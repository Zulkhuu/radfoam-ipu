// main.cpp

#include <iostream>
#include <vector>
#include <iomanip>
#include <cstdint>

#include <highfive/H5File.hpp>
#include <highfive/H5DataType.hpp>

#include <poplar/Device.hpp>
#include <poplar/DeviceManager.hpp>
#include <poplar/Graph.hpp>
#include <poplar/Engine.hpp>
#include <popops/Reduce.hpp>
#include <popops/Operation.hpp>
#include <popops/codelets.hpp>

#include "ipu/ipu_utils.hpp"
#include <radfoam_types.hpp>

#include <remote_ui/InterfaceServer.hpp>
#include <remote_ui/AsyncTask.hpp>

template <typename T>
std::string vector_slice_to_string(const std::vector<T>& vec, size_t start, size_t end) {
    std::ostringstream oss;
    oss << "[";
    if (start >= vec.size()) {
        oss << "]";
        return oss.str(); // empty slice
    }

    end = std::min(end, vec.size());
    for (size_t i = start; i < end; ++i) {
        if constexpr (std::is_same<T, uint8_t>::value) {
            oss << static_cast<int>(vec[i]);  // Safe cast for uint8_t
        } else {
            oss << vec[i];
        }

        if (i + 1 < end)
            oss << ", ";
    }
    oss << "]";
    return oss.str();
}


// ─── Point type ────────────────────────────────────────────────────
static inline HighFive::CompoundType makeLocalPointType()
{
    using namespace HighFive;
    return CompoundType{
        {"x",       create_datatype<float>(),   offsetof(LocalPoint, x)},
        {"y",       create_datatype<float>(),   offsetof(LocalPoint, y)},
        {"z",       create_datatype<float>(),   offsetof(LocalPoint, z)},
        {"r",       create_datatype<uint8_t>(), offsetof(LocalPoint, r)},
        {"g",       create_datatype<uint8_t>(), offsetof(LocalPoint, g)},
        {"b",       create_datatype<uint8_t>(), offsetof(LocalPoint, b)},
        {"_pad",    create_datatype<uint8_t>(), offsetof(LocalPoint, _pad)},
        {"density", create_datatype<float>(),   offsetof(LocalPoint, density)},
        {"adj_end", create_datatype<uint32_t>(),offsetof(LocalPoint, adj_end)},
    };
}
HIGHFIVE_REGISTER_TYPE(LocalPoint, makeLocalPointType);

// ─── Register the compound type with HighFive ─────────────────────
static inline HighFive::CompoundType makeNeighborPointType()
{
    using namespace HighFive;
    return CompoundType{
        {"x",       create_datatype<float>(),   offsetof(NeighborPoint, x)},
        {"y",       create_datatype<float>(),   offsetof(NeighborPoint, y)},
        {"z",       create_datatype<float>(),   offsetof(NeighborPoint, z)},
        {"gid",       create_datatype<uint8_t>(), offsetof(NeighborPoint, gid)},
    };
}
HIGHFIVE_REGISTER_TYPE(NeighborPoint, makeNeighborPointType);

// ─── Acquire an IPU (or an IPU Model if no hardware is available) ─
poplar::Device acquireDevice()
{
    auto dm = poplar::DeviceManager::createDeviceManager();

    for (auto& d : dm.getDevices(poplar::TargetType::IPU, 1))
        if (d.attach()) return std::move(d);

    throw std::runtime_error("No available IPU device");
}

void h5_load_test_print(const std::vector<LocalPoint>& local_points,
                        const std::vector<NeighborPoint>& neighbor_points,
                        const std::vector<uint16_t>& adjacency_list,
                        size_t index) {
    std::cout << "" << std::endl;
    ipu_utils::logger()->info("Local points size: {}", local_points.size());
    if (index < local_points.size()) {
        const auto& pt = local_points[index];
        ipu_utils::logger()->info("Local point[{}]: x={}, y={}, z={}", index, pt.x, pt.y, pt.z);
    } else {
        ipu_utils::logger()->warn("Index {} out of bounds for local_points", index);
    }

    ipu_utils::logger()->info("Neighbor points size: {}", neighbor_points.size());
    if (index < neighbor_points.size()) {
        const auto& nb = neighbor_points[index];
        ipu_utils::logger()->info("Neighbor point[{}]: x={}, y={}, z={}", index, nb.x, nb.y, nb.z);
    } else {
        ipu_utils::logger()->warn("Index {} out of bounds for neighbor_points", index);
    }

    ipu_utils::logger()->info("Loaded {} adjacency entries", adjacency_list.size());
    ipu_utils::logger()->info("Adjacency [{}:{}]: {}", 
                              0, std::min<size_t>(10, adjacency_list.size()), 
                              vector_slice_to_string(adjacency_list, 0, 10));
    std::cout << "" << std::endl;
}

class RadfoamBuilder : public ipu_utils::BuilderInterface {
public:
  explicit RadfoamBuilder(const std::string& filename) : h5file(filename) {}

  void build(poplar::Graph& graph, const poplar::Target& target) override {
    using namespace ipu_utils;
    logger()->info("Building Radfoam graph");

    // Load HDF5 datasets
    HighFive::File file(h5file, HighFive::File::ReadOnly);
    file.getGroup("part0462").getDataSet("local_pts").read(local_points);
    file.getGroup("part0462").getDataSet("neighbor_pts").read(neighbor_points);
    file.getGroup("part0462").getDataSet("adjacency_list").read(adjacency_list);

    h5_load_test_print(local_points, neighbor_points, adjacency_list, 0);

    // Add codelets
    std::string codeletFile = std::string(POPC_PREFIX) + "/src/codelets/codelets.cpp";
    std::string incPath = std::string(POPC_PREFIX) + "/include/";
    std::string glmPath = std::string(POPC_PREFIX) + "/external/glm/";
    std::string includes = " -I " + incPath + " -I " + glmPath;

    graph.addCodelets(codeletFile, poplar::CodeletFileType::Auto, "-O3" + includes);
    popops::addCodelets(graph);

    // Build and map tensors
    inputLocal.buildTensor(graph, poplar::UNSIGNED_CHAR, {sizeof(LocalPoint) * local_points.size()});
    inputNeighbors.buildTensor(graph, poplar::UNSIGNED_CHAR, {sizeof(NeighborPoint) * neighbor_points.size()});
    inputAdjList.buildTensor(graph, poplar::UNSIGNED_SHORT, {adjacency_list.size()});
    inputSceneSizes.buildTensor(graph, poplar::UNSIGNED_SHORT, {3});
    result.buildTensor(graph, poplar::FLOAT, {1});
    result_uint16.buildTensor(graph, poplar::UNSIGNED_SHORT, {1});
    framebuffer.buildTensor(graph, poplar::UNSIGNED_CHAR, {128 * 72 * 3});

    raysA = graph.addVariable(poplar::UNSIGNED_CHAR, {numRays * sizeof(Ray)}, "raysA");
    raysB = graph.addVariable(poplar::UNSIGNED_CHAR, {numRays * sizeof(Ray)}, "raysB");
    auto zeroConst = graph.addConstant(poplar::UNSIGNED_CHAR, {numRays * sizeof(Ray)}, 0);

    // Tile mapping
    graph.setTileMapping(zeroConst, 0);
    graph.setTileMapping(raysA, 0);
    graph.setTileMapping(raysB, 0);
    graph.setTileMapping(inputLocal, 0);
    graph.setTileMapping(inputNeighbors, 0);
    graph.setTileMapping(inputAdjList, 0);
    graph.setTileMapping(inputSceneSizes, 0);
    graph.setTileMapping(result, 0);
    graph.setTileMapping(result_uint16, 0);
    graph.setTileMapping(framebuffer, 1);

    // Zero rays
    getPrograms().add("zero_rays", poplar::program::Sequence{
      poplar::program::Copy(zeroConst, raysA),
      poplar::program::Copy(zeroConst, raysB)
    });

    // RayGen
    auto rayGenCS = graph.addComputeSet("RayGenCS");
    auto rayGenVertex = graph.addVertex(rayGenCS, "RayGen");
    graph.connect(rayGenVertex["raysIn"], raysA);
    graph.connect(rayGenVertex["raysOut"], raysB);
    graph.connect(rayGenVertex["framebuffer"], framebuffer.get());
    graph.setTileMapping(rayGenVertex, 1);
    getPrograms().add("RayGenCS", poplar::program::Execute(rayGenCS));

    // RayTrace
    auto traceCS = graph.addComputeSet("RayTraceCS");
    auto vertex = graph.addVertex(traceCS, "RayTrace");
    graph.connect(vertex["local_pts"], inputLocal.get());
    graph.connect(vertex["neighbor_pts"], inputNeighbors.get());
    graph.connect(vertex["adjacency"], inputAdjList.get());
    graph.connect(vertex["scene_sizes"], inputSceneSizes.get());
    graph.connect(vertex["result_float"], result.get().reshape({}));
    graph.connect(vertex["result_u16"], result_uint16.get().reshape({}));
    graph.connect(vertex["raysIn"], raysB);
    graph.connect(vertex["raysOut"], raysA);
    graph.setTileMapping(vertex, 0);
    getPrograms().add("RayTraceCS", poplar::program::Execute(traceCS));

    // Readbacks
    getPrograms().add("readouts", poplar::program::Sequence{
      framebuffer.buildRead(graph, true),
      result.buildRead(graph, true),
      result_uint16.buildRead(graph, true)
    });

    // Writes
    getPrograms().add("write", poplar::program::Sequence{
      inputLocal.buildWrite(graph, true),
      inputNeighbors.buildWrite(graph, true),
      inputAdjList.buildWrite(graph, true)
    });
  }

  void execute(poplar::Engine& engine, const poplar::Device& device) override {
    framebuffer_host.resize(128 * 72 * 3);
    inputLocal.connectWriteStream(engine, local_points);
    inputNeighbors.connectWriteStream(engine, neighbor_points);
    inputAdjList.connectWriteStream(engine, adjacency_list);
    result.connectReadStream(engine, &result_float);
    result_uint16.connectReadStream(engine, &result_u16);
    framebuffer.connectReadStream(engine, framebuffer_host.data());

    engine.run(getPrograms().getOrdinals().at("zero_rays"));
    engine.run(getPrograms().getOrdinals().at("write"));
    engine.run(getPrograms().getOrdinals().at("RayGenCS"));
    engine.run(getPrograms().getOrdinals().at("RayTraceCS"));
    engine.run(getPrograms().getOrdinals().at("readouts"));

    ipu_utils::logger()->info("result_u16: {}", result_u16);
    ipu_utils::logger()->info("result_float: {}", result_float);
    ipu_utils::logger()->info("Framebuffer sample: {}", vector_slice_to_string(framebuffer_host, 0, 12));
    //engine.printProfileSummary(std::cout, {{"showExecutionSteps", "true"}});
  }

private:
  std::string h5file;
  std::vector<LocalPoint> local_points;
  std::vector<NeighborPoint> neighbor_points;
  std::vector<uint16_t> adjacency_list;
  std::vector<uint8_t> framebuffer_host;
  float result_float = 0;
  uint16_t result_u16 = 0;

  ipu_utils::StreamableTensor inputLocal{"input_local"};
  ipu_utils::StreamableTensor inputNeighbors{"input_neighbors"};
  ipu_utils::StreamableTensor inputAdjList{"input_adjacency"};
  ipu_utils::StreamableTensor inputSceneSizes{"input_scene_sizes"};
  ipu_utils::StreamableTensor result{"result_float"};
  ipu_utils::StreamableTensor result_uint16{"result_u16"};
  ipu_utils::StreamableTensor framebuffer{"framebuffer"};
  poplar::Tensor raysA, raysB;
  constexpr static size_t numRays = 512;
};

int main(int argc, char** argv) {
    const std::string inputFile = argc > 1 ? argv[1] : "./data/garden.h5";
    RadfoamBuilder builder(inputFile);

    ipu_utils::RuntimeConfig cfg;
    cfg.numIpus = 1;
    cfg.useIpuModel = false;
    cfg.numReplicas = 1; 
    cfg.loadExe = false;
    cfg.saveExe = false;
    cfg.compileOnly = false;
    cfg.deferredAttach = false;
    cfg.exeName = "radfoam_exe";
    builder.setRuntimeConfig(cfg);

    ipu_utils::GraphManager manager;
    int exit_code = manager.run(builder, {
        {"debug.instrument", "true"}
    });
    return exit_code;
}
