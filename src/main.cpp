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


int main(int argc, char** argv)
{
    try {
        const std::string h5file = argc > 1 ? argv[1] : "./data/garden.h5";

        // ─── Load the first point from the file ────────────────────
        std::cout << "Opening file:" << h5file << '\n';
        HighFive::File file(h5file, HighFive::File::ReadOnly);

        auto local_ds = file.getGroup("part0000").getDataSet("local_pts");
        std::vector<LocalPoint> local_points;
        local_ds.read(local_points);
        
        auto neighbor_ds = file.getGroup("part0000").getDataSet("neighbor_pts");
        std::vector<NeighborPoint> neighbor_points;
        neighbor_ds.read(neighbor_points);

        auto adj_ds = file.getGroup("part0000").getDataSet("adjacency_list");
        std::vector<uint16_t> adjacency_list;
        adj_ds.read(adjacency_list);

        h5_load_test_print(local_points, neighbor_points, adjacency_list, 0);

        // ─── Build a simple Poplar graph ───────────────────────────
        ipu_utils::StreamableTensor inputLocal("input_local");
        ipu_utils::StreamableTensor inputNeighbors("input_neighbors");
        ipu_utils::StreamableTensor inputAdjList("input_adjacency");
        ipu_utils::StreamableTensor inputSceneSizes("input_scene_sizes");
        ipu_utils::StreamableTensor result("sum_result");
        ipu_utils::StreamableTensor result_uint16("sum_result_u16");

        ipu_utils::StreamableTensor framebuffer("framebuffer");

        auto device = acquireDevice();
        poplar::Graph graph(device.getTarget());
        const std::string codeletFile = std::string(POPC_PREFIX) + "/src/codelets/codelets.cpp";
        const std::string incPath = std::string(POPC_PREFIX) + "/include/";
        const std::string glmPath = std::string(POPC_PREFIX) + "/external/glm/";
        const std::string includes = " -I " + incPath + " -I " + glmPath;
        ipu_utils::logger()->debug("POPC_PREFIX: {}", POPC_PREFIX);
        popops::addCodelets(graph);
        graph.addCodelets(codeletFile, poplar::CodeletFileType::Auto, "-O3" + includes);
        poplar::program::Sequence prog;

        // Build tensors
        inputLocal.buildTensor(graph, poplar::UNSIGNED_CHAR, {sizeof(LocalPoint) * local_points.size()});
        inputNeighbors.buildTensor(graph, poplar::UNSIGNED_CHAR, {sizeof(NeighborPoint) * neighbor_points.size()});
        inputSceneSizes.buildTensor(graph, poplar::UNSIGNED_SHORT, {3});
        inputAdjList.buildTensor(graph, poplar::UNSIGNED_SHORT, { adjacency_list.size()});
        result.buildTensor(graph, poplar::FLOAT, {1});
        result_uint16.buildTensor(graph, poplar::UNSIGNED_SHORT, {1});

        framebuffer.buildTensor(graph, poplar::UNSIGNED_CHAR, {128 * 72 * 3});

        constexpr size_t numRays = 512;

        // These rays live entirely on device (no host stream)
        auto raysA = graph.addVariable(poplar::UNSIGNED_CHAR, {numRays*sizeof(Ray)}, "raysA");
        auto raysB = graph.addVariable(poplar::UNSIGNED_CHAR, {numRays*sizeof(Ray)}, "raysB");
        auto zeroConst = graph.addConstant(poplar::UNSIGNED_CHAR, {numRays*sizeof(Ray)}, 0, "rays_zero");
        graph.setTileMapping(zeroConst, 0);



        graph.setTileMapping(raysA, 0);
        graph.setTileMapping(raysB, 0);

        // Map to tile 0
        graph.setTileMapping(inputLocal, 0);
        graph.setTileMapping(inputNeighbors, 0);
        graph.setTileMapping(inputAdjList, 0);
        graph.setTileMapping(inputSceneSizes, 0);
        graph.setTileMapping(result, 0);
        graph.setTileMapping(result_uint16, 0);

        graph.setTileMapping(framebuffer, 1);

        // Host -> Device programs
        auto writeLocal = inputLocal.buildWrite(graph, true);
        auto writeNeighbors = inputNeighbors.buildWrite(graph, true);
        auto writeAdjList = inputAdjList.buildWrite(graph, true);
        auto writeSceneSizes = inputSceneSizes.buildWrite(graph, true);

        // Device computation: sum both tensors
        auto cs = graph.addComputeSet("sum_xyz");
        auto vertex = graph.addVertex(cs, "RayTrace");

        graph.connect(vertex["local_pts"], inputLocal.get());
        graph.connect(vertex["neighbor_pts"], inputNeighbors.get());
        graph.connect(vertex["adjacency"], inputAdjList.get());
        graph.connect(vertex["scene_sizes"], inputSceneSizes.get());  
        graph.connect(vertex["result_float"], result.get().reshape({}));
        graph.connect(vertex["result_u16"], result_uint16.get().reshape({}));  
        graph.connect(vertex["raysIn"], raysB);
        graph.connect(vertex["raysOut"], raysA);
        
        graph.setTileMapping(vertex, 0);


        auto rayGenCS = graph.addComputeSet("RayGenCS");
        auto rayGenVertex = graph.addVertex(rayGenCS, "RayGen");

        graph.connect(rayGenVertex["raysIn"], raysA);
        graph.connect(rayGenVertex["raysOut"], raysB);
        graph.connect(rayGenVertex["framebuffer"], framebuffer.get());
        graph.setTileMapping(rayGenVertex, 1);

        // Copy zeroConst → raysA and raysB at runtime
        prog.add(poplar::program::Copy(zeroConst, raysA));
        prog.add(poplar::program::Copy(zeroConst, raysB));

        prog.add(writeLocal);
        prog.add(writeNeighbors);
        prog.add(writeAdjList);
        // RayGen runs first
        prog.add(poplar::program::Execute(rayGenCS));

        // Then RayTrace
        prog.add(poplar::program::Execute(cs));

        // Device -> Host
        auto readResult = result.buildRead(graph, true);
        auto readResultU16 = result_uint16.buildRead(graph, true);
        auto readFramebuffer = framebuffer.buildRead(graph, true);
        
        prog.add(readFramebuffer);
        prog.add(readResult);
        prog.add(readResultU16);

        poplar::Engine engine(graph, prog);

        float result_float = 0.0f;
        uint16_t result_u16 = 0;

        inputLocal.connectWriteStream(engine, local_points);
        inputNeighbors.connectWriteStream(engine, neighbor_points);
        inputAdjList.connectWriteStream(engine, adjacency_list);
        result.connectReadStream(engine, &result_float);
        result_uint16.connectReadStream(engine, &result_u16); 

        std::vector<uint8_t> hostFramebuffer(128 * 72 * 3);
        framebuffer.connectReadStream(engine, hostFramebuffer.data());

        engine.load(device);
        engine.run();

        ipu_utils::logger()->info("result_u16: {}", result_u16);
        ipu_utils::logger()->info("result_float: {}", result_float);
        ipu_utils::logger()->info("Framebuffer sample: {}", vector_slice_to_string(hostFramebuffer, 0, 12));
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << '\n';
        return 1;
    }
}
