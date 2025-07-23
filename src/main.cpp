// main.cpp

#include <iostream>
#include <vector>
#include <iomanip>
#include <cstdint>

#include <spdlog/fmt/fmt.h>

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
#include "ipu/tile_config.hpp"
#include <radfoam_types.hpp>

#include <remote_ui/InterfaceServer.hpp>
#include <remote_ui/AsyncTask.hpp>

#include <glm/glm.hpp>
#include <glm/mat4x4.hpp>
#include <glm/gtc/matrix_transform.hpp> 

#include <pvti/pvti.hpp>


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
    explicit RadfoamBuilder(const std::string& filename,
                            int tileToCompute_ = 0)
        : h5file(filename), tileToCompute(tileToCompute_), initialised(false) {}

    // ─── BuilderInterface overrides ───────────────────────────────────────────
    void build(poplar::Graph& graph, const poplar::Target& target) override;
    void execute(poplar::Engine& engine, const poplar::Device& device) override;

private:
    // ─── Helper types & constants ─────────────────────────────────────────────
    static constexpr unsigned NUM_TRACE_TILES = 1024;
    static constexpr unsigned RAYGEN_TILE     = 1024;
    static constexpr std::size_t FB_SIZE      = 40 * 23 * 3;
    static constexpr std::size_t numRays      = 1500;

    // ─── HDF5 input -----------------------------------------------------------
    std::string h5file;

    // ─── Host-side scene data for each tile --------------------------
    std::vector<std::vector<LocalPoint>>    hostLocal;   // [tile][points]
    std::vector<std::vector<NeighborPoint>> hostNbr;     // [tile][points]
    std::vector<std::vector<uint16_t>>      hostAdj;     // [tile][indices]

    // ─── Poplar tensors & streams --------------------------------------------
    poplar::Tensor raysA, raysB;
    poplar::Tensor resultFloatAll;   // FLOAT[1024]
    poplar::Tensor resultU16All;     // UINT16[1024]
    poplar::Tensor framebufferAll;   // UCHAR[1024][FB_SIZE]

    // Write streams (one per tile)
    std::vector<ipu_utils::StreamableTensor> hostLocalT;
    std::vector<ipu_utils::StreamableTensor> hostNbrT;
    std::vector<ipu_utils::StreamableTensor> hostAdjT;

    // Read-back streams (only chosen tile)
    ipu_utils::StreamableTensor fbReadT {"fb_read"};
    ipu_utils::StreamableTensor f32ReadT{"f32_read"};
    ipu_utils::StreamableTensor u16ReadT{"u16_read"};

    // Host-visible result buffers for selected tile
    std::vector<uint8_t> framebuffer_host;
    float                result_float = 0.f;
    uint16_t             result_u16   = 0;

    // Poplar program sequences
    poplar::program::Sequence perTileWrites;
    poplar::program::Sequence readSeq;

    // State
    int                    tileToCompute;
    std::atomic<bool>      initialised;
};

/*═══════════════════════════════════════════════════════════════════════════
  build()
═══════════════════════════════════════════════════════════════════════════*/
inline void RadfoamBuilder::build(poplar::Graph& graph,
                                  const poplar::Target& /*target*/) {
    using namespace ipu_utils;
    logger()->info("Building Radfoam graph (tileToCompute = {})", tileToCompute);
    poplar::program::Sequence zeroSeq;  // Accumulate zeroing copies

    // ─── Load HDF5 partitions part0000 … part1023 ────────────────────────────
    hostLocal .resize(NUM_TRACE_TILES);
    hostNbr   .resize(NUM_TRACE_TILES);
    hostAdj   .resize(NUM_TRACE_TILES);

    {
        HighFive::File file(h5file, HighFive::File::ReadOnly);
        for (unsigned tid = 0; tid < NUM_TRACE_TILES; ++tid) {
            std::string ds = fmt::format("part{:04}", tid);
            file.getGroup(ds).getDataSet("local_pts")      .read(hostLocal[tid]);
            file.getGroup(ds).getDataSet("neighbor_pts")   .read(hostNbr [tid]);
            file.getGroup(ds).getDataSet("adjacency_list") .read(hostAdj     [tid]);
        }
        h5_load_test_print(hostLocal[0], hostNbr[0], hostAdj[0], 0);
    }

    // ─── Compile codelets & popops  ──────────────────────────────────────────
    std::string codeletFile = std::string(POPC_PREFIX) + "/src/codelets/codelets.cpp";
    std::string incPath     = std::string(POPC_PREFIX) + "/include/";
    std::string glmPath     = std::string(POPC_PREFIX) + "/external/glm/";
    graph.addCodelets(codeletFile, poplar::CodeletFileType::Auto,
                    "-O3 -I " + incPath + " -I " + glmPath);
    popops::addCodelets(graph);

    // ─── Shared ray buffers (two-buffer ping-pong) ───────────────────────────
    constexpr std::size_t rayBytesPerTile = numRays * sizeof(Ray);
    constexpr std::size_t totalRayBytes = NUM_TRACE_TILES * rayBytesPerTile;

    raysA = graph.addVariable(poplar::UNSIGNED_CHAR, {totalRayBytes}, "raysA");
    raysB = graph.addVariable(poplar::UNSIGNED_CHAR, {totalRayBytes}, "raysB");

    auto zeroConst = graph.addConstant(poplar::UNSIGNED_CHAR, {rayBytesPerTile}, 0);
    graph.setTileMapping(zeroConst, 1024);

    // ─── Result tensors common to all tiles ──────────────────────────────────
    resultFloatAll = graph.addVariable(poplar::FLOAT,          {NUM_TRACE_TILES});
    resultU16All   = graph.addVariable(poplar::UNSIGNED_SHORT, {NUM_TRACE_TILES});
    framebufferAll = graph.addVariable(poplar::UNSIGNED_CHAR,
                                        {NUM_TRACE_TILES, FB_SIZE});
                                     
    // map every scalar / framebuffer row to its owning tile
    for (unsigned tid = 0; tid < NUM_TRACE_TILES; ++tid) {
        graph.setTileMapping(resultFloatAll.slice({tid}, {tid + 1}), tid);
        graph.setTileMapping(resultU16All  .slice({tid}, {tid + 1}), tid);
        graph.setTileMapping(framebufferAll.slice({tid, 0},
                                                    {tid + 1, FB_SIZE}), tid);
    }

  // —— Build per-tile inputs and RayTrace vertices ————————————————
  auto traceCS = graph.addComputeSet("RayTraceCS");

  for (unsigned tid = 0; tid < NUM_TRACE_TILES; ++tid) {

    // Input tensors
    ipu_utils::StreamableTensor inLocal(fmt::format("local_{}", tid));
    ipu_utils::StreamableTensor inNbr  (fmt::format("nbr_{}"  , tid));
    ipu_utils::StreamableTensor inAdj  (fmt::format("adj_{}"  , tid));

    inLocal.buildTensor(graph, poplar::UNSIGNED_CHAR, {sizeof(LocalPoint) * hostLocal[tid].size()});
    inNbr.buildTensor(graph, poplar::UNSIGNED_CHAR, {sizeof(NeighborPoint) * hostNbr[tid].size()});
    inAdj.buildTensor(graph, poplar::UNSIGNED_SHORT, {hostAdj[tid].size()});

    graph.setTileMapping(inLocal.get(), tid);
    graph.setTileMapping(inNbr.get(), tid);
    graph.setTileMapping(inAdj.get(), tid);

    // Add vertex
    auto v = graph.addVertex(traceCS, "RayTrace");
    graph.connect(v["local_pts"],    inLocal.get());
    graph.connect(v["neighbor_pts"], inNbr  .get());
    graph.connect(v["adjacency"],    inAdj  .get());

    auto raysA_slice = raysA.slice(tid * rayBytesPerTile,
                                (tid + 1) * rayBytesPerTile);
    auto raysB_slice = raysB.slice(tid * rayBytesPerTile,
                                (tid + 1) * rayBytesPerTile);

    graph.setTileMapping(raysA_slice, tid);
    graph.setTileMapping(raysB_slice, tid);

    zeroSeq.add(poplar::program::Copy(zeroConst, raysA_slice));
    zeroSeq.add(poplar::program::Copy(zeroConst, raysB_slice));

    graph.connect(v["raysIn"],  raysB_slice);
    graph.connect(v["raysOut"], raysA_slice);


    // Per-tile outputs
    graph.connect(v["result_float"], resultFloatAll.slice({tid},{tid+1}).reshape({}));
    graph.connect(v["result_u16"], resultU16All.slice({tid},{tid+1}).reshape({}));
    graph.connect(v["framebuffer"], framebufferAll.slice({tid,0},{tid+1,FB_SIZE}).reshape({FB_SIZE}));

    // tile_id constant
    auto tileConst = graph.addConstant(poplar::UNSIGNED_SHORT, {}, tid);
    graph.setTileMapping(tileConst, tid);
    graph.connect(v["tile_id"], tileConst);

    graph.setTileMapping(v, tid);

    // accumulate writes
    // accumulate writes (Sequence::add)
    perTileWrites.add(inLocal.buildWrite(graph, true));
    perTileWrites.add(inNbr  .buildWrite(graph, true));
    perTileWrites.add(inAdj  .buildWrite(graph, true));

    // store tensors for stream connection
    hostLocalT.push_back(std::move(inLocal));
    hostNbrT  .push_back(std::move(inNbr));
    hostAdjT  .push_back(std::move(inAdj));
  }
  // Add the Trace program
  getPrograms().add("RayTraceCS", poplar::program::Execute(traceCS));

  // ─── RayGen vertex (tile 1024) ───────────────────────────────────────────
  {
    auto rayGenCS = graph.addComputeSet("RayGenCS");
    auto rayGenV  = graph.addVertex(rayGenCS, "RayGen");
    auto raysA_gen_slice = raysA.slice(tileToCompute * rayBytesPerTile,
                                   (tileToCompute + 1) * rayBytesPerTile);
    auto raysB_gen_slice = raysB.slice(tileToCompute * rayBytesPerTile,
                                    (tileToCompute + 1) * rayBytesPerTile);

    graph.connect(rayGenV["raysIn"],  raysA_gen_slice);
    graph.connect(rayGenV["raysOut"], raysB_gen_slice);
    graph.setTileMapping(rayGenV, RAYGEN_TILE);
    getPrograms().add("RayGenCS", poplar::program::Execute(rayGenCS));
  }

    // ─── Read-back (only selected tile) ───────────────────────────────────────
    const std::size_t tid = static_cast<std::size_t>(tileToCompute);
    auto fbSlice  = framebufferAll.slice({tid,0},{tid+1, FB_SIZE}).reshape({FB_SIZE});
    auto f32Slice = resultFloatAll.slice({tid}, {tid+1}).reshape({});
    auto u16Slice = resultU16All.slice({tid}, {tid+1}).reshape({});

    fbReadT  = fbSlice;
    f32ReadT = f32Slice;
    u16ReadT = u16Slice;

    auto fbReadProg  = fbReadT .buildRead(graph, true);
    auto f32ReadProg = f32ReadT.buildRead(graph, true);
    auto u16ReadProg = u16ReadT.buildRead(graph, true);
    readSeq = poplar::program::Sequence({fbReadProg, f32ReadProg, u16ReadProg});

    getPrograms().add("readouts", readSeq);

    // ─── Zero-rays & bulk writes ──────────────────────────────────────────────
    getPrograms().add("zero_rays", zeroSeq);
    getPrograms().add("write", perTileWrites);
}

/*═══════════════════════════════════════════════════════════════════════════
  execute()
═══════════════════════════════════════════════════════════════════════════*/
inline void RadfoamBuilder::execute(poplar::Engine& engine,
                                    const poplar::Device& /*device*/) {
  if (!initialised) {
    framebuffer_host.resize(FB_SIZE);

    // Connect write streams for all tiles
    for (unsigned tid = 0; tid < hostLocalT.size(); ++tid) {
      hostLocalT[tid].connectWriteStream(engine, hostLocal[tid]);
      hostNbrT  [tid].connectWriteStream(engine, hostNbr [tid]);
      hostAdjT  [tid].connectWriteStream(engine, hostAdj [tid]);
    }

    // Connect read streams for chosen tile
    fbReadT .connectReadStream(engine, framebuffer_host.data());
    f32ReadT.connectReadStream(engine, &result_float);
    u16ReadT.connectReadStream(engine, &result_u16);

    // Initialise IPU buffers
    engine.run(getPrograms().getOrdinals().at("zero_rays"));
    engine.run(getPrograms().getOrdinals().at("write"));
    initialised = true;
  }

  // One frame
  engine.run(getPrograms().getOrdinals().at("RayGenCS"));
  engine.run(getPrograms().getOrdinals().at("RayTraceCS"));
  engine.run(getPrograms().getOrdinals().at("readouts"));

  // ─── Debug prints ────────────────────────────────────────────────────────
  ipu_utils::logger()->info("TILE {} → result_u16  : {}", tileToCompute, result_u16);
  ipu_utils::logger()->info("TILE {} → result_float: {}", tileToCompute, result_float);
  ipu_utils::logger()->info("Framebuffer sample   : {}", vector_slice_to_string(framebuffer_host, 0, 12));
}

int main(int argc, char** argv) {
    pvti::TraceChannel traceChannel = {"radfoam_ipu_tracer"};
    const std::string inputFile = argc > 1 ? argv[1] : "./data/garden.h5";
    const int         tileToCompute = 462;
    RadfoamBuilder builder(inputFile, tileToCompute);

    auto imagePtr = std::make_unique<cv::Mat>(kFullImageHeight, kFullImageWidth, CV_8UC3);
    // std::unique_ptr<InterfaceServer> uiServer;
    // InterfaceServer::State state;
    // state.fov = glm::radians(40.f);
    // state.device = "ipu";
    // auto uiPort = 5000;
    // if (uiPort) {
    //     uiServer.reset(new InterfaceServer(uiPort));
    //     uiServer->start();
    //     uiServer->initialiseVideoStream(imagePtr->cols, imagePtr->rows);
    //     uiServer->updateFov(state.fov);
    // }
    // AsyncTask hostProcessing;
    // auto uiUpdateFunc = [&]() {
    //     {
    //     pvti::Tracepoint scoped(&traceChannel, "ui_update");
    //     uiServer->sendPreviewImage(*imagePtrBuffered);
    //     }
    // };

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
