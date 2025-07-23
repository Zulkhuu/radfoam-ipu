#include <iostream>
#include <vector>
#include <iomanip>
#include <cstdint>
#include <cstring>

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

#include <opencv2/opencv.hpp> 


// -----------------------------------------------------------------------------
// Debug helper
template <typename T>
std::string vector_slice_to_string(const std::vector<T>& vec, size_t start, size_t end) {
    std::ostringstream oss;
    oss << "[";
    if (start >= vec.size()) {
        oss << "]";
        return oss.str();
    }

    end = std::min(end, vec.size());
    for (size_t i = start; i < end; ++i) {
        if constexpr (std::is_same<T, uint8_t>::value) {
            oss << static_cast<int>(vec[i]);
        } else {
            oss << vec[i];
        }

        if (i + 1 < end)
            oss << ", ";
    }
    oss << "]";
    return oss.str();
}

// -----------------------------------------------------------------------------
// HDF5 compound types
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

static inline HighFive::CompoundType makeNeighborPointType()
{
    using namespace HighFive;
    return CompoundType{
        {"x",       create_datatype<float>(),   offsetof(NeighborPoint, x)},
        {"y",       create_datatype<float>(),   offsetof(NeighborPoint, y)},
        {"z",       create_datatype<float>(),   offsetof(NeighborPoint, z)},
        {"gid",     create_datatype<uint8_t>(), offsetof(NeighborPoint, gid)},
    };
}
HIGHFIVE_REGISTER_TYPE(NeighborPoint, makeNeighborPointType);

// -----------------------------------------------------------------------------
// RadfoamBuilder
class RadfoamBuilder : public ipu_utils::BuilderInterface {
public:
    explicit RadfoamBuilder(const std::string& filename,
                            int tileToCompute_ = 0)
        : h5file(filename), tileToCompute(tileToCompute_), initialised(false) {}

    void build(poplar::Graph& graph, const poplar::Target& target) override;
    void execute(poplar::Engine& engine, const poplar::Device& device) override;

    void readAllTiles(poplar::Engine &engine);

    std::vector<uint8_t> framebuffer_all_tiles; // Full framebuffer

private:
    static constexpr unsigned RAYGEN_TILE     = 1024;
    static constexpr std::size_t numRays      = 1500;

    std::string h5file;

    std::vector<std::vector<LocalPoint>>    hostLocal;
    std::vector<std::vector<NeighborPoint>> hostNbr;
    std::vector<std::vector<uint16_t>>      hostAdj;

    poplar::Tensor raysA, raysB;
    poplar::Tensor resultFloatAll;
    poplar::Tensor resultU16All;
    poplar::Tensor framebufferAll;

    std::vector<ipu_utils::StreamableTensor> hostLocalT;
    std::vector<ipu_utils::StreamableTensor> hostNbrT;
    std::vector<ipu_utils::StreamableTensor> hostAdjT;

    poplar::program::Sequence perTileWrites;

    int                    tileToCompute;
    std::atomic<bool>      initialised;
};

// -----------------------------------------------------------------------------
// Build graph
inline void RadfoamBuilder::build(poplar::Graph& graph,
                                  const poplar::Target& /*target*/) {
    using namespace ipu_utils;
    logger()->info("Building Radfoam graph (tileToCompute = {})", tileToCompute);
    poplar::program::Sequence zeroSeq;

    // Load HDF5 partitions
    hostLocal.resize(kNumTraceTiles);
    hostNbr.resize(kNumTraceTiles);
    hostAdj.resize(kNumTraceTiles);

    {
        HighFive::File file(h5file, HighFive::File::ReadOnly);
        for (unsigned tid = 0; tid < kNumTraceTiles; ++tid) {
            std::string ds = fmt::format("part{:04}", tid);
            file.getGroup(ds).getDataSet("local_pts").read(hostLocal[tid]);
            file.getGroup(ds).getDataSet("neighbor_pts").read(hostNbr[tid]);
            file.getGroup(ds).getDataSet("adjacency_list").read(hostAdj[tid]);
        }
    }

    // Compile codelets
    std::string codeletFile = std::string(POPC_PREFIX) + "/src/codelets/codelets.cpp";
    std::string incPath     = std::string(POPC_PREFIX) + "/include/";
    std::string glmPath     = std::string(POPC_PREFIX) + "/external/glm/";
    graph.addCodelets(codeletFile, poplar::CodeletFileType::Auto,
                      "-O3 -I " + incPath + " -I " + glmPath);
    popops::addCodelets(graph);

    constexpr std::size_t rayBytesPerTile = numRays * sizeof(Ray);
    constexpr std::size_t totalRayBytes = kNumTraceTiles * rayBytesPerTile;

    raysA = graph.addVariable(poplar::UNSIGNED_CHAR, {totalRayBytes}, "raysA");
    raysB = graph.addVariable(poplar::UNSIGNED_CHAR, {totalRayBytes}, "raysB");

    auto zeroConst = graph.addConstant(poplar::UNSIGNED_CHAR, {rayBytesPerTile}, 0);
    graph.setTileMapping(zeroConst, 1024);

    // Result tensors
    resultFloatAll = graph.addVariable(poplar::FLOAT,          {kNumTraceTiles});
    resultU16All   = graph.addVariable(poplar::UNSIGNED_SHORT, {kNumTraceTiles});
    framebufferAll = graph.addVariable(poplar::UNSIGNED_CHAR,  {kNumTraceTiles, kTileFramebufferSize});
                                     
    for (unsigned tid = 0; tid < kNumTraceTiles; ++tid) {
        graph.setTileMapping(resultFloatAll.slice({tid}, {tid + 1}), tid);
        graph.setTileMapping(resultU16All.slice({tid}, {tid + 1}), tid);
        graph.setTileMapping(framebufferAll.slice({tid, 0}, {tid + 1, kTileFramebufferSize}), tid);
    }

    // Per-tile RayTrace vertices
    auto traceCS = graph.addComputeSet("RayTraceCS");

    for (unsigned tid = 0; tid < kNumTraceTiles; ++tid) {
        ipu_utils::StreamableTensor inLocal(fmt::format("local_{}", tid));
        ipu_utils::StreamableTensor inNbr(fmt::format("nbr_{}", tid));
        ipu_utils::StreamableTensor inAdj(fmt::format("adj_{}", tid));

        inLocal.buildTensor(graph, poplar::UNSIGNED_CHAR, {sizeof(LocalPoint) * hostLocal[tid].size()});
        inNbr.buildTensor(graph, poplar::UNSIGNED_CHAR, {sizeof(NeighborPoint) * hostNbr[tid].size()});
        inAdj.buildTensor(graph, poplar::UNSIGNED_SHORT, {hostAdj[tid].size()});

        graph.setTileMapping(inLocal.get(), tid);
        graph.setTileMapping(inNbr.get(), tid);
        graph.setTileMapping(inAdj.get(), tid);

        auto v = graph.addVertex(traceCS, "RayTrace");
        graph.connect(v["local_pts"], inLocal.get());
        graph.connect(v["neighbor_pts"], inNbr.get());
        graph.connect(v["adjacency"], inAdj.get());

        auto raysA_slice = raysA.slice(tid * rayBytesPerTile, (tid + 1) * rayBytesPerTile);
        auto raysB_slice = raysB.slice(tid * rayBytesPerTile, (tid + 1) * rayBytesPerTile);

        graph.setTileMapping(raysA_slice, tid);
        graph.setTileMapping(raysB_slice, tid);

        zeroSeq.add(poplar::program::Copy(zeroConst, raysA_slice));
        zeroSeq.add(poplar::program::Copy(zeroConst, raysB_slice));

        graph.connect(v["raysIn"],  raysB_slice);
        graph.connect(v["raysOut"], raysA_slice);

        graph.connect(v["result_float"], resultFloatAll.slice({tid},{tid+1}).reshape({}));
        graph.connect(v["result_u16"], resultU16All.slice({tid},{tid+1}).reshape({}));
        graph.connect(v["framebuffer"], framebufferAll.slice({tid,0},{tid+1,kTileFramebufferSize}).reshape({kTileFramebufferSize}));

        auto tileConst = graph.addConstant(poplar::UNSIGNED_SHORT, {}, tid);
        graph.setTileMapping(tileConst, tid);
        graph.connect(v["tile_id"], tileConst);

        graph.setTileMapping(v, tid);

        perTileWrites.add(inLocal.buildWrite(graph, true));
        perTileWrites.add(inNbr.buildWrite(graph, true));
        perTileWrites.add(inAdj.buildWrite(graph, true));

        hostLocalT.push_back(std::move(inLocal));
        hostNbrT.push_back(std::move(inNbr));
        hostAdjT.push_back(std::move(inAdj));
    }

    getPrograms().add("RayTraceCS", poplar::program::Execute(traceCS));

    // RayGen vertex
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

    // Register zero_rays and write sequences
    getPrograms().add("zero_rays", zeroSeq);
    getPrograms().add("write", perTileWrites);

    // Build one unified read program for all tiles
    ipu_utils::StreamableTensor fbReadAll("fb_read_all");
    fbReadAll = framebufferAll.reshape({kNumTraceTiles * kTileFramebufferSize});
    auto prog = fbReadAll.buildRead(graph, true);
    getPrograms().add("fb_read_all", prog);

}

// -----------------------------------------------------------------------------
// Execute graph
inline void RadfoamBuilder::execute(poplar::Engine& engine,
                                    const poplar::Device& /*device*/) {
    if (!initialised) {
        for (unsigned tid = 0; tid < hostLocalT.size(); ++tid) {
            hostLocalT[tid].connectWriteStream(engine, hostLocal[tid]);
            hostNbrT[tid].connectWriteStream(engine, hostNbr[tid]);
            hostAdjT[tid].connectWriteStream(engine, hostAdj[tid]);
        }

        engine.run(getPrograms().getOrdinals().at("zero_rays"));
        engine.run(getPrograms().getOrdinals().at("write"));
        initialised = true;
    }

    engine.run(getPrograms().getOrdinals().at("RayGenCS"));
    engine.run(getPrograms().getOrdinals().at("RayTraceCS"));

    // After compute, read all tile data
    readAllTiles(engine);
}

// -----------------------------------------------------------------------------
// Read all tiles
void RadfoamBuilder::readAllTiles(poplar::Engine &engine) {
    framebuffer_all_tiles.resize(kTileFramebufferSize * kNumTraceTiles);

    ipu_utils::StreamableTensor fbReadAll("fb_read_all");
    fbReadAll.connectReadStream(engine, framebuffer_all_tiles.data());
    engine.run(getPrograms().getOrdinals().at("fb_read_all"));

}

// -----------------------------------------------------------------------------
// Assemble full image
cv::Mat assembleFullImage(const std::vector<uint8_t>& tiles) {
    cv::Mat fullImage((int)kFullImageHeight, (int)kFullImageWidth, CV_8UC3);

    for (int ty = 0; ty < kNumTilesY; ++ty) {
        for (int tx = 0; tx < kNumTilesX; ++tx) {
            int tileIndex = ty * kNumTilesX + tx;
            const uint8_t* tileData = tiles.data() + tileIndex * kTileFramebufferSize;

            for (int y = 0; y < kTileHeight; ++y) {
                uint8_t* dstRow = fullImage.ptr<uint8_t>(ty * kTileHeight + y) + (tx * kTileWidth * 3);
                const uint8_t* srcRow = tileData + y * kTileWidth * 3;
                std::memcpy(dstRow, srcRow, kTileWidth * 3);
            }
        }
    }
    return fullImage;
}

// -----------------------------------------------------------------------------
// Main
int main(int argc, char** argv) {
    pvti::TraceChannel traceChannel = {"radfoam_ipu_tracer"};
    const std::string inputFile = argc > 1 ? argv[1] : "./data/garden.h5";
    const int tileToCompute = 462;
    RadfoamBuilder builder(inputFile, tileToCompute);

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
        {"debug.instrument", "false"}
    });

    // Assemble and save full framebuffer
    cv::Mat fullImage = assembleFullImage(builder.framebuffer_all_tiles);
    cv::imwrite("framebuffer_full.png", fullImage);

    return exit_code;
}
