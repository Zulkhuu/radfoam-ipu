#include "RadiantFoamIpuBuilder.hpp"

#include <spdlog/fmt/fmt.h>
#include <poplar/Program.hpp>

// Utility macro for logging inside the class implementation
#define RF_LOG(...) ipu_utils::logger()->info(__VA_ARGS__)

template <typename T>
static void SetInit(poplar::Graph& g, const poplar::Tensor& t, T value) {
  std::vector<T> tmp(t.numElements(), value);
  g.setInitialValue(t, poplar::ArrayRef<T>(tmp));
}

using namespace radfoam::ipu;
using namespace radfoam::geometry;
using namespace radfoam::config;
using ipu_utils::logger;
using poplar::DebugContext;

RadiantFoamIpuBuilder::RadiantFoamIpuBuilder(std::string h5_scene_file, bool debug)
    : h5_file_(std::move(h5_scene_file)),
      debug_(debug) {
        exec_counter_ = 0;
      }

void RadiantFoamIpuBuilder::updateViewMatrix(const glm::mat4& m) {
    const float* ptr = glm::value_ptr(m);
    for (size_t i = 0; i < 16; ++i) {
        hostViewMatrix_[i] = ptr[i];
    }
}

void RadiantFoamIpuBuilder::updateProjectionMatrix(const glm::mat4& m) {
    const float* ptr = glm::value_ptr(m);
    for (size_t i = 0; i < 16; ++i) {
        hostProjMatrix_[i] = ptr[i];
    }
}

void RadiantFoamIpuBuilder::updateCameraCell(const GenericPoint& cell) {
    hostCameraCellInfo_[0] = static_cast<uint8_t>( cell.cluster_id        & 0xFF);
    hostCameraCellInfo_[1] = static_cast<uint8_t>((cell.cluster_id >> 8)  & 0xFF);
    hostCameraCellInfo_[2] = static_cast<uint8_t>( cell.local_id          & 0xFF);
    hostCameraCellInfo_[3] = static_cast<uint8_t>((cell.local_id   >> 8)  & 0xFF);
}

// ─────────────────────────────────────────────────────────────────────────────
//  build() – graph construction (called exactly once)
// ─────────────────────────────────────────────────────────────────────────────
void RadiantFoamIpuBuilder::build(poplar::Graph& g, const poplar::Target&) {
    RF_LOG("Building RadiantFoam graph");

    loadScenePartitions();
    registerCodeletsAndOps(g);
    allocateGlobalTensors(g);

    poplar::ComputeSet cs = g.addComputeSet("RayTraceCS");

    buildRayTracers(g, cs);
    buildRayRoutersL0(g, cs);
    buildRayRoutersL1(g, cs);
    buildRayRoutersL2(g, cs);
    buildRayRoutersL3(g, cs);
    buildRayRoutersL4(g, cs);
    buildRayGenerator(g, cs);
    buildDataExchange(g);
    setupHostStreams(g);

    poplar::program::Sequence frame_;
    // frame_.add(poplar::program::Repeat(kSubsteps, frameStep_));
    frame_.add(poplar::program::Execute(cs));
    frame_.add(data_exchange_seq);
    frame_.add(poplar::program::Copy(inStreamFinishedRays, inStream));
    getPrograms().add("frame", frame_);

    stopFlag_.buildTensor(g, poplar::UNSIGNED_INT, {});
    g.setTileMapping(stopFlag_.get(), 0);
    g.setInitialValue(stopFlag_.get(), poplar::ArrayRef<unsigned>({1}));
    auto condProgram = stopFlag_.buildWrite(g, true);
    // poplar::program::Sequence condProgram2;
    // auto condProgram = poplar::program::Copy(stopFlagStream, stopFlag_);
    auto frameLoop = poplar::program::RepeatWhileTrue(condProgram, stopFlag_.get(), frame_);
    getPrograms().add("frame_loop", frameLoop);
}

// ----------------------------------------------------------------------------
//  execute() – per‑frame invocation
// ----------------------------------------------------------------------------
void RadiantFoamIpuBuilder::execute(poplar::Engine& eng, const poplar::Device&) {
    if (exec_counter_ == 0) {
        connectHostStreams(eng);

        eng.run(getPrograms().getOrdinals().at("write_scene_data"),
            fmt::format("frame_{:03d}/write_scene_data", exec_counter_));
        eng.run(getPrograms().getOrdinals().at("broadcast_matrices"),
            fmt::format("frame_{:03d}/broadcast_matrices", exec_counter_));
        eng.run(getPrograms().getOrdinals().at("write_camera_cell_info"),
            fmt::format("frame_{:03d}/write_camera_cell_info", exec_counter_));

        eng.run(getPrograms().getOrdinals().at("frame_loop"));
    }

    // eng.run(getPrograms().getOrdinals().at("frame"),
    //         fmt::format("frame_{:03d}", exec_counter_));
    
    // readAllTiles(eng);

    // if(debug_)

    exec_counter_++;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Private helper implementations
// ─────────────────────────────────────────────────────────────────────────────

void RadiantFoamIpuBuilder::loadScenePartitions() {
    local_pts_.resize(kNumRayTracerTiles);
    neighbor_pts_.resize(kNumRayTracerTiles);
    adjacency_.resize(kNumRayTracerTiles);
    paths_.resize(kNumRayTracerTiles);

    HighFive::File f(h5_file_, HighFive::File::ReadOnly);
    for (size_t tid = 0; tid < kNumRayTracerTiles; ++tid) {
        const auto g = f.getGroup(fmt::format("part{:04}", tid));
        g.getDataSet("local_pts")       .read(local_pts_[tid]);
        g.getDataSet("neighbor_pts")    .read(neighbor_pts_[tid]);
        g.getDataSet("adjacency_list")  .read(adjacency_[tid]);
        g.getDataSet("path")            .read(paths_[tid]);
    }
}

void RadiantFoamIpuBuilder::registerCodeletsAndOps(poplar::Graph& g) {
    const std::string codeletFile = std::string(POPC_PREFIX) + "/src/codelets/codelets.cpp";
    const std::string incPath     = std::string(POPC_PREFIX) + "/include/";
    const std::string glmPath     = std::string(POPC_PREFIX) + "/external/glm/";
    g.addCodelets(codeletFile, poplar::CodeletFileType::Auto,
                  "-O3 -finline-functions -funroll-loops -I " + incPath + " -I " + glmPath);
    popops::addCodelets(g);
}

void RadiantFoamIpuBuilder::allocateGlobalTensors(poplar::Graph& g) {
    const size_t kRayIOBytesPerTile = kNumRays * sizeof(Ray);

    result_f32_read_.buildTensor(g, poplar::FLOAT,          {kNumRayTracerTiles});
    poputil::mapTensorLinearlyWithOffset(g, result_f32_read_.get(), 0);

    result_u16_read_.buildTensor(g, poplar::UNSIGNED_SHORT, {kNumRayTracerTiles});
    poputil::mapTensorLinearlyWithOffset(g, result_u16_read_.get(), 0);

    fb_read_all_.buildTensor(g,  poplar::UNSIGNED_CHAR, {kNumRayTracerTiles, kTileFramebufferSize});
    poputil::mapTensorLinearlyWithOffset(g, fb_read_all_.get().reshape({kNumRayTracerTiles, kTileFramebufferSize}), 0);
    
    const size_t kFinishedRayBytesPerTile = kNumRays * kFinishedFactor * sizeof(FinishedRay);
    // finishedRaysRead_.buildTensor(g,  poplar::UNSIGNED_CHAR, {kNumRayTracerTiles * kFinishedRayBytesPerTile});

    exec_counts_.buildTensor(g, poplar::UNSIGNED_INT, {kNumRayTracerTiles + 1});
    for (std::size_t tid = 0; tid < kNumRayTracerTiles; ++tid) {
        g.setTileMapping(exec_counts_.get().slice({tid},{tid+1}).reshape({}), tid);
    }
    g.setTileMapping(exec_counts_.get().slice({kNumRayTracerTiles},{kNumRayTracerTiles+1}).reshape({}), kRaygenTile);
    std::vector<unsigned> zeros(kNumRayTracerTiles + 1, 0u);
    g.setInitialValue(exec_counts_.get(), poplar::ArrayRef<unsigned>(zeros));

    finishedWriteOffsets_.buildTensor(g, poplar::UNSIGNED_INT, {kNumRayTracerTiles});
    poputil::mapTensorLinearlyWithOffset(g, finishedWriteOffsets_.get(), 0);
    std::vector<unsigned> zeros_offsets(kNumRayTracerTiles, 0u);
    g.setInitialValue(finishedWriteOffsets_.get(), poplar::ArrayRef<unsigned>(zeros_offsets));
  
    viewMatrix_.buildTensor(g, poplar::FLOAT, {4,4});
    g.setTileMapping(viewMatrix_.get(), 0);
    broadcastMatrices_.add(viewMatrix_.buildWrite(g, true));

    projMatrix_.buildTensor(g, poplar::FLOAT, {4,4});
    g.setTileMapping(projMatrix_.get(), 0);
    broadcastMatrices_.add(projMatrix_.buildWrite(g, true));

    inStreamFinishedRays = g.addVariable(poplar::UNSIGNED_CHAR, {kNumRayTracerTiles * kFinishedRayBytesPerTile}, "instream-finished-rays");
    poplar::OptionFlags inOpts = {
        {"bufferingDepth", "2"},
        {"splitLimit", std::to_string(100 * 1024 * 1024)} // 100 MiB
    };
    inStream =  g.addDeviceToHostFIFO(
        "read-finished-rays-stream",
        poplar::UNSIGNED_CHAR,
        kNumRayTracerTiles * kFinishedRayBytesPerTile,
        inOpts);

}

void RadiantFoamIpuBuilder::buildRayTracers(poplar::Graph& g, poplar::ComputeSet& cs) {
    const size_t kRayIOBytesPerTile = kNumRays * sizeof(Ray);
    const size_t kRayTracerIoBytesAllTiles = kRayIOBytesPerTile * kNumRayTracerTiles;
    const size_t kFinishedRayBytesPerTile = kNumRays * kFinishedFactor * sizeof(FinishedRay);

    rayTracerOutputRays_ = g.addVariable(poplar::UNSIGNED_CHAR, {kRayTracerIoBytesAllTiles}, "rt_out");
    rayTracerInputRays_  = g.addVariable(poplar::UNSIGNED_CHAR, {kRayTracerIoBytesAllTiles}, "rt_in");
    SetInit<uint8_t>(g, rayTracerOutputRays_,  0xFF);
    SetInit<uint8_t>(g, rayTracerInputRays_,  0xFF);

    std::vector<poplar::Tensor> viewDests;  
    viewDests.reserve(kNumRayTracerTiles);

    std::vector<poplar::Tensor> projDests;  
    projDests.reserve(kNumRayTracerTiles);

    for (size_t tid = 0; tid < kNumRayTracerTiles; ++tid) {
        // H2D tensors --------------------------------------------------------
        ipu_utils::StreamableTensor in_local (fmt::format("local_{}", tid));
        ipu_utils::StreamableTensor in_nbr   (fmt::format("nbr_{}",   tid));
        ipu_utils::StreamableTensor in_adj   (fmt::format("adj_{}",   tid));

        in_local.buildTensor(g, poplar::UNSIGNED_CHAR, {sizeof(LocalPoint)  * local_pts_[tid].size()});
        in_nbr  .buildTensor(g, poplar::UNSIGNED_CHAR, {sizeof(GenericPoint) * neighbor_pts_[tid].size()});
        in_adj  .buildTensor(g, poplar::UNSIGNED_SHORT,{adjacency_[tid].size()});

        g.setTileMapping(in_local.get(), tid);
        g.setTileMapping(in_nbr.get(),   tid);
        g.setTileMapping(in_adj.get(),   tid);

        auto v = g.addVertex(cs, "RayTrace");

        // Broadcasted matrices (tile‑local clones)
        auto localView = g.clone(viewMatrix_.get(), "view_mat_t"+std::to_string(tid));
        g.setTileMapping(localView, tid);
        auto localProj = g.clone(projMatrix_.get(), "proj_mat_t"+std::to_string(tid));
        g.setTileMapping(localProj, tid);

        g.connect(v["view_matrix"],      localView.flatten());
        g.connect(v["projection_matrix"],localProj.flatten());

        // Save flattened 1×16 views for one-shot big copy later
        viewDests.push_back(localView.reshape({1, 16}));
        projDests.push_back(localProj.reshape({1, 16}));

        // Connect point clouds / adjacency
        g.connect(v["local_pts"],    in_local.get());
        g.connect(v["neighbor_pts"], in_nbr.get());
        g.connect(v["adjacency"],    in_adj.get());

        // Ray IO slices ------------------------------------------------------
        const auto out_slice = rayTracerOutputRays_.slice(tid * kRayIOBytesPerTile,
                                                          (tid+1)*kRayIOBytesPerTile);
        const auto in_slice  = rayTracerInputRays_ .slice(tid * kRayIOBytesPerTile,
                                                          (tid+1)*kRayIOBytesPerTile);
        g.setTileMapping(out_slice, tid);
        g.setTileMapping(in_slice,  tid);
        g.connect(v["raysOut"], out_slice);
        g.connect(v["raysIn" ], in_slice);

        // Per‑tile result scalars & framebuffer slice -----------------------
        g.connect(v["result_float"], result_f32_read_.get().slice({tid},{tid+1}).reshape({}));
        g.connect(v["result_u16"  ], result_u16_read_.get().slice({tid},{tid+1}).reshape({}));
        auto fb_slice = fb_read_all_.get()
                            .slice({tid,0},{tid+1,kTileFramebufferSize})
                            .reshape({kTileFramebufferSize});
        g.connect(v["framebuffer"], fb_slice);

        // auto finished_slice = finishedRaysRead_.get().slice(
        //     tid * kFinishedRayBytesPerTile, (tid + 1) * kFinishedRayBytesPerTile);
        auto finished_slice = inStreamFinishedRays.slice(
            tid * kFinishedRayBytesPerTile, (tid + 1) * kFinishedRayBytesPerTile);
        g.setTileMapping(finished_slice, tid);
        g.connect(v["finishedRays"], finished_slice);

        // Constant tile_id param
        auto tile_const = g.addConstant(poplar::UNSIGNED_SHORT, {}, static_cast<unsigned>(tid));
        g.setTileMapping(tile_const, tid);
        g.connect(v["tile_id"], tile_const);
        g.setTileMapping(v, tid);

        auto execCnt = exec_counts_.get().slice({tid},{tid+1}).reshape({});
        g.connect(v["exec_count"], execCnt);

        auto writeOffs = finishedWriteOffsets_.get().slice({tid},{tid+1}).reshape({});
        g.connect(v["finishedWriteOffset"],  writeOffs);

        per_tile_writes_.add(in_local.buildWrite(g, true));
        per_tile_writes_.add(in_nbr  .buildWrite(g, true));
        per_tile_writes_.add(in_adj  .buildWrite(g, true));

        local_tensors_.push_back(std::move(in_local));
        neighbor_tensors_.push_back(std::move(in_nbr));
        adj_tensors_.push_back(std::move(in_adj));
    }

    // Source: 1×16 broadcast to [kNumTiles,16]; Dest: concat of all per-tile clones → [kNumTiles,16]
    auto srcView = viewMatrix_.get().reshape({1, 16}).broadcast(kNumRayTracerTiles, 0);
    auto dstView = poplar::concat(viewDests, 0);
    broadcastMatrices_.add(poplar::program::Copy(srcView, dstView, /*dontOutline*/true,
                                                DebugContext{"broadcast_matrices/view"}));

    auto srcProj = projMatrix_.get().reshape({1, 16}).broadcast(kNumRayTracerTiles, 0);
    auto dstProj = poplar::concat(projDests, 0);
    broadcastMatrices_.add(poplar::program::Copy(srcProj, dstProj, /*dontOutline*/true,
                                                DebugContext{"broadcast_matrices/proj"}));

    // getPrograms().add("RayTraceCSExecution", poplar::program::Execute(cs));
    getPrograms().add("broadcast_matrices", broadcastMatrices_);
    getPrograms().add("write_scene_data", per_tile_writes_);
}

void RadiantFoamIpuBuilder::buildRayGenerator(poplar::Graph& g, poplar::ComputeSet& cs) {
    const size_t kRayIOBytesPerTile = kNumRays * sizeof(Ray);
    const unsigned kPendingFactor = 2;

    auto v = g.addVertex(cs, "RayGen");
    raygenInput  = g.addVariable(poplar::UNSIGNED_CHAR,{kRayIOBytesPerTile}, "raygen_in");
    raygenOutput = g.addVariable(poplar::UNSIGNED_CHAR,{kRayIOBytesPerTile}, "raygen_out");
    SetInit<uint8_t>(g, raygenInput,  0xFF);
    SetInit<uint8_t>(g, raygenOutput, 0xFF);
    g.setTileMapping(raygenInput,  kRaygenTile);
    g.setTileMapping(raygenOutput, kRaygenTile);
    g.connect(v["RaysIn"],  raygenInput);
    g.connect(v["RaysOut"], raygenOutput);

    auto rgPending = g.addVariable(poplar::UNSIGNED_CHAR, {kPendingFactor * kRayIOBytesPerTile}, "rg_pending_rays");
    SetInit<uint8_t>(g, rgPending, 0xFF); 
    g.setTileMapping(rgPending, kRaygenTile);
    g.connect(v["pendingRays"],  rgPending);

    auto rgHead = g.addVariable(poplar::UNSIGNED_INT, {}, "rg_pending_head");
    auto rgTail = g.addVariable(poplar::UNSIGNED_INT, {}, "rg_pending_tail");
    g.setInitialValue(rgHead, poplar::ArrayRef<unsigned>({0u}));
    g.setInitialValue(rgTail, poplar::ArrayRef<unsigned>({0u}));
    g.setTileMapping(rgHead, kRaygenTile);
    g.setTileMapping(rgTail, kRaygenTile);
    g.connect(v["pendingHead"], rgHead);
    g.connect(v["pendingTail"], rgTail);

    cameraCellInfo_.buildTensor(g, poplar::UNSIGNED_CHAR,{4});
    g.setTileMapping(cameraCellInfo_.get(),kRaygenTile);
    g.connect(v["camera_cell_info"], cameraCellInfo_.get());
    getPrograms().add("write_camera_cell_info", cameraCellInfo_.buildWrite(g,true));
    
    auto raygen_exec = exec_counts_.get().slice({kNumRayTracerTiles},{kNumRayTracerTiles+1}).reshape({});
    g.connect(v["exec_count"], raygen_exec);

    raygenDebugRead_.buildTensor(g, poplar::UNSIGNED_CHAR, {kRouterDebugSize});
    g.setTileMapping(raygenDebugRead_.get(), kRaygenTile);
    g.connect(v["debugBytes"], raygenDebugRead_.get());

    g.setTileMapping(v, kRaygenTile);

    frameStep_.add(poplar::program::Execute(cs));
}

void RadiantFoamIpuBuilder::buildRayRoutersL0(poplar::Graph& g, poplar::ComputeSet& cs) {
    constexpr uint16_t router_tile_offset = 1024;
    constexpr size_t   kChildrenPerRouter = 4;
    constexpr size_t   kRayIOBytesPerTile = kNumRays * sizeof(Ray);

    const size_t kRouterPerTileBuffer = kRayIOBytesPerTile * 5; // parent + 4 children
    const size_t kTotalBuffer         = kRouterPerTileBuffer * kNumL0RouterTiles;

    L0RouterOut = g.addVariable(poplar::UNSIGNED_CHAR,{kTotalBuffer},"router_out");
    L0RouterIn  = g.addVariable(poplar::UNSIGNED_CHAR,{kTotalBuffer},"router_in");
    SetInit<uint8_t>(g, L0RouterOut,  0xFF);
    SetInit<uint8_t>(g, L0RouterIn,  0xFF);

    l0routerDebugRead_.buildTensor(g, poplar::UNSIGNED_CHAR,{kNumL0RouterTiles,kRouterDebugSize});
    poputil::mapTensorLinearlyWithOffset(g,
        l0routerDebugRead_.get().reshape({kNumL0RouterTiles,kRouterDebugSize}),
        router_tile_offset);

    std::vector<uint16_t> clusterIds;
    clusterIds.reserve(kNumL0RouterTiles * kChildrenPerRouter);
    for (size_t r = 0; r < kNumL0RouterTiles; ++r) {
        for (size_t i = 0; i < kChildrenPerRouter; ++i) {
            clusterIds.push_back(static_cast<uint16_t>(r * kChildrenPerRouter + i));
        }
    }

    for (size_t router_id = 0; router_id < kNumL0RouterTiles; ++router_id) {
        const uint16_t tile = router_tile_offset + router_id;
        auto v = g.addVertex(cs, "RayRouter");

        // Slice IO buffers ---------------------------------------------------
        const size_t base = router_id * kRouterPerTileBuffer;
        auto sliceIn  = [&](size_t idx){
            return L0RouterIn.slice (base+idx*kRayIOBytesPerTile, base+(idx+1)*kRayIOBytesPerTile);} ;
        auto sliceOut = [&](size_t idx){
            return L0RouterOut.slice(base+idx*kRayIOBytesPerTile, base+(idx+1)*kRayIOBytesPerTile);} ;
        auto parentIn  = sliceIn(0);
        auto parentOut = sliceOut(0);
        g.setTileMapping(parentIn, tile); 
        g.setTileMapping(parentOut, tile);
        g.connect(v["parentRaysIn" ], parentIn);
        g.connect(v["parentRaysOut"], parentOut);

        for (int i = 0; i < 4; ++i) {
            auto in  = sliceIn(i+1);
            auto out = sliceOut(i+1);
            g.setTileMapping(in,  tile);
            g.setTileMapping(out, tile);
            g.connect(v[fmt::format("childRaysIn{}",  i)], in);
            g.connect(v[fmt::format("childRaysOut{}", i)], out);
        }

        // Constant child cluster IDs (4)
        poplar::Tensor idsConst = g.addConstant(poplar::UNSIGNED_SHORT,{4},
                                                clusterIds.data() + router_id*4);
        g.setTileMapping(idsConst, tile);
        g.connect(v["childClusterIds"], idsConst);

        auto levelConst = g.addConstant(poplar::UNSIGNED_CHAR, {}, 0);
        g.setTileMapping(levelConst, tile);
        g.connect(v["level"], levelConst);

        const auto kNumWorkers = 6;
        auto sharedCounts  = g.addVariable(poplar::UNSIGNED_INT, {kNumWorkers*5}, "sharedCounts");
        auto sharedOffsets = g.addVariable(poplar::UNSIGNED_INT, {kNumWorkers*5 + 2*kNumWorkers}, "sharedOffsets");
        auto readyFlags    = g.addVariable(poplar::UNSIGNED_INT, {kNumWorkers}, "readyFlags");
        g.setTileMapping(sharedCounts, tile);
        g.setTileMapping(sharedOffsets, tile);
        g.setTileMapping(readyFlags, tile);
        g.connect(v["sharedCounts"], sharedCounts);
        g.connect(v["sharedOffsets"], sharedOffsets);
        g.connect(v["readyFlags"], readyFlags);

        auto dbgSlice = l0routerDebugRead_.get()
                         .slice({router_id,0},{router_id+1,kRouterDebugSize})
                         .reshape({kRouterDebugSize});
        g.connect(v["debugBytes"], dbgSlice);
        g.setTileMapping(v, tile);
    }
}

void RadiantFoamIpuBuilder::buildRayRoutersL1(poplar::Graph& g, poplar::ComputeSet& cs) {
    constexpr uint16_t router_tile_offset = kNumRayTracerTiles + kNumL0RouterTiles; // 1024 + 256; 
    constexpr size_t   kChildrenPerRouter = 4;
    constexpr size_t   kRayIOBytesPerTile = kNumRays * sizeof(Ray);

    const size_t kRouterPerTileBuffer = kRayIOBytesPerTile * 5; // parent + 4 children
    const size_t kTotalBuffer         = kRouterPerTileBuffer * kNumL1RouterTiles;

    L1RouterOut = g.addVariable(poplar::UNSIGNED_CHAR, {kTotalBuffer}, "l1_router_out");
    L1RouterIn  = g.addVariable(poplar::UNSIGNED_CHAR, {kTotalBuffer}, "l1_router_in");
    SetInit<uint8_t>(g, L1RouterOut,  0xFF);
    SetInit<uint8_t>(g, L1RouterIn,  0xFF);

    l1routerDebugRead_.buildTensor(g, poplar::UNSIGNED_CHAR,{kNumL1RouterTiles,kRouterDebugSize});
    poputil::mapTensorLinearlyWithOffset(g,
        l1routerDebugRead_.get().reshape({kNumL1RouterTiles,kRouterDebugSize}),
        router_tile_offset
    );

    // --- Generate child cluster IDs with increment of 4 ---
    std::vector<uint16_t> clusterIds;
    clusterIds.reserve(kNumL1RouterTiles * kChildrenPerRouter);
    for (size_t r = 0; r < kNumL1RouterTiles; ++r) {
        for (size_t i = 0; i < kChildrenPerRouter; ++i) {
            clusterIds.push_back(static_cast<uint16_t>((r * 16) + i * 4));
        }
    }

    for (size_t router_id = 0; router_id < kNumL1RouterTiles; ++router_id) {
        const uint16_t tile = router_tile_offset + router_id;
        auto v = g.addVertex(cs, "RayRouter");

        // IO buffer slicing logic same as L0
        const size_t base = router_id * kRouterPerTileBuffer;
        auto sliceIn  = [&](size_t idx){return L1RouterIn.slice (base+idx*kRayIOBytesPerTile,
                                                                 base+(idx+1)*kRayIOBytesPerTile);} ;
        auto sliceOut = [&](size_t idx){return L1RouterOut.slice(base+idx*kRayIOBytesPerTile,
                                                                 base+(idx+1)*kRayIOBytesPerTile);} ;

        auto parentIn  = sliceIn(0);
        auto parentOut = sliceOut(0);
        g.setTileMapping(parentIn, tile);
        g.setTileMapping(parentOut, tile);
        g.connect(v["parentRaysIn" ], parentIn);
        g.connect(v["parentRaysOut"], parentOut);

        // Child buffers
        for (int i = 0; i < 4; ++i) {
            auto in  = sliceIn(i+1);
            auto out = sliceOut(i+1);
            g.setTileMapping(in,  tile);
            g.setTileMapping(out, tile);
            g.connect(v[fmt::format("childRaysIn{}",  i)], in);
            g.connect(v[fmt::format("childRaysOut{}", i)], out);
        }

        // Child cluster IDs
        poplar::Tensor idsConst = g.addConstant(poplar::UNSIGNED_SHORT, {4},
            clusterIds.data() + router_id * 4);
        g.setTileMapping(idsConst, tile);
        g.connect(v["childClusterIds"], idsConst);

        // Level constant (1 for L1)
        auto levelConst = g.addConstant(poplar::UNSIGNED_CHAR, {}, 1);
        g.setTileMapping(levelConst, tile);
        g.connect(v["level"], levelConst);

        const auto kNumWorkers = 6;
        auto sharedCounts  = g.addVariable(poplar::UNSIGNED_INT, {kNumWorkers*5}, "sharedCounts");
        auto sharedOffsets = g.addVariable(poplar::UNSIGNED_INT, {kNumWorkers*5 + 2*kNumWorkers}, "sharedOffsets");
        auto readyFlags    = g.addVariable(poplar::UNSIGNED_INT, {kNumWorkers}, "readyFlags");
        g.setTileMapping(sharedCounts, tile);
        g.setTileMapping(sharedOffsets, tile);
        g.setTileMapping(readyFlags, tile);
        g.connect(v["sharedCounts"], sharedCounts);
        g.connect(v["sharedOffsets"], sharedOffsets);
        g.connect(v["readyFlags"], readyFlags);

        auto dbgSlice = l1routerDebugRead_.get()
                         .slice({router_id,0},{router_id+1,kRouterDebugSize})
                         .reshape({kRouterDebugSize});
        g.connect(v["debugBytes"], dbgSlice);

        g.setTileMapping(v, tile);
    }
}

void RadiantFoamIpuBuilder::buildRayRoutersL2(poplar::Graph& g, poplar::ComputeSet& cs) {
    constexpr uint16_t router_tile_offset =
        kNumRayTracerTiles + kNumL0RouterTiles + kNumL1RouterTiles;
    constexpr size_t kChildrenPerRouter = 4;
    constexpr size_t kRayIOBytesPerTile = kNumRays * sizeof(Ray);

    const size_t kRouterPerTileBuffer = kRayIOBytesPerTile * 5; // parent + 4 children
    const size_t kTotalBuffer = kRouterPerTileBuffer * kNumL2RouterTiles;

    // Allocate IO tensors
    L2RouterOut = g.addVariable(poplar::UNSIGNED_CHAR, {kTotalBuffer}, "l2_router_out");
    L2RouterIn  = g.addVariable(poplar::UNSIGNED_CHAR, {kTotalBuffer}, "l2_router_in");
    SetInit<uint8_t>(g, L2RouterOut,  0xFF);
    SetInit<uint8_t>(g, L2RouterIn, 0xFF);

    // Debug bytes
    l2routerDebugRead_.buildTensor(g, poplar::UNSIGNED_CHAR, {kNumL2RouterTiles, kRouterDebugSize});
    poputil::mapTensorLinearlyWithOffset(g,
        l2routerDebugRead_.get().reshape({kNumL2RouterTiles, kRouterDebugSize}),
        router_tile_offset
    );

    // Child cluster IDs: spacing by 64 (since L2 handles 64 clusters)
    std::vector<uint16_t> clusterIds;
    clusterIds.reserve(kNumL2RouterTiles * kChildrenPerRouter);
    for (size_t r = 0; r < kNumL2RouterTiles; ++r) {
        for (size_t i = 0; i < kChildrenPerRouter; ++i) {
            clusterIds.push_back(static_cast<uint16_t>((r * 64) + i * 16));
        }
    }

    for (size_t router_id = 0; router_id < kNumL2RouterTiles; ++router_id) {
        const uint16_t tile = router_tile_offset + router_id;
        auto v = g.addVertex(cs, "RayRouter");

        const size_t base = router_id * kRouterPerTileBuffer;
        auto sliceIn  = [&](size_t idx){
            return L2RouterIn.slice (base+idx*kRayIOBytesPerTile, base+(idx+1)*kRayIOBytesPerTile);};
        auto sliceOut = [&](size_t idx){
            return L2RouterOut.slice(base+idx*kRayIOBytesPerTile, base+(idx+1)*kRayIOBytesPerTile);};

        // Parent IO
        auto parentIn  = sliceIn(0);
        auto parentOut = sliceOut(0);
        g.setTileMapping(parentIn, tile);
        g.setTileMapping(parentOut, tile);
        g.connect(v["parentRaysIn"], parentIn);
        g.connect(v["parentRaysOut"], parentOut);

        // Child IO
        for (int i = 0; i < 4; ++i) {
            auto in  = sliceIn(i+1);
            auto out = sliceOut(i+1);
            g.setTileMapping(in, tile);
            g.setTileMapping(out, tile);
            g.connect(v[fmt::format("childRaysIn{}", i)], in);
            g.connect(v[fmt::format("childRaysOut{}", i)], out);
        }

        // Child cluster IDs
        poplar::Tensor idsConst = g.addConstant(poplar::UNSIGNED_SHORT, {4},
            clusterIds.data() + router_id * 4);
        g.setTileMapping(idsConst, tile);
        g.connect(v["childClusterIds"], idsConst);

        // Level constant = 2
        auto levelConst = g.addConstant(poplar::UNSIGNED_CHAR, {}, 2);
        g.setTileMapping(levelConst, tile);
        g.connect(v["level"], levelConst);

        const auto kNumWorkers = 6;
        auto sharedCounts  = g.addVariable(poplar::UNSIGNED_INT, {kNumWorkers*5}, "sharedCounts");
        auto sharedOffsets = g.addVariable(poplar::UNSIGNED_INT, {kNumWorkers*5 + 2*kNumWorkers}, "sharedOffsets");
        auto readyFlags    = g.addVariable(poplar::UNSIGNED_INT, {kNumWorkers}, "readyFlags");
        g.setTileMapping(sharedCounts, tile);
        g.setTileMapping(sharedOffsets, tile);
        g.setTileMapping(readyFlags, tile);
        g.connect(v["sharedCounts"], sharedCounts);
        g.connect(v["sharedOffsets"], sharedOffsets);
        g.connect(v["readyFlags"], readyFlags);

        auto dbgSlice = l2routerDebugRead_.get()
                         .slice({router_id,0},{router_id+1,kRouterDebugSize})
                         .reshape({kRouterDebugSize});
        g.connect(v["debugBytes"], dbgSlice);

        g.setTileMapping(v, tile);
    }
}

void RadiantFoamIpuBuilder::buildRayRoutersL3(poplar::Graph& g, poplar::ComputeSet& cs) {
    constexpr uint16_t L3RouterTileOffset =
        kNumRayTracerTiles + kNumL0RouterTiles + kNumL1RouterTiles + kNumL2RouterTiles;
    constexpr size_t kChildrenPerRouter = 4;
    constexpr size_t kRayIOBytesPerTile = kNumRays * sizeof(Ray);

    const size_t kRouterPerTileBuffer = kRayIOBytesPerTile * 5; // parent + 4 children
    const size_t kTotalBuffer = kRouterPerTileBuffer * kNumL3RouterTiles;

    L3RouterOut = g.addVariable(poplar::UNSIGNED_CHAR, {kTotalBuffer}, "l3_router_out");
    L3RouterIn  = g.addVariable(poplar::UNSIGNED_CHAR, {kTotalBuffer}, "l3_router_in");
    SetInit<uint8_t>(g, L3RouterOut,  0xFF);
    SetInit<uint8_t>(g, L3RouterIn, 0xFF);

    l3routerDebugRead_.buildTensor(g, poplar::UNSIGNED_CHAR, {kNumL3RouterTiles, kRouterDebugSize});
    poputil::mapTensorLinearlyWithOffset(
        g,
        l3routerDebugRead_.get().reshape({kNumL3RouterTiles, kRouterDebugSize}),
        L3RouterTileOffset
    );

    // Generate cluster IDs (increment by 64 per L3 router, as each step quadruples)
    std::vector<uint16_t> clusterIds;
    clusterIds.reserve(kNumL3RouterTiles * kChildrenPerRouter);
    for (size_t r = 0; r < kNumL3RouterTiles; ++r) {
        for (size_t i = 0; i < kChildrenPerRouter; ++i) {
            clusterIds.push_back(static_cast<uint16_t>((r * 256) + i * 64));
        }
    }

    for (size_t router_id = 0; router_id < kNumL3RouterTiles; ++router_id) {
        const uint16_t tile = L3RouterTileOffset + router_id;
        auto v = g.addVertex(cs, "RayRouter");

        // IO buffer slicing
        const size_t base = router_id * kRouterPerTileBuffer;
        auto sliceIn  = [&](size_t idx){ 
            return L3RouterIn.slice(base + idx*kRayIOBytesPerTile, base + (idx+1)*kRayIOBytesPerTile); };
        auto sliceOut = [&](size_t idx){ 
            return L3RouterOut.slice(base + idx*kRayIOBytesPerTile, base + (idx+1)*kRayIOBytesPerTile); };

        // Parent IO
        auto parentIn = sliceIn(0);
        auto parentOut = sliceOut(0);
        g.setTileMapping(parentIn, tile);
        g.setTileMapping(parentOut, tile);
        g.connect(v["parentRaysIn"], parentIn);
        g.connect(v["parentRaysOut"], parentOut);

        // Child IO
        for (int i = 0; i < 4; ++i) {
            auto in = sliceIn(i+1);
            auto out = sliceOut(i+1);
            g.setTileMapping(in, tile);
            g.setTileMapping(out, tile);
            g.connect(v[fmt::format("childRaysIn{}", i)], in);
            g.connect(v[fmt::format("childRaysOut{}", i)], out);
        }

        // Child cluster IDs
        poplar::Tensor idsConst = g.addConstant(poplar::UNSIGNED_SHORT, {4},
            clusterIds.data() + router_id * 4);
        g.setTileMapping(idsConst, tile);
        g.connect(v["childClusterIds"], idsConst);

        // Level constant = 3
        auto levelConst = g.addConstant(poplar::UNSIGNED_CHAR, {}, 3);
        g.setTileMapping(levelConst, tile);
        g.connect(v["level"], levelConst);

        // Debug slice
        auto dbgSlice = l3routerDebugRead_.get()
            .slice({router_id, 0}, {router_id+1, kRouterDebugSize})
            .reshape({kRouterDebugSize});
        g.connect(v["debugBytes"], dbgSlice);

        const auto kNumWorkers = 6;
        auto sharedCounts  = g.addVariable(poplar::UNSIGNED_INT, {kNumWorkers*5}, "sharedCounts");
        auto sharedOffsets = g.addVariable(poplar::UNSIGNED_INT, {kNumWorkers*5 + 2*kNumWorkers}, "sharedOffsets");
        auto readyFlags    = g.addVariable(poplar::UNSIGNED_INT, {kNumWorkers}, "readyFlags");
        g.setTileMapping(sharedCounts, tile);
        g.setTileMapping(sharedOffsets, tile);
        g.setTileMapping(readyFlags, tile);
        g.connect(v["sharedCounts"], sharedCounts);
        g.connect(v["sharedOffsets"], sharedOffsets);
        g.connect(v["readyFlags"], readyFlags);

        g.setTileMapping(v, tile);
    }
}

void RadiantFoamIpuBuilder::buildRayRoutersL4(poplar::Graph& g, poplar::ComputeSet& cs) {
  constexpr uint16_t L4TileOffset =
      kNumRayTracerTiles + kNumL0RouterTiles + kNumL1RouterTiles + kNumL2RouterTiles + kNumL3RouterTiles;
  constexpr size_t kChildrenPerRouter = 4;
  constexpr size_t kRayIOBytesPerTile = kNumRays * sizeof(Ray);
  const size_t kRouterPerTileBuffer = kRayIOBytesPerTile * 5; // parent + 4 children
  const size_t kTotalBuffer = kRouterPerTileBuffer * 1;       // single L4 router

  L4RouterOut = g.addVariable(poplar::UNSIGNED_CHAR, {kTotalBuffer}, "l4_router_out");
  L4RouterIn  = g.addVariable(poplar::UNSIGNED_CHAR, {kTotalBuffer}, "l4_router_in");
  SetInit<uint8_t>(g, L4RouterOut, 0xFF);
  SetInit<uint8_t>(g, L4RouterIn,  0xFF);

  l4routerDebugRead_.buildTensor(g, poplar::UNSIGNED_CHAR, {kNumL4RouterTiles, kRouterDebugSize});
  poputil::mapTensorLinearlyWithOffset(
      g,
      l4routerDebugRead_.get().reshape({kNumL4RouterTiles, kRouterDebugSize}),
      L4TileOffset);
  // cluster ids for level 4: r*4^4 + i*4^3  → r=0 → {0,256,512,768}
  std::array<uint16_t,4> clusterIds = {0, 256, 512, 768};

  const uint16_t tile = L4TileOffset;
  auto v = g.addVertex(cs, "RayRouter");

  auto sliceIn  = [&](size_t idx){ return L4RouterIn .slice(idx*kRayIOBytesPerTile,  (idx+1)*kRayIOBytesPerTile); };
  auto sliceOut = [&](size_t idx){ return L4RouterOut.slice(idx*kRayIOBytesPerTile,  (idx+1)*kRayIOBytesPerTile); };

  // parent (slot 0)
  auto parentIn  = sliceIn(0);
  auto parentOut = sliceOut(0);
  g.setTileMapping(parentIn,  tile);
  g.setTileMapping(parentOut, tile);
  g.connect(v["parentRaysIn" ], parentIn);
  g.connect(v["parentRaysOut"], parentOut);

  // children (slots 1..4)
  for (int i = 0; i < 4; ++i) {
    auto in  = sliceIn(i+1);
    auto out = sliceOut(i+1);
    g.setTileMapping(in,  tile);
    g.setTileMapping(out, tile);
    g.connect(v[fmt::format("childRaysIn{}",  i)], in);
    g.connect(v[fmt::format("childRaysOut{}", i)], out);
  }

  // child cluster IDs
  poplar::Tensor idsConst = g.addConstant(poplar::UNSIGNED_SHORT, {4}, clusterIds.data());
  g.setTileMapping(idsConst, tile);
  g.connect(v["childClusterIds"], idsConst);

  // level = 4
  auto levelConst = g.addConstant(poplar::UNSIGNED_CHAR, {}, 4);
  g.setTileMapping(levelConst, tile);
  g.connect(v["level"], levelConst);

  const auto kNumWorkers = 6;
  auto sharedCounts  = g.addVariable(poplar::UNSIGNED_INT, {kNumWorkers*5},                 "l4_sharedCounts");
  auto sharedOffsets = g.addVariable(poplar::UNSIGNED_INT, {kNumWorkers*5 + 2*kNumWorkers}, "l4_sharedOffsets");
  auto readyFlags    = g.addVariable(poplar::UNSIGNED_INT, {kNumWorkers},                   "l4_readyFlags");
  g.setTileMapping(sharedCounts, tile);
  g.setTileMapping(sharedOffsets, tile);
  g.setTileMapping(readyFlags, tile);
  g.connect(v["sharedCounts"],  sharedCounts);
  g.connect(v["sharedOffsets"], sharedOffsets);
  g.connect(v["readyFlags"],    readyFlags);

  // Debug slice
  auto dbgSlice = l4routerDebugRead_.get()
      .slice({0, 0}, {1, kRouterDebugSize})
      .reshape({kRouterDebugSize});
  g.connect(v["debugBytes"], dbgSlice);

  g.setTileMapping(v, tile);
}

void RadiantFoamIpuBuilder::buildDataExchange(poplar::Graph& g) {
  constexpr size_t   kChildrenPerRouter  = 4;
  constexpr size_t   kRayIOBytesPerTile  = kNumRays * sizeof(Ray);
  const     size_t   kRouterPerTileBuffer = kRayIOBytesPerTile * 5; // [parent | child0..3]

  data_exchange_seq = poplar::program::Sequence(DebugContext{"DataExchangeSeq"});

  auto childBlock = [&](poplar::Tensor &buf, uint16_t rid) -> poplar::Tensor {
    const size_t base = static_cast<size_t>(rid) * kRouterPerTileBuffer;
    return buf.slice(base + kRayIOBytesPerTile, base + 5 * kRayIOBytesPerTile); // slots 1..4
  };
  auto parentSlot = [&](poplar::Tensor &buf, uint16_t rid) -> poplar::Tensor {
    const size_t base = static_cast<size_t>(rid) * kRouterPerTileBuffer;
    return buf.slice(base, base + kRayIOBytesPerTile); // slot 0
  };

  // ────────────────────────────────────────────────────────────────────────────
  // L0 children → RT input   (DOWN)   and   RT output → L0 children (UP)
  // ────────────────────────────────────────────────────────────────────────────
  {
    std::vector<poplar::Tensor> srcAll, dstAll;
    srcAll.reserve(kNumL0RouterTiles);
    dstAll.reserve(kNumL0RouterTiles);

    for (uint16_t l0 = 0; l0 < kNumL0RouterTiles; ++l0) {
      const size_t baseRT = static_cast<size_t>(l0) * kChildrenPerRouter * kRayIOBytesPerTile;
      srcAll.push_back(childBlock(L0RouterOut, l0)); // 4×bytes
      dstAll.push_back(rayTracerInputRays_.slice(baseRT, baseRT + kChildrenPerRouter * kRayIOBytesPerTile));
    }
    data_exchange_seq.add(poplar::program::Copy(poplar::concat(srcAll), poplar::concat(dstAll),
                                  /*dontOutline*/true, DebugContext{"DX/L0->RT/down"}));

    srcAll.clear(); dstAll.clear();
    for (uint16_t l0 = 0; l0 < kNumL0RouterTiles; ++l0) {
      const size_t baseRT = static_cast<size_t>(l0) * kChildrenPerRouter * kRayIOBytesPerTile;
      srcAll.push_back(rayTracerOutputRays_.slice(baseRT, baseRT + kChildrenPerRouter * kRayIOBytesPerTile));
      dstAll.push_back(childBlock(L0RouterIn, l0)); // 4×bytes
    }
    data_exchange_seq.add(poplar::program::Copy(poplar::concat(srcAll), poplar::concat(dstAll),
                                  /*dontOutline*/true, DebugContext{"DX/RT->L0/up"}));
  }

  // ────────────────────────────────────────────────────────────────────────────
  // L0 parents → L1 children (DOWN)   and   L1 children → L0 parents (UP)
  // ────────────────────────────────────────────────────────────────────────────
  {
    std::vector<poplar::Tensor> srcAll, dstAll; // DOWN
    srcAll.reserve(kNumL1RouterTiles);
    dstAll.reserve(kNumL1RouterTiles);

    for (uint16_t l1 = 0; l1 < kNumL1RouterTiles; ++l1) {
      const size_t l1_base = l1 * kRouterPerTileBuffer;

      // L1 child block (slots 1..4) is contiguous
      poplar::Tensor l1ChildInBlock = L1RouterIn.slice(l1_base + kRayIOBytesPerTile, l1_base + 5 * kRayIOBytesPerTile);

      // Gather 4 L0 parents for this L1
      std::array<poplar::Tensor,4> l0Parents{};
      for (int c = 0; c < 4; ++c) {
        const uint16_t l0 = l1 * 4 + c;
        l0Parents[c] = parentSlot(L0RouterOut, l0);
      }
      srcAll.push_back(poplar::concat(l0Parents));
      dstAll.push_back(l1ChildInBlock);
    }
    data_exchange_seq.add(poplar::program::Copy(poplar::concat(srcAll), poplar::concat(dstAll),
                                  true, DebugContext{"DX/L0->L1/down"}));

    // UP
    srcAll.clear(); dstAll.clear();
    for (uint16_t l1 = 0; l1 < kNumL1RouterTiles; ++l1) {
      const size_t l1_base = l1 * kRouterPerTileBuffer;
      poplar::Tensor l1ChildOutBlock = L1RouterOut.slice(l1_base + kRayIOBytesPerTile, l1_base + 5 * kRayIOBytesPerTile);

      std::array<poplar::Tensor,4> l0ParentsIn{};
      for (int c = 0; c < 4; ++c) {
        const uint16_t l0 = l1 * 4 + c;
        l0ParentsIn[c] = parentSlot(L0RouterIn, l0);
      }
      srcAll.push_back(l1ChildOutBlock);
      dstAll.push_back(poplar::concat(l0ParentsIn));
    }
    data_exchange_seq.add(poplar::program::Copy(poplar::concat(srcAll), poplar::concat(dstAll),
                                  true, DebugContext{"DX/L1->L0/up"}));
  }

  // ────────────────────────────────────────────────────────────────────────────
  // L1 parents → L2 children (DOWN)   and   L2 children → L1 parents (UP)
  // ────────────────────────────────────────────────────────────────────────────
  {
    std::vector<poplar::Tensor> srcAll, dstAll; // DOWN
    srcAll.reserve(kNumL2RouterTiles);
    dstAll.reserve(kNumL2RouterTiles);

    for (uint16_t l2 = 0; l2 < kNumL2RouterTiles; ++l2) {
      const size_t l2_base = l2 * kRouterPerTileBuffer;
      poplar::Tensor l2ChildInBlock = L2RouterIn.slice(l2_base + kRayIOBytesPerTile, l2_base + 5 * kRayIOBytesPerTile);

      std::array<poplar::Tensor,4> l1Parents{};
      for (int c = 0; c < 4; ++c) {
        const uint16_t l1 = l2 * 4 + c;
        l1Parents[c] = parentSlot(L1RouterOut, l1);
      }
      srcAll.push_back(poplar::concat(l1Parents));
      dstAll.push_back(l2ChildInBlock);
    }
    data_exchange_seq.add(poplar::program::Copy(poplar::concat(srcAll), poplar::concat(dstAll),
                                  true, DebugContext{"DX/L1->L2/down"}));

    // UP
    srcAll.clear(); dstAll.clear();
    for (uint16_t l2 = 0; l2 < kNumL2RouterTiles; ++l2) {
      const size_t l2_base = l2 * kRouterPerTileBuffer;
      poplar::Tensor l2ChildOutBlock = L2RouterOut.slice(l2_base + kRayIOBytesPerTile, l2_base + 5 * kRayIOBytesPerTile);

      std::array<poplar::Tensor,4> l1ParentsIn{};
      for (int c = 0; c < 4; ++c) {
        const uint16_t l1 = l2 * 4 + c;
        l1ParentsIn[c] = parentSlot(L1RouterIn, l1);
      }
      srcAll.push_back(l2ChildOutBlock);
      dstAll.push_back(poplar::concat(l1ParentsIn));
    }
    data_exchange_seq.add(poplar::program::Copy(poplar::concat(srcAll), poplar::concat(dstAll),
                                  true, DebugContext{"DX/L2->L1/up"}));
  }

  // ────────────────────────────────────────────────────────────────────────────
  // L2 parents → L3 children (DOWN)   and   L3 children → L2 parents (UP)
  // ────────────────────────────────────────────────────────────────────────────
  {
    std::vector<poplar::Tensor> srcAll, dstAll; // DOWN
    srcAll.reserve(kNumL3RouterTiles);
    dstAll.reserve(kNumL3RouterTiles);

    for (uint16_t l3 = 0; l3 < kNumL3RouterTiles; ++l3) {
      const size_t l3_base = l3 * kRouterPerTileBuffer;
      poplar::Tensor l3ChildInBlock = L3RouterIn.slice(l3_base + kRayIOBytesPerTile, l3_base + 5 * kRayIOBytesPerTile);

      std::array<poplar::Tensor,4> l2Parents{};
      for (int c = 0; c < 4; ++c) {
        const uint16_t l2 = l3 * 4 + c;
        l2Parents[c] = parentSlot(L2RouterOut, l2);
      }
      srcAll.push_back(poplar::concat(l2Parents));
      dstAll.push_back(l3ChildInBlock);
    }
    data_exchange_seq.add(poplar::program::Copy(poplar::concat(srcAll), poplar::concat(dstAll),
                                  true, DebugContext{"DX/L2->L3/down"}));

    // UP
    srcAll.clear(); dstAll.clear();
    for (uint16_t l3 = 0; l3 < kNumL3RouterTiles; ++l3) {
      const size_t l3_base = l3 * kRouterPerTileBuffer;
      poplar::Tensor l3ChildOutBlock = L3RouterOut.slice(l3_base + kRayIOBytesPerTile, l3_base + 5 * kRayIOBytesPerTile);

      std::array<poplar::Tensor,4> l2ParentsIn{};
      for (int c = 0; c < 4; ++c) {
        const uint16_t l2 = l3 * 4 + c;
        l2ParentsIn[c] = parentSlot(L2RouterIn, l2);
      }
      srcAll.push_back(l3ChildOutBlock);
      dstAll.push_back(poplar::concat(l2ParentsIn));
    }
    data_exchange_seq.add(poplar::program::Copy(poplar::concat(srcAll), poplar::concat(dstAll),
                                  true, DebugContext{"DX/L3->L2/up"}));
  }

  // ────────────────────────────────────────────────────────────────────────────
  // L3 parents → L4 children (DOWN)   and   L4 children → L3 parents (UP)
  // ────────────────────────────────────────────────────────────────────────────
  {
    std::vector<poplar::Tensor> srcAll, dstAll; // DOWN
    srcAll.reserve(kNumL4RouterTiles);
    dstAll.reserve(kNumL4RouterTiles);
    // DOWN: L4 childOut block -> each L3 parentIn
    for (uint16_t l3 = 0; l3 < kNumL3RouterTiles; ++l3) {
        const size_t kRayIOBytesPerTile = kNumRays*sizeof(Ray);
        const size_t baseL4 = 0; // single L4 router, base index 0
        // L4 child block slots 1..4 laid contiguously
        poplar::Tensor l4ChildOutBlock = L4RouterOut.slice(baseL4 + kRayIOBytesPerTile, baseL4 + 5*kRayIOBytesPerTile);
        // pick child c==l3
        srcAll.push_back(l4ChildOutBlock.slice(l3*kRayIOBytesPerTile, (l3+1)*kRayIOBytesPerTile));
        dstAll.push_back( parentSlot(L3RouterIn, l3) );
    }
    data_exchange_seq.add(poplar::program::Copy(poplar::concat(srcAll), poplar::concat(dstAll),
                                true, DebugContext{"DX/L4->L3/down"}));

    // UP: L3 parentOut -> L4 childIn block
    srcAll.clear(); dstAll.clear();
    for (uint16_t l3 = 0; l3 < kNumL3RouterTiles; ++l3) {
        const size_t kRayIOBytesPerTile = kNumRays*sizeof(Ray);
        const size_t baseL4 = 0;
        poplar::Tensor l4ChildInBlock = L4RouterIn.slice(baseL4 + kRayIOBytesPerTile, baseL4 + 5*kRayIOBytesPerTile);
        srcAll.push_back( parentSlot(L3RouterOut, l3) );
        dstAll.push_back( l4ChildInBlock.slice(l3*kRayIOBytesPerTile, (l3+1)*kRayIOBytesPerTile) );
    }
    data_exchange_seq.add(poplar::program::Copy(poplar::concat(srcAll), poplar::concat(dstAll),
                                true, DebugContext{"DX/L3->L4/up"}));
    }

    // ────────────────────────────────────────────────────────────────────────────
    // L4 parent ↔ RayGen child   (DOWN: RG→L4 parent,  UP: L4 parent→RG)
    // ────────────────────────────────────────────────────────────────────────────
    {
        // DOWN: RayGen RaysOut -> L4 parentIn
        data_exchange_seq.add(poplar::program::Copy(
            raygenOutput,
            L4RouterIn.slice(0, kNumRays*sizeof(Ray)),
            true, DebugContext{"DX/RG->L4/down"}));

        // UP: L4 parentOut -> RayGen RaysIn
        data_exchange_seq.add(poplar::program::Copy(
            L4RouterOut.slice(0, kNumRays*sizeof(Ray)),
            raygenInput,
            true, DebugContext{"DX/L4->RG/up"}));
    }
  // Register the program (use this if you still call it by name from execute())
    //   getPrograms().add("DataExchange", seq);

  frameStep_.add(data_exchange_seq);
}

void RadiantFoamIpuBuilder::setupHostStreams(poplar::Graph& g) {
    // Debug reads
    framebuffer_host.resize(kTileFramebufferSize * kNumRayTracerTiles);
    getPrograms().add("fb_read_all", fb_read_all_.buildRead(g,true));
    result_f32_host.resize(kNumRayTracerTiles);
    getPrograms().add("read_result_f32", result_f32_read_.buildRead(g,true));    
    result_u16_host.resize(kNumRayTracerTiles);
    getPrograms().add("read_result_u16", result_u16_read_.buildRead(g,true));
    l0routerDebugBytesHost_.resize(kNumL0RouterTiles * kRouterDebugSize);
    getPrograms().add("read_l0_router_debug_bytes", l0routerDebugRead_.buildRead(g,true));
    l1routerDebugBytesHost_.resize(kNumL1RouterTiles * kRouterDebugSize);
    getPrograms().add("read_l1_router_debug_bytes", l1routerDebugRead_.buildRead(g,true));
    l2routerDebugBytesHost_.resize(kNumL2RouterTiles * kRouterDebugSize);
    getPrograms().add("read_l2_router_debug_bytes", l2routerDebugRead_.buildRead(g,true));
    l3routerDebugBytesHost_.resize(kNumL3RouterTiles * kRouterDebugSize);
    getPrograms().add("read_l3_router_debug_bytes", l3routerDebugRead_.buildRead(g,true));
    l4routerDebugBytesHost_.resize(kNumL4RouterTiles * kRouterDebugSize);
    getPrograms().add("read_l4_router_debug_bytes", l4routerDebugRead_.buildRead(g,true));
    raygenDebugBytesHost_.resize(kRouterDebugSize);
    getPrograms().add("read_raygen_router_debug_bytes", raygenDebugRead_.buildRead(g,true));

    finishedRaysHost_.resize(kNumRayTracerTiles * kFinishedRayBytesPerTile);

    hostViewMatrix_.assign(16, 0.0f);
    hostProjMatrix_.assign(16, 0.0f);
}

void RadiantFoamIpuBuilder::connectHostStreams(poplar::Engine& eng) {
    for (size_t t = 0; t < local_tensors_.size(); ++t) {
        local_tensors_[t].connectWriteStream(eng, local_pts_[t]);
        neighbor_tensors_[t].connectWriteStream(eng, neighbor_pts_[t]);
        adj_tensors_[t].connectWriteStream(eng, adjacency_[t]);
    }

    fb_read_all_.connectReadStream(eng, framebuffer_host.data());
    result_f32_read_.connectReadStream(eng, result_f32_host.data());
    result_u16_read_.connectReadStream(eng, result_u16_host.data());
    // finishedRaysRead_.connectReadStream(eng, finishedRaysHost_.data());
    eng.connectStream("read-finished-rays-stream", finishedRaysHost_.data(), finishedRaysHost_.data() + finishedRaysHost_.size());

    l0routerDebugRead_.connectReadStream(eng, l0routerDebugBytesHost_.data());
    l1routerDebugRead_.connectReadStream(eng, l1routerDebugBytesHost_.data());
    l2routerDebugRead_.connectReadStream(eng, l2routerDebugBytesHost_.data());
    l3routerDebugRead_.connectReadStream(eng, l3routerDebugBytesHost_.data());
    l4routerDebugRead_.connectReadStream(eng, l4routerDebugBytesHost_.data());
    raygenDebugRead_.connectReadStream(eng, raygenDebugBytesHost_.data());
    viewMatrix_.connectWriteStream(eng, hostViewMatrix_.data());
    projMatrix_.connectWriteStream(eng, hostProjMatrix_.data());
    cameraCellInfo_.connectWriteStream(eng, hostCameraCellInfo_.data());

    stopFlag_.connectWriteStream(eng, &stopFlagHost_);
}

// ----------------------------------------------------------------------------
//  readAllTiles() – simple helper to print data for debugging
// ----------------------------------------------------------------------------
void RadiantFoamIpuBuilder::readAllTiles(poplar::Engine& eng) {
    eng.run(getPrograms().getOrdinals().at("fb_read_all"));
    eng.run(getPrograms().getOrdinals().at("read_result_f32"));
    eng.run(getPrograms().getOrdinals().at("read_result_u16"));
    eng.run(getPrograms().getOrdinals().at("read_l0_router_debug_bytes"));
    eng.run(getPrograms().getOrdinals().at("read_l1_router_debug_bytes"));
    eng.run(getPrograms().getOrdinals().at("read_l2_router_debug_bytes"));
    eng.run(getPrograms().getOrdinals().at("read_l3_router_debug_bytes"));
    eng.run(getPrograms().getOrdinals().at("read_l4_router_debug_bytes"));
    eng.run(getPrograms().getOrdinals().at("read_raygen_router_debug_bytes"));

    RF_LOG("================ Frame {} =================================================", exec_counter_);
    int overall_cntr = 0;
    constexpr uint16_t kLeadMask   = 0xFC00u;   // 11111 00000000000₂
    constexpr unsigned kShift = 10;

    // Cell tracking
    // for (int tid = 0; tid < 1024; ++tid) {
    //     const size_t offset = static_cast<size_t>(tid) * kTileFramebufferSize;
    //     uint8_t cnt = framebuffer_host[offset];

    //     if (cnt > 0) {
    //         // Print summary for this tile
    //         RF_LOG("Tile {} result f32: {:.9f}, u16: {}", tid,
    //             result_f32_host[tid], result_u16_host[tid]);

    //         // Iterate through points written in framebuffer
    //         for (uint8_t i = 1; i <= cnt; ++i) {
    //             uint16_t x = (framebuffer_host[offset + i * 6] << 8) |
    //                             framebuffer_host[offset + i * 6 + 1];
    //             uint16_t y = (framebuffer_host[offset + i * 6 + 2] << 8) |
    //                             framebuffer_host[offset + i * 6 + 3];
    //             auto x_data = x & ~kLeadMask;
    //             auto x_cluster_cnt = (x & kLeadMask) >> kShift;
    //             uint16_t pt_idx = (framebuffer_host[offset + i * 6 + 4] << 8) |
    //                             framebuffer_host[offset + i * 6 + 5];

    //             if (pt_idx < local_pts_[tid].size()) {
    //                 const auto &pt = local_pts_[tid][pt_idx];
    //                 fmt::print("[{}] ({}, {}) {}: {:4} → ({:8.6f}, {:8.6f}, {:8.6f})\n",
    //                         i, x_data, y, x_cluster_cnt, pt_idx, pt.x, pt.y, pt.z);
    //             } else {
    //                 fmt::print("[{}]: {:4} → (INVALID INDEX)\n",
    //                         i, pt_idx);
    //             }

    //             ++overall_cntr;
    //         }
    //     }
    // }

    // -----------------------------------------------------------------------------
    //  Router-lane saturation test
    // -----------------------------------------------------------------------------
    constexpr std::uint16_t kWarnCap        = 5000;  // threshold
    constexpr int           kWordsPerRouter = 10;    // 0..4 = IN  / 5..9 = OUT

    auto dumpRouters =
        [&](const char                       *lvl,
            const std::vector<std::uint8_t>  &dbg,
            std::size_t                       numRouters)
    {
        for (std::size_t rid = 0; rid < numRouters; ++rid)
        {
            const std::uint8_t *base = &dbg[rid * kRouterDebugSize];

            std::array<std::uint16_t, kWordsPerRouter> w{};
            for (std::size_t i = 0; i < kWordsPerRouter; ++i)
            {
                /* build little-endian 16-bit word   byte0 = LSB, byte1 = MSB */
                w[i] = static_cast<std::uint16_t>(base[2*i])         |
                    (static_cast<std::uint16_t>(base[2*i + 1]) << 8);
            }

            bool over = std::any_of(w.begin(), w.end(),
                                    [](std::uint16_t v){ return v > kWarnCap; });

            if (over || lvl == "RG")
            {
              if(lvl == "RG" && w[9] != 0)
                fmt::print("{} router {:4}: "
                    "In: {:4}  {:4}  {:4}  {:4} {:4}\t"
                    "Out: {:4}  {:4}  {:4}  {:4} {:4}====================================================\n",
                    lvl, rid,
                    w[0], w[1], w[2], w[3], w[4],
                    w[5], w[6], w[7], w[8], w[9]);
              else 
                fmt::print("{} router {:4}: "
                    "In: {:4}  {:4}  {:4}  {:4} {:4}\t"
                    "Out: {:4}  {:4}  {:4}  {:4} {:4}\n",
                    lvl, rid,
                    w[0], w[1], w[2], w[3], w[4],
                    w[5], w[6], w[7], w[8], w[9]);
            }
        }
    };
    // call once per level ----------------------------------------------------------
    dumpRouters("L0", l0routerDebugBytesHost_, kNumL0RouterTiles);
    dumpRouters("L1", l1routerDebugBytesHost_, kNumL1RouterTiles);
    dumpRouters("L2", l2routerDebugBytesHost_, kNumL2RouterTiles);
    dumpRouters("L3", l3routerDebugBytesHost_, kNumL3RouterTiles);
    dumpRouters("L4", l4routerDebugBytesHost_, kNumL4RouterTiles);
    dumpRouters("RG",  raygenDebugBytesHost_,  1);   

}

#undef RF_LOG
