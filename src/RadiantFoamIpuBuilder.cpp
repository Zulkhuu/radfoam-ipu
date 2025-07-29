#include "RadiantFoamIpuBuilder.hpp"

#include <spdlog/fmt/fmt.h>

// Utility macro for logging inside the class implementation
#define RF_LOG(...) ipu_utils::logger()->info(__VA_ARGS__)

using namespace radfoam::ipu;
using namespace radfoam::geometry;
using ipu_utils::logger;

// ----------------------------------------------------------------------------
//  ctor
// ----------------------------------------------------------------------------
RadiantFoamIpuBuilder::RadiantFoamIpuBuilder(std::string h5_scene_file, int debug_tile)
    : h5_file_(std::move(h5_scene_file)),
      tile_to_debug_(debug_tile) {
        initialised_ = false;
        hostViewMatrix_.assign(16, 0.0f);
        hostProjMatrix_.assign(16, 0.0f);
      }

// ----------------------------------------------------------------------------
//  Public update helpers (called once per frame from host)
// ----------------------------------------------------------------------------
void RadiantFoamIpuBuilder::updateViewMatrix(const glm::mat4& m) {
    glm::mat4 transposed = glm::transpose(m); // Poplar expects row‑major
    const float* ptr = glm::value_ptr(transposed);
    for (size_t i = 0; i < 16; ++i) {
        hostViewMatrix_[i] = ptr[i];
    }
}

void RadiantFoamIpuBuilder::updateProjectionMatrix(const glm::mat4& m) {
    glm::mat4 transposed = glm::transpose(m);
    const float* ptr = glm::value_ptr(transposed);
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
void RadiantFoamIpuBuilder::build(poplar::Graph& graph, const poplar::Target&) {
    RF_LOG("Building RadiantFoam graph (debug tile {})", tile_to_debug_);

    loadScenePartitions();
    registerCodeletsAndOps(graph);
    allocateGlobalTensors(graph);

    poplar::ComputeSet cs = graph.addComputeSet("RayTraceCS");

    createRayTraceVertices(graph, cs);
    createRayGenVertex(graph, cs);
    createRayRoutersLevel0(graph, cs);

    createDataExchangePrograms(graph);
    setupHostStreams(graph);

    std::string filepath = "../debugchains.txt";
	std::ifstream file(filepath);
	if (!file.is_open()) {
        throw std::runtime_error("Failed to open debug chains file: " + filepath);
	}

	unsigned value;
	while (file >> value) {
        debug_chains.push_back(value);
    }
}

// ----------------------------------------------------------------------------
//  execute() – per‑frame invocation
// ----------------------------------------------------------------------------
void RadiantFoamIpuBuilder::execute(poplar::Engine& eng, const poplar::Device&) {
    if (!initialised_) {
        // Connect once‑off host streams and run zero/write programs
        for (size_t t = 0; t < local_tensors_.size(); ++t) {
            local_tensors_[t].connectWriteStream(eng, local_pts_[t]);
            neighbor_tensors_[t].connectWriteStream(eng, neighbor_pts_[t]);
            adj_tensors_[t].connectWriteStream(eng, adjacency_[t]);
        }
        fb_read_all_.connectReadStream(eng, framebuffer_host.data());
        result_f32_read_.connectReadStream(eng, result_f32_host.data());
        result_u16_read_.connectReadStream(eng, result_u16_host.data());
        l0routerDebugBytesHost_.resize(kNumL0RouterTiles * kRouterDebugSize);
        l0routerDebugBytesRead_.connectReadStream(eng, l0routerDebugBytesHost_.data());
        viewMatrix_.connectWriteStream(eng, hostViewMatrix_.data());
        projMatrix_.connectWriteStream(eng, hostProjMatrix_.data());
        execCountT_.connectWriteStream(eng, &exec_counter_);
        cameraCellInfo_.connectWriteStream(eng, hostCameraCellInfo_.data());

        eng.run(getPrograms().getOrdinals().at("zero_rays"));
        eng.run(getPrograms().getOrdinals().at("write"));
        initialised_ = true;
    }

    // Per‑frame updates ------------------------------------------------------
    eng.run(getPrograms().getOrdinals().at("broadcast_matrices"));
    eng.run(getPrograms().getOrdinals().at("write_exec_count"));
    eng.run(getPrograms().getOrdinals().at("write_camera_cell_info"));
    eng.run(getPrograms().getOrdinals().at("RayTraceCS"));
    eng.run(getPrograms().getOrdinals().at("DataExchange"));
    eng.run(getPrograms().getOrdinals().at("read_router_debug_bytes"));
    readAllTiles(eng);
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
                  "-O3 -I " + incPath + " -I " + glmPath);
    popops::addCodelets(g);
}

void RadiantFoamIpuBuilder::allocateGlobalTensors(poplar::Graph& g) {
    const size_t kRayIOBytesPerTile   = kNumRays * sizeof(Ray);
    const size_t kRayTracerBufferSize = kRayIOBytesPerTile * kNumRayTracerTiles;

    rayTracerOutputRays_ = g.addVariable(poplar::UNSIGNED_CHAR, {kRayTracerBufferSize}, "rt_out");
    rayTracerInputRays_  = g.addVariable(poplar::UNSIGNED_CHAR, {kRayTracerBufferSize}, "rt_in");

    result_f32_read_.buildTensor(g, poplar::FLOAT,          {kNumRayTracerTiles});
    result_u16_read_.buildTensor(g, poplar::UNSIGNED_SHORT, {kNumRayTracerTiles});
    fb_read_all_.buildTensor(g,  poplar::UNSIGNED_CHAR,
                             {kNumRayTracerTiles, kTileFramebufferSize});

    // Map linearly for fast access
    poputil::mapTensorLinearlyWithOffset(g, result_f32_read_.get(), 0);
    poputil::mapTensorLinearlyWithOffset(g, result_u16_read_.get(), 0);
    poputil::mapTensorLinearlyWithOffset(g,
        fb_read_all_.get().reshape({kNumRayTracerTiles, kTileFramebufferSize}), 0);

    // View / projection matrices live on tile 0 and are broadcast each frame
    viewMatrix_.buildTensor(g, poplar::FLOAT, {4,4});
    projMatrix_.buildTensor(g, poplar::FLOAT, {4,4});
    g.setTileMapping(viewMatrix_.get(), 0);
    g.setTileMapping(projMatrix_.get(), 0);
    broadcastMatrices_.add(viewMatrix_.buildWrite(g, true));
    broadcastMatrices_.add(projMatrix_.buildWrite(g, true));
}

void RadiantFoamIpuBuilder::createRayTraceVertices(poplar::Graph& g, poplar::ComputeSet& cs) {
    const size_t kRayIOBytesPerTile = kNumRays * sizeof(Ray);
    const auto zero_const = g.addConstant(poplar::UNSIGNED_CHAR, {kRayIOBytesPerTile}, 0);
    g.setTileMapping(zero_const, kRaygenTile);
    poplar::program::Sequence zero_seq;

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
        auto localProj = g.clone(projMatrix_.get(), "proj_mat_t"+std::to_string(tid));
        g.setTileMapping(localView, tid);
        g.setTileMapping(localProj, tid);
        broadcastMatrices_.add(poplar::program::Copy(viewMatrix_.get(), localView));
        broadcastMatrices_.add(poplar::program::Copy(projMatrix_.get(), localProj));
        g.connect(v["view_matrix"],      localView.flatten());
        g.connect(v["projection_matrix"],localProj.flatten());

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
        zero_seq.add(poplar::program::Copy(zero_const, out_slice));
        zero_seq.add(poplar::program::Copy(zero_const, in_slice));
        g.connect(v["raysOut"], out_slice);
        g.connect(v["raysIn" ], in_slice);

        // Per‑tile result scalars & framebuffer slice -----------------------
        g.connect(v["result_float"], result_f32_read_.get().slice({tid},{tid+1}).reshape({}));
        g.connect(v["result_u16"  ], result_u16_read_.get().slice({tid},{tid+1}).reshape({}));
        auto fb_slice = fb_read_all_.get()
                            .slice({tid,0},{tid+1,kTileFramebufferSize})
                            .reshape({kTileFramebufferSize});
        g.connect(v["framebuffer"], fb_slice);

        // Constant tile_id param
        auto tile_const = g.addConstant(poplar::UNSIGNED_SHORT, {}, static_cast<unsigned>(tid));
        g.setTileMapping(tile_const, tid);
        g.connect(v["tile_id"], tile_const);
        g.setTileMapping(v, tid);

        per_tile_writes_.add(in_local.buildWrite(g, true));
        per_tile_writes_.add(in_nbr  .buildWrite(g, true));
        per_tile_writes_.add(in_adj  .buildWrite(g, true));

        local_tensors_.push_back(std::move(in_local));
        neighbor_tensors_.push_back(std::move(in_nbr));
        adj_tensors_.push_back(std::move(in_adj));
    }

    getPrograms().add("RayTraceCS", poplar::program::Execute(cs));
    getPrograms().add("broadcast_matrices", broadcastMatrices_);
    getPrograms().add("zero_rays", zero_seq);
    getPrograms().add("write", per_tile_writes_);
}

void RadiantFoamIpuBuilder::createRayGenVertex(poplar::Graph& g, poplar::ComputeSet& cs) {
    const size_t kRayIOBytesPerTile = kNumRays * sizeof(Ray);
    raygenInput  = g.addVariable(poplar::UNSIGNED_CHAR,{kRayIOBytesPerTile}, "raygen_in");
    raygenOutput = g.addVariable(poplar::UNSIGNED_CHAR,{kRayIOBytesPerTile}, "raygen_out");
    g.setTileMapping(raygenInput,  kRaygenTile);
    g.setTileMapping(raygenOutput, kRaygenTile);

    auto v = g.addVertex(cs, "RayGen");
    g.connect(v["raysIn" ], raygenInput);
    g.connect(v["raysOut"], raygenOutput);

    execCountT_.buildTensor(g, poplar::UNSIGNED_INT, {});
    g.setTileMapping(execCountT_.get(),   kRaygenTile);
    getPrograms().add("write_exec_count", execCountT_.buildWrite(g,true));
    
    cameraCellInfo_.buildTensor(g, poplar::UNSIGNED_CHAR,{4});
    g.setTileMapping(cameraCellInfo_.get(),kRaygenTile);
    getPrograms().add("write_camera_cell_info", cameraCellInfo_.buildWrite(g,true));
    
    g.connect(v["exec_count"], execCountT_.get());
    g.connect(v["camera_cell_info"], cameraCellInfo_.get());
    g.setTileMapping(v, kRaygenTile);
}

void RadiantFoamIpuBuilder::createRayRoutersLevel0(poplar::Graph& g, poplar::ComputeSet& cs) {
    constexpr uint16_t router_tile_offset = 1024;
    constexpr size_t   kChildrenPerRouter = 4;
    constexpr size_t   kRayIOBytesPerTile = kNumRays * sizeof(Ray);

    const size_t kRouterPerTileBuffer = kRayIOBytesPerTile * 5; // parent + 4 children
    const size_t kTotalBuffer         = kRouterPerTileBuffer * kNumL0RouterTiles;

    L0RouterOut = g.addVariable(poplar::UNSIGNED_CHAR,{kTotalBuffer},"router_out");
    L0RouterIn  = g.addVariable(poplar::UNSIGNED_CHAR,{kTotalBuffer},"router_in");

    l0routerDebugBytesRead_.buildTensor(g, poplar::UNSIGNED_CHAR,{kNumL0RouterTiles,kRouterDebugSize});
    poputil::mapTensorLinearlyWithOffset(g,
        l0routerDebugBytesRead_.get().reshape({kNumL0RouterTiles,kRouterDebugSize}),
        router_tile_offset);

    allClusterIds_.reserve(kNumL0RouterTiles * kChildrenPerRouter);
    for (size_t r = 0; r < kNumL0RouterTiles; ++r) {
        for (size_t i = 0; i < kChildrenPerRouter; ++i) {
            allClusterIds_.push_back(static_cast<uint16_t>(r * kChildrenPerRouter + i));
        }
    }

    for (size_t router_id = 0; router_id < kNumL0RouterTiles; ++router_id) {
        const uint16_t tile = router_tile_offset + router_id;
        auto v = g.addVertex(cs, "RayRouter");

        // Slice IO buffers ---------------------------------------------------
        const size_t base = router_id * kRouterPerTileBuffer;
        auto sliceIn  = [&](size_t idx){return L0RouterIn.slice (base+idx*kRayIOBytesPerTile,
                                                                 base+(idx+1)*kRayIOBytesPerTile);} ;
        auto sliceOut = [&](size_t idx){return L0RouterOut.slice(base+idx*kRayIOBytesPerTile,
                                                                 base+(idx+1)*kRayIOBytesPerTile);} ;
        auto parentIn  = sliceIn(0);
        auto parentOut = sliceOut(0);
        g.setTileMapping(parentIn, tile); g.setTileMapping(parentOut, tile);

        for (int i = 0; i < 4; ++i) {
            auto in  = sliceIn(i+1);
            auto out = sliceOut(i+1);
            g.setTileMapping(in,  tile);
            g.setTileMapping(out, tile);
            g.connect(v[fmt::format("childRaysIn{}",  i)], in);
            g.connect(v[fmt::format("childRaysOut{}", i)], out);
        }

        g.connect(v["parentRaysIn" ], parentIn);
        g.connect(v["parentRaysOut"], parentOut);

        // Constant child cluster IDs (4)
        poplar::Tensor idsConst = g.addConstant(poplar::UNSIGNED_SHORT,{4},
                                                allClusterIds_.data() + router_id*4);
        g.setTileMapping(idsConst, tile);
        g.connect(v["childClusterIds"], idsConst);

        auto levelConst = g.addConstant(poplar::UNSIGNED_CHAR, {}, 0);
        g.setTileMapping(levelConst, tile);
        g.connect(v["level"], levelConst);

        auto dbgSlice = l0routerDebugBytesRead_.get()
                         .slice({router_id,0},{router_id+1,kRouterDebugSize})
                         .reshape({kRouterDebugSize});
        g.connect(v["debugBytes"], dbgSlice);
        g.setTileMapping(v, tile);
    }

    getPrograms().add("read_router_debug_bytes", l0routerDebugBytesRead_.buildRead(g,true));
}

void RadiantFoamIpuBuilder::createDataExchangePrograms(poplar::Graph& g) {
    constexpr uint16_t router_tile_offset = 1024;
    constexpr size_t   kChildrenPerRouter = 4;
    constexpr size_t   kRayIOBytesPerTile = kNumRays * sizeof(Ray);
    const size_t kRouterPerTileBuffer = kRayIOBytesPerTile * 5;

    poplar::program::Sequence seq;

    for (uint16_t routerId = 0; routerId < kNumL0RouterTiles; ++routerId) {
        const size_t base = routerId * kRouterPerTileBuffer;
        auto childOut = [&](int i){return L0RouterOut.slice(base+(i+1)*kRayIOBytesPerTile, base+(i+2)*kRayIOBytesPerTile);} ;
        auto childIn  = [&](int i){return L0RouterIn .slice(base+(i+1)*kRayIOBytesPerTile, base+(i+2)*kRayIOBytesPerTile);} ;

        // Router ‑> RayTracer
        for (int c = 0; c < 4; ++c) {
            uint16_t tracerTile = routerId * kChildrenPerRouter + c;
            auto rtIn = rayTracerInputRays_.slice(tracerTile * kRayIOBytesPerTile,
                                                  (tracerTile+1)*kRayIOBytesPerTile);
            seq.add(poplar::program::Copy(childOut(c), rtIn));
        }

        // RayTracer ‑> Router
        for (int c = 0; c < 4; ++c) {
            uint16_t tracerTile = routerId * kChildrenPerRouter + c;
            auto rtOut = rayTracerOutputRays_.slice(tracerTile * kRayIOBytesPerTile,
                                                    (tracerTile+1)*kRayIOBytesPerTile);
            seq.add(poplar::program::Copy(rtOut, childIn(c)));
        }

        // Special: RayGen output into router 130 parent input (proof‑of‑concept)
        if (routerId == 130) {
            auto parentIn  = L0RouterIn.slice(base, base + kRayIOBytesPerTile);
            seq.add(poplar::program::Copy(raygenOutput, parentIn));
        }
    }
    getPrograms().add("DataExchange", seq);
}

void RadiantFoamIpuBuilder::setupHostStreams(poplar::Graph& g) {
    framebuffer_host.resize(kTileFramebufferSize * kNumRayTracerTiles);
    result_f32_host.resize(kNumRayTracerTiles);
    result_u16_host.resize(kNumRayTracerTiles);

    getPrograms().add("fb_read_all", fb_read_all_.buildRead(g,true));
    getPrograms().add("read_result_f32", result_f32_read_.buildRead(g,true));
    getPrograms().add("read_result_u16", result_u16_read_.buildRead(g,true));
}

// ----------------------------------------------------------------------------
//  readAllTiles() – simple debugging helper to print framebuffer summary
// ----------------------------------------------------------------------------
void RadiantFoamIpuBuilder::readAllTiles(poplar::Engine& eng) {
    eng.run(getPrograms().getOrdinals().at("fb_read_all"));
    eng.run(getPrograms().getOrdinals().at("read_result_f32"));
    eng.run(getPrograms().getOrdinals().at("read_result_u16"));

    RF_LOG("================ Frame {} =================================================", exec_counter_);
    int overall_cntr = 0;
    for (int tid : debug_chains) {
        RF_LOG("Tile {} result f32: {:.6f}, u16: {}", tid, result_f32_host[tid], result_u16_host[tid]);
        const size_t offset = static_cast<size_t>(tid) * kTileFramebufferSize;
        uint8_t cnt = framebuffer_host[offset];
        for (uint8_t i = 1; i <= cnt; ++i) {
            uint16_t pt_idx = (framebuffer_host[offset+i*2] <<8) | framebuffer_host[offset+i*2+1];
            if (pt_idx < local_pts_[tid].size()) {
                const auto& pt = local_pts_[tid][pt_idx];
                fmt::print("[{}] {}: {:4} → ({:8.6f}, {:8.6f}, {:8.6f})\n",
                           overall_cntr, i, pt_idx, pt.x, pt.y, pt.z);
            }
            ++overall_cntr;
        }
    }
}

#undef RF_LOG
