// ============================================================================
// RadiantFoamIpuBuilder – full, refactored implementation
//   • Stand‑alone translation unit (add to your CMakeLists/src list)
//   • Public API identical to the original, so existing call‑sites keep working
//   • Internal logic split into well‑named helper functions with early‑exit
//     checks and clear ownership semantics.
// ============================================================================
//                         BSD‑3‑Clause License
// ============================================================================
// 2025‑07‑29 – Foam Refactor Team
// ============================================================================

// ─────────────────────────────────────────────────────────────────────────────
//  Header
// ─────────────────────────────────────────────────────────────────────────────
#ifndef RADIANT_FOAM_IPU_BUILDER_HPP
#define RADIANT_FOAM_IPU_BUILDER_HPP

// C++ std
#include <string>
#include <vector>
#include <array>
#include <memory>
#include <cstdint>
#include <utility>
#include <fstream>
#include <stdexcept>

// Third‑party
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <highfive/H5File.hpp>
#include <poplar/Graph.hpp>
#include <poplar/Engine.hpp>
#include <popops/codelets.hpp>
#include <poputil/TileMapping.hpp>

// Local
#include "ipu/tile_config.hpp"
#include "ipu/ipu_utils.hpp"
#include "geometry/primitives.hpp"
#include "io/hdf5_types.hpp"

namespace radfoam {
namespace ipu {

/// RadiantFoamIpuBuilder – constructs & drives the Poplar graph used by the
/// real‑time RadiantFoam renderer. One instance lives on the host and may be
/// reused across multiple frames; build() runs once, execute() runs every frame.
class RadiantFoamIpuBuilder final : public ipu_utils::BuilderInterface {
public:
    /// @param h5_scene_file    Path to the baked HDF5 scene file.
    /// @param debug_tile       Optional: tile ID to emit extra debug for.
    explicit RadiantFoamIpuBuilder(std::string h5_scene_file, int debug_tile = 0, bool debug = false);

    // BuilderInterface -------------------------------------------------------
    void build(poplar::Graph& graph, const poplar::Target& target) override;
    void execute(poplar::Engine& eng, const poplar::Device& dev) override;

    // Per‑frame host‑side updates -------------------------------------------
    void updateViewMatrix(const glm::mat4& view);
    void updateProjectionMatrix(const glm::mat4& proj);
    void updateCameraCell(const radfoam::geometry::GenericPoint& cell);

    // Host‑visible frame results -------------------------------------------
    std::vector<uint8_t>  finishedRaysHost_;
    std::vector<uint8_t>  framebuffer_host;   ///< RGB888 for every tile
    std::vector<float>    result_f32_host;    ///< Scratch debug channel
    std::vector<uint16_t> result_u16_host;    ///< Scratch debug channel
    std::vector<unsigned> debug_chains;       ///< Tiles to print in readAllTiles()

private:
    // ───────────── helper sections used by build() ─────────────
    void loadScenePartitions();
    void registerCodeletsAndOps(poplar::Graph& g);
    void allocateGlobalTensors(poplar::Graph& g);
    void createRayTraceVertices(poplar::Graph& g, poplar::ComputeSet& cs);
    void createRayGenVertex(poplar::Graph& g, poplar::ComputeSet& cs);
    void createRayRoutersLevel0(poplar::Graph& g, poplar::ComputeSet& cs);
    void createRayRoutersLevel1(poplar::Graph& g, poplar::ComputeSet& cs);
    void createRayRoutersLevel2(poplar::Graph& g, poplar::ComputeSet& cs);
    void createRayRoutersLevel3(poplar::Graph& g, poplar::ComputeSet& cs);
    void createDataExchangePrograms(poplar::Graph& g);
    void setupHostStreams(poplar::Graph& g);
    void readAllTiles(poplar::Engine& engine);

    // ───────────── members ─────────────
    // Construction‑time constants
    const std::string h5_file_;
    const int         tile_to_debug_;
    bool debug_;
    unsigned substeps_ = 10;

    // Scene data -------------------------------------------------------------
    std::vector<std::vector<radfoam::geometry::LocalPoint>>   local_pts_;
    std::vector<std::vector<radfoam::geometry::GenericPoint>> neighbor_pts_;
    std::vector<std::vector<uint16_t>>                        adjacency_;
    std::vector<std::string>                                  paths_;

    // Poplar tensors ---------------------------------------------------------
    poplar::Tensor rayTracerOutputRays_;
    poplar::Tensor rayTracerInputRays_;

    std::vector<ipu_utils::StreamableTensor> local_tensors_;
    std::vector<ipu_utils::StreamableTensor> neighbor_tensors_;
    std::vector<ipu_utils::StreamableTensor> adj_tensors_;

    ipu_utils::StreamableTensor execCountT_{"exec_count"};
    ipu_utils::StreamableTensor fb_read_all_{"fb_read_all"};
    ipu_utils::StreamableTensor result_f32_read_{"result_f32_read"};
    ipu_utils::StreamableTensor result_u16_read_{"result_u16_read"};
    ipu_utils::StreamableTensor viewMatrix_{"view_matrix"};
    ipu_utils::StreamableTensor projMatrix_{"proj_matrix"};
    ipu_utils::StreamableTensor cameraCellInfo_{"camera_cell_info"};
    ipu_utils::StreamableTensor l0routerDebugBytesRead_{"l0_router_debug_bytes_read"};
    ipu_utils::StreamableTensor l1routerDebugBytesRead_{"l1_router_debug_bytes_read"};
    ipu_utils::StreamableTensor l2routerDebugBytesRead_{"l2_router_debug_bytes_read"};
    ipu_utils::StreamableTensor l3routerDebugBytesRead_{"l3_router_debug_bytes_read"};
    ipu_utils::StreamableTensor raygenDebugBytesRead_{"raygen_debug_bytes_read"};
    ipu_utils::StreamableTensor finishedRaysRead_{"finished_rays_read"};
    ipu_utils::StreamableTensor rtExecCounts_{"exec_count_rt"};   
    ipu_utils::StreamableTensor finishedWriteOffsets_{"finished_rays_write_offsets"};

    // CPU‑side mirrors -------------------------------------------------------
    bool initialised_      = false;
    unsigned exec_counter_ = 0;
    std::vector<float> hostViewMatrix_;
    std::vector<float> hostProjMatrix_;
    std::array<uint8_t, 4> hostCameraCellInfo_{{0,0,0,0}};
    std::vector<uint8_t> l0routerDebugBytesHost_;
    std::vector<uint8_t> l1routerDebugBytesHost_;
    std::vector<uint8_t> l2routerDebugBytesHost_;
    std::vector<uint8_t> l3routerDebugBytesHost_;
    std::vector<uint8_t> raygenDebugBytesHost_;



    // Helper program sequences ----------------------------------------------
    poplar::program::Sequence per_tile_writes_;
    poplar::program::Sequence broadcastMatrices_;
    poplar::program::Sequence zero_seq;

    poplar::program::Sequence frameStep_;
    // poplar::program::Sequence frame_;

    // Router bookkeeping -----------------------------------------------------
    static constexpr uint16_t kRouterDebugSize = 24;
    // std::vector<uint16_t> allClusterIds_;

    poplar::Tensor L0RouterOut;
    poplar::Tensor L0RouterIn; 
    poplar::Tensor L1RouterOut;
    poplar::Tensor L1RouterIn; 
    poplar::Tensor L2RouterOut;
    poplar::Tensor L2RouterIn; 
    poplar::Tensor L3RouterOut;
    poplar::Tensor L3RouterIn; 
    poplar::Tensor raygenOutput;
    poplar::Tensor raygenInput;
    poplar::Tensor zero_const;
    poplar::Tensor zero_constf;
};

} // namespace ipu
} // namespace radfoam

#endif // RADIANT_FOAM_IPU_BUILDER_HPP
