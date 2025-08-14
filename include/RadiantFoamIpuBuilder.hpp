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
#include "ipu/rf_config.hpp"
#include "ipu/ipu_utils.hpp"
#include "geometry/primitives.hpp"
#include "io/hdf5_types.hpp"

namespace radfoam {
namespace ipu {

class RadiantFoamIpuBuilder final : public ipu_utils::BuilderInterface {
public:
    explicit RadiantFoamIpuBuilder(std::string h5_scene_file, bool loop_frames = false, bool debug = false);

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

    unsigned stopFlagHost_;
    std::atomic<uint32_t> frameFenceHost_{0};

private:
    // ───────────── helper sections used by build() ─────────────
    void loadScenePartitions();
    void registerCodeletsAndOps(poplar::Graph& g);
    void allocateGlobalTensors(poplar::Graph& g);
    void buildRayTracers(poplar::Graph& g, poplar::ComputeSet& cs);
    void buildRayGenerator(poplar::Graph& g, poplar::ComputeSet& cs);
    void buildRayRoutersL0(poplar::Graph& g, poplar::ComputeSet& cs);
    void buildRayRoutersL1(poplar::Graph& g, poplar::ComputeSet& cs);
    void buildRayRoutersL2(poplar::Graph& g, poplar::ComputeSet& cs);
    void buildRayRoutersL3(poplar::Graph& g, poplar::ComputeSet& cs);
    void buildRayRoutersL4(poplar::Graph& g, poplar::ComputeSet& cs);
    poplar::program::Sequence buildDataExchange(poplar::Graph& g);
    void setupHostStreams(poplar::Graph& g);
    void connectHostStreams(poplar::Engine& eng);
    void readAllTiles(poplar::Engine& engine);

    // ───────────── members ─────────────
    // Construction‑time constants
    const std::string h5_file_;
    bool debug_;
    bool loop_frames_;
    static constexpr int kSubsteps = 1;
    static constexpr int kRouterDebugSize = 15;

    // Scene data -------------------------------------------------------------
    std::vector<std::vector<radfoam::geometry::LocalPoint>>   local_pts_;
    std::vector<std::vector<radfoam::geometry::GenericPoint>> neighbor_pts_;
    std::vector<std::vector<uint16_t>>                        adjacency_;
    std::vector<std::string>                                  paths_;

    // Poplar tensors ---------------------------------------------------------
    poplar::Tensor rayTracerOutputRays_;
    poplar::Tensor rayTracerInputRays_;
    poplar::Tensor L0RouterOut;
    poplar::Tensor L0RouterIn; 
    poplar::Tensor L1RouterOut;
    poplar::Tensor L1RouterIn; 
    poplar::Tensor L2RouterOut;
    poplar::Tensor L2RouterIn; 
    poplar::Tensor L3RouterOut;
    poplar::Tensor L3RouterIn; 
    poplar::Tensor L4RouterOut;
    poplar::Tensor L4RouterIn; 
    poplar::Tensor raygenOutput;
    poplar::Tensor raygenInput;

    std::vector<ipu_utils::StreamableTensor> local_tensors_;
    std::vector<ipu_utils::StreamableTensor> neighbor_tensors_;
    std::vector<ipu_utils::StreamableTensor> adj_tensors_;

    ipu_utils::StreamableTensor exec_counts_{"exec_count"};
    ipu_utils::StreamableTensor fb_read_all_{"fb_read_all"};
    ipu_utils::StreamableTensor result_f32_read_{"result_f32_read"};
    ipu_utils::StreamableTensor result_u16_read_{"result_u16_read"};
    ipu_utils::StreamableTensor viewMatrix_{"view_matrix"};
    ipu_utils::StreamableTensor projMatrix_{"proj_matrix"};
    ipu_utils::StreamableTensor cameraCellInfo_{"camera_cell_info"};
    ipu_utils::StreamableTensor l0routerDebugRead_{"l0_router_debug_bytes_read"};
    ipu_utils::StreamableTensor l1routerDebugRead_{"l1_router_debug_bytes_read"};
    ipu_utils::StreamableTensor l2routerDebugRead_{"l2_router_debug_bytes_read"};
    ipu_utils::StreamableTensor l3routerDebugRead_{"l3_router_debug_bytes_read"};
    ipu_utils::StreamableTensor l4routerDebugRead_{"l4_router_debug_bytes_read"};
    ipu_utils::StreamableTensor raygenDebugRead_{"raygen_debug_bytes_read"};
    ipu_utils::StreamableTensor finishedRaysRead_{"finished_rays_read"};
    ipu_utils::StreamableTensor finishedWriteOffsets_{"finished_rays_write_offsets"};

    poplar::Tensor inStreamFinishedRays;
    ipu_utils::StreamableTensor stopFlag_{"stopFlag"};
    poplar::DataStream inStream, stopFlagStream;

    // CPU‑side mirrors -------------------------------------------------------
    int exec_counter_;
    std::vector<float> hostViewMatrix_;
    std::vector<float> hostProjMatrix_;
    std::array<uint8_t, 4> hostCameraCellInfo_{{0,0,0,0}};
    std::vector<unsigned> l0routerDebugBytesHost_;
    std::vector<unsigned> l1routerDebugBytesHost_;
    std::vector<unsigned> l2routerDebugBytesHost_;
    std::vector<unsigned> l3routerDebugBytesHost_;
    std::vector<unsigned> l4routerDebugBytesHost_;
    std::vector<unsigned> raygenDebugBytesHost_;

    // Helper program sequences ----------------------------------------------
    poplar::program::Sequence per_tile_writes_;
    poplar::program::Sequence broadcastMatrices_;

};

} // namespace ipu
} // namespace radfoam

#endif // RADIANT_FOAM_IPU_BUILDER_HPP
