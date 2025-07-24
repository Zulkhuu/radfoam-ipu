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
#include <poputil/TileMapping.hpp>

#include <remote_ui/InterfaceServer.hpp>
#include <remote_ui/AsyncTask.hpp>

#include <glm/glm.hpp>
#include <glm/mat4x4.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp> 

#include <pvti/pvti.hpp>

#include <opencv2/opencv.hpp> 

#include "util/debug_utils.hpp"
#include "geometry/primitives.hpp"
#include "ipu/ipu_utils.hpp"
#include "ipu/tile_config.hpp"
#include "io/hdf5_types.hpp"
#include "util/debug_utils.hpp"

using namespace radfoam::geometry;

using ipu_utils::logger;

// -----------------------------------------------------------------------------
// RadiantFoamIpuBuilder
class RadiantFoamIpuBuilder : public ipu_utils::BuilderInterface {
 public:
  RadiantFoamIpuBuilder(std::string h5_file, int tile_to_compute = 0)
      : h5_file_(std::move(h5_file)),
        tile_to_compute_(tile_to_compute),
				initialized_(false),
				hostViewMatrix_(16),
				hostProjMatrix_(16) {}

  // BuilderInterface overrides
  void build(poplar::Graph& graph, const poplar::Target& target) override;
  void execute(poplar::Engine& eng, const poplar::Device& dev) override;
	void updateViewMatrix(const glm::mat4& view);
	void updateProjectionMatrix(const glm::mat4& proj);

  // Host-visible full framebuffer
  std::vector<uint8_t> framebuffer_host_;
	std::vector<float> result_f32_host_;
	std::vector<uint16_t> result_u16_host_;

 private:
  // ---- constants -----------------------------------------------------------
  static constexpr unsigned kRaygenTile = 1024;
  static constexpr size_t   kNumRays    = 1500;

  // ---- helper --------------------------------------------------------------
  void readAllTiles(poplar::Engine& eng);

  // ---- members -------------------------------------------------------------
  std::string h5_file_;
  int         tile_to_compute_;
  bool        initialized_;//     = false;
	
  std::vector<std::vector<LocalPoint>>    local_pts_;
  std::vector<std::vector<NeighborPoint>> neighbor_pts_;
  std::vector<std::vector<uint16_t>>      adjacency_;

  poplar::Tensor rays_a_, rays_b_;

  std::vector<ipu_utils::StreamableTensor> local_tensors_;
  std::vector<ipu_utils::StreamableTensor> neighbor_tensors_;
  std::vector<ipu_utils::StreamableTensor> adj_tensors_;

	ipu_utils::StreamableTensor execCountT {"exec_count"};
	ipu_utils::StreamableTensor fb_read_all {"fb_read_all"};
	ipu_utils::StreamableTensor result_f32_read {"result_f32_read"};
	ipu_utils::StreamableTensor result_u16_read {"result_u16_read"};
	unsigned execCounterHost = 0;
	
  poplar::program::Sequence per_tile_writes_;
	
	std::vector<float> hostViewMatrix_;
	std::vector<float> hostProjMatrix_;
	ipu_utils::StreamableTensor viewMatrix_{"view_matrix"};
	ipu_utils::StreamableTensor projMatrix_{"proj_matrix"};
};

// -----------------------------------------------------------------------------
// build()
void RadiantFoamIpuBuilder::build(poplar::Graph& graph,
                                  const poplar::Target& /*target*/) {
  logger()->info("Building RadiantFoam graph (compute tile {})", tile_to_compute_);

  // ── Load HDF5 scene partitions ────────────────────────────────────────────
  local_pts_.resize(kNumTraceTiles);
  neighbor_pts_.resize(kNumTraceTiles);
  adjacency_.resize(kNumTraceTiles);

  {
    HighFive::File f(h5_file_, HighFive::File::ReadOnly);
    for (size_t tid = 0; tid < kNumTraceTiles; ++tid) {
      const auto g = f.getGroup(fmt::format("part{:04}", tid));
      g.getDataSet("local_pts").read(local_pts_[tid]);
      g.getDataSet("neighbor_pts").read(neighbor_pts_[tid]);
      g.getDataSet("adjacency_list").read(adjacency_[tid]);
    }
  }

  // ── Register codelets & popops ────────────────────────────────────────────
	std::string codeletFile = std::string(POPC_PREFIX) + "/src/codelets/codelets.cpp";
	std::string incPath     = std::string(POPC_PREFIX) + "/include/";
	std::string glmPath     = std::string(POPC_PREFIX) + "/external/glm/";
	graph.addCodelets(codeletFile, poplar::CodeletFileType::Auto,
										"-O3 -I " + incPath + " -I " + glmPath);
	popops::addCodelets(graph);

  // ── Allocate global tensors ───────────────────────────────────────────────
  const size_t kBytesPerTile = kNumRays * sizeof(Ray);
  const size_t kTotalBytes   = kBytesPerTile * kNumTraceTiles;
  rays_a_ = graph.addVariable(poplar::UNSIGNED_CHAR, {kTotalBytes}, "raysA");
  rays_b_ = graph.addVariable(poplar::UNSIGNED_CHAR, {kTotalBytes}, "raysB");

	result_f32_read.buildTensor(graph, poplar::FLOAT, {kNumTraceTiles});
	result_u16_read.buildTensor(graph, poplar::UNSIGNED_SHORT, {kNumTraceTiles});
  fb_read_all.buildTensor(graph, poplar::UNSIGNED_CHAR, {kNumTraceTiles, kTileFramebufferSize});

  // Faster mapping via poputil helpers (instead of manual loop):
  poputil::mapTensorLinearlyWithOffset(graph, result_f32_read.get(), 0);
  poputil::mapTensorLinearlyWithOffset(graph, result_u16_read.get(), 0);
  poputil::mapTensorLinearlyWithOffset(graph, fb_read_all.get().reshape({kNumTraceTiles, kTileFramebufferSize}), 0);

	// Create master matrices on tile 0
	poplar::program::Sequence broadcastMatrices;
	viewMatrix_.buildTensor(graph, poplar::FLOAT, {4, 4});
	graph.setTileMapping(viewMatrix_.get(), 0);
	broadcastMatrices.add(viewMatrix_.buildWrite(graph, true));

	projMatrix_.buildTensor(graph, poplar::FLOAT, {4, 4});
	graph.setTileMapping(projMatrix_.get(), 0);
	broadcastMatrices.add(projMatrix_.buildWrite(graph, true));


  // Zero-init rays buffers once per run
  const auto zero_const = graph.addConstant(poplar::UNSIGNED_CHAR, {kBytesPerTile}, 0);
  graph.setTileMapping(zero_const, 1024);

  poplar::program::Sequence zero_seq;

  // ── Create RayTrace vertices per tile ─────────────────────────────────────
  auto trace_cs = graph.addComputeSet("RayTraceCS");
	
  
	for (size_t tid = 0; tid < kNumTraceTiles; ++tid) {
    // ---- H2D tensors -------------------------------------------------------
    ipu_utils::StreamableTensor in_local(fmt::format("local_{}", tid));
    ipu_utils::StreamableTensor in_nbr(fmt::format("nbr_{}", tid));
    ipu_utils::StreamableTensor in_adj(fmt::format("adj_{}", tid));

    in_local.buildTensor(graph, poplar::UNSIGNED_CHAR, {sizeof(LocalPoint) * local_pts_[tid].size()});
    in_nbr.buildTensor(graph,  poplar::UNSIGNED_CHAR, {sizeof(NeighborPoint) * neighbor_pts_[tid].size()});
    in_adj.buildTensor(graph,  poplar::UNSIGNED_SHORT, {adjacency_[tid].size()});

    graph.setTileMapping(in_local.get(), tid);
    graph.setTileMapping(in_nbr.get(), tid);
    graph.setTileMapping(in_adj.get(), tid);

    // ---- Vertex -----------------------------------------------------------
    auto v = graph.addVertex(trace_cs, "RayTrace");

	// Clone view matrix and broadcast
		auto localView = graph.clone(viewMatrix_.get(), "view_matrix_tile_" + std::to_string(tid));
		graph.setTileMapping(localView, tid);
		broadcastMatrices.add(poplar::program::Copy(viewMatrix_.get(), localView));
		graph.connect(v["view_matrix"], localView.flatten());

		// Clone projection matrix and broadcast
		auto localProj = graph.clone(projMatrix_.get(), "proj_matrix_tile_" + std::to_string(tid));
		graph.setTileMapping(localProj, tid);
		broadcastMatrices.add(poplar::program::Copy(projMatrix_.get(), localProj));
		graph.connect(v["projection_matrix"], localProj.flatten());

    graph.connect(v["local_pts"],    in_local.get());
    graph.connect(v["neighbor_pts"], in_nbr.get());
    graph.connect(v["adjacency"],    in_adj.get());

    const auto slice_a = rays_a_.slice(tid * kBytesPerTile, (tid + 1) * kBytesPerTile);
    const auto slice_b = rays_b_.slice(tid * kBytesPerTile, (tid + 1) * kBytesPerTile);

    graph.setTileMapping(slice_a, tid);
    graph.setTileMapping(slice_b, tid);

    zero_seq.add(poplar::program::Copy(zero_const, slice_a));
    zero_seq.add(poplar::program::Copy(zero_const, slice_b));

    graph.connect(v["raysIn"],  slice_b);
    graph.connect(v["raysOut"], slice_a);

    graph.connect(v["result_float"], result_f32_read.get().slice({tid}, {tid + 1}).reshape({}));
    graph.connect(v["result_u16"], result_u16_read.get().slice({tid}, {tid + 1}).reshape({}));

    auto framebuffer_slice = fb_read_all.get().slice({tid, 0}, {tid + 1, kTileFramebufferSize}).reshape({kTileFramebufferSize});
    graph.connect(v["framebuffer"], framebuffer_slice);

    auto tile_const = graph.addConstant(poplar::UNSIGNED_SHORT, {}, static_cast<unsigned>(tid));
    graph.setTileMapping(tile_const, tid);
    graph.connect(v["tile_id"], tile_const);

    graph.setTileMapping(v, tid);

    per_tile_writes_.add(in_local.buildWrite(graph, true));
    per_tile_writes_.add(in_nbr.buildWrite(graph, true));
    per_tile_writes_.add(in_adj.buildWrite(graph, true));

    local_tensors_.push_back(std::move(in_local));
    neighbor_tensors_.push_back(std::move(in_nbr));
    adj_tensors_.push_back(std::move(in_adj));
  }

  getPrograms().add("RayTraceCS", poplar::program::Execute(trace_cs));
	getPrograms().add("broadcast_matrices", broadcastMatrices);

  // ── RayGen (single tile) ─────────────────────────────────────────────────
  {
    auto raygen_cs = graph.addComputeSet("RayGenCS");
    auto v         = graph.addVertex(raygen_cs, "RayGen");

    const auto slice_a = rays_a_.slice(tile_to_compute_ * kBytesPerTile, (tile_to_compute_ + 1) * kBytesPerTile);
    const auto slice_b = rays_b_.slice(tile_to_compute_ * kBytesPerTile, (tile_to_compute_ + 1) * kBytesPerTile);

		execCountT.buildTensor(graph, poplar::UNSIGNED_INT, {});
		graph.setTileMapping(execCountT.get(), kRaygenTile);
		graph.connect(v["exec_count"], execCountT.get());
		getPrograms().add("write_exec_count", execCountT.buildWrite(graph, true));

    graph.connect(v["raysIn"],  slice_a);
    graph.connect(v["raysOut"], slice_b);
    graph.setTileMapping(v, kRaygenTile);

    getPrograms().add("RayGenCS", poplar::program::Execute(raygen_cs));
  }

  // ── Host-device streams & zero-init program ──────────────────────────────
  getPrograms().add("zero_rays", zero_seq);
  getPrograms().add("write",     per_tile_writes_);

  // ── Unified framebuffer read ─────────────────────────────────────────────
  framebuffer_host_.resize(kTileFramebufferSize * kNumTraceTiles);
	result_f32_host_.resize(kNumTraceTiles);
	result_u16_host_.resize(kNumTraceTiles);

  getPrograms().add("fb_read_all", fb_read_all.buildRead(graph, true));
	getPrograms().add("read_result_f32", result_f32_read.buildRead(graph, true));
	getPrograms().add("read_result_u16", result_u16_read.buildRead(graph, true));
}

// -----------------------------------------------------------------------------
// execute()
void RadiantFoamIpuBuilder::execute(poplar::Engine& engine,
                                    const poplar::Device&) {
  if (!initialized_) {
    // Connect all scene-write streams once
    for (size_t tid = 0; tid < local_tensors_.size(); ++tid) {
      local_tensors_[tid].connectWriteStream(engine, local_pts_[tid]);
      neighbor_tensors_[tid].connectWriteStream(engine, neighbor_pts_[tid]);
      adj_tensors_[tid].connectWriteStream(engine, adjacency_[tid]);
			fb_read_all.connectReadStream(engine, framebuffer_host_.data());
			result_f32_read.connectReadStream(engine, result_f32_host_.data());
			result_u16_read.connectReadStream(engine, result_u16_host_.data());

    }
		viewMatrix_.connectWriteStream(engine, hostViewMatrix_);
  	projMatrix_.connectWriteStream(engine, hostProjMatrix_);
    execCountT.connectWriteStream(engine, &execCounterHost);
    engine.run(getPrograms().getOrdinals().at("zero_rays"));
    engine.run(getPrograms().getOrdinals().at("write"));
    initialized_ = true;
  }

	engine.run(getPrograms().getOrdinals().at("broadcast_matrices"));
	engine.run(getPrograms().getOrdinals().at("write_exec_count"));
  engine.run(getPrograms().getOrdinals().at("RayGenCS"));
  engine.run(getPrograms().getOrdinals().at("RayTraceCS"));
  readAllTiles(engine);
	execCounterHost++;
}

void RadiantFoamIpuBuilder::updateViewMatrix(const glm::mat4& view) {
  // Graphcore expects row-major floats; glm::value_ptr is column-major
  glm::mat4 transposed = glm::transpose(view);
  const float* ptr = glm::value_ptr(transposed);
  for (size_t i = 0; i < 16; ++i) {
    hostViewMatrix_[i] = ptr[i];
  }
}

void RadiantFoamIpuBuilder::updateProjectionMatrix(const glm::mat4& proj) {
  glm::mat4 transposed = glm::transpose(proj);
  const float* ptr = glm::value_ptr(transposed);
  for (size_t i = 0; i < 16; ++i) {
    hostProjMatrix_[i] = ptr[i];
  }
}

// -----------------------------------------------------------------------------
// readAllTiles()
void RadiantFoamIpuBuilder::readAllTiles(poplar::Engine& engine) {
	engine.run(getPrograms().getOrdinals().at("fb_read_all"));
	engine.run(getPrograms().getOrdinals().at("read_result_f32"));
	engine.run(getPrograms().getOrdinals().at("read_result_u16"));

	logger()->info("Computed results for tile {} -> float: {}, u16: {}",
								tile_to_compute_,
								result_f32_host_[tile_to_compute_],
								result_u16_host_[tile_to_compute_]);
	const size_t offset = tile_to_compute_ * kTileFramebufferSize;
	logger()->info("Framebuffer[0:10]: {}",
               	radfoam::util::VectorSliceToString(framebuffer_host_, offset, offset + 10));
}

// -----------------------------------------------------------------------------
// Assemble full framebuffer → cv::Mat
static cv::Mat AssembleFullImage(const std::vector<uint8_t>& tiles) {
  cv::Mat img(kFullImageHeight, kFullImageWidth, CV_8UC3);

  for (size_t ty = 0; ty < kNumTilesY; ++ty) {
    for (size_t tx = 0; tx < kNumTilesX; ++tx) {
      const size_t idx  = ty * kNumTilesX + tx;
      const uint8_t* src = tiles.data() + idx * kTileFramebufferSize;

      for (size_t y = 0; y < kTileHeight; ++y) {
        std::memcpy(img.ptr<uint8_t>(ty * kTileHeight + y) +
                        tx * kTileWidth * 3,
                    src + y * kTileWidth * 3,
                    kTileWidth * 3);
      }
    }
  }
  return img;
}

int main(int argc, char** argv) {

	glm::mat4 ViewMatrix(
    glm::vec4(-0.034899f,  0.000000f, -0.999391f, 0.000000f),
    glm::vec4( 0.484514f, -0.874620f, -0.016920f, 0.000000f),
    glm::vec4(-0.874087f, -0.484810f,  0.030524f, 0.000000f),
    glm::vec4(-0.000000f, -0.000000f, -6.700000f, 1.000000f)
	);

	glm::mat4 ProjectionMatrix(
    glm::vec4(1.299038f, 0.000000f,  0.000000f,  0.000000f),
    glm::vec4(0.000000f, 1.732051f,  0.000000f,  0.000000f),
    glm::vec4(0.000000f, 0.000000f, -1.002002f, -1.000000f),
    glm::vec4(0.000000f, 0.000000f, -0.200200f,  0.000000f)
	);

  // ------------------------------
  // Profiling Trace Setup (PVTI)
  // ------------------------------
  pvti::TraceChannel traceChannel = {"RadiantFoamIpu"};

  // ------------------------------
  // Input Arguments
  // ------------------------------
  const std::string inputFile = (argc > 1) ? argv[1] : "./data/garden.h5";
  const int tileToDebug       = (argc > 2) ? std::stoi(argv[2]) : 0;
	bool enableUI = true;
	for (int i = 1; i < argc; ++i) {
		if (std::string(argv[i]) == "--no-ui") {
			enableUI = false;
		}
	}
  // ------------------------------
  // Poplar Engine Options
  // ------------------------------
  poplar::OptionFlags engineOptions;
  if (radfoam::util::isPoplarEngineOptionsEnabled()) {
    logger()->info("Poplar auto-reporting is enabled (POPLAR_ENGINE_OPTIONS set)");
    engineOptions = {{"debug.instrument", "true"}};
  } else {
    logger()->info("Poplar auto-reporting is NOT enabled");
    engineOptions = {};
  }

  // ------------------------------
  // Build and Configure IPU Graph
  // ------------------------------
  RadiantFoamIpuBuilder builder(inputFile, tileToDebug);

  ipu_utils::RuntimeConfig cfg{
    /*numIpus=*/1,
    /*numReplicas=*/1,
    /*exeName=*/"radiantfoam_ipu",
    /*useIpuModel=*/false,
    /*saveExe=*/false,
    /*loadExe=*/false,
    /*compileOnly=*/false,
    /*deferredAttach=*/false
  };
  builder.setRuntimeConfig(cfg);

  ipu_utils::GraphManager mgr;

  // Compile and Prepare Engine with Tracepoints
  pvti::Tracepoint::begin(&traceChannel, "constructing_graph");
  mgr.compileOrLoad(builder, engineOptions);
  mgr.prepareEngine(engineOptions);
  pvti::Tracepoint::end(&traceChannel, "constructing_graph");

  // ------------------------------
  // UI Setup
  // ------------------------------
  auto imagePtr         = std::make_unique<cv::Mat>(kFullImageHeight, kFullImageWidth, CV_8UC3);
  auto imagePtrBuffered = std::make_unique<cv::Mat>(kFullImageHeight, kFullImageWidth, CV_8UC3);

  std::unique_ptr<InterfaceServer> uiServer;
  InterfaceServer::State state;
  state.fov    = glm::radians(40.f);
  state.device = "cpu"; // Could parameterize if needed
  const int uiPort = 5000;

	if (enableUI && uiPort) {
    uiServer = std::make_unique<InterfaceServer>(uiPort);
    uiServer->start();
    uiServer->initialiseVideoStream(imagePtr->cols, imagePtr->rows);
    uiServer->updateFov(state.fov);
  }

  // ------------------------------
  // Main Execution & UI Loop
  // ------------------------------
	AsyncTask hostProcessing;
	auto uiUpdateFunc = [&]() {
		if (enableUI && uiServer) {
			uiServer->sendPreviewImage(*imagePtrBuffered);
		}
	};

	do {
		ViewMatrix[2][2] += 2;
		ProjectionMatrix[2][2] += 1;
		builder.updateViewMatrix(ViewMatrix);
		builder.updateProjectionMatrix(ProjectionMatrix);

		if (enableUI) hostProcessing.waitForCompletion();
		mgr.execute(builder);
		*imagePtr = AssembleFullImage(builder.framebuffer_host_);
		std::swap(imagePtr, imagePtrBuffered);
		if (enableUI) hostProcessing.run(uiUpdateFunc);

		state = enableUI && uiServer ? uiServer->consumeState() : InterfaceServer::State{};
	} while (!enableUI || (uiServer && !state.stop));

	if (enableUI) hostProcessing.waitForCompletion();

  cv::imwrite("framebuffer_full.png", AssembleFullImage(builder.framebuffer_host_));

  return 0;
}
