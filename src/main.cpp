#include <iostream>
#include <vector>
#include <iomanip>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <random>
#include <future>

#include <cxxopts.hpp>
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
#include "ipu/rf_config.hpp"
#include "util/debug_utils.hpp"
#include "util/nanoflann.hpp"
#include "KDTreeManager.hpp"
#include "RadiantFoamIpuBuilder.hpp"

using namespace radfoam::geometry;
using namespace radfoam::config;

using ipu_utils::logger;

// Assemble full image from finished rays buffer, persistent between frames
static cv::Mat AssembleFinishedRaysImage(const std::vector<uint8_t>& finishedRaysHost, int mode) {
  // Persistent images (kept across frames)
  static cv::Mat rgb_img   (kFullImageHeight, kFullImageWidth, CV_8UC3, cv::Scalar(0,0,0));
  static cv::Mat depth_img (kFullImageHeight, kFullImageWidth, CV_8UC3, cv::Scalar(0,0,0));

  // ---- Build a small LUT once: t in [0,80] -> BGR ----
  // Using OpenCV just 81 times at startup is fine; the hot path uses the LUT.
  static std::array<cv::Vec3b, 81> T2BGR_LUT = []{
    std::array<cv::Vec3b, 81> lut{};
    for (int t = 0; t <= 80; ++t) {
      const int hue = static_cast<int>(std::lround(t * (179.0 / 80.0))); // OpenCV H: 0..179
      cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(hue, 255, 255));
      cv::Mat bgr;
      cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
      lut[t] = bgr.at<cv::Vec3b>(0, 0);
    }
    return lut;
  }();

  const size_t numFinishedRays = finishedRaysHost.size() / sizeof(FinishedRay);
  const FinishedRay* rays = reinterpret_cast<const FinishedRay*>(finishedRaysHost.data());

  // Rays are laid out as [tile0 rays][tile1 rays] ... with a per-tile sentinel x==0xFFFF
  const size_t num_finished_ray_per_tile = (kNumRayTracerTiles > 0)
        ? (numFinishedRays / kNumRayTracerTiles)
        : 0;

  const int W = kFullImageWidth;
  const int H = kFullImageHeight;
  const size_t rgb_step   = rgb_img.step;   // bytes per row
  const size_t depth_step = depth_img.step; // bytes per row
  uchar* const rgb_data   = rgb_img.data;
  uchar* const depth_data = depth_img.data;

  // ---- Parallel over tiles (each thread processes whole tiles) ----
  #pragma omp parallel for schedule(static)
  for (ptrdiff_t tid = 0; tid < static_cast<ptrdiff_t>(kNumRayTracerTiles); ++tid) {
    const FinishedRay* tileRays = rays + tid * num_finished_ray_per_tile;

    for (size_t i = 0; i < num_finished_ray_per_tile; ++i) {
      const FinishedRay& r = tileRays[i];

      // sentinel: no more rays in this tile
      if (r.x == 0xFFFF) break;

      // bounds check (kept branchy for safety; remove if guaranteed valid)
      if (r.x >= static_cast<uint16_t>(W) || r.y >= static_cast<uint16_t>(H)) continue;

      // Write RGB
      const size_t off_rgb = static_cast<size_t>(r.y) * rgb_step + static_cast<size_t>(r.x) * 3;
      uchar* const dst_rgb = rgb_data + off_rgb;
      dst_rgb[0] = r.b;
      dst_rgb[1] = r.g;
      dst_rgb[2] = r.r;

      // Write depth color via LUT
      int tt = static_cast<int>(std::lround(r.t));
      if (tt < 0)   tt = 0;
      if (tt > 80)  tt = 80;
      const cv::Vec3b bgr = T2BGR_LUT[tt];

      const size_t off_d = static_cast<size_t>(r.y) * depth_step + static_cast<size_t>(r.x) * 3;
      uchar* const dst_d = depth_data + off_d;
      dst_d[0] = bgr[0];
      dst_d[1] = bgr[1];
      dst_d[2] = bgr[2];
    }
  }

  return (mode == 0) ? rgb_img : depth_img;
}


int main(int argc, char** argv) {

  // glm::mat4 ViewMatrix(
  //   glm::vec4(-0.034899f,  0.000000f, -0.999391f, 0.000000f),
  //   glm::vec4( 0.484514f, -0.874620f, -0.016920f, 0.000000f),
  //   glm::vec4(-0.874087f, -0.484810f,  0.030524f, 0.000000f),
  //   glm::vec4(-0.000000f, -0.000000f, -6.700000f, 1.000000f)
  // );

  // glm::mat4 ViewMatrix(
  //   glm::vec4(-0.034899458f,  0.000000000f, -0.999390781f, 0.000000000f),
  //   glm::vec4( 0.484514207f, -0.874619782f, -0.016919592f, 0.000000000f),
  //   glm::vec4(-0.874086976f, -0.484809548f,  0.030523760f, 0.000000000f),
  //   glm::vec4(-0.000000000f, -0.000000000f, -6.699999809f, 1.000000000f)
  // );

  glm::mat4 ViewMatrix(
      glm::vec4(-0.995107710f,  0.000000000f,  0.098795786f, 0.000000000f),
      glm::vec4(-0.067882277f, -0.726565778f, -0.683735430f, 0.000000000f),
      glm::vec4( 0.071781643f, -0.687096834f,  0.723011255f, 0.000000000f),
      glm::vec4( 0.206200063f,  1.675502419f, -3.500002146f, 1.000000000f)
  );

  glm::mat4 ProjectionMatrix(
      glm::vec4(1.299038053f, 0.000000000f,  0.000000000f,  0.000000000f),
      glm::vec4(0.000000000f, 1.732050657f,  0.000000000f,  0.000000000f),
      glm::vec4(0.000000000f, 0.000000000f, -1.002002001f, -1.000000000f),
      glm::vec4(0.000000000f, 0.000000000f, -0.200200200f,  0.000000000f)
  );

  // ------------------------------
  // Profiling Trace Setup (PVTI)
  // ------------------------------
  pvti::TraceChannel traceChannel = {"RadiantFoamIpu"};

  // ------------------------------
  // Input Arguments
  // ------------------------------
  cxxopts::Options options("radiantfoam_ipu", "RadiantFoam IPU Renderer");

  options.add_options()
    ("i,input", "Input HDF5 file", cxxopts::value<std::string>()->default_value("./data/garden.h5"))
    ("n,nruns", "Number of runs to execute 0=inf", cxxopts::value<int>()->default_value("0"))
    ("no-ui", "Disable UI server", cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
    ("debug", "Enable debug reporting", cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
    ("p,port", "UI port", cxxopts::value<int>()->default_value("5000"))
    ("h,help", "Print usage");

  auto result = options.parse(argc, argv);

  if (result.count("help")) {
    std::cout << options.help() << std::endl;
    return 0;
  }

  std::string inputFile = result["input"].as<std::string>();
  int nRuns = result["nruns"].as<int>();
  int vis_mode = 0;
  bool enableUI = !result["no-ui"].as<bool>();
  bool enableDebug = result["debug"].as<bool>();
  int uiPort = result["port"].as<int>();

  glm::mat4 inverseView = glm::inverse(ViewMatrix);
  glm::vec3 cameraPos = glm::vec3(inverseView[3]);
  
  // KDTreeManager kdtree(inputFile);
  // 
  // auto camera_cell = kdtree.getNearestNeighbor(cameraPos);
  // fmt::print("Camera position ({:>8.4f}, {:>8.4f}, {:>8.4f})\n",
  // 						cameraPos.x, cameraPos.y, cameraPos.z);
  // fmt::print("Closest Point to Camera:\n"
  // 						"  Cluster: {:>5}\n"
  // 						"  Local  : {:>5}\n"
  // 						"  Position: ({:>8.4f}, {:>8.4f}, {:>8.4f})\n",
  // 						camera_cell.cluster_id, camera_cell.local_id, 
  // 						camera_cell.x, camera_cell.y, camera_cell.z);
  radfoam::geometry::GenericPoint camera_cell{
     0.6503f, -1.2979f,  3.3524f, 
     static_cast<uint16_t>(779),   // cluster_id
     static_cast<uint16_t>(3532)   // local_id
  };

  // ------------------------------
  // Poplar Engine Options
  // ------------------------------
  poplar::OptionFlags engineOptions = {};
  if (enableDebug) {
    logger()->info("Enabling Poplar auto-reporting (POPLAR_ENGINE_OPTIONS set)");
    setenv("POPLAR_ENGINE_OPTIONS", R"({"autoReport.all":"true", "autoReport.executionProfileProgramRunCount":"5","debug.retainDebugInformation":"true","autoReport.directory":"./report"})", 1);
    setenv("PVTI_OPTIONS", R"({"enable":"true"})", 1);
    engineOptions = {{"debug.instrument", "true"}};
  } else {
    unsetenv("POPLAR_ENGINE_OPTIONS");
    unsetenv("PVTI_OPTIONS");
    logger()->info("Poplar auto-reporting is NOT enabled");
    engineOptions.set("streamCallbacks.multiThreadMode", "collaborative");   // host side
    engineOptions.set("streamCallbacks.numWorkerThreads", "auto");
    engineOptions.set("streamCallbacks.numaAware", "true");
    engineOptions.set("debug.instrument",           "false");
    engineOptions.set("debug.verify",               "false");
    engineOptions.set("target.deterministicWorkers","false");
  }
  // ------------------------------
  // Build and Configure IPU Graph
  // ------------------------------
  radfoam::ipu::RadiantFoamIpuBuilder builder(inputFile, enableDebug);

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

  builder.stopFlagHost_ = 1;
  std::thread ipuThread([&] {
    mgr.execute(builder);
  });

  int step=0;
  do {
    step++;
    if(step==nRuns) {
      builder.stopFlagHost_ = 0;
      break;
    }
    glm::mat4 inverseView = glm::inverse(ViewMatrix);
    glm::mat4 inverseProj = glm::inverse(ProjectionMatrix);
    glm::vec3 cameraPos = glm::vec3(inverseView[3]);
    // auto camera_cell = kdtree.getNearestNeighbor(cameraPos);

    builder.updateViewMatrix(inverseView);
    builder.updateProjectionMatrix(inverseProj);
    builder.updateCameraCell(camera_cell);

    if (enableUI) {
      hostProcessing.waitForCompletion();
      state = uiServer->consumeState();
      if (state.device == "rgb") {
        vis_mode = 0;
      } else if(state.device == "depth") {
        vis_mode = 1;
      }
    }

    auto startTime = std::chrono::steady_clock::now();
    *imagePtr = AssembleFinishedRaysImage(builder.finishedRaysHost_, vis_mode);
    std::swap(imagePtr, imagePtrBuffered);
    if (enableUI) hostProcessing.run(uiUpdateFunc);
    
    state = enableUI && uiServer ? uiServer->consumeState() : InterfaceServer::State{};

    logger()->info("UI execution time: {}", std::chrono::duration<double>(std::chrono::steady_clock::now() - startTime).count());

    std::this_thread::sleep_for(std::chrono::milliseconds(16)); 
  } while (!enableUI || (uiServer && !state.stop));

  if (enableUI) hostProcessing.waitForCompletion();

  ipuThread.join();

  // cv::imwrite("framebuffer_full.png", AssembleFullImage(builder.framebuffer_host));
  cv::imwrite("framebuffer_full.png", AssembleFinishedRaysImage(builder.finishedRaysHost_, vis_mode));

  return 0;
}
