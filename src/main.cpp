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

#include "geometry/primitives.hpp"
#include "ipu/ipu_utils.hpp"
#include "ipu/rf_config.hpp"
#include "util/debug_utils.hpp"
#include "util/nanoflann.hpp"
#include "util/cmdline_options.hpp"
#include "util/ui_state.hpp"
#include "KDTreeManager.hpp"
#include "RadiantFoamIpuBuilder.hpp"

using namespace radfoam::geometry;
using namespace radfoam::config;
using radfoam::geometry::FinishedPixel;

using ipu_utils::logger;

static std::pair<cv::Mat, size_t> AssembleFinishedRaysImage(const std::vector<uint8_t>& finishedRaysHost, std::string mode) {
  static cv::Mat rgb_img   (kFullImageHeight, kFullImageWidth, CV_8UC3, cv::Scalar(0,0,0));
  static cv::Mat depth_img (kFullImageHeight, kFullImageWidth, CV_8UC3, cv::Scalar(0,0,0));

  static std::array<cv::Vec3b, 81> T2BGR_LUT = []{
    std::array<cv::Vec3b, 81> lut{};
    for (int t = 0; t <= 80; ++t) {
      const int hue = static_cast<int>(std::lround(t * (179.0 / 80.0)));
      cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(hue, 255, 255));
      cv::Mat bgr;
      cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
      lut[t] = bgr.at<cv::Vec3b>(0, 0);
    }
    return lut;
  }();

  const size_t numFinishedRays = finishedRaysHost.size() / sizeof(FinishedRay);
  const FinishedRay* rays = reinterpret_cast<const FinishedRay*>(finishedRaysHost.data());
  const size_t num_finished_ray_per_tile = (kNumRayTracerTiles > 0)
        ? (numFinishedRays / kNumRayTracerTiles)
        : 0;

  const int W = kFullImageWidth;
  const int H = kFullImageHeight;
  const size_t rgb_step   = rgb_img.step;
  const size_t depth_step = depth_img.step;
  uchar* const rgb_data   = rgb_img.data;
  uchar* const depth_data = depth_img.data;

  std::atomic<size_t> updatedPixels{0};

  #pragma omp parallel for schedule(static)
  for (ptrdiff_t tid = 0; tid < static_cast<ptrdiff_t>(kNumRayTracerTiles); ++tid) {
    const FinishedRay* tileRays = rays + tid * num_finished_ray_per_tile;

    const int NW = 6;
    const int kRayPerWorker = num_finished_ray_per_tile / NW;
    for(int wid = 0; wid<NW; wid++) {
      for (size_t i = wid*kRayPerWorker; i < (wid+1)*kRayPerWorker; ++i) {
        const FinishedRay& r = tileRays[i];

        if (r.x == 0xFFFF) continue;
        if (r.x >= static_cast<uint16_t>(W) || r.y >= static_cast<uint16_t>(H)) continue;

        const size_t off_rgb = static_cast<size_t>(r.y) * rgb_step + static_cast<size_t>(r.x) * 3;
        uchar* const dst_rgb = rgb_data + off_rgb;
        dst_rgb[0] = r.b;
        dst_rgb[1] = r.g;
        dst_rgb[2] = r.r;

        int tt = static_cast<int>(std::lround(r.t));
        if (tt < 0)   tt = 0;
        if (tt > 80)  tt = 80;
        const cv::Vec3b bgr = T2BGR_LUT[tt];

        const size_t off_d = static_cast<size_t>(r.y) * depth_step + static_cast<size_t>(r.x) * 3;
        uchar* const dst_d = depth_data + off_d;
        dst_d[0] = bgr[0];
        dst_d[1] = bgr[1];
        dst_d[2] = bgr[2];

        updatedPixels.fetch_add(1, std::memory_order_relaxed);
      }
    }
  }

  return {(mode == "depth") ? depth_img : rgb_img, updatedPixels.load()};
}

static std::pair<cv::Mat, size_t> AssembleFramebufferImage(const std::vector<uint8_t>& fbBytes, std::string mode)
{
  const int W = static_cast<int>(radfoam::config::kFullImageWidth);
  const int H = static_cast<int>(radfoam::config::kFullImageHeight);
  const int tileW = static_cast<int>(radfoam::config::kTileImageWidth);
  const int tileH = static_cast<int>(radfoam::config::kTileImageHeight);
  const int tilesX = static_cast<int>(radfoam::config::kNumRayTracerTilesX);
  const int tilesY = static_cast<int>(radfoam::config::kNumRayTracerTilesY);
  const std::size_t bytesPerTile = radfoam::config::kTileFramebufferSize;

  cv::Mat rgb_img   (H, W, CV_8UC3, cv::Scalar(0,0,0));
  cv::Mat depth_img;        
  cv::Mat depth_gray(H, W, CV_8UC1, cv::Scalar(0)); // per-pixel 0..255 depth intensity

  // Set your scene max depth here (was 80 earlier). Increase/decrease as desired.
  const float kMaxDepth = 20.f;

  std::atomic<size_t> updated{0};

  // Walk tiles
  for (int ty = 0; ty < tilesY; ++ty) {
    for (int tx = 0; tx < tilesX; ++tx) {
      const int tid = ty * tilesX + tx;
      const std::size_t base = static_cast<std::size_t>(tid) * bytesPerTile;
      const FinishedPixel* tile = reinterpret_cast<const FinishedPixel*>(fbBytes.data() + base);

      for (int ly = 0; ly < tileH; ++ly) {
        const int y = ty * tileH + ly;
        uint8_t* dstRGB   = rgb_img.ptr<uint8_t>(y);
        uint8_t* dstGray  = depth_gray.ptr<uint8_t>(y);

        for (int lx = 0; lx < tileW; ++lx) {
          const int x = tx * tileW + lx;
          const FinishedPixel& p = tile[ly * tileW + lx];

          // consider non-zero alpha or any nonzero rgb a valid update
          const bool nonzero = (p.a != 0) || (p.r|p.g|p.b);
          if (nonzero) updated.fetch_add(1, std::memory_order_relaxed);

          // RGB
          uint8_t* rgb = dstRGB + x*3;
          rgb[2] = p.r; rgb[1] = p.g; rgb[0] = p.b;

          // Grayscale depth 0..255 from p.t in [0..kMaxDepth]
          const float t_clamped = std::max(0.f, std::min(kMaxDepth, p.t));
          const float norm = t_clamped / kMaxDepth;                      // 0..1
          const uint8_t g = static_cast<uint8_t>(std::lround(norm * 255.f));
          dstGray[x] = g;
        }
      }
    }
  }

  cv::applyColorMap(255-depth_gray, depth_img, cv::COLORMAP_TURBO);

  return {(mode == "depth") ? depth_img : rgb_img, updated.load()};
}

size_t CountNonZeroPixels(const cv::Mat& img) {
    // Convert to grayscale mask (any nonzero channel -> 255)
    cv::Mat gray, mask;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, mask, 0, 255, cv::THRESH_BINARY);
    return static_cast<size_t>(cv::countNonZero(mask));
}

static poplar::OptionFlags makeEngineOptions(bool enableDebug) {
  poplar::OptionFlags engineOptions{};
  if (enableDebug) {
    logger()->info("Enabling Poplar auto-reporting (POPLAR_ENGINE_OPTIONS set)");
    setenv("POPLAR_ENGINE_OPTIONS",
           R"({"autoReport.all":"true","autoReport.executionProfileProgramRunCount":"10","target.hostSyncTimeout":"30","debug.retainDebugInformation":"true","autoReport.directory":"./report"})", 1);
    setenv("PVTI_OPTIONS", R"({"enable":"true"})", 1);
    engineOptions = {{"debug.instrument", "true"}};
  } else {
    unsetenv("POPLAR_ENGINE_OPTIONS");
    unsetenv("PVTI_OPTIONS");
    logger()->info("Poplar auto-reporting is NOT enabled");
    engineOptions.set("streamCallbacks.multiThreadMode", "collaborative");
    engineOptions.set("streamCallbacks.numWorkerThreads", "auto");
    engineOptions.set("streamCallbacks.numaAware",  "true");
    engineOptions.set("debug.instrument",           "false");
    engineOptions.set("debug.verify",               "false");
    engineOptions.set("target.deterministicWorkers","false");
  }
  return engineOptions;
}

static inline void checkRenderCompletion(const cv::Mat& img,
                                         size_t updatedCount,
                                         int iterTag,
                                         const std::chrono::steady_clock::time_point& startTime,
                                         size_t totalPixels,
                                         bool& fullImageUpdated,
                                         float threshold = 0.9995f)
{
  if (fullImageUpdated) return;

  const size_t nonZeroCount = CountNonZeroPixels(img);
  const double pct = (100.0 * static_cast<double>(nonZeroCount)) / static_cast<double>(totalPixels);

  std::cout << iterTag
            << ": Updated pixels: " << updatedCount
            << "  Non-zero: " << nonZeroCount << " (" << std::fixed << std::setprecision(2) << pct << "%)"
            << std::endl;

  if (nonZeroCount >= static_cast<size_t>(threshold * static_cast<double>(totalPixels))) {
    const auto now = std::chrono::steady_clock::now();
    const double elapsedSec = std::chrono::duration<double>(now - startTime).count();
    std::cout << "Full image updated in " << elapsedSec << " seconds." << std::endl;
    fullImageUpdated = true;
  }
}

int main(int argc, char** argv) {

  pvti::TraceChannel traceChannel = {"RadiantFoamIpu"};

  const rf::cli::CliOptions opt = rf::cli::parseOptions(argc, argv);
  
  std::unique_ptr<KDTreeManager> kdtree;
  kdtree = std::make_unique<KDTreeManager>(opt.inputFile);

  // ------------------------------
  // Build and Configure IPU Graph
  // ------------------------------
  auto engineOptions = makeEngineOptions(opt.enableDebug);
  radfoam::ipu::RadiantFoamIpuBuilder builder(opt.inputFile, opt.enableDeviceLoop, opt.repeatcounts, opt.enableDebug);

  ipu_utils::RuntimeConfig cfg{
    /*numIpus=*/1, /*numReplicas=*/1, /*exeName=*/"radiantfoam_ipu",
    /*useIpuModel=*/false, /*saveExe=*/false, /*loadExe=*/false,
    /*compileOnly=*/false, /*deferredAttach=*/false
  };
  builder.setRuntimeConfig(cfg);
  ipu_utils::GraphManager mgr;
  mgr.compileOrLoad(builder, engineOptions);
  mgr.prepareEngine(engineOptions);

  // ------------------------------
  // UI Setup
  // ------------------------------
  auto imagePtr         = std::make_unique<cv::Mat>(kFullImageHeight, kFullImageWidth, CV_8UC3);

  std::unique_ptr<InterfaceServer> uiServer;
  InterfaceServer::State state;
  state.fov    = glm::radians(60.f);
  if (opt.loadUIState) {
    if (rf::ui::LoadStateFromFile(state, opt.uiStatePath)) {
      logger()->info("Loaded UI state from {}", opt.uiStatePath);
    } else {
      logger()->warn("No UI state loaded (file not found or unreadable): {}", opt.uiStatePath);
    }
    state.stop = false;
  }

  if (opt.enableUI && opt.uiPort) {
    uiServer = std::make_unique<InterfaceServer>(opt.uiPort);
    uiServer->start();
    uiServer->initialiseVideoStream(imagePtr->cols, imagePtr->rows);
    uiServer->updateFov(state.fov);
  }
  
  // ------------------------------	
  // Main Execution & UI Loop
  // ------------------------------
  AsyncTask hostProcessing;
  auto uiUpdateFunc = [&]() {
    if (opt.enableUI && uiServer) {
      uiServer->sendPreviewImage(*imagePtr);
    }
  };
  
  std::cout << "IPU process started" << std::endl;
  auto startTime = std::chrono::steady_clock::now();
  const size_t totalPixels = kFullImageWidth * kFullImageHeight;
  bool fullImageUpdated = false;
  int step=0;
  
  if(opt.enableDeviceLoop) { // use repeatwhiletrue for IPU execution
    {
      if (opt.enableUI) {
        hostProcessing.waitForCompletion();
        if(!opt.loadUIState)
          state = uiServer->consumeState();
      }
      builder.updateCameraParameters(state);
      auto camera_position = builder.getCameraPos();
      auto camera_cell = kdtree->getNearestNeighbor(camera_position);
      builder.updateCameraCell(camera_cell);
    }
    builder.stopFlagHost_ = 1;
    std::thread ipuThread([&] {
      mgr.execute(builder);
    });

    do {
      step++;
      if(step==opt.nRuns) {
        builder.stopFlagHost_ = 0;
        break;
      }
      
      builder.updateCameraParameters(state);
      auto camera_position = builder.getCameraPos();
      auto camera_cell = kdtree->getNearestNeighbor(camera_position);
      builder.updateCameraCell(camera_cell);

      if (opt.enableUI) {
        hostProcessing.waitForCompletion();
        if(!opt.loadUIState)
          state = uiServer->consumeState();
      }
      
      static unsigned lastFence = 0;
      unsigned f;
      do {
        f = builder.frameFenceHost_.load(std::memory_order_acquire);
      } while (f == lastFence);   // wait for a new frame to complete its copy
      lastFence = f;

      // Take a stable snapshot of the streamed framebuffer bytes
      static std::vector<uint8_t> localCopy(builder.framebuffer_host.size());
      std::memcpy(localCopy.data(), builder.framebuffer_host.data(), localCopy.size());

      auto [imageMat, updatedCount] = AssembleFramebufferImage(localCopy, state.mode);
      *imagePtr = imageMat;

      if (opt.enableUI) hostProcessing.run(uiUpdateFunc);
      
      auto state2 = opt.enableUI && uiServer ? uiServer->consumeState() : InterfaceServer::State{};
      state.stop = state2.stop;

      checkRenderCompletion(*imagePtr, updatedCount, static_cast<int>(lastFence),
                      startTime, totalPixels, fullImageUpdated);

      // std::this_thread::sleep_for(std::chrono::milliseconds(1)); 

    } while (!opt.enableUI || (uiServer && !state.stop));

    builder.stopFlagHost_ = 0;
    ipuThread.join();

  } else { // execute IPU engine every time

    do {
      step++;
      if(step==opt.nRuns) {
        break;
      }
            
      builder.updateCameraParameters(state);
      auto camera_position = builder.getCameraPos();
      auto camera_cell = kdtree->getNearestNeighbor(camera_position);
      builder.updateCameraCell(camera_cell);

      if (opt.enableUI) {
        hostProcessing.waitForCompletion();
        state = uiServer->consumeState();
      }
      // per frame IPU program execution
      mgr.execute(builder);

      if (opt.enableDebug && opt.enableUI) {
        uiServer->sendHistogram(builder.raysCount_);
      }

      auto [imageMat, updatedCount] = AssembleFramebufferImage(builder.framebuffer_host, state.mode);
      *imagePtr = imageMat;

      if (opt.enableUI) hostProcessing.run(uiUpdateFunc);
      
      state = opt.enableUI && uiServer ? uiServer->consumeState() : InterfaceServer::State{};

      checkRenderCompletion(*imagePtr, updatedCount, step,
                      startTime, totalPixels, fullImageUpdated);

    } while (!opt.enableUI || (uiServer && !state.stop));
  }

  if (opt.saveUIState) {
    rf::ui::SaveStateToFile(state, opt.uiStatePath);
    logger()->info("Saved UI state to {}", opt.uiStatePath);
  }

  if (opt.enableUI) hostProcessing.waitForCompletion();

  auto [rgbMat, updatedCountRGB] = AssembleFramebufferImage(builder.framebuffer_host, "rgb");
  cv::imwrite("rgb_image.png", rgbMat);

  auto [depthMat, updatedCountDepth] = AssembleFramebufferImage(builder.framebuffer_host, "depth");
  cv::imwrite("depth_image.png", depthMat);

  return 0;
}
