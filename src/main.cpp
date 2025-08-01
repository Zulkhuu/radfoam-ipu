#include <iostream>
#include <vector>
#include <iomanip>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <random>

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
#include "ipu/tile_config.hpp"
#include "util/debug_utils.hpp"
#include "util/nanoflann.hpp"
#include "KDTreeManager.hpp"
#include "RadiantFoamIpuBuilder.hpp"

using namespace radfoam::geometry;

using ipu_utils::logger;

// -----------------------------------------------------------------------------
// Assemble full framebuffer → cv::Mat
static cv::Mat AssembleFullImage(const std::vector<uint8_t>& tiles) {
  cv::Mat img(kFullImageHeight, kFullImageWidth, CV_8UC3);

  for (size_t ty = 0; ty < kNumRayTracerTilesY; ++ty) {
    for (size_t tx = 0; tx < kNumRayTracerTilesX; ++tx) {
      const size_t idx  = ty * kNumRayTracerTilesX + tx;
      const uint8_t* src = tiles.data() + idx * kTileFramebufferSize;

      for (size_t y = 0; y < kTileImageHeight; ++y) {
        std::memcpy(img.ptr<uint8_t>(ty * kTileImageHeight + y) +
										tx * kTileImageWidth * 3,
                    src + y * kTileImageWidth * 3,
                    kTileImageWidth * 3);
      }
    }
  }
  return img;
}

inline cv::Vec3b mapTtoHSV(float t) {
    if (t < 0) t = 0;
    if (t > 80) t = 80;

    // Map to hue range (0-179)
    float hue = (t / 80.0f) * 179.0f;

    // Full saturation and value
    return cv::Vec3b(static_cast<uint8_t>(hue), 255, 255);
}

// Assemble full image from finished rays buffer, persistent between frames
static cv::Mat AssembleFinishedRaysImage(const std::vector<uint8_t>& finishedRaysHost, int mode) {
    static cv::Mat img(kFullImageHeight, kFullImageWidth, CV_8UC3, cv::Scalar(0,0,0)); 
    // Retains previous content across calls

    const size_t numFinishedRays = finishedRaysHost.size() / sizeof(FinishedRay);
    const FinishedRay* rays = reinterpret_cast<const FinishedRay*>(finishedRaysHost.data());
    for (size_t i = 0; i < numFinishedRays; ++i) {
        const FinishedRay& r = rays[i];

        // Skip invalid rays
        if (r.x == 0xFFFF || r.y == 0xFFFF) continue;
        if (r.x >= kFullImageWidth || r.y >= kFullImageHeight) continue;

        // Update pixel color in persistent image
        if(mode == 0) {
          cv::Vec3b& pixel = img.at<cv::Vec3b>(r.y, r.x);
          pixel[0] = r.b; // B
          pixel[1] = r.g; // G
          pixel[2] = r.r; // R
        } else if (mode == 1) {
          // Convert t to HSV color
          cv::Vec3b hsv = mapTtoHSV(r.t);
  
          // Convert HSV to BGR (OpenCV expects H,S,V format in 8-bit)
          cv::Mat hsvMat(1, 1, CV_8UC3, hsv);
          cv::Mat bgrMat;
          cv::cvtColor(hsvMat, bgrMat, cv::COLOR_HSV2BGR);
  
          // Assign pixel color
          img.at<cv::Vec3b>(r.y, r.x) = bgrMat.at<cv::Vec3b>(0, 0);
        }

				// fmt::print("Ray {:3}: (x={}, y={}) RGB=({}, {}, {})\n",
        //                i, r.x, r.y, r.r, r.g, r.b);
    }

    return img; // Returns reference (copy-on-write in OpenCV)
}



int main(int argc, char** argv) {

	// glm::mat4 ViewMatrix(
  //   glm::vec4(-0.034899f,  0.000000f, -0.999391f, 0.000000f),
  //   glm::vec4( 0.484514f, -0.874620f, -0.016920f, 0.000000f),
  //   glm::vec4(-0.874087f, -0.484810f,  0.030524f, 0.000000f),
  //   glm::vec4(-0.000000f, -0.000000f, -6.700000f, 1.000000f)
	// );

	// glm::mat4 ProjectionMatrix(
  //   glm::vec4(1.299038f, 0.000000f,  0.000000f,  0.000000f),
  //   glm::vec4(0.000000f, 1.732051f,  0.000000f,  0.000000f),
  //   glm::vec4(0.000000f, 0.000000f, -1.002002f, -1.000000f),
  //   glm::vec4(0.000000f, 0.000000f, -0.200200f,  0.000000f)
	// );
	// glm::mat4 ViewMatrix(
  //   glm::vec4(-0.034899458f,  0.000000000f, -0.999390781f, 0.000000000f),
  //   glm::vec4( 0.484514207f, -0.874619782f, -0.016919592f, 0.000000000f),
  //   glm::vec4(-0.874086976f, -0.484809548f,  0.030523760f, 0.000000000f),
  //   glm::vec4(-0.000000000f, -0.000000000f, -6.699999809f, 1.000000000f)
	// );

	// glm::mat4 ProjectionMatrix(
	// 	glm::vec4(1.299038053f, 0.000000000f,  0.000000000f,  0.000000000f),
	// 	glm::vec4(0.000000000f, 1.732050657f,  0.000000000f,  0.000000000f),
	// 	glm::vec4(0.000000000f, 0.000000000f, -1.002002001f, -1.000000000f),
	// 	glm::vec4(0.000000000f, 0.000000000f, -0.200200200f,  0.000000000f)
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


	fmt::print("Ray size: {}, kNumRays: {}, buffer size: {}\n",
       sizeof(Ray), kNumRays, kNumRays * sizeof(Ray));
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
    ("t,tile", "Tile to debug", cxxopts::value<int>()->default_value("0"))
    ("m,mode", "Display mode rgb/depth", cxxopts::value<int>()->default_value("0"))
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
	int tileToDebug = result["tile"].as<int>();
	int vis_mode = result["mode"].as<int>();
	bool enableUI = !result["no-ui"].as<bool>();
	bool enableDebug = result["debug"].as<bool>();
	int uiPort = result["port"].as<int>();

	KDTreeManager kdtree(inputFile);

	glm::mat4 inverseView = glm::inverse(ViewMatrix);
	glm::vec3 cameraPos = glm::vec3(inverseView[3]);
	auto camera_cell = kdtree.getNearestNeighbor(cameraPos);
	fmt::print("Camera position ({:>8.4f}, {:>8.4f}, {:>8.4f})\n",
							cameraPos.x, cameraPos.y, cameraPos.z);
	fmt::print("Closest Point to Camera:\n"
							"  Cluster: {:>5}\n"
							"  Local  : {:>5}\n"
							"  Position: ({:>8.4f}, {:>8.4f}, {:>8.4f})\n",
							camera_cell.cluster_id, camera_cell.local_id, 
							camera_cell.x, camera_cell.y, camera_cell.z);

  // ------------------------------
  // Poplar Engine Options
  // ------------------------------
  poplar::OptionFlags engineOptions = {};
  // if (radfoam::util::isPoplarEngineOptionsEnabled()) {
  //   logger()->info("Poplar auto-reporting is enabled (POPLAR_ENGINE_OPTIONS set)");
  //   engineOptions = {{"debug.instrument", "true"}};
  // } else {
  //   logger()->info("Poplar auto-reporting is NOT enabled");
  //   engineOptions = {};
  // }
  if (enableDebug) {
    logger()->info("Enabling Poplar auto-reporting (POPLAR_ENGINE_OPTIONS set)");
		setenv("POPLAR_ENGINE_OPTIONS", R"({"autoReport.all":"true","autoReport.directory":"./report"})", 1);
    engineOptions = {{"debug.instrument", "true"}};
    // engineOptions.set("debug.instrument", "true");
    // engineOptions.set("autoReport.all", "true");
    // engineOptions.set("autoReport.directory", "./report");
  } else {
		unsetenv("POPLAR_ENGINE_OPTIONS");
    logger()->info("Poplar auto-reporting is NOT enabled");
    // engineOptions.set("autoReport.all", "false");
    // engineOptions = {};
  }
  // ------------------------------
  // Build and Configure IPU Graph
  // ------------------------------
  radfoam::ipu::RadiantFoamIpuBuilder builder(inputFile, tileToDebug, enableDebug);

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

	int i=0;
	do {
		// ViewMatrix[2][2] += 2;
		// ProjectionMatrix[2][2] += 1;
		i++;
		// if(i==builder.debug_chains_.size()+2) break;
		if(i==tileToDebug) break;
		glm::mat4 inverseView = glm::inverse(ViewMatrix);
		glm::mat4 inverseProj = glm::inverse(ProjectionMatrix);
		glm::vec3 cameraPos = glm::vec3(inverseView[3]);
		auto camera_cell = kdtree.getNearestNeighbor(cameraPos);

		builder.updateViewMatrix(inverseView);
		builder.updateProjectionMatrix(inverseProj);
		builder.updateCameraCell(camera_cell);

		if (enableUI) hostProcessing.waitForCompletion();
		mgr.execute(builder);
		// *imagePtr = AssembleFullImage(builder.framebuffer_host);
		*imagePtr = AssembleFinishedRaysImage(builder.finishedRaysHost_, vis_mode);
		std::swap(imagePtr, imagePtrBuffered);
		if (enableUI) hostProcessing.run(uiUpdateFunc);

		state = enableUI && uiServer ? uiServer->consumeState() : InterfaceServer::State{};
	} while (!enableUI || (uiServer && !state.stop));

	if (enableUI) hostProcessing.waitForCompletion();

  // cv::imwrite("framebuffer_full.png", AssembleFullImage(builder.framebuffer_host));
  cv::imwrite("framebuffer_full.png", AssembleFinishedRaysImage(builder.finishedRaysHost_, vis_mode));

  return 0;
}
