#include "util/cmdline_options.hpp"
#include <cxxopts.hpp>
#include <algorithm>
#include <cstdlib>
#include <iostream>

namespace rf::cli {

CliOptions parseOptions(int argc, char** argv) {
  cxxopts::Options options("radiantfoam_ipu", "RadiantFoam IPU Renderer");
  options.add_options()
    ("i,input", "Input HDF5 file",
      cxxopts::value<std::string>()->default_value("./data/garden.h5"))
    ("n,nruns", "Number of runs to execute 0=inf",
      cxxopts::value<int>()->default_value("0"))
    ("no-ui", "Disable UI server",
      cxxopts::value<bool>()->default_value("false")->implicit_value("true"))

    ("device-loop", "Enable multi frame IPU loop before CPU read",
      cxxopts::value<bool>()->default_value("true")->implicit_value("true"))
    ("no-device-loop", "Disable multi frame IPU loop",
      cxxopts::value<bool>()->default_value("false")->implicit_value("true"))

    ("debug", "Enable debug reporting",
      cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
    ("p,port", "UI port",
      cxxopts::value<int>()->default_value("5000"))

    ("r,repeatcounts", "Number of CS-DX repeats per frame (>=1)",
      cxxopts::value<unsigned>()->default_value("20"))
    ("repeatcount", "Alias of --repeatcounts (deprecated)",
      cxxopts::value<unsigned>())
    ("s,substeps", "DEPRECATED: use --repeatcounts",
      cxxopts::value<unsigned>())

    ("ui-state", "Path to UI state file",
      cxxopts::value<std::string>()->default_value("ui_state.txt"))
    ("load-ui-state", "Load UI state from file at startup",
      cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
    ("save-ui-state", "Save UI state to file at shutdown",
      cxxopts::value<bool>()->default_value("false")->implicit_value("true"))

    ("h,help", "Print usage");

  auto result = options.parse(argc, argv);
  if (result.count("help")) {
    std::cout << options.help() << std::endl;
    std::exit(0);
  }

  CliOptions opts;
  opts.inputFile        = result["input"].as<std::string>();
  opts.nRuns            = result["nruns"].as<int>();
  opts.enableUI         = !result["no-ui"].as<bool>();

  opts.enableDeviceLoop = result["device-loop"].as<bool>();
  if (result["no-device-loop"].as<bool>()) {
    opts.enableDeviceLoop = false;
  }

  opts.enableDebug      = result["debug"].as<bool>();
  opts.uiPort           = result["port"].as<int>();

  unsigned rc = result["repeatcounts"].as<unsigned>();
  if (result.count("repeatcount")) rc = result["repeatcount"].as<unsigned>();
  if (result.count("substeps"))    rc = result["substeps"].as<unsigned>();
  opts.repeatcounts     = std::max(1u, rc);

  opts.uiStatePath      = result["ui-state"].as<std::string>();
  opts.loadUIState      = result["load-ui-state"].as<bool>();
  opts.saveUIState      = result["save-ui-state"].as<bool>();

  return opts;
}

} // namespace rf::cli
