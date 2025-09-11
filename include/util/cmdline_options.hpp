#pragma once
#include <string>

namespace rf::cli {

struct CliOptions {
  std::string inputFile = "./data/garden.h5";
  int  nRuns            = 0;     // 0 == infinite
  bool enableUI         = true;
  bool enableDeviceLoop = true;  // default ON
  bool enableDebug      = false;
  int  uiPort           = 5000;

  // UI state persistence
  std::string uiStatePath = "ui_state.txt";
  bool loadUIState = false;
  bool saveUIState = false;

  // Device repeat count
  unsigned repeatcounts = 20;    // default 20
};

// Parse argv into CliOptions. Exits after printing help if --help is given.
CliOptions parseOptions(int argc, char** argv);

} // namespace rf::cli
