cmake_minimum_required(VERSION 3.14)
project(radfoam_ipu LANGUAGES CXX)

# ───── Compiler Settings ───────────────────────────────────────────
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ───── Poplar Setup ────────────────────────────────────────────────
include(utils.cmake)                    # Poplar helper macros
check_for_submodules()
get_poplar_version()
message(STATUS "Poplar version ${POPLAR_VERSION_MAJOR}.${POPLAR_VERSION_MINOR}.${POPLAR_VERSION_PATCH}")

# ───── Dependencies ────────────────────────────────────────────────
find_package(Boost REQUIRED COMPONENTS program_options unit_test_framework)
find_package(OpenCV REQUIRED)
find_package(spdlog REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C HL)

include(FetchContent)
FetchContent_Declare(
  HighFive
  GIT_REPOSITORY https://github.com/BlueBrain/HighFive.git
  GIT_TAG        v2.10.0
)
FetchContent_MakeAvailable(HighFive)

FetchContent_Declare(
  cxxopts
  GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
  GIT_TAG        v3.2.0  # Use latest stable tag or specific version you need
)
FetchContent_MakeAvailable(cxxopts)

# External Libraries:
add_subdirectory(external/packetcomms)
add_subdirectory(external/videolib)

include_directories(
  ${PACKETCOMMS_INCLUDES}
  ${VIDEOLIB_INCLUDES}
)


# ───── Main Executable ─────────────────────────────────────────────
add_executable(main src/main.cpp)

# ───── Target Configuration ────────────────────────────────────────
target_include_directories(main
  PRIVATE
    ${IPU_INCLUDES}                         # set by utils.cmake
    ${OpenCV_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/external/glm
)

target_link_libraries(main
  PRIVATE
    HighFive
    ${HDF5_LIBRARIES}
    ${OpenCV_LIBS}
    ${PACKETCOMMS_LIBRARIES}
    ${VIDEOLIB_LIBRARIES}
    cxxopts
    spdlog::spdlog
    Boost::program_options
    Boost::unit_test_framework
    poplar popops poputil poprand pvti      # Link Poplar libraries
)

target_compile_definitions(main
  PRIVATE POPC_PREFIX="${PROJECT_SOURCE_DIR}"
)

target_compile_features(main PRIVATE cxx_std_20)
