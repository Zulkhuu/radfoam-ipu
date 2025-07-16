cmake_minimum_required(VERSION 3.1)
project(hdf5_ipu_add LANGUAGES CXX)

# ───── compiler flags ──────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Poplar helper macros (from the Graphcore SDK examples)
include(utils.cmake)
check_for_submodules()
get_poplar_version()
message(STATUS "Poplar version ${POPLAR_VERSION_MAJOR}.${POPLAR_VERSION_MINOR}.${POPLAR_VERSION_PATCH}")

find_package(Boost REQUIRED COMPONENTS program_options)
find_package(Boost COMPONENTS unit_test_framework REQUIRED)
find_package(glm REQUIRED)


# ───── dependencies ────────────────────────────────────────────────
find_package(HDF5 COMPONENTS C HL REQUIRED)           # host-side HDF5
include(FetchContent)                                 # header-only HighFive
FetchContent_Declare(
  HighFive
  GIT_REPOSITORY https://github.com/BlueBrain/HighFive.git
  GIT_TAG        v2.10.0
)
FetchContent_MakeAvailable(HighFive)

include_directories(
  ${Boost_INCLUDE_DIRS}
  ${HDF5_INCLUDE_DIRS}
  ${IPU_INCLUDES}                         # set by utils.cmake
  ${HighFive_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/include
  external/glm
)

# ───── executable ──────────────────────────────────────────────────
add_executable(main main.cpp)

target_link_libraries(main
  PUBLIC
    HighFive
    ${HDF5_LIBRARIES}
    -lpoplar -lpopops -lpoputil -lpoprand     # core Poplar libs
)
target_compile_features(main PRIVATE cxx_std_17)

# Make the absolute path available to popc inside IPU code
add_compile_definitions(POPC_PREFIX="${PROJECT_SOURCE_DIR}")
