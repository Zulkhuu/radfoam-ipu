#pragma once
#include <cassert>
#include <cstddef>

// -----------------------------------------------------------------------------
// Full image dimensions (pixels)
// -----------------------------------------------------------------------------
constexpr std::size_t kFullImageWidth  = 640;
constexpr std::size_t kFullImageHeight = 480;

// -----------------------------------------------------------------------------
// Tile grid configuration
// -----------------------------------------------------------------------------

// Ray tracers
constexpr std::size_t kNumRayTracerTilesX = 32;  
constexpr std::size_t kNumRayTracerTilesY = 32; 
constexpr std::size_t kNumRayTracerTiles = 1024;  

// Ray router
constexpr std::size_t kNumL0RouterTiles = 256;  // Total number of L0 router tiles
constexpr std::size_t kNumL1RouterTiles = 64;  // Total number of L1 router tiles
constexpr std::size_t kNumL2RouterTiles = 16;  // Total number of L2 router tiles
constexpr std::size_t kNumL3RouterTiles = 4;  // Total number of L3 router tiles

// Ray generator
constexpr unsigned kRaygenTile = 1470;

// Ray
constexpr std::size_t kNumRays = 2080;

// -----------------------------------------------------------------------------
// Tile dimensions (pixels)
// -----------------------------------------------------------------------------
constexpr std::size_t kTileImageWidth  = kFullImageWidth  / kNumRayTracerTilesX;
constexpr std::size_t kTileImageHeight = kFullImageHeight / kNumRayTracerTilesY;

// Per-tile framebuffer size (RGB: 3 bytes per pixel)
constexpr std::size_t kTileFramebufferSize = kTileImageWidth * kTileImageHeight * 3;
// 40 * 24 * 3; for 1280x768
// 20 * 15 * 3; for 640x480


// -----------------------------------------------------------------------------
// Sanity checks
// -----------------------------------------------------------------------------
static_assert(kNumRayTracerTilesX  * kNumRayTracerTilesY == kNumRayTracerTiles,
              "Number of tiles does not match total trace tiles");
static_assert(kTileImageWidth * kTileImageHeight * 3 == kTileFramebufferSize,
              "Full imagebuffer slice size does not match slice width and height");
static_assert(kTileImageWidth  * kNumRayTracerTilesX == kFullImageWidth,
              "Tile width does not evenly divide full image width");
static_assert(kTileImageHeight * kNumRayTracerTilesY == kFullImageHeight,
              "Tile height does not evenly divide full image height");

