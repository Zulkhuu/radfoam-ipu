#pragma once
#include <cassert>
#include <cstddef>

// -----------------------------------------------------------------------------
// Full image dimensions (pixels)
// -----------------------------------------------------------------------------
constexpr std::size_t kFullImageWidth  = 1280;
constexpr std::size_t kFullImageHeight = 768;

// -----------------------------------------------------------------------------
// Tile grid configuration
// -----------------------------------------------------------------------------
constexpr std::size_t kNumTilesX = 32;  // Horizontal tile count
constexpr std::size_t kNumTilesY = 32;  // Vertical tile count
constexpr std::size_t kNumTraceTiles = 1024;  // Total number of tiles

// -----------------------------------------------------------------------------
// Tile dimensions (pixels)
// -----------------------------------------------------------------------------
constexpr std::size_t kTileWidth  = 40; // kFullImageWidth  / kNumTilesX;
constexpr std::size_t kTileHeight = 24; // kFullImageHeight / kNumTilesY;

// Per-tile framebuffer size (RGB: 3 bytes per pixel)
constexpr std::size_t kTileFramebufferSize = 960; // kTileWidth * kTileHeight * 3;

// -----------------------------------------------------------------------------
// Sanity checks
// -----------------------------------------------------------------------------
static_assert(kNumTilesX  * kNumTilesY == kNumTraceTiles,
              "Number of tiles does not match total trace tiles");
static_assert(kTileWidth  * kNumTilesX == kFullImageWidth,
              "Tile width does not evenly divide full image width");
static_assert(kTileHeight * kNumTilesY == kFullImageHeight,
              "Tile height does not evenly divide full image height");
