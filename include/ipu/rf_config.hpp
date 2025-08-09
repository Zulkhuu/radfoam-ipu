// rf_config.h
#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <geometry/primitives.hpp>  // defines radfoam::geometry::Ray

namespace radfoam::config {
// ============================================================================
// Topology & image geometry 
// ============================================================================

// Full image (pixels)
inline constexpr std::size_t kFullImageWidth  = 640;
inline constexpr std::size_t kFullImageHeight = 480;

// Ray tracer tiles
inline constexpr std::size_t kNumRayTracerTilesX = 32;
inline constexpr std::size_t kNumRayTracerTilesY = 32;
inline constexpr std::size_t kNumRayTracerTiles  = 1024;

// Router tiles per level
inline constexpr std::size_t kChildrenPerRouter = 4;
inline constexpr std::size_t kNumL0RouterTiles = kNumRayTracerTiles / kChildrenPerRouter; // 256
inline constexpr std::size_t kNumL1RouterTiles = kNumL0RouterTiles / kChildrenPerRouter; // 64
inline constexpr std::size_t kNumL2RouterTiles = kNumL1RouterTiles / kChildrenPerRouter; // 16;
inline constexpr std::size_t kNumL3RouterTiles = kNumL2RouterTiles / kChildrenPerRouter; // 4
inline constexpr std::size_t kNumL4RouterTiles = kNumL3RouterTiles / kChildrenPerRouter; // 1

// Ray generator tile
inline constexpr std::uint16_t kRaygenTile = 1470;

// Rays per IO buffer
inline constexpr std::size_t kNumRays = 2400;

// Per-tile image geometry (pixels)
inline constexpr std::size_t kTileImageWidth  = kFullImageWidth  / kNumRayTracerTilesX;
inline constexpr std::size_t kTileImageHeight = kFullImageHeight / kNumRayTracerTilesY;

// Per-tile framebuffer size (RGB bytes)
inline constexpr std::size_t kTileFramebufferSize = kTileImageWidth * kTileImageHeight * 3;

// Sanity checks
static_assert(kNumRayTracerTilesX * kNumRayTracerTilesY == kNumRayTracerTiles,
              "Trace tile grid does not match total tiles.");
static_assert(kTileImageWidth * kTileImageHeight * 3 == kTileFramebufferSize,
              "Framebuffer slice size mismatch.");
static_assert(kTileImageWidth  * kNumRayTracerTilesX == kFullImageWidth,
              "Tile width must evenly divide image width.");
static_assert(kTileImageHeight * kNumRayTracerTilesY == kFullImageHeight,
              "Tile height must evenly divide image height.");


// ============================================================================
// Derived memory layouts for ray buffers
// ============================================================================

// Element sizes
inline constexpr std::size_t kRayBytesPerRay = sizeof(radfoam::geometry::Ray);

// Per-tile ray IO buffer (one buffer: in OR out)
inline constexpr std::size_t kRayIoBytesPerTile  = kNumRays * kRayBytesPerRay;

// All tiles (one buffer kind across all tiles)
inline constexpr std::size_t kRayTracerIoBytesAllTiles = kNumRayTracerTiles * kRayIoBytesPerTile;

// Router (1 parent + 4 children)
inline constexpr std::size_t kRouterSlotsPerTile   = 5; // 4 children + parent
inline constexpr std::size_t kRouterIoBytesPerTile = kRouterSlotsPerTile * kRayIoBytesPerTile;

// Router totals per level
inline constexpr std::size_t kL0RouterIoBytesAllTiles = kNumL0RouterTiles * kRouterIoBytesPerTile;
inline constexpr std::size_t kL1RouterIoBytesAllTiles = kNumL1RouterTiles * kRouterIoBytesPerTile;
inline constexpr std::size_t kL2RouterIoBytesAllTiles = kNumL2RouterTiles * kRouterIoBytesPerTile;
inline constexpr std::size_t kL3RouterIoBytesAllTiles = kNumL3RouterTiles * kRouterIoBytesPerTile;
inline constexpr std::size_t kL4RouterIoBytesAllTiles = kNumL4RouterTiles * kRouterIoBytesPerTile;

// Optional: router tile base offsets (useful for mapping)
inline constexpr std::size_t kL0RouterTileBase = kNumRayTracerTiles;
inline constexpr std::size_t kL1RouterTileBase = kNumRayTracerTiles + kNumL0RouterTiles;
inline constexpr std::size_t kL2RouterTileBase = kNumRayTracerTiles + kNumL0RouterTiles + kNumL1RouterTiles;
inline constexpr std::size_t kL3RouterTileBase = kNumRayTracerTiles + kNumL0RouterTiles + kNumL1RouterTiles + kNumL2RouterTiles;
inline constexpr std::size_t kL4RouterTileBase = kNumRayTracerTiles + kNumL0RouterTiles + kNumL1RouterTiles + kNumL2RouterTiles + kL3RouterTileBase;

inline constexpr std::size_t kFinishedFactor  = 1;
inline constexpr std::size_t kFinishedRayBytesPerTile = kNumRays * kFinishedFactor * sizeof(radfoam::geometry::FinishedRay);
}  // namespace rf