#include <cassert>

// ───── Image Dimensions (Pixels) ──────────────────────────────────
constexpr float kFullImageWidth  = 1280.0f;
constexpr float kFullImageHeight = 768.0f;

// ───── Tiling Configuration ───────────────────────────────────────
// Number of horizontal and vertical tiles used to divide the image
constexpr int kNumTilesX = 32;
constexpr int kNumTilesY = 32;

// ───── Tile Dimensions (Pixels) ───────────────────────────────────
// Dimensions of a framebuffer slice in a single tile 
constexpr int kTileWidth  = 40;
constexpr int kTileHeight = 24;

// ───── Sanity Check ───────────────────────────────────────────────
// Ensure that tiling evenly divides the image
static_assert(  kTileWidth  * kNumTilesX == static_cast<int>(kFullImageWidth),  
                "Tile width doesn't evenly divide image width");
static_assert(  kTileHeight * kNumTilesY == static_cast<int>(kFullImageHeight), 
                "Tile height doesn't evenly divide image height");
