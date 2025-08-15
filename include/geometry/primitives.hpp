#pragma once
#include <cstdint>

namespace radfoam::geometry {

// ── Full-precision structures ───────────────────────────────────────────────
struct LocalPoint {
  float   x, y, z;
  uint8_t r, g, b, _pad;
  float   density;
  uint32_t adj_end;
};
static_assert(sizeof(LocalPoint) == 24, "LocalPoint size mismatch");

struct GenericPoint {
  float x, y, z;
  uint16_t cluster_id;
  uint16_t local_id;
};
static_assert(sizeof(GenericPoint) == 16, "GenericPoint size mismatch");

struct Ray {
  uint16_t x, y;
  uint16_t t, transmittance; // uint16_t is placeholder for float16
  float r, g, b;
  uint16_t next_cluster; 
  uint16_t next_local;
};
static_assert(sizeof(Ray) == 24, "Ray size mismatch");

struct FinishedRay {
  uint16_t x, y;
  uint8_t r, g, b, a;
  float t;
};
static_assert(sizeof(FinishedRay) == 12, "Finished Ray size mismatch");

struct FinishedPixel {
  uint8_t r, g, b, a;
  float t;
};
static_assert(sizeof(FinishedPixel) == 8, "Finished Ray size mismatch");
// ── TOOD: Half-precision structures ───────────────────────────────────────────────

}  // namespace radfoam::geometry
