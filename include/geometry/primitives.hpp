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

struct NeighborPoint {
  float x, y, z;
  uint32_t gid;
};
static_assert(sizeof(NeighborPoint) == 16, "NeighborPoint size mismatch");

struct Ray {
  uint16_t x, y;
  float t;
  float transmittance;
  float r, g, b;
  uint32_t next_cell;
};
static_assert(sizeof(Ray) == 28, "Ray size mismatch");

struct GenericPoint {
  float x, y, z;        // 3D position
  uint16_t cluster_id;  // Which cluster this point belongs to
  uint16_t local_id;    // Index inside the cluster
};
static_assert(sizeof(GenericPoint) == 16, "GenericPoint size mismatch");

// ── TOOD: Half-precision structures ───────────────────────────────────────────────

}  // namespace radfoam::geometry
