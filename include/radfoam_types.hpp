#pragma once

#include <cstdint>

// ─── Full Precision Types ──────────────────────────────────────────

// Local primal point
struct LocalPoint {
    float   x, y, z;
    uint8_t r, g, b, _pad;
    float   density;
    uint32_t adj_end;
};
static_assert(sizeof(LocalPoint) == 24, "Unexpected LocalPoint size");

// Neighbor primal point 
struct NeighborPoint {
    float x, y, z;
    uint32_t gid;
};
static_assert(sizeof(NeighborPoint) == 16, "Unexpected NeighborPoint size");

// A 28-byte packed ray for IPU processing
struct Ray {
    uint16_t x, y;
    float t;
    float transmittance;
    float r, g, b;
    uint32_t next_cell;
};
static_assert(sizeof(Ray) == 28, "Ray must be 28B");
