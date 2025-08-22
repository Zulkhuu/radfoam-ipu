#include <poplar/Vertex.hpp>

#include <glm/mat4x4.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <geometry/primitives.hpp>
#include <ipu/rf_config.hpp>
#include <ipudef.h>
#include <ipu_builtins.h>
#include <ipu_vector_math>
#include <ipu_memory_intrinsics>

// __builtin_assume(kNumRays <= 4096);

using namespace radfoam::config;
using radfoam::geometry::LocalPoint;
using radfoam::geometry::GenericPoint;
using radfoam::geometry::FinishedRay;

struct alignas(4) Ray {
  uint16_t x, y;
  half t, transmittance;
  float r, g, b;
  uint16_t next_cluster; 
  uint16_t next_local;
};
struct alignas(4) SeedRay {
  uint16_t x, y;
  uint16_t ncols, nrows;
  float pad1, pad2, pad3;
  uint16_t next_cluster; 
  uint16_t next_local;
};
struct alignas(4) FinishedPixel {
  uint8_t r, g, b, a;
  float t;
};

inline constexpr uint16_t INVALID_RAY_ID = 0xFFFF;
inline constexpr uint16_t FINISHED_RAY_ID = 0xFFFE;
inline constexpr uint16_t INVALID_SEEN_ID = 0xFFFD;
inline constexpr unsigned NW = 6;

class RayTracer : public poplar::MultiVertex {
public:
  // Inputs
  poplar::Input<poplar::Vector<float>> view_matrix;
  poplar::Input<poplar::Vector<float>> projection_matrix;

  poplar::Input<poplar::Vector<uint8_t>>            local_pts;
  poplar::Input<poplar::Vector<uint8_t>>            neighbor_pts;
  poplar::Input<poplar::Vector<unsigned short>>     adjacency;
  poplar::Input<unsigned short>                     tile_id;
  poplar::Input<poplar::Vector<uint8_t>>            raysIn;
  poplar::Input<poplar::Vector<uint8_t>>            seedRay;

  // Outputs
  poplar::InOut<poplar::Vector<uint8_t>>            raysOut;
  poplar::InOut<poplar::Vector<uint8_t>>            finishedRays;
  // Debug outputs
  poplar::Output<float>                             result_float;
  poplar::Output<unsigned short>                    result_u16;
  poplar::Output<poplar::Vector<uint8_t>>           debugBytes;

  // InOut
  poplar::InOut<poplar::Vector<uint8_t>>            framebuffer;
  poplar::InOut<unsigned>                           exec_count;
  poplar::InOut<unsigned>                           finishedWriteOffset;

  poplar::InOut<poplar::Vector<unsigned>>           sharedCounts;
  poplar::InOut<poplar::Vector<unsigned>>           sharedOffsets;
  poplar::InOut<poplar::Vector<unsigned>>           readyFlags;

  // aliasing guards (unchanged)
  [[poplar::constraint("elem(*raysIn)!=elem(*raysOut)")]]
  [[poplar::constraint("elem(*framebuffer)!=elem(*raysIn)")]]
  [[poplar::constraint("elem(*framebuffer)!=elem(*raysOut)")]]
  [[poplar::constraint("elem(*readyFlags)!=elem(*raysIn)")]]
  [[poplar::constraint("elem(*readyFlags)!=elem(*raysOut)")]]
  [[poplar::constraint("elem(*readyFlags)!=elem(*framebuffer)")]]
  [[poplar::constraint("elem(*readyFlags)!=elem(*view_matrix)")]]
  [[poplar::constraint("elem(*readyFlags)!=elem(*projection_matrix)")]]
  [[poplar::constraint("elem(*readyFlags)!=elem(*adjacency)")]]
  [[poplar::constraint("elem(*readyFlags)!=elem(*local_pts)")]]
  [[poplar::constraint("elem(*readyFlags)!=elem(*neighbor_pts)")]]
  [[poplar::constraint("elem(*readyFlags)!=elem(*result_float)")]]
  [[poplar::constraint("elem(*readyFlags)!=elem(*result_u16)")]]
  [[poplar::constraint("elem(*readyFlags)!=elem(*exec_count)")]]
  
  bool compute(unsigned workerId) {
    const unsigned NW = poplar::MultiVertex::numWorkers();

    const unsigned C         = raysOut.size() / sizeof(Ray);
    const unsigned FR_CAP    = finishedRays.size() / sizeof(FinishedRay);
    (void)FR_CAP;
    const uint16_t nLocalPts = local_pts.size() / sizeof(LocalPoint);
    const uint16_t nNbrPts   = neighbor_pts.size() / sizeof(GenericPoint);
    (void)nNbrPts;

    glm::mat4 invView = glm::make_mat4(view_matrix.data());
    glm::mat4 invProj = glm::make_mat4(projection_matrix.data());
    const glm::vec3 rayOrigin = glm::vec3(invView[3]);
    const uint16_t  myTile    = *tile_id;

    // barrier(/*phase*/1, workerId);

    // If a seed exists, generat columns.
    const SeedRay* seed = readStructAt<SeedRay>(seedRay, 0);
    const bool haveSeed = (seed->next_cluster != INVALID_RAY_ID);
    if (haveSeed) {
      const uint16_t x0        = seed->x;
      const uint16_t seedLocal = static_cast<uint16_t>(seed->next_local);

      unsigned nCols = seed->ncols;
      const unsigned totalRays = nCols * kFullImageHeight;
      const unsigned writeCount = (totalRays < C) ? totalRays : C;

      for (unsigned i = workerId; i < writeCount; i += NW) {
        const unsigned  col = i / kFullImageHeight;          // 0..(nCols-1)
        const unsigned  row = i % kFullImageHeight;          // 0..H-1
        const uint16_t  x   = static_cast<uint16_t>((x0 + col) % kFullImageWidth);
        const uint16_t  y   = static_cast<uint16_t>(row);

        glm::vec3 color(0.f); 
        float trans = 1.f, t0 = 0.f;
        const glm::vec3 rayDir = computeRayDir(x, y, invProj, invView);

        int current = seedLocal;
        Ray* out = reinterpret_cast<Ray*>(raysOut.data() + i*sizeof(Ray));

        if (current < 0) {
          writeFinishedOut(out, x, y, color, t0, trans, tileOfXY(x,y));
          continue;
        }

        while (true) {
          const LocalPoint* cur = readStructAt<LocalPoint>(local_pts, current);
          float closestT;
          const int next = findNextAndT(current, t0, closestT, rayOrigin, rayDir, nLocalPts);

          accumulateColor(cur, closestT, t0, trans, color);

          if (trans < 0.01f || next == -1 || next >= (int)nLocalPts) {
            if (trans < 0.01f || next == -1) {
              writeFinishedOut(out, x, y, color, t0, trans, tileOfXY(x,y));
            } else {
              const GenericPoint* nb = readStructAt<GenericPoint>(neighbor_pts, static_cast<std::size_t>(next - nLocalPts));
              writeNeighborOut(out, x, y, color, t0, trans, nb);
            }
            break;
          }
          current = next;
        }
      }

      // Invalidate any unused tail (head-packed list)
      for (unsigned i = writeCount + workerId; i < C; i += NW) {
        Ray* r = reinterpret_cast<Ray*>(raysOut.data() + i*sizeof(Ray));
        if (r->next_cluster == INVALID_RAY_ID) break;
        r->next_cluster = INVALID_RAY_ID;
      }

      for (uint16_t idx = workerId; idx < C; idx += NW) {
        const Ray* in = readStructAt<Ray>(raysIn, idx);
        if (in->next_cluster == INVALID_RAY_ID) break;
        if (in->next_local == FINISHED_RAY_ID) {
          const uint16_t y = unpackYCoord(in->y);
          if (in->next_cluster != myTile) {
            Ray* out = reinterpret_cast<Ray*>(raysOut.data() + idx*sizeof(Ray));
            *out = *in; //spillover out collides with generated columns, need to handle
            continue;
          }
          uint16_t lx, ly;
          localXY(in->x, y, myTile, lx, ly);
          if (lx >= kTileImageWidth || ly >= kTileImageHeight) return false;

          FinishedPixel* p = fbAt(lx, ly);
          p->r = clampToU8(in->r);
          p->g = clampToU8(in->g);
          p->b = clampToU8(in->b);
          p->a = 255;
          p->t = __builtin_ipu_f16tof32(in->t);
        }
      }
      if (workerId == 0) { *exec_count = *exec_count + 1; }
      return true;
    }
  

    // Normal path: march your stripe and write results back to same index.
    unsigned lastSeenIdx = INVALID_SEEN_ID; 
    marchStripe(workerId, NW, C, myTile, nLocalPts, invProj, invView, rayOrigin, lastSeenIdx);

    // Invalidate tail so downstream routers can stop early
    invalidateTail(workerId, NW, C, lastSeenIdx);

    if (workerId == 0) { 
      *exec_count = *exec_count + 1; 
    }
    return true;
  }

private:
  // ---------- Small, always-inlined helpers (kept trivial to avoid perf regressions) ----------

  template <typename T>
  [[gnu::always_inline]] inline const T*
  readStructAt(const poplar::Input<poplar::Vector<uint8_t>>& buf, std::size_t idx) const {
    const std::size_t stride = sizeof(T);
    const uint8_t* base = buf.data() + idx * stride;
    return reinterpret_cast<const T*>(base);
  }

  [[gnu::always_inline]] inline void
  barrier(unsigned phase, unsigned workerId) {
    volatile unsigned* flags = readyFlags.data();
    asm volatile("" ::: "memory");
    while (true) {
      flags[workerId] = phase;
      bool all = true;
      for (unsigned i = 0; i < NW; ++i) if (flags[i] < phase) { all = false; break; }
      if (all) break;
    }
    asm volatile("" ::: "memory");
  }

  [[gnu::always_inline]] static inline uint16_t
  tileOfXY(uint16_t x, uint16_t y) {
    const uint16_t tx = x / kTileImageWidth;
    const uint16_t ty = y / kTileImageHeight;
    return static_cast<uint16_t>(ty * kNumRayTracerTilesX + tx);
  }

  [[gnu::always_inline]] static inline void
  localXY(uint16_t x, uint16_t y, uint16_t tile, uint16_t &lx, uint16_t &ly) {
    const uint16_t tileX = tile % kNumRayTracerTilesX;
    const uint16_t tileY = tile / kNumRayTracerTilesX;
    lx = static_cast<uint16_t>(x - tileX * kTileImageWidth);
    ly = static_cast<uint16_t>(y - tileY * kTileImageHeight);
  }

  [[gnu::always_inline]] inline FinishedPixel*
  fbAt(uint16_t lx, uint16_t ly) const {
    auto* base = const_cast<uint8_t*>(framebuffer.data());
    auto* fb   = reinterpret_cast<FinishedPixel*>(base);
    // FinishedPixel* fb = reinterpret_cast<FinishedPixel*>(framebuffer.data());
    return &fb[static_cast<std::size_t>(ly) * kTileImageWidth + lx];
  }

  [[gnu::always_inline]] static inline uint16_t unpackYCoord(uint16_t packed) { return packed & 0x03FF; }
  [[gnu::always_inline]] static inline uint8_t  unpackYInfo (uint16_t packed) { return (packed >> 10) & 0x3F; }
  [[gnu::always_inline]] static inline uint16_t packY(uint16_t y, uint8_t info) { return ((info & 0x3F) << 10) | (y & 0x03FF); }

  [[gnu::always_inline]] static inline uint8_t clampToU8(float v01) {
    return static_cast<uint8_t>(__builtin_ipu_max(0.0f, __builtin_ipu_min(255.0f, v01 * 255.0f)));
  }

  [[gnu::always_inline]] static inline glm::vec3
  computeRayDir(uint16_t x, uint16_t y, glm::mat4 &invProj, glm::mat4 &invView) {
    const float ndcX = (2.0f * x) / kFullImageWidth  - 1.0f;
    const float ndcY = 1.0f   - (2.0f * y) / kFullImageHeight;
    glm::vec4 clipRay(ndcX, ndcY, -1.0f, 1.0f);
    glm::vec4 eyeRay = invProj * clipRay;
    eyeRay.z = -1.0f; eyeRay.w = 0.0f;
    return glm::normalize(glm::vec3(invView * eyeRay));
  }

  [[gnu::always_inline]] inline bool
  writeFinishedToFramebuffer(const Ray* in, uint16_t myTile) {
    const uint16_t y = unpackYCoord(in->y);
    if (in->next_cluster != myTile) return false;
    uint16_t lx, ly;
    localXY(in->x, y, myTile, lx, ly);
    if (lx >= kTileImageWidth || ly >= kTileImageHeight) return false;

    FinishedPixel* p = fbAt(lx, ly);
    p->r = clampToU8(in->r);
    p->g = clampToU8(in->g);
    p->b = clampToU8(in->b);
    p->a = 255;
    p->t = __builtin_ipu_f16tof32(in->t);
    return true;
  }

  [[gnu::always_inline]] static inline void
  writeFinishedOut(Ray* out, uint16_t x, uint16_t y, const glm::vec3& color,
                   float t0, float trans, uint16_t destTile) {
    out->x = x;
    out->y = packY(y, /*info*/1);
    out->r = color.x; out->g = color.y; out->b = color.z;
    out->t = __builtin_ipu_f32tof16(t0);
    out->transmittance = __builtin_ipu_f32tof16(trans);
    out->next_cluster  = destTile;
    out->next_local    = FINISHED_RAY_ID;
  }

  [[gnu::always_inline]] static inline void
  writeNeighborOut(Ray* out, uint16_t x, uint16_t y, const glm::vec3& color,
                   float t0, float trans, const GenericPoint* nb) {
    out->x = x;
    out->y = packY(y, /*info*/1);
    out->r = color.x; out->g = color.y; out->b = color.z;
    out->t = __builtin_ipu_f32tof16(t0);
    out->transmittance = __builtin_ipu_f32tof16(trans);
    out->next_cluster  = nb->cluster_id;
    out->next_local    = nb->local_id;
  }

  [[gnu::always_inline]] inline void
  accumulateColor(const LocalPoint* cur, float closestT, float &t0, float &transmittance, glm::vec3 &color) const {
    const float delta = closestT - t0;
    const float alpha = 1.f - __builtin_ipu_exp(-cur->density * delta);
    const float ta    = transmittance * alpha;
    color.x += ta * (cur->r / 255.f);
    color.y += ta * (cur->g / 255.f);
    color.z += ta * (cur->b / 255.f);
    transmittance *= (1.f - alpha);
    t0 = __builtin_ipu_max(t0, closestT);
  }

  [[gnu::always_inline]] inline int
  findNextAndT(int current, float t0, float &closestT,
               const glm::vec3& rayOrigin, const glm::vec3& rayDir,
               uint16_t nLocalPts_) const {
    const LocalPoint* cur = readStructAt<LocalPoint>(local_pts, current);
    const glm::vec3   p0(cur->x, cur->y, cur->z);
    const uint16_t    adjStart = (current == 0) ? 0 : readStructAt<LocalPoint>(local_pts, current-1)->adj_end;
    const uint16_t    adjEnd   = cur->adj_end;

    closestT = std::numeric_limits<float>::max();
    int next = -1;

    for (uint16_t j = adjStart; j < adjEnd; ++j) {
      const uint16_t nbrIdx = adjacency[j];
      glm::vec3 p1;
      if (nbrIdx < nLocalPts_) {
        const LocalPoint* nb = readStructAt<LocalPoint>(local_pts, nbrIdx);
        p1 = glm::vec3(nb->x, nb->y, nb->z);
      } else {
        const GenericPoint* nb = readStructAt<GenericPoint>(neighbor_pts, nbrIdx - nLocalPts_);
        p1 = glm::vec3(nb->x, nb->y, nb->z);
      }
      const glm::vec3 faceNormal = p1 - p0;
      const glm::vec3 faceOrigin = 0.5f * (p1 + p0);
      const float dn = glm::dot(faceNormal, rayDir);
      if (dn <= 0.f) continue;
      const float t = glm::dot(faceOrigin - rayOrigin, faceNormal) / dn;
      if (t > 0.f && t < closestT) { closestT = t; next = nbrIdx; }
    }
    return next;
  }

  // ---------- Phase helpers used by compute() ----------

  [[gnu::always_inline]] inline void
  tryCaptureSeed(unsigned workerId, const Ray* in, uint16_t myTile) {
    // capture one seed per worker (first seen)
    if (in->next_cluster == myTile && unpackYInfo(in->y) == 0 && sharedCounts[workerId] == 0) {
      sharedCounts[workerId]     = 1;
      sharedOffsets[workerId]    = in->x;          // seed X
      sharedOffsets[workerId+NW] = in->next_local; // seed starting cell
    }
  }

  inline void passScanAndConsume(unsigned workerId, unsigned NW_, unsigned C,
                                 uint16_t myTile, uint16_t nLocalPts,
                                 glm::mat4 &invProj, glm::mat4 &invView,
                                 const glm::vec3 &rayOrigin) {
    (void)nLocalPts; (void)invProj; (void)invView; (void)rayOrigin; // (seed pass only consumes finished)
    for (uint16_t idx = workerId; idx < C; idx += NW_) {
      const Ray* in = readStructAt<Ray>(raysIn, idx);
      if (in->next_cluster == INVALID_RAY_ID) break;

      tryCaptureSeed(workerId, in, myTile);

      if (in->next_local == FINISHED_RAY_ID) {
        if (writeFinishedToFramebuffer(in, myTile)) {
          // ok
        }
        // mirror to raysOut with FINISHED_RAY_ID so routers skip it
        Ray* out = reinterpret_cast<Ray*>(raysOut.data() + idx*sizeof(Ray));
        *out = *in;
        out->next_cluster = FINISHED_RAY_ID;
      }
    }
  }

  inline void publishFirstSeed(unsigned NW_) {
    unsigned haveSeed = 0, seedX = 0, seedLocal = 0;
    for (unsigned w = 0; w < NW_; ++w) {
      if (sharedCounts[w]) { seedX = sharedOffsets[w]; seedLocal = sharedOffsets[w+NW_]; haveSeed = 1; break; }
    }
    sharedCounts[NW_]      = haveSeed;
    sharedOffsets[2*NW_]   = seedX;
    sharedOffsets[2*NW_+1] = seedLocal;
  }

  inline void generateColumnsFromSeed(unsigned workerId, unsigned NW_, unsigned C,
                                      uint16_t myTile, uint16_t nLocalPts,
                                      glm::mat4 &invProj, glm::mat4 &invView,
                                      const glm::vec3 &rayOrigin) {
    const uint16_t x0        = static_cast<uint16_t>(sharedOffsets[2*NW_]   & 0xFFFF);
    const uint16_t seedLocal = static_cast<uint16_t>(sharedOffsets[2*NW_+1] & 0xFFFF);

    constexpr unsigned nCols = 3;
    const unsigned totalRays = nCols * kFullImageHeight;
    const unsigned writeCount = (totalRays < C) ? totalRays : C;

    for (unsigned i = workerId; i < writeCount; i += NW_) {
      const unsigned  col = i / kFullImageHeight;
      const unsigned  row = i % kFullImageHeight;
      const uint16_t  x   = static_cast<uint16_t>((x0 + col) % kFullImageWidth);
      const uint16_t  y   = static_cast<uint16_t>(row);

      glm::vec3 color(0.f); float trans = 1.f; float t0 = 0.f;
      const glm::vec3 rayDir = computeRayDir(x, y, invProj, invView);

      int current = seedLocal;
      Ray* out = reinterpret_cast<Ray*>(raysOut.data() + i*sizeof(Ray));

      if (current < 0) { writeFinishedOut(out, x, y, color, t0, trans, tileOfXY(x,y)); continue; }

      while (true) {
        const LocalPoint* cur = readStructAt<LocalPoint>(local_pts, current);
        float closestT;
        const int next = findNextAndT(current, t0, closestT, rayOrigin, rayDir, nLocalPts);

        accumulateColor(cur, closestT, t0, trans, color);

        if (trans < 0.01f || next == -1 || next >= nLocalPts) {
          if (trans < 0.01f || next == -1) {
            writeFinishedOut(out, x, y, color, t0, trans, tileOfXY(x,y));
          } else {
            const GenericPoint* nb = readStructAt<GenericPoint>(neighbor_pts, next - nLocalPts);
            writeNeighborOut(out, x, y, color, t0, trans, nb);
          }
          break;
        }
        current = next;
      }
    }

    // invalidate unused tail
    for (unsigned i = writeCount + workerId; i < C; i += NW_) {
      Ray* r = reinterpret_cast<Ray*>(raysOut.data() + i*sizeof(Ray));
      if (r->next_cluster == INVALID_RAY_ID) break;
      r->next_cluster = INVALID_RAY_ID;
    }
  }

  inline void marchStripe(unsigned workerId, unsigned NW_, unsigned C,
                          uint16_t myTile, uint16_t nLocalPts,
                          glm::mat4 &invProj, glm::mat4 &invView,
                          const glm::vec3 &rayOrigin,
                          unsigned &lastSeenIdx_out) {
    unsigned lastSeenIdx = 0xFFFD;

    for (uint16_t idx = workerId; idx < C; idx += NW_) {
      const Ray* in = readStructAt<Ray>(raysIn, idx);
      if (in->next_cluster == INVALID_RAY_ID) break;
      lastSeenIdx = idx;

      const uint16_t y = unpackYCoord(in->y);
      Ray* out = reinterpret_cast<Ray*>(raysOut.data() + idx*sizeof(Ray));

      // Not my tile? forward as-is.
      if (in->next_cluster != myTile) { *out = *in; continue; }

      // Already finished? Write to framebuffer if mine, then mark FINISHED in stream.
      if (in->next_local == FINISHED_RAY_ID) {
        if (writeFinishedToFramebuffer(in, myTile)) { /*count if needed*/ }
        *out = *in; out->next_cluster = FINISHED_RAY_ID;
        continue;
      }

      // March
      glm::vec3 color(in->r, in->g, in->b);
      float trans = __builtin_ipu_f16tof32(in->transmittance);
      float t0    = __builtin_ipu_f16tof32(in->t);
      int   current = in->next_local;

      if (current < 0 || current >= nLocalPts) { *out = *in; out->next_cluster = FINISHED_RAY_ID; continue; }

      const glm::vec3 rayDir = computeRayDir(in->x, y, invProj, invView);

      while (true) {
        const LocalPoint* cur = readStructAt<LocalPoint>(local_pts, current);
        float closestT;
        const int next = findNextAndT(current, t0, closestT, rayOrigin, rayDir, nLocalPts);

        accumulateColor(cur, closestT, t0, trans, color);

        if (trans < 0.01f || next == -1 || next >= nLocalPts) {
          if (trans < 0.01f || next == -1) {
            writeFinishedOut(out, in->x, y, color, t0, trans, tileOfXY(in->x, y));
            *result_u16   = FINISHED_RAY_ID;
            *result_float = color.x;
          } else {
            const GenericPoint* nb = readStructAt<GenericPoint>(neighbor_pts, next - nLocalPts);
            writeNeighborOut(out, in->x, y, color, t0, trans, nb);
            *result_u16   = nb->cluster_id;
            *result_float = color.x;
          }
          break;
        }
        current = next;
      }
    }

    lastSeenIdx_out = lastSeenIdx;
  }

  inline void invalidateTail(unsigned workerId, unsigned NW_, unsigned C, unsigned lastSeenIdx) {
    const unsigned startTail = (lastSeenIdx == INVALID_SEEN_ID) ? workerId : (static_cast<unsigned>(lastSeenIdx) + NW_);
    for (unsigned idx = startTail; idx < C; idx += NW_) {
      Ray* ray = reinterpret_cast<Ray*>(raysOut.data() + idx*sizeof(Ray));
      if (ray->next_cluster == INVALID_RAY_ID) break;
      ray->next_cluster = INVALID_RAY_ID;
    }
  }

  // Optional: reserved for future batch writes to finishedRays ring buffer
  [[gnu::always_inline]] inline void computeWriteOffsets() {
    // Exclusive scan over workers to reserve a contiguous slice (kept for future use).
    unsigned head = *finishedWriteOffset;
    unsigned scan = 0;
    for (unsigned w = 0; w < NW; ++w) {
      sharedOffsets[w] = head + scan; // start for worker w
      scan += sharedCounts[w];
    }
    *finishedWriteOffset = head + scan;
  }
};
