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
struct alignas(4) FinishedPixel {
  uint8_t r, g, b, a;
  float t;
};

inline constexpr uint16_t INVALID_RAY_ID = 0xFFFF;
inline constexpr uint16_t FINISHED_RAY_ID = 0xFFFE;
inline constexpr unsigned NW = 6;

class RayTrace : public poplar::MultiVertex {
public:
  // Inputs
  poplar::Input<poplar::Vector<float>> view_matrix;
  poplar::Input<poplar::Vector<float>> projection_matrix;

  poplar::Input<poplar::Vector<uint8_t>> local_pts;
  poplar::Input<poplar::Vector<uint8_t>> neighbor_pts;
  poplar::Input<poplar::Vector<unsigned short>> adjacency;
  poplar::Input<unsigned short> tile_id;
  poplar::Input<poplar::Vector<uint8_t>> raysIn;

  // Outputs
  poplar::InOut<poplar::Vector<uint8_t>> raysOut;
  poplar::InOut<poplar::Vector<uint8_t>> finishedRays;
  // Debug outputs
  poplar::Output<float> result_float;
  poplar::Output<unsigned short> result_u16;
  poplar::Output<poplar::Vector<uint8_t>> debugBytes;

  // InOut
  poplar::InOut<poplar::Vector<uint8_t>> framebuffer;
  poplar::InOut<unsigned>       exec_count;          
  poplar::InOut<unsigned>       finishedWriteOffset; 

  poplar::InOut<poplar::Vector<unsigned>> sharedCounts;
  poplar::InOut<poplar::Vector<unsigned>> sharedOffsets;
  poplar::InOut<poplar::Vector<unsigned>> readyFlags;

  // poplar::Input<unsigned> kSubsteps;

  // aliasing guards
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
    const unsigned kSubSteps = 6;
    const unsigned C  = raysOut.size() / sizeof(Ray);
    const unsigned FR_CAP = finishedRays.size() / sizeof(FinishedRay);
    const uint16_t nLocalPts = local_pts.size() / sizeof(LocalPoint);
    const uint16_t nNbrPts   = neighbor_pts.size() / sizeof(GenericPoint);
    const unsigned adjSize   = adjacency.size(); 
    const uint16_t tileX = tile_id % kNumRayTracerTilesX;
    const uint16_t tileY = tile_id / kNumRayTracerTilesX;

    // helpers for tile ownership and local coords
    auto tileOfXY = [&](uint16_t x10, uint16_t y)->uint16_t {
      const uint16_t tx = x10 / kTileImageWidth;
      const uint16_t ty = y   / kTileImageHeight;
      return static_cast<uint16_t>(ty * kNumRayTracerTilesX + tx);
    };
    auto localXY = [&](uint16_t x, uint16_t y, uint16_t tile, uint16_t &lx, uint16_t &ly){
      lx = static_cast<uint16_t>(x - tileX * kTileImageWidth);
      ly = static_cast<uint16_t>(y   - tileY * kTileImageHeight);
    };
    auto fbAt = [&](uint16_t lx, uint16_t ly)->FinishedPixel* {
      FinishedPixel* fb = reinterpret_cast<FinishedPixel*>(framebuffer.data());
      return &fb[static_cast<std::size_t>(ly) * kTileImageWidth + lx];
    };

    // Setup matrices
    glm::mat4 invView = glm::make_mat4(view_matrix.data());
    glm::mat4 invProj = glm::make_mat4(projection_matrix.data());
    glm::vec3 rayOrigin = glm::vec3(invView[3]);
    const uint16_t myTile = *tile_id;

    // if (workerId == 0) {
    //   // Clear global mailboxes
    //   sharedCounts[NW]      = 0;
    //   sharedOffsets[2*NW]   = 0;
    //   sharedOffsets[2*NW+1] = 0;
    // }
    
    // sharedCounts[workerId] = 0;
    // barrier(/*phase*/1, workerId);

    // for (uint16_t idx = workerId; idx < C; idx += NW) {
    //   const Ray* in = readStructAt<Ray>(raysIn, idx);
    //   if (in->next_cluster == INVALID_RAY_ID) break;

    //   const uint16_t y_coord = unpackYCoord(in->y);
    //   const uint8_t  y_info  = unpackYInfo(in->y);

    //   // 1a) Capture one seed per worker (first seen)
    //   if (in->next_cluster == myTile && y_info == 0 && sharedCounts[workerId] == 0) {
    //     sharedCounts[workerId]      = 1;
    //     sharedOffsets[workerId]     = in->x;           // seed X
    //     sharedOffsets[workerId+NW]  = in->next_local;  // seed starting cell
    //     // do NOT early return; still handle finished rays in this stripe
    //   }

    //   // 1b) Consume finished rays (write to FB and mark finished)
    //   if (in->next_local == FINISHED_RAY_ID) {
    //     // Only write FB if it belongs to me (route says which tile owns the pixel)
    //     if (in->next_cluster == myTile) {
    //       uint16_t lx, ly;
    //       localXY(in->x, y_coord, myTile, lx, ly);
    //       if (lx < kTileImageWidth && ly < kTileImageHeight) {
    //         FinishedPixel* p = fbAt(lx, ly);
    //         p->r = clampToU8(in->r);
    //         p->g = clampToU8(in->g);
    //         p->b = clampToU8(in->b);
    //         p->a = 255;
    //         p->t = __builtin_ipu_f16tof32(in->t);
    //       }
    //     }
    //     // Mirror back to raysOut so the routers skip it downstream
    //     Ray* out = reinterpret_cast<Ray*>(raysOut.data() + idx*sizeof(Ray));
    //     *out = *in;
    //     out->next_cluster = FINISHED_RAY_ID;
    //   }
    // }

    // barrier(/*phase*/2, workerId);

    // if (workerId == 0) {
    //   unsigned haveSeed = 0;
    //   unsigned seedX = 0;
    //   unsigned seedLocal = 0;
    //   for (unsigned w = 0; w < NW; ++w) {
    //     if (sharedCounts[w]) {   // this worker saw a seed
    //       seedX     = sharedOffsets[w];
    //       seedLocal = sharedOffsets[w + NW];
    //       haveSeed  = 1;
    //       break;
    //     }
    //   }
    //   sharedCounts[NW]      = haveSeed;
    //   sharedOffsets[2*NW]   = seedX;
    //   sharedOffsets[2*NW+1] = seedLocal;
    // }

    // // Make sure everyone sees the global seed decision
    // barrier(/*phase*/3, workerId);

    
    // const Ray* in = readStructAt<Ray>(raysIn, 0);
    // const uint8_t  y_info  = unpackYInfo(in->y);
    
    // if (sharedCounts[NW]) {
    //   const uint16_t x0        = static_cast<uint16_t>(sharedOffsets[2*NW]   & 0xFFFF);
    //   const uint16_t seedLocal = static_cast<uint16_t>(sharedOffsets[2*NW+1] & 0xFFFF);
    // // if (in->next_cluster != FINISHED_RAY_ID && in->next_cluster == myTile && y_info == 0) {
    // //   const uint16_t x0        = in->x;
    // //   const uint16_t seedLocal = in->next_local;
    //   constexpr unsigned nCols = 3;
    //   const unsigned totalRays = nCols * kFullImageHeight; 
    //   const unsigned writeCount = (totalRays < C) ? totalRays : C;

    //   for (unsigned i = workerId; i < writeCount; i += NW) {
    //     // Map linear i -> (col, row)
    //     const unsigned col = i / kFullImageHeight;       // 0..2
    //     const unsigned row = i % kFullImageHeight;       // 0..H-1
    //     const uint16_t x   = static_cast<uint16_t>((x0 + col) % kFullImageWidth);
    //     const uint16_t y   = static_cast<uint16_t>(row);

    //     // Initialize ray state
    //     glm::vec3 color(0.f, 0.f, 0.f);
    //     float transmittance = 1.f;
    //     float t0 = 0.f;
    //     glm::vec3 rayDir = computeRayDir(x, y, invProj, invView);

    //     int current = seedLocal;
    //     bool finished = false;

    //     Ray* out = reinterpret_cast<Ray*>(raysOut.data() + i * sizeof(Ray));

    //     // March like the normal path
    //     while (!finished) {
    //       if (current < 0 || current >= nLocalPts) {
    //         // Bad start → mark finished (black) on this tile
    //         out->x = x;
    //         out->y = packY(y, /*info=*/1);
    //         out->r = color.x; out->g = color.y; out->b = color.z;
    //         out->t = __builtin_ipu_f32tof16(t0);
    //         out->transmittance = __builtin_ipu_f32tof16(transmittance);
    //         out->next_cluster  = tileOfXY(x, y);
    //         out->next_local    = FINISHED_RAY_ID;
    //         break;
    //       }

    //       const LocalPoint* cur = readStructAt<LocalPoint>(local_pts, current);
    //       glm::vec3 p0(cur->x, cur->y, cur->z);

    //       uint16_t adjStart = (current==0) ? 0 : readStructAt<LocalPoint>(local_pts, current-1)->adj_end;
    //       uint16_t adjEnd   = cur->adj_end;

    //       float closestT = std::numeric_limits<float>::max();
    //       int   next     = -1;

    //       for (uint16_t j = adjStart; j < adjEnd; ++j) {
    //         const uint16_t nbrIdx = adjacency[j];
    //         glm::vec3 p1;
    //         if (nbrIdx < nLocalPts) {
    //           const LocalPoint* nb = readStructAt<LocalPoint>(local_pts, nbrIdx);
    //           p1 = glm::vec3(nb->x, nb->y, nb->z);
    //         } else {
    //           const GenericPoint* nb = readStructAt<GenericPoint>(neighbor_pts, nbrIdx - nLocalPts);
    //           p1 = glm::vec3(nb->x, nb->y, nb->z);
    //         }

    //         const glm::vec3 faceNormal = p1 - p0;
    //         const glm::vec3 faceOrigin = 0.5f*(p1 + p0);
    //         const float dn = glm::dot(faceNormal, rayDir);
    //         if (dn <= 0.f) continue;

    //         const float t = glm::dot(faceOrigin - rayOrigin, faceNormal) / dn;
    //         if (t > 0.f && t < closestT) { closestT = t; next = nbrIdx; }
    //       }

    //       const float delta = closestT - t0;
    //       const float alpha = 1.f - __builtin_ipu_exp(-cur->density * delta);
    //       const float ta    = transmittance * alpha;

    //       color.x += ta * (cur->r / 255.f);
    //       color.y += ta * (cur->g / 255.f);
    //       color.z += ta * (cur->b / 255.f);
    //       transmittance *= (1.f - alpha);
    //       t0 = __builtin_ipu_max(t0, closestT);

    //       // Finish or cross boundary?
    //       if (transmittance < 0.01f || next == -1 || next >= nLocalPts) {
    //         if (transmittance < 0.01f || next == -1) {
    //           // Finished on this tile: route to FB owner
    //           out->x = x;
    //           out->y = packY(y, /*info=*/1);
    //           out->r = color.x; out->g = color.y; out->b = color.z;
    //           out->t = __builtin_ipu_f32tof16(t0);
    //           out->transmittance = __builtin_ipu_f32tof16(transmittance);
    //           out->next_cluster  = tileOfXY(x, y);
    //           out->next_local    = FINISHED_RAY_ID;
    //         } else {
    //           // Cross to neighbor cluster
    //           const GenericPoint* nb = readStructAt<GenericPoint>(neighbor_pts, next - nLocalPts);
    //           out->x = x;
    //           out->y = packY(y, /*info=*/1);
    //           out->r = color.x; out->g = color.y; out->b = color.z;
    //           out->t = __builtin_ipu_f32tof16(t0);
    //           out->transmittance = __builtin_ipu_f32tof16(transmittance);
    //           out->next_cluster  = nb->cluster_id;
    //           out->next_local    = nb->local_id;
    //         }
    //         finished = true;
    //         break;
    //       }

    //       current = next;
    //     } // while
    //   } // worker stripe over writeCount

    //   // Invalidate any unused tail (so routers stop early)
    //   for (unsigned i = writeCount + workerId; i < C; i += NW) {
    //     Ray* r = reinterpret_cast<Ray*>(raysOut.data() + i*sizeof(Ray));
    //     if (r->next_cluster == INVALID_RAY_ID) break;
    //     r->next_cluster = INVALID_RAY_ID;
    //   }
    //   if (workerId == 0) {
    //     *exec_count = *exec_count + 1;
    //   }
    //   return true;
    // }

    unsigned consumedFinishedOnThisTile = 0;
    unsigned localFinished = 0;
    unsigned localMaxSeenPlus1 = 0;
    uint16_t lastSeenIdx = 65533;
    int cell_cntr = 0;
    bool debug = false;
    if(workerId == 0)
      debugBytes[0] = cell_cntr & 0xFF;
    

    // ---- Pass: march rays; write back to SAME index in raysOut; record finished into my slice ----
    for (uint16_t idx = workerId; idx < C; idx += NW) {
      const Ray* in = readStructAt<Ray>(raysIn, idx);
      if (in->next_cluster == INVALID_RAY_ID) break; 
      lastSeenIdx = idx;      
      localMaxSeenPlus1 = idx + 1;
      uint16_t y_coord = unpackYCoord(in->y);
      uint8_t y_info = unpackYInfo(in->y);
      Ray* out = reinterpret_cast<Ray*>(raysOut.data() + idx*sizeof(Ray));
      
      if (in->next_cluster != myTile) { // spillover, immediately return
        std::memcpy(out, in, sizeof(Ray));
        continue;
      }

      if (in->next_local == FINISHED_RAY_ID) {
        if (in->next_cluster == myTile) {
          // write pixel to my framebuffer, then mark as FINISHED_RAY_ID so routers skip it
          const uint16_t x   = in->x;
          const uint16_t y   = y_coord;
          uint16_t lx, ly;
          localXY(x, y, myTile, lx, ly);
          if (lx < kTileImageWidth && ly < kTileImageHeight) {
            FinishedPixel* p = fbAt(lx, ly);
            p->r = clampToU8(in->r);
            p->g = clampToU8(in->g);
            p->b = clampToU8(in->b);
            p->a = 255;
            p->t = __builtin_ipu_f16tof32(in->t);
            ++consumedFinishedOnThisTile;
          }
          // keep head-packed: write FINISHED_RAY_ID into raysOut
          *out = *in;
          out->next_cluster = FINISHED_RAY_ID;
        } else {
          // not my tile → just forward unchanged
          *out = *in;
        }
        continue;
      }

      // --- march ---
      // const uint16_t x10 = data10(in->x);
      glm::vec3 rayDir = computeRayDir(in->x, y_coord, invProj, invView);

      glm::vec3 color(in->r, in->g, in->b);
      float transmittance = __builtin_ipu_f16tof32(in->transmittance);
      float t0    = __builtin_ipu_f16tof32(in->t);

      int current = in->next_local;
      if (current < 0 || current >= nLocalPts) {
        // Bad input → drop the ray safely
        *reinterpret_cast<Ray*>(raysOut.data() + idx*sizeof(Ray)) = *in;
        reinterpret_cast<Ray*>(raysOut.data() + idx*sizeof(Ray))->next_cluster = FINISHED_RAY_ID;
        continue;
      }
      bool finished = false;

      while (!finished) {
        const LocalPoint* cur = readStructAt<LocalPoint>(local_pts, current);
        glm::vec3 p0(cur->x, cur->y, cur->z);

        uint16_t adjStart = (current==0) ? 0 : readStructAt<LocalPoint>(local_pts, current-1)->adj_end;
        uint16_t adjEnd   = cur->adj_end;

        float closestT = std::numeric_limits<float>::max();
        int   next     = -1;

        if(debug && workerId==0) {
          cell_cntr++;
          debugBytes[6 * cell_cntr]     = static_cast<uint8_t>((in->x >> 8) & 0xFF);
          debugBytes[6 * cell_cntr + 1] = static_cast<uint8_t>(in->x & 0xFF);
          debugBytes[6 * cell_cntr + 2]     = static_cast<uint8_t>((y_coord >> 8) & 0xFF);
          debugBytes[6 * cell_cntr + 3] = static_cast<uint8_t>(y_coord & 0xFF);
          debugBytes[6 * cell_cntr + 4]     = static_cast<uint8_t>((current >> 8) & 0xFF);
          debugBytes[6 * cell_cntr + 5] = static_cast<uint8_t>(current & 0xFF);
        }
        for (uint16_t j = adjStart; j < adjEnd; ++j) {
          const uint16_t nbrIdx = adjacency[j];
          glm::vec3 p1;
          if (nbrIdx < local_pts.size()/sizeof(LocalPoint)) {
            const LocalPoint* nb = readStructAt<LocalPoint>(local_pts, nbrIdx);
            p1.x = nb->x;
            p1.y = nb->y;
            p1.z = nb->z;
          } else {
            const GenericPoint* nb = readStructAt<GenericPoint>(neighbor_pts, nbrIdx - nLocalPts);
            p1.x = nb->x;
            p1.y = nb->y;
            p1.z = nb->z;
          }

          const glm::vec3 faceNormal = p1 - p0;
          const glm::vec3 faceOrigin = 0.5f*(p1 + p0);
          const float dn = glm::dot(faceNormal, rayDir);
          if (dn <= 0.f) continue;

          const float t = glm::dot(faceOrigin - rayOrigin, faceNormal) / dn;
          if (t > 0.f && t < closestT) { closestT = t; next = nbrIdx; }
        }

        const float delta = closestT - t0;
        const float alpha = 1.f - __builtin_ipu_exp(-cur->density * delta);
        // const float x_ = cur->density * delta;
        // const float alpha = (x_ > -0.125f && x_ < 0.125f)
        //                   ? (x_ - 0.5f*x_*x_)              // 2-term expm1 approx
        //                   : (1.f - __builtin_ipu_exp(-x_));

        const float ta    = transmittance * alpha;

        color.x += ta * (cur->r / 255.f);
        color.y += ta * (cur->g / 255.f);
        color.z += ta * (cur->b / 255.f);
        transmittance   *= (1.f - alpha);
        t0       = __builtin_ipu_max(t0, closestT);

        // Finish or cross boundary?
        if (transmittance < 0.01f || next == -1 || next >= nLocalPts) {
          if (transmittance < 0.01f || next == -1) {
            // Finished on this tile
            out->x = in->x; 
            out->y = packY(y_coord, 2);
            out->r = color.x; 
            out->g = color.y; 
            out->b = color.z;
            out->t = __builtin_ipu_f32tof16(t0);
            out->transmittance = __builtin_ipu_f32tof16(transmittance);
            out->next_cluster  = tileOfXY(in->x, y_coord);   // << route to FB owner tile
            out->next_local  = FINISHED_RAY_ID;   //
            ++localFinished;
            *result_u16 = FINISHED_RAY_ID;
            *result_float = color.x;
          } else {
            // Cross to neighbor cluster
            const GenericPoint* nb = readStructAt<GenericPoint>(neighbor_pts, next - nLocalPts);
            out->x = in->x; 
            out->y = packY(y_coord, 2);
            out->r = color.x; 
            out->g = color.y; 
            out->b = color.z;
            out->t = __builtin_ipu_f32tof16(t0);
            out->transmittance = __builtin_ipu_f32tof16(transmittance); //__builtin_ipu_max(0.f, __builtin_ipu_min(1.f, transmittance)));
            out->next_cluster  = nb->cluster_id;
            out->next_local    = nb->local_id;

            *result_u16 = nb->cluster_id;
            *result_float = color.x;
          }
          finished = true;
          break;
        }

        current = next;
      } // while(!finished)
    } // for stripe

    unsigned startTail = (lastSeenIdx == 65533) ? workerId : (static_cast<unsigned>(lastSeenIdx) + NW);
    for (unsigned idx = startTail; idx < C; idx += NW) {
      Ray* ray = reinterpret_cast<Ray*>(raysOut.data() + idx*sizeof(Ray));
      if (ray->next_cluster == INVALID_RAY_ID) break;
      ray->next_cluster = INVALID_RAY_ID;
    }

    if(debug && workerId==0){
      debugBytes[0] = cell_cntr & 0xFF;
    }
    if (workerId == 0) {
      *exec_count = *exec_count + 1;
      // *result_u16  = consumedFinishedOnThisTile; 
    }
    return true;
  }

private:
  template <typename T>
  [[gnu::always_inline]] inline const T* readStructAt(const poplar::Input<poplar::Vector<uint8_t>>& buffer, std::size_t index) {
    constexpr std::size_t stride = sizeof(T);
    const uint8_t* base = buffer.data() + index * stride;
    return reinterpret_cast<const T*>(base);
  }

  // [[gnu::always_inline]]
  void barrier(unsigned phase, unsigned workerId) {
    volatile unsigned* flags = readyFlags.data();
    // const unsigned NW = poplar::MultiVertex::numWorkers();
    asm volatile("" ::: "memory");
    while (true) {
      flags[workerId] = phase;
      bool all = true;
      for (unsigned i = 0; i < NW; ++i) if (flags[i] < phase) { all = false; break; }
      if (all) break;
    }
    asm volatile("" ::: "memory");
  }

  [[gnu::always_inline]] inline uint8_t clampToU8(float v) {
    return static_cast<uint8_t>(__builtin_ipu_max(0.0f, __builtin_ipu_min(255.0f, v * 255.0f)));
  }

  [[gnu::always_inline]] inline uint16_t unpackYCoord(uint16_t packed) {
      return packed & 0x03FF; // low 10 bits
  }

  [[gnu::always_inline]] inline uint8_t unpackYInfo(uint16_t packed) {
      return (packed >> 10) & 0x3F; // high 6 bits
  }

  [[gnu::always_inline]] inline uint16_t packY(uint16_t y, uint8_t info) {
    return ( (info & 0x3F) << 10 ) | (y & 0x03FF);
  }

  [[gnu::always_inline]]
  glm::vec3 computeRayDir(uint16_t x, uint16_t y, glm::mat4& invProj, glm::mat4& invView) {
    float ndcX = (2.0f * x) / kFullImageWidth - 1.0f;
    float ndcY = 1.0f - (2.0f * y) / kFullImageHeight;
    glm::vec4 clipRay(ndcX, ndcY, -1.0f, 1.0f);
    glm::vec4 eyeRay = invProj * clipRay;
    eyeRay.z = -1.0f;
    eyeRay.w = 0.0f;
    return glm::normalize(glm::vec3(invView * eyeRay));
  };
 
  [[gnu::always_inline]]
  void computeWriteOffsets() {
      // const unsigned NW = poplar::MultiVertex::numWorkers();
      // exclusive scan over workers
      unsigned head = *finishedWriteOffset;
      unsigned scan = 0, total = 0;
      for (unsigned w = 0; w < NW; ++w) {
        sharedOffsets[w] = head + scan;   // start for worker w
        scan  += sharedCounts[w];
      }
      total = scan;

      // capacity check (assume no wrap during a burst)
      *finishedWriteOffset = head + total;
  }

};

class RayGenMT : public poplar::MultiVertex {
public:
  using BytesIn  = poplar::Input< poplar::Vector<uint8_t,  poplar::VectorLayout::ONE_PTR, 8> >;
  using BytesIO  = poplar::InOut<  poplar::Vector<uint8_t,  poplar::VectorLayout::ONE_PTR, 8> >;

  // I/O
  BytesIn  RaysIn;                     // spillovers from L4 parentOut
  BytesIO  RaysOut;                    // to L4 parentIn

  poplar::InOut< poplar::Vector<uint8_t> > pendingRays; // ring buffer storage
  poplar::InOut<unsigned> pendingHead;                  // ring indices
  poplar::InOut<unsigned> pendingTail;

  poplar::InOut<unsigned> exec_count;
  poplar::Input< poplar::Vector<uint8_t> > camera_cell_info;

  // Scratch
  poplar::InOut< poplar::Vector<unsigned> > readyFlags;    // >= NW
  poplar::InOut< poplar::Vector<unsigned> > sharedCounts;  // >= (2*NW + 3)
  poplar::InOut< poplar::Vector<unsigned> > sharedOffsets; // >= (5*NW + 1)

  // Debug (optional)
  poplar::Output< poplar::Vector<unsigned> > debugBytes;

  // HW workers per tile
  inline static constexpr unsigned NW = 6;

  // Ring capacity (in Rays), pick a fixed multiple of kNumRays you actually allocate
  inline static constexpr unsigned kQueueRays = 2 * kNumRays;

  // sharedOffsets layout (rows):
  //  [0 .. NW-1]      : SPILL emit absolute OUT starts per worker
  //  [NW .. 2NW-1]    : GEN  OUT starts per worker
  //  [2NW .. 3NW-1]   : GEN  OUT counts per worker
  //  [3NW .. 4NW-1]   : GEN  RING starts per worker (absolute ring index)
  //  [4NW .. 5NW-1]   : GEN  RING counts per worker
  //  [5NW]            : RING tail value after all reserved pushes
  inline static constexpr unsigned OFF_SPILL_OUT_START = 0;
  inline static constexpr unsigned OFF_GEN_OUT_START   = NW;
  inline static constexpr unsigned OFF_GEN_OUT_COUNT   = 2*NW;
  inline static constexpr unsigned OFF_RING_START      = 3*NW;
  inline static constexpr unsigned OFF_RING_COUNT      = 4*NW;
  inline static constexpr unsigned OFF_RING_TAIL_NEXT  = 5*NW;

  bool compute(unsigned workerId) {
    // ---------- basics ----------
    const uint16_t cluster_id = (uint16_t)camera_cell_info[0] | ((uint16_t)camera_cell_info[1] << 8);
    const uint16_t local_id   = (uint16_t)camera_cell_info[2] | ((uint16_t)camera_cell_info[3] << 8);

    Ray*       __restrict out  = reinterpret_cast<Ray*>(RaysOut.data());
    const Ray* __restrict inSp = reinterpret_cast<const Ray*>(RaysIn.data());
    Ray*       __restrict ring = reinterpret_cast<Ray*>(pendingRays.data());

    // Queue helpers
    auto qNext = [&](unsigned v)->unsigned {
      unsigned nv = v + 1;
      return (nv == kQueueRays) ? 0u : nv;
    };
    auto qEmpty = [&](unsigned h, unsigned t)->bool {
      return h == t;
    };
    auto qFree = [&](unsigned h, unsigned t)->unsigned {
      // Max storable without overlapping head (one slot kept empty)
      return (kQueueRays - 1) - ((t + kQueueRays - h) % kQueueRays);
    };

    // Modes (change to try different patterns)
    unsigned raygen_mode = 3;

    // ---------- locals shared via barriers ----------
    unsigned head = 0, tail = 0;      // ring indices
    unsigned outBase = 0;             // next free slot in RaysOut during this run

    // Stats (debug)
    unsigned pendEmitted=0, spillSeen=0, spillEmitted=0, spillQueued=0, genEmitted=0, genQueued=0, drops=0;

    // ---------- PHASE 0: reset flags + load ring indices ----------
    if (workerId == 0) {
      head = *pendingHead;
      tail = *pendingTail;
      for (unsigned w=0; w<NW; ++w) {
        readyFlags[w]   = 0;
        sharedCounts[w] = 0;        // spill counts
        sharedCounts[NW + w] = 0;   // spill emit counts
      }
    }
    barrier(1, workerId);

    // ---------- PHASE 1: PENDING -> OUT (worker 0) ----------
    if (workerId == 0) {
      while (outBase < kNumRays && !qEmpty(head, tail)) {
        out[outBase] = ring[head];
        head = qNext(head);
        ++outBase;
        ++pendEmitted;
      }
    }
    barrier(2, workerId);

    // ---------- PHASE 2: count SPILLOVERS in parallel ----------
    unsigned mySpillCnt = 0;
    for (unsigned i = workerId; i < kNumRays; i += NW) {
      const Ray* r = &inSp[i];
      if (r->next_cluster == INVALID_RAY_ID) break;
      ++spillSeen;
      ++mySpillCnt;
    }
    sharedCounts[workerId] = mySpillCnt;
    barrier(3, workerId);

    // ---------- PHASE 3: assign spill EMIT vs QUEUE; compute emit starts ----------
    if (workerId == 0) {
      unsigned outLeft = (kNumRays > outBase) ? (kNumRays - outBase) : 0;

      // Greedy: each worker emits up to min(mySpillCnt, outLeft-leftover)
      unsigned scanEmit = 0;
      for (unsigned w=0; w<NW; ++w) {
        unsigned take = sharedCounts[w];
        if (take > outLeft) take = outLeft;
        sharedCounts[NW + w] = take;                             // per-worker spill EMIT count
        sharedOffsets[OFF_SPILL_OUT_START + w] = outBase + scanEmit; // absolute out start
        scanEmit += take;
        outLeft  -= take;
      }
      outBase       += scanEmit;  // consumed these OUT slots
      spillEmitted  += scanEmit;

      // Everything else is queued (in input order) if ring has space
      unsigned totalSpill = 0;
      for (unsigned w=0; w<NW; ++w) totalSpill += sharedCounts[w];
      unsigned toQueue = totalSpill - spillEmitted;

      unsigned freeSlots = qFree(head, tail);
      unsigned canQueue  = (toQueue <= freeSlots) ? toQueue : freeSlots;
      spillQueued = canQueue;
      drops       = toQueue - canQueue;

      // Worker 0 will push exactly 'canQueue' later in input order (Phase 4)
      // Remember how many to skip (the ones emitted): store in sharedCounts[2*NW]
      sharedCounts[2*NW] = spillEmitted;   // "skip" count for Phase 4
      sharedCounts[2*NW+1] = canQueue;     // "queue" count for Phase 4
    }
    barrier(4, workerId);

    // ---------- PHASE 4: emit my SPILL slice to OUT (parallel) ----------
    {
      unsigned myEmit = sharedCounts[NW + workerId];
      if (myEmit) {
        unsigned outStart = sharedOffsets[OFF_SPILL_OUT_START + workerId];
        unsigned written = 0;
        for (unsigned i = workerId; i < kNumRays && written < myEmit; i += NW) {
          const Ray* r = &inSp[i];
          if (r->next_cluster == INVALID_RAY_ID) break;
          out[outStart + written] = *r;
          ++written;
        }
      }
    }
    barrier(5, workerId);

    // ---------- PHASE 5: queue remaining SPILLOVERS (worker 0, preserve order) ----------
    if (workerId == 0) {
      unsigned skip   = sharedCounts[2*NW];     // how many already emitted
      unsigned toPush = sharedCounts[2*NW+1];   // how many to queue
      unsigned skipped = 0, queued = 0;

      for (unsigned i = 0; i < kNumRays && queued < toPush; ++i) {
        const Ray* r = &inSp[i];
        if (r->next_cluster == INVALID_RAY_ID) break;
        if (skipped < skip) { ++skipped; continue; }
        // push to ring
        ring[tail] = *r;
        tail = qNext(tail);
        ++queued;
      }
    }
    barrier(6, workerId);

    // ---------- PHASE 6: pre-slice GENERATION output and ring (worker 0) ----------
    if (workerId == 0) {
      // Remaining OUT capacity
      unsigned outLeft = (kNumRays > outBase) ? (kNumRays - outBase) : 0;

      // Slice OUT among workers
      unsigned scan = outBase;
      for (unsigned w=0; w<NW; ++w) {
        unsigned q = splitQuota(outLeft, w);
        sharedOffsets[OFF_GEN_OUT_START + w] = scan;
        sharedOffsets[OFF_GEN_OUT_COUNT + w] = q;
        scan += q;
      }
      // Remaining RING capacity (after Phase 5)
      unsigned ringLeft = qFree(head, tail);

      // Slice RING among workers (parallel push; tail advanced once at the end)
      unsigned rscan = 0;
      for (unsigned w=0; w<NW; ++w) {
        unsigned q = splitQuota(ringLeft, w);
        sharedOffsets[OFF_RING_START + w] = (tail + rscan) % kQueueRays; // absolute ring index
        sharedOffsets[OFF_RING_COUNT + w] = q;
        rscan += q;
      }
      sharedOffsets[OFF_RING_TAIL_NEXT] = (tail + rscan) % kQueueRays;

      // Publish for debug/invalidation
      sharedCounts[2*NW]   = outBase;       // old base
      sharedCounts[2*NW+1] = outLeft;       // how many gen OUT slots
      sharedCounts[2*NW+2] = ringLeft;      // how many gen RING slots
    }
    barrier(7, workerId);

    // ---------- PHASE 7: GENERATION (each worker uses its own slices) ----------
    {
      // OUT slice
      unsigned myOutStart = sharedOffsets[OFF_GEN_OUT_START + workerId];
      unsigned myOutCount = sharedOffsets[OFF_GEN_OUT_COUNT + workerId];
      unsigned myOutEnd   = myOutStart + myOutCount;
      unsigned myOutPos   = myOutStart;

      auto emit_out = [&](const Ray& g)->bool {
        if (myOutPos >= myOutEnd) return false;
        out[myOutPos++] = g;
        ++genEmitted;
        return true;
      };

      // RING slice
      unsigned myRingStart = sharedOffsets[OFF_RING_START + workerId];
      unsigned myRingCount = sharedOffsets[OFF_RING_COUNT + workerId];
      unsigned myRingIdx   = myRingStart;

      auto push_ring_reserved = [&](const Ray& g)->bool {
        if (!myRingCount) return false;
        ring[myRingIdx] = g;
        myRingIdx = (myRingIdx + 1 == kQueueRays) ? 0u : (myRingIdx + 1);
        --myRingCount;
        ++genQueued;
        return true;
      };

      // Build the template ray (constant fields)
      Ray g0{};
      g0.r = g0.g = g0.b = 0.0f;
      g0.t = __builtin_ipu_f32tof16(0.0f);
      g0.transmittance = __builtin_ipu_f32tof16(1.0f);
      g0.next_cluster = cluster_id;
      g0.next_local   = local_id;

      // Modes (kept as in your code, but split work among workers)
      auto gen_mode_0 = [&](){
        const unsigned interval = 1;
        const bool genThisFrame = ((*exec_count % interval) == 0);
        if (!genThisFrame) return;
        const unsigned nRowsPerFrame = 1;
        const unsigned rowBase = ((*exec_count)/interval) * nRowsPerFrame;

        for (uint16_t cy = 0; cy < nRowsPerFrame; ++cy) {
          const uint16_t y = (rowBase + cy) % kFullImageHeight;
          for (uint16_t x = workerId; x < kFullImageWidth; x += NW) {
            Ray g = g0; g.x = x; g.y = y;
            if (emit_out(g)) continue;
            if (push_ring_reserved(g)) continue;
            return; // both slices exhausted
          }
        }
      };

      auto gen_mode_1 = [&](){
        const unsigned interval = 1;
        const bool genThisFrame = ((*exec_count % interval) == 0);
        if (!genThisFrame) return;
        const unsigned nColsPerFrame = 3;
        const unsigned colBase = ((*exec_count)/interval) * nColsPerFrame;

        for (uint16_t cx = 0; cx < nColsPerFrame; ++cx) {
          const uint16_t x = (colBase + cx) % kFullImageWidth;
          for (uint16_t y = workerId; y < kFullImageHeight; y += NW) {
            Ray g = g0; g.x = x; g.y = y;
            if (emit_out(g)) continue;
            if (push_ring_reserved(g)) continue;
            return;
          }
        }
      };

      auto gen_mode_2 = [&](){
        const bool genThisFrame = (*exec_count == 1);
        if (!genThisFrame) return;
        // Single specific ray on worker 0 to match your original behavior
        if (workerId != 0) return;
        Ray g = g0; g.x = 200; g.y = 440;
        if (emit_out(g)) return;
        (void)push_ring_reserved(g);
      };

      auto gen_mode_3 = [&](){
        const unsigned interval = 1;
        const bool genThisFrame = ((*exec_count % interval) == 0 && *exec_count < 1024);
        if (!genThisFrame) return;

        const unsigned bx = 32, by = 32;
        const unsigned total = bx * by;
        const unsigned step  = *exec_count % total;

        // Partition outer loop by worker (basex)
        for (unsigned basex = workerId; basex < (kFullImageWidth / bx); basex += NW) {
          for (unsigned basey = 0; basey < (kFullImageHeight / by); ++basey) {
            Ray g = g0;
            g.x = basex*bx + (step % bx);
            g.y = basey*by + (step / bx);
            if (emit_out(g)) continue;
            if (push_ring_reserved(g)) continue;
            return;
          }
        }
      };

      switch (raygen_mode) {
        case 0: gen_mode_0(); break;
        case 1: gen_mode_1(); break;
        case 2: gen_mode_2(); break;
        case 3: gen_mode_3(); break;
        default: break;
      }
    }
    barrier(8, workerId);

    // Advance ring tail once (after all reserved writes)
    if (workerId == 0) {
      tail = sharedOffsets[OFF_RING_TAIL_NEXT];
    }
    barrier(9, workerId);

    // ---------- PHASE 8: invalidate tail of RaysOut & persist ring indices (worker 0) ----------
    if (workerId == 0) {
      // OUT end after generation
      unsigned oldBase = sharedCounts[2*NW];
      unsigned genOut  = sharedCounts[2*NW+1];
      unsigned finalOutEnd = oldBase + genOut; // we reserved exactly 'genOut' across workers

      // Invalidate remaining tail
      for (unsigned i = finalOutEnd; i < kNumRays; ++i) {
        if (out[i].next_cluster == INVALID_RAY_ID) break;
        out[i].next_cluster = INVALID_RAY_ID;
      }

      *pendingHead = head;
      *pendingTail = tail;

      // Debug (coarse totals)
      unsigned* dbg = reinterpret_cast<unsigned*>(debugBytes.data());
      dbg[0] = *exec_count;
      dbg[1] = finalOutEnd;                 // total emitted to out
      dbg[2] = (kQueueRays - 1) - ((tail + kQueueRays - head) % kQueueRays); // ring free
      dbg[3] = pendEmitted;
      dbg[4] = spillSeen;
      dbg[5] = spillEmitted;
      dbg[6] = spillQueued;
      dbg[7] = genEmitted;   // local to worker 0, but OK as indicative
      dbg[8] = genQueued;    // local to worker 0, indicative
      dbg[9] = drops;

      *exec_count = *exec_count + 1;
    }
    barrier(10, workerId);

    return true;
  }

private:
  // Simple fair split: ceil(total/NW) distribution
  [[gnu::always_inline]]
  static unsigned splitQuota(unsigned total, unsigned w) {
    unsigned q = total / NW;
    unsigned r = total % NW;
    return q + (w < r ? 1u : 0u);
  }

  // Robust barrier (same style you used that worked)
  void barrier(unsigned phase, unsigned workerId) {
    volatile unsigned* flags = readyFlags.data();
    asm volatile("" ::: "memory");
    while (true) {
      flags[workerId] = phase;
      bool all = true;
      for (unsigned i = 0; i < NW; ++i) {
        if (flags[i] < phase) { all = false; break; }
      }
      if (all) break;
    }
    asm volatile("" ::: "memory");
  }
};

class RayGen : public poplar::Vertex {
public:
  poplar::Input<poplar::Vector<uint8_t>>  RaysIn;
  poplar::Output<poplar::Vector<uint8_t>> RaysOut;

  poplar::InOut<poplar::Vector<uint8_t>>  pendingRays; 
  poplar::InOut<unsigned>                 pendingHead; 
  poplar::InOut<unsigned>                 pendingTail;  

  poplar::InOut<unsigned> exec_count;
  poplar::Input<poplar::Vector<uint8_t>> camera_cell_info;
  poplar::Output<poplar::Vector<unsigned>> debugBytes;

  poplar::InOut< poplar::Vector<unsigned> > readyFlags;
  poplar::InOut< poplar::Vector<unsigned> > sharedCounts;
  poplar::InOut< poplar::Vector<unsigned> > sharedOffsets; 

  bool compute() {
    const uint16_t cluster_id = camera_cell_info[0] | (camera_cell_info[1] << 8);
    const uint16_t local_id   = camera_cell_info[2] | (camera_cell_info[3] << 8);

    const unsigned C = RaysOut.size()    / sizeof(Ray);
    const unsigned P = pendingRays.size()/ sizeof(Ray);
    
    unsigned head = *pendingHead;
    unsigned tail = *pendingTail;

    auto qNext  = [&](unsigned v) -> unsigned { unsigned nv = v + 1; return (nv == P) ? 0u : nv; };
    auto qEmpty = [&](unsigned h, unsigned t) -> bool { return h == t; };
    auto qFree  = [&](unsigned h, unsigned t) -> unsigned { return (P - 1) - ((t + P - h) % P); }; // max storable without overlap

    auto qPop = [&](Ray &r) -> bool {
      if (qEmpty(head, tail)) return false;
      std::memcpy(&r, pendingRays.data() + head*sizeof(Ray), sizeof(Ray));
      head = qNext(head);
      return true;
    };
    auto qPush = [&](const Ray &r) -> bool {
      unsigned nt = qNext(tail);
      if (nt == head) return false; // full
      std::memcpy(pendingRays.data() + tail*sizeof(Ray), &r, sizeof(Ray));
      tail = nt;
      return true;
    };
    auto emit = [&](const Ray &r, unsigned &outCnt) -> bool {
      if (outCnt >= C) return false;
      std::memcpy(RaysOut.data() + outCnt*sizeof(Ray), &r, sizeof(Ray));
      ++outCnt;
      return true;
    };

    unsigned outCnt = 0;

    // Stats (debug)
    unsigned pendEmitted=0, spillSeen=0, spillEmitted=0, spillQueued=0, genEmitted=0, genQueued=0, drops=0;

    // ── 1) send pending first
    Ray tmp{};
    while (outCnt < C && qPop(tmp)) {
      emit(tmp, outCnt);
      ++pendEmitted;
    }

    // ── 2) forward spillovers next; queue overflow
    for (unsigned i = 0; i < C; ++i) {
      const Ray* r = reinterpret_cast<const Ray*>(RaysIn.data() + i*sizeof(Ray));
      if (r->next_cluster == INVALID_RAY_ID) break;
      ++spillSeen;

      if (emit(*r, outCnt)) { ++spillEmitted; }
      else if (qPush(*r))   { ++spillQueued;  }
      else                  { ++drops;        } // ring full
    }

    // ── 3) fill remaining with generated rays
    int raygen_mode = 1;
    if(raygen_mode == 0) {
      const unsigned interval = 1;
      const bool genThisFrame = ((*exec_count % interval) == 0);
      if (genThisFrame) {
        const unsigned nRowsPerFrame = 1;
        const unsigned rowBase = ((*exec_count)/interval) * nRowsPerFrame;

        // Don’t generate more than remaining output capacity + queue free slots
        unsigned budget = (C - outCnt) + qFree(head, tail);

        Ray g{};
        g.r = g.g = g.b = 0.0f;
        g.t = __builtin_ipu_f32tof16(0.0f);
        g.transmittance = __builtin_ipu_f32tof16(1.0f);
        g.next_cluster = cluster_id;
        g.next_local   = local_id;
        for (uint16_t cy = 0; cy < nRowsPerFrame && budget; ++cy) {
          const uint16_t y = (rowBase + cy) % kFullImageWidth;
          for (uint16_t x = 0; x < kFullImageWidth && budget; ++x) {
            g.x = x; 
            g.y = y;

            if (emit(g, outCnt)) { ++genEmitted; }
            else if (qPush(g))   { ++genQueued;  }
            else                 { ++drops;      } // shouldn’t happen 
            --budget;
          }
        }
      }
    }
    if(raygen_mode == 1) {
      const unsigned interval = 1;
      const bool genThisFrame = ((*exec_count % interval) == 0);
      if (genThisFrame) {
        const unsigned nColsPerFrame = 2;
        const unsigned colBase = ((*exec_count)/interval) * nColsPerFrame;

        // Don’t generate more than remaining output capacity + queue free slots
        unsigned budget = (C - outCnt) + qFree(head, tail);

        Ray g{};
        g.r = g.g = g.b = 0.0f;
        g.t = __builtin_ipu_f32tof16(0.0f);
        g.transmittance = __builtin_ipu_f32tof16(1.0f);
        g.next_cluster = cluster_id;
        g.next_local   = local_id;
        for (uint16_t cx = 0; cx < nColsPerFrame && budget; ++cx) {
          const uint16_t x = (colBase + cx) % kFullImageWidth;
          for (uint16_t y = 0; y < kFullImageHeight && budget; ++y) {
            g.x = x; 
            g.y = y;

            if (emit(g, outCnt)) { ++genEmitted; }
            else if (qPush(g))   { ++genQueued;  }
            else                 { ++drops;      } // shouldn’t happen 
            --budget;
          }
        }
      }
    }
    if(raygen_mode == 2) {
      const bool genThisFrame = (*exec_count  == 1);
      if (genThisFrame) {
        unsigned budget = (C - outCnt) + qFree(head, tail);
        Ray g{};
        g.r = g.g = g.b = 0.0f;
        g.t = __builtin_ipu_f32tof16(0.0f);
        g.transmittance = __builtin_ipu_f32tof16(1.0f);
        g.next_cluster = cluster_id;
        g.next_local   = local_id;
        g.x = 200; 
        g.y = 440;

        if (emit(g, outCnt)) { ++genEmitted; }
        else if (qPush(g))   { ++genQueued;  }
        else                 { ++drops;      } 
        --budget;
      }
    }
    if(raygen_mode == 3) {
      const unsigned interval = 1;
      const bool genThisFrame = ((*exec_count % interval) == 0 && *exec_count < 1024);
      if (genThisFrame) {
        unsigned budget = (C - outCnt) + qFree(head, tail);
        unsigned bx = 32, by = 32;
        unsigned total = bx * by;
        unsigned step = *exec_count % total;
        Ray g{};
        g.r = g.g = g.b = 0.0f;
        g.t = __builtin_ipu_f32tof16(0.0f);
        g.transmittance = __builtin_ipu_f32tof16(1.0f);
        g.next_cluster = cluster_id;
        g.next_local   = local_id;
        for(int basex=0; basex < kFullImageWidth/bx; basex++) {
          for(int basey=0; basey < kFullImageHeight/by; basey++) {
            g.x = basex*bx + step%bx; 
            g.y = basey*by + step/bx;
    
            if (emit(g, outCnt)) { ++genEmitted; }
            else if (qPush(g))   { ++genQueued;  }
            else                 { ++drops;      } 
            --budget;
          }
        }
      }
    }
    if(raygen_mode == 4) {
      const unsigned interval = 1;
      const bool genThisFrame = ((*exec_count % interval) == 0);
      if (genThisFrame) {
        const unsigned nColsPerFrame = 3;
        const unsigned colBase = ((*exec_count)/interval) * nColsPerFrame;
        Ray g{};
        g.r = g.g = g.b = 0.0f;
        g.t = __builtin_ipu_f32tof16(0.0f);
        g.transmittance = __builtin_ipu_f32tof16(1.0f);
        g.next_cluster = cluster_id;
        g.next_local   = local_id;
        g.x = colBase% kFullImageWidth; 
        g.y = 0;

        if (emit(g, outCnt)) { ++genEmitted; }
        else if (qPush(g))   { ++genQueued;  }
        else                 { ++drops;      }
      }
    }
    // ── Invalidate tail of RaysOut
    for (unsigned i = outCnt; i < C; ++i) {
      Ray* r = reinterpret_cast<Ray*>(RaysOut.data() + i*sizeof(Ray));
      if (r->next_cluster == INVALID_RAY_ID) break;
      r->next_cluster = INVALID_RAY_ID;
    }

    // Persist ring indices
    *pendingHead = head;
    *pendingTail = tail;

    // Debug (10 x uint16)
    unsigned* dbg = reinterpret_cast<unsigned*>(debugBytes.data());
    dbg[0] = *exec_count;
    dbg[1] = outCnt;                 // total emitted
    dbg[2] = (P - 1) - qFree(head, tail); // backlog size
    dbg[3] = pendEmitted;
    dbg[4] = spillSeen;
    dbg[5] = spillEmitted;
    dbg[6] = spillQueued;
    dbg[7] = genEmitted;
    dbg[8] = genQueued;
    dbg[9] = drops;

    *exec_count = *exec_count + 1;
    return true;
  }
};

class RayRouter : public poplar::MultiVertex {
  
public:
  using RayVecIn = poplar::Input< poplar::Vector<uint8_t, poplar::VectorLayout::ONE_PTR, 8> >;
  using RayVecIO = poplar::InOut< poplar::Vector<uint8_t, poplar::VectorLayout::ONE_PTR, 8> >;
  // Incoming and outcoming rays
  RayVecIn parentRaysIn, childRaysIn0, childRaysIn1, childRaysIn2, childRaysIn3;
  RayVecIO parentRaysOut, childRaysOut0, childRaysOut1, childRaysOut2, childRaysOut3;

  // ID mapping to know which cluster IDs belong to which child
  poplar::Input<poplar::Vector<unsigned short, poplar::VectorLayout::SPAN, 4>> childClusterIds;
  poplar::Input<uint8_t> level;

  poplar::InOut<poplar::Vector<unsigned>> debugBytes;

  poplar::InOut<poplar::Vector<unsigned>> sharedCounts; 
  poplar::InOut<poplar::Vector<unsigned>> sharedOffsets; 
  poplar::InOut<poplar::Vector<unsigned>> readyFlags; 

  inline static constexpr unsigned kNumLanes = 5;
  inline static constexpr unsigned PARENT = 4;
  inline static constexpr unsigned NW = 6;//MultiVertex::numWorkers(); 

  struct LanePartition {
    unsigned base[kNumLanes];             // primary writes per lane, clamped to capacity C
    unsigned freePrefix[kNumLanes + 1];   // prefix-scan of free capacities across lanes
  };

  uint16_t myChildPrefix;
  uint8_t shift;

  [[poplar::constraint("elem(*parentRaysIn)!=elem(*parentRaysOut)")]]
  [[poplar::constraint("elem(*parentRaysIn)!=elem(*childRaysOut0)")]]
  [[poplar::constraint("elem(*parentRaysIn)!=elem(*childRaysOut1)")]]
  [[poplar::constraint("elem(*parentRaysIn)!=elem(*childRaysOut2)")]]
  [[poplar::constraint("elem(*parentRaysIn)!=elem(*childRaysOut3)")]]
  [[poplar::constraint("elem(*childRaysIn0)!=elem(*parentRaysOut)")]]
  [[poplar::constraint("elem(*childRaysIn0)!=elem(*childRaysOut0)")]]
  [[poplar::constraint("elem(*childRaysIn0)!=elem(*childRaysOut1)")]]
  [[poplar::constraint("elem(*childRaysIn0)!=elem(*childRaysOut2)")]]
  [[poplar::constraint("elem(*childRaysIn0)!=elem(*childRaysOut3)")]]
  [[poplar::constraint("elem(*childRaysIn1)!=elem(*parentRaysOut)")]]
  [[poplar::constraint("elem(*childRaysIn1)!=elem(*childRaysOut0)")]]
  [[poplar::constraint("elem(*childRaysIn1)!=elem(*childRaysOut1)")]]
  [[poplar::constraint("elem(*childRaysIn1)!=elem(*childRaysOut2)")]]
  [[poplar::constraint("elem(*childRaysIn1)!=elem(*childRaysOut3)")]]
  [[poplar::constraint("elem(*childRaysIn2)!=elem(*parentRaysOut)")]]
  [[poplar::constraint("elem(*childRaysIn2)!=elem(*childRaysOut0)")]]
  [[poplar::constraint("elem(*childRaysIn2)!=elem(*childRaysOut1)")]]
  [[poplar::constraint("elem(*childRaysIn2)!=elem(*childRaysOut2)")]]
  [[poplar::constraint("elem(*childRaysIn2)!=elem(*childRaysOut3)")]]
  [[poplar::constraint("elem(*childRaysIn3)!=elem(*parentRaysOut)")]]
  [[poplar::constraint("elem(*childRaysIn3)!=elem(*childRaysOut0)")]]
  [[poplar::constraint("elem(*childRaysIn3)!=elem(*childRaysOut1)")]]
  [[poplar::constraint("elem(*childRaysIn3)!=elem(*childRaysOut2)")]]
  [[poplar::constraint("elem(*childRaysIn3)!=elem(*childRaysOut3)")]]

  [[poplar::constraint("elem(*parentRaysOut)!=elem(*childRaysOut0)")]]
  [[poplar::constraint("elem(*parentRaysOut)!=elem(*childRaysOut1)")]]
  [[poplar::constraint("elem(*parentRaysOut)!=elem(*childRaysOut2)")]]
  [[poplar::constraint("elem(*parentRaysOut)!=elem(*childRaysOut3)")]]
  [[poplar::constraint("elem(*childRaysOut0)!=elem(*childRaysOut1)")]]
  [[poplar::constraint("elem(*childRaysOut0)!=elem(*childRaysOut2)")]]
  [[poplar::constraint("elem(*childRaysOut0)!=elem(*childRaysOut3)")]]
  [[poplar::constraint("elem(*childRaysOut1)!=elem(*childRaysOut2)")]]
  [[poplar::constraint("elem(*childRaysOut1)!=elem(*childRaysOut3)")]]
  [[poplar::constraint("elem(*childRaysOut2)!=elem(*childRaysOut3)")]]

  [[poplar::constraint("elem(*readyFlags)!=elem(*sharedCounts)")]]
  [[poplar::constraint("elem(*readyFlags)!=elem(*sharedOffsets)")]]
  [[poplar::constraint("elem(*readyFlags)!=elem(*debugBytes)")]]
  [[poplar::constraint("elem(*sharedCounts)!=elem(*sharedOffsets)")]]
  [[poplar::constraint("elem(*sharedCounts)!=elem(*debugBytes)")]]
  [[poplar::constraint("elem(*sharedOffsets)!=elem(*debugBytes)")]]
  
  bool compute(unsigned workerId) {
    shift = *level * 2;
    myChildPrefix = childClusterIds[0] >> (shift + 2);

    if (workerId == 0) {
      volatile unsigned* f = readyFlags.data();
      for (unsigned i=0;i<NW;++i) f[i] = 0;
    }
    barrier(/*phase=*/0, workerId);  
    
    if (workerId == 0) {
      volatile unsigned* f = readyFlags.data();
      for (unsigned i=0;i<NW;++i) f[i] = 0;
    }
    countRays(workerId);
    barrier(/*phase=*/1, workerId);  

    unsigned outCnt[kNumLanes] = {0,0,0,0,0};
    if(workerId == 0) {
      for(int i=0; i<NW; i++) {
        for(int lane=0; lane<kNumLanes; lane++) {
          outCnt[lane] += sharedCounts[i*kNumLanes + lane];
        }
      }
      computeWriteOffsets();
    }
    barrier(/*phase=*/2, workerId);  

    routeRays(workerId);
    barrier(/*phase=*/3, workerId);  

    if(workerId == 0) {
      invalidateAfterRouting();      
      writeDebugInfos();
    }

    return true;
  }
private:
  [[gnu::always_inline]] unsigned getSharedIdx(unsigned workerId, unsigned lane) {
    return workerId * kNumLanes + lane;
  }

  [[gnu::always_inline]] unsigned spillStartIdx(unsigned w) const {
    // const unsigned NW = poplar::MultiVertex::numWorkers();
    return NW * kNumLanes + w;
  }

  [[gnu::always_inline]] unsigned spillEndIdx(unsigned w) const {
    // const unsigned NW = poplar::MultiVertex::numWorkers();
    return NW * kNumLanes + NW + w;
  }

  [[gnu::always_inline]] unsigned laneHeadShiftIdx(unsigned lane) const {
    // needs 5 extra slots after existing spillStart/End ranges
    return NW * kNumLanes + 2 * NW + lane;
  }

  [[gnu::always_inline]]
  void barrier(unsigned phase, unsigned workerId) {
    volatile unsigned* flags = readyFlags.data();
    // const unsigned NW = poplar::MultiVertex::numWorkers();
    flags[workerId] = phase;
    asm volatile("" ::: "memory");
    while (true) {
      bool all = true;
      for (unsigned i = 0; i < NW; ++i) {
        if (flags[i] < phase) { 
          all = false; 
          break; 
        }
      }
      if (all) break;
    }
    asm volatile("" ::: "memory");
  }
  
  [[gnu::always_inline]]
  static inline void copy_ray(const Ray* __restrict src, Ray* __restrict dst) {
    const unsigned* __restrict s = reinterpret_cast<const unsigned*>(src);
    unsigned*       __restrict d_raw = reinterpret_cast<unsigned*>(dst);
    unsigned *d = d_raw; 
    // exact unroll (6 x 32-bit)
    ipu::store_postinc(&d, s[0], 1);
    ipu::store_postinc(&d, s[1], 1);
    ipu::store_postinc(&d, s[2], 1);
    ipu::store_postinc(&d, s[3], 1);
    ipu::store_postinc(&d, s[4], 1);
    ipu::store_postinc(&d, s[5], 1);
  }

  [[gnu::always_inline]]
  unsigned findChildForCluster (uint16_t cluster_id) {
      unsigned childIdx  = (cluster_id >> shift) & 0x3;
      bool isChild = ((cluster_id >> (shift + 2)) == myChildPrefix);
      return isChild ? childIdx : PARENT;
  };
  [[gnu::always_inline]]
  void totalsPerLane(const poplar::Vector<unsigned> &sharedCounts,
                            unsigned NW,
                            unsigned outTot[kNumLanes]) {
    for (unsigned r = 0; r < kNumLanes; ++r) outTot[r] = 0;
    for (unsigned r = 0; r < kNumLanes; ++r)
      for (unsigned w = 0; w < NW; ++w)
        outTot[r] += sharedCounts[getSharedIdx(w, r)];
  }

  [[gnu::always_inline]]
  void mapGlobalFreeToLane(unsigned g, const LanePartition &P,
                                  unsigned &r, unsigned &slotInR) {
    r = 0;
    while (g >= P.freePrefix[r + 1]) ++r;             // 5 lanes max
    slotInR = P.base[r] + (g - P.freePrefix[r]);
  }

  [[gnu::always_inline]]
  LanePartition makePartition(const unsigned tot[kNumLanes], unsigned C) {
    LanePartition P{};
    P.freePrefix[0] = 0;
    for (unsigned r = 0; r < kNumLanes; ++r) {
      unsigned used = (tot[r] < C) ? tot[r] : C;
      P.base[r] = used;
      unsigned freeCap = C - used;
      P.freePrefix[r + 1] = P.freePrefix[r] + freeCap;
    }
    return P;
  }

  struct SpillCursor { unsigned r; unsigned nextBoundary; };

  [[gnu::always_inline]]
  static void initSpillCursor(const LanePartition& P, unsigned g, SpillCursor& sc) {
    sc.r = 0;
    while (g >= P.freePrefix[sc.r + 1]) ++sc.r;
    sc.nextBoundary = P.freePrefix[sc.r + 1];
  }

  [[gnu::always_inline]]
  static void spillAdvanceIfNeeded(const LanePartition& P, unsigned g, SpillCursor& sc) {
    if (g >= sc.nextBoundary) {
      do { ++sc.r; } while (g >= P.freePrefix[sc.r + 1]);
      sc.nextBoundary = P.freePrefix[sc.r + 1];
    }
  }

  [[gnu::always_inline]]
  void accumulateSpillAssigned(const poplar::Vector<unsigned> &so,
                              unsigned NW,
                              const LanePartition &P,
                              unsigned assigned[kNumLanes]) {
    for (unsigned r = 0; r < kNumLanes; ++r) assigned[r] = 0;
    for (unsigned w = 0; w < NW; ++w) {
      unsigned a = so[spillStartIdx(w)];      // [a, b)
      unsigned b = so[spillEndIdx(w)];
      unsigned r = 0;
      while (a < b) {
        while (a >= P.freePrefix[r + 1]) ++r; // find bin containing 'a'
        unsigned R = P.freePrefix[r + 1];
        unsigned take = ((b < R) ? b : R) - a;
        assigned[r] += take;
        a += take;
      }
    }
  }

  void countRays(unsigned workerId) {
    for (unsigned lane = 0; lane < kNumLanes; ++lane)
      sharedCounts[getSharedIdx(workerId, lane)] = 0;

    // Cast bases once (you’re using uint8_t buffers)
    const Ray* __restrict inBase[kNumLanes] = {
      reinterpret_cast<const Ray*>(childRaysIn0.data()),
      reinterpret_cast<const Ray*>(childRaysIn1.data()),
      reinterpret_cast<const Ray*>(childRaysIn2.data()),
      reinterpret_cast<const Ray*>(childRaysIn3.data()),
      reinterpret_cast<const Ray*>(parentRaysIn.data())
    };

    __builtin_assume(kNumRays <= 4096);

    for (unsigned lane = 0; lane < kNumLanes; ++lane) {
      const Ray* __restrict in = inBase[lane];
      // worker-strided: i = workerId, workerId+NW, ...
      for (rptsize_t i = (rptsize_t)workerId; i < (rptsize_t)kNumRays; i += (rptsize_t)NW) {
        const Ray* ray = &in[(unsigned)i];
        // Sentinel means end of valid rays for this lane → safe to break for this worker’s stride
        if (__builtin_expect(ray->next_cluster == INVALID_RAY_ID, 0)) break;
        if (__builtin_expect(ray->next_cluster == FINISHED_RAY_ID, 0)) continue;
        const unsigned dst = findChildForCluster(ray->next_cluster);
        ++sharedCounts[getSharedIdx(workerId, dst)];
      }
    }
  }
  
  void computeWriteOffsets() {
  // const unsigned NW = poplar::MultiVertex::numWorkers();
  const unsigned C = kNumRays; // raysOut.size()    / sizeof(Ray);

  // compute offsets without considering spillovers
  for (unsigned lane = 0; lane < kNumLanes; ++lane) {
    unsigned offset = 0;
    for (unsigned worker = 0; worker < NW; ++worker) {
      unsigned count = sharedCounts[getSharedIdx(worker, lane)];
      sharedOffsets[getSharedIdx(worker, lane)] = offset;
      offset += count;
    }
  }

  // 2) totals and free per lane
  unsigned tot[kNumLanes] = {0,0,0,0,0};
  totalsPerLane(sharedCounts, NW, tot);
  LanePartition P = makePartition(tot, C);
  const unsigned totalFree = P.freePrefix[kNumLanes];

  // 3) Spill needed by each worker = sum_lanes max(0, start+cnt - C)
  unsigned workerSpill[NW];  // kNumWorkers >= NW
  for (unsigned w = 0; w < NW; ++w) {
    unsigned s = 0;
    for (unsigned lane = 0; lane < kNumLanes; ++lane) {
      const unsigned start = sharedOffsets[getSharedIdx(w, lane)];
      const unsigned cnt   = sharedCounts[getSharedIdx(w, lane)];
      const unsigned allow = (start < C) ? ((cnt < (C - start)) ? cnt : (C - start)) : 0;
      s += (cnt - allow);
    }
    workerSpill[w] = s;
  }

  // 4) Assign each worker a contiguous slice of the global free list (clamped)
  unsigned scan = 0;
  unsigned remaining = totalFree;
  for (unsigned w = 0; w < NW; ++w) {
    const unsigned take = (workerSpill[w] < remaining) ? workerSpill[w] : remaining;
    sharedOffsets[spillStartIdx(w)] = scan;  // start (inclusive)
    scan += take;
    sharedOffsets[spillEndIdx(w)]   = scan;  // end (exclusive)
    remaining -= take;
  }

  // 5) how much of those spills land in each lane?
  unsigned assigned[kNumLanes] = {0,0,0,0,0};
  accumulateSpillAssigned(sharedOffsets, NW, P, assigned); // sums spill slices via freePrefix

  // 6) Store head shift and shift all primary starts by that amount
  for (unsigned lane = 0; lane < kNumLanes; ++lane) {
    sharedOffsets[laneHeadShiftIdx(lane)] = assigned[lane];   // spill head size
    for (unsigned w = 0; w < NW; ++w) {
      sharedOffsets[getSharedIdx(w, lane)] += assigned[lane]; // move primaries after spills
    }
  }
}

  void routeRays(unsigned workerId) {
    // const unsigned NW = poplar::MultiVertex::numWorkers();
    const unsigned C = kNumRays; //.size()    / sizeof(Ray);

    const Ray* __restrict inBase[kNumLanes] = {
      reinterpret_cast<const Ray*>(childRaysIn0.data()),
      reinterpret_cast<const Ray*>(childRaysIn1.data()),
      reinterpret_cast<const Ray*>(childRaysIn2.data()),
      reinterpret_cast<const Ray*>(childRaysIn3.data()),
      reinterpret_cast<const Ray*>(parentRaysIn.data())
    };
    Ray* __restrict outBase[kNumLanes] = {
      reinterpret_cast<Ray*>(childRaysOut0.data()),
      reinterpret_cast<Ray*>(childRaysOut1.data()),
      reinterpret_cast<Ray*>(childRaysOut2.data()),
      reinterpret_cast<Ray*>(childRaysOut3.data()),
      reinterpret_cast<Ray*>(parentRaysOut.data())
    };
        
    // totals per lane
    unsigned tot[kNumLanes];
    totalsPerLane(sharedCounts, NW, tot);
    LanePartition P = makePartition(tot, C);

    // my spill global range
    unsigned g    = sharedOffsets[spillStartIdx(workerId)];
    unsigned gEnd = sharedOffsets[spillEndIdx(workerId)];
    SpillCursor sc{}; if (g < gEnd) initSpillCursor(P, g, sc);

    // My primary ranges per lane (start/end), kept in registers
    const unsigned s0 = sharedOffsets[getSharedIdx(workerId, 0)];
    const unsigned c0 = sharedCounts [getSharedIdx(workerId, 0)];
    const unsigned e0 = (s0 + c0 < kNumRays ? s0 + c0 : kNumRays);
    const unsigned s1 = sharedOffsets[getSharedIdx(workerId, 1)];
    const unsigned c1 = sharedCounts [getSharedIdx(workerId, 1)];
    const unsigned e1 = (s1 + c1 < kNumRays ? s1 + c1 : kNumRays);
    const unsigned s2 = sharedOffsets[getSharedIdx(workerId, 2)];
    const unsigned c2 = sharedCounts [getSharedIdx(workerId, 2)];
    const unsigned e2 = (s2 + c2 < kNumRays ? s2 + c2 : kNumRays);
    const unsigned s3 = sharedOffsets[getSharedIdx(workerId, 3)];
    const unsigned c3 = sharedCounts [getSharedIdx(workerId, 3)];
    const unsigned e3 = (s3 + c3 < kNumRays ? s3 + c3 : kNumRays);
    const unsigned sP = sharedOffsets[getSharedIdx(workerId, 4)];
    const unsigned cP = sharedCounts [getSharedIdx(workerId, 4)];
    const unsigned eP = (sP + cP < kNumRays ? sP + cP : kNumRays);

    unsigned wr0=0, wr1=0, wr2=0, wr3=0, wrP=0;
    __builtin_assume(kNumRays <= 4096);

    for (rptsize_t lane = 0; lane < (rptsize_t)kNumLanes; ++lane) {
      const Ray* __restrict in = inBase[(unsigned)lane];
      for (rptsize_t i = (rptsize_t)workerId; i < (rptsize_t)kNumRays; i += (rptsize_t)NW) {
        const Ray* ray = &in[(unsigned)i];
        if (__builtin_expect(ray->next_cluster == INVALID_RAY_ID, 0)) break;
        if (__builtin_expect(ray->next_cluster == FINISHED_RAY_ID, 0)) continue;

        const unsigned dst = findChildForCluster(ray->next_cluster);
        bool primaryOK = true;
        unsigned slot = 0;

        switch (dst) {
          case 0: slot = s0 + wr0; primaryOK = (slot < e0); if (primaryOK) { copy_ray(ray, &outBase[0][slot]); ++wr0; } break;
          case 1: slot = s1 + wr1; primaryOK = (slot < e1); if (primaryOK) { copy_ray(ray, &outBase[1][slot]); ++wr1; } break;
          case 2: slot = s2 + wr2; primaryOK = (slot < e2); if (primaryOK) { copy_ray(ray, &outBase[2][slot]); ++wr2; } break;
          case 3: slot = s3 + wr3; primaryOK = (slot < e3); if (primaryOK) { copy_ray(ray, &outBase[3][slot]); ++wr3; } break;
          default:slot = sP + wrP; primaryOK = (slot < eP); if (primaryOK) { copy_ray(ray, &outBase[4][slot]); ++wrP; } break;
        }

        if (!primaryOK && (g < gEnd)) {
          spillAdvanceIfNeeded(P, g, sc);
          // const unsigned slotInR = P.base[sc.r] + (g - P.freePrefix[sc.r]);
          const unsigned slotInR = (g - P.freePrefix[sc.r]);
          copy_ray(ray, &outBase[sc.r][slotInR]);
          ++g;
        }

      }
    }
  }

  [[gnu::always_inline]]
  void wipeTail(Ray* __restrict out, unsigned written) {
    if (written >= kNumRays) return;
    for (rptsize_t i = written; i < (rptsize_t)kNumRays; ++i) {
      Ray* rr = &out[(unsigned)i];
      if (rr->next_cluster == INVALID_RAY_ID) break;
      rr->next_cluster = INVALID_RAY_ID;
    }
  }

  void invalidateAfterRouting() {
    // totals + partition
    unsigned tot[kNumLanes];
    totalsPerLane(sharedCounts, NW, tot);
    LanePartition P = makePartition(tot, kNumRays);

    // how much spill each lane received
    unsigned assigned[kNumLanes] = {0,0,0,0,0};
    for (unsigned w = 0; w < NW; ++w) {
      unsigned a = sharedOffsets[spillStartIdx(w)];
      unsigned b = sharedOffsets[spillEndIdx(w)];
      unsigned r = 0;
      while (a < b) {
        while (a >= P.freePrefix[r + 1]) ++r;
        unsigned R = P.freePrefix[r + 1];
        unsigned take = ((b < R) ? b : R) - a;
        assigned[r] += take;
        a += take;
      }
    }
    
    Ray* __restrict outBase[kNumLanes] = {
      reinterpret_cast<Ray*>(childRaysOut0.data()),
      reinterpret_cast<Ray*>(childRaysOut1.data()),
      reinterpret_cast<Ray*>(childRaysOut2.data()),
      reinterpret_cast<Ray*>(childRaysOut3.data()),
      reinterpret_cast<Ray*>(parentRaysOut.data())
    };
    // wipe tails past written = base + assigned
    for (unsigned lane = 0; lane < kNumLanes; ++lane) {
      const unsigned written = P.base[lane] + assigned[lane];
      wipeTail(outBase[lane], written);
    }
  }

  [[gnu::always_inline]] 
  void writeDebugInfos() {
    unsigned inCnt[6] = {0,0,0,0,0,0}; 
    unsigned outCnt2[5] = {0,0,0,0,0}; 
    const Ray* __restrict inBase[kNumLanes] = {
      reinterpret_cast<const Ray*>(childRaysIn0.data()),
      reinterpret_cast<const Ray*>(childRaysIn1.data()),
      reinterpret_cast<const Ray*>(childRaysIn2.data()),
      reinterpret_cast<const Ray*>(childRaysIn3.data()),
      reinterpret_cast<const Ray*>(parentRaysIn.data())
    };

    for (unsigned lane = 0; lane < kNumLanes; ++lane) {
      const Ray* __restrict in = inBase[(unsigned)lane];

      for (unsigned i = 0; i < kNumRays; i ++) {
        // const Ray *ray = reinterpret_cast<const Ray *>(buf.data() + i * sizeof(Ray));
        const Ray* ray = &in[i];
        if (ray->next_cluster == FINISHED_RAY_ID) {
          inCnt[5]++;
          continue;
        }
        if (ray->next_cluster == INVALID_RAY_ID) break;

        unsigned dst = findChildForCluster(ray->next_cluster);  // 0..4
        outCnt2[dst]++;
        inCnt[lane]++;
      }
    }

    unsigned* dbg = reinterpret_cast<unsigned*>(debugBytes.data());

    dbg[0] = inCnt[0];
    dbg[1] = inCnt[1];
    dbg[2] = inCnt[2];
    dbg[3] = inCnt[3];
    dbg[4] = inCnt[4];
    dbg[5] = inCnt[5];
    dbg[6] = outCnt2[0];
    dbg[7] = outCnt2[1];
    dbg[8] = outCnt2[2];
    dbg[9] = outCnt2[3];
    dbg[10] = outCnt2[4];
  }

};

class RayRouter2 : public poplar::MultiVertex {
  using RayVecIn = poplar::Input< poplar::Vector<uint8_t, poplar::VectorLayout::ONE_PTR, 8> >;
  using RayVecIO = poplar::InOut< poplar::Vector<uint8_t, poplar::VectorLayout::ONE_PTR, 8> >;

public:
  // Incoming rays (5 lanes)
  RayVecIn  parentRaysIn, childRaysIn0, childRaysIn1, childRaysIn2, childRaysIn3;
  // Outgoing rays (5 lanes)
  RayVecIO  parentRaysOut, childRaysOut0, childRaysOut1, childRaysOut2, childRaysOut3;

  // Mapping info
  poplar::Input< poplar::Vector<unsigned short, poplar::VectorLayout::SPAN, 4> > childClusterIds;
  poplar::Input<uint8_t> level;

  // Shared scratch
  poplar::InOut< poplar::Vector<unsigned> > sharedCounts; 
  poplar::InOut< poplar::Vector<unsigned> > sharedOffsets; 
  poplar::InOut< poplar::Vector<unsigned> > readyFlags; 
  poplar::InOut< poplar::Vector<unsigned> > debugBytes; // optional

  inline static constexpr unsigned kNumLanes = 5;
  inline static constexpr unsigned PARENT    = 4;
  inline static constexpr unsigned NW        = 6; // MultiVertex workers
  // NOTE: assume capacity per lane == kNumRays
  inline static constexpr unsigned kNumRays  = /* your capacity here */ 1024;

  struct LanePartition {
    unsigned base[kNumLanes];
    unsigned freePrefix[kNumLanes + 1];
  };

  uint16_t myChildPrefix;
  uint8_t  shift;

  // ---- aliasing constraints (unchanged) ----
  [[poplar::constraint("elem(*readyFlags)!=elem(*sharedCounts)")]]
  [[poplar::constraint("elem(*readyFlags)!=elem(*sharedOffsets)")]]
  [[poplar::constraint("elem(*sharedCounts)!=elem(*sharedOffsets)")]]
  // (add others you had if needed)

  bool compute(unsigned workerId) {
    // derive prefix mapping
    shift = *level * 2;
    myChildPrefix = childClusterIds[0] >> (shift + 2);

    if (workerId == 0) {
      volatile unsigned* f = readyFlags.data();
      for (unsigned i = 0; i < NW; ++i) f[i] = 0;
    }
    barrier(0, workerId);

    countRays(workerId);
    barrier(1, workerId);

    if (workerId == 0) computeWriteOffsets();
    barrier(2, workerId);

    routeRays(workerId);
    barrier(3, workerId);

    if (workerId == 0) invalidateAfterRouting();
    return true;
  }

private:
  // ---------- helpers ----------
  [[gnu::always_inline]] unsigned getSharedIdx(unsigned workerId, unsigned lane) const {
    return workerId * kNumLanes + lane;
  }
  [[gnu::always_inline]] unsigned spillStartIdx(unsigned w) const { return NW * kNumLanes + w; }
  [[gnu::always_inline]] unsigned spillEndIdx  (unsigned w) const { return NW * kNumLanes + NW + w; }

  [[gnu::always_inline]] 
  static void workerRange(unsigned workerId, unsigned &begin, unsigned &end) {
    begin = (kNumRays * workerId) / NW;
    end   = (kNumRays * (workerId + 1)) / NW;
  }

  [[gnu::always_inline]]
  static inline void copy_ray(const Ray* __restrict src, Ray* __restrict dst) {
    // 24 bytes -> 6 x 32-bit words
    const unsigned* __restrict s = reinterpret_cast<const unsigned*>(src);
    unsigned* d = reinterpret_cast<unsigned*>(dst);  // NOTE: no __restrict here

    // exact unroll helps codegen
    ipu::store_postinc(&d, s[0], 1);
    ipu::store_postinc(&d, s[1], 1);
    ipu::store_postinc(&d, s[2], 1);
    ipu::store_postinc(&d, s[3], 1);
    ipu::store_postinc(&d, s[4], 1);
    ipu::store_postinc(&d, s[5], 1);
  }

  [[gnu::always_inline]]
  unsigned findChildForCluster (uint16_t cluster_id) const {
    unsigned childIdx = (cluster_id >> shift) & 0x3;
    bool isChild = ((cluster_id >> (shift + 2)) == myChildPrefix);
    return isChild ? childIdx : PARENT;
  }

  [[gnu::always_inline]]
  void totalsPerLane(const poplar::Vector<unsigned>& cnt, unsigned outTot[kNumLanes]) const {
    for (unsigned r = 0; r < kNumLanes; ++r) outTot[r] = 0;
    for (unsigned r = 0; r < kNumLanes; ++r)
      for (unsigned w = 0; w < NW; ++w)
        outTot[r] += cnt[getSharedIdx(w, r)];
  }

  [[gnu::always_inline]]
  static LanePartition makePartition(const unsigned tot[kNumLanes]) {
    LanePartition P{};
    P.freePrefix[0] = 0;
    for (unsigned r = 0; r < kNumLanes; ++r) {
      const unsigned used = (tot[r] < kNumRays) ? tot[r] : kNumRays;
      P.base[r] = used;
      const unsigned freeCap = kNumRays - used;
      P.freePrefix[r + 1] = P.freePrefix[r] + freeCap;
    }
    return P;
  }

  struct SpillCursor { unsigned r; unsigned nextBoundary; };

  [[gnu::always_inline]]
  static void initSpillCursor(const LanePartition& P, unsigned g, SpillCursor& sc) {
    sc.r = 0;
    while (g >= P.freePrefix[sc.r + 1]) ++sc.r;
    sc.nextBoundary = P.freePrefix[sc.r + 1];
  }

  [[gnu::always_inline]]
  static void spillAdvanceIfNeeded(const LanePartition& P, unsigned g, SpillCursor& sc) {
    if (g >= sc.nextBoundary) {
      do { ++sc.r; } while (g >= P.freePrefix[sc.r + 1]);
      sc.nextBoundary = P.freePrefix[sc.r + 1];
    }
  }

  // ---------- phases ----------
  void countRays(unsigned workerId) {
    // zero my row
    for (unsigned lane = 0; lane < kNumLanes; ++lane)
      sharedCounts[getSharedIdx(workerId, lane)] = 0;

    // typed bases (if you kept uint8_t ABI, reinterpret_cast to Ray* once)
    const Ray* __restrict inBase[kNumLanes] = {
      reinterpret_cast<const Ray*>(childRaysIn0.data()),
      reinterpret_cast<const Ray*>(childRaysIn1.data()),
      reinterpret_cast<const Ray*>(childRaysIn2.data()),
      reinterpret_cast<const Ray*>(childRaysIn3.data()),
      reinterpret_cast<const Ray*>(parentRaysIn.data())
    };

    unsigned begin, end; workerRange(workerId, begin, end);
    __builtin_assume(kNumRays <= 4096);

    for (unsigned lane = 0; lane < kNumLanes; ++lane) {
      const Ray* __restrict in = inBase[lane];
      for (rptsize_t i = begin; i < (rptsize_t)end; ++i) {
        const Ray* ray = &in[(unsigned)i];
        if (__builtin_expect(ray->next_cluster == INVALID_RAY_ID, 0)) break;
        if (__builtin_expect(ray->next_cluster == FINISHED_RAY_ID, 0)) continue;
        const unsigned dst = findChildForCluster(ray->next_cluster);
        ++sharedCounts[getSharedIdx(workerId, dst)];
      }
    }
  }

  void computeWriteOffsets() {
    // primary offsets (no spill yet)
    for (unsigned lane = 0; lane < kNumLanes; ++lane) {
      unsigned offset = 0;
      for (unsigned w = 0; w < NW; ++w) {
        unsigned count = sharedCounts[getSharedIdx(w, lane)];
        sharedOffsets[getSharedIdx(w, lane)] = offset;
        offset += count;
      }
    }

    // free space partition
    unsigned tot[kNumLanes];
    totalsPerLane(sharedCounts, tot);
    LanePartition P = makePartition(tot);
    const unsigned totalFree = P.freePrefix[kNumLanes];

    // compute per-worker spill need
    unsigned workerSpill[NW];
    for (unsigned w = 0; w < NW; ++w) {
      unsigned s = 0;
      for (unsigned lane = 0; lane < kNumLanes; ++lane) {
        const unsigned start = sharedOffsets[getSharedIdx(w, lane)];
        const unsigned cnt   = sharedCounts [getSharedIdx(w, lane)];
        const unsigned allow = (start < kNumRays) ? ((cnt < (kNumRays - start)) ? cnt : (kNumRays - start)) : 0;
        s += (cnt - allow);
      }
      workerSpill[w] = s;
    }

    // assign slices of global free space
    unsigned scan = 0, remaining = totalFree;
    for (unsigned w = 0; w < NW; ++w) {
      const unsigned take = (workerSpill[w] < remaining) ? workerSpill[w] : remaining;
      sharedOffsets[spillStartIdx(w)] = scan;
      scan += take;
      sharedOffsets[spillEndIdx(w)]   = scan;
      remaining -= take;
    }
  }

  void routeRays(unsigned workerId) {
    // typed bases
    const Ray* __restrict inBase[kNumLanes] = {
      reinterpret_cast<const Ray*>(childRaysIn0.data()),
      reinterpret_cast<const Ray*>(childRaysIn1.data()),
      reinterpret_cast<const Ray*>(childRaysIn2.data()),
      reinterpret_cast<const Ray*>(childRaysIn3.data()),
      reinterpret_cast<const Ray*>(parentRaysIn.data())
    };
    Ray* __restrict outBase[kNumLanes] = {
      reinterpret_cast<Ray*>(childRaysOut0.data()),
      reinterpret_cast<Ray*>(childRaysOut1.data()),
      reinterpret_cast<Ray*>(childRaysOut2.data()),
      reinterpret_cast<Ray*>(childRaysOut3.data()),
      reinterpret_cast<Ray*>(parentRaysOut.data())
    };

    // partition + spill
    unsigned tot[kNumLanes]; totalsPerLane(sharedCounts, tot);
    LanePartition P = makePartition(tot);

    unsigned g    = sharedOffsets[spillStartIdx(workerId)];
    unsigned gEnd = sharedOffsets[spillEndIdx(workerId)];
    SpillCursor sc{}; if (g < gEnd) initSpillCursor(P, g, sc);

    // my primary ranges per lane (start/end)
    const unsigned s0 = sharedOffsets[getSharedIdx(workerId, 0)];
    const unsigned c0 = sharedCounts [getSharedIdx(workerId, 0)];
    const unsigned e0 = (s0 + c0 < kNumRays ? s0 + c0 : kNumRays);
    const unsigned s1 = sharedOffsets[getSharedIdx(workerId, 1)];
    const unsigned c1 = sharedCounts [getSharedIdx(workerId, 1)];
    const unsigned e1 = (s1 + c1 < kNumRays ? s1 + c1 : kNumRays);
    const unsigned s2 = sharedOffsets[getSharedIdx(workerId, 2)];
    const unsigned c2 = sharedCounts [getSharedIdx(workerId, 2)];
    const unsigned e2 = (s2 + c2 < kNumRays ? s2 + c2 : kNumRays);
    const unsigned s3 = sharedOffsets[getSharedIdx(workerId, 3)];
    const unsigned c3 = sharedCounts [getSharedIdx(workerId, 3)];
    const unsigned e3 = (s3 + c3 < kNumRays ? s3 + c3 : kNumRays);
    const unsigned sP = sharedOffsets[getSharedIdx(workerId, 4)];
    const unsigned cP = sharedCounts [getSharedIdx(workerId, 4)];
    const unsigned eP = (sP + cP < kNumRays ? sP + cP : kNumRays);

    unsigned wr0=0, wr1=0, wr2=0, wr3=0, wrP=0;
    unsigned begin, end; workerRange(workerId, begin, end);
    __builtin_assume(kNumRays <= 4096);

    for (rptsize_t lane = 0; lane < (rptsize_t)kNumLanes; ++lane) {
      const Ray* __restrict in = inBase[(unsigned)lane];
      for (rptsize_t i = begin; i < (rptsize_t)end; ++i) {
        const Ray* ray = &in[(unsigned)i];
        if (__builtin_expect(ray->next_cluster == INVALID_RAY_ID, 0)) break;
        if (__builtin_expect(ray->next_cluster == FINISHED_RAY_ID, 0)) continue;

        const unsigned dst = findChildForCluster(ray->next_cluster);
        bool primaryOK = true;
        unsigned slot = 0;

        switch (dst) {
          case 0: slot = s0 + wr0; primaryOK = (slot < e0); if (primaryOK) { copy_ray(ray, &outBase[0][slot]); ++wr0; } break;
          case 1: slot = s1 + wr1; primaryOK = (slot < e1); if (primaryOK) { copy_ray(ray, &outBase[1][slot]); ++wr1; } break;
          case 2: slot = s2 + wr2; primaryOK = (slot < e2); if (primaryOK) { copy_ray(ray, &outBase[2][slot]); ++wr2; } break;
          case 3: slot = s3 + wr3; primaryOK = (slot < e3); if (primaryOK) { copy_ray(ray, &outBase[3][slot]); ++wr3; } break;
          default:slot = sP + wrP; primaryOK = (slot < eP); if (primaryOK) { copy_ray(ray, &outBase[4][slot]); ++wrP; } break;
        }

        if (!primaryOK && (g < gEnd)) {
          spillAdvanceIfNeeded(P, g, sc);
          const unsigned slotInR = P.base[sc.r] + (g - P.freePrefix[sc.r]);
          copy_ray(ray, &outBase[sc.r][slotInR]);
          ++g;
        }
      }
    }
  }

  [[gnu::always_inline]]
  void wipeTail(Ray* __restrict out, unsigned written) {
    if (written >= kNumRays) return;
    for (rptsize_t i = written; i < (rptsize_t)kNumRays; ++i) {
      Ray* rr = &out[(unsigned)i];
      if (rr->next_cluster == INVALID_RAY_ID) break;
      rr->next_cluster = INVALID_RAY_ID;
    }
  }

  void invalidateAfterRouting() {
    // totals + partition
    unsigned tot[kNumLanes];
    totalsPerLane(sharedCounts, tot);
    LanePartition P = makePartition(tot);

    // how much spill each lane received
    unsigned assigned[kNumLanes] = {0,0,0,0,0};
    for (unsigned w = 0; w < NW; ++w) {
      unsigned a = sharedOffsets[spillStartIdx(w)];
      unsigned b = sharedOffsets[spillEndIdx(w)];
      unsigned r = 0;
      while (a < b) {
        while (a >= P.freePrefix[r + 1]) ++r;
        unsigned R = P.freePrefix[r + 1];
        unsigned take = ((b < R) ? b : R) - a;
        assigned[r] += take;
        a += take;
      }
    }

    Ray* __restrict outBase[kNumLanes] = {
      reinterpret_cast<Ray*>(childRaysOut0.data()),
      reinterpret_cast<Ray*>(childRaysOut1.data()),
      reinterpret_cast<Ray*>(childRaysOut2.data()),
      reinterpret_cast<Ray*>(childRaysOut3.data()),
      reinterpret_cast<Ray*>(parentRaysOut.data())
    };
    // wipe tails past written = base + assigned
    for (unsigned lane = 0; lane < kNumLanes; ++lane) {
      const unsigned written = P.base[lane] + assigned[lane];
      wipeTail(outBase[lane], written);
    }
  }

  [[gnu::always_inline]]
  void barrier(unsigned phase, unsigned workerId) {
    volatile unsigned* flags = readyFlags.data();
    flags[workerId] = phase;
    asm volatile("" ::: "memory");
    while (true) {
      bool all = true;
      for (unsigned i = 0; i < NW; ++i) if (flags[i] < phase) { all = false; break; }
      if (all) break;
    }
    asm volatile("" ::: "memory");
  }
};
