#include <poplar/Vertex.hpp>

#include <glm/mat4x4.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <geometry/primitives.hpp>
#include <ipu/rf_config.hpp>
#include <ipu/ipu_primitives.hpp>
#include <ipudef.h>
#include <ipu_builtins.h>
#include <ipu_vector_math>
#include <ipu_memory_intrinsics>

// __builtin_assume(kNumRays <= 4096);

using namespace radfoam::config;
using radfoam::geometry::LocalPoint;
using radfoam::geometry::GenericPoint;
using radfoam::geometry::FinishedRay;
using radfoam::geometry::FinishedPixel;
using ipu::geometry::Ray;

class RayTracer : public poplar::MultiVertex {
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
      out->d = in->d;
      
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
            p->t = __builtin_ipu_f16tof32(out->d);
            ++consumedFinishedOnThisTile;
          }
          *out = *in;
          out->next_cluster = FINISHED_RAY_ID;
        } else {
          *out = *in;
        }
        continue;
      }

      // --- march ---
      glm::vec3 rayDir = computeRayDir(in->x, y_coord, invProj, invView);

      glm::vec3 color(in->r, in->g, in->b);
      float transmittance = in->transmittance; //__builtin_ipu_f16tof32(in->transmittance);
      float t0    = in->t; //__builtin_ipu_f16tof32(in->t);

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

        const float transmittance0 = transmittance;
        const float t_seg_start   = t0;

        const float delta = closestT - t0;
        const float alpha = 1.f - __builtin_ipu_exp(-cur->density * delta);
        const float ta    = transmittance * alpha;

        color.x += ta * (cur->r / 255.f);
        color.y += ta * (cur->g / 255.f);
        color.z += ta * (cur->b / 255.f);
        transmittance   *= (1.f - alpha);
        t0       = __builtin_ipu_max(t0, closestT);

        float depth_quantile = 0.5f;
        float transmittance_threshold = 0.01f;
        
        // if ((transmittance0 > depth_quantile) && (transmittance <= depth_quantile)) {
          //   const float t_cross = t_seg_start + (1.f / cur->density) * __builtin_ipu_ln(transmittance0 / depth_quantile);
        //   out->d = t_cross;
        // } 
        if ((transmittance0 > depth_quantile) && (transmittance <= depth_quantile) && cur->density > 0 && in->d == 0) {
          const float ratio   = __builtin_ipu_max(1e-6f, transmittance0 / depth_quantile);
          float t_cross = t_seg_start + (1.f / cur->density) * __builtin_ipu_ln(ratio);
          // numerical safety: keep inside [segment start, segment end]
          t_cross = __builtin_ipu_min(__builtin_ipu_max(t_cross, t_seg_start), closestT);
          out->d =  __builtin_ipu_f32tof16(t_cross);
        }
        
        // Finish or cross boundary?
        if (transmittance < transmittance_threshold || next == -1 || next >= nLocalPts) {
          if (transmittance < transmittance_threshold || next == -1) {
            // Finished on this tile
            out->x = in->x; 
            out->y = packY(y_coord, 2);
            out->r = color.x; 
            out->g = color.y; 
            out->b = color.z;
            out->t = closestT; //__builtin_ipu_f32tof16(closestT);
            // out->d = closestT;
            out->transmittance = transmittance; //__builtin_ipu_f32tof16(transmittance);
            out->next_cluster  = tileOfXY(in->x, y_coord);   // << route to FB owner tile
            out->next_local  = FINISHED_RAY_ID;
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
            out->t = t0; //__builtin_ipu_f32tof16(t0);
            out->transmittance = transmittance; //__builtin_ipu_f32tof16(transmittance); //__builtin_ipu_max(0.f, __builtin_ipu_min(1.f, transmittance)));
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
