#include <poplar/Vertex.hpp>

#include <glm/mat4x4.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <geometry/primitives.hpp>
#include <ipu/rf_config.hpp>
#include <ipudef.h>
#include <ipu_builtins.h>

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

inline constexpr uint16_t kLeadMask   = 0xFC00u;   // 11111 00000000000₂
inline constexpr uint8_t kShift = 10;  
inline constexpr uint16_t INVALID_RAY_ID = 0xFFFF;
inline constexpr uint16_t FINISHED_RAY_ID = 0xFFFE;

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

  // InOut
  poplar::InOut<poplar::Vector<uint8_t>> framebuffer;
  poplar::InOut<unsigned>       exec_count;          
  poplar::InOut<unsigned>       finishedWriteOffset; 

  poplar::InOut<poplar::Vector<unsigned>> sharedCounts;
  poplar::InOut<poplar::Vector<unsigned>> sharedOffsets;
  poplar::InOut<poplar::Vector<unsigned>> readyFlags;

  // ---- aliasing guards ----
  [[poplar::constraint("elem(*raysIn)!=elem(*raysOut)")]]
  [[poplar::constraint("elem(*raysIn)!=elem(*finishedRays)")]]
  [[poplar::constraint("elem(*raysOut)!=elem(*finishedRays)")]]

  bool compute(unsigned workerId) {
    const unsigned NW = poplar::MultiVertex::numWorkers();
    const unsigned C  = raysOut.size() / sizeof(Ray);
    const unsigned FR_CAP = finishedRays.size() / sizeof(FinishedRay);
    const uint16_t nLocalPts = local_pts.size() / sizeof(LocalPoint);
    const uint16_t nNbrPts   = neighbor_pts.size() / sizeof(GenericPoint);
    const unsigned adjSize   = adjacency.size(); // elements (uint16)

    // Per-worker fixed slice in finishedRays (even split; last worker takes remainder)
    const unsigned basePer = FR_CAP / NW;
    const unsigned myStart = workerId * basePer; // + (workerId < rem ? workerId : rem);
    const unsigned myCount = 0; //basePer + (workerId < rem ? 1u : 0u);
    const unsigned myEnd   = (workerId+1) * basePer; //(myStart + myCount <= FR_CAP) ? (myStart + myCount) : FR_CAP;
    unsigned       myWrite = myStart; // next write index within finishedRays

    // Clear only my finished slice for this frame
    // for (unsigned i = myStart; i < myEnd; ++i) {
    //   FinishedRay* fr = reinterpret_cast<FinishedRay*>(finishedRays.data() + i*sizeof(FinishedRay));
    //   fr->x = INVALID_RAY_ID;
    // }
    
    // for (unsigned idx = workerId; idx < C; idx += NW) {
    //   auto *ro = reinterpret_cast<Ray*>(raysOut.data() + idx*sizeof(Ray));
    //   ro->next_cluster = INVALID_RAY_ID;
    // }


    // Setup matrices
    glm::mat4 invView = glm::make_mat4(view_matrix.data());
    glm::mat4 invProj = glm::make_mat4(projection_matrix.data());
    glm::vec3 rayOrigin = glm::vec3(invView[3]);
    const uint16_t myTile = *tile_id;

    unsigned localFinished = 0;
    unsigned localMaxSeenPlus1 = 0;

    // Track last index we actually touched in our stripe to clean the tail later
    // unsigned out_ray_cntr  = 0;
    uint16_t lastSeenIdx = 65533;

    // ---- Pass: march rays; write back to SAME index in raysOut; record finished into my slice ----
    for (uint16_t idx = workerId; idx < C; idx += NW) {
      const Ray* in = readStructAt<Ray>(raysIn, idx);
      if (in->next_cluster == INVALID_RAY_ID) break; 
      lastSeenIdx = idx;      
      localMaxSeenPlus1 = idx + 1;
      Ray* out = reinterpret_cast<Ray*>(raysOut.data() + idx*sizeof(Ray));
      
      // Spillover passthrough (ray belongs to another tile)
      if (in->next_cluster != myTile) {
        std::memcpy(out, in, sizeof(Ray));
        // out_ray_cntr++;
        continue;
      }

      // --- march ---
      const uint16_t x10 = data10(in->x);
      glm::vec3 rayDir = computeRayDir(x10, in->y, invProj, invView);

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
      unsigned step = 0;

      while (!finished) {
        ++step;
        if (current < 0 || current >= nLocalPts) { finished = true; break; }
        const LocalPoint* cur = readStructAt<LocalPoint>(local_pts, current);
        glm::vec3 p0(cur->x, cur->y, cur->z);

        uint16_t adjStart = (current==0) ? 0 : readStructAt<LocalPoint>(local_pts, current-1)->adj_end;
        uint16_t adjEnd   = cur->adj_end;
        if (adjStart > adjEnd) adjStart = adjEnd;
        if (adjEnd   > adjSize) adjEnd  = adjSize;

        float closestT = std::numeric_limits<float>::max();
        int   next     = -1;

        for (uint16_t j = adjStart; j < adjEnd; ++j) {
          const uint16_t nbrIdx = adjacency[j];
          glm::vec3 p1;
          if (nbrIdx < local_pts.size()/sizeof(LocalPoint)) {
            const LocalPoint* nb = readStructAt<LocalPoint>(local_pts, nbrIdx);
            p1 = glm::vec3(nb->x, nb->y, nb->z);
          } else {
            const uint16_t off = nbrIdx - nLocalPts;
            if (off >= nNbrPts) { next = -1; continue; }
            const GenericPoint* nb = readStructAt<GenericPoint>(neighbor_pts, off);
            p1 = glm::vec3(nb->x, nb->y, nb->z);
          }

          const glm::vec3 faceNormal = p1 - p0;
          const glm::vec3 faceOrigin = p0 + 0.5f*faceNormal;
          const float dn = glm::dot(faceNormal, rayDir);
          if (dn <= 0.f) continue;

          const float t = glm::dot(faceOrigin - rayOrigin, faceNormal) / dn;
          if (t > 0.f && t < closestT) { closestT = t; next = nbrIdx; }
        }

        const float delta = closestT - t0;
        const float alpha = 1.f - __builtin_ipu_exp(-cur->density * delta);
        const float ta    = transmittance * alpha;

        color.x += ta * (cur->r / 255.f);
        color.y += ta * (cur->g / 255.f);
        color.z += ta * (cur->b / 255.f);
        transmittance   *= (1.f - alpha);
        t0       = __builtin_ipu_max(t0, closestT);

        // Finish or cross boundary?
        const uint16_t nLocalPts = local_pts.size()/sizeof(LocalPoint);
        if (transmittance < 0.01f || next == -1 || next >= nLocalPts) {
          if (transmittance < 0.01f || next == -1) {
            // Finished on this tile
            out->x = in->x; 
            out->y = in->y;
            out->r = color.x; 
            out->g = color.y; 
            out->b = color.z;
            out->t = __builtin_ipu_f32tof16(t0);
            out->transmittance = __builtin_ipu_f32tof16(transmittance);
            out->next_cluster  = FINISHED_RAY_ID;   // router will skip
            out->next_local    = 0;

            // Emit into my finished slice if space remains
            if (myWrite < myEnd) {
              FinishedRay* fr = reinterpret_cast<FinishedRay*>(
                  finishedRays.data() + myWrite*sizeof(FinishedRay));
              const uint16_t x10w = data10(in->x);
              fr->x = x10w;
              fr->y = in->y;
              fr->r = clampToU8(color.x);
              fr->g = clampToU8(color.y);
              fr->b = clampToU8(color.z);
              fr->t = t0;
              ++myWrite;
              ++localFinished;
            }
          } else {
            // Cross to neighbor cluster
            const uint16_t nLocal = local_pts.size()/sizeof(LocalPoint);
            const GenericPoint* nb = readStructAt<GenericPoint>(neighbor_pts, next - nLocal);
            out->x = in->x; 
            out->y = in->y;
            out->r = color.x; 
            out->g = color.y; 
            out->b = color.z;
            out->t = __builtin_ipu_f32tof16(t0);
            out->transmittance = __builtin_ipu_f32tof16(__builtin_ipu_max(0.f, __builtin_ipu_min(1.f, transmittance)));
            out->next_cluster  = nb->cluster_id;
            out->next_local    = nb->local_id;
          }
          finished = true;
          break;
        }

        // Safety cap to avoid pathological loops
        // if (step >= 120) finished = true;
        current = next;
      } // while(!finished)
    } // for stripe

    // sharedCounts[workerId]  = localFinished;
    // barrier(0, workerId);

    // if(workerId == 0) {
    //   uint16_t totalFinished = 0;
    //   for (unsigned w = 0; w < NW; ++w) {
    //     totalFinished += sharedCounts[w];
    //   }
    // }
    // barrier(1, workerId);
    // Invalidate the remainder of my raysOut stripe to avoid stale entries
    unsigned startTail = (lastSeenIdx == 65533) ? workerId : (static_cast<unsigned>(lastSeenIdx) + NW);
    for (unsigned idx = startTail; idx < C; idx += NW) {
      Ray* ray = reinterpret_cast<Ray*>(raysOut.data() + idx*sizeof(Ray));
      if (ray->next_cluster == INVALID_RAY_ID) break;
      ray->next_cluster = INVALID_RAY_ID;
    }

    for (unsigned i = myWrite; i < myEnd; ++i) {
      FinishedRay* ray = reinterpret_cast<FinishedRay*>(finishedRays.data() + i*sizeof(FinishedRay));
      if (ray->x == INVALID_RAY_ID) break;
      ray->x = INVALID_RAY_ID;
    }

    // Optional debug
    if (workerId == 0) {
      *exec_count = *exec_count + 1;
      *result_u16  = localFinished; 
      // *result_float = static_cast<float>(*exec_count);
      // if(lastSeenIdx != 65533)
      //   framebuffer[0] = lastSeenIdx;
      // else
      //   framebuffer[0] = 0;
    }
    return true;
  }

private:
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
  
  // [[gnu::always_inline]]
  void barrier2(unsigned phase, unsigned workerId) {
    volatile unsigned* flags = readyFlags.data();
    const unsigned NW = poplar::MultiVertex::numWorkers();
    asm volatile("" ::: "memory");
    while (true) {
      flags[workerId] = phase;
      bool all = true;
      for (unsigned i = 0; i < NW; ++i) if (flags[i] != phase) { all = false; break; }
      if (all) break;
    }
    asm volatile("" ::: "memory");
  }

  template <typename T>
  [[gnu::always_inline]] inline const T* readStructAt(const poplar::Input<poplar::Vector<uint8_t>>& buffer, std::size_t index) {
    constexpr std::size_t stride = sizeof(T);
    const uint8_t* base = buffer.data() + index * stride;
    return reinterpret_cast<const T*>(base);
  }

  [[gnu::always_inline]] inline uint8_t clampToU8(float v) {
    return static_cast<uint8_t>(__builtin_ipu_max(0.0f, __builtin_ipu_min(255.0f, std::round(v * 255.0f))));
  }

  [[gnu::always_inline]] inline uint8_t getLead6(uint16_t v) { 
    return v >> kShift; 
  }

  [[gnu::always_inline]] inline uint16_t setLead6(const uint16_t &v, uint8_t x) {
    return (v & ~kLeadMask) | ((x & 0x3F) << kShift); 
  }

  [[gnu::always_inline]] inline uint16_t data10(uint16_t v) {
    return v & ~kLeadMask; 
  }

};

class RayGen : public poplar::Vertex {
public:
  poplar::Input<poplar::Vector<uint8_t>>  RaysIn;   // from L4 parentOut (spillovers)
  poplar::Output<poplar::Vector<uint8_t>> RaysOut;  // to L4 parentIn

  poplar::InOut<poplar::Vector<uint8_t>>  pendingRays; 
  poplar::InOut<unsigned>                 pendingHead;   // [0..P-1]
  poplar::InOut<unsigned>                 pendingTail;   // [0..P-1]

  poplar::InOut<unsigned> exec_count;
  poplar::Input<poplar::Vector<uint8_t>> camera_cell_info;
  poplar::Output<poplar::Vector<unsigned>> debugBytes;

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

    // ── 3) fill remaining with generated rays (simple column scan), queue rest
    const unsigned interval = 2;
    const bool genThisFrame = ((*exec_count % interval) == 0);
    if (genThisFrame) {
      const unsigned nColsPerFrame = 4;
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

template <unsigned Port>
[[gnu::always_inline]] 
inline void emit(const Ray &r, unsigned &cnt,
          poplar::Output<poplar::Vector<uint8_t>>* const (&outPorts)[5])
{
    // Port is a compile-time constant, so the expression below
    // becomes a constant address + a run-time offset.
    std::memcpy(outPorts[Port]->data() + cnt * sizeof(Ray), &r, sizeof(Ray));
    ++cnt;
}

class RayRouter : public poplar::MultiVertex {
public:
  // Incoming rays
  poplar::Input<poplar::Vector<uint8_t>> parentRaysIn;
  poplar::Input<poplar::Vector<uint8_t>> childRaysIn0;
  poplar::Input<poplar::Vector<uint8_t>> childRaysIn1;
  poplar::Input<poplar::Vector<uint8_t>> childRaysIn2;
  poplar::Input<poplar::Vector<uint8_t>> childRaysIn3;

  // Outgoing rays
  poplar::InOut<poplar::Vector<uint8_t>> parentRaysOut;
  poplar::InOut<poplar::Vector<uint8_t>> childRaysOut0;
  poplar::InOut<poplar::Vector<uint8_t>> childRaysOut1;
  poplar::InOut<poplar::Vector<uint8_t>> childRaysOut2;
  poplar::InOut<poplar::Vector<uint8_t>> childRaysOut3;

  // ID mapping to know which cluster IDs belong to which child
  poplar::Input<poplar::Vector<unsigned short>> childClusterIds;
  poplar::Input<uint8_t> level;

  poplar::InOut<poplar::Vector<unsigned>> debugBytes;

  poplar::InOut<poplar::Vector<unsigned>> sharedCounts; 
  poplar::InOut<poplar::Vector<unsigned>> sharedOffsets; 
  poplar::InOut<poplar::Vector<unsigned>> readyFlags; 

  static constexpr unsigned kNumLanes = 5;
  // static constexpr unsigned kNumWorkers = 6; //MultiVertex::numWorkers(); 
  static constexpr unsigned PARENT = 4;

  struct LanePartition {
    unsigned base[kNumLanes];             // primary writes per lane, clamped to capacity C
    unsigned freePrefix[kNumLanes + 1];   // prefix-scan of free capacities across lanes
  };

  uint16_t myChildPrefix;
  uint8_t shift;

  // ---- Forbid any Input -> Output aliasing (5 x 5 = 25) ----
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


  // ---- Forbid any Output -> Output aliasing (C(5,2) = 10) ----
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
    const unsigned NW = poplar::MultiVertex::numWorkers();

    if (workerId == 0) {
      volatile unsigned* f = readyFlags.data();
      for (unsigned i=0;i<NW;++i) f[i] = 0;
    }
    if(!barrier(0, workerId))
      return false;
    
    if (workerId == 0) {
      volatile unsigned* f = readyFlags.data();
      for (unsigned i=0;i<NW;++i) f[i] = 0;
    }
    countRays(workerId);

    if(!barrier(1, workerId))
      return false;
    // barrier(/*phase=*/696, workerId);  // countRays(workerId);

    unsigned outCnt[kNumLanes] = {0,0,0,0,0};
    if(workerId == 0) {
      for(int i=0; i<NW; i++) {
        for(int lane=0; lane<kNumLanes; lane++) {
          outCnt[lane] += sharedCounts[i*kNumLanes + lane];
        }
      }
      computeWriteOffsets();
    }
    // barrier(/*phase=*/2, workerId);  
    if(!barrier(2, workerId))
      return false;

    routeRays(workerId);
    // barrier(/*phase=*/3, workerId);  
    if(!barrier(3, workerId))
      return false;

    if(workerId == 0) {
      invalidateAfterRouting();

      unsigned inCnt[6] = {0,0,0,0,0,0}; 
      unsigned outCnt2[5] = {0,0,0,0,0}; 
      const poplar::Input<poplar::Vector<uint8_t>> *inputs[kNumLanes] = {
          &childRaysIn0, &childRaysIn1, &childRaysIn2, &childRaysIn3, &parentRaysIn};

      for (unsigned lane = 0; lane < kNumLanes; ++lane) {
        const auto &buf  = *inputs[lane];

        for (unsigned i = 0; i < kNumRays; i ++) {
          const Ray *ray = reinterpret_cast<const Ray *>(buf.data() + i * sizeof(Ray));
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

      // Fill debugBytes (10 counts = 20 bytes)
      unsigned* dbg = reinterpret_cast<unsigned*>(debugBytes.data());

      dbg[0] = inCnt[0];
      dbg[1] = inCnt[1];
      dbg[2] = inCnt[2];
      dbg[3] = inCnt[3];
      dbg[4] = inCnt[4];
      dbg[5] = inCnt[5];
      // dbg[0] = outCnt2[0];
      // dbg[1] = outCnt2[1];
      // dbg[2] = outCnt2[2];
      // dbg[3] = outCnt2[3];
      // dbg[4] = outCnt2[4];
      dbg[6] = outCnt[0];
      dbg[7] = outCnt[1];
      dbg[8] = outCnt[2];
      dbg[9] = outCnt[3];
      dbg[10] = outCnt[4];
      // dbg[10] = 0;
    }

    return true;
  }
private:
  [[gnu::always_inline]] unsigned getSharedIdx(unsigned workerId, unsigned lane) {
    return workerId * kNumLanes + lane;
  }

  [[gnu::always_inline]] unsigned spillStartIdx(unsigned w) const {
    const unsigned NW = poplar::MultiVertex::numWorkers();
    return NW * kNumLanes + w;
  }
  [[gnu::always_inline]] unsigned spillEndIdx(unsigned w) const {
    const unsigned NW = poplar::MultiVertex::numWorkers();
    return NW * kNumLanes + NW + w;
  }

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

  void routeRays(unsigned workerId) {
    const unsigned NW = poplar::MultiVertex::numWorkers();
    const unsigned C = kNumRays; //.size()    / sizeof(Ray);
        
    // totals per lane
    unsigned tot[kNumLanes];
    totalsPerLane(sharedCounts, NW, tot);
    LanePartition P = makePartition(tot, C);

    // my spill global range
    unsigned g    = sharedOffsets[spillStartIdx(workerId)];
    unsigned gEnd = sharedOffsets[spillEndIdx(workerId)];

    unsigned  wrCtr[kNumLanes] = {0,0,0,0,0};

    const poplar::Input<poplar::Vector<uint8_t>> *inBuf[kNumLanes] = {
        &childRaysIn0, &childRaysIn1, &childRaysIn2, &childRaysIn3, &parentRaysIn};

    poplar::InOut<poplar::Vector<uint8_t>> *outBuf[kNumLanes] = {
        &childRaysOut0, &childRaysOut1, &childRaysOut2, &childRaysOut3, &parentRaysOut};

    for (unsigned lane = 0; lane < kNumLanes; ++lane) {
      // indices: workerId, workerId+NW, …
      for (unsigned idx = workerId; idx < C; idx += NW) {
        const Ray *ray = reinterpret_cast<const Ray*>(inBuf[lane]->data() + idx * sizeof(Ray));
        if (ray->next_cluster == INVALID_RAY_ID) break;
        if (ray->next_cluster == FINISHED_RAY_ID) continue; 

        unsigned dst = findChildForCluster(ray->next_cluster);  // 0..4

        const unsigned wi = getSharedIdx(workerId, dst);
        const unsigned start = sharedOffsets[wi];
        const unsigned plannedEnd = start + sharedCounts[wi];
        const unsigned end   = plannedEnd < C ? plannedEnd : C;
        
        // decide once per ray whether we need to spill
        bool needSpill = false;

        // primary attempt
        // if (wrCtr[dst] < sharedCounts[wi]) {
        unsigned slot = start + wrCtr[dst];
        if (slot < end) {
          // primary write OK
          std::memcpy(outBuf[dst]->data() + slot*sizeof(Ray), ray, sizeof(Ray));
          ++wrCtr[dst];
        } else {
          needSpill = true; // this worker's primary quota used up
        }
        // } else {
        //   needSpill = true;   // this worker's primary quota used up
        // }

        // spill only if required
        if (needSpill) {
          if (g < gEnd) {
            unsigned r, slotInR;
            mapGlobalFreeToLane(g, P, r, slotInR);
            std::memcpy(outBuf[r]->data() + slotInR*sizeof(Ray), ray, sizeof(Ray));
            ++g;
          } else {
            // optional: count a drop
          }
        }

      }
    }
  }

  [[gnu::always_inline]]
  bool barrier(unsigned phase, unsigned workerId) {
    volatile unsigned* flags = readyFlags.data();
    const unsigned NW = poplar::MultiVertex::numWorkers();
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
    return true;
  }

  // bool barrier(unsigned phase, unsigned workerId) {
  //   volatile unsigned* flags = readyFlags.data();
  //   const unsigned NW = poplar::MultiVertex::numWorkers();

  //   flags[workerId] = phase;
  //   asm volatile ("" ::: "memory"); 

  //   // Debug watchdog to avoid infinite spin during bring-up
  //   unsigned spins = 0;
  //   for (;;) {
  //     bool all = true;
  //     for (unsigned i = 0; i < NW; ++i) {
  //       if (flags[i] < phase) { all = false; break; }
  //     }
  //     if (all) break;
  //     if (++spins == 100000) { 
  //       // drop a breadcrumb
  //       unsigned* dbg = reinterpret_cast<unsigned*>(debugBytes.data());
  //       // auto dbg = reinterpret_cast<volatile uint16_t*>(debugBytes.data());
  //       dbg[0] = flags[0];
  //       dbg[1] = flags[1];
  //       dbg[2] = flags[2];
  //       dbg[3] = flags[3];
  //       dbg[4] = flags[4];
  //       dbg[5] = flags[5];
  //       dbg[11] = 999;
  //       dbg[12] = phase*100 + workerId;
  //       // spins = 0;
  //       return false;
  //     }
  //   }

  //   asm volatile ("" ::: "memory");
  //   return true;
  // }

  [[gnu::always_inline]]
  unsigned findChildForCluster (uint16_t cluster_id) {
      unsigned childIdx  = (cluster_id >> shift) & 0x3;
      bool isChild = ((cluster_id >> (shift + 2)) == myChildPrefix);
      return isChild ? childIdx : PARENT;
  };

  void computeWriteOffsets() {
    const unsigned NW = poplar::MultiVertex::numWorkers();
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

  [[gnu::always_inline]]
  void wipeTail(poplar::InOut<poplar::Vector<uint8_t>> &out, unsigned written) {
    const unsigned capacity = out.size() / sizeof(Ray);
    if (written >= capacity) return;
    for (unsigned i = written; i < capacity; ++i) {
      Ray *rr = reinterpret_cast<Ray*>(out.data() + i*sizeof(Ray));
      if (rr->next_cluster == INVALID_RAY_ID) break;  // already clean
      rr->next_cluster = INVALID_RAY_ID;              // sentinel
    }
  }

  void invalidateAfterRouting() {
    const unsigned NW = poplar::MultiVertex::numWorkers();
    const unsigned C = kNumRays; // raysOut.size()    / sizeof(Ray);

    poplar::InOut<poplar::Vector<uint8_t>> *outBuf[kNumLanes] = {
      &childRaysOut0, &childRaysOut1, &childRaysOut2, &childRaysOut3, &parentRaysOut
    };

    // 1) Totals per lane and partition (base, freePrefix)
    unsigned tot[kNumLanes];
    totalsPerLane(sharedCounts, NW, tot);
    LanePartition P = makePartition(tot, C);

    // 2) How much global free space each lane received via spills
    unsigned assigned[kNumLanes];
    accumulateSpillAssigned(sharedOffsets, NW, P, assigned);

    // 3) Wipe the tails past written = base + assigned
    for (unsigned lane = 0; lane < kNumLanes; ++lane) {
      const unsigned written = P.base[lane] + assigned[lane]; // <= C
      wipeTail(*outBuf[lane], written);
    }
  }


  void countRays(unsigned workerId) {
    const unsigned NW = poplar::MultiVertex::numWorkers();
    const unsigned C = kNumRays; //raysOut.size()    / sizeof(Ray);
    for (unsigned lane = 0; lane < kNumLanes; ++lane)
      sharedCounts[getSharedIdx(workerId, lane)] = 0;

    const poplar::Input<poplar::Vector<uint8_t>> *inputs[kNumLanes] = {
        &childRaysIn0, &childRaysIn1, &childRaysIn2, &childRaysIn3, &parentRaysIn};

    for (unsigned lane = 0; lane < kNumLanes; ++lane) {
      const auto &buf  = *inputs[lane];

      for (unsigned i = workerId; i < C; i += NW) {
        const Ray *ray = reinterpret_cast<const Ray *>(buf.data() + i * sizeof(Ray));
        if (ray->next_cluster == INVALID_RAY_ID) break;
        if (ray->next_cluster == FINISHED_RAY_ID) continue;

        unsigned dst = findChildForCluster(ray->next_cluster);
        sharedCounts[getSharedIdx(workerId, dst)]++; 
      }
    }
  }

};
