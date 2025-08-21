#include <poplar/Vertex.hpp>

#include <glm/mat4x4.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <geometry/primitives.hpp>
#include <ipu/rf_config.hpp>
#include <ipudef.h>
#include <ipu_builtins.h>
#include <ipu_vector_math>
#include <ipu_memory_intrinsics>

using namespace radfoam::config;
using radfoam::geometry::LocalPoint;
using radfoam::geometry::GenericPoint;

struct alignas(4) Ray {
  uint16_t x, y;
  half t, transmittance;
  float r, g, b;
  uint16_t next_cluster; 
  uint16_t next_local;
};

inline constexpr uint16_t INVALID_RAY_ID = 0xFFFF;
inline constexpr uint16_t FINISHED_RAY_ID = 0xFFFE;
inline constexpr unsigned NW = 6;

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
