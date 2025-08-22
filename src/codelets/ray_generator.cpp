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

inline constexpr uint16_t INVALID_RAY_ID = 0xFFFF;
inline constexpr uint16_t FINISHED_RAY_ID = 0xFFFE;
inline constexpr unsigned NW = 6;

class RayGenMT : public poplar::MultiVertex {
public:
  using BytesIn  = poplar::Input< poplar::Vector<uint8_t,  poplar::VectorLayout::ONE_PTR, 8> >;
  using BytesIO  = poplar::InOut<  poplar::Vector<uint8_t,  poplar::VectorLayout::ONE_PTR, 8> >;

  // I/O
  BytesIn  RaysIn;                     // spillovers from L4 parentOut
  BytesIO  RaysOut;                    // to L4 parentIn
  BytesIn  seedRaysOut;

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

class RayGenerator2 : public poplar::Vertex {
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
    int raygen_mode = 4;
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

// RayGen.cpp (device codelet)
class RayGenerator : public poplar::MultiVertex {
public:
  // inputs you already had (camera_cell_info, matrices, etc.)
  poplar::Input<poplar::Vector<uint8_t>> camera_cell_info;

  // NEW: 1024 seeds — one slot per tracer. Each slot is sizeof(Ray).
  poplar::InOut<poplar::Vector<uint8_t>> seedRaysOut;

  poplar::InOut< poplar::Vector<unsigned>> readyFlags;    // >= NW
  poplar::InOut< poplar::Vector<unsigned>> sharedCounts;  // >= (2*NW + 3)
  poplar::InOut< poplar::Vector<unsigned>> sharedOffsets; // >= (5*NW + 1)
  poplar::Output< poplar::Vector<unsigned>> debugBytes;

  poplar::InOut<unsigned> exec_count;

  bool compute(unsigned wid) {
    const unsigned NW = poplar::MultiVertex::numWorkers();
    const unsigned numTiles = kNumRayTracerTiles; // kNumRayTracerTiles
    const unsigned bytesPerRay = sizeof(Ray);

    // 1) Clear all seeds to INVALID in parallel
    for (unsigned t = wid; t < numTiles; t += NW) {
      Ray* slot = reinterpret_cast<Ray*>(seedRaysOut.data() + t * bytesPerRay);
      slot->next_cluster = INVALID_RAY_ID;
    }

    // 2) Pick the destination tile and craft the seed (example uses camera cell)
    // camera_cell_info = [cluster_lo, cluster_hi, local_lo, local_hi]
    const uint16_t destTile = static_cast<uint16_t>(camera_cell_info[0] | (uint16_t(camera_cell_info[1]) << 8));
    const uint16_t seedLocal = static_cast<uint16_t>(camera_cell_info[2] | (uint16_t(camera_cell_info[3]) << 8));

    // choose x you want to seed from (for your 3-column mode)
    const unsigned nColsPerFrame = 2;
    const uint16_t colBase = ((*exec_count)) * nColsPerFrame;

    if (wid == 0) {
      SeedRay* slot = reinterpret_cast<SeedRay*>(seedRaysOut.data() + destTile * bytesPerRay);
      slot->x            = colBase%kFullImageWidth;
      slot->y            = 0;  
      slot->nrows        = 0;
      slot->ncols        = nColsPerFrame;
      slot->next_cluster = destTile;      // consumed by that tile
      slot->next_local   = seedLocal;     // starting cell
      *exec_count = *exec_count + 1;
    }
    return true;
  }
};
