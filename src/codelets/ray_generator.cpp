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


class RayGenerator : public poplar::Vertex {
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
      const unsigned nRowsPerFrame = 1;
      const unsigned idleFramesAfterSweep = 200;
      const unsigned sweepFrames = (kFullImageHeight + nRowsPerFrame - 1) / nRowsPerFrame;
      const unsigned cycleLen = sweepFrames + idleFramesAfterSweep;
      const unsigned phase    = (*exec_count) % cycleLen;
      if (phase < sweepFrames) {
        const unsigned rowBase = phase * nRowsPerFrame;

        // Don’t generate more than remaining output capacity + queue free slots
        unsigned budget = (C - outCnt) + qFree(head, tail);

        Ray g{};
        g.r = g.g = g.b = 0.0f;
        g.t = 0.0f; g.d = 0.0f;
        g.transmittance = 1.0f;
        g.next_cluster = cluster_id;
        g.next_local   = local_id;
        for (uint16_t cy = 0; cy < nRowsPerFrame; ++cy) {
          const uint16_t y = (rowBase + cy) % kFullImageHeight;
          for (uint16_t x = 0; x < kFullImageWidth; ++x) {
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
      const unsigned nColsPerFrame = 2;
      const unsigned idleFramesAfterSweep = 200;
      const unsigned sweepFrames = (kFullImageWidth + nColsPerFrame - 1) / nColsPerFrame;
      const unsigned cycleLen = sweepFrames + idleFramesAfterSweep;
      const unsigned phase    = (*exec_count) % cycleLen;
      if (phase < sweepFrames) {
        const unsigned colBase = phase * nColsPerFrame;

        // Don’t generate more than remaining output capacity + queue free slots
        unsigned budget = (C - outCnt) + qFree(head, tail);

        Ray g{};
        g.r = g.g = g.b = 0.0f;
        g.t = 0.0f; g.d = 0.0f;
        g.transmittance = 1.0f;
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
        g.t = 0.0f;
        g.transmittance = 1.0f;
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
        g.t = 0.0f;
        g.transmittance = 1.0f;
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
        g.t = 0.0f;
        g.transmittance = 1.0f;
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

    // if(*exec_count == 480)
    //   *exec_count = 0;
    // else
    *exec_count = *exec_count + 1;
    return true;
  }
};
