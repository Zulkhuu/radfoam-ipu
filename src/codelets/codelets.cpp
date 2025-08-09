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

class RayTrace : public poplar::Vertex {
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
  poplar::Output<poplar::Vector<uint8_t>> raysOut;
  poplar::InOut<poplar::Vector<uint8_t>> finishedRays;
  // Debug outputs
  poplar::Output<float> result_float;
  poplar::Output<unsigned short> result_u16;

  // InOut
  poplar::InOut<poplar::Vector<uint8_t>> framebuffer;
  poplar::InOut<unsigned>       exec_count;          
  poplar::InOut<unsigned>       finishedWriteOffset; 

  // [[poplar::constraint("elem(*local_pts)!=elem(*finishedRays)")]]
  // [[poplar::constraint("elem(*local_pts)!=elem(*raysOut)")]]
  bool compute() {
    constexpr unsigned substeps = 1;
    constexpr unsigned finishedRaysCap = kNumRays * kFinishedFactor;
    unsigned exec_step_id = exec_count % substeps;
    bool startOfFrame = (exec_step_id == 0);

    const uint16_t nLocalPts = local_pts.size() / sizeof(LocalPoint);
    const uint16_t nNeighborPts = neighbor_pts.size() / sizeof(GenericPoint);

    if (startOfFrame) {
      *finishedWriteOffset = 0;
      invalidateRemainingFinishedRays(finishedRays, 0);
    }

    // Prepare matrices and ray origin
    glm::mat4 invView = glm::make_mat4(view_matrix.data());
    glm::mat4 invProj = glm::make_mat4(projection_matrix.data());
    glm::vec3 rayOrigin = glm::vec3(invView[3]);

    unsigned base = finishedWriteOffset;
    unsigned finished_ray_cntr = 0;
    unsigned out_ray_cntr = 0;
    unsigned cell_cntr = 0;
    bool debug = false;

    *result_u16 = 65535;

    // Loop over input rays
    for (int ray_index = 0; ray_index < kNumRays; ++ray_index) {
      const Ray* ray_in = readStructAt<Ray>(raysIn, ray_index);
      if (ray_in->x == INVALID_RAY_ID) break; // End of rays

      // Compute ray direction in world space
      uint16_t x_data = data10(ray_in->x); 
      uint8_t n_passed_clusters = getLead6(ray_in->x);
      glm::vec3 rayDir = computeRayDir(x_data, ray_in->y, invProj, invView);

      // Initialize accumulation
      glm::vec3 color(ray_in->r, ray_in->g, ray_in->b);
      float transmittance = __builtin_ipu_f16tof32(ray_in->transmittance);
      float t0 = __builtin_ipu_f16tof32(ray_in->t);

      int current = ray_in->next_local;
      int steps = 0;
      bool finished = false;

      while (!finished) {
        // Access current cell
        const LocalPoint* cur_cell = readStructAt<LocalPoint>(local_pts, current);
        glm::vec3 currentPos(cur_cell->x, cur_cell->y, cur_cell->z);

        // Determine adjacency range
        uint16_t start = (current == 0) ? 0 : readStructAt<LocalPoint>(local_pts, current - 1)->adj_end;
        uint16_t end   = cur_cell->adj_end;

        if(debug) {
          if(6 * (cell_cntr + 1) + 5 < framebuffer.size()) {
            cell_cntr++;
            framebuffer[6 * cell_cntr]     = static_cast<uint8_t>((ray_in->x >> 8) & 0xFF);
            framebuffer[6 * cell_cntr + 1] = static_cast<uint8_t>(ray_in->x & 0xFF);
            framebuffer[6 * cell_cntr + 2] = static_cast<uint8_t>((ray_in->y >> 8) & 0xFF);
            framebuffer[6 * cell_cntr + 3] = static_cast<uint8_t>(ray_in->y & 0xFF);
            framebuffer[6 * cell_cntr + 4] = static_cast<uint8_t>((current >> 8) & 0xFF);
            framebuffer[6 * cell_cntr + 5] = static_cast<uint8_t>(current & 0xFF);
          }
        }

        // Traverse neighbors
        float closestT = std::numeric_limits<float>::max();
        int nextIdx = -1;

        for (uint16_t j = start; j < end; ++j) {
          uint16_t neighborIdx = adjacency[j];
          glm::vec3 nbrPos;

          if (neighborIdx < nLocalPts) {
            const LocalPoint* nbrPt = readStructAt<LocalPoint>(local_pts, neighborIdx);
            nbrPos.x = nbrPt->x;
            nbrPos.y = nbrPt->y;
            nbrPos.z = nbrPt->z;
          } else {
            const GenericPoint* nbrPt = readStructAt<GenericPoint>(neighbor_pts, neighborIdx - nLocalPts);
            nbrPos.x = nbrPt->x;
            nbrPos.y = nbrPt->y;
            nbrPos.z = nbrPt->z;
          }

          glm::vec3 offset = nbrPos - currentPos;
          glm::vec3 faceNormal = offset;
          glm::vec3 faceOrigin = currentPos + 0.5f * offset;

          float dotND = glm::dot(faceNormal, rayDir);
          if (dotND <= 0.0f) continue;

          float t = glm::dot(faceOrigin - rayOrigin, faceNormal) / dotND;
          if (t > 0 && t < closestT ) {
            closestT = t;
            nextIdx = neighborIdx;
          }
        }

        // Accumulate color
        float delta = closestT - t0;
        float alpha = 1.0f - __builtin_ipu_exp(-cur_cell->density * delta);
        // color += transmittance * alpha *
        //          glm::vec3(cur_cell->r / 255.0f, cur_cell->g / 255.0f, cur_cell->b / 255.0f);
        auto ta = transmittance * alpha;
        color.x += ta * (cur_cell->r/255.0f);
        color.y += ta * (cur_cell->g/255.0f);
        color.z += ta * (cur_cell->b/255.0f);


        transmittance *= (1.0f - alpha);
        t0 = __builtin_ipu_max(t0, closestT);

        // Termination conditions
        if (transmittance < 0.01f || nextIdx == -1 || nextIdx >= nLocalPts) {
          if (transmittance < 0.01f || nextIdx == -1) {
            // Ray finished
            FinishedRay* finished_ray = reinterpret_cast<FinishedRay*>(finishedRays.data() + sizeof(FinishedRay) * (base + finished_ray_cntr));
            finished_ray->x = x_data;
            finished_ray->y = ray_in->y;
            finished_ray->r = clampToU8(color.x);
            finished_ray->g = clampToU8(color.y);
            finished_ray->b = clampToU8(color.z);
            finished_ray->t = t0;
            finished_ray_cntr++;
            
            *result_u16 = 65533;
            if(nextIdx == -1)
              *result_u16 = 65534; 
            *result_float = color.x;
          }
          else if (nextIdx >= nLocalPts) {
            // Move to next cluster
            const GenericPoint* nbrPt = readStructAt<GenericPoint>(neighbor_pts, nextIdx - nLocalPts);
            Ray* ray_out = reinterpret_cast<Ray*>(raysOut.data()+sizeof(Ray)*out_ray_cntr);
            ray_out->next_cluster = nbrPt->cluster_id;
            ray_out->next_local   = nbrPt->local_id;
            transmittance = __builtin_ipu_max(0.0f, __builtin_ipu_min(1.0f, transmittance));
            ray_out->transmittance = __builtin_ipu_f32tof16(transmittance);
            if(n_passed_clusters == 5 || n_passed_clusters == 10 || n_passed_clusters == 15)
              ray_out->x = setLead6(x_data, n_passed_clusters);
            else
              ray_out->x = setLead6(x_data, n_passed_clusters+1);
            ray_out->y = ray_in->y;
            ray_out->t = __builtin_ipu_f32tof16(t0);
            ray_out->r = color.x;
            ray_out->g = color.y;
            ray_out->b = color.z;
            out_ray_cntr++;

            *result_u16 = nbrPt->cluster_id; // y;
            *result_float = color.x;
          }
          finished = true;
        }

        ++steps;
        current = nextIdx;
      }

    }

    if(debug){
      framebuffer[0] = cell_cntr & 0xFF;
    }

    // Invalidate unused slots
    invalidateRemainingRays(raysOut, out_ray_cntr);
    // invalidateRemainingFinishedRays(finishedRays, finished_ray_cntr);

    // Update append pointer (clamped)
    unsigned newOffset = base + finished_ray_cntr;
    if (newOffset > finishedRaysCap) newOffset = finishedRaysCap;
    *finishedWriteOffset = newOffset;

    *exec_count = exec_count + 1;

    return true;
  }

private:
  [[gnu::always_inline]]
  void writeFinishedRay(poplar::Output<poplar::Vector<uint8_t>>& buffer,
                        int& counter, uint16_t x, uint16_t y,
                        const glm::vec3& color, float t) {
    FinishedRay* ray = reinterpret_cast<FinishedRay*>(buffer.data() + sizeof(FinishedRay) * counter);
    ray->x = x;
    ray->y = y;
    ray->r = clampToU8(color.x);
    ray->g = clampToU8(color.y);
    ray->b = clampToU8(color.z);
    ray->t = t;
    counter++;
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
  void invalidateRemainingRays(poplar::Output<poplar::Vector<uint8_t>>& buffer, int count) {
    for (int i = count; i < buffer.size()/sizeof(Ray); i++) {
      Ray* ray = reinterpret_cast<Ray*>(buffer.data() + sizeof(Ray) * i);
      if(ray->x == INVALID_RAY_ID)
        break;
      ray->x = INVALID_RAY_ID;
    }
  }

  [[gnu::always_inline]]
  inline void invalidateRemainingFinishedRays(poplar::InOut<poplar::Vector<uint8_t>>& buffer, int count) {
    for (int i = count; i < buffer.size()/sizeof(FinishedRay); i++) {
      FinishedRay* ray = reinterpret_cast<FinishedRay*>(buffer.data() + sizeof(FinishedRay) * i);
      if(ray->x == INVALID_RAY_ID)
        break;
      ray->x = INVALID_RAY_ID;
    }
  }
};

class RayGen : public poplar::Vertex {
public:
  poplar::Input<poplar::Vector<uint8_t>> childRaysIn0;
  poplar::Input<poplar::Vector<uint8_t>> childRaysIn1;
  poplar::Input<poplar::Vector<uint8_t>> childRaysIn2;
  poplar::Input<poplar::Vector<uint8_t>> childRaysIn3;

  poplar::Output<poplar::Vector<uint8_t>> childRaysOut0;
  poplar::Output<poplar::Vector<uint8_t>> childRaysOut1;
  poplar::Output<poplar::Vector<uint8_t>> childRaysOut2;
  poplar::Output<poplar::Vector<uint8_t>> childRaysOut3;

  poplar::InOut<unsigned> exec_count; 
  poplar::Input<poplar::Vector<uint8_t>> camera_cell_info;
  poplar::Output<poplar::Vector<uint8_t>> debugBytes;

  bool compute() {
    constexpr uint8_t shift = 8; // lvl 4*2
    uint16_t outCountC0 = 0, outCountC1 = 0, outCountC2 = 0, outCountC3 = 0;
    uint16_t cluster_id = camera_cell_info[0] | (camera_cell_info[1] << 8);
    uint16_t local_id   = camera_cell_info[2] | (camera_cell_info[3] << 8);

    // Helper: Determine target child
    auto findChildForCluster = [&](uint16_t cid) __attribute__((always_inline)) -> unsigned {
        return (cid >> shift) & 0x3;
    };

    // Helper: Route a ray
    auto routeRay = [&](const Ray* ray) {
      int targetChild = findChildForCluster(ray->next_cluster);
      if (targetChild == 0) {
        if (outCountC0 < kNumRays) {
          std::memcpy(childRaysOut0.data() + outCountC0 * sizeof(Ray), ray, sizeof(Ray));
          outCountC0++;
        }
      } else if (targetChild == 1) {
        if (outCountC1 < kNumRays) {
          std::memcpy(childRaysOut1.data() + outCountC1 * sizeof(Ray), ray, sizeof(Ray));
          outCountC1++;
        }
      } else if (targetChild == 2) {
        if (outCountC2 < kNumRays) {
          std::memcpy(childRaysOut2.data() + outCountC2 * sizeof(Ray), ray, sizeof(Ray));
          outCountC2++;
        }
      } else if (targetChild == 3) {
        if (outCountC3 < kNumRays) {
          std::memcpy(childRaysOut3.data() + outCountC3 * sizeof(Ray), ray, sizeof(Ray));
          outCountC3++;
        }
      } 
    };

    auto routeChildRays = [&](const poplar::Input<poplar::Vector<uint8_t>>& childIn) __attribute__((always_inline)) -> uint16_t {
      uint16_t count = 0;
      for (int i = 0; i < kNumRays; ++i) {
        const Ray* ray = reinterpret_cast<const Ray*>(childIn.data() + i * sizeof(Ray));
        if (ray->x == INVALID_RAY_ID) break;
        count++;
        routeRay(ray);

      }
      return count;
    };

    int mode = 1;
    if(mode == 0) { // Single ray test
      if(exec_count == 0) {
        Ray genRay{};
        genRay.x = 304;
        genRay.y = 288;
        genRay.r = 0.0f;
        genRay.g = 0.0f;
        genRay.b = 0.0f;
        genRay.t = 0.0f;
        genRay.transmittance = 1.0f;
        genRay.next_cluster = cluster_id;
        genRay.next_local   = local_id;
        routeRay(&genRay);
      }
    }
    if(mode == 1) { // Row scan
      const int interval = 3;
      const int nRowsPerFrame = 2;
      if(exec_count%interval == 0) {
        for(uint16_t x=0; x<kFullImageWidth; x++) {
          for(uint16_t y=0; y<nRowsPerFrame; y++) {
            Ray genRay{};

            genRay.x = x; //(x+(exec_count/interval)*3)%kFullImageWidth;
            genRay.y = (y+(exec_count/interval)*nRowsPerFrame)%kFullImageHeight;
            genRay.r = 0.0f;
            genRay.g = 0.0f;
            genRay.b = 0.0f;
            genRay.t = 0.0;
            genRay.transmittance = 1.0;
            genRay.next_cluster = cluster_id;
            genRay.next_local   = local_id;

            routeRay(&genRay);
          }
        }
      } 
    }
    if(mode == 2) {
      const int interval = 3;
      const int nx = 40;
      const int ny = 30;
      if(exec_count%interval == 0) {
        for(uint16_t x=0; x<nx; x++) {
          for(uint16_t y=0; y<ny; y++) {
            Ray genRay{};
            genRay.x = x*16 + exec_count%16;
            genRay.y = y*16 + (exec_count/16)%16;
            genRay.r = 0.0f;
            genRay.g = 0.0f;
            genRay.b = 0.0f;
            genRay.t = 0.0f;
            genRay.transmittance = 1.0f;
            genRay.next_cluster = cluster_id;
            genRay.next_local   = local_id;

            routeRay(&genRay);
          }
        }
      } 
    }
    if(mode == 3) {
      const int interval = 15;
      const int nx = 40;
      const int ny = 30;
      if(exec_count == 0) {
        for(uint16_t x=0; x<nx; x++) {
          for(uint16_t y=0; y<ny; y++) {
            Ray genRay{};
            genRay.x = x*16; // + ((exec_count/interval)%4)*4 ;
            genRay.y = y*16; // + (((exec_count/interval)/4)%4)*4;
            genRay.r = 0.0f;
            genRay.g = 0.0f;
            genRay.b = 0.0f;
            genRay.t = 0.0f;
            genRay.transmittance = 1.0f;
            genRay.next_cluster = cluster_id;
            genRay.next_local   = local_id;

            routeRay(&genRay);
          }
        }
      } 
    }

    // Route each child and capture their input counts
    uint16_t inCountC0 = routeChildRays(childRaysIn0);
    uint16_t inCountC1 = routeChildRays(childRaysIn1);
    uint16_t inCountC2 = routeChildRays(childRaysIn2);
    uint16_t inCountC3 = routeChildRays(childRaysIn3);

    auto invalidateRemaining = [&](poplar::Output<poplar::Vector<uint8_t>>& out, uint16_t count) __attribute__((always_inline)) {
      for (uint16_t i = count; i < kNumRays; ++i) {
        Ray* ray = reinterpret_cast<Ray*>(out.data() + sizeof(Ray) * i);
        if(ray->x == INVALID_RAY_ID)
          break;
        ray->x = INVALID_RAY_ID;
      }
    };

    invalidateRemaining(childRaysOut0, outCountC0);
    invalidateRemaining(childRaysOut1, outCountC1);
    invalidateRemaining(childRaysOut2, outCountC2);
    invalidateRemaining(childRaysOut3, outCountC3);

    // --- Debug bytes ---
    uint16_t* dbg = reinterpret_cast<uint16_t*>(debugBytes.data());
    dbg[0] = *exec_count;
    dbg[1] = inCountC0;
    dbg[2] = inCountC1;
    dbg[3] = inCountC2;
    dbg[4] = inCountC3;
    dbg[5] = 0;
    dbg[6] = outCountC0;
    dbg[7] = outCountC1;
    dbg[8] = outCountC2;
    dbg[9] = outCountC3;

    *exec_count = exec_count+1;
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
  poplar::Output<poplar::Vector<uint8_t>> parentRaysOut;
  poplar::Output<poplar::Vector<uint8_t>> childRaysOut0;
  poplar::Output<poplar::Vector<uint8_t>> childRaysOut1;
  poplar::Output<poplar::Vector<uint8_t>> childRaysOut2;
  poplar::Output<poplar::Vector<uint8_t>> childRaysOut3;

  // ID mapping to know which cluster IDs belong to which child
  poplar::Input<poplar::Vector<unsigned short>> childClusterIds;
  poplar::Input<uint8_t> level;

  poplar::Output<poplar::Vector<uint8_t>> debugBytes;

  poplar::InOut<poplar::Vector<unsigned>> sharedCounts; 
  poplar::InOut<poplar::Vector<unsigned>> sharedOffsets; 
  poplar::InOut<poplar::Vector<unsigned>> readyFlags; 

  static constexpr unsigned kNumLanes = 5;
  static constexpr unsigned kNumWorkers = 6; //MultiVertex::numWorkers(); 
  static constexpr unsigned PARENT = 4;
    
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

  bool compute(unsigned workerId) {
    shift = *level * 2;
    myChildPrefix = childClusterIds[0] >> (shift + 2);

    if (workerId == 0) {
      volatile unsigned* f = readyFlags.data();
      for (unsigned i=0;i<kNumWorkers;++i) f[i] = 0;
    }
    barrier(0, workerId);
    
    countRays(workerId);
    barrier(/*phase=*/1, workerId);  // countRays(workerId);

    unsigned outCnt2[kNumLanes] = {0,0,0,0,0};
    if(workerId == 0) {
      for(int i=0; i<kNumWorkers; i++) {
        for(int lane=0; lane<kNumLanes; lane++) {
          outCnt2[lane] += sharedCounts[i*kNumLanes + lane];
        }
      }
      computeWriteOffsets();
    }
    barrier(/*phase=*/2, workerId);  

    routeRays(workerId);
    barrier(/*phase=*/3, workerId);  

    if(workerId == 0) {
      unsigned outCnt[5] = {0,0,0,0,0}; 

      // invalidateRemainingRays(childRaysOut0, outCnt2[0]);
      // invalidateRemainingRays(childRaysOut1, outCnt2[1]);
      // invalidateRemainingRays(childRaysOut2, outCnt2[2]);
      // invalidateRemainingRays(childRaysOut3, outCnt2[3]);
      // invalidateRemainingRays(parentRaysOut, outCnt2[4]);

      // Fill debugBytes (10 counts = 20 bytes)
      uint16_t* dbg = reinterpret_cast<uint16_t*>(debugBytes.data());

      dbg[0] = outCnt[0];
      dbg[1] = outCnt[1];
      dbg[2] = outCnt[2];
      dbg[3] = outCnt[3];
      dbg[4] = outCnt[4];
      dbg[5] = outCnt2[0];
      dbg[6] = outCnt2[1];
      dbg[7] = outCnt2[2];
      dbg[8] = outCnt2[3];
      dbg[9] = outCnt2[4];
    }

    return true;
  }
private:

  void routeRays(unsigned workerId) {
    //------------------------------------------------------------------
    //  1.  Local state
    //------------------------------------------------------------------
    unsigned  wrCtr[kNumLanes] = {0,0,0,0,0};             // per-lane counter

    const poplar::Input<poplar::Vector<uint8_t>> *inBuf[kNumLanes] = {
        &childRaysIn0, &childRaysIn1, &childRaysIn2, &childRaysIn3, &parentRaysIn};

    poplar::Output<poplar::Vector<uint8_t>> *outBuf[kNumLanes] = {
        &childRaysOut0, &childRaysOut1, &childRaysOut2, &childRaysOut3, &parentRaysOut};

    //------------------------------------------------------------------
    //  2.  Strided walk over *all* input lanes
    //------------------------------------------------------------------
    for (unsigned lane = 0; lane < kNumLanes; ++lane) {
      // const unsigned N = in.size() / sizeof(Ray);

      // indices: workerId, workerId+kNumWorkers, …
      for (unsigned idx = workerId; idx < kNumRays; idx += kNumWorkers) {
        const Ray *ray = reinterpret_cast<const Ray*>(inBuf[lane]->data() + idx * sizeof(Ray));
        if (ray->x == INVALID_RAY_ID) break;

        unsigned dst = findChildForCluster(ray->next_cluster);  // 0..4
        const unsigned wi = getSharedIdx(workerId, dst);
        const unsigned start = sharedOffsets[wi];
        const unsigned plannedEnd = start + sharedCounts[wi];
        const unsigned end   = plannedEnd < kNumRays ? plannedEnd : kNumRays;
        unsigned slot = start + wrCtr[dst];
        if(slot < end) {
          std::memcpy(outBuf[dst]->data() + slot * sizeof(Ray), ray, sizeof(Ray));
          wrCtr[dst]++;
        }
      }
    }

    //------------------------------------------------------------------
    // 3. Last worker invalidates tail
    //------------------------------------------------------------------
    if (workerId == kNumWorkers - 1) {
      for (unsigned lane = 0; lane < kNumLanes; ++lane) {
        unsigned written  = sharedOffsets[getSharedIdx(kNumWorkers-1, lane)] + wrCtr[lane];   // total
        unsigned capacity = outBuf[lane]->size() / sizeof(Ray);
        for (unsigned i = written; i < capacity; ++i) {
          Ray *r = reinterpret_cast<Ray*>(outBuf[lane]->data() + i*sizeof(Ray));
          if(r->x == INVALID_RAY_ID)
            break;
          r->x = INVALID_RAY_ID;
        }
      }
    }
  }

  [[gnu::always_inline]]
  void barrier(unsigned phase, unsigned workerId) {
    volatile unsigned* flags = readyFlags.data();
    const unsigned NW = poplar::MultiVertex::numWorkers();
    flags[workerId] = phase;
    asm volatile("" ::: "memory");
    while (true) {
      bool all = true;
      for (unsigned i = 0; i < NW; ++i) if (flags[i] != phase) { all = false; break; }
      if (all) break;
    }
    asm volatile("" ::: "memory");
  }

  [[gnu::always_inline]]
  unsigned findChildForCluster (uint16_t cluster_id) {
      unsigned childIdx  = (cluster_id >> shift) & 0x3;
      bool isChild = ((cluster_id >> (shift + 2)) == myChildPrefix);
      return isChild ? childIdx : PARENT;
  };

  void computeWriteOffsets() {
    // first worker offsets are zero -> discarded
    // sum of all is added last instead
    for (unsigned lane = 0; lane < kNumLanes; ++lane) {
      unsigned offset = 0;
      for (unsigned worker = 0; worker < kNumWorkers; ++worker) {
        unsigned count = sharedCounts[getSharedIdx(worker, lane)];
        sharedOffsets[getSharedIdx(worker, lane)] = offset;
        offset += count;
      }
    }
  }

  [[gnu::always_inline]]
  unsigned getSharedIdx(unsigned workerId, unsigned lane) {
    return workerId * kNumLanes + lane;
  }

  void countRays(unsigned workerId) {
    // Clear previous values
    for (unsigned lane = 0; lane < kNumLanes; ++lane)
      sharedCounts[getSharedIdx(workerId, lane)] = 0;

    // Convenient arrays of input buffers
    const poplar::Input<poplar::Vector<uint8_t>> *inputs[kNumLanes] = {
        &childRaysIn0, &childRaysIn1, &childRaysIn2, &childRaysIn3, &parentRaysIn};

    // Visit every lane
    for (unsigned lane = 0; lane < kNumLanes; ++lane) {
      const auto &buf  = *inputs[lane];

      // Strided loop: workerId, workerId+kNumWorkers, …
      for (unsigned i = workerId; i < kNumRays; i += kNumWorkers) {
        const Ray *ray = reinterpret_cast<const Ray *>(buf.data() + i * sizeof(Ray));
        if (ray->x == INVALID_RAY_ID) break;

        unsigned dst = findChildForCluster(ray->next_cluster);
        sharedCounts[getSharedIdx(workerId, dst)]++; 
      }
    }
  }

};


