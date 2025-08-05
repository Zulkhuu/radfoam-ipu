#include <poplar/Vertex.hpp>

#include <glm/mat4x4.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <geometry/primitives.hpp>
#include <ipu/rf_config.hpp>

using namespace radfoam::geometry;
using namespace radfoam::config;

// static const glm::mat4 View2(
//     glm::vec4(-0.034899f,  0.000000f, -0.999391f, 0.000000f),
//     glm::vec4( 0.484514f, -0.874620f, -0.016920f, 0.000000f),
//     glm::vec4(-0.874087f, -0.484810f,  0.030524f, 0.000000f),
//     glm::vec4(-0.000000f, -0.000000f, -6.700000f, 1.000000f)
// );

// static const glm::mat4 Proj2(
//     glm::vec4(1.299038f, 0.000000f,  0.000000f,  0.000000f),
//     glm::vec4(0.000000f, 1.732051f,  0.000000f,  0.000000f),
//     glm::vec4(0.000000f, 0.000000f, -1.002002f, -1.000000f),
//     glm::vec4(0.000000f, 0.000000f, -0.200200f,  0.000000f)
// );

// inline __attribute__((always_inline))
// const LocalPoint* readLocalPointAt(const poplar::Input<poplar::Vector<uint8_t>>& buffer, std::size_t index) {
//     constexpr std::size_t stride = sizeof(LocalPoint);
//     const uint8_t* base = buffer.data() + index * stride;
//     return reinterpret_cast<const LocalPoint*>(base);
// }

template <typename T>
inline __attribute__((always_inline))
const T* readStructAt(const poplar::Input<poplar::Vector<uint8_t>>& buffer, std::size_t index) {
    constexpr std::size_t stride = sizeof(T);
    const uint8_t* base = buffer.data() + index * stride;
    return reinterpret_cast<const T*>(base);
}

inline __attribute__((always_inline))
uint8_t clampToU8(float v) {
    return static_cast<uint8_t>(std::fmax(0.0f, std::fmin(255.0f, std::round(v * 255.0f))));
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
  poplar::InOut<unsigned>       exec_count;           // increments every sub-iteration
  poplar::InOut<unsigned>       finishedWriteOffset;  // append pointer within finishedRays
  // poplar::Input<unsigned short> substeps;             // e.g., 10


  [[poplar::constraint("elem(*local_pts)!=elem(*framebuffer)")]]
  bool compute() {
    constexpr int RaySize = sizeof(Ray);
    constexpr int LocalPointSize = sizeof(LocalPoint);
    constexpr int GenericPointSize = sizeof(GenericPoint); 

    constexpr unsigned substeps = 20;
    constexpr unsigned finishedRaysCap = kNumRays * 3;
    unsigned exec_step_id = exec_count % substeps;
    bool startOfFrame = (exec_step_id == 0);

    const uint16_t nLocalPts = local_pts.size() / LocalPointSize;
    const uint16_t nNeighborPts = neighbor_pts.size() / GenericPointSize;

    if (startOfFrame) {
      *finishedWriteOffset = 0;
      invalidateRemainingFinishedRays(finishedRays, 0);
    }

    // Prepare matrices and ray origin
    glm::mat4 invView = glm::make_mat4(view_matrix.data());
    glm::mat4 invProj = glm::make_mat4(projection_matrix.data());
    glm::vec3 rayOrigin = glm::vec3(invView[3]);

    auto computeRayDir = [&](uint16_t x, uint16_t y) __attribute__((always_inline)) -> glm::vec3 {
      float ndcX = (2.0f * x) / kFullImageWidth - 1.0f;
      float ndcY = 1.0f - (2.0f * y) / kFullImageHeight;
      glm::vec4 clipRay(ndcX, ndcY, -1.0f, 1.0f);
      glm::vec4 eyeRay = invProj * clipRay;
      eyeRay.z = -1.0f;
      eyeRay.w = 0.0f;
      return glm::normalize(glm::vec3(invView * eyeRay));
    };

    unsigned base = finishedWriteOffset;
    unsigned finished_ray_cntr = 0;
    unsigned out_ray_cntr = 0;
    unsigned cell_cntr = 0;
    bool debug = false;

    *result_u16 = 65535;

    // Loop over input rays
    for (int ray_index = 0; ray_index < kNumRays; ++ray_index) {
      const Ray* ray_in = readStructAt<Ray>(raysIn, ray_index);
      if (ray_in->x == 0xFFFF) break; // End of rays

      // Compute ray direction in world space
      glm::vec3 rayDir = computeRayDir(ray_in->x, ray_in->y);

      // Initialize accumulation
      glm::vec3 color(ray_in->r, ray_in->g, ray_in->b);
      float transmittance = ray_in->transmittance;
      float t0 = ray_in->t;

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
          cell_cntr++;
          framebuffer[2 * cell_cntr]     = static_cast<uint8_t>((current >> 8) & 0xFF);
          framebuffer[2 * cell_cntr + 1] = static_cast<uint8_t>(current & 0xFF);
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
          if (t > 0 && t < closestT  && t >= t0) {
            closestT = t;
            nextIdx = neighborIdx;
          }
        }

        // // Guard for stuck rays
        // if (steps > 1050) {
        //   nextIdx = -1;
        // }

        // Accumulate color
        // float delta = __builtin_fmaxf(0.0f, closestT - t0);
        float delta = closestT - t0;
        float alpha = 1.0f - expf(-cur_cell->density * delta);
        // color += transmittance * alpha *
        //          glm::vec3(cur_cell->r / 255.0f, cur_cell->g / 255.0f, cur_cell->b / 255.0f);
        color.x += transmittance * alpha * (cur_cell->r/255.0f);
        color.y += transmittance * alpha * (cur_cell->g/255.0f);
        color.z += transmittance * alpha * (cur_cell->b/255.0f);

        transmittance *= (1.0f - alpha);
        t0 = __builtin_fmaxf(t0, closestT);

        // Termination conditions
        if (transmittance < 0.01f || nextIdx == -1 || nextIdx >= nLocalPts) {
          if (transmittance < 0.01f || nextIdx == -1) {
            // Ray finished
            FinishedRay* finished_ray = reinterpret_cast<FinishedRay*>(finishedRays.data() + sizeof(FinishedRay) * (base + finished_ray_cntr));
            finished_ray->x = ray_in->x;
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
            ray_out->transmittance = transmittance;
            ray_out->x = ray_in->x;
            ray_out->y = ray_in->y;
            ray_out->t = t0;
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
        t0 = __builtin_fmaxf(t0, closestT);
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

  void invalidateRemainingRays(poplar::Output<poplar::Vector<uint8_t>>& buffer, int count) {
    for (int i = count; i < kNumRays; i++) {
      Ray* ray = reinterpret_cast<Ray*>(buffer.data() + sizeof(Ray) * i);
      if(ray->x == 0xFFFF)
        break;
      ray->x = 0xFFFF;
    }
  }

  inline void invalidateRemainingFinishedRays(poplar::InOut<poplar::Vector<uint8_t>>& buffer, int count) {
    for (int i = count; i < kNumRays*3; i++) {
      FinishedRay* ray = reinterpret_cast<FinishedRay*>(buffer.data() + sizeof(FinishedRay) * i);
      if(ray->x == 0xFFFF)
        break;
      ray->x = 0xFFFF;
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
    constexpr int RaySize = sizeof(Ray);  
    constexpr uint16_t INVALID_RAY_ID = 0xFFFF;
    const uint16_t childClusterIds[4] = {0, 256, 512, 768};
    const int lvl = 4;
    uint8_t shift = lvl * 2;
    uint16_t inCountC0 = 0, inCountC1 = 0, inCountC2 = 0, inCountC3 = 0;
    uint16_t outCountC0 = 0, outCountC1 = 0, outCountC2 = 0, outCountC3 = 0;

    uint16_t cluster_id = camera_cell_info[0] | (camera_cell_info[1] << 8);
    uint16_t local_id   = camera_cell_info[2] | (camera_cell_info[3] << 8);

    // Helper: Determine target child
    auto findChildForCluster = [&](uint16_t clusterId) -> int {
      for (int i = 0; i < 4; ++i) {
        if ((clusterId >> 8) == (childClusterIds[i] >> 8))
          return i;
      }
      return -1;
    };

    // Helper: Route a ray
    auto routeRay = [&](const Ray* ray) {
      int targetChild = findChildForCluster(ray->next_cluster);
      if (targetChild == 0) {
        if (outCountC0 < kNumRays) {
          std::memcpy(childRaysOut0.data() + outCountC0 * RaySize, ray, RaySize);
          outCountC0++;
        }
      } else if (targetChild == 1) {
        if (outCountC1 < kNumRays) {
          std::memcpy(childRaysOut1.data() + outCountC1 * RaySize, ray, RaySize);
          outCountC1++;
        }
      } else if (targetChild == 2) {
        if (outCountC2 < kNumRays) {
          std::memcpy(childRaysOut2.data() + outCountC2 * RaySize, ray, RaySize);
          outCountC2++;
        }
      } else if (targetChild == 3) {
        if (outCountC3 < kNumRays) {
          std::memcpy(childRaysOut3.data() + outCountC3 * RaySize, ray, RaySize);
          outCountC3++;
        }
      } 
    };

    auto routeChildRays = [&](const poplar::Input<poplar::Vector<uint8_t>>& childIn) -> uint16_t {
      uint16_t count = 0;
      const int numChildRays = childIn.size() / RaySize;
      for (int i = 0; i < numChildRays; ++i) {
        const Ray* ray = reinterpret_cast<const Ray*>(childIn.data() + i * RaySize);
        if (ray->x == INVALID_RAY_ID) break;
        count++;
        routeRay(ray);
      }
      return count;
    };

    int mode = 1;
    if(mode == 0) { // Single ray test
      Ray genRay{};
      genRay.x = 343;
      genRay.y = 428;
      genRay.r = 0.0f;
      genRay.g = 0.0f;
      genRay.b = 0.0f;
      genRay.t = 0.0f;
      genRay.transmittance = 1.0f;
      genRay.next_cluster = cluster_id;
      genRay.next_local   = local_id;
      routeRay(&genRay);
    }
    if(mode == 1) { // Row scan
      const int interval = 3;
      const int nRowsPerFrame = 1;
      if(exec_count%interval == 0) {
        for(uint16_t x=0; x<kFullImageWidth; x++) {
          for(uint16_t y=0; y<nRowsPerFrame; y++) {
            Ray genRay{};

            genRay.x = x; //(x+(exec_count/interval)*3)%kFullImageWidth;
            genRay.y = (y+(exec_count/interval)*nRowsPerFrame)%kFullImageHeight;
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

    // Route each child and capture their input counts
    inCountC0 = routeChildRays(childRaysIn0);
    inCountC1 = routeChildRays(childRaysIn1);
    inCountC2 = routeChildRays(childRaysIn2);
    inCountC3 = routeChildRays(childRaysIn3);

    auto invalidateRemaining = [&](poplar::Output<poplar::Vector<uint8_t>>& out, uint16_t count) {
      for (uint16_t i = count; i < kNumRays; ++i) {
        Ray* ray = reinterpret_cast<Ray*>(out.data() + sizeof(Ray) * i);
        if(ray->x == 0xFFFF)
          break;
        ray->x = 0xFFFF;
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

class RayRouter : public poplar::Vertex {
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
  poplar::Input<poplar::Vector<unsigned short>> childClusterIds; // 4 IDs
  poplar::Input<uint8_t> level;

  poplar::Output<poplar::Vector<uint8_t>> debugBytes;

  bool compute() {
    const uint8_t lvl = *level;
    uint8_t shift = lvl * 2;
    constexpr int RaySize = sizeof(Ray);
    constexpr uint16_t INVALID_RAY_ID = 0xFFFF;
    const int kNumRays = parentRaysIn.size() / RaySize;

    // --- Counts ---
    uint16_t inCountParent = 0;
    uint16_t inCountC0 = 0, inCountC1 = 0, inCountC2 = 0, inCountC3 = 0;
    uint16_t outCountParent = 0;
    uint16_t outCountC0 = 0, outCountC1 = 0, outCountC2 = 0, outCountC3 = 0;

    // Helper: Determine which child a cluster ID belongs to
    auto findChildForCluster = [&](uint16_t clusterId) -> int {
      for (int i = 0; i < 4; ++i) {
        if ((childClusterIds[i] >> shift) == (clusterId >> shift))
            return i;
      }
      return -1;
    };

    // Helper: Route a ray
    auto routeRay = [&](const Ray* ray) {
      int targetChild = findChildForCluster(ray->next_cluster);
      if (targetChild == 0) {
        if (outCountC0 < kNumRays) {
          std::memcpy(childRaysOut0.data() + outCountC0 * RaySize, ray, RaySize);
          outCountC0++;
        }
      } else if (targetChild == 1) {
        if (outCountC1 < kNumRays) {
          std::memcpy(childRaysOut1.data() + outCountC1 * RaySize, ray, RaySize);
          outCountC1++;
        }
      } else if (targetChild == 2) {
        if (outCountC2 < kNumRays) {
          std::memcpy(childRaysOut2.data() + outCountC2 * RaySize, ray, RaySize);
          outCountC2++;
        }
      } else if (targetChild == 3) {
        if (outCountC3 < kNumRays) {
          std::memcpy(childRaysOut3.data() + outCountC3 * RaySize, ray, RaySize);
          outCountC3++;
        }
      } else {
        if (outCountParent < kNumRays) {
          std::memcpy(parentRaysOut.data() + outCountParent * RaySize, ray, RaySize);
          outCountParent++;
        }
      }
    };

    // Process parent rays
    {
      const int numParentRays = parentRaysIn.size() / RaySize;
      for (int i = 0; i < numParentRays; ++i) {
        const Ray* ray = reinterpret_cast<const Ray*>(parentRaysIn.data() + i * RaySize);
        if (ray->x == INVALID_RAY_ID) break;
        inCountParent++;
        routeRay(ray);
      }
    }

    // New routeChildRays: count + route, return count
    auto routeChildRays = [&](const poplar::Input<poplar::Vector<uint8_t>>& childIn) -> uint16_t {
      uint16_t count = 0;
      const int numChildRays = childIn.size() / RaySize;
      for (int i = 0; i < numChildRays; ++i) {
        const Ray* ray = reinterpret_cast<const Ray*>(childIn.data() + i * RaySize);
        if (ray->x == INVALID_RAY_ID) break;
        count++;
        routeRay(ray);
      }
      return count;
    };

    // Route each child and capture their input counts
    inCountC0 = routeChildRays(childRaysIn0);
    inCountC1 = routeChildRays(childRaysIn1);
    inCountC2 = routeChildRays(childRaysIn2);
    inCountC3 = routeChildRays(childRaysIn3);

    // Invalidate remaining rays
    auto invalidateRemaining = [&](poplar::Output<poplar::Vector<uint8_t>>& out, uint16_t count) {
      for (uint16_t i = count; i < kNumRays; i++) {
        Ray* ray = reinterpret_cast<Ray*>(out.data() + i * RaySize);
        if (ray->x == INVALID_RAY_ID) break;
        ray->x = INVALID_RAY_ID;
      }
    };
    invalidateRemaining(childRaysOut0, outCountC0);
    invalidateRemaining(childRaysOut1, outCountC1);
    invalidateRemaining(childRaysOut2, outCountC2);
    invalidateRemaining(childRaysOut3, outCountC3);
    invalidateRemaining(parentRaysOut, outCountParent);

    // Fill debugBytes (10 counts = 20 bytes)
    uint16_t* dbg = reinterpret_cast<uint16_t*>(debugBytes.data());
    dbg[0] = inCountParent;
    dbg[1] = inCountC0;
    dbg[2] = inCountC1;
    dbg[3] = inCountC2;
    dbg[4] = inCountC3;
    dbg[5] = outCountParent;
    dbg[6] = outCountC0;
    dbg[7] = outCountC1;
    dbg[8] = outCountC2;
    dbg[9] = outCountC3;

    dbg[10] = 0; // spare
    dbg[11] = 0; // spare

    return true;
  }

};


