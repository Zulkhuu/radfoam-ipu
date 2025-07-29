#include <poplar/Vertex.hpp>

#include <glm/mat4x4.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <geometry/primitives.hpp>
#include <ipu/tile_config.hpp>

using namespace radfoam::geometry;

static const glm::mat4 View2(
    glm::vec4(-0.034899f,  0.000000f, -0.999391f, 0.000000f),
    glm::vec4( 0.484514f, -0.874620f, -0.016920f, 0.000000f),
    glm::vec4(-0.874087f, -0.484810f,  0.030524f, 0.000000f),
    glm::vec4(-0.000000f, -0.000000f, -6.700000f, 1.000000f)
);

static const glm::mat4 Proj2(
    glm::vec4(1.299038f, 0.000000f,  0.000000f,  0.000000f),
    glm::vec4(0.000000f, 1.732051f,  0.000000f,  0.000000f),
    glm::vec4(0.000000f, 0.000000f, -1.002002f, -1.000000f),
    glm::vec4(0.000000f, 0.000000f, -0.200200f,  0.000000f)
);

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

class RayTrace : public poplar::Vertex {
public:
  poplar::Input<poplar::Vector<float>> view_matrix;
  poplar::Input<poplar::Vector<float>> projection_matrix;

  poplar::Input<poplar::Vector<uint8_t>> local_pts;
  poplar::Input<poplar::Vector<uint8_t>> neighbor_pts;
  poplar::Input<poplar::Vector<unsigned short>> adjacency;
  poplar::Input<unsigned short> tile_id;

  poplar::Input<poplar::Vector<uint8_t>> raysIn;
  poplar::Output<poplar::Vector<uint8_t>> raysOut;

  poplar::InOut<poplar::Vector<uint8_t>> framebuffer;
  
  poplar::Output<float> result_float;
  poplar::Output<unsigned short> result_u16;
  
  [[poplar::constraint("elem(*local_pts)!=elem(*framebuffer)")]]
  bool compute() {
    constexpr int RaySize = sizeof(Ray);
    constexpr int LocalPointSize = sizeof(LocalPoint);
    constexpr int GenericPointSize = sizeof(GenericPoint); 
    glm::mat4 invView2 = glm::inverse(View2);
    // glm::mat4 invProj2 = glm::inverse(Proj2);
    // glm::vec3 rayOrigin = glm::vec3(invView2[3]);
    const uint16_t nLocalPts = local_pts.size() / LocalPointSize;
    const uint16_t nNeighborPts = neighbor_pts.size() / GenericPointSize;
    const uint16_t nTotalPts = nLocalPts + nNeighborPts;

    const LocalPoint* local_pt = readStructAt<LocalPoint>(local_pts, 0);  
    const GenericPoint* nbr_pt = readStructAt<GenericPoint>(neighbor_pts, 2);  

    glm::mat4 View = glm::transpose(glm::make_mat4(view_matrix.data()));
    glm::mat4 Proj = glm::transpose(glm::make_mat4(projection_matrix.data()));
    glm::mat4 invView = glm::inverse(View);
    glm::mat4 invProj = glm::inverse(Proj);
    glm::vec3 rayOrigin = glm::vec3(invView[3]);

    int index = 0;
    const Ray* ray_in1 = readStructAt<Ray>(raysIn, index); 
    Ray* ray_out1 = reinterpret_cast<Ray*>(raysOut.data()+sizeof(Ray)*index);
    
    uint16_t local_id = ray_in1->next_local;
    uint16_t cluster_id = ray_in1->next_cluster;
    
    const uint16_t& x = ray_in1->x;
    const uint16_t& y = ray_in1->y;

    float ndcX = (2.0f * x) / kFullImageWidth - 1.0f;
    float ndcY = 1.0f - (2.0f * y) / kFullImageHeight;
    glm::vec4 clipRay(ndcX, ndcY, -1.0f, 1.0f);
    glm::vec4 eyeRay = invProj * clipRay;
    eyeRay.z = -1.0f;
    eyeRay.w = 0.0f;
    glm::vec3 rayDir = glm::normalize(glm::vec3(invView * eyeRay));
    
    glm::vec3 color(ray_in1->r, ray_in1->g, ray_in1->b);
    float transmittance = ray_in1->transmittance;
    float t0 = ray_in1->t;
    int current = local_id;

    *result_u16 = 65535;
    int cntr = 1;
    while (true) {
        const LocalPoint* cur_cell = readStructAt<LocalPoint>(local_pts, current); 
        glm::vec3 currentPos(cur_cell->x, cur_cell->y, cur_cell->z);

        uint16_t end = cur_cell->adj_end;
        uint16_t start;
        if(current == 0) {
          start = 0;
        } else {
          const LocalPoint* prev = readStructAt<LocalPoint>(local_pts, current-1); 
          start = prev->adj_end;
        }

        framebuffer[2 * cntr]     = static_cast<uint8_t>((current >> 8) & 0xFF);
        framebuffer[2 * cntr + 1] = static_cast<uint8_t>(current & 0xFF);

        float closestT = std::numeric_limits<float>::max();
        int nextIdx = -1;

        for (uint16_t j = start; j < end; ++j) {
          uint16_t neighborIdx = adjacency[j];
          glm::vec3 nbrPos;
          if (neighborIdx < nLocalPts) {
            // neighbor is a local point in the cluster     
            const LocalPoint* nbrPt = readStructAt<LocalPoint>(local_pts, neighborIdx);
            nbrPos.x = nbrPt->x;
            nbrPos.y = nbrPt->y;
            nbrPos.z = nbrPt->z;
          } else {
            // neighbor is a point from neighboring cluster
            const GenericPoint* nbrPt = readStructAt<GenericPoint>(neighbor_pts, neighborIdx-nLocalPts);
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

          if (t > 0 && t < closestT) {
              closestT = t;
              nextIdx = neighborIdx;
          }
        }       
        
        float delta = closestT - t0;
        float alpha = 1.0f - expf(-cur_cell->density * delta);
        // glm::vec3 pointColor = glm::vec3(cur_cell->r, cur_cell->g, cur_cell->b) / 255.0f;

        color.x += transmittance * alpha * (cur_cell->r/255.0f);
        color.y += transmittance * alpha * (cur_cell->g/255.0f);
        color.z += transmittance * alpha * (cur_cell->b/255.0f);

        transmittance *= (1.0f - alpha);

        t0 = __builtin_fmaxf(t0, closestT);

        if (transmittance < 0.01f || nextIdx == -1 || nextIdx >= nLocalPts) {
          // finished ray tracing for this one
          // transmittance < 0.01f: too opaque, light cant pass anymore
          // nextIdx == -1: ray has left the entire scene, depth = inf
          if(nextIdx >= nLocalPts) {
            // nextIdx >= nLocalPts : ray moves to next cluster/tile
            const GenericPoint* nbrPt = readStructAt<GenericPoint>(neighbor_pts, nextIdx-nLocalPts);
            *result_u16 = nbrPt->cluster_id; 
            *result_float = color.z;
            ray_out1->next_cluster = nbrPt->cluster_id; 
            if(tile_id == 521) {
              *result_u16 = ray_in1->y; 
              *result_float = color.z;
              ray_out1->next_cluster = 523; 
            } 
            if(tile_id == 523) {
              *result_u16 = ray_in1->y; 
              *result_float = color.z;
              ray_out1->next_cluster = 520; 
            } 


            ray_out1->next_local = nbrPt->local_id; 
            ray_out1->transmittance = transmittance; 
            ray_out1->x = ray_in1->x;
            ray_out1->y = ray_in1->y;
            ray_out1->t = t0;
            ray_out1->r = color.x;
            ray_out1->g = color.y;
            ray_out1->b = color.z;
          } else {
            if(nextIdx == -1)
              *result_u16 = 65534; 
          }
          break;
        }

        current = nextIdx;
        cntr++;
    }
    framebuffer[0] = cntr & 0xFF;

    // *result_u16 = nNeighborPts; // y;
    // *result_float = rayDir.x;

    // for (unsigned i = 0; i < framebuffer.size(); ++i)
    //   framebuffer[i] = (255 - tile_id/4 + ray_in1->x)%256;
  
    for(int i=1; i<kNumRays; i++) {
      Ray* ray_ = reinterpret_cast<Ray*>(raysOut.data()+sizeof(Ray)*i);
      ray_->x = 0xFFFF;
    }
    return true;
  }
};

class RayGen : public poplar::Vertex {
public:
  poplar::Input<poplar::Vector<uint8_t>> raysIn;
  poplar::Output<poplar::Vector<uint8_t>> raysOut;
  poplar::Input<unsigned> exec_count; 
  poplar::Input<poplar::Vector<uint8_t>> camera_cell_info;

  bool compute() {
    constexpr int RaySize = sizeof(Ray);  

    uint16_t cluster_id = camera_cell_info[0] | (camera_cell_info[1] << 8);
    uint16_t local_id   = camera_cell_info[2] | (camera_cell_info[3] << 8);

    int index = 0;
    Ray* ray_ = reinterpret_cast<Ray*>(raysOut.data()+sizeof(Ray)*index);

    ray_->x = 14;
    ray_->y = 300;
    ray_->r = 0.0;
    ray_->g = 0.0;
    ray_->b = 0.0;
    ray_->t = 0.0;
    ray_->transmittance = 1.0f;
    ray_->next_cluster = cluster_id;
    ray_->next_local = local_id;

    for(int i=1; i<kNumRays; i++) {
      Ray* ray_ = reinterpret_cast<Ray*>(raysOut.data()+sizeof(Ray)*i);
      ray_->x = 0xFFFF;
    }
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
        if ((childClusterIds[i] >> lvl) == (clusterId >> lvl))
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
      uint16_t invalidRay = INVALID_RAY_ID;
      for (uint16_t i = count; i < kNumRays; i++) {
        *reinterpret_cast<uint16_t*>(out.data() + i * RaySize) = invalidRay;
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


