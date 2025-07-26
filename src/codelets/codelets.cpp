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
  
  bool compute() {
    constexpr int RaySize = sizeof(Ray);
    constexpr int LocalPointSize = sizeof(LocalPoint);
    constexpr int GenericPointSize = sizeof(GenericPoint); 
    // glm::mat4 invView2 = glm::inverse(View2);
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

    int index = 1;
    const Ray* ray_in1 = readStructAt<Ray>(raysIn, 1); 
    
    uint16_t local_id = ray_in1->next_cell & 0xFFFF;
    uint16_t cluster_id = (ray_in1->next_cell >> 16) & 0xFFFF;
    
    const uint16_t& x = ray_in1->x;
    const uint16_t& y = ray_in1->y;

    float ndcX = (2.0f * x) / kFullImageWidth - 1.0f;
    float ndcY = 1.0f - (2.0f * y) / kFullImageHeight;
    glm::vec4 clipRay(ndcX, ndcY, -1.0f, 1.0f);
    glm::vec4 eyeRay = invProj * clipRay;
    eyeRay.z = -1.0f;
    eyeRay.w = 0.0f;
    glm::vec3 rayDir = glm::normalize(glm::vec3(invView * eyeRay));
        
    float t0 = ray_in1->t;
    int current = local_id;

    *result_u16 = 0;
    int cntr = 1;
    while (true) {
        const LocalPoint* p0 = readStructAt<LocalPoint>(local_pts, current); 
        glm::vec3 currentPos(p0->x, p0->y, p0->z);

        uint16_t end = p0->adj_end;
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

        if (ray_in1->transmittance < 0.01f || nextIdx == -1 || nextIdx >= nLocalPts) {
          // finished ray tracing for this one
          // ray_in1->transmittance < 0.01f: too opaque, light cant pass anymore
          // nextIdx == -1: ray has left the entire scene, depth = inf
          if(nextIdx >= nLocalPts) {
            // nextIdx >= nLocalPts : ray moves to next cluster/tile
            const GenericPoint* nbrPt = readStructAt<GenericPoint>(neighbor_pts, nextIdx-nLocalPts);
            *result_u16 = nbrPt->local_id; 
          }
          break;
        }

        t0 = std::fmax(t0, closestT);
        current = nextIdx;
        cntr++;
    }
    framebuffer[0] = cntr & 0xFF;

    // *result_u16 = nNeighborPts; // y;
    *result_float = rayDir.x;

    // for (unsigned i = 0; i < framebuffer.size(); ++i)
    //   framebuffer[i] = (255 - tile_id/4 + ray_in1->x)%256;
    
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

    int index = 1;
    Ray* ray_ = reinterpret_cast<Ray*>(raysOut.data()+sizeof(Ray)*index);

    ray_->x = 14;
    ray_->y = 28;
    ray_->r = 0.0;
    ray_->g = 0.0;
    ray_->b = 0.0;
    ray_->transmittance = 1.0f;
    ray_->next_cell = cluster_id << 16 | local_id;

    return true;
  }
};
