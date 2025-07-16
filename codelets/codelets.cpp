#include <poplar/Vertex.hpp>

#include <glm/mat4x4.hpp>
#include <glm/gtc/type_ptr.hpp>

struct LocalPoint {
    float   x, y, z;
    uint8_t r, g, b, _pad;
    float   density;
    uint32_t adj_end;
};

struct NeighborPoint {
    float x, y, z;
    uint32_t gid;
};

struct Ray {
  uint16_t x, y;         // pixel coords
  float    t;            // ray distance
  float    transmittance;
  float    r, g, b;      // accumulated color
  uint32_t next_cell;    // for traversal
};

static const glm::mat4 View(
    glm::vec4(-0.034899f,  0.000000f, -0.999391f, 0.000000f),
    glm::vec4( 0.484514f, -0.874620f, -0.016920f, 0.000000f),
    glm::vec4(-0.874087f, -0.484810f,  0.030524f, 0.000000f),
    glm::vec4(-0.000000f, -0.000000f, -6.700000f, 1.000000f)
);

static const glm::mat4 Proj(
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
  poplar::Input<poplar::Vector<uint8_t>> local_pts;
  poplar::Input<poplar::Vector<uint8_t>> neighbor_pts;
  poplar::Input<poplar::Vector<unsigned short>> adjacency;
  poplar::Input<poplar::Vector<unsigned short>> scene_sizes;

  poplar::Input<poplar::Vector<uint8_t>> raysIn;
  poplar::Output<poplar::Vector<uint8_t>> raysOut;
  
  poplar::Output<float> result_float;
  poplar::Output<unsigned short> result_u16;

  bool compute() {
    constexpr int RaySize = sizeof(Ray);
    constexpr int LocalPointSize = sizeof(LocalPoint);
    constexpr int NeighborPointSize = sizeof(NeighborPoint); 
    glm::mat4 invView = glm::inverse(View);
    glm::mat4 invProj = glm::inverse(Proj);
    glm::vec3 rayOrigin = glm::vec3(invView[3]);

    const LocalPoint* local_pt = readStructAt<LocalPoint>(local_pts, 1000);  
    const NeighborPoint* nbr_pt = readStructAt<NeighborPoint>(neighbor_pts, 2);  

    //memcpy(&pt, local_pts.data()+LocalPointSize, LocalPointSize);
    // float p1x  = *reinterpret_cast<const float*>(&local_pts[LocalPointSize + 0]);
    // float p1y  = *reinterpret_cast<const float*>(&local_pts[LocalPointSize + 4]);

    int index = 1;
    Ray ray_in1  = *reinterpret_cast<const Ray*>(raysIn.data()+sizeof(Ray)*index);

    // *result_u16 = adjacency.size(); //local_pt->adj_end;
    // *result_float = nbr_pt->x;
    *result_u16 = ray_in1.x; //local_pt->adj_end;
    *result_float = invView[3][0];
    return true;
  }
};


class RayGen : public poplar::Vertex {
public:
  poplar::Input<poplar::Vector<uint8_t>> raysIn;
  poplar::Output<poplar::Vector<uint8_t>> raysOut;

  poplar::InOut<poplar::Vector<uint8_t>> framebuffer;

  bool compute() {
    constexpr int RaySize = sizeof(Ray);  
    constexpr int fb_width = 128;   
    constexpr int fb_height = 72;   

    int index = 1;
    Ray* ray_out1  = reinterpret_cast<Ray*>(raysOut.data()+sizeof(Ray)*index);
    ray_out1->x = 12;
    ray_out1->r = 0.345;

    for (unsigned i = 0; i < 12; ++i)
      framebuffer[i] = i*2;

    return true;
  }
};
