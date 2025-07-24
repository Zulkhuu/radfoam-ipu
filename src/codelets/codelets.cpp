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
    constexpr int NeighborPointSize = sizeof(NeighborPoint); 
    glm::mat4 invView2 = glm::inverse(View2);
    glm::mat4 invProj2 = glm::inverse(Proj2);
    glm::vec3 rayOrigin = glm::vec3(invView2[3]);
    
    const LocalPoint* local_pt = readStructAt<LocalPoint>(local_pts, 0);  
    const NeighborPoint* nbr_pt = readStructAt<NeighborPoint>(neighbor_pts, 2);  

    glm::mat4 View = glm::transpose(glm::make_mat4(view_matrix.data()));
    glm::mat4 Proj = glm::transpose(glm::make_mat4(projection_matrix.data()));
    glm::mat4 invView = glm::inverse(View);
    glm::mat4 invProj = glm::inverse(Proj);
    //memcpy(&pt, local_pts.data()+LocalPointSize, LocalPointSize);
    // float p1x  = *reinterpret_cast<const float*>(&local_pts[LocalPointSize + 0]);
    // float p1y  = *reinterpret_cast<const float*>(&local_pts[LocalPointSize + 4]);

    int index = 1;
    Ray ray_in1  = *reinterpret_cast<const Ray*>(raysIn.data()+sizeof(Ray)*index);
    
    // *result_u16 = adjacency.size(); //local_pt->adj_end;
    // *result_float = nbr_pt->x;
    *result_u16 = ray_in1.x; //local_pt->adj_end;
    *result_float = View[0][2]; // local_pt->x;
    
    for (unsigned i = 10; i < framebuffer.size(); ++i)
      framebuffer[i] = (255 - tile_id/4 + ray_in1.x)%256;
    
    return true;
  }
};

class RayGen : public poplar::Vertex {
public:
  poplar::Input<poplar::Vector<uint8_t>> raysIn;
  poplar::Output<poplar::Vector<uint8_t>> raysOut;
  poplar::Input<unsigned> exec_count; 

  bool compute() {
    constexpr int RaySize = sizeof(Ray);  

    int index = 1;
    Ray* ray_out1  = reinterpret_cast<Ray*>(raysOut.data()+sizeof(Ray)*index);
    ray_out1->x = exec_count;
    ray_out1->r = 0.345;


    return true;
  }
};
