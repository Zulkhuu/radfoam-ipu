#include <ipudef.h>
#include <ipu_builtins.h>
#include <ipu_vector_math>
#include <ipu_memory_intrinsics>

inline constexpr uint16_t INVALID_RAY_ID = 0xFFFF;
inline constexpr uint16_t FINISHED_RAY_ID = 0xFFFE;
inline constexpr unsigned NW = 6;

namespace ipu::geometry {

struct alignas(4) Ray {
uint16_t x, y;
half t, transmittance;
half r, g, b, d;
uint16_t next_cluster; 
uint16_t next_local;
};


}