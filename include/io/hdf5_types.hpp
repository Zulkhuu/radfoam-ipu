#include "geometry/primitives.hpp"
#include <highfive/H5File.hpp>
#include <highfive/H5DataType.hpp>

namespace h5  = HighFive;
using radfoam::geometry::LocalPoint;
using radfoam::geometry::NeighborPoint;

namespace radfoam::io {

HighFive::CompoundType MakeLocalPointType() {
  using namespace HighFive;
  return CompoundType{
      {"x", create_datatype<float>(),   offsetof(LocalPoint, x)},
      {"y", create_datatype<float>(),   offsetof(LocalPoint, y)},
      {"z", create_datatype<float>(),   offsetof(LocalPoint, z)},
      {"r", create_datatype<uint8_t>(), offsetof(LocalPoint, r)},
      {"g", create_datatype<uint8_t>(), offsetof(LocalPoint, g)},
      {"b", create_datatype<uint8_t>(), offsetof(LocalPoint, b)},
      {"_pad", create_datatype<uint8_t>(), offsetof(LocalPoint, _pad)},
      {"density", create_datatype<float>(), offsetof(LocalPoint, density)},
      {"adj_end", create_datatype<uint32_t>(), offsetof(LocalPoint, adj_end)}};
}

HighFive::CompoundType MakeNeighborPointType() {
  using namespace HighFive;
  return CompoundType{
      {"x", create_datatype<float>(),   offsetof(NeighborPoint, x)},
      {"y", create_datatype<float>(),   offsetof(NeighborPoint, y)},
      {"z", create_datatype<float>(),   offsetof(NeighborPoint, z)},
      {"gid", create_datatype<uint32_t>(), offsetof(NeighborPoint, gid)}};
}

}  // namespace radfoam::io

HIGHFIVE_REGISTER_TYPE(LocalPoint,   radfoam::io::MakeLocalPointType)
HIGHFIVE_REGISTER_TYPE(NeighborPoint,radfoam::io::MakeNeighborPointType)
