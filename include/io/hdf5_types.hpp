#pragma once

#include "geometry/primitives.hpp"
#include <highfive/H5File.hpp>
#include <highfive/H5DataType.hpp>

namespace h5  = HighFive;
using radfoam::geometry::LocalPoint;
using radfoam::geometry::GenericPoint;

namespace radfoam::io {

inline HighFive::CompoundType MakeLocalPointType() {
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

inline HighFive::CompoundType MakeGenericPointType() {
  using namespace HighFive;
  return CompoundType{
      {"x", create_datatype<float>(),   offsetof(GenericPoint, x)},
      {"y", create_datatype<float>(),   offsetof(GenericPoint, y)},
      {"z", create_datatype<float>(),   offsetof(GenericPoint, z)},
      {"cluster_id", create_datatype<uint16_t>(), offsetof(GenericPoint, cluster_id)},
      {"local_id", create_datatype<uint16_t>(), offsetof(GenericPoint, local_id)}};
}

}  // namespace radfoam::io

HIGHFIVE_REGISTER_TYPE(LocalPoint,   radfoam::io::MakeLocalPointType)
HIGHFIVE_REGISTER_TYPE(GenericPoint, radfoam::io::MakeGenericPointType)
