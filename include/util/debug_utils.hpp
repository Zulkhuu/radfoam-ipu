#pragma once
#include <sstream>
#include <vector>

namespace radfoam::util {

template <typename T>
std::string VectorSliceToString(const std::vector<T>& v,
                                size_t begin, size_t end) {
  std::ostringstream oss;
  oss << '[';
  if (begin >= v.size()) return oss.str() + ']';

  end = std::min(end, v.size());
  for (size_t i = begin; i < end; ++i) {
    if constexpr (std::is_same_v<T, uint8_t>)
      oss << static_cast<int>(v[i]);
    else
      oss << v[i];
    if (i + 1 < end) oss << ", ";
  }
  oss << ']';
  return oss.str();
}

bool isPoplarEngineOptionsEnabled() {
    const char* env = std::getenv("POPLAR_ENGINE_OPTIONS");
    return (env != nullptr && std::string(env).find("autoReport") != std::string::npos);
}

}  // namespace radfoam::util
