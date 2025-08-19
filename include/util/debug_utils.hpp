#pragma once
#include <sstream>
#include <vector>
#include <glm/glm.hpp>
#include <spdlog/fmt/fmt.h>

template <>
struct fmt::formatter<glm::mat4> {
    constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

    template <typename FormatContext>
    auto format(const glm::mat4& mat, FormatContext& ctx) {
        for (int r = 0; r < 4; r++) {
            fmt::format_to(ctx.out(), "[{: .6f}, {: .6f}, {: .6f}, {: .6f}]{}\n",
                           mat[r][0], mat[r][1], mat[r][2], mat[r][3],
                           (r == 3 ? "" : ""));
        }
        return ctx.out();
    }
};

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

