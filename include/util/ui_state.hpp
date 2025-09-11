// util/ui_state.hpp
#pragma once
#include <type_traits>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <glm/vec3.hpp>   // for glm::vec3

namespace rf::ui {

// --------- detection idiom for optional State members ----------
template<typename T, typename = void> struct has_fov    : std::false_type{};
template<typename T> struct has_fov<T,    std::void_t<decltype(std::declval<T>().fov)>> : std::true_type{};

template<typename T, typename = void> struct has_mode   : std::false_type{};
template<typename T> struct has_mode<T,   std::void_t<decltype(std::declval<T>().mode)>> : std::true_type{};

template<typename T, typename = void> struct has_stop   : std::false_type{};
template<typename T> struct has_stop<T,   std::void_t<decltype(std::declval<T>().stop)>> : std::true_type{};

template<typename T, typename = void> struct has_envYaw : std::false_type{};
template<typename T> struct has_envYaw<T, std::void_t<decltype(std::declval<T>().envRotationDegrees)>> : std::true_type{};

template<typename T, typename = void> struct has_envPitch : std::false_type{};
template<typename T> struct has_envPitch<T, std::void_t<decltype(std::declval<T>().envRotationDegrees2)>> : std::true_type{};

template<typename T, typename = void> struct has_X : std::false_type{};
template<typename T> struct has_X<T, std::void_t<decltype(std::declval<T>().X)>> : std::true_type{};
template<typename T, typename = void> struct has_Y : std::false_type{};
template<typename T> struct has_Y<T, std::void_t<decltype(std::declval<T>().Y)>> : std::true_type{};
template<typename T, typename = void> struct has_Z : std::false_type{};
template<typename T> struct has_Z<T, std::void_t<decltype(std::declval<T>().Z)>> : std::true_type{};

template<typename T, typename = void> struct has_eye    : std::false_type{};
template<typename T> struct has_eye<T,    std::void_t<decltype(std::declval<T>().eye)>> : std::true_type{};
template<typename T, typename = void> struct has_center : std::false_type{};
template<typename T> struct has_center<T, std::void_t<decltype(std::declval<T>().center)>> : std::true_type{};
template<typename T, typename = void> struct has_up     : std::false_type{};
template<typename T> struct has_up<T,     std::void_t<decltype(std::declval<T>().up)>> : std::true_type{};

// --------- small helpers ----------
inline std::string vec3_to_csv(const glm::vec3& v) {
  std::ostringstream ss; ss << v.x << "," << v.y << "," << v.z; return ss.str();
}
inline glm::vec3 csv_to_vec3(const std::string& s) {
  glm::vec3 v{0}; char c;
  std::istringstream ss(s);
  if (!(ss >> v.x)) return v;
  if (ss >> c && c == ',') ss >> v.y;
  if (ss >> c && c == ',') ss >> v.z;
  return v;
}

// --------- main API ----------
template<typename StateT>
inline void SaveStateToFile(const StateT& s, const std::string& path) {
  std::ofstream out(path, std::ios::trunc);
  if (!out) { std::cerr << "Failed to open UI state file for write: " << path << "\n"; return; }
  out << "# RadiantFoam UI State\n";
  if constexpr (has_fov<StateT>::value)        out << "fov=" << s.fov << "\n";
  if constexpr (has_mode<StateT>::value)       out << "mode=" << s.mode << "\n";
  if constexpr (has_stop<StateT>::value)       out << "stop=" << (s.stop ? 1 : 0) << "\n";
  if constexpr (has_envYaw<StateT>::value)     out << "envRotationDegrees="  << s.envRotationDegrees  << "\n";
  if constexpr (has_envPitch<StateT>::value)   out << "envRotationDegrees2=" << s.envRotationDegrees2 << "\n";
  if constexpr (has_X<StateT>::value)          out << "X=" << s.X << "\n";
  if constexpr (has_Y<StateT>::value)          out << "Y=" << s.Y << "\n";
  if constexpr (has_Z<StateT>::value)          out << "Z=" << s.Z << "\n";
}

template<typename StateT>
inline bool LoadStateFromFile(StateT& s, const std::string& path) {
  std::ifstream in(path);
  if (!in) return false;

  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    auto eq = line.find('=');
    if (eq == std::string::npos) continue;
    std::string k = line.substr(0, eq);
    std::string v = line.substr(eq + 1);

    if constexpr (has_fov<StateT>::value)      { if (k=="fov") { s.fov = std::stof(v);        continue; } }
    if constexpr (has_mode<StateT>::value)     { if (k=="mode"){ s.mode = v;                  continue; } }
    if constexpr (has_stop<StateT>::value)     { if (k=="stop"){ s.stop = (v=="1"||v=="true");continue; } }
    if constexpr (has_envYaw<StateT>::value)   { if (k=="envRotationDegrees")  { s.envRotationDegrees  = std::stof(v); continue; } }
    if constexpr (has_envPitch<StateT>::value) { if (k=="envRotationDegrees2") { s.envRotationDegrees2 = std::stof(v); continue; } }
    if constexpr (has_X<StateT>::value)        { if (k=="X") { s.X = std::stof(v); continue; } }
    if constexpr (has_Y<StateT>::value)        { if (k=="Y") { s.Y = std::stof(v); continue; } }
    if constexpr (has_Z<StateT>::value)        { if (k=="Z") { s.Z = std::stof(v); continue; } }
  }
  return true;
}

} // namespace rf::ui
