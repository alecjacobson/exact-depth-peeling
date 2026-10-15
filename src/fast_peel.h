#pragma once
#include "peel_options.h"
#include <array>
#include <boost/container/small_vector.hpp>
#include <cstdint>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace peel::fast {
using Json = nlohmann::json;
struct Point {
  int32_t x = 0, y = 0;
  bool operator==(Point b) const { return x == b.x && y == b.y; }
  bool operator!=(Point b) const { return !(*this == b); }
  bool operator<(Point b) const { return x == b.x ? y < b.y : x < b.x; }
};
using Ring = boost::container::small_vector<Point, 8>;
struct Patch {
  int source;
  Ring points;
}; // dense source index, not external ID
struct Source {
  int id, original_id;
  std::array<double, 3> score; // affine score in grid coordinates
  Ring initial;
};
struct Scene {
  bool perspective = false;
  int bits = 30, input_sources = 0, collapsed_sources = 0;
  double near = 0.01;
  std::array<double, 4> window;
  std::array<std::array<double, 4>, 3> inverse;
  std::vector<Source> sources;
  std::vector<std::array<Point, 2>> contours;
  Json topology;
};
struct Face {
  int source;
  std::array<std::array<double, 2>, 3> points;
};
struct Layer {
  std::vector<Patch> patches;
  std::vector<Face> faces;
  Json metrics;
};
struct Result {
  std::vector<Layer> layers;
  std::vector<Patch> residual;
  Json classification;
};
Json validate_reference(const std::string &path, const Scene &, const Result &, bool oracle,
                        int cap, bool complete);
int run_cli(const std::string &path, const std::string &out, int layers, int threads, int repeat,
            int bits, const std::string &order, unsigned seed, bool accelerate, bool metrics_only,
            bool verify, bool validate, int cap, bool resolved_mesh, bool mesh_only,
            PeelOptions options = {});
} // namespace peel::fast
