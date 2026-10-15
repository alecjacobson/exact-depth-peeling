#pragma once
#include "peel_options.h"
#include <CGAL/Cartesian.h>
#include <CGAL/Exact_rational.h>
#include <CGAL/Lazy_exact_nt.h>
#include <CGAL/Polygon_set_2.h>
#include <array>
#include <chrono>
#include <map>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
namespace peel {
#ifdef PEEL_LAZY_EXACT
using Q = CGAL::Lazy_exact_nt<CGAL::Exact_rational>;
#else
using Q = CGAL::Exact_rational;
#endif
using K = CGAL::Cartesian<Q>;
using P = K::Point_2;
using P3 = K::Point_3;
using Poly = CGAL::Polygon_2<K>;
using Pwh = CGAL::Polygon_with_holes_2<K>;
using Set = CGAL::Polygon_set_2<K>;
using Json = nlohmann::json;
using Domains = std::map<int, Set>;
using Clock = std::chrono::steady_clock;
inline double seconds(Clock::time_point t) {
  return std::chrono::duration<double>(Clock::now() - t).count();
}
struct Score {
  Q a = 0, b = 0, c = 0;
  Q at(const P &p) const { return a * p.x() + b * p.y() + c; }
};
struct Box {
  Q xmin, ymin, xmax, ymax;
  std::array<P, 4> corners() const {
    return {P(xmin, ymin), P(xmax, ymin), P(xmax, ymax), P(xmin, ymax)};
  }
};
struct Camera {
  bool perspective = false;
  Q near = Q(1) / 100;
  std::array<std::array<Q, 4>, 3> world_to_camera{}, camera_to_world{};
};
struct Source {
  int id, mesh_face;
  std::array<P3, 3> world, cam;
  Score score;
  Set initial;
  Poly initial_polygon;
};
struct Scene {
  Camera camera;
  Box window;
  std::vector<Source> sources;
  std::map<int, size_t> source_lookup;
  int input_sources = 0, zero_area = 0, clipped_empty = 0;
  Json manifest, topology;
  std::vector<std::array<P, 2>> contours;
  const Source &source(int id) const;
};
struct Face {
  int source;
  std::array<P, 3> points;
};
struct Patch {
  int source;
  std::vector<P> points;
};
struct Stats {
  long candidate_tests = 0, box_rejections = 0, depth_separated = 0, occlusion_rejections = 0,
       index_box_tests = 0, index_rebuilds = 0, certificate_tests = 0, intersections = 0,
       differences = 0, joins = 0, halfplane_clips = 0;
  long adapter_triangles = 0, result_boundary_vertices = 0, result_boundary_edges = 0,
       max_map_vertices = 0, peak_live_boundary_vertices = 0, arrangement_vertices = 0,
       arrangement_edges = 0, arrangement_faces = 0, raw_triangles = 0, raw_regions = 0,
       raw_boundary_vertices = 0;
  double adapter_seconds = 0, envelope_construction_seconds = 0, normalization_seconds = 0,
         visibility_seconds = 0, residual_seconds = 0, triangulation_seconds = 0,
         validation_seconds = 0;
  Json json() const;
};
struct Layer {
  Domains regions;
  std::vector<Patch> patches;
  std::vector<Face> faces;
  Stats stats;
  Json residual_complexity;
  Json phase_metrics;
};
struct Result {
  std::vector<Layer> layers;
  Domains residual;
  Stats eager_stats;
  Json classification;
  Json global_statistics;
  Json oracle_cells = Json::array();
  std::vector<Patch> residual_patches;
  bool convex_representation = false;
};
Q rational(const Json &j);
std::string str(const Q &q);
Json jp(const P &p);
Json jp3(const P3 &p);
Poly polygon(std::vector<P> p);
Set domain(const Poly &p);
Set regularized_cycle(const std::vector<P> &ring);
std::vector<Pwh> pieces(const Set &s);
Set intersect(const Set &a, const Set &b, Stats *stats = nullptr);
Set subtract(const Set &a, const Set &b, Stats *stats = nullptr);
void unite(Set &a, const Set &b, Stats *stats = nullptr);
bool equal(const Set &a, const Set &b);
bool contains(const Set &a, const P &p);
long vertices(const Set &s);
long vertices(const Domains &d);
Q area(const Set &s);
Json complexity(const Domains &d);
Box bounds(const Set &s);
bool overlaps(const Box &a, const Box &b);
Set halfplane(const Score &s, const Box &w);
Score difference(const Score &a, const Score &b);
Json set_json(const Set &s);
Set read_set(const Json &j);
Scene read_scene(const std::string &path, bool convex_only = false, bool resolved_mesh = false);
void materialize_scene(Scene &s);
P3 lift(const Scene &s, const Source &src, const P &p);
std::array<Q, 3> barycentric(const Source &s, const P3 &p);
P project(const Scene &s, const P3 &p);
std::vector<Face> triangulate(const Domains &regions, Stats *stats = nullptr);
std::vector<K::Segment_2> noded_edges(const Domains &d, Stats *stats = nullptr);
Domains visible(const Scene &s, const Domains &r, const std::string &order, unsigned seed,
                bool accelerate, Stats &stats);
Domains envelope(const Scene &s, const Domains &r, Stats &stats);
Result arrangement(const Scene &s, int layers, int cap = 64);
// Scene must have contours extracted (read_scene third argument true), even on unresolved input.
Result global_arrangement(const Scene &s, int layers, int threads, bool accelerate = true,
                          bool use_contours = true);
std::vector<Face> triangulate_patches(const std::vector<Patch> &patches, int threads);
Result run(const Scene &s, const std::string &method, int layers, const std::string &order,
           unsigned seed, bool accelerate, int cap = 64);
void validate(const Scene &s, Result &r, bool complete);
void compare(const Result &a, const Result &b);
void compare_files(const std::string &a, const std::string &b);
void write_result(const Scene &s, Result &r, const std::string &path, Json metadata);
Result bvh_peel(const Scene &s, int layers, const std::string &order, unsigned seed,
                bool accelerate, int threads, bool incremental = false, PeelOptions options = {});
void materialize_regions(Result &r);
Json audit_convex(const Scene &s, const Result &r, bool complete);
void write_convex_result(const Scene &s, Result &r, const std::string &path, Json metadata,
                         bool meshes);
} // namespace peel
