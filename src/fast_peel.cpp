#include "fast_peel.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <sys/resource.h>

namespace peel::fast {
namespace {
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point t) {
  return std::chrono::duration<double>(Clock::now() - t).count();
}
using V3 = std::array<double, 3>;
using V2 = std::array<double, 2>;
using Matrix = std::array<std::array<double, 4>, 3>;
int64_t side(Point a, Point b, Point p) {
  return int64_t(b.x - a.x) * (p.y - a.y) - int64_t(b.y - a.y) * (p.x - a.x);
}
int64_t area2(const Ring &r) {
  int64_t area = 0;
  for (size_t j = 1; j + 1 < r.size(); ++j)
    area += side(r[0], r[j], r[j + 1]);
  return area;
}
struct Counts {
  uint64_t candidates = 0, nodes = 0, cuts = 0, rejected = 0, repairs = 0, collapsed = 0, ties = 0;
  void add(const Counts &b) {
    candidates += b.candidates;
    nodes += b.nodes;
    cuts += b.cuts;
    rejected += b.rejected;
    repairs += b.repairs;
    collapsed += b.collapsed;
    ties += b.ties;
  }
};
// All coordinates lie in [0,2^bits], bits<=30. Integer determinants cannot overflow.
void clean(Ring &r, Counts &c) {
  auto end = std::unique(r.begin(), r.end());
  r.erase(end, r.end());
  if (r.size() > 1 && r.front() == r.back())
    r.pop_back();
  if (r.size() < 3) {
    if (!r.empty())
      ++c.collapsed;
    r.clear();
    return;
  }
  bool convex = true;
  for (size_t i = 0; i < r.size(); ++i)
    if (side(r[i], r[(i + 1) % r.size()], r[(i + 2) % r.size()]) < 0) {
      convex = false;
      break;
    }
  if (!convex) {
    ++c.repairs;
    std::sort(r.begin(), r.end());
    r.erase(std::unique(r.begin(), r.end()), r.end());
    Ring hull;
    for (auto p : r) {
      while (hull.size() > 1 && side(hull[hull.size() - 2], hull.back(), p) <= 0)
        hull.pop_back();
      hull.push_back(p);
    }
    auto lower = hull.size();
    for (auto i = r.rbegin() + 1; i != r.rend(); ++i) {
      while (hull.size() > lower && side(hull[hull.size() - 2], hull.back(), *i) <= 0)
        hull.pop_back();
      hull.push_back(*i);
    }
    hull.pop_back();
    r = std::move(hull);
  }
  if (r.size() < 3 || area2(r) <= 2) {
    ++c.collapsed;
    r.clear();
  }
  // Rounded partitions can leave long one-cell ribbons. Treat these as below
  // the fast mode's spatial resolution, rather than re-peeling them forever.
  for (size_t i = 0; i < r.size(); ++i) {
    auto a = r[i], b = r[(i + 1) % r.size()];
    int64_t threshold = 2 * (int64_t(std::abs(b.x - a.x)) + std::abs(b.y - a.y));
    if (threshold == 0)
      continue;
    int64_t maximum = 0;
    for (auto p : r)
      maximum = std::max(maximum, side(a, b, p));
    if (maximum <= threshold) {
      ++c.collapsed;
      r.clear();
      break;
    }
  }
}
double number(const Json &j) {
  double x;
  if (j.is_string()) {
    auto v = j.get<std::string>();
    auto slash = v.find('/');
    size_t used = 0;
    x = std::stod(v.substr(0, slash), &used);
    if (used != v.substr(0, slash).size())
      throw std::runtime_error("Invalid numeric input");
    if (slash != std::string::npos) {
      auto tail = v.substr(slash + 1);
      auto denominator = std::stod(tail, &used);
      if (used != tail.size() || denominator == 0)
        throw std::runtime_error("Invalid denominator");
      x /= denominator;
    }
  } else if (j.is_number())
    x = j.get<double>();
  else
    throw std::runtime_error("Expected numeric coordinate");
  if (!std::isfinite(x))
    throw std::runtime_error("Input outside finite binary64 range");
  return x;
}
V3 transform(const Matrix &m, V3 p) {
  V3 q{};
  for (int i = 0; i < 3; ++i)
    q[i] = m[i][0] * p[0] + m[i][1] * p[1] + m[i][2] * p[2] + m[i][3];
  return q;
}
template <class V, class F> std::vector<V> clip_float(const std::vector<V> &p, F f) {
  std::vector<V> out;
  for (size_t i = 0; i < p.size(); ++i) {
    auto a = p[i], b = p[(i + 1) % p.size()];
    double da = f(a), db = f(b);
    if (da >= 0)
      out.push_back(a);
    if ((da < 0 && db > 0) || (da > 0 && db < 0)) {
      double t = da / (da - db);
      V q;
      for (size_t k = 0; k < q.size(); ++k)
        q[k] = a[k] + t * (b[k] - a[k]);
      out.push_back(q);
    }
  }
  return out;
}
Scene read(const std::string &path, int bits, bool resolved_mesh) {
  std::ifstream file(path);
  if (!file)
    throw std::runtime_error("Cannot open scene: " + path);
  Json data;
  file >> data;
  Scene s;
  s.bits = bits;
  auto cam = data.value("camera", Json::object());
  auto type = cam.value("type", std::string("orthographic"));
  if (type != "orthographic" && type != "perspective")
    throw std::runtime_error("Unknown camera");
  s.perspective = type == "perspective";
  s.near = number(cam.value("near", Json(0.01)));
  if (s.near <= 0)
    throw std::runtime_error("Near plane must be positive");
  Matrix m{};
  for (int i = 0; i < 3; ++i)
    m[i][i] = 1;
  if (cam.contains("world_to_camera")) {
    auto a = cam["world_to_camera"];
    if (a.size() != 3)
      throw std::runtime_error("Camera matrix must be 3x4");
    for (int i = 0; i < 3; ++i) {
      if (a[i].size() != 4)
        throw std::runtime_error("Camera matrix must be 3x4");
      for (int j = 0; j < 4; ++j)
        m[i][j] = number(a[i][j]);
    }
  }
  double a[3][6]{};
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j)
      a[i][j] = m[i][j];
    a[i][i + 3] = 1;
  }
  for (int k = 0; k < 3; ++k) {
    int pivot = k;
    for (int i = k + 1; i < 3; ++i)
      if (std::abs(a[i][k]) > std::abs(a[pivot][k]))
        pivot = i;
    if (a[pivot][k] == 0)
      throw std::runtime_error("Singular camera");
    for (int j = 0; j < 6; ++j)
      std::swap(a[k][j], a[pivot][j]);
    double divisor = a[k][k];
    for (auto &v : a[k])
      v /= divisor;
    for (int i = 0; i < 3; ++i)
      if (i != k) {
        double v = a[i][k];
        for (int j = 0; j < 6; ++j)
          a[i][j] -= v * a[k][j];
      }
  }
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      s.inverse[i][j] = a[i][j + 3];
      s.inverse[i][3] -= a[i][j + 3] * m[j][3];
    }
  if (data.at("window").size() != 4)
    throw std::runtime_error("Invalid window");
  for (int i = 0; i < 4; ++i)
    s.window[i] = number(data["window"][i]);
  auto w = s.window;
  double grid = double(int64_t(1) << bits);
  if (!(w[2] > w[0] && w[3] > w[1]))
    throw std::runtime_error("Invalid window");
  std::set<int> ids;
  std::map<V3, int> vertex_ids;
  struct Incident {
    int direction, facing;
    std::array<Point, 2> projected;
  };
  std::map<std::pair<int, int>, std::vector<Incident>> edges;
  size_t clipping_boundaries = 0;
  for (auto &t : data.at("triangles")) {
    int id = t.value("id", s.input_sources++);
    if (!ids.insert(id).second)
      throw std::runtime_error("Duplicate source ID");
    if (t.at("vertices").size() != 3)
      throw std::runtime_error("Expected triangle");
    std::vector<V3> vertices;
    std::array<int, 3> connectivity{};
    bool clipped = false;
    for (auto &v : t["vertices"]) {
      if (v.size() != 3)
        throw std::runtime_error("Expected 3D vertex");
      V3 world{number(v[0]), number(v[1]), number(v[2])};
      if (resolved_mesh) {
        auto entry = vertex_ids.emplace(world, int(vertex_ids.size()));
        connectivity[vertices.size()] = entry.first->second;
      }
      vertices.push_back(transform(m, world));
      clipped |= s.perspective && vertices.back()[2] < s.near;
    }
    if (s.perspective)
      vertices = clip_float(vertices, [&](V3 v) { return v[2] - s.near; });
    if (vertices.size() < 3) {
      ++s.collapsed_sources;
      continue;
    }
    std::vector<V2> p;
    std::vector<double> depths;
    for (auto v : vertices) {
      double q = s.perspective ? 1 / v[2] : 1;
      p.push_back(
          {(v[0] * q - w[0]) * grid / (w[2] - w[0]), (v[1] * q - w[1]) * grid / (w[3] - w[1])});
      depths.push_back(s.perspective ? -q : v[2]);
    }
    size_t k = 2;
    double det = 0;
    for (; k < p.size(); ++k) {
      det = (p[1][0] - p[0][0]) * (p[k][1] - p[0][1]) - (p[1][1] - p[0][1]) * (p[k][0] - p[0][0]);
      if (det != 0)
        break;
    }
    if (k == p.size()) {
      ++s.collapsed_sources;
      continue;
    }
    double x1 = p[1][0] - p[0][0], y1 = p[1][1] - p[0][1], x2 = p[k][0] - p[0][0],
           y2 = p[k][1] - p[0][1];
    Source src;
    src.id = id;
    src.original_id = t.value("original_source_id", id);
    src.score = {((depths[1] - depths[0]) * y2 - (depths[k] - depths[0]) * y1) / det,
                 (x1 * (depths[k] - depths[0]) - x2 * (depths[1] - depths[0])) / det, 0};
    src.score[2] = depths[0] - src.score[0] * p[0][0] - src.score[1] * p[0][1];
    for (double v : src.score)
      if (!std::isfinite(v))
        throw std::runtime_error("Unrepresentable depth plane");
    for (auto v : p)
      clipped |= v[0] < 0 || v[0] > grid || v[1] < 0 || v[1] > grid;
    std::array<Point, 3> original_projection{};
    if (!clipped)
      for (int i = 0; i < 3; ++i)
        original_projection[i] = {int32_t(std::llround(p[i][0])), int32_t(std::llround(p[i][1]))};
    for (int axis = 0; axis < 2; ++axis) {
      p = clip_float(p, [&](V2 v) { return v[axis]; });
      p = clip_float(p, [&](V2 v) { return grid - v[axis]; });
    }
    for (auto v : p)
      src.initial.push_back({int32_t(std::llround(std::clamp(v[0], 0., grid))),
                             int32_t(std::llround(std::clamp(v[1], 0., grid)))});
    if (src.initial.size() > 2 && area2(src.initial) < 0)
      std::reverse(src.initial.begin(), src.initial.end());
    Counts c;
    clean(src.initial, c);
    if (src.initial.empty()) {
      ++s.collapsed_sources;
      continue;
    }
    if (resolved_mesh) {
      if (clipped) {
        for (size_t i = 0; i < src.initial.size(); ++i) {
          s.contours.push_back({src.initial[i], src.initial[(i + 1) % src.initial.size()]});
          ++clipping_boundaries;
        }
      } else
        for (int i = 0; i < 3; ++i) {
          int a = connectivity[i], b = connectivity[(i + 1) % 3];
          edges[{std::min(a, b), std::max(a, b)}].push_back(
              {a < b ? 1 : -1,
               det > 0 ? 1 : -1,
               {original_projection[i], original_projection[(i + 1) % 3]}});
        }
    }
    s.sources.push_back(std::move(src));
  }
  size_t boundary = 0, silhouette = 0, nonmanifold = 0, ignored = 0;
  for (auto &entry : edges) {
    auto &faces = entry.second;
    if (faces.size() == 2 && faces[0].direction != faces[1].direction &&
        faces[0].facing == faces[1].facing) {
      ++ignored;
      continue;
    }
    if (faces.size() == 1)
      ++boundary;
    else if (faces.size() != 2 || faces[0].direction == faces[1].direction)
      ++nonmanifold;
    else
      ++silhouette;
    s.contours.push_back(faces.front().projected);
  }
  s.topology = {{"resolved_mesh_contract", resolved_mesh},
                {"boundary_edges", boundary},
                {"silhouette_edges", silhouette},
                {"nonmanifold_or_inconsistent_edges", nonmanifold},
                {"same_facing_internal_edges_ignored", ignored},
                {"camera_clipping_boundary_edges", clipping_boundaries},
                {"contour_segments", s.contours.size()}};
  return s;
}
struct Box {
  int32_t lo[2] = {INT32_MAX, INT32_MAX}, hi[2] = {INT32_MIN, INT32_MIN};
  double near = INFINITY, far = -INFINITY;
  void add(Point p) {
    lo[0] = std::min(lo[0], p.x);
    lo[1] = std::min(lo[1], p.y);
    hi[0] = std::max(hi[0], p.x);
    hi[1] = std::max(hi[1], p.y);
  }
  void add(const Box &b) {
    for (int k = 0; k < 2; ++k) {
      lo[k] = std::min(lo[k], b.lo[k]);
      hi[k] = std::max(hi[k], b.hi[k]);
    }
    near = std::min(near, b.near);
    far = std::max(far, b.far);
  }
};
bool overlap(const Box &a, const Box &b) {
  return a.lo[0] <= b.hi[0] && b.lo[0] <= a.hi[0] && a.lo[1] <= b.hi[1] && b.lo[1] <= a.hi[1];
}
Box bound(const Ring &r) {
  Box b;
  for (auto p : r)
    b.add(p);
  return b;
}
class BVH {
public:
  struct Node {
    Box box;
    int left = -1, right = -1, begin = 0, end = 0;
  };
  std::vector<Node> nodes;
  std::vector<int> ids;
  const std::vector<Box> &boxes;
  explicit BVH(const std::vector<Box> &b) : boxes(b) {
    ids.resize(b.size());
    std::iota(ids.begin(), ids.end(), 0);
    nodes.reserve(b.size());
    if (!b.empty())
      build(0, b.size());
  }
  int build(int begin, int end) {
    int i = nodes.size();
    nodes.emplace_back();
    Box box;
    for (int j = begin; j < end; ++j)
      box.add(boxes[ids[j]]);
    nodes[i].box = box;
    if (end - begin <= 4) {
      nodes[i].begin = begin;
      nodes[i].end = end;
      return i;
    }
    int axis = box.hi[1] - box.lo[1] > box.hi[0] - box.lo[0], mid = (begin + end) / 2;
    std::nth_element(ids.begin() + begin, ids.begin() + mid, ids.begin() + end, [&](int a, int b) {
      int64_t x = int64_t(boxes[a].lo[axis]) + boxes[a].hi[axis],
              y = int64_t(boxes[b].lo[axis]) + boxes[b].hi[axis];
      return x == y ? a < b : x < y;
    });
    int a = build(begin, mid), b = build(mid, end);
    nodes[i].left = a;
    nodes[i].right = b;
    return i;
  }
  template <class F> void visit(const Box &b, F &&f, int i = 0) const {
    if (nodes.empty() || !overlap(nodes[i].box, b))
      return;
    auto &n = nodes[i];
    if (n.left < 0) {
      for (int j = n.begin; j < n.end; ++j)
        if (overlap(boxes[ids[j]], b))
          f(ids[j]);
    } else {
      visit(b, f, n.left);
      visit(b, f, n.right);
    }
  }
};
template <class F> std::pair<Ring, Ring> split(const Ring &r, F eval, Counts &c) {
  boost::container::small_vector<double, 8> d;
  bool pos = false, neg = false;
  for (auto p : r) {
    d.push_back(eval(p));
    pos |= d.back() > 0;
    neg |= d.back() < 0;
  }
  if (!neg)
    return {r, {}};
  if (!pos)
    return {{}, r};
  Ring in, out;
  for (size_t i = 0; i < r.size(); ++i) {
    size_t j = (i + 1) % r.size();
    if (d[i] >= 0)
      in.push_back(r[i]);
    if (d[i] <= 0)
      out.push_back(r[i]);
    if ((d[i] > 0 && d[j] < 0) || (d[i] < 0 && d[j] > 0)) {
      double t = d[i] / (d[i] - d[j]);
      Point p{int32_t(std::llround(r[i].x + t * (r[j].x - r[i].x))),
              int32_t(std::llround(r[i].y + t * (r[j].y - r[i].y)))};
      in.push_back(p);
      out.push_back(p);
    }
  }
  clean(in, c);
  clean(out, c);
  return {std::move(in), std::move(out)};
}
bool separated(const Ring &a, const Ring &b) {
  for (size_t i = 0; i < a.size(); ++i) {
    bool pos = false;
    for (auto p : b)
      if (side(a[i], a[(i + 1) % a.size()], p) > 0) {
        pos = true;
        break;
      }
    if (!pos)
      return true;
  }
  return false;
}
using Rings = boost::container::small_vector<Ring, 2>;
struct Work {
  Rings visible, hidden;
  Counts counts;
};
double at(const std::array<double, 3> &s, Point p) { return s[0] * p.x + s[1] * p.y + s[2]; }
void cut(Work &w, const Patch &candidate, const Source &target, const Source &other) {
  Box cb = bound(candidate.points);
  Rings next;
  std::array<double, 3> d;
  for (int k = 0; k < 3; ++k)
    d[k] = target.score[k] - other.score[k];
  // A binary64-scale tie band, not a screen-space visibility threshold.
  double tol = 32 * std::numeric_limits<double>::epsilon() *
               std::max({std::abs(target.score[2]), std::abs(other.score[2]), 1e-300});
  for (auto &p : w.visible) {
    if (!overlap(bound(p), cb) || separated(p, candidate.points) ||
        separated(candidate.points, p)) {
      next.push_back(std::move(p));
      continue;
    }
    bool pos = false, neg = false;
    for (auto v : p) {
      double x = at(d, v);
      pos |= x > tol;
      neg |= x < -tol;
    }
    if (!pos && !neg) {
      ++w.counts.ties;
      pos = target.id > other.id;
    }
    if (!pos) {
      next.push_back(std::move(p));
      continue;
    }
    Ring inside = p;
    Rings outside;
    if (neg) {
      auto parts = split(
          inside, [&](Point v) { return at(d, v); }, w.counts);
      inside = std::move(parts.first);
      if (!parts.second.empty())
        outside.push_back(std::move(parts.second));
    }
    for (size_t j = 0; j < candidate.points.size() && !inside.empty(); ++j) {
      auto parts = split(
          inside,
          [&](Point v) {
            return double(
                side(candidate.points[j], candidate.points[(j + 1) % candidate.points.size()], v));
          },
          w.counts);
      inside = std::move(parts.first);
      if (!parts.second.empty())
        outside.push_back(std::move(parts.second));
    }
    if (inside.empty())
      next.push_back(std::move(p));
    else {
      ++w.counts.cuts;
      w.hidden.push_back(std::move(inside));
      for (auto &v : outside)
        next.push_back(std::move(v));
    }
  }
  w.visible = std::move(next);
}
void query(const Scene &s, const std::vector<Patch> &input, const BVH &tree, const Patch &target,
           const Box &target_box, Work &w, int index, const std::vector<int> *ranks = nullptr,
           int layer = 0) {
  if (w.visible.empty() || tree.nodes.empty())
    return;
  auto &node = tree.nodes[index];
  ++w.counts.nodes;
  // Strict separation retains equal-depth candidates for deterministic ID ties.
  if (!overlap(node.box, target_box) || node.box.near > target_box.far)
    return;
  if (node.left < 0) {
    for (int j = node.begin; j < node.end && !w.visible.empty(); ++j) {
      int candidate = tree.ids[j];
      if (input[candidate].source == target.source ||
          (ranks && (*ranks)[input[candidate].source] < layer) ||
          !overlap(tree.boxes[candidate], target_box) ||
          tree.boxes[candidate].near > target_box.far)
        continue;
      ++w.counts.candidates;
      cut(w, input[candidate], s.sources[target.source], s.sources[input[candidate].source]);
    }
  } else {
    int a = node.left, b = node.right;
    if (tree.nodes[a].box.near > tree.nodes[b].box.near)
      std::swap(a, b);
    query(s, input, tree, target, target_box, w, a, ranks, layer);
    query(s, input, tree, target, target_box, w, b, ranks, layer);
  }
}
std::vector<Face> mesh(const std::vector<Patch> &patches, int threads) {
  std::vector<Point> points;
  for (auto &p : patches)
    points.insert(points.end(), p.points.begin(), p.points.end());
  std::sort(points.begin(), points.end());
  points.erase(std::unique(points.begin(), points.end()), points.end());
  std::vector<Box> boxes(points.size());
  for (size_t j = 0; j < points.size(); ++j)
    boxes[j].add(points[j]);
  BVH tree(boxes);
  std::vector<boost::container::small_vector<Face, 4>> output(patches.size());
#pragma omp parallel for schedule(dynamic, 64) num_threads(threads)
  for (size_t j = 0; j < patches.size(); ++j) {
    auto &p = patches[j];
    Ring ring;
    for (size_t i = 0; i < p.points.size(); ++i) {
      auto a = p.points[i], b = p.points[(i + 1) % p.points.size()];
      Box box;
      box.add(a);
      box.add(b);
      Ring along;
      tree.visit(box, [&](int k) {
        auto v = points[k];
        if (v != b && side(a, b, v) == 0)
          along.push_back(v);
      });
      std::sort(along.begin(), along.end());
      if (b < a)
        std::reverse(along.begin(), along.end());
      ring.insert(ring.end(), along.begin(), along.end());
    }
    auto point = [](Point v) { return V2{double(v.x), double(v.y)}; };
    if (ring.size() == 3)
      output[j].push_back({p.source, {point(ring[0]), point(ring[1]), point(ring[2])}});
    else if (ring.size() > 3) {
      V2 center{};
      for (auto v : ring) {
        center[0] += v.x;
        center[1] += v.y;
      }
      for (auto &v : center)
        v /= ring.size();
      for (size_t i = 0; i < ring.size(); ++i)
        output[j].push_back(
            {p.source, {center, point(ring[i]), point(ring[(i + 1) % ring.size()])}});
    }
  }
  size_t count = 0;
  for (auto &v : output)
    count += v.size();
  std::vector<Face> out;
  out.reserve(count);
  for (auto &v : output)
    out.insert(out.end(), v.begin(), v.end());
  return out;
}
std::vector<int> classify_ranks(const Scene &s, int threads, Json &metrics, const BVH &sources) {
  auto start = Clock::now();
  std::vector<int> ranks(s.sources.size(), -1);
  if (!s.topology.value("resolved_mesh_contract", false))
    return ranks;
  std::vector<Box> edges;
  for (auto e : s.contours) {
    Box b;
    b.add(e[0]);
    b.add(e[1]);
    for (int d = 0; d < 2; ++d) {
      b.lo[d] -= 4;
      b.hi[d] += 4;
    }
    edges.push_back(b);
  }
  BVH contour_tree(edges);
  std::vector<uint64_t> candidates(ranks.size(), 0);
#pragma omp parallel for schedule(dynamic, 64) num_threads(threads)
  for (size_t i = 0; i < s.sources.size(); ++i) {
    auto &src = s.sources[i];
    bool boundary = false;
    contour_tree.visit(sources.boxes[i], [&](int j) {
      if (boundary)
        return;
      auto e = s.contours[j];
      // Conservative four-cell guard around each contour, including endpoints.
      for (size_t k = 0; k < src.initial.size(); ++k) {
        auto a = src.initial[k], b = src.initial[(k + 1) % src.initial.size()];
        int64_t margin = 4 * (int64_t(std::abs(b.x - a.x)) + std::abs(b.y - a.y));
        if (side(a, b, e[0]) < -margin && side(a, b, e[1]) < -margin)
          return;
      }
      bool positive = false, negative = false;
      int64_t margin = 4 * (int64_t(std::abs(e[1].x - e[0].x)) + std::abs(e[1].y - e[0].y));
      for (auto p : src.initial) {
        auto value = side(e[0], e[1], p);
        positive |= value >= -margin;
        negative |= value <= margin;
      }
      if (positive && negative)
        boundary = true;
    });
    if (boundary || src.initial.size() != 3)
      continue;
    int64_t px = 0, py = 0;
    for (auto p : src.initial) {
      px += p.x;
      py += p.y;
    }
    Box point;
    point.add(Point{int32_t(px / 3), int32_t(py / 3)});
    point.add(Point{int32_t((px + 2) / 3), int32_t((py + 2) / 3)});
    double target = src.score[0] * (px / 3.) + src.score[1] * (py / 3.) + src.score[2];
    int rank = 0;
    bool ambiguous = false;
    sources.visit(point, [&](int j) {
      if (j == int(i))
        return;
      ++candidates[i];
      auto &other = s.sources[j];
      bool edge = false;
      for (size_t k = 0; k < other.initial.size(); ++k) {
        auto a = other.initial[k], b = other.initial[(k + 1) % other.initial.size()];
        int64_t cross = int64_t(b.x - a.x) * (py - 3 * int64_t(a.y)) -
                        int64_t(b.y - a.y) * (px - 3 * int64_t(a.x));
        if (cross < 0)
          return;
        edge |= cross == 0;
      }
      if (edge) {
        ambiguous = true;
        return;
      }
      double z = other.score[0] * (px / 3.) + other.score[1] * (py / 3.) + other.score[2];
      double tol = 64 * std::numeric_limits<double>::epsilon() *
                   std::max({std::abs(z), std::abs(target), 1e-300});
      if (std::abs(z - target) <= tol) {
        ambiguous = true;
        return;
      }
      rank += z < target;
    });
    if (!ambiguous)
      ranks[i] = rank;
  }
  size_t known = std::count_if(ranks.begin(), ranks.end(), [](int r) { return r >= 0; });
  metrics = {{"constant_rank_sources", known},
             {"sources_requiring_clipping", ranks.size() - known},
             {"point_query_candidates",
              std::accumulate(candidates.begin(), candidates.end(), uint64_t(0))},
             {"classification_seconds", elapsed(start)}};
  return ranks;
}
Result extract(const Scene &s, int layers, int threads, const std::string &order, unsigned seed,
               bool accelerate) {
  auto prepare_boxes = [&](const std::vector<Patch> &patches) {
    std::vector<Box> boxes(patches.size());
    for (size_t i = 0; i < patches.size(); ++i) {
      boxes[i] = bound(patches[i].points);
      auto &score = s.sources[patches[i].source].score;
      for (auto p : patches[i].points) {
        double z = at(score, p);
        double err = 64 * std::numeric_limits<double>::epsilon() *
                     (std::abs(score[0] * p.x) + std::abs(score[1] * p.y) + std::abs(score[2]));
        boxes[i].near = std::min(boxes[i].near, z - err);
        boxes[i].far = std::max(boxes[i].far, z + err);
      }
    }
    return boxes;
  };
  std::vector<Patch> originals;
  originals.reserve(s.sources.size());
  for (size_t i = 0; i < s.sources.size(); ++i)
    originals.push_back({int(i), s.sources[i].initial});
  auto original_boxes = prepare_boxes(originals);
  BVH original_tree(original_boxes);
  Result r;
  auto ranks = classify_ranks(s, threads, r.classification, original_tree);
  std::vector<Patch> residual;
  std::vector<std::vector<int>> direct;
  for (size_t i = 0; i < originals.size(); ++i) {
    if (ranks[i] < 0)
      residual.push_back(originals[i]);
    else {
      if (direct.size() <= size_t(ranks[i]))
        direct.resize(ranks[i] + 1);
      direct[ranks[i]].push_back(i);
    }
  }
  int limit = layers ? layers : s.sources.size();
  for (int k = 0; k < limit && (!residual.empty() || size_t(k) < direct.size()); ++k) {
    auto t = Clock::now();
    size_t initial_count = residual.size();
    auto boxes = prepare_boxes(residual);
    BVH tree(boxes);
    std::vector<Work> work(residual.size());
    std::vector<int> schedule(residual.size());
    std::iota(schedule.begin(), schedule.end(), 0);
    if (order == "reverse")
      std::reverse(schedule.begin(), schedule.end());
    if (order == "random") {
      std::mt19937 rng(seed);
      std::shuffle(schedule.begin(), schedule.end(), rng);
    }
#pragma omp parallel for schedule(dynamic, 32) num_threads(threads)
    for (size_t j = 0; j < schedule.size(); ++j) {
      int i = schedule[j];
      auto &w = work[i];
      w.visible.push_back(residual[i].points);
      if (accelerate) {
        query(s, originals, original_tree, residual[i], boxes[i], w, 0, &ranks, k);
        query(s, residual, tree, residual[i], boxes[i], w, 0);
      } else {
        for (size_t ci = 0; ci < originals.size() && !w.visible.empty(); ++ci)
          if (ranks[ci] >= k) {
            ++w.counts.candidates;
            cut(w, originals[ci], s.sources[residual[i].source], s.sources[ci]);
          }
        for (int ci : schedule) {
          if (residual[ci].source == residual[i].source || w.visible.empty())
            continue;
          ++w.counts.candidates;
          cut(w, residual[ci], s.sources[residual[i].source], s.sources[residual[ci].source]);
        }
      }
      if (w.visible.empty()) {
        w.hidden.clear();
        w.hidden.push_back(residual[i].points);
        ++w.counts.rejected;
      }
    }
    Layer l;
    if (size_t(k) < direct.size())
      for (int i : direct[k])
        l.patches.push_back(originals[i]);
    std::vector<Patch> next;
    Counts counts;
    for (size_t i = 0; i < work.size(); ++i) {
      counts.add(work[i].counts);
      for (auto &p : work[i].visible)
        l.patches.push_back({residual[i].source, std::move(p)});
      for (auto &p : work[i].hidden)
        next.push_back({residual[i].source, std::move(p)});
    }
    if (l.patches.empty())
      throw std::runtime_error("Fast peeling stalled after rounding; use finer grid or exact mode");
    double visibility = elapsed(t);
    t = Clock::now();
    l.faces = mesh(l.patches, threads);
    l.metrics = {{"layer_index", k + 1},
                 {"visibility_seconds", visibility},
                 {"triangulation_seconds", elapsed(t)},
                 {"candidate_tests", counts.candidates},
                 {"index_box_tests", counts.nodes},
                 {"polygon_differences", counts.cuts},
                 {"complete_occlusion_rejections", counts.rejected},
                 {"convexity_repairs", counts.repairs},
                 {"collapsed_polygons", counts.collapsed},
                 {"depth_tie_breaks", counts.ties},
                 {"input_residual", {{"convex_patches", initial_count}}},
                 {"output", {{"convex_patches", l.patches.size()}}},
                 {"exported_triangles", l.faces.size()}};
    r.layers.push_back(std::move(l));
    residual = std::move(next);
  }
  if (layers == 0 && !residual.empty())
    throw std::runtime_error("Fast peeling exceeded source-count layer limit");
  r.residual = std::move(residual);
  for (size_t k = r.layers.size(); k < direct.size(); ++k)
    for (int i : direct[k])
      r.residual.push_back(originals[i]);
  return r;
}
V2 uv(const Scene &s, V2 p) {
  double g = double(int64_t(1) << s.bits);
  return {s.window[0] + p[0] / g * (s.window[2] - s.window[0]),
          s.window[1] + p[1] / g * (s.window[3] - s.window[1])};
}
V3 lift(const Scene &s, const Source &src, V2 p) {
  double score = src.score[0] * p[0] + src.score[1] * p[1] + src.score[2];
  auto u = uv(s, p);
  double z = s.perspective ? -1 / score : score;
  return transform(s.inverse,
                   {s.perspective ? u[0] * z : u[0], s.perspective ? u[1] * z : u[1], z});
}
} // namespace
int run_cli(const std::string &path, const std::string &out, int layers, int threads, int repeat,
            int bits, const std::string &order, unsigned seed, bool accelerate, bool metrics_only,
            bool verify, bool validate, int cap, bool resolved_mesh, bool mesh_only) {
  if (bits < 12 || bits > 30)
    throw std::runtime_error("Fast grid bits must be between 12 and 30");
#ifndef _OPENMP
  if (threads != 1)
    throw std::runtime_error("This build has no OpenMP; use --threads 1");
#endif
  if (order != "input" && order != "near-first" && order != "random" && order != "reverse")
    throw std::runtime_error("Unknown order");
  auto t = Clock::now();
  auto s = read(path, bits, resolved_mesh);
  double setup = elapsed(t);
  Result r;
  std::vector<double> times;
  for (int i = 0; i < repeat; ++i) {
    r = Result();
    t = Clock::now();
    r = extract(s, layers, threads, order, seed, accelerate);
    times.push_back(elapsed(t));
  }
  double validation = 0;
  Json audit;
  if (verify || validate) {
    t = Clock::now();
    audit = validate_reference(path, s, r, verify, cap, layers == 0);
    validation = elapsed(t);
  }
  auto sorted = times;
  std::sort(sorted.begin(), sorted.end());
  Json metadata = {
      {"method", "fast-bvh-peel"},
      {"arithmetic",
       "finite-precision: fixed-point projected geometry, binary64 depth/constructions"},
      {"exact_geometry", false},
      {"grid_bits", bits},
      {"grid_cells_per_axis", int64_t(1) << bits},
      {"collapsed_initial_sources", s.collapsed_sources},
      {"minimum_polygon_area_grid_cells", 1},
      {"minimum_polygon_width_l1_grid_cells", 2},
      {"mesh_topology", s.topology},
      {"rank_classification", r.classification},
      {"depth_tie_policy", "binary64 uncertainty band; lower source ID wins"},
      {"input_sources", s.input_sources},
      {"active_sources", s.sources.size()},
      {"threads", threads},
      {"order", order},
      {"seed", seed},
      {"accelerated", accelerate},
      {"requested_layers", layers ? Json(layers) : Json("all")},
      {"projection_preflight_seconds", setup},
      {"algorithm_seconds", sorted[sorted.size() / 2]},
      {"algorithm_repetitions_seconds", times},
      {"validation_seconds", validation},
      {"exact_invariants_validated", false},
      {"validated_against_oracle", false},
      {"fast_validation", audit},
      {"remaining_patches", r.residual.size()},
      {"scene", path},
      {"output_mode", metrics_only ? "metrics-only"
                      : mesh_only  ? "obj-and-metrics"
                                   : "approximate-json-and-obj"}};
  t = Clock::now();
  std::filesystem::create_directories(out);
  Json passes = Json::array(), geometry = Json::array();
  for (size_t k = 0; k < r.layers.size(); ++k) {
    auto &l = r.layers[k];
    passes.push_back(l.metrics);
    if (metrics_only)
      continue;
    std::ofstream obj(out + "/layer-" + std::to_string(k + 1) + ".obj");
    obj << std::setprecision(17);
    size_t corner = 1;
    for (auto &f : l.faces) {
      for (auto p : f.points) {
        auto w = lift(s, s.sources[f.source], p);
        obj << "v " << w[0] << ' ' << w[1] << ' ' << w[2] << '\n';
      }
      obj << "g source_" << s.sources[f.source].original_id << "\nf " << corner << ' ' << corner + 1
          << ' ' << corner + 2 << '\n';
      corner += 3;
    }
    Json patches = Json::array();
    for (auto &p : l.patches) {
      if (mesh_only)
        break;
      Json points = Json::array();
      for (auto v : p.points)
        points.push_back({v.x, v.y});
      patches.push_back({{"source", s.sources[p.source].original_id},
                         {"piece", s.sources[p.source].id},
                         {"grid_vertices", points}});
    }
    geometry.push_back({{"layer", k + 1}, {"patches", patches}});
  }
  if (!metrics_only && !mesh_only) {
    Json j = {{"format", "approximate-fixed-grid-v1"},
              {"exact_geometry", false},
              {"grid_bits", bits},
              {"window", s.window},
              {"layers", geometry}};
    std::ofstream(out + "/approximate.json") << j.dump() << '\n';
  }
  metadata["passes"] = passes;
  metadata["serialization_seconds"] = elapsed(t);
  struct rusage usage {};
  getrusage(RUSAGE_SELF, &usage);
  metadata["peak_rss_kib"] = usage.ru_maxrss;
  std::ofstream(out + "/metrics.json") << metadata.dump(2) << '\n';
  std::cout << Json({{"layers", r.layers.size()},
                     {"algorithm_seconds", metadata["algorithm_seconds"]},
                     {"validation_seconds", validation},
                     {"exact_geometry", false},
                     {"output", out}})
                   .dump()
            << '\n';
  return audit.is_object() && audit.value("status", std::string()) != "passed" ? 3 : 0;
}
} // namespace peel::fast
