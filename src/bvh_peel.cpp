#include "geometry.h"
#include "peel_common.h"
#include <CGAL/Interval_nt.h>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <random>
#include <sys/resource.h>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace peel {
namespace {
using Ring = std::vector<P>;
// Every floating-point bound encloses the exact coordinate/score. Bounds only
// reject candidates; all geometry and ambiguous decisions use exact rationals.
struct Bounds {
  double center(int axis) const { return lo[axis] + hi[axis]; }
  double lo[2] = {INFINITY, INFINITY}, hi[2] = {-INFINITY, -INFINITY};
  double depth_lo = INFINITY, depth_hi = -INFINITY;
  void add(const P &p) {
    for (int k = 0; k != 2; ++k) {
      auto v = CGAL::to_interval(p[k]);
      lo[k] = std::min(lo[k], v.first);
      hi[k] = std::max(hi[k], v.second);
    }
  }
  void add(const Bounds &b) {
    for (int k = 0; k != 2; ++k) {
      lo[k] = std::min(lo[k], b.lo[k]);
      hi[k] = std::max(hi[k], b.hi[k]);
    }
    depth_lo = std::min(depth_lo, b.depth_lo);
    depth_hi = std::max(depth_hi, b.depth_hi);
  }
};
bool overlap(const Bounds &a, const Bounds &b) {
  return a.lo[0] <= b.hi[0] && b.lo[0] <= a.hi[0] && a.lo[1] <= b.hi[1] && b.lo[1] <= a.hi[1];
}
Bounds bound(const Ring &p) {
  Bounds b;
  for (auto &v : p)
    b.add(v);
  return b;
}
using BVH = common::BVH<Bounds>;
Q side(const P &a, const P &b, const P &p) {
  return (b.x() - a.x()) * (p.y() - a.y()) - (b.y() - a.y()) * (p.x() - a.x());
}
using Interval = CGAL::Interval_nt<false>;
Interval interval(const Q &q) {
  auto b = CGAL::to_interval(q);
  return Interval(b.first, b.second);
}
int side_sign(const P &a, const P &b, const P &p) {
  if (p == a || p == b)
    return 0;
  Interval ax = interval(a.x()), ay = interval(a.y());
  auto d = (interval(b.x()) - ax) * (interval(p.y()) - ay) -
           (interval(b.y()) - ay) * (interval(p.x()) - ax);
  if (d.inf() > 0)
    return 1;
  if (d.sup() < 0)
    return -1;
  return CGAL::sign(side(a, b, p));
}
int score_sign(const Score &s, const P &p) {
  auto d = interval(s.a) * interval(p.x()) + interval(s.b) * interval(p.y()) + interval(s.c);
  if (d.inf() > 0)
    return 1;
  if (d.sup() < 0)
    return -1;
  return CGAL::sign(s.at(p));
}
void clean(Ring &r) {
  Ring out;
  for (const auto &p : r)
    if (out.empty() || p != out.back())
      out.push_back(p);
  if (out.size() > 1 && out.front() == out.back())
    out.pop_back();
  r = std::move(out);
  if (r.size() < 3)
    r.clear();
}
// Partition a convex polygon by an affine half-plane. Positive is inside.
template <class F, class S> std::pair<Ring, Ring> split(const Ring &r, F &&eval, S &&sign) {
  std::vector<int> d;
  bool pos = false, neg = false;
  for (auto &p : r) {
    d.push_back(sign(p));
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
      Q di = eval(r[i]), dj = eval(r[j]);
      Q t = di / (di - dj);
      P p(r[i].x() + t * (r[j].x() - r[i].x()), r[i].y() + t * (r[j].y() - r[i].y()));
      in.push_back(p);
      out.push_back(p);
    }
  }
  clean(in);
  clean(out);
  return {std::move(in), std::move(out)};
}
bool separated(const Ring &a, const Ring &b) {
  for (size_t i = 0; i < a.size(); ++i) {
    bool positive = false;
    for (const auto &p : b)
      if (side_sign(a[i], a[(i + 1) % a.size()], p) > 0) {
        positive = true;
        break;
      }
    if (!positive)
      return true;
  }
  return false;
}
struct Work {
  uint64_t retired_subtrees = 0;
  std::vector<Ring> visible, hidden;
  Stats stats;
};
void cut(Work &w, const Ring &occluder, const Score &delta, bool accelerate) {
  // delta = target score - occluder score: positive means occluded.
  Bounds cb = bound(occluder);
  std::vector<Ring> next;
  for (auto &p : w.visible) {
    if (accelerate && !overlap(bound(p), cb)) {
      next.push_back(std::move(p));
      continue;
    }
    if (separated(p, occluder) || separated(occluder, p)) {
      next.push_back(std::move(p));
      continue;
    }
    bool nearer = false, behind = false;
    for (const auto &v : p) {
      int d = score_sign(delta, v);
      nearer |= d > 0;
      behind |= d < 0;
    }
    if (!nearer) {
      next.push_back(std::move(p));
      continue;
    }
    Ring inside = p;
    std::vector<Ring> outside;
    if (behind) {
      ++w.stats.halfplane_clips;
      auto parts = split(
          inside, [&](const P &v) { return delta.at(v); },
          [&](const P &v) { return score_sign(delta, v); });
      inside = std::move(parts.first);
      if (!parts.second.empty())
        outside.push_back(std::move(parts.second));
    } else {
      ++w.stats.depth_separated;
    }
    for (size_t j = 0; j < occluder.size() && !inside.empty(); ++j) {
      auto parts = split(
          inside,
          [&](const P &v) { return side(occluder[j], occluder[(j + 1) % occluder.size()], v); },
          [&](const P &v) {
            return side_sign(occluder[j], occluder[(j + 1) % occluder.size()], v);
          });
      inside = std::move(parts.first);
      if (!parts.second.empty())
        outside.push_back(std::move(parts.second));
    }
    if (inside.empty()) {
      // No coverage change: do not retain gratuitous cuts of a visible polygon.
      next.push_back(std::move(p));
    } else {
      ++w.stats.differences;
      w.hidden.push_back(std::move(inside));
      for (auto &v : outside)
        next.push_back(std::move(v));
    }
  }
  w.visible = std::move(next);
}
void add_stats(Stats &a, const Stats &b) {
  a.candidate_tests += b.candidate_tests;
  a.index_box_tests += b.index_box_tests;
  a.box_rejections += b.box_rejections;
  a.depth_separated += b.depth_separated;
  a.differences += b.differences;
  a.halfplane_clips += b.halfplane_clips;
  a.occlusion_rejections += b.occlusion_rejections;
}
// A balanced dynamic AABB tree. Leaves are convex visible cells (including
// uncovered background), so the map partitions the entire image window.
class VisibilityTree {
public:
  struct Node {
    Bounds box;
    Patch patch;
    bool background = false, active = false;
    int parent = -1, left = -1, right = -1, height = 0;
  };
  std::vector<Node> nodes;
  std::vector<int> free;
  int root = -1;
  int alloc() {
    int i;
    if (free.empty()) {
      i = nodes.size();
      nodes.emplace_back();
    } else {
      i = free.back();
      free.pop_back();
      nodes[i] = Node();
    }
    nodes[i].active = true;
    return i;
  }
  void release(int i) {
    nodes[i].active = false;
    nodes[i].patch.points.clear();
    free.push_back(i);
  }
  static double cost(const Bounds &b) { return (b.hi[0] - b.lo[0]) + (b.hi[1] - b.lo[1]); }
  void refit(int i) {
    if (nodes[i].left < 0)
      return;
    auto &n = nodes[i];
    n.box = nodes[n.left].box;
    n.box.add(nodes[n.right].box);
    n.height = 1 + std::max(nodes[n.left].height, nodes[n.right].height);
  }
  void replace_child(int parent, int from, int to) {
    if (parent < 0)
      root = to;
    else if (nodes[parent].left == from)
      nodes[parent].left = to;
    else
      nodes[parent].right = to;
    if (to >= 0)
      nodes[to].parent = parent;
  }
  int balance(int a) {
    if (nodes[a].left < 0 || nodes[a].height < 2)
      return a;
    int b = nodes[a].left, c = nodes[a].right;
    int d = nodes[c].height - nodes[b].height;
    if (d > 1) {
      int f = nodes[c].left, g = nodes[c].right;
      replace_child(nodes[a].parent, a, c);
      nodes[c].left = a;
      nodes[a].parent = c;
      if (nodes[f].height > nodes[g].height) {
        nodes[c].right = f;
        nodes[a].right = g;
        nodes[g].parent = a;
      } else {
        nodes[c].right = g;
        nodes[a].right = f;
        nodes[f].parent = a;
      }
      refit(a);
      refit(c);
      return c;
    }
    if (d < -1) {
      int f = nodes[b].left, g = nodes[b].right;
      replace_child(nodes[a].parent, a, b);
      nodes[b].right = a;
      nodes[a].parent = b;
      if (nodes[f].height > nodes[g].height) {
        nodes[b].left = f;
        nodes[a].left = g;
        nodes[g].parent = a;
      } else {
        nodes[b].left = g;
        nodes[a].left = f;
        nodes[f].parent = a;
      }
      refit(a);
      refit(b);
      return b;
    }
    return a;
  }
  void ascend(int i) {
    while (i >= 0) {
      refit(i);
      i = balance(i);
      i = nodes[i].parent;
    }
  }
  void insert(Patch patch, bool background = false) {
    int leaf = alloc();
    nodes[leaf].box = bound(patch.points);
    nodes[leaf].patch = std::move(patch);
    nodes[leaf].background = background;
    if (root < 0) {
      root = leaf;
      return;
    }
    int sibling = root;
    while (nodes[sibling].left >= 0) {
      int a = nodes[sibling].left, b = nodes[sibling].right;
      Bounds aa = nodes[a].box, bb = nodes[b].box;
      aa.add(nodes[leaf].box);
      bb.add(nodes[leaf].box);
      sibling = cost(aa) - cost(nodes[a].box) < cost(bb) - cost(nodes[b].box) ? a : b;
    }
    int parent = nodes[sibling].parent;
    int joined = alloc();
    replace_child(parent, sibling, joined);
    nodes[joined].left = sibling;
    nodes[joined].right = leaf;
    nodes[sibling].parent = joined;
    nodes[leaf].parent = joined;
    ascend(joined);
  }
  void erase(int leaf) {
    if (leaf == root) {
      root = -1;
      release(leaf);
      return;
    }
    int parent = nodes[leaf].parent;
    int sibling = nodes[parent].left == leaf ? nodes[parent].right : nodes[parent].left;
    int grand = nodes[parent].parent;
    replace_child(grand, parent, sibling);
    release(leaf);
    release(parent);
    ascend(grand);
  }
  void query(const Bounds &b, std::vector<int> &out, Stats &stats, int i) const {
    if (i < 0)
      return;
    ++stats.index_box_tests;
    auto &n = nodes[i];
    if (!overlap(n.box, b)) {
      ++stats.box_rejections;
      return;
    }
    if (n.left < 0)
      out.push_back(i);
    else {
      query(b, out, stats, n.left);
      query(b, out, stats, n.right);
    }
  }
};
// Subtract a convex polygon; preserve the subject unchanged on no overlap.
Work partition(const Ring &subject, const Ring &clip) {
  Work w;
  w.visible.push_back(subject);
  cut(w, clip, Score{0, 0, 1}, true);
  return w;
}
std::pair<std::vector<Patch>, std::vector<Patch>>
incremental_pass(const Scene &s, const std::vector<Patch> &input, const std::string &order,
                 unsigned seed, Stats &stats, bool accelerate) {
  CGAL::Protect_FPU_rounding<true> rounding(CGAL_FE_UPWARD);
  VisibilityTree tree;
  auto wc = s.window.corners();
  tree.insert({0, {wc.begin(), wc.end()}}, true);
  std::vector<int> ids(input.size());
  std::iota(ids.begin(), ids.end(), 0);
  if (order == "reverse")
    std::reverse(ids.begin(), ids.end());
  if (order == "random") {
    std::mt19937 rng(seed);
    std::shuffle(ids.begin(), ids.end(), rng);
  }
  if (order == "near-first") {
    std::vector<double> near(input.size(), INFINITY);
    for (size_t i = 0; i < input.size(); ++i)
      for (auto &p : input[i].points)
        near[i] = std::min(near[i], CGAL::to_interval(s.source(input[i].source).score.at(p)).first);
    std::stable_sort(ids.begin(), ids.end(), [&](int a, int b) { return near[a] < near[b]; });
  }
  std::vector<Patch> hidden;
  for (int i : ids) {
    auto &incoming = input[i];
    struct Candidate {
      int node, winner; // +1 incoming, -1 old, 0 depth crossing
    };
    std::vector<int> found;
    if (accelerate)
      tree.query(bound(incoming.points), found, stats, tree.root);
    else
      for (size_t j = 0; j < tree.nodes.size(); ++j)
        if (tree.nodes[j].active && tree.nodes[j].left < 0)
          found.push_back(j);
    std::vector<Candidate> candidates;
    bool all_wins = true, all_loses = true;
    for (int n : found) {
      auto &old = tree.nodes[n];
      ++stats.candidate_tests;
      if (separated(old.patch.points, incoming.points) ||
          separated(incoming.points, old.patch.points))
        continue;
      int winner = 1;
      if (!old.background) {
        Score d = difference(s.source(old.patch.source).score, s.source(incoming.source).score);
        bool pos = false, neg = false;
        for (auto &p : incoming.points) {
          int sign = score_sign(d, p);
          pos |= sign > 0;
          neg |= sign < 0;
        }
        winner = !neg ? 1 : !pos ? -1 : 0;
      }
      all_wins &= winner == 1;
      all_loses &= winner == -1;
      candidates.push_back({n, winner});
    }
    if (candidates.empty())
      throw std::runtime_error("Visibility partition lost incoming coverage");
    if (all_loses) {
      ++stats.occlusion_rejections;
      hidden.push_back(incoming);
      continue;
    }
    std::vector<Patch> visible;
    for (auto c : candidates) {
      // Copy before mutations: dynamic tree storage can grow or recycle nodes.
      auto old = tree.nodes[c.node];
      if (c.winner == -1) {
        auto parts = partition(incoming.points, old.patch.points);
        for (auto &p : parts.hidden)
          hidden.push_back({incoming.source, std::move(p)});
        continue;
      }
      auto parts = partition(old.patch.points, incoming.points);
      if (parts.hidden.empty())
        throw std::runtime_error("Positive overlap classification disagrees with clipping");
      Ring common = std::move(parts.hidden.front());
      if (c.winner == 0) {
        Score d = difference(s.source(old.patch.source).score, s.source(incoming.source).score);
        auto split_depth = split(
            common, [&](const P &p) { return d.at(p); },
            [&](const P &p) { return score_sign(d, p); });
        common = std::move(split_depth.first);
        if (!split_depth.second.empty()) {
          hidden.push_back({incoming.source, split_depth.second});
          parts.visible.push_back(std::move(split_depth.second));
        }
        ++stats.halfplane_clips;
      }
      if (common.empty())
        continue;
      if (!old.background)
        hidden.push_back({old.patch.source, common});
      if (!all_wins)
        visible.push_back({incoming.source, std::move(common)});
      tree.erase(c.node);
      for (auto &p : parts.visible)
        tree.insert({old.patch.source, std::move(p)}, old.background);
      ++stats.differences;
    }
    if (all_wins)
      tree.insert(incoming);
    else
      for (auto &p : visible)
        tree.insert(std::move(p));
  }
  std::vector<Patch> visible;
  for (auto &n : tree.nodes)
    if (n.active && n.left < 0 && !n.background)
      visible.push_back(std::move(n.patch));
  return {std::move(visible), std::move(hidden)};
}
// Node all boundary vertices on patch edges without a global arrangement or
// searching all source labels. The patches already carry their source IDs.
std::vector<Face> mesh(const std::vector<Patch> &patches, int threads, Json *phases = nullptr) {
  return common::mesh<P, Bounds, Ring, Face, std::vector<Face>>(
      patches, threads, [] { return CGAL::Protect_FPU_rounding<true>(CGAL_FE_UPWARD); },
      [](const P &a, const P &b, const P &v) {
        return side_sign(a, b, v) == 0 && K::Segment_2(a, b).has_on(v);
      },
      [](int source, const Ring &ring, std::vector<Face> &out) {
        if (ring.size() == 3) {
          out.push_back({source, {ring[0], ring[1], ring[2]}});
        } else if (ring.size() > 3) {
          Q x = 0, y = 0;
          for (const auto &p : ring) {
            x += p.x();
            y += p.y();
          }
          P center(x / Q(ring.size()), y / Q(ring.size()));
          for (size_t i = 0; i < ring.size(); ++i)
            out.push_back({source, {center, ring[i], ring[(i + 1) % ring.size()]}});
        }
      },
      phases);
}

Json patch_complexity(const std::vector<Patch> &ps) {
  long n = 0;
  for (auto &p : ps)
    n += p.points.size();
  return {{"convex_patches", ps.size()}, {"patch_vertices", n}};
}
struct ExactPolicy {
  using PatchType = Patch;
  using BoxType = Bounds;
  using WorkType = Work;
  using LayerType = Layer;
  using ResultType = Result;
  const Scene &s;
  auto guard() const { return CGAL::Protect_FPU_rounding<true>(CGAL_FE_UPWARD); }
  int turn(const P &a, const P &b, const P &c) const { return side_sign(a, b, c); }
  bool resolved() const { return s.topology.value("resolved_mesh_contract", false); }
  void initialize(Result &r) const { r.convex_representation = true; }
  void residual(Result &r, std::vector<Patch> p) const { r.residual_patches = std::move(p); }
  std::vector<Patch> originals() const {
    std::vector<Patch> out;
    out.reserve(s.sources.size());
    for (auto &src : s.sources)
      out.push_back({src.id, {src.initial_polygon.begin(), src.initial_polygon.end()}});
    return out;
  }
  std::vector<Bounds> boxes(const std::vector<Patch> &patches) const {
    std::vector<Bounds> boxes(patches.size());
    for (size_t i = 0; i < patches.size(); ++i) {
      boxes[i] = bound(patches[i].points);
      auto &score = s.source(patches[i].source).score;
      for (auto &p : patches[i].points) {
        auto interval = CGAL::to_interval(score.at(p));
        boxes[i].depth_lo = std::min(boxes[i].depth_lo, interval.first);
        boxes[i].depth_hi = std::max(boxes[i].depth_hi, interval.second);
      }
    }
    return boxes;
  }
  std::vector<Bounds> contour_boxes() const {
    std::vector<Bounds> out(s.contours.size());
    for (size_t i = 0; i < s.contours.size(); ++i) {
      out[i].add(s.contours[i][0]);
      out[i].add(s.contours[i][1]);
    }
    return out;
  }
  bool rank_eligible(int i) const { return s.sources[i].initial_polygon.size() == 3; }
  bool touches_contour(int i, int j) const {
    auto &polygon = s.sources[i].initial_polygon;
    auto &e = s.contours[j];
    // Exact convex polygon / segment separating axes. Contact is conservative.
    for (size_t k = 0; k < polygon.size(); ++k)
      if (side_sign(polygon[k], polygon[(k + 1) % polygon.size()], e[0]) < 0 &&
          side_sign(polygon[k], polygon[(k + 1) % polygon.size()], e[1]) < 0)
        return false;
    bool positive = false, negative = false;
    for (auto &v : polygon) {
      int sign = side_sign(e[0], e[1], v);
      positive |= sign >= 0;
      negative |= sign <= 0;
    }
    return positive && negative;
  }
  auto sample(int i) const {
    auto &r = s.sources[i].initial_polygon;
    P p((r[0].x() + r[1].x() + r[2].x()) / 3, (r[0].y() + r[1].y() + r[2].y()) / 3);
    Bounds b;
    b.add(p);
    return std::make_pair(p, b);
  }
  int ray_order(int i, int j, const P &p) const {
    auto &other = s.sources[j];
    auto &ring = other.initial_polygon;
    bool edge = false;
    for (size_t k = 0; k < ring.size(); ++k) {
      int sign = side_sign(ring[k], ring[(k + 1) % ring.size()], p);
      if (sign < 0)
        return 0;
      edge |= sign == 0;
    }
    if (edge)
      return -1;
    int sign = score_sign(difference(s.sources[i].score, other.score), p);
    return sign == 0 ? -1 : int(sign > 0);
  }
  double near(const Bounds &b) const { return b.depth_lo; }
  bool behind(const Bounds &a, const Bounds &b) const { return a.depth_lo >= b.depth_hi; }
  void node_test(Work &w) const { ++w.stats.index_box_tests; }
  void box_reject(Work &w) const { ++w.stats.box_rejections; }
  void occluded(Work &w) const { ++w.stats.occlusion_rejections; }
  void cut(Work &w, const Patch &target, const Patch &candidate, bool accelerate) const {
    ++w.stats.candidate_tests;
    peel::cut(w, candidate.points,
              difference(s.source(target.source).score, s.source(candidate.source).score),
              accelerate);
  }
  void finish(Layer &layer, int, const std::vector<Patch> &input, const std::vector<Work> &work,
              double visibility, int threads, const Json &phases) const {
    layer.residual_complexity = patch_complexity(input);
    ++layer.stats.index_rebuilds;
    for (auto &w : work)
      add_stats(layer.stats, w.stats);
    layer.stats.visibility_seconds = visibility;
    auto t = Clock::now();
    layer.phase_metrics = phases;
    layer.faces = mesh(layer.patches, threads, &layer.phase_metrics["meshing"]);
    layer.stats.triangulation_seconds = seconds(t);
  }
};
} // namespace

std::vector<Face> triangulate_patches(const std::vector<Patch> &patches, int threads) {
  return mesh(patches, threads);
}

Result bvh_peel(const Scene &s, int layers, const std::string &order, unsigned seed,
                bool accelerate, int threads, bool incremental, PeelOptions options) {
  if (threads < 1)
    throw std::runtime_error("threads must be positive");
#ifndef _OPENMP
  if (threads != 1)
    throw std::runtime_error("This build has no OpenMP support; use --threads 1");
#endif
  if (order != "input" && order != "near-first" && order != "random" && order != "reverse")
    throw std::runtime_error("Unknown insertion order");
  if (!incremental) {
    ExactPolicy policy{s};
    return common::extract(policy, layers, threads, order, seed, accelerate, options);
  }
  Result result;
  result.convex_representation = true;
  std::vector<Patch> residual;
  for (auto &src : s.sources)
    residual.push_back({src.id, {src.initial_polygon.begin(), src.initial_polygon.end()}});
  const int limit = layers ? layers : s.sources.size();
  for (int k = 0; k < limit && !residual.empty(); ++k) {
    Layer layer;
    layer.residual_complexity = patch_complexity(residual);
    auto t = Clock::now();
    if (incremental) {
      auto partitions = incremental_pass(s, residual, order, seed, layer.stats, accelerate);
      layer.patches = std::move(partitions.first);
      residual = std::move(partitions.second);
      if (layer.patches.empty())
        throw std::runtime_error("Incremental BVH peeling made no progress");
      layer.stats.visibility_seconds = seconds(t);
      t = Clock::now();
      layer.faces = mesh(layer.patches, threads);
      layer.stats.triangulation_seconds = seconds(t);
      result.layers.push_back(std::move(layer));
      continue;
    }
  }
  result.residual_patches = std::move(residual);
  return result;
}
void materialize_regions(Result &r) {
  if (!r.convex_representation)
    return;
  for (auto &l : r.layers) {
    if (!l.regions.empty())
      continue;
    auto t = Clock::now();
    for (auto &p : l.patches)
      unite(l.regions[p.source], domain(polygon(p.points)));
    l.stats.normalization_seconds += seconds(t);
  }
  if (r.residual.empty())
    for (auto &p : r.residual_patches)
      unite(r.residual[p.source], domain(polygon(p.points)));
}
Json audit_convex(const Scene &s, const Result &r, bool complete) {
  CGAL::Protect_FPU_rounding<true> rounding(CGAL_FE_UPWARD);
  auto require = [](bool ok, const char *msg) {
    if (!ok)
      throw std::runtime_error(std::string("Convex audit failed: ") + msg);
  };
  auto area2 = [](const Ring &p) {
    Q a = 0;
    for (size_t j = 1; j + 1 < p.size(); ++j)
      a += side(p[0], p[j], p[j + 1]);
    return a;
  };
  std::map<int, Ring> original;
  std::vector<Bounds> original_bounds;
  for (auto &src : s.sources) {
    original[src.id] = {src.initial_polygon.begin(), src.initial_polygon.end()};
    original_bounds.push_back(bound(original[src.id]));
  }
  BVH originals(original_bounds);
  std::map<int, std::vector<const Patch *>> per_source;
  size_t patches_checked = 0, rays_checked = 0, boundary_rays_skipped = 0;
  auto collect = [&](const std::vector<Patch> &patches) {
    for (auto &p : patches) {
      require(p.points.size() >= 3 && area2(p.points) > 0, "nonpositive patch");
      auto &base = original.at(p.source);
      for (size_t j = 0; j < base.size(); ++j)
        for (auto &v : p.points)
          require(side_sign(base[j], base[(j + 1) % base.size()], v) >= 0,
                  "patch outside original source");
      per_source[p.source].push_back(&p);
      ++patches_checked;
    }
  };
  for (size_t k = 0; k < r.layers.size(); ++k) {
    const auto &l = r.layers[k];
    collect(l.patches);
    std::vector<Bounds> boxes;
    for (auto &p : l.patches)
      boxes.push_back(bound(p.points));
    BVH tree(boxes);
    for (size_t i = 0; i < l.patches.size(); ++i)
      tree.visit(boxes[i], [&](int j) {
        if (j > static_cast<int>(i))
          require(separated(l.patches[i].points, l.patches[j].points) ||
                      separated(l.patches[j].points, l.patches[i].points),
                  "positive-area overlap within layer");
      });
    std::map<int, Q> patch_area, face_area;
    for (auto &p : l.patches)
      patch_area[p.source] += area2(p.points);
    for (auto &f : l.faces) {
      Q a = side(f.points[0], f.points[1], f.points[2]);
      require(a > 0, "nonpositive output triangle");
      face_area[f.source] += a;
    }
    require(patch_area == face_area, "mesh and patch areas disagree");
    // Independent camera-space Moller-Trumbore ray/triangle intersections.
    // This checks sampled ranks, not an all-region rank certificate.
    size_t count = std::min<size_t>(64, l.faces.size());
    for (size_t j = 0; j < count; ++j) {
      const auto &f = l.faces[j * l.faces.size() / count];
      P p((f.points[0].x() + f.points[1].x() + f.points[2].x()) / 3,
          (f.points[0].y() + f.points[1].y() + f.points[2].y()) / 3);
      P3 origin = s.camera.perspective ? P3(0, 0, 0) : P3(p.x(), p.y(), 0);
      K::Vector_3 direction =
          s.camera.perspective ? K::Vector_3(p.x(), p.y(), 1) : K::Vector_3(0, 0, 1);
      Bounds point_box;
      point_box.add(p);
      std::vector<std::pair<Q, int>> hits;
      bool boundary = false;
      originals.visit(point_box, [&](int index) {
        auto &src = s.sources[index];
        auto e1 = src.cam[1] - src.cam[0], e2 = src.cam[2] - src.cam[0];
        auto h = CGAL::cross_product(direction, e2);
        Q det = e1 * h;
        if (det == 0)
          return;
        auto v = origin - src.cam[0];
        Q u = (v * h) / det;
        if (u < 0 || u > 1)
          return;
        auto q = CGAL::cross_product(v, e1);
        Q b = (direction * q) / det;
        if (b < 0 || u + b > 1)
          return;
        Q depth = (e2 * q) / det;
        if (s.camera.perspective && depth < s.camera.near)
          return;
        if (u == 0 || b == 0 || u + b == 1)
          boundary = true;
        hits.emplace_back(depth, src.id);
      });
      std::sort(hits.begin(), hits.end());
      for (size_t i = 1; i < hits.size(); ++i)
        boundary |= hits[i - 1].first == hits[i].first;
      if (boundary) {
        ++boundary_rays_skipped;
        continue;
      }
      require(hits.size() > k && hits[k].second == f.source, "independent ray rank mismatch");
      ++rays_checked;
    }
  }
  collect(r.residual_patches);
  for (auto &src : s.sources) {
    Q emitted = 0;
    auto &ps = per_source[src.id];
    for (size_t i = 0; i < ps.size(); ++i) {
      emitted += area2(ps[i]->points);
      for (size_t j = 0; j < i; ++j)
        require(separated(ps[i]->points, ps[j]->points) || separated(ps[j]->points, ps[i]->points),
                "source overlaps itself across ranks");
    }
    require(emitted == area2(original[src.id]), "source conservation");
  }
  require(!complete || r.residual_patches.empty(), "all-layer extraction incomplete");
  require(r.layers.size() <= s.sources.size(), "too many layers");
  return {{"status", "passed"},
          {"patches_checked", patches_checked},
          {"source_containment_disjointness_conservation", true},
          {"layer_patch_disjointness", true},
          {"positive_triangle_and_patch_area_agreement", true},
          {"independent_exact_ray_ranks_checked", rays_checked},
          {"boundary_ray_samples_skipped", boundary_rays_skipped},
          {"full_region_rank_certificate", false},
          {"all_layers_complete", complete}};
}
void write_convex_result(const Scene &s, Result &r, const std::string &path, Json metadata,
                         bool meshes) {
  std::filesystem::create_directories(path);
  auto t = Clock::now();
  Json passes = Json::array();
  for (size_t k = 0; k < r.layers.size(); ++k) {
    auto &l = r.layers[k];
    auto p = l.stats.json();
    p["layer_index"] = k + 1;
    p["input_residual"] = l.residual_complexity;
    p["phases"] = l.phase_metrics;
    p["output"] = patch_complexity(l.patches);
    p["exported_triangles"] = l.faces.size();
    passes.push_back(p);
    if (meshes) {
      std::ofstream obj(path + "/layer-" + std::to_string(k + 1) + ".obj");
      obj << std::setprecision(17);
      size_t corner = 1;
      for (auto &f : l.faces) {
        for (auto &p : f.points) {
          auto w = lift(s, s.source(f.source), p);
          obj << "v " << CGAL::to_double(w.x()) << ' ' << CGAL::to_double(w.y()) << ' '
              << CGAL::to_double(w.z()) << '\n';
        }
        obj << "g source_" << f.source << "\nf " << corner << ' ' << corner + 1 << ' ' << corner + 2
            << '\n';
        corner += 3;
      }
    }
  }
  metadata["passes"] = passes;
  metadata["input_sources"] = s.input_sources;
  metadata["active_sources"] = s.sources.size();
  metadata["zero_projected_area_sources"] = s.zero_area;
  metadata["clipped_empty_sources"] = s.clipped_empty;
  metadata["remaining_patches"] = r.residual_patches.size();
  metadata["serialization_seconds"] = seconds(t);
  metadata["output_mode"] = meshes ? "obj-and-metrics" : "metrics-only";
  struct rusage usage {};
  getrusage(RUSAGE_SELF, &usage);
  metadata["peak_rss_kib"] = usage.ru_maxrss;
  std::ofstream mf(path + "/metrics.json");
  mf << metadata.dump(2) << '\n';
}
} // namespace peel
