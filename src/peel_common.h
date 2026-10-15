#pragma once
// Shared combinatorial algorithm. Numeric policies own predicates, constructions,
// outward bounds and rounding; this layer never converts exact coordinates.
#include <algorithm>
#include <chrono>
#include <map>
#include <nlohmann/json.hpp>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>
namespace peel::common {
using Clock = std::chrono::steady_clock;
inline double seconds(Clock::time_point t) {
  return std::chrono::duration<double>(Clock::now() - t).count();
}
struct NoRoundingGuard {};
template <class Box> class BVH {
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
      // Widen integer grid centers without changing floating-point exact bounds.
      auto x = boxes[a].center(axis), y = boxes[b].center(axis);
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
// Topological classification is identical for exact coordinates and grid points.
template <class World, class Point> struct Contours {
  struct Incident {
    int direction, facing;
    std::array<Point, 2> projected;
  };
  std::map<World, int> vertices;
  std::map<std::pair<int, int>, std::vector<Incident>> edges;
  size_t clipped = 0;
  int vertex(const World &p) { return vertices.emplace(p, vertices.size()).first->second; }
  void edge(int a, int b, int facing, Point p, Point q) {
    edges[{std::min(a, b), std::max(a, b)}].push_back({a < b ? 1 : -1, facing, {p, q}});
  }
  nlohmann::json finish(bool enabled, std::vector<std::array<Point, 2>> &out) {
    size_t boundary = 0, silhouette = 0, nonmanifold = 0, ignored = 0;
    for (auto &entry : edges) {
      auto &f = entry.second;
      if (f.size() == 2 && f[0].direction != f[1].direction && f[0].facing == f[1].facing) {
        ++ignored;
        continue;
      }
      if (f.size() == 1)
        ++boundary;
      else if (f.size() != 2 || f[0].direction == f[1].direction)
        ++nonmanifold;
      else
        ++silhouette;
      out.push_back(f.front().projected);
    }
    return {{"resolved_mesh_contract", enabled},
            {"boundary_edges", boundary},
            {"silhouette_edges", silhouette},
            {"nonmanifold_or_inconsistent_edges", nonmanifold},
            {"same_facing_internal_edges_ignored", ignored},
            {"camera_clipping_boundary_edges", clipped},
            {"contour_segments", out.size()}};
  }
};
// One ray only after the entire source has been certified clear of contours.
// Numeric policy returns -1 for an ambiguous edge/tie and falls back to clipping.
template <class Policy, class Tree>
std::vector<int> classify(Policy &p, const Tree &sources, int threads, nlohmann::json &metrics) {
  auto t = Clock::now();
  std::vector<int> ranks(sources.boxes.size(), -1);
  if (!p.resolved())
    return ranks;
  auto edges = p.contour_boxes();
  Tree contours(edges);
  std::vector<uint64_t> candidates(ranks.size(), 0);
#pragma omp parallel for schedule(dynamic, 64) num_threads(threads)
  for (size_t i = 0; i < ranks.size(); ++i) {
    [[maybe_unused]] auto guard = p.guard();
    bool boundary = false;
    contours.visit(sources.boxes[i], [&](int j) {
      if (!boundary)
        boundary = p.touches_contour(i, j);
    });
    if (boundary || !p.rank_eligible(i))
      continue;
    auto sample = p.sample(i);
    int rank = 0;
    bool ambiguous = false;
    sources.visit(sample.second, [&](int j) {
      if (j == int(i))
        return;
      ++candidates[i];
      int hit = p.ray_order(i, j, sample.first);
      ambiguous |= hit < 0;
      rank += hit > 0;
    });
    if (!ambiguous)
      ranks[i] = rank;
  }
  size_t known = std::count_if(ranks.begin(), ranks.end(), [](int r) { return r >= 0; });
  metrics = {{"constant_rank_sources", known},
             {"sources_requiring_clipping", ranks.size() - known},
             {"point_query_candidates",
              std::accumulate(candidates.begin(), candidates.end(), uint64_t(0))},
             {"classification_seconds", seconds(t)}};
  return ranks;
}
// Near-first traversal; rank mask retires static whole triangles without rebuilding.
template <class Policy, class Patches, class Tree, class Patch, class Box, class Work>
void query(Policy &p, const Patches &input, const Tree &tree, const Patch &target, const Box &box,
           Work &w, const std::vector<int> *ranks = nullptr, int layer = 0, int i = 0) {
  if (w.visible.empty() || tree.nodes.empty())
    return;
  auto &node = tree.nodes[i];
  p.node_test(w);
  if (!overlap(node.box, box) || p.behind(node.box, box)) {
    p.box_reject(w);
    return;
  }
  if (node.left < 0) {
    for (int j = node.begin; j < node.end && !w.visible.empty(); ++j) {
      int c = tree.ids[j];
      if (input[c].source == target.source || (ranks && (*ranks)[c] < layer) ||
          !overlap(tree.boxes[c], box) || p.behind(tree.boxes[c], box))
        continue;
      p.cut(w, target, input[c], true);
    }
  } else {
    int a = node.left, b = node.right;
    if (p.near(tree.nodes[a].box) > p.near(tree.nodes[b].box))
      std::swap(a, b);
    query(p, input, tree, target, box, w, ranks, layer, a);
    query(p, input, tree, target, box, w, ranks, layer, b);
  }
}
// Shared conforming-boundary traversal. Geometry policy supplies exact/grid
// segment membership and centroid construction, with the same labelled topology.
template <class Point, class Box, class Ring, class Face, class FaceBuffer, class Patches,
          class Guard, class Contains, class Emit>
std::vector<Face> mesh(const Patches &patches, int threads, Guard guard, Contains contains,
                       Emit emit) {
  std::vector<Point> points;
  for (auto &p : patches)
    points.insert(points.end(), p.points.begin(), p.points.end());
  std::sort(points.begin(), points.end());
  points.erase(std::unique(points.begin(), points.end()), points.end());
  std::vector<Box> boxes(points.size());
  for (size_t i = 0; i < points.size(); ++i)
    boxes[i].add(points[i]);
  BVH<Box> tree(boxes);
  std::vector<FaceBuffer> output(patches.size());
#pragma omp parallel for schedule(dynamic, 32) num_threads(threads)
  for (size_t k = 0; k < patches.size(); ++k) {
    [[maybe_unused]] auto rounding = guard();
    auto &patch = patches[k];
    Ring ring;
    for (size_t i = 0; i < patch.points.size(); ++i) {
      auto &a = patch.points[i];
      auto &b = patch.points[(i + 1) % patch.points.size()];
      Box box;
      box.add(a);
      box.add(b);
      Ring along;
      tree.visit(box, [&](int j) {
        auto &v = points[j];
        if (v != b && contains(a, b, v))
          along.push_back(v);
      });
      std::sort(along.begin(), along.end());
      if (b < a)
        std::reverse(along.begin(), along.end());
      ring.insert(ring.end(), along.begin(), along.end());
    }
    emit(patch.source, ring, output[k]);
  }
  size_t n = 0;
  for (auto &v : output)
    n += v.size();
  std::vector<Face> out;
  out.reserve(n);
  for (auto &v : output)
    out.insert(out.end(), std::make_move_iterator(v.begin()), std::make_move_iterator(v.end()));
  return out;
}
// Shared all-layer scheduler, fragment conservation and prefix residual handling.
template <class Policy>
auto extract(Policy &p, int layers, int threads, const std::string &order, unsigned seed,
             bool accelerate) {
  using Patch = typename Policy::PatchType;
  using Tree = BVH<typename Policy::BoxType>;
  typename Policy::ResultType result;
  p.initialize(result);
  auto originals = p.originals();
  // No extra static index in the clipping-only reference/control path.
  auto original_boxes = p.resolved() ? p.boxes(originals) : std::vector<typename Policy::BoxType>{};
  Tree original_tree(original_boxes);
  auto ranks = p.resolved() ? classify(p, original_tree, threads, result.classification)
                            : std::vector<int>(originals.size(), -1);
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
  int limit = layers ? layers : originals.size();
  for (int k = 0; k < limit && (!residual.empty() || size_t(k) < direct.size()); ++k) {
    auto t = Clock::now();
    auto boxes = p.boxes(residual);
    Tree tree(boxes);
    std::vector<typename Policy::WorkType> work(residual.size());
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
      [[maybe_unused]] auto guard = p.guard();
      int i = schedule[j];
      auto &w = work[i];
      auto &target = residual[i];
      w.visible.push_back(target.points);
      if (accelerate) {
        query(p, originals, original_tree, target, boxes[i], w, &ranks, k);
        query(p, residual, tree, target, boxes[i], w);
      } else {
        for (size_t c = 0; c < originals.size() && !w.visible.empty(); ++c)
          if (ranks[c] >= k && originals[c].source != target.source)
            p.cut(w, target, originals[c], false);
        for (int c : schedule)
          if (residual[c].source != target.source && !w.visible.empty())
            p.cut(w, target, residual[c], false);
      }
      if (w.visible.empty()) {
        w.hidden.clear();
        w.hidden.push_back(target.points);
        p.occluded(w);
      }
    }
    typename Policy::LayerType layer;
    if (size_t(k) < direct.size())
      for (int i : direct[k])
        layer.patches.push_back(originals[i]);
    std::vector<Patch> next;
    for (size_t i = 0; i < work.size(); ++i) {
      for (auto &ring : work[i].visible)
        layer.patches.push_back({residual[i].source, std::move(ring)});
      for (auto &ring : work[i].hidden)
        next.push_back({residual[i].source, std::move(ring)});
    }
    if (layer.patches.empty())
      throw std::runtime_error("BVH peeling made no progress");
    p.finish(layer, k, residual, work, seconds(t), threads);
    result.layers.push_back(std::move(layer));
    residual = std::move(next);
  }
  if (!layers && !residual.empty())
    throw std::runtime_error("Peeling exceeded source-count layer limit");
  for (size_t k = result.layers.size(); k < direct.size(); ++k)
    for (int i : direct[k])
      residual.push_back(originals[i]);
  p.residual(result, std::move(residual));
  return result;
}
} // namespace peel::common
