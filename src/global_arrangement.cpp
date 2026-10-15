#include "peel_common.h"
#include "topology.h"
#include <CGAL/FPU.h>
#include <limits>

namespace peel {
namespace {
using Ring = std::vector<P>;
// Broad phase only: every bound encloses the represented exact geometry.
struct Bounds {
  double lo[2] = {INFINITY, INFINITY}, hi[2] = {-INFINITY, -INFINITY};
  double center(int axis) const { return lo[axis] + hi[axis]; }
  void add(const P &p) {
    for (int k = 0; k < 2; ++k) {
      auto v = CGAL::to_interval(p[k]);
      lo[k] = std::min(lo[k], v.first);
      hi[k] = std::max(hi[k], v.second);
    }
  }
  void add(const Bounds &b) {
    for (int k = 0; k < 2; ++k) {
      lo[k] = std::min(lo[k], b.lo[k]);
      hi[k] = std::max(hi[k], b.hi[k]);
    }
  }
};
bool overlap(const Bounds &a, const Bounds &b) {
  return a.lo[0] <= b.hi[0] && b.lo[0] <= a.hi[0] && a.lo[1] <= b.hi[1] && b.lo[1] <= a.hi[1];
}
template <class Range> Bounds bounds_of(const Range &ring) {
  Bounds b;
  for (auto &p : ring)
    b.add(p);
  return b;
}
using Tree = common::BVH<Bounds>;
template <class F> void visit(const Tree &tree, const Bounds &box, bool accelerate, F &&f) {
  if (accelerate)
    tree.visit(box, f);
  else
    for (size_t i = 0; i < tree.boxes.size(); ++i)
      f(int(i));
}
Q cross(const P &a, const P &b, const P &p) {
  return (b.x() - a.x()) * (p.y() - a.y()) - (b.y() - a.y()) * (p.x() - a.x());
}
// Independent Sutherland-Hodgman intersection; no peeling visibility kernel.
Ring intersect_convex(Ring ring, const Ring &clip) {
  for (size_t i = 0; i < clip.size() && !ring.empty(); ++i) {
    Ring next;
    const auto &a = clip[i], &b = clip[(i + 1) % clip.size()];
    for (size_t j = 0; j < ring.size(); ++j) {
      const auto &u = ring[j], &v = ring[(j + 1) % ring.size()];
      Q du = cross(a, b, u), dv = cross(a, b, v);
      if (du >= 0 && (next.empty() || next.back() != u))
        next.push_back(u);
      if ((du < 0 && dv > 0) || (du > 0 && dv < 0)) {
        Q t = du / (du - dv);
        next.emplace_back(u.x() + t * (v.x() - u.x()), u.y() + t * (v.y() - u.y()));
      }
    }
    if (next.size() > 1 && next.front() == next.back())
      next.pop_back();
    ring = std::move(next);
  }
  auto p = polygon(std::move(ring));
  return {p.begin(), p.end()};
}
bool straddles(const Ring &ring, const Score &s) {
  bool pos = false, neg = false;
  for (auto &p : ring) {
    auto sign = CGAL::sign(s.at(p));
    pos |= sign > 0;
    neg |= sign < 0;
  }
  return pos && neg;
}
struct Work {
  std::vector<std::pair<int, Patch>> patches;
  uint64_t cell_candidates = 0, ray_candidates = 0, sample_retries = 0;
};
// A point on an omitted internal mesh edge would count both incident triangles.
// Choose a generic exact interior sample instead of using an epsilon/tie-break.
int rank_of(const Scene &s, const Tree &tree, size_t source, const Ring &ring, bool accelerate,
            Work &work) {
  Q x = 0, y = 0;
  for (auto &p : ring) {
    x += p.x();
    y += p.y();
  }
  P center(x / Q(ring.size()), y / Q(ring.size()));
  for (size_t attempt = 0;; ++attempt) {
    P sample = center;
    if (attempt) {
      // Noncollinear vertices define an interior parabola. Any input edge line
      // meets it at most twice, so finitely many forbidden samples cannot stall it.
      Q t = Q(1) / Q(attempt + 2), u = t * t, v = 1 - t - u;
      sample = P(v * ring[0].x() + t * ring[1].x() + u * ring[2].x(),
                 v * ring[0].y() + t * ring[1].y() + u * ring[2].y());
    }
    Q depth = s.sources[source].score.at(sample);
    Bounds box;
    box.add(sample);
    int rank = 0;
    bool ambiguous = false;
    visit(tree, box, accelerate, [&](int j) {
      if (j == int(source) || ambiguous)
        return;
      ++work.ray_candidates;
      auto side = s.sources[j].initial_polygon.bounded_side(sample);
      if (side == CGAL::ON_BOUNDARY) {
        ambiguous = true;
      } else if (side == CGAL::ON_BOUNDED_SIDE) {
        Q other = s.sources[j].score.at(sample);
        ambiguous |= other == depth;
        rank += other < depth;
      }
    });
    if (!ambiguous)
      return rank;
    ++work.sample_retries;
  }
}
} // namespace

Result global_arrangement(const Scene &s, int layers, int threads, bool accelerate,
                          bool use_contours) {
  Result result;
  result.convex_representation = true;
  auto start = Clock::now();
  std::vector<Ring> sources;
  std::vector<Bounds> source_boxes;
  for (auto &src : s.sources) {
    sources.emplace_back(src.initial_polygon.begin(), src.initial_polygon.end());
    source_boxes.push_back(bounds_of(sources.back()));
  }
  Tree source_tree(source_boxes);
  std::vector<K::Segment_2> edges;
  if (use_contours) {
    for (auto &e : s.contours)
      if (e[0] != e[1])
        edges.emplace_back(e[0], e[1]);
  } else {
    for (auto &ring : sources)
      for (size_t i = 0; i < ring.size(); ++i)
        edges.emplace_back(ring[i], ring[(i + 1) % ring.size()]);
  }
  size_t boundary_segments = edges.size();
  auto window = s.window.corners();
  for (int i = 0; i < 4; ++i)
    edges.emplace_back(window[i], window[(i + 1) % 4]);
  // Do not trust a resolved-mesh assertion for oracle correctness. Find all
  // interior depth-order changes, including unprocessed self-intersections.
  std::vector<std::vector<K::Segment_2>> equality(sources.size());
  std::vector<uint64_t> pairs(sources.size()), overlaps(sources.size());
#pragma omp parallel for schedule(dynamic, 32) num_threads(threads)
  for (size_t i = 0; i < sources.size(); ++i) {
    CGAL::Protect_FPU_rounding<true> rounding(CGAL_FE_UPWARD);
    visit(source_tree, source_boxes[i], accelerate, [&](int j) {
      if (j >= int(i))
        return;
      ++pairs[i];
      auto delta = difference(s.sources[i].score, s.sources[j].score);
      if (!straddles(sources[i], delta) && !straddles(sources[j], delta))
        return;
      ++overlaps[i];
      auto overlap = intersect_convex(sources[i], sources[j]);
      if (overlap.empty())
        return;
      std::vector<P> hits;
      for (size_t k = 0; k < overlap.size(); ++k) {
        auto &a = overlap[k], &b = overlap[(k + 1) % overlap.size()];
        Q da = delta.at(a), db = delta.at(b);
        if (da == 0)
          hits.push_back(a);
        if ((da < 0 && db > 0) || (da > 0 && db < 0)) {
          Q t = da / (da - db);
          hits.emplace_back(a.x() + t * (b.x() - a.x()), a.y() + t * (b.y() - a.y()));
        }
      }
      std::sort(hits.begin(), hits.end());
      if (hits.size() > 1 && hits.front() != hits.back())
        equality[i].emplace_back(hits.front(), hits.back());
    });
  }
  uint64_t equality_count = 0;
  for (auto &v : equality) {
    equality_count += v.size();
    edges.insert(edges.end(), v.begin(), v.end());
  }
  double boundary_seconds = seconds(start);
  start = Clock::now();
  auto cells = triangle_cells(node(edges, &result.eager_stats));
  double arrangement_seconds = seconds(start);
  result.eager_stats.raw_triangles = cells.size();
  start = Clock::now();
  std::vector<Bounds> cell_boxes;
  for (auto &cell : cells)
    cell_boxes.push_back(bounds_of(cell));
  Tree cell_tree(cell_boxes);
  std::vector<Work> work(sources.size());
#pragma omp parallel for schedule(dynamic, 16) num_threads(threads)
  for (size_t i = 0; i < sources.size(); ++i) {
    CGAL::Protect_FPU_rounding<true> rounding(CGAL_FE_UPWARD);
    auto &w = work[i];
    visit(cell_tree, source_boxes[i], accelerate, [&](int j) {
      ++w.cell_candidates;
      auto &cell = cells[j];
      auto piece = intersect_convex(sources[i], {cell.begin(), cell.end()});
      if (piece.empty())
        return;
      int rank = rank_of(s, source_tree, i, piece, accelerate, w);
      w.patches.push_back({rank, {s.sources[i].id, std::move(piece)}});
    });
  }
  uint64_t cell_candidates = 0, ray_candidates = 0, retries = 0, fragments = 0;
  for (auto &w : work) {
    cell_candidates += w.cell_candidates;
    ray_candidates += w.ray_candidates;
    retries += w.sample_retries;
    fragments += w.patches.size();
    for (auto &item : w.patches) {
      int rank = item.first;
      if (layers && rank >= layers) {
        result.residual_patches.push_back(std::move(item.second));
      } else {
        if (result.layers.size() <= size_t(rank))
          result.layers.resize(rank + 1);
        result.layers[rank].patches.push_back(std::move(item.second));
      }
    }
  }
  double labeling_seconds = seconds(start);
  double meshing_seconds = 0;
  for (auto &layer : result.layers) {
    start = Clock::now();
    layer.faces = triangulate_patches(layer.patches, threads);
    layer.stats.triangulation_seconds = seconds(start);
    meshing_seconds += layer.stats.triangulation_seconds;
  }
  result.global_statistics = {
      {"boundary_segments", boundary_segments},
      {"contour_reduction", use_contours},
      {"pair_candidates", std::accumulate(pairs.begin(), pairs.end(), uint64_t(0))},
      {"pair_polygon_intersections",
       std::accumulate(overlaps.begin(), overlaps.end(), uint64_t(0))},
      {"depth_equality_segments", equality_count},
      {"boundary_seconds", boundary_seconds},
      {"arrangement_seconds", arrangement_seconds},
      {"labeling_seconds", labeling_seconds},
      {"triangulation_seconds", meshing_seconds},
      {"cell_candidates", cell_candidates},
      {"ray_candidates", ray_candidates},
      {"generic_sample_retries", retries},
      {"source_cell_fragments", fragments},
      {"arrangement", result.eager_stats.json()}};
  return result;
}
} // namespace peel
