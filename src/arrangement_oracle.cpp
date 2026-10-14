#include "topology.h"
#include <algorithm>
namespace peel {
Result arrangement(const Scene &s, int layers, int cap) {
  if (static_cast<int>(s.sources.size()) > cap)
    throw std::runtime_error("Arrangement source cap exceeded (" + std::to_string(cap) + ")");
  Result r;
  auto t = Clock::now();
  Domains initial;
  for (auto &src : s.sources) {
    initial[src.id] = src.initial;
    r.residual[src.id] = src.initial;
  }
  auto edges = boundaries(initial);
  auto wc = s.window.corners();
  for (int i = 0; i < 4; ++i)
    edges.emplace_back(wc[i], wc[(i + 1) % 4]);
  for (size_t i = 0; i < s.sources.size(); ++i)
    for (size_t j = 0; j < i; ++j) {
      ++r.eager_stats.candidate_tests;
      auto &a = s.sources[i];
      auto &b = s.sources[j];
      if (!overlaps(bounds(a.initial), bounds(b.initial))) {
        ++r.eager_stats.box_rejections;
        continue;
      }
      auto delta = difference(a.score, b.score);
      if (delta.a == 0 && delta.b == 0)
        continue;
      Set overlap = intersect(a.initial, b.initial, &r.eager_stats);
      if (overlap.is_empty())
        continue;
      ++r.eager_stats.halfplane_clips;
      // Initial domains are convex (triangle clipped by near plane and window).
      // Find the equality segment directly, preserving exact rational endpoints.
      for (auto &p : pieces(overlap)) {
        std::vector<P> hits;
        for (auto e = p.outer_boundary().edges_begin(); e != p.outer_boundary().edges_end(); ++e) {
          P u = e->source(), v = e->target();
          Q du = delta.at(u), dv = delta.at(v);
          if (du == 0)
            hits.push_back(u);
          if ((du < 0 && dv > 0) || (du > 0 && dv < 0)) {
            Q q = du / (du - dv);
            hits.emplace_back(u.x() + q * (v.x() - u.x()), u.y() + q * (v.y() - u.y()));
          }
        }
        std::sort(hits.begin(), hits.end());
        hits.erase(std::unique(hits.begin(), hits.end()), hits.end());
        if (hits.size() > 1)
          edges.emplace_back(hits.front(), hits.back());
      }
    }
  auto cells = triangle_cells(node(edges, &r.eager_stats));
  r.eager_stats.raw_triangles = cells.size();
  for (auto &cell : cells) {
    P p = centroid(cell);
    std::vector<std::pair<Q, int>> hits;
    for (auto &src : s.sources)
      if (contains(src.initial, p))
        hits.emplace_back(src.score.at(p), src.id);
    std::sort(hits.begin(), hits.end());
    Json ids = Json::array();
    for (auto &h : hits)
      ids.push_back(h.second);
    r.oracle_cells.push_back({{"interior_point", jp(p)},
                              {"depth_order", ids},
                              {"triangle", Json::array({jp(cell[0]), jp(cell[1]), jp(cell[2])})}});
    int count = layers ? std::min<int>(layers, hits.size()) : hits.size();
    if (static_cast<int>(r.layers.size()) < count)
      r.layers.resize(count);
    Set piece = domain(polygon({cell.begin(), cell.end()}));
    for (int k = 0; k < count; ++k) {
      ++r.layers[k].stats.raw_regions;
      r.layers[k].stats.raw_boundary_vertices += 3;
      unite(r.layers[k].regions[hits[k].second], piece, &r.eager_stats);
    }
  }
  r.eager_stats.visibility_seconds = seconds(t);
  for (auto &l : r.layers) {
    l.residual_complexity = complexity(r.residual);
    t = Clock::now();
    for (auto &kv : l.regions)
      r.residual[kv.first] = subtract(r.residual[kv.first], kv.second, &l.stats);
    l.stats.residual_seconds = seconds(t);
    t = Clock::now();
    l.faces = triangulate(l.regions);
    l.stats.triangulation_seconds = seconds(t);
  }
  return r;
}
} // namespace peel
