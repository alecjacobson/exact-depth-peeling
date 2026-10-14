#include "geometry.h"
#include <algorithm>
#include <random>
#include <set>
namespace peel {
static bool nearer_box(const Score &d, const Box &b) {
  for (auto &p : b.corners())
    if (d.at(p) >= 0)
      return false;
  return true;
}
static bool certificate(const Set &d, const Box &b) {
  for (auto &p : pieces(d)) {
    if (p.number_of_holes() || !p.outer_boundary().is_convex())
      continue;
    bool inside = true;
    for (auto &corner : b.corners())
      if (p.outer_boundary().bounded_side(corner) == CGAL::ON_UNBOUNDED_SIDE) {
        inside = false;
        break;
      }
    if (inside)
      return true;
  }
  return false;
}
// Exact component-box index. Rebuilt after map mutation; queries scan the
// xmin-sorted prefix and test the remaining bounds conservatively.
class ComponentIndex {
  struct Entry {
    Box box;
    int source;
  };
  std::vector<Entry> entries;

public:
  void rebuild(const Domains &map, Stats &stats) {
    entries.clear();
    for (auto &kv : map)
      for (auto &component : pieces(kv.second)) {
        entries.push_back({bounds(Set(component)), kv.first});
      }
    std::sort(entries.begin(), entries.end(),
              [](const Entry &a, const Entry &b) { return a.box.xmin < b.box.xmin; });
    ++stats.index_rebuilds;
  }
  std::vector<int> query(const Box &box, Stats &stats) const {
    std::set<int> found;
    for (auto &e : entries) {
      if (e.box.xmin >= box.xmax)
        break;
      ++stats.index_box_tests;
      if (overlaps(e.box, box))
        found.insert(e.source);
      else
        ++stats.box_rejections;
    }
    return {found.begin(), found.end()};
  }
};
Domains visible(const Scene &s, const Domains &r, const std::string &order, unsigned seed,
                bool accelerate, Stats &stats) {
  std::vector<int> ids;
  for (auto &src : s.sources) {
    auto it = r.find(src.id);
    if (it != r.end() && !it->second.is_empty())
      ids.push_back(src.id);
  }
  if (order == "random") {
    std::mt19937 gen(seed);
    std::shuffle(ids.begin(), ids.end(), gen);
  } else if (order == "reverse")
    std::reverse(ids.begin(), ids.end());
  else if (order == "near-first") {
    std::map<int, Q> mins;
    for (int i : ids) {
      auto cs = bounds(r.at(i)).corners();
      Q v = s.source(i).score.at(cs[0]);
      for (auto &p : cs)
        v = std::min(v, s.source(i).score.at(p));
      mins[i] = v;
    }
    std::stable_sort(ids.begin(), ids.end(), [&](int a, int b) { return mins[a] < mins[b]; });
  } else if (order != "input")
    throw std::runtime_error("Unknown insertion order");
  Domains m;
  ComponentIndex index;
  const long residual_vertices = vertices(r);
  for (int i : ids) {
    Set pending = r.at(i);
    Box cb = bounds(pending);
    std::vector<int> candidates;
    if (accelerate)
      candidates = index.query(cb, stats);
    else
      for (auto &kv : m)
        if (!kv.second.is_empty())
          candidates.push_back(kv.first);
    if (accelerate) {
      bool reject = false;
      for (int old_id : candidates) {
        auto &old = *m.find(old_id);
        if (old.second.is_empty())
          continue;
        ++stats.certificate_tests;
        if (nearer_box(difference(s.source(old.first).score, s.source(i).score), cb) &&
            certificate(old.second, cb)) {
          reject = true;
          ++stats.occlusion_rejections;
          break;
        }
      }
      if (reject)
        continue;
    }
    // Each old source is visited once. Updating it cannot affect another old region;
    // incoming geometry is inserted only after this snapshot traversal finishes.
    for (int old_id : candidates) {
      auto &old = *m.find(old_id);
      if (pending.is_empty())
        break;
      if (old.second.is_empty())
        continue;
      ++stats.candidate_tests;
      Box pb = bounds(pending);
      if (accelerate && !overlaps(pb, bounds(old.second))) {
        ++stats.box_rejections;
        continue;
      }
      Score delta = difference(s.source(i).score, s.source(old.first).score);
      if (accelerate && nearer_box(delta, pb)) {
        ++stats.depth_separated;
        old.second = subtract(old.second, pending, &stats);
        continue;
      }
      if (accelerate && nearer_box(difference(s.source(old.first).score, s.source(i).score), pb)) {
        ++stats.depth_separated;
        pending = subtract(pending, old.second, &stats);
        continue;
      }
      Set overlap = intersect(pending, old.second, &stats);
      if (overlap.is_empty())
        continue;
      ++stats.halfplane_clips;
      Set wins = intersect(overlap, halfplane(delta, s.window), &stats);
      Set loses = subtract(overlap, wins, &stats);
      if (!wins.is_empty())
        old.second = subtract(old.second, wins, &stats);
      pending = subtract(pending, loses, &stats);
    }
    if (!pending.is_empty())
      m.emplace(i, std::move(pending));
    if (accelerate)
      index.rebuild(m, stats);
    long n = vertices(m);
    stats.max_map_vertices = std::max(stats.max_map_vertices, n);
    stats.peak_live_boundary_vertices =
        std::max(stats.peak_live_boundary_vertices, n + residual_vertices);
  }
  for (auto it = m.begin(); it != m.end();)
    if (it->second.is_empty())
      it = m.erase(it);
    else
      ++it;
  return m;
}
Result run(const Scene &s, const std::string &method, int layers, const std::string &order,
           unsigned seed, bool accelerate, int cap) {
  if (method == "arrangement")
    return arrangement(s, layers, cap);
  if (method != "visible-peel" && method != "envelope-peel")
    throw std::runtime_error("Unknown method");
  Result out;
  for (auto &src : s.sources)
    out.residual[src.id] = src.initial;
  int limit = layers ? layers : s.sources.size();
  for (int k = 0; k < limit; ++k) {
    bool empty = true;
    for (auto &kv : out.residual)
      if (!kv.second.is_empty())
        empty = false;
    if (empty)
      break;
    Layer layer;
    layer.residual_complexity = complexity(out.residual);
    auto t = Clock::now();
    layer.regions = method == "visible-peel"
                        ? visible(s, out.residual, order, seed, accelerate, layer.stats)
                        : envelope(s, out.residual, layer.stats);
    layer.stats.visibility_seconds = seconds(t);
    if (layer.regions.empty())
      throw std::runtime_error("Peeling made no progress");
    t = Clock::now();
    for (auto &kv : layer.regions)
      out.residual[kv.first] = subtract(out.residual.at(kv.first), kv.second, &layer.stats);
    layer.stats.residual_seconds = seconds(t);
    t = Clock::now();
    layer.faces = triangulate(layer.regions);
    layer.stats.triangulation_seconds = seconds(t);
    out.layers.push_back(std::move(layer));
  }
  return out;
}
} // namespace peel
