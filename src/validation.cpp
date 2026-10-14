#include "topology.h"
#include <fstream>
#include <set>
namespace peel {
static void require(bool v, const std::string &m) {
  if (!v)
    throw std::runtime_error("Validation failed: " + m);
}
void validate(const Scene &s, Result &r, bool complete) {
  Domains emitted;
  Q total = 0, initial_area = 0;
  for (auto &src : s.sources)
    initial_area += area(src.initial);
  for (auto &l : r.layers) {
    auto timer = Clock::now();
    Set covered;
    Domains mesh;
    std::set<P> points;
    std::vector<K::Segment_2> edges;
    for (auto &kv : l.regions) {
      require(subtract(kv.second, s.source(kv.first).initial).is_empty(), "source containment");
      require(intersect(covered, kv.second).is_empty(), "layer uniqueness");
      unite(covered, kv.second);
      require(intersect(emitted[kv.first], kv.second).is_empty(), "per-source rank disjointness");
      unite(emitted[kv.first], kv.second);
      total += area(kv.second);
    }
    Set mesh_all;
    for (auto &f : l.faces) {
      require(CGAL::orientation(f.points[0], f.points[1], f.points[2]) == CGAL::LEFT_TURN,
              "positive triangle area");
      Set tri = domain(polygon({f.points.begin(), f.points.end()}));
      require(intersect(mesh_all, tri).is_empty(), "triangle interior overlap");
      unite(mesh_all, tri);
      require(l.regions.count(f.source) && subtract(tri, l.regions.at(f.source)).is_empty(),
              "triangle label and containment");
      unite(mesh[f.source], tri);
      for (int i = 0; i < 3; ++i) {
        P p = f.points[i];
        points.insert(p);
        edges.emplace_back(p, f.points[(i + 1) % 3]);
        auto w = lift(s, s.source(f.source), p);
        require(project(s, w) == p, "exact reprojection");
        auto b = barycentric(s.source(f.source), w);
        require(b[0] >= 0 && b[1] >= 0 && b[2] >= 0 && b[0] + b[1] + b[2] == 1,
                "source barycentrics");
        auto &src = s.source(f.source);
        for (int axis = 0; axis < 3; ++axis)
          require(b[0] * src.world[0][axis] + b[1] * src.world[1][axis] +
                          b[2] * src.world[2][axis] ==
                      w[axis],
                  "original source plane");
      }
    }
    for (auto &kv : l.regions)
      require(equal(kv.second, mesh[kv.first]), "triangulated region symmetric difference");
    for (auto &p : points)
      for (auto &e : edges)
        if (p != e.source() && p != e.target())
          require(!e.has_on(p), "T-junction");
    l.stats.validation_seconds = seconds(timer);
  }
  for (auto &src : s.sources) {
    Set remainder;
    auto it = r.residual.find(src.id);
    if (it != r.residual.end())
      remainder = it->second;
    require(intersect(emitted[src.id], remainder).is_empty(), "residual disjointness");
    unite(emitted[src.id], remainder);
    require(equal(emitted[src.id], src.initial), "per-source conservation");
    if (complete)
      require(remainder.is_empty(), "all-layer completion");
  }
  require(r.layers.size() <= s.sources.size(), "maximum layer count");
  if (complete)
    require(total == initial_area, "projected multiplicity area");
}
void compare(const Result &a, const Result &b) {
  require(a.layers.size() == b.layers.size(), "different layer counts");
  for (size_t k = 0; k < a.layers.size(); ++k) {
    std::set<int> ids;
    for (auto &kv : a.layers[k].regions)
      ids.insert(kv.first);
    for (auto &kv : b.layers[k].regions)
      ids.insert(kv.first);
    for (int id : ids) {
      Set x, y;
      auto i = a.layers[k].regions.find(id), j = b.layers[k].regions.find(id);
      if (i != a.layers[k].regions.end())
        x = i->second;
      if (j != b.layers[k].regions.end())
        y = j->second;
      require(equal(x, y), "labelled region difference at layer " + std::to_string(k + 1) +
                               ", source " + std::to_string(id));
    }
  }
}
void compare_files(const std::string &a, const std::string &b) {
  auto read = [](const std::string &path) {
    std::ifstream f(path);
    if (!f)
      throw std::runtime_error("Cannot open exact output: " + path);
    Json j;
    f >> j;
    Result r;
    for (auto &l : j.at("layers")) {
      Layer layer;
      for (auto &v : l.at("regions"))
        layer.regions[v.at("source_id").get<int>()] = read_set(v.at("domain"));
      r.layers.push_back(std::move(layer));
    }
    return r;
  };
  compare(read(a), read(b));
}
} // namespace peel
