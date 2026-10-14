#include "geometry.h"
#include <CGAL/Env_surface_data_traits_3.h>
#include <CGAL/Env_triangle_traits_3.h>
#include <CGAL/envelope_3.h>
#include <set>
namespace peel {
Domains envelope(const Scene &s, const Domains &r, Stats &stats) {
  using Traits = CGAL::Env_triangle_traits_3<K>;
  using Data = CGAL::Env_surface_data_traits_3<Traits, int>;
  using Diagram = CGAL::Envelope_diagram_2<Data>;
  std::vector<Data::Surface_3> triangles;
  auto timer = Clock::now();
  // Residuals overlap: triangulate each original source independently as a solver
  // adapter. The final exporter always triangulates the whole labelled layer.
  for (auto &kv : r) {
    if (kv.second.is_empty())
      continue;
    for (auto &f : triangulate(Domains{{kv.first, kv.second}}, &stats)) {
      std::array<P3, 3> v;
      for (int i = 0; i < 3; ++i)
        v[i] = P3(f.points[i].x(), f.points[i].y(), s.source(kv.first).score.at(f.points[i]));
      triangles.emplace_back(Traits::Surface_3(v[0], v[1], v[2]), kv.first);
    }
  }
  stats.adapter_seconds = seconds(timer);
  timer = Clock::now();
  Diagram diagram;
  CGAL::lower_envelope_3(triangles.begin(), triangles.end(), diagram);
  stats.envelope_construction_seconds = seconds(timer);
  timer = Clock::now();
  stats.arrangement_vertices = diagram.number_of_vertices();
  stats.arrangement_edges = diagram.number_of_edges();
  stats.arrangement_faces = diagram.number_of_faces();
  Domains out;
  auto cycle = [](auto begin) {
    std::vector<P> ps;
    auto c = begin;
    do {
      ps.push_back(c->source()->point());
      ++c;
    } while (c != begin);
    return regularized_cycle(ps);
  };
  for (auto f = diagram.faces_begin(); f != diagram.faces_end(); ++f) {
    if (f->is_unbounded() || !f->number_of_surfaces())
      continue;
    std::set<int> ids;
    for (auto surf = f->surfaces_begin(); surf != f->surfaces_end(); ++surf)
      ids.insert(surf->data());
    if (ids.size() != 1)
      throw std::runtime_error("Unsupported positive-area envelope tie");
    Set part = cycle(f->outer_ccb());
    for (auto h = f->holes_begin(); h != f->holes_end(); ++h)
      part = subtract(part, cycle(*h), &stats);
    if (!part.is_empty()) {
      ++stats.raw_regions;
      stats.raw_boundary_vertices += vertices(part);
      unite(out[*ids.begin()], part, &stats);
    }
  }
  stats.normalization_seconds = seconds(timer);
  return out;
}
} // namespace peel
