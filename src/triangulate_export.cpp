#include "topology.h"
#include <CGAL/Arr_segment_traits_2.h>
#include <CGAL/Arrangement_2.h>
#include <CGAL/Constrained_Delaunay_triangulation_2.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sys/resource.h>
namespace peel {
std::vector<K::Segment_2> boundaries(const Domains &d) {
  std::vector<K::Segment_2> v;
  auto add = [&](const Poly &p) {
    for (auto e = p.edges_begin(); e != p.edges_end(); ++e)
      if (!e->is_degenerate())
        v.push_back(*e);
  };
  for (auto &kv : d)
    for (auto &p : pieces(kv.second)) {
      add(p.outer_boundary());
      for (auto h = p.holes_begin(); h != p.holes_end(); ++h)
        add(*h);
    }
  return v;
}
std::vector<K::Segment_2> node(const std::vector<K::Segment_2> &segments, Stats *stats) {
  using Arr = CGAL::Arrangement_2<CGAL::Arr_segment_traits_2<K>>;
  Arr arr;
  CGAL::insert(arr, segments.begin(), segments.end());
  if (stats) {
    stats->arrangement_vertices += arr.number_of_vertices();
    stats->arrangement_edges += arr.number_of_edges();
    stats->arrangement_faces += arr.number_of_faces();
  }
  std::vector<K::Segment_2> out;
  for (auto e = arr.edges_begin(); e != arr.edges_end(); ++e)
    out.emplace_back(e->source()->point(), e->target()->point());
  return out;
}
std::vector<K::Segment_2> noded_edges(const Domains &d, Stats *stats) {
  return node(boundaries(d), stats);
}
std::vector<std::array<P, 3>> triangle_cells(const std::vector<K::Segment_2> &edges) {
  using V = CGAL::Triangulation_vertex_base_2<K>;
  using F = CGAL::Constrained_triangulation_face_base_2<K>;
  using T = CGAL::Triangulation_data_structure_2<V, F>;
  using CDT = CGAL::Constrained_Delaunay_triangulation_2<K, T, CGAL::Exact_intersections_tag>;
  CDT cdt;
  for (auto &e : edges) {
    cdt.insert(e.source());
    cdt.insert(e.target());
  }
  for (auto &e : edges)
    cdt.insert_constraint(e.source(), e.target());
  if (!cdt.is_valid())
    throw std::runtime_error("Invalid constrained triangulation");
  std::vector<std::array<P, 3>> out;
  for (auto f = cdt.finite_faces_begin(); f != cdt.finite_faces_end(); ++f)
    out.push_back({f->vertex(0)->point(), f->vertex(1)->point(), f->vertex(2)->point()});
  return out;
}
std::vector<Face> triangulate(const Domains &regions, Stats *stats) {
  auto cells = triangle_cells(noded_edges(regions));
  std::vector<Face> out;
  for (auto &t : cells) {
    P p = centroid(t);
    int id = 0, count = 0;
    for (auto &kv : regions)
      if (contains(kv.second, p)) {
        id = kv.first;
        ++count;
      }
    if (count > 1)
      throw std::runtime_error("Overlapping labelled regions in triangulation");
    if (count)
      out.push_back({id, t});
  }
  if (stats)
    stats->adapter_triangles += out.size();
  return out;
}
void write_result(const Scene &s, Result &r, const std::string &path, Json metadata) {
  std::filesystem::create_directories(path);
  auto start = Clock::now();
  auto matrix_json = [](const auto &m) {
    Json rows = Json::array();
    for (const auto &row : m) {
      Json r = Json::array();
      for (const auto &q : row)
        r.push_back(str(q));
      rows.push_back(r);
    }
    return rows;
  };
  Json exact_camera = {{"type", s.camera.perspective ? "perspective" : "orthographic"},
                       {"near", str(s.camera.near)},
                       {"world_to_camera", matrix_json(s.camera.world_to_camera)},
                       {"camera_to_world", matrix_json(s.camera.camera_to_world)}};
  Json exact = {{"format", "exact-depth-peeling-v1"},
                {"camera", exact_camera},
                {"window", Json::array({str(s.window.xmin), str(s.window.ymin), str(s.window.xmax),
                                        str(s.window.ymax)})},
                {"sources", Json::array()},
                {"layers", Json::array()},
                {"residual", Json::array()},
                {"oracle_cells", r.oracle_cells}};
  for (auto &src : s.sources) {
    Json w = Json::array();
    for (auto p : src.world)
      w.push_back(jp3(p));
    exact["sources"].push_back(
        {{"source_id", src.id},
         {"mesh_face_id", src.mesh_face},
         {"world_vertices", w},
         {"score", Json::array({str(src.score.a), str(src.score.b), str(src.score.c)})},
         {"initial_domain", set_json(src.initial)}});
  }
  Json passes = Json::array();
  for (size_t k = 0; k < r.layers.size(); ++k) {
    auto &l = r.layers[k];
    Json lj = {{"layer_index", k + 1}, {"regions", Json::array()}, {"faces", Json::array()}};
    for (auto &kv : l.regions)
      lj["regions"].push_back({{"source_id", kv.first}, {"domain", set_json(kv.second)}});
    std::ofstream obj(path + "/layer-" + std::to_string(k + 1) + ".obj");
    obj << std::setprecision(17);
    size_t corner = 1;
    for (auto &f : l.faces) {
      Json uv = Json::array(), world = Json::array(), bary = Json::array();
      for (auto &p : f.points) {
        auto w = lift(s, s.source(f.source), p);
        auto b = barycentric(s.source(f.source), w);
        uv.push_back(jp(p));
        world.push_back(jp3(w));
        bary.push_back(Json::array({str(b[0]), str(b[1]), str(b[2])}));
        obj << "v " << CGAL::to_double(w.x()) << ' ' << CGAL::to_double(w.y()) << ' '
            << CGAL::to_double(w.z()) << '\n';
      }
      lj["faces"].push_back(
          {{"source_id", f.source}, {"uv", uv}, {"world", world}, {"barycentric", bary}});
      obj << "g source_" << f.source << "\nf " << corner << ' ' << corner + 1 << ' ' << corner + 2
          << '\n';
      corner += 3;
    }
    std::ofstream svg(path + "/layer-" + std::to_string(k + 1) + ".svg");
    double x = CGAL::to_double(s.window.xmin), y = CGAL::to_double(s.window.ymin),
           w = CGAL::to_double(s.window.xmax - s.window.xmin),
           h = CGAL::to_double(s.window.ymax - s.window.ymin);
    svg << "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"" << x << ' ' << y << ' ' << w
        << ' ' << h << "\"><rect x=\"" << x << "\" y=\"" << y << "\" width=\"" << w
        << "\" height=\"" << h << "\" fill=\"white\"/>";
    for (auto &kv : l.regions) {
      svg << "<path fill=\"hsl(" << (static_cast<unsigned>(kv.first) * 137u) % 360
          << ",60%,65%)\" fill-rule=\"evenodd\" stroke=\"#222\" stroke-width=\"" << w / 800
          << "\" d=\"";
      auto ring = [&](const Poly &p) {
        bool first = true;
        for (auto v : p) {
          svg << (first ? "M " : "L ") << CGAL::to_double(v.x()) << ' ' << CGAL::to_double(v.y())
              << ' ';
          first = false;
        }
        svg << "Z ";
      };
      for (auto &p : pieces(kv.second)) {
        ring(p.outer_boundary());
        for (auto hi = p.holes_begin(); hi != p.holes_end(); ++hi)
          ring(*hi);
      }
      svg << "\"><title>source " << kv.first << "</title></path>";
    }
    svg << "</svg>\n";
    exact["layers"].push_back(lj);
    auto p = l.stats.json();
    p["layer_index"] = k + 1;
    p["input_residual"] = l.residual_complexity;
    p["phases"] = l.phase_metrics;
    p["output"] = complexity(l.regions);
    p["exported_triangles"] = l.faces.size();
    passes.push_back(p);
  }
  for (auto &kv : r.residual)
    exact["residual"].push_back({{"source_id", kv.first}, {"domain", set_json(kv.second)}});
  std::ofstream ef(path + "/exact.json");
  ef << exact.dump(2) << '\n';
  ef.close();
  metadata["passes"] = passes;
  metadata["eager_arrangement"] = r.eager_stats.json();
  metadata["input_sources"] = s.input_sources;
  metadata["active_sources"] = s.sources.size();
  metadata["zero_projected_area_sources"] = s.zero_area;
  metadata["clipped_empty_sources"] = s.clipped_empty;
  metadata["serialization_seconds"] = seconds(start);
  struct rusage usage {};
  getrusage(RUSAGE_SELF, &usage);
  metadata["peak_rss_kib"] = usage.ru_maxrss;
  std::ofstream mf(path + "/metrics.json");
  mf << metadata.dump(2) << '\n';
}
} // namespace peel
