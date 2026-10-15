#include "fast_peel.h"
#include "geometry.h"
#include <algorithm>
#include <cmath>

namespace peel::fast {
Json validate_reference(const std::string &path, const Scene &fs, const Result &fr, bool oracle,
                        int cap, bool complete) {
  auto s = peel::read_scene(path, true);
  const Q grid = Q(int64_t(1) << fs.bits);
  auto point = [&](double x, double y) {
    return P(Q(fs.window[0]) + Q(x) / grid * Q(fs.window[2] - fs.window[0]),
             Q(fs.window[1]) + Q(y) / grid * Q(fs.window[3] - fs.window[1]));
  };
  double cell = std::max(fs.window[2] - fs.window[0], fs.window[3] - fs.window[1]) /
                double(int64_t(1) << fs.bits);
  struct Bounds {
    double xmin, ymin, xmax, ymax;
  };
  std::vector<Bounds> boxes;
  for (auto &src : s.sources) {
    auto b = src.initial_polygon.bbox();
    boxes.push_back({b.xmin(), b.ymin(), b.xmax(), b.ymax()});
  }
  size_t checked = 0, mismatches = 0, near_boundary = 0;
  Json examples = Json::array();
  for (size_t k = 0; k < fr.layers.size(); ++k) {
    auto &faces = fr.layers[k].faces;
    size_t count = std::min<size_t>(128, faces.size());
    for (size_t j = 0; j < count; ++j) {
      auto &face = faces[j * faces.size() / count];
      double x = 0, y = 0;
      for (auto v : face.points) {
        x += v[0] / 3;
        y += v[1] / 3;
      }
      P p = point(x, y);
      double u = CGAL::to_double(p.x()), v = CGAL::to_double(p.y());
      P3 origin = s.camera.perspective ? P3(0, 0, 0) : P3(p.x(), p.y(), 0);
      K::Vector_3 direction =
          s.camera.perspective ? K::Vector_3(p.x(), p.y(), 1) : K::Vector_3(0, 0, 1);
      std::vector<std::pair<Q, int>> hits;
      bool boundary = false;
      for (size_t i = 0; i < s.sources.size(); ++i) {
        auto &b = boxes[i];
        if (u < b.xmin - 2 * cell || u > b.xmax + 2 * cell || v < b.ymin - 2 * cell ||
            v > b.ymax + 2 * cell)
          continue;
        auto &src = s.sources[i];
        for (auto e = src.initial_polygon.edges_begin(); e != src.initial_polygon.edges_end();
             ++e) {
          double ax = CGAL::to_double(e->source().x()), ay = CGAL::to_double(e->source().y()),
                 dx = CGAL::to_double(e->target().x()) - ax,
                 dy = CGAL::to_double(e->target().y()) - ay;
          if (std::abs(dx * (v - ay) - dy * (u - ax)) <= 2 * cell * std::hypot(dx, dy))
            boundary = true;
        }
        auto e1 = src.cam[1] - src.cam[0], e2 = src.cam[2] - src.cam[0],
             h = CGAL::cross_product(direction, e2);
        Q det = e1 * h;
        if (det == 0)
          continue;
        auto offset = origin - src.cam[0];
        Q a = (offset * h) / det;
        if (a < 0 || a > 1)
          continue;
        auto q = CGAL::cross_product(offset, e1);
        Q bcoord = (direction * q) / det;
        if (bcoord < 0 || a + bcoord > 1)
          continue;
        Q depth = (e2 * q) / det;
        if (s.camera.perspective && depth < s.camera.near)
          continue;
        hits.emplace_back(depth, src.id);
      }
      if (boundary) {
        ++near_boundary;
        continue;
      }
      std::sort(hits.begin(), hits.end());
      ++checked;
      if (hits.size() <= k || hits[k].second != fs.sources[face.source].id) {
        ++mismatches;
        if (examples.size() < 8)
          examples.push_back(
              {{"layer", k + 1},
               {"uv", {u, v}},
               {"actual_source", fs.sources[face.source].id},
               {"expected_source", hits.size() > k ? Json(hits[k].second) : Json()}});
      }
    }
  }
  std::map<int, double> actual_area;
  const double grid_area_scale = (fs.window[2] - fs.window[0]) * (fs.window[3] - fs.window[1]) /
                                 std::pow(double(int64_t(1) << fs.bits), 2);
  auto collect = [&](const std::vector<Patch> &patches) {
    for (auto &p : patches) {
      double area2 = 0;
      for (size_t j = 1; j + 1 < p.points.size(); ++j)
        area2 +=
            double(int64_t(p.points[j].x - p.points[0].x) * (p.points[j + 1].y - p.points[0].y) -
                   int64_t(p.points[j].y - p.points[0].y) * (p.points[j + 1].x - p.points[0].x));
      actual_area[fs.sources[p.source].id] += 0.5 * area2 * grid_area_scale;
    }
  };
  for (auto &layer : fr.layers)
    collect(layer.patches);
  collect(fr.residual);
  double area_error = 0, reference_area = 0;
  for (auto &src : s.sources) {
    double a = CGAL::to_double(src.initial_polygon.area());
    reference_area += a;
    area_error += std::abs(actual_area[src.id] - a);
  }
  Json report = {
      {"exact_reference", true},
      {"summed_patch_area_per_source_l1_error", area_error},
      {"summed_reference_source_area", reference_area},
      {"relative_source_area_l1_error", reference_area ? area_error / reference_area : 0},
      {"source_area_check_is_region_certificate", false},
      {"full_exact_geometry_certificate", false},
      {"independent_exact_ray_ranks_checked", checked},
      {"rank_mismatches_away_from_boundaries", mismatches},
      {"boundary_samples_skipped", near_boundary},
      {"boundary_exclusion_cells", 2},
      {"mismatch_examples", examples},
      {"remaining_patches", fr.residual.size()}};
  bool passed = mismatches == 0 && (!complete || fr.residual.empty());
  if (oracle) {
    if (s.sources.size() > size_t(cap))
      throw std::runtime_error("Fast reference verification exceeds oracle cap");
    materialize_scene(s);
    auto reference = arrangement(s, complete ? 0 : int(fr.layers.size()), cap);
    double error = 0, perimeter = 0;
    size_t count = std::max(reference.layers.size(), fr.layers.size());
    for (size_t k = 0; k < count; ++k) {
      Domains actual;
      if (k < fr.layers.size())
        for (auto &patch : fr.layers[k].patches) {
          std::vector<P> vertices;
          for (auto v : patch.points)
            vertices.push_back(point(v.x, v.y));
          unite(actual[fs.sources[patch.source].id], domain(polygon(vertices)));
        }
      Domains expected;
      if (k < reference.layers.size())
        expected = reference.layers[k].regions;
      std::set<int> ids;
      for (auto &p : actual)
        ids.insert(p.first);
      for (auto &p : expected)
        ids.insert(p.first);
      for (int id : ids) {
        Set diff = actual[id];
        diff.symmetric_difference(expected[id]);
        error += CGAL::to_double(area(diff));
        for (auto *set : {&actual[id], &expected[id]})
          for (auto &poly : pieces(*set)) {
            auto measure = [&](const Poly &ring) {
              for (auto e = ring.edges_begin(); e != ring.edges_end(); ++e)
                perimeter += std::sqrt(CGAL::to_double(e->squared_length()));
            };
            measure(poly.outer_boundary());
            for (auto h = poly.holes_begin(); h != poly.holes_end(); ++h)
              measure(*h);
          }
      }
    }
    // An explicit acceptance budget for tests, not a proven global error bound.
    double budget = 8 * cell * perimeter +
                    1e-12 * (fs.window[2] - fs.window[0]) * (fs.window[3] - fs.window[1]);
    report["oracle_layers"] = reference.layers.size();
    report["fast_layers"] = fr.layers.size();
    report["labelled_region_symmetric_difference_area"] = error;
    report["area_acceptance_budget"] = budget;
    report["area_budget_is_proven_bound"] = false;
    passed &= error <= budget;
  }
  report["status"] = passed ? "passed" : "numerical_mismatch";
  return report;
}
} // namespace peel::fast
