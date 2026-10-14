#include "geometry.h"
#include <cmath>
#include <sstream>
namespace peel {
Q rational(const Json &j) {
  if (j.is_number_float()) {
    double d = j.get<double>();
    if (!std::isfinite(d))
      throw std::runtime_error("Nonfinite coordinate");
    return Q(d);
  }
  std::string s = j.is_string() ? j.get<std::string>() : j.dump();
  // Strings deliberately accept integer or numerator/denominator only.
  auto slash = s.find('/');
  auto integer = [](const std::string &v) {
    if (v.empty())
      return false;
    size_t i = (v[0] == '-' || v[0] == '+');
    if (i == v.size())
      return false;
    for (; i < v.size(); ++i)
      if (v[i] < '0' || v[i] > '9')
        return false;
    return true;
  };
  if (!integer(s.substr(0, slash)) || (slash != std::string::npos && !integer(s.substr(slash + 1))))
    throw std::runtime_error("Expected integer or rational string: " + s);
  Q n;
  std::istringstream ns(s.substr(0, slash));
  ns >> n;
  if (slash == std::string::npos)
    return n;
  Q d;
  std::istringstream ds(s.substr(slash + 1));
  ds >> d;
  if (d == 0)
    throw std::runtime_error("Zero rational denominator");
  return n / d;
}
std::string str(const Q &q) {
  std::ostringstream o;
  o << q;
  auto s = o.str();
  return s.find('/') == std::string::npos ? s + "/1" : s;
}
Json jp(const P &p) { return Json::array({str(p.x()), str(p.y())}); }
Json jp3(const P3 &p) { return Json::array({str(p.x()), str(p.y()), str(p.z())}); }
Poly polygon(std::vector<P> p) {
  if (p.size() > 1 && p.front() == p.back())
    p.pop_back();
  bool changed = true;
  while (changed && p.size() >= 3) {
    changed = false;
    for (size_t i = 0; i < p.size(); ++i) {
      auto a = p[(i + p.size() - 1) % p.size()], b = p[i], c = p[(i + 1) % p.size()];
      if (a == b || b == c || CGAL::collinear(a, b, c)) {
        p.erase(p.begin() + i);
        changed = true;
        break;
      }
    }
  }
  Poly out(p.begin(), p.end());
  if (out.size() < 3 || out.area() == 0)
    return Poly();
  if (!out.is_simple())
    throw std::runtime_error("Non-simple polygon boundary");
  if (out.is_clockwise_oriented())
    out.reverse_orientation();
  return out;
}
Set domain(const Poly &p) { return p.size() < 3 ? Set() : Set(p); }
// Polygon-set boundaries can touch themselves at a vertex. Split their
// weakly simple walks into signed simple cycles before reconstructing a set.
Set regularized_cycle(const std::vector<P> &ring) {
  std::vector<std::vector<P>> pending{ring};
  Set positive, negative;
  Q signed_area = 0;
  while (!pending.empty()) {
    auto v = std::move(pending.back());
    pending.pop_back();
    if (v.size() > 1 && v.front() == v.back())
      v.pop_back();
    bool split = false;
    std::map<P, size_t> seen;
    for (size_t i = 0; i < v.size(); ++i) {
      auto found = seen.find(v[i]);
      if (found != seen.end()) {
        size_t j = found->second;
        pending.emplace_back(v.begin() + j, v.begin() + i);
        std::vector<P> rest(v.begin(), v.begin() + j);
        rest.insert(rest.end(), v.begin() + i, v.end());
        pending.push_back(std::move(rest));
        split = true;
        break;
      }
      seen[v[i]] = i;
    }
    if (split || v.size() < 3)
      continue;
    Poly raw(v.begin(), v.end());
    Q a = raw.area();
    signed_area += a;
    Poly clean = polygon(v);
    if (clean.is_empty())
      continue;
    if (a > 0)
      positive.join(clean);
    else
      negative.join(clean);
  }
  if (signed_area < 0) {
    negative.difference(positive);
    return negative;
  }
  positive.difference(negative);
  return positive;
}
std::vector<Pwh> pieces(const Set &s) {
  std::vector<Pwh> r;
  s.polygons_with_holes(std::back_inserter(r));
  return r;
}
long vertices(const Set &s) {
  long n = 0;
  for (auto &p : pieces(s)) {
    n += p.outer_boundary().size();
    for (auto h = p.holes_begin(); h != p.holes_end(); ++h)
      n += h->size();
  }
  return n;
}
long vertices(const Domains &d) {
  long n = 0;
  for (auto &kv : d)
    n += vertices(kv.second);
  return n;
}
static void count(const Set &s, Stats *c) {
  if (c) {
    auto n = vertices(s);
    c->result_boundary_vertices += n;
    c->result_boundary_edges += n;
  }
}
Set intersect(const Set &a, const Set &b, Stats *c) {
  if (c)
    ++c->intersections;
  Set r = a;
  r.intersection(b);
  count(r, c);
  return r;
}
Set subtract(const Set &a, const Set &b, Stats *c) {
  if (c)
    ++c->differences;
  Set r = a;
  r.difference(b);
  count(r, c);
  return r;
}
void unite(Set &a, const Set &b, Stats *c) {
  if (c)
    ++c->joins;
  a.join(b);
  count(a, c);
}
bool equal(const Set &a, const Set &b) {
  Set r = a;
  r.symmetric_difference(b);
  return r.is_empty();
}
bool contains(const Set &a, const P &p) { return a.oriented_side(p) != CGAL::ON_NEGATIVE_SIDE; }
Q area(const Set &s) {
  Q a = 0;
  for (auto &p : pieces(s)) {
    a += p.outer_boundary().area();
    for (auto h = p.holes_begin(); h != p.holes_end(); ++h)
      a += h->area();
  }
  return a;
}
Json complexity(const Domains &d) {
  long comps = 0, holes = 0, src = 0;
  Q a = 0;
  for (auto &kv : d) {
    if (kv.second.is_empty())
      continue;
    ++src;
    a += area(kv.second);
    for (auto &p : pieces(kv.second)) {
      ++comps;
      holes += p.number_of_holes();
    }
  }
  return {{"sources", src},
          {"components", comps},
          {"holes", holes},
          {"boundary_vertices", vertices(d)},
          {"area", str(a)}};
}
Box bounds(const Set &s) {
  bool first = true;
  Box b;
  for (auto &poly : pieces(s))
    for (auto p : poly.outer_boundary()) {
      if (first) {
        b = {p.x(), p.y(), p.x(), p.y()};
        first = false;
      } else {
        b.xmin = std::min(b.xmin, p.x());
        b.xmax = std::max(b.xmax, p.x());
        b.ymin = std::min(b.ymin, p.y());
        b.ymax = std::max(b.ymax, p.y());
      }
    }
  if (first)
    throw std::runtime_error("Bounds of empty domain");
  return b;
}
bool overlaps(const Box &a, const Box &b) {
  return a.xmin < b.xmax && b.xmin < a.xmax && a.ymin < b.ymax && b.ymin < a.ymax;
}
Score difference(const Score &a, const Score &b) { return {a.a - b.a, a.b - b.b, a.c - b.c}; }
Set halfplane(const Score &s, const Box &w) {
  auto cs = w.corners();
  std::vector<P> in(cs.begin(), cs.end()), out;
  for (size_t i = 0; i < in.size(); ++i) {
    P a = in[i], b = in[(i + 1) % in.size()];
    Q da = s.at(a), db = s.at(b);
    if (da <= 0)
      out.push_back(a);
    if ((da < 0 && db > 0) || (da > 0 && db < 0)) {
      Q t = da / (da - db);
      out.emplace_back(a.x() + t * (b.x() - a.x()), a.y() + t * (b.y() - a.y()));
    }
  }
  return domain(polygon(out));
}
Json set_json(const Set &s) {
  Json out = Json::array();
  for (auto &p : pieces(s)) {
    Json o = Json::array(), hs = Json::array();
    for (auto v : p.outer_boundary())
      o.push_back(jp(v));
    for (auto h = p.holes_begin(); h != p.holes_end(); ++h) {
      Json a = Json::array();
      for (auto v : *h)
        a.push_back(jp(v));
      hs.push_back(a);
    }
    out.push_back({{"outer", o}, {"holes", hs}});
  }
  return out;
}
Set read_set(const Json &j) {
  Set s;
  for (auto &p : j) {
    auto ring = [](const Json &a) {
      std::vector<P> v;
      for (auto &x : a)
        v.emplace_back(rational(x[0]), rational(x[1]));
      return regularized_cycle(v);
    };
    Set part = ring(p.at("outer"));
    for (auto &h : p.at("holes"))
      part.difference(ring(h));
    s.join(part);
  }
  return s;
}
const Source &Scene::source(int id) const {
  auto it = source_lookup.find(id);
  if (it == source_lookup.end())
    throw std::runtime_error("Unknown source ID");
  return sources.at(it->second);
}
Json Stats::json() const {
  return {{"index_box_tests", index_box_tests},
          {"index_rebuilds", index_rebuilds},
          {"adapter_seconds", adapter_seconds},
          {"envelope_construction_seconds", envelope_construction_seconds},
          {"normalization_seconds", normalization_seconds},
          {"raw_regions", raw_regions},
          {"raw_boundary_vertices", raw_boundary_vertices},
          {"candidate_tests", candidate_tests},
          {"box_rejections", box_rejections},
          {"depth_separated", depth_separated},
          {"complete_occlusion_rejections", occlusion_rejections},
          {"certificate_tests", certificate_tests},
          {"polygon_intersections", intersections},
          {"polygon_differences", differences},
          {"polygon_unions", joins},
          {"depth_halfplane_clips", halfplane_clips},
          {"adapter_triangles", adapter_triangles},
          {"materialized_result_boundary_vertices", result_boundary_vertices},
          {"materialized_result_boundary_edges", result_boundary_edges},
          {"max_visible_map_boundary_vertices", max_map_vertices},
          {"peak_live_boundary_vertices_lower_bound", peak_live_boundary_vertices},
          {"arrangement_vertices", arrangement_vertices},
          {"arrangement_edges", arrangement_edges},
          {"arrangement_faces", arrangement_faces},
          {"raw_triangles", raw_triangles},
          {"visibility_seconds", visibility_seconds},
          {"residual_seconds", residual_seconds},
          {"triangulation_seconds", triangulation_seconds},
          {"validation_seconds", validation_seconds},
          {"cgal_internal_constructions", nullptr},
          {"cgal_internal_peak_geometry", nullptr}};
}
} // namespace peel
