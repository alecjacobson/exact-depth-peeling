#include "geometry.h"
#include <fstream>
#include <set>
namespace peel {
static P3 transform(const std::array<std::array<Q, 4>, 3> &m, const P3 &p) {
  std::array<Q, 3> v;
  for (int i = 0; i < 3; ++i)
    v[i] = m[i][0] * p.x() + m[i][1] * p.y() + m[i][2] * p.z() + m[i][3];
  return {v[0], v[1], v[2]};
}
static Camera camera(const Json &j) {
  Camera c;
  auto type = j.value("type", std::string("orthographic"));
  if (type != "orthographic" && type != "perspective")
    throw std::runtime_error("Unknown camera type");
  c.perspective = type == "perspective";
  if (j.contains("near"))
    c.near = rational(j["near"]);
  if (c.near <= 0)
    throw std::runtime_error("Near plane must be strictly positive");
  for (int i = 0; i < 3; ++i)
    c.world_to_camera[i][i] = 1;
  if (j.contains("world_to_camera")) {
    auto a = j["world_to_camera"];
    if (a.size() != 3)
      throw std::runtime_error("world_to_camera must be 3x4 affine matrix");
    for (int i = 0; i < 3; ++i) {
      if (a[i].size() != 4)
        throw std::runtime_error("world_to_camera must be 3x4");
      for (int k = 0; k < 4; ++k)
        c.world_to_camera[i][k] = rational(a[i][k]);
    }
  }
  std::array<std::array<Q, 6>, 3> m{};
  for (int i = 0; i < 3; ++i) {
    for (int k = 0; k < 3; ++k)
      m[i][k] = c.world_to_camera[i][k];
    m[i][i + 3] = 1;
  }
  for (int k = 0; k < 3; ++k) {
    int pivot = k;
    while (pivot < 3 && m[pivot][k] == 0)
      ++pivot;
    if (pivot == 3)
      throw std::runtime_error("Singular camera transform");
    std::swap(m[k], m[pivot]);
    Q d = m[k][k];
    for (auto &x : m[k])
      x /= d;
    for (int i = 0; i < 3; ++i)
      if (i != k) {
        Q v = m[i][k];
        for (int t = 0; t < 6; ++t)
          m[i][t] -= v * m[k][t];
      }
  }
  for (int i = 0; i < 3; ++i) {
    for (int k = 0; k < 3; ++k) {
      c.camera_to_world[i][k] = m[i][k + 3];
      c.camera_to_world[i][3] -= m[i][k + 3] * c.world_to_camera[k][3];
    }
  }
  return c;
}
static Score fit(const std::array<P, 3> &p, const std::array<Q, 3> &z) {
  Q x1 = p[1].x() - p[0].x(), y1 = p[1].y() - p[0].y(), x2 = p[2].x() - p[0].x(),
    y2 = p[2].y() - p[0].y(), det = x1 * y2 - x2 * y1;
  Score s{((z[1] - z[0]) * y2 - (z[2] - z[0]) * y1) / det,
          (x1 * (z[2] - z[0]) - x2 * (z[1] - z[0])) / det, 0};
  s.c = z[0] - s.a * p[0].x() - s.b * p[0].y();
  return s;
}
Scene read_scene(const std::string &path, bool convex_only) {
  std::ifstream f(path);
  if (!f)
    throw std::runtime_error("Cannot open scene: " + path);
  Json j;
  f >> j;
  Scene s;
  s.camera = camera(j.value("camera", Json::object()));
  auto w = j.at("window");
  if (w.size() != 4)
    throw std::runtime_error("Window must be [xmin,ymin,xmax,ymax]");
  s.window = {rational(w[0]), rational(w[1]), rational(w[2]), rational(w[3])};
  if (s.window.xmin >= s.window.xmax || s.window.ymin >= s.window.ymax)
    throw std::runtime_error("Invalid window");
  auto wc = s.window.corners();
  Set win = domain(polygon({wc.begin(), wc.end()}));
  std::set<int> ids;
  for (auto &t : j.at("triangles")) {
    Source src;
    src.id = t.value("id", s.input_sources);
    src.mesh_face = t.value("mesh_face_id", src.id);
    ++s.input_sources;
    if (!ids.insert(src.id).second)
      throw std::runtime_error("Duplicate source ID");
    if (t.at("vertices").size() != 3)
      throw std::runtime_error("Expected three vertices");
    for (int i = 0; i < 3; ++i) {
      auto v = t["vertices"][i];
      if (v.size() != 3)
        throw std::runtime_error("Expected 3D vertex");
      src.world[i] = P3(rational(v[0]), rational(v[1]), rational(v[2]));
      src.cam[i] = transform(s.camera.world_to_camera, src.world[i]);
    }
    std::vector<P3> clipped(src.cam.begin(), src.cam.end());
    if (s.camera.perspective) {
      clipped.clear();
      for (int i = 0; i < 3; ++i) {
        auto a = src.cam[i], b = src.cam[(i + 1) % 3];
        Q da = a.z() - s.camera.near, db = b.z() - s.camera.near;
        if (da >= 0)
          clipped.push_back(a);
        if ((da < 0 && db > 0) || (da > 0 && db < 0)) {
          Q t = da / (da - db);
          clipped.push_back(a + (b - a) * t);
        }
      }
    }
    if (clipped.size() < 3) {
      ++s.clipped_empty;
      continue;
    }
    std::vector<P> projected;
    std::vector<Q> scores;
    for (auto p : clipped) {
      projected.emplace_back(s.camera.perspective ? p.x() / p.z() : p.x(),
                             s.camera.perspective ? p.y() / p.z() : p.y());
      scores.push_back(s.camera.perspective ? -Q(1) / p.z() : p.z());
    }
    size_t k = 2;
    while (k < projected.size() && CGAL::collinear(projected[0], projected[1], projected[k]))
      ++k;
    if (k == projected.size()) {
      ++s.zero_area;
      continue;
    }
    src.score = fit({projected[0], projected[1], projected[k]}, {scores[0], scores[1], scores[k]});
    bool in_window = true;
    for (auto &p : projected)
      in_window &= p.x() >= s.window.xmin && p.x() <= s.window.xmax && p.y() >= s.window.ymin &&
                   p.y() <= s.window.ymax;
    src.initial_polygon = polygon(projected);
    if (!in_window) {
      Set clipped = intersect(domain(src.initial_polygon), win);
      auto components = pieces(clipped);
      src.initial_polygon = components.empty() ? Poly() : components[0].outer_boundary();
    }
    if (src.initial_polygon.is_empty()) {
      ++s.clipped_empty;
      continue;
    }
    if (!convex_only)
      src.initial = domain(src.initial_polygon);
    s.source_lookup[src.id] = s.sources.size();
    s.sources.push_back(src);
  }
  // Group exact affine planes; even fully hidden duplicates must fail.
  std::map<std::array<Q, 3>, std::vector<const Source *>> groups;
  for (auto &src : s.sources)
    groups[{src.score.a, src.score.b, src.score.c}].push_back(&src);
  auto separated = [](const Poly &a, const Poly &b) {
    for (auto e = a.edges_begin(); e != a.edges_end(); ++e) {
      bool positive = false;
      for (auto &p : b)
        if (CGAL::orientation(e->source(), e->target(), p) == CGAL::LEFT_TURN) {
          positive = true;
          break;
        }
      if (!positive)
        return true;
    }
    return false;
  };
  for (auto &group : groups) {
    struct Entry {
      const Source *source;
      Box box;
    };
    std::vector<Entry> entries;
    for (auto src : group.second) {
      auto &p = src->initial_polygon;
      Box b{p[0].x(), p[0].y(), p[0].x(), p[0].y()};
      for (auto &v : p) {
        b.xmin = std::min(b.xmin, v.x());
        b.xmax = std::max(b.xmax, v.x());
        b.ymin = std::min(b.ymin, v.y());
        b.ymax = std::max(b.ymax, v.y());
      }
      entries.push_back({src, b});
    }
    std::sort(entries.begin(), entries.end(),
              [](const Entry &a, const Entry &b) { return a.box.xmin < b.box.xmin; });
    for (size_t i = 0; i < entries.size(); ++i)
      for (size_t j = i + 1; j < entries.size() && entries[j].box.xmin < entries[i].box.xmax; ++j) {
        auto &a = entries[i], &b = entries[j];
        if (overlaps(a.box, b.box) &&
            !separated(a.source->initial_polygon, b.source->initial_polygon) &&
            !separated(b.source->initial_polygon, a.source->initial_polygon))
          throw std::runtime_error("Unsupported positive-area depth tie between sources " +
                                   std::to_string(a.source->id) + " and " +
                                   std::to_string(b.source->id));
      }
  }
  return s;
}
void materialize_scene(Scene &s) {
  for (auto &src : s.sources)
    if (src.initial.is_empty())
      src.initial = domain(src.initial_polygon);
}
P3 lift(const Scene &s, const Source &src, const P &p) {
  Q score = src.score.at(p);
  P3 c;
  if (s.camera.perspective) {
    Q q = -score;
    if (q <= 0)
      throw std::runtime_error("Invalid inverse depth");
    c = P3(p.x() / q, p.y() / q, Q(1) / q);
  } else
    c = P3(p.x(), p.y(), score);
  return transform(s.camera.camera_to_world, c);
}
P project(const Scene &s, const P3 &p) {
  auto c = transform(s.camera.world_to_camera, p);
  return s.camera.perspective ? P(c.x() / c.z(), c.y() / c.z()) : P(c.x(), c.y());
}
std::array<Q, 3> barycentric(const Source &s, const P3 &p) {
  auto u = s.world[1] - s.world[0], v = s.world[2] - s.world[0], w = p - s.world[0];
  Q uu = u * u, uv = u * v, vv = v * v, wu = w * u, wv = w * v, den = uu * vv - uv * uv;
  if (den == 0)
    throw std::runtime_error("Degenerate source plane");
  Q b = (wu * vv - wv * uv) / den, c = (wv * uu - wu * uv) / den;
  return {1 - b - c, b, c};
}
} // namespace peel
