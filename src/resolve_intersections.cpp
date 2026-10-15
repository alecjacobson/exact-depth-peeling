#include "geometry.h"
#include <CGAL/Polygon_mesh_processing/corefinement.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/boost/graph/helpers.h>
#include <fstream>
#include <iostream>

namespace {
using Mesh = CGAL::Surface_mesh<peel::P3>;
using Face = Mesh::Face_index;
struct Visitor : CGAL::Polygon_mesh_processing::Corefinement::Default_visitor<Mesh> {
  Mesh::Property_map<Face, int> labels;
  int parent = -1;
  explicit Visitor(Mesh::Property_map<Face, int> p) : labels(p) {}
  void before_subface_creations(Face f, Mesh &) { parent = labels[f]; }
  void after_subface_created(Face f, Mesh &) { labels[f] = parent; }
};
} // namespace
int main(int argc, char **argv) {
  using namespace peel;
  try {
    if (argc != 3) {
      std::cerr
          << "resolve_intersections INPUT.json OUTPUT.json\nExact camera-independent intersection "
             "splitting; input must be an orientable manifold with optional boundary.\n";
      return 2;
    }
    auto start = Clock::now();
    std::ifstream f(argv[1]);
    if (!f)
      throw std::runtime_error("Cannot open input");
    Json input;
    f >> input;
    Mesh mesh;
    std::map<P3, Mesh::Vertex_index> vertices;
    auto labels = mesh.add_property_map<Mesh::Face_index, int>("f:source", -1).first;
    auto constrained = mesh.add_property_map<Mesh::Edge_index, bool>("e:intersection", false).first;
    auto triangles = input.at("triangles");
    for (size_t i = 0; i < triangles.size(); ++i) {
      auto t = triangles[i].at("vertices");
      if (t.size() != 3)
        throw std::runtime_error("Expected triangle");
      std::vector<Mesh::Vertex_index> v;
      for (auto q : t) {
        if (q.size() != 3)
          throw std::runtime_error("Expected 3D vertex");
        P3 p(rational(q[0]), rational(q[1]), rational(q[2]));
        auto it = vertices.find(p);
        if (it == vertices.end())
          it = vertices.emplace(p, mesh.add_vertex(p)).first;
        v.push_back(it->second);
      }
      auto face = mesh.add_face(v);
      if (face == Mesh::null_face())
        throw std::runtime_error(
            "Preprocessor input has non-manifold or inconsistently oriented connectivity; provide "
            "resolved pieces directly for --resolved-mesh");
      labels[face] = i;
    }
    double import = seconds(start);
    start = Clock::now();
    Visitor visitor(labels);
    CGAL::Polygon_mesh_processing::experimental::autorefine(
        mesh, CGAL::parameters::visitor(visitor).edge_is_constrained_map(constrained));
    double refine = seconds(start);
    Json output = input;
    output["triangles"] = Json::array();
    if (!output.contains("original_triangles"))
      output["original_triangles"] = triangles;
    int id = 0;
    for (auto face : mesh.faces()) {
      if (labels[face] < 0 || size_t(labels[face]) >= triangles.size())
        throw std::runtime_error("Lost preprocessing provenance");
      auto &original = triangles[labels[face]];
      int source = original.value("original_source_id", original.value("id", labels[face]));
      Json points = Json::array();
      for (auto v : CGAL::vertices_around_face(mesh.halfedge(face), mesh))
        points.push_back(jp3(mesh.point(v)));
      if (points.size() != 3)
        throw std::runtime_error("Autorefinement produced nontriangular face");
      output["triangles"].push_back({{"id", id++},
                                     {"original_source_id", source},
                                     {"mesh_face_id", original.value("mesh_face_id", source)},
                                     {"vertices", points}});
    }
    size_t intersection_edges = 0;
    for (auto e : mesh.edges())
      intersection_edges += constrained[e];
    output["intersection_preprocessing"] = {{"method", "CGAL PMP experimental::autorefine"},
                                            {"arithmetic", "exact rational constructions"},
                                            {"input_triangles", triangles.size()},
                                            {"output_triangles", id},
                                            {"marked_intersection_edges", intersection_edges},
                                            {"import_seconds", import},
                                            {"refinement_seconds", refine},
                                            {"positive_area_coincident_surfaces_supported", false}};
    std::ofstream out(argv[2]);
    if (!out)
      throw std::runtime_error("Cannot open output");
    out << output.dump() << '\n';
    std::cout << output["intersection_preprocessing"].dump() << '\n';
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << '\n';
    return 1;
  }
}
