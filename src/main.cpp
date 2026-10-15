#include "fast_peel.h"
#include "geometry.h"
#include <CGAL/version.h>
#include <fstream>
#include <iostream>
#include <sys/utsname.h>
namespace {
void help() {
  std::cout
      << "geometric_peel --scene FILE --method "
         "global-arrangement|fast-bvh-peel|vbvh-peel|bvh-peel|visible-peel|envelope-peel|"
         "arrangement\n  --layers "
         "all|N --order input|random|near-first|reverse --seed N\n  --out DIR "
         "[--verify|--verify-global|--validate] "
         "[--no-accelerate] [--oracle-cap N] [--threads N] [--repeat N]\n"
         "  [--metrics-only|--mesh-only] (BVH/global methods) [--fast-grid-bits 12..30]\n"
         "  [--no-rank-pruning] [--merge-residuals] (bvh-peel/fast-bvh-peel controls)\n"
         "  [--global-all-edges] (global-arrangement control: disable contour reduction)\n"
         "  [--resolved-mesh] (bvh-peel/fast-bvh-peel: all intersections must already be split "
         "into edges)\n  "
         "--compare "
         "EXACT_JSON "
         "EXACT_JSON\nVerification runs "
         "exact geometry checks and an independent arrangement oracle.\n";
}
} // namespace
int main(int argc, char **argv) {
  using namespace peel;
  try {
    std::string scene, method = "visible-peel", order = "near-first", out = "out/peel";
    int layers = 0, cap = 64, threads = 1, repeat = 1, fast_bits = 30;
    PeelOptions peel_options;
    bool metrics_only = false, mesh_only = false, resolved_mesh = false;
    unsigned seed = 1729;
    bool verify = false, verify_global = false, global_all_edges = false;
    bool validate_only = false, accelerate = true;
    for (int i = 1; i < argc; ++i) {
      std::string a = argv[i];
      auto value = [&]() {
        if (i + 1 == argc)
          throw std::runtime_error("Missing value for " + a);
        return std::string(argv[++i]);
      };
      if (a == "--help") {
        help();
        return 0;
      } else if (a == "--compare") {
        auto x = value(), y = value();
        compare_files(x, y);
        std::cout << "Exact labelled regions agree.\n";
        return 0;
      } else if (a == "--scene")
        scene = value();
      else if (a == "--method")
        method = value();
      else if (a == "--order")
        order = value();
      else if (a == "--out")
        out = value();
      else if (a == "--seed")
        seed = std::stoul(value());
      else if (a == "--oracle-cap")
        cap = std::stoi(value());
      else if (a == "--threads")
        threads = std::stoi(value());
      else if (a == "--repeat")
        repeat = std::stoi(value());
      else if (a == "--fast-grid-bits")
        fast_bits = std::stoi(value());
      else if (a == "--no-rank-pruning")
        peel_options.prune_rank_subtrees = false;
      else if (a == "--merge-residuals")
        peel_options.merge_residuals = true;
      else if (a == "--resolved-mesh")
        resolved_mesh = true;
      else if (a == "--metrics-only")
        metrics_only = true;
      else if (a == "--mesh-only")
        mesh_only = true;
      else if (a == "--layers") {
        auto v = value();
        layers = v == "all" ? 0 : std::stoi(v);
        if (v != "all" && layers <= 0)
          throw std::runtime_error("Layer count must be positive or all");
      } else if (a == "--global-all-edges")
        global_all_edges = true;
      else if (a == "--verify-global")
        verify_global = true;
      else if (a == "--verify")
        verify = true;
      else if (a == "--validate")
        validate_only = true;
      else if (a == "--no-accelerate")
        accelerate = false;
      else
        throw std::runtime_error("Unknown option: " + a);
    }
    if (scene.empty()) {
      help();
      return 2;
    }
    if (repeat < 1 || threads < 1)
      throw std::runtime_error("repeat and threads must be positive");
    if (metrics_only && mesh_only)
      throw std::runtime_error("Choose only one compact output mode");
    if ((global_all_edges && method != "global-arrangement") ||
        (verify_global && (method == "fast-bvh-peel" || method == "global-arrangement")))
      throw std::runtime_error("Global controls require the corresponding exact solver");
    if ((!peel_options.prune_rank_subtrees || peel_options.merge_residuals) &&
        method != "bvh-peel" && method != "fast-bvh-peel")
      throw std::runtime_error("Peeling controls require bvh-peel or fast-bvh-peel");
    if (method == "fast-bvh-peel")
      return fast::run_cli(scene, out, layers, threads, repeat, fast_bits, order, seed, accelerate,
                           metrics_only, verify, validate_only, cap, resolved_mesh, mesh_only,
                           peel_options);
    if (resolved_mesh && method != "bvh-peel" && method != "global-arrangement")
      throw std::runtime_error(
          "--resolved-mesh requires a BVH peeling or global-arrangement method");
    if ((metrics_only || mesh_only) && method != "bvh-peel" && method != "vbvh-peel" &&
        method != "global-arrangement")
      throw std::runtime_error("Compact output requires a BVH or global-arrangement method");
    auto t = Clock::now();
    auto s = read_scene(
        scene, method == "bvh-peel" || method == "vbvh-peel" || method == "global-arrangement",
        resolved_mesh || method == "global-arrangement" || verify_global);
    s.topology["resolved_mesh_contract"] = resolved_mesh;
    double setup = seconds(t);
    Result r;
    std::vector<double> timings;
    for (int i = 0; i < repeat; ++i) {
      r = Result();
      t = Clock::now();
      r = method == "global-arrangement"
              ? global_arrangement(s, layers, threads, accelerate, !global_all_edges)
          : (method == "bvh-peel" || method == "vbvh-peel")
              ? bvh_peel(s, layers, order, seed, accelerate, threads, method == "vbvh-peel",
                         peel_options)
              : run(s, method, layers, order, seed, accelerate, cap);
      timings.push_back(seconds(t));
    }
    auto sorted_timings = timings;
    std::sort(sorted_timings.begin(), sorted_timings.end());
    double algorithm = sorted_timings[sorted_timings.size() / 2], validation = 0;
    Json audit;
    if (verify || verify_global || validate_only) {
      t = Clock::now();
      if (r.convex_representation && !verify && !verify_global)
        audit = audit_convex(s, r, layers == 0);
      else {
        materialize_scene(s);
        materialize_regions(r);
        validate(s, r, layers == 0);
      }
      if (verify && method != "arrangement")
        compare(r, arrangement(s, layers, cap));
      if (verify_global) {
        auto reference = global_arrangement(s, layers, threads);
        materialize_regions(reference);
        compare(r, reference);
      }
      validation = seconds(t);
    }
    struct utsname sys {};
    uname(&sys);
    Json meta = {{"method", method},
                 {"order", order},
                 {"mesh_topology", s.topology},
                 {"rank_classification", r.classification},
                 {"global_arrangement", r.global_statistics},
                 {"seed", seed},
                 {"requested_layers", layers ? Json(layers) : Json("all")},
                 {"accelerated", accelerate},
                 {"camera", s.camera.perspective ? "perspective" : "orthographic"},
                 {"exact_number_type",
#ifdef PEEL_LAZY_EXACT
                  "CGAL::Lazy_exact_nt<CGAL::Exact_rational>"
#else
                  "CGAL::Exact_rational (GMP rational)"
#endif
                 },
                 {"kernel", "CGAL::Cartesian (exact constructions)"},
                 {"cgal_version", CGAL_VERSION_STR},
                 {"cgal_upstream_release", std::string("v") + CGAL_VERSION_STR},
                 {"dependency_lock", "dependencies.json"},
                 {"compiler", __VERSION__},
                 {"build_type",
#ifdef NDEBUG
                  "Release"
#else
                  "Debug"
#endif
                 },
                 {"platform", std::string(sys.sysname) + " " + sys.release + " " + sys.machine},
                 {"projection_preflight_seconds", setup},
                 {"algorithm_seconds", algorithm},
                 {"algorithm_repetitions_seconds", timings},
                 {"threads", threads},
                 {"validation_seconds", validation},
                 {"validated_against_oracle", verify},
                 {"validated_against_global", verify_global},
                 {"exact_invariants_validated", verify || verify_global || validate_only},
                 {"convex_audit", audit},
                 {"scene", scene},
                 {"oracle_cap", cap}};
    if (metrics_only || mesh_only)
      write_convex_result(s, r, out, meta, mesh_only);
    else {
      t = Clock::now();
      materialize_scene(s);
      materialize_regions(r);
      meta["region_materialization_seconds"] = seconds(t);
      write_result(s, r, out, meta);
    }
    std::cout << Json({{"layers", r.layers.size()},
                       {"algorithm_seconds", algorithm},
                       {"validation_seconds", validation},
                       {"output", out}})
                     .dump()
              << '\n';
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << '\n';
    return 1;
  }
}
