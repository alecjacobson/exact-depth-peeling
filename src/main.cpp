#include "geometry.h"
#include <CGAL/version.h>
#include <fstream>
#include <iostream>
#include <sys/utsname.h>
namespace {
void help() {
  std::cout
      << "geometric_peel --scene FILE --method visible-peel|envelope-peel|arrangement\n  --layers "
         "all|N --order input|random|near-first|reverse --seed N\n  --out DIR "
         "[--verify|--validate] "
         "[--no-accelerate] [--oracle-cap N]\n  --compare EXACT_JSON EXACT_JSON\nVerification runs "
         "exact geometry checks and an independent arrangement oracle.\n";
}
} // namespace
int main(int argc, char **argv) {
  using namespace peel;
  try {
    std::string scene, method = "visible-peel", order = "near-first", out = "out/peel";
    int layers = 0, cap = 64;
    unsigned seed = 1729;
    bool verify = false, validate_only = false, accelerate = true;
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
      else if (a == "--layers") {
        auto v = value();
        layers = v == "all" ? 0 : std::stoi(v);
        if (v != "all" && layers <= 0)
          throw std::runtime_error("Layer count must be positive or all");
      } else if (a == "--verify")
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
    auto t = Clock::now();
    auto s = read_scene(scene);
    double setup = seconds(t);
    t = Clock::now();
    auto r = run(s, method, layers, order, seed, accelerate, cap);
    double algorithm = seconds(t), validation = 0;
    if (verify || validate_only) {
      t = Clock::now();
      validate(s, r, layers == 0);
      if (verify && method != "arrangement")
        compare(r, arrangement(s, layers, cap));
      validation = seconds(t);
    }
    struct utsname sys {};
    uname(&sys);
    Json meta = {{"method", method},
                 {"order", order},
                 {"seed", seed},
                 {"requested_layers", layers ? Json(layers) : Json("all")},
                 {"accelerated", accelerate},
                 {"camera", s.camera.perspective ? "perspective" : "orthographic"},
                 {"exact_number_type", "CGAL::Exact_rational (GMP rational)"},
                 {"kernel", "CGAL::Cartesian"},
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
                 {"validation_seconds", validation},
                 {"validated_against_oracle", verify},
                 {"exact_invariants_validated", verify || validate_only},
                 {"scene", scene},
                 {"oracle_cap", cap}};
    write_result(s, r, out, meta);
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
