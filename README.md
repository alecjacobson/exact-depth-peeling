# Exact geometric depth peeling

A CPU research prototype implementing the three methods in
[the design](geometric_depth_peeling_design.md): a global arrangement oracle,
CGAL triangle-envelope peeling, and incremental visible-map peeling. All core
geometry uses `CGAL::Cartesian<CGAL::Exact_rational>`. Orthographic and normalized
pinhole cameras, intersecting triangles, holes, disconnected residuals, and
original-source reconstruction are implemented.

## Build and validate

Generated result files larger than 1 MiB are stored with Git LFS. After cloning,
install Git LFS and run `git lfs install` followed by `git lfs pull` to download
the complete recorded artifacts.

On Debian/Ubuntu:

```sh
sudo apt-get install libcgal-dev libgmp-dev libmpfr-dev nlohmann-json3-dev ninja-build
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 2
python scripts/generate_fixtures.py
ctest --test-dir build --output-on-failure
```

The tested dependency versions and external reference revisions are recorded in
[dependencies.json](dependencies.json). CGAL 5.4 is installed in this environment;
the design links to 6.2 documentation. No CGAL internals or external renderer
sources are copied into this repository. Building the envelope templates can take
several minutes; limit build parallelism on machines with little memory.

```sh
build/geometric_peel --scene tests/fixtures/holes.json \
  --method visible-peel --order near-first --layers all --verify --out out/selective
build/geometric_peel --scene tests/fixtures/holes.json \
  --method envelope-peel --layers all --verify --out out/envelope
build/geometric_peel --scene tests/fixtures/holes.json \
  --method arrangement --layers all --verify --out out/oracle
build/geometric_peel --compare out/selective/exact.json out/oracle/exact.json
```

`--verify` checks exact invariants and labelled symmetric differences against the
independent arrangement method. `--validate` performs the invariant checks without
constructing the oracle, useful above the default 64-source oracle cap.
`--oracle-cap N` explicitly changes that cap. `--no-accelerate` exercises the
unaccelerated insertion algorithm. Orders are `input`, `random`, `near-first`, and
`reverse`; the default seed is 1729. No backface culling is performed.

## Input and output

Scenes are JSON triangle soups; connectivity is optional. Coordinates may be
integers, rational strings such as `"1/3"`, or finite JSON floating-point numbers.
Floating-point values are converted from their stored binary64 values into exact
rationals **before** transformations. Rational strings deliberately do not accept
decimal syntax. Example:

```json
{
  "window": ["-2", "-2", "2", "2"],
  "camera": {"type": "orthographic"},
  "triangles": [
    {"id": 7, "mesh_face_id": 19,
     "vertices": [["-1", "-1", "2"], ["1", "-1", "2"], ["0", "1", "2"]]}
  ]
}
```

The camera can instead specify `"type": "perspective"` and a strictly positive
`"near": "1/10"`. An optional `world_to_camera` is an invertible exact 3×4 affine
matrix; the default is identity. Image coordinates are normalized: orthographic
`(x,y)` or perspective `(x/z,y/z)`. Express intrinsics through the input image
coordinate convention. Perspective clipping precedes division, and score is
`-1/z`; world positions and barycentrics are reconstructed on original triangles.

Each output directory contains:

- `exact.json`: authoritative rational source domains, scores, both camera
  transforms, labelled layers, residuals, and face-corner UV/world/barycentric data.
  Coordinates use numerator/denominator strings. Arrangement outputs also retain
  exact verification triangles and their complete depth lists.
- `layer-N.svg`: a floating-point image-plane region preview, including holes.
- `layer-N.obj`: a floating-point lifted mesh with source groups and independent
  face corners, preserving depth discontinuities.
- `metrics.json`: setup, algorithm, triangulation, validation, serialization,
  memory, output complexity, and construction/test counters.

Floating-point previews never participate in exact validation. JSON boundaries
can be weakly simple at touching vertices; the exact reader splits signed cycles
without dropping their positive-area components. Exported faces form one conforming
2D constrained triangulation for each layer, with source boundaries preserved.

Positive-area coincident surface overlaps are explicitly rejected, including
fully hidden duplicates. Coplanar adjacent faces pass. Zero projected-area faces
are counted and omitted. Sets are regularized: isolated points and line-only
remnants do not represent standalone surfaces. This does not define boundary-ray
ownership or reproduce a GPU fill convention.

## Algorithms and validation

`arrangement` inserts domain boundaries and equality segments clipped to pairwise
overlap, nodes them exactly, then triangulates the whole subdivision. Its triangle
barycenters are guaranteed interior samples. Membership and depth order are
constant in each resulting triangle, so assigning every rank and taking labelled
unions yields the oracle regions. This triangulation refines arrangement cells;
raw triangle counts and arrangement face counts are reported separately.

`envelope-peel` triangulates each residual source, builds its affine score graph,
uses CGAL's lower envelope, extracts outer and hole cycles, then unions faces by
original source. Residuals never retain adapter triangulation edges.

`visible-peel` inserts each residual once into a disjoint visible map. Exact affine
half-planes resolve crossings. Conservative component-box queries, strict depth
separation, and bounding-rectangle containment in a single convex visible component
avoid unnecessary Boolean operations. The xmin-sorted component index is rebuilt
after each mutation. It is deliberately simple; it is not a hierarchical BVH.
Persistent residuals change only after the pass is finalized.

The fixture driver compares all methods and insertion orders, unaccelerated
variants, and independent K=1,2,4 prefixes against all-layer results. Exact checks
cover source containment, per-source conservation and disjointness, layer
uniqueness, rank regions, positive triangulation area, triangle overlap, T-junctions,
region equality, original-plane barycentrics, and reprojection. All-layer runs
also check completion and projected area with multiplicity. The cyclic fixture
certifies strict cyclic depth inequalities over overlap closures and absence of a
triple overlap. Fixed-seed random soups supplement the named degeneracy fixtures.

## Experiments

```sh
python -m pip install 'matplotlib==3.10.9'
python scripts/benchmark.py --suite all --output results/benchmark \
  --timeout 5 --budget 600 --memory-mib 2048
python scripts/plot_results.py results/benchmark
```

The runner uses one validation/warm-up run and three measured repetitions, with
separate processes, a per-process wall-clock and address-space cap, and a total
suite budget. Failed, timed-out, capped, and budget-exhausted cases remain in
`status.jsonl`. Successful measurements are in `runs.jsonl` and `summary.csv`;
plots show median and range. All fixture/camera manifests are saved in `scenes/`.
The default foreground sweep uses g=2,4,8,16,32,64, all insertion orders, and
K=1,2,4,all, with and without the foreground. Other suites exercise stacks,
disjoint projections, holes, crossings, perspective, and generated closed meshes.

Algorithm timing includes visibility, residual updates, and final triangulation.
Projection/preflight, validation, and serialization are reported separately.
Envelope adapter triangulation and envelope construction have separate sub-timers.
`materialized_result_boundary_vertices` counts vertices in the returned Boolean
results, including repeated materialization; it is **not** CGAL's number of
internal constructed vertices. Actual arrangement vertices/edges are separately
available. Internal CGAL construction and peak-live-geometry counters are `null`.
`peak_live_boundary_vertices_lower_bound` counts visible-map and residual boundaries;
process peak RSS provides the whole-process memory observation. Certificate tests,
index box tests, candidate comparisons, intersections, differences, and half-plane
clips are distinct counters. Validation never contributes to algorithm counters.

See [results/findings.md](results/findings.md) for measured results and limitations.
This is a reference prototype: no GPU implementation, differentiation, production
mesh loader, or claim of novelty in polygon visibility is included.

## Reference implementation and licenses

The CGAL envelope and triangulation API examples linked in the design informed the
adapters. The implementation uses exact constructions and labelled classification,
including hole cycles, rather than the example's outer-boundary printer and
single-polygon parity marking. CGAL's Envelope_3, Boolean_set_operations_2,
Arrangement_on_surface_2, and Triangulation_2 packages are GPL-3.0-or-later or
commercially licensed; see their installed headers and the CGAL license files
before redistributing a linked build. This repository does not vendor CGAL.

The vectorizer archive at revision recorded in `dependencies.json` was inspected
for BVH traversal, convex splitting, and merging. Its integer rounding, convex-hull
repair, and area thresholds are not used here. No archive source was incorporated.
All generated scenes are original synthetic fixtures provided as CC0-1.0 data;
connected meshes are rational gridded, sheared closed boxes, with camera and
backface settings recorded in their manifests.

For additional many-hole and nested-surface workloads, run
`python scripts/benchmark.py --suite broader --output results/broader`. These
include 17/65-source perforated scenes and closed nested boxes with 96/1176 faces
and up to four ray hits. The supplemental suite is separate from the default
budgeted `all` sweep so it can receive its own explicit budget.

`bash scripts/reproduce.sh` runs the validation, separately budgeted suites, plots,
and retained example exports in sequence. Benchmark filters `--methods`, `--orders`,
`--layers`, and `--only-workloads` support focused reruns. The recorded benchmark
configuration and executable SHA-256 identify each run; results are machine- and
load-dependent.

The identical-footprint stack is an implementation diagnostic, not a representative
3D mesh workload or a lower bound for peeling. Constant-depth identical footprints
can be sorted once and emitted without repeated source scans. The next comparison
should include that control and persistent depth-aware traversal with certified
bounds; projected-box overlap alone cannot distinguish a stack. Practical assessment
prioritizes connected geometry and K=1,2,4. The `practical` benchmark suite adds an
oblique normalized pinhole camera and a nonconvex, connected rational torus alongside
the 1,200-face box:

```sh
python scripts/benchmark.py --suite practical --methods visible-peel envelope-peel \
  --orders near-first --layers 1 2 4 --timeout 30 --budget 300 --output results/practical
```

The total suite budget stops admitting new configurations; an in-flight repetition
group can finish afterward. Per-process wall-clock and address-space caps still apply.

The 384-face faceted torus has small six-hit regions. Its K=4 benchmark is a prefix;
the retained examples also contain complete six-layer outputs from both methods.
`scripts/check_rays.py` independently intersects rational camera rays with original
3D triangles. The recorded checks cover every exported face in layers 5 and 6;
these samples supplement, and do not replace, exact region comparisons.
