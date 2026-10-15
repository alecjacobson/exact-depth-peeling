# Findings: exact geometric depth peeling

The discussion below records the original polygon-set experiments. The new
convex-fragment solvers and complete-layer measurements on 10K/100K connected
meshes are documented in the [current README](../README.md#large-connected-mesh-example)
and [large-mesh results](large/README.md). The faster new method, `bvh-peel`, uses
independent BVH visibility queries; `vbvh-peel` uses an R3-inspired dynamic visible
map. Exact all-layer extraction now scales to 102,400 triangles, but the measured
seconds-per-extraction costs still miss the near-real-time goal.

The prototype supports output selectivity, and demonstrates computational
selectivity for the foreground-cover workload with a favorable insertion order.
Practical usefulness is assessed with connected geometry, camera/depth complexity,
and K=1,2,4. On the measured connected scenes the current CGAL envelope baseline
is faster than this incremental prototype; this is not a conclusion about
optimal peeling algorithms. Identical stacks are a separate implementation diagnostic.

## Correctness and reproducibility

The release executable passed 37 fixtures and
2,281 validation/comparison subprocess runs: 35
accepted fixtures plus 2 expected positive-area-tie rejections. Checks include
exact labelled symmetric difference with the oracle, three independently requested
prefixes, insertion-order invariance, unaccelerated variants, holes/components,
strict nonintersecting cyclic occlusion, crossing planes, exact reconstruction,
conforming triangulations, perspective clipping, and 12 fixed-seed random soups.
See [validation.json](validation.json) and [retained examples](examples/).

Measurements use compiler version 11.4.0, CGAL 5.4,
exact GMP rationals, and an Intel(R) Xeon(R) Platinum 8362 CPU @ 2.80GHz shared host. Each successful configuration has one warm-up and three measured
repetitions. The [dependency record](../dependencies.json), scene manifests, and
executable hashes identify the environment and inputs. Run
[reproduce.sh](../scripts/reproduce.sh) to repeat the experiment.

Across the separately budgeted suites there are 615 successful
configuration-runs, 112 process timeouts, 40
oracle-cap skips, and 423 total-budget skips. There
were no geometry-error results. Some configurations recur in focused reruns.
The standard process cap is 5 seconds / 2 GiB; focused connected-mesh runs use
30 seconds. Warm-up includes validation: its timeouts are **not** algorithm-only
lower bounds. Larger cases receive exact invariant checks but no arrangement-rank
certificate above the 64-source oracle cap. All statuses remain in the data.

## Foreground cover

Every completed cover run, including normalized oracle output, retained exactly
three boundary vertices and one output triangle in layer 1. Thus small final
output alone cannot distinguish the algorithms' work.

For near-first K=1, every measured g rejected all 4g background triangles by the
convex-cell certificate, with zero visibility polygon-intersection calls and a
three-vertex visible map. At g=8, reverse insertion instead reached
601 visible-map boundary vertices,
while the oracle constructed 1190 triangulated
verification cells in the window. Final foreground output was still one triangle.
Internal CGAL envelope constructions are unavailable; its small final diagram
must not be interpreted as zero hidden work.

Median **algorithm milliseconds**, including residual updates and final
triangulation, excluding setup, validation, and serialization:

| g | Sources | Visible near-first | Visible reverse | Envelope | Arrangement |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 9 | 0.313 | 3.033 | 1.974 | 13.757 |
| 4 | 17 | 0.455 | 13.468 | 4.211 | 68.191 |
| 8 | 33 | 0.667 | 71.070 | 10.752 | 381.248 |
| 16 | 65 | 1.148 | 455.100 | 33.860 | — |
| 32 | 129 | 2.105 | 3272.768 | 115.052 | — |
| 64 | 257 | 4.138 | — | 498.944 | — |

A dash denotes an unmeasured/capped case, not zero time. The main sweep exhausted
its 600-second budget before g=64, so that first-layer case received a separate
budget. At g=64, setup/preflight alone took 44.120 ms (median), versus 4.138 ms for the algorithm. Mandatory tie preflight itself
performs overlap tests; the zero-intersection result applies to visibility.
Processing hidden inputs is not constant-time. Full medians/ranges and all orders
are in [the main table](benchmark/summary.csv) and
[the g=64 rerun](large-first/summary.csv).

[Cover plots](benchmark/foreground-cover.svg) and
[insertion-order plots](benchmark/insertion-order.svg) show the output/work
separation. Removing the foreground makes the crossings relevant immediately:
[no-foreground plots](benchmark/no-foreground.svg). The g=8 all-layer near-first
run spent about 70.5% of algorithm time in
final triangulation and source labelling; the later geometry is real work.

## Connected-scene measurements

The practical suite uses a normalized pinhole camera, an exact (5,12,13) rotation,
a positive near plane, and connected closed geometry: a nonconvex 384-face torus
and a 1,200-face box. These are controlled synthetic probes, not a representative
sample of typical production models. K=1,2,4 measurements are:

| Scene | K | Visible near-first ms: median [min, max] | Envelope ms: median [min, max] |
| --- | ---: | ---: | ---: |
| perspective-torus | 1 | 695.712 [695.561, 698.340] | 313.079 [312.778, 313.597] |
| perspective-torus | 2 | 1250.421 [1244.603, 1255.447] | 554.592 [553.632, 557.600] |
| perspective-torus | 4 | 1393.482 [1383.940, 1393.666] | 725.771 [724.727, 729.904] |
| perspective-box | 1 | 5313.780 [5296.501, 5326.802] | 1002.245 [983.443, 1032.092] |
| perspective-box | 2 | 10384.111 [10333.824, 10400.947] | 1904.463 [1895.701, 1910.581] |
| perspective-box | 4 | 10400.751 [10379.327, 10460.522] | 1885.555 [1883.745, 1902.407] |

[Practical measurements](practical/summary.csv), [camera manifests](practical/scenes/),
and [connected-perspective plots](practical/connected-perspective.svg) take priority
for assessing this prototype; the stack diagnostic below is kept separate.
The small perspective torus also receives full exact arrangement comparisons
across methods, orders, and prefixes. Larger scenes receive invariant validation.

The larger faceted torus retains small positive-area residuals after K=4. Both methods agree exactly on all six layers in a separate completion check. 45 independent rational ray/3D-triangle samples across layers 5 and 6 confirm the expected source ranks and six-hit regions. These are spot checks, not a full verification arrangement. See [ray checks](practical/late-layer-ray-checks.json) and [full-layer exports](examples/perspective-torus-384/). K=4 is a requested prefix, not an all-layer result for this mesh.

The generated closed meshes retain entry and exit surfaces; nested boxes have
up to four hits. These are original CC0 rational fixtures, not downloaded models.
The longer-cap paired measurements are:

| Scene | K | Visible near-first ms: median [min, max] | Envelope ms: median [min, max] |
| --- | ---: | ---: | ---: |
| connected-mesh, 1200 faces | 1 | 5715.549 [5683.396, 5753.093] | 894.257 [886.996, 895.793] |
| connected-mesh, 1200 faces | all | 8082.914 [8053.427, 8092.932] | 1332.094 [1330.561, 1362.117] |
| nested-mesh, 1176 faces | 1 | 2813.903 [2807.319, 2814.439] | 749.380 [743.274, 753.791] |
| nested-mesh, 1176 faces | all | 7594.365 [7548.037, 7631.144] | 1810.082 [1806.806, 1820.113] |

See [connected data](connected/summary.csv), [nested data](nested/summary.csv),
and [perforated/other workloads](broader/summary.csv). Source boundaries are
preserved, so many small visible source faces limit the single-convex-cell
occlusion certificate. The component index currently rebuilds after every
noncertified insertion, even when clipping ultimately leaves the map unchanged.
This implementation cost is part of the measured result.

## Identical-footprint implementation diagnostic

Identical stacked footprints are not representative of most 3D models. This
fixture diagnoses repeated scanning/rebuilding and absent cross-pass reuse in
these implementations; it is not an inherent lower bound for geometric peeling.
All 64 layers produce just 64 output triangles:

| Method | Algorithm ms: median [min, max] |
| --- | ---: |
| visible-peel | 138.444 [137.185, 140.582] |
| envelope-peel | 295.503 [294.073, 295.830] |
| arrangement | 21.723 [21.717, 21.908] |

Visible peeling performed 2,080 polygon differences; the envelope adapter built
2,080 triangles across repeated passes. The oracle sorted a small common
subdivision once. Those triangular sums count this code's source visits/adapter
work, not unavoidable geometric work. A sort-once baseline can sort constant
depths in O(N log N) and emit all layers in O(N) total; a persistent ordered
structure can find/remove the next source in O(log N) per layer. A depth-aware
hierarchy with certified bounds can prune subtrees, whereas projected boxes alone
cannot distinguish these identical footprints. An arbitrary BVH does not guarantee
logarithmic queries. These alternatives have not been benchmarked here. See [stack table](stacks/summary.csv) and
[stack plots](stacks/stacked.svg).

## Next investment and limits

Improve spatial indexing and source labelling first: avoid unchanged-map index
rebuilds, update only changed components, and index labels during final
triangulation. Next profile residual/adapter triangulation reuse with explicit
invalidation, and collective-coverage certificates for finely tessellated covers.
Add a sort-once stack control and a persistent, depth-aware hierarchy before
using stack timings to assess peeling. Reuse across passes and subtree depth/
coverage certificates deserve investigation. General meshes with crossings and
cyclic occlusion still require geometric visibility decisions; the stack's simple
scalar ordering is not assumed for those cases.

Construction counters are explicitly scoped. Returned Boolean-result boundary
vertices count repeated materializations, not CGAL internal constructions. Visible
map maxima are sampled after insertions; peak-live boundary counts are lower
bounds. Whole-process peak RSS and actual arrangement sizes are recorded. The
oracle includes straightforward incremental cell unions and retained depth-list
diagnostics; this is a comparison of prototypes, not optimal implementations or
all possible all-layer algorithms. No 10k-face scaling, GPU comparison, or claim of
novel polygon-visibility mathematics is made.
