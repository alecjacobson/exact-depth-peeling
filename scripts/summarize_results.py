#!/usr/bin/env python3
"""Write the short findings report from completed benchmark tables and raw metrics."""

import csv
import json
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DIRECTORIES = (
    "benchmark",
    "stacks",
    "other",
    "broader",
    "connected",
    "nested",
    "large-first",
    "practical",
)
rows = []
statuses = []
for directory in DIRECTORIES:
    path = RESULTS / directory
    if (path / "summary.csv").exists():
        with (path / "summary.csv").open() as stream:
            rows.extend(
                dict(row, directory=directory) for row in csv.DictReader(stream)
            )
    if (path / "status.jsonl").exists():
        statuses.extend(
            json.loads(line)
            for line in (path / "status.jsonl").read_text().splitlines()
        )


def select(suite, size, layers, method, order="input", directory=None):
    candidates = [
        row
        for row in rows
        if row["suite"] == suite
        and row["size"] == str(size)
        and row["layers"] == str(layers)
        and row["method"] == method
        and row["order"] == order
        and (directory is None or row["directory"] == directory)
    ]
    return candidates[-1] if candidates else None


def milliseconds(row, with_range=False):
    if row is None:
        return "—"
    value = 1000 * float(row["median_seconds"])
    text = f"{value:.3f}"
    if with_range:
        text += f" [{1000 * float(row['min_seconds']):.3f}, {1000 * float(row['max_seconds']):.3f}]"
    return text


cover_rows = [r for r in rows if r["suite"] == "foreground-cover"]
assert all(
    r["layer1_boundary_vertices"] == "3" and r["layer1_triangles"] == "1"
    for r in cover_rows
)
near_rows = [
    r
    for r in cover_rows
    if r["method"] == "visible-peel"
    and r["order"] == "near-first"
    and r["layers"] == "1"
]
assert all(
    r["polygon_intersections"] == "0"
    and int(r["complete_occlusion_rejections"]) == 4 * int(r["size"])
    for r in near_rows
)

cover_table = []
for size in (2, 4, 8, 16, 32, 64):
    selected = [
        select("foreground-cover", size, 1, method, order)
        for method, order in (
            ("visible-peel", "near-first"),
            ("visible-peel", "reverse"),
            ("envelope-peel", "input"),
            ("arrangement", "input"),
        )
    ]
    cover_table.append(
        f"| {size} | {4 * size + 1} | "
        + " | ".join(milliseconds(r) for r in selected)
        + " |"
    )

stack_table = []
for method, order in (
    ("visible-peel", "near-first"),
    ("envelope-peel", "input"),
    ("arrangement", "input"),
):
    r = select("stacked", 64, "all", method, order, "stacks")
    stack_table.append(f"| {method} | {milliseconds(r, True)} |")

mesh_table = []
for suite, size, directory in (
    ("connected-mesh", 10, "connected"),
    ("nested-mesh", 7, "nested"),
):
    for layers in (1, "all"):
        v = select(suite, size, layers, "visible-peel", "near-first", directory)
        e = select(suite, size, layers, "envelope-peel", "input", directory)
        n = 1200 if suite == "connected-mesh" else 1176
        mesh_table.append(
            f"| {suite}, {n} faces | {layers} | {milliseconds(v, True)} | {milliseconds(e, True)} |"
        )

practical_table = []
for suite, size in (("perspective-torus", 4), ("perspective-box", 10)):
    for layers in (1, 2, 4):
        v = select(suite, size, layers, "visible-peel", "near-first", "practical")
        e = select(suite, size, layers, "envelope-peel", "input", "practical")
        practical_table.append(
            f"| {suite} | {layers} | {milliseconds(v, True)} | {milliseconds(e, True)} |"
        )

fractions = []
for line in (RESULTS / "benchmark/runs.jsonl").read_text().splitlines():
    sample = json.loads(line)
    if (
        sample["suite"] == "foreground-cover"
        and sample["size"] == 8
        and sample["layers"] == "all"
        and sample["method"] == "visible-peel"
        and sample["order"] == "near-first"
    ):
        m = sample["metrics"]
        fractions.append(
            sum(p["triangulation_seconds"] for p in m["passes"])
            / m["algorithm_seconds"]
        )
counts = Counter(s["status"] for s in statuses)
errors = [s for s in statuses if s["status"] == "error"]
assert not errors, errors
validation = json.loads((RESULTS / "validation.json").read_text())
large = select("foreground-cover", 64, 1, "visible-peel", "near-first")
setup_note = (
    f"At g=64, setup/preflight alone took {1000 * float(large['setup_median_seconds']):.3f} ms (median), versus {milliseconds(large)} ms for the algorithm. "
    if large
    else ""
)
reverse = select("foreground-cover", 8, 1, "visible-peel", "reverse")
oracle = select("foreground-cover", 8, 1, "arrangement")
with (RESULTS / "benchmark/runs.jsonl").open() as stream:
    environment = json.loads(next(stream))["metrics"]
cpu = json.loads((RESULTS / "stacks/config.json").read_text()).get(
    "cpu", "CPU recorded in configuration"
)
ray_report_path = RESULTS / "practical/late-layer-ray-checks.json"
ray_note = ""
if ray_report_path.exists():
    ray_report = json.loads(ray_report_path.read_text())
    ray_note = f"The larger faceted torus retains small positive-area residuals after K=4. Both methods agree exactly on all six layers in a separate completion check. {ray_report['query_count']} independent rational ray/3D-triangle samples across layers 5 and 6 confirm the expected source ranks and six-hit regions. These are spot checks, not a full verification arrangement. See [ray checks](practical/late-layer-ray-checks.json) and [full-layer exports](examples/perspective-torus-384/). K=4 is a requested prefix, not an all-layer result for this mesh."
report = f"""# Findings: exact geometric depth peeling

The prototype supports output selectivity, and demonstrates computational
selectivity for the foreground-cover workload with a favorable insertion order.
Practical usefulness is assessed with connected geometry, camera/depth complexity,
and K=1,2,4. On the measured connected scenes the current CGAL envelope baseline
is faster than this incremental prototype; this is not a conclusion about
optimal peeling algorithms. Identical stacks are a separate implementation diagnostic.

## Correctness and reproducibility

The release executable passed {len(validation["fixtures"])} fixtures and
{validation["subprocess_checks"]:,} validation/comparison subprocess runs: {sum(x["status"] == "passed" for x in validation["fixtures"])}
accepted fixtures plus {sum(x["status"] == "expected_rejection" for x in validation["fixtures"])} expected positive-area-tie rejections. Checks include
exact labelled symmetric difference with the oracle, three independently requested
prefixes, insertion-order invariance, unaccelerated variants, holes/components,
strict nonintersecting cyclic occlusion, crossing planes, exact reconstruction,
conforming triangulations, perspective clipping, and 12 fixed-seed random soups.
See [validation.json](validation.json) and [retained examples](examples/).

Measurements use compiler version {environment["compiler"]}, CGAL {environment["cgal_version"]},
exact GMP rationals, and an {cpu} shared host. Each successful configuration has one warm-up and three measured
repetitions. The [dependency record](../dependencies.json), scene manifests, and
executable hashes identify the environment and inputs. Run
[reproduce.sh](../scripts/reproduce.sh) to repeat the experiment.

Across the separately budgeted suites there are {counts["ok"]} successful
configuration-runs, {counts["timeout"]} process timeouts, {counts["oracle_cap"]}
oracle-cap skips, and {counts["total_budget_exhausted"]} total-budget skips. There
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
{reverse["max_visible_map_boundary_vertices"]} visible-map boundary vertices,
while the oracle constructed {oracle["raw_arrangement_triangles"]} triangulated
verification cells in the window. Final foreground output was still one triangle.
Internal CGAL envelope constructions are unavailable; its small final diagram
must not be interpreted as zero hidden work.

Median **algorithm milliseconds**, including residual updates and final
triangulation, excluding setup, validation, and serialization:

| g | Sources | Visible near-first | Visible reverse | Envelope | Arrangement |
| ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(cover_table)}

A dash denotes an unmeasured/capped case, not zero time. The main sweep exhausted
its 600-second budget before g=64, so that first-layer case received a separate
budget. {setup_note}Mandatory tie preflight itself
performs overlap tests; the zero-intersection result applies to visibility.
Processing hidden inputs is not constant-time. Full medians/ranges and all orders
are in [the main table](benchmark/summary.csv) and
[the g=64 rerun](large-first/summary.csv).

[Cover plots](benchmark/foreground-cover.svg) and
[insertion-order plots](benchmark/insertion-order.svg) show the output/work
separation. Removing the foreground makes the crossings relevant immediately:
[no-foreground plots](benchmark/no-foreground.svg). The g=8 all-layer near-first
run spent about {100 * statistics.median(fractions):.1f}% of algorithm time in
final triangulation and source labelling; the later geometry is real work.

## Connected-scene measurements

The practical suite uses a normalized pinhole camera, an exact (5,12,13) rotation,
a positive near plane, and connected closed geometry: a nonconvex 384-face torus
and a 1,200-face box. These are controlled synthetic probes, not a representative
sample of typical production models. K=1,2,4 measurements are:

| Scene | K | Visible near-first ms: median [min, max] | Envelope ms: median [min, max] |
| --- | ---: | ---: | ---: |
{chr(10).join(practical_table)}

[Practical measurements](practical/summary.csv), [camera manifests](practical/scenes/),
and [connected-perspective plots](practical/connected-perspective.svg) take priority
for assessing this prototype; the stack diagnostic below is kept separate.
The small perspective torus also receives full exact arrangement comparisons
across methods, orders, and prefixes. Larger scenes receive invariant validation.

{ray_note}

The generated closed meshes retain entry and exit surfaces; nested boxes have
up to four hits. These are original CC0 rational fixtures, not downloaded models.
The longer-cap paired measurements are:

| Scene | K | Visible near-first ms: median [min, max] | Envelope ms: median [min, max] |
| --- | ---: | ---: | ---: |
{chr(10).join(mesh_table)}

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
{chr(10).join(stack_table)}

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
"""
(RESULTS / "findings.md").write_text(report)
print("Wrote results/findings.md")
