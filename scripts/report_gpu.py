#!/usr/bin/env python3
"""Render the retained GPU timing records; no benchmark execution."""

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/gpu"


def main():
    records = json.loads((OUT / "benchmark.json").read_text())["cases"]
    profile = json.loads((OUT / "phases.json").read_text())
    audits = json.loads((OUT / "audits.json").read_text())
    suite = ET.parse(OUT / "validation.xml").getroot().find("testsuite")
    validation = {
        "tests": int(suite.attrib["tests"]),
        "failures": int(suite.attrib["failures"]),
        "errors": int(suite.attrib["errors"]),
        "seconds": float(suite.attrib["time"]),
        "cases": [t.attrib["name"] for t in suite.findall("testcase")],
        "kernel_sha256": hashlib.sha256(
            (ROOT / "warp_peel/kernels.py").read_bytes()
        ).hexdigest(),
        "reference_executable_sha256": hashlib.sha256(
            (ROOT / "build/geometric_peel").read_bytes()
        ).hexdigest(),
        "oracle": "original exact arrangement with --verify",
        "normalized_area_tolerance": 1e-9,
    }
    (OUT / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")

    timing_rows = "\n".join(
        f"| {'Dragon' if i < 4 else 'Sculpted mesh'} | {c['sources']:,} | {c['output']['layers']} | {c['median_ms'] / 1000:.3f} s | {c['output']['triangles']:,} |"
        for i, c in enumerate(records)
    )
    audit_rows = "\n".join(
        f"| {'Dragon' if i < 4 else 'Sculpted mesh'} | {records[i]['sources']:,} | {a['checked_rays']} | {a['rank_mismatches']} |"
        for i, a in enumerate(audits)
    )
    clipping = sum(p["clip_ms"] for p in profile["layers"])
    hierarchy = sum(p["hierarchies_ms"] for p in profile["layers"])
    compact = sum(p["compact_ms"] for p in profile["layers"])
    total = clipping + hierarchy + compact + profile["prepare_ms"] + profile["mesh_ms"]
    (OUT / "README.md").write_text(f"""# Warp GPU results

Measured on NVIDIA L40, Warp 1.15.0, driver 570.158.01, on a shared machine.
This implementation completes all-layer, analytic GPU extraction and graph
replay. **It does not yet meet the near-real-time performance goal and is not
an exact correctness oracle.** The CPU implementations remain available.

## Full graph replay

| Input | Resolved triangles | Layers | Median replay | Output triangles |
| --- | ---: | ---: | ---: | ---: |
{timing_rows}

Three CUDA-event repetitions follow JIT, capture and two complete warm-up replays.
Timing includes device input validation, welding, projection, topology/contours,
all hierarchy work, all peeling passes, boundary noding and world-space mesh
construction. Upload, allocation, JIT, graph capture, readback and audits are
excluded and setup costs are recorded separately. The earlier CPU measurements
exclude projection/topology setup, so these are not matched speedup comparisons.
The GPU route remains slower than the earlier CPU fast route on the 64K dragon.

[Raw repetitions, capacities, input/source hashes and process snapshots](benchmark.json).
No GPU reservation or interference control was applied. Other simulation/render
jobs ran during development; inspect the retained case-start snapshots for the
final run. The machine's CPU quota is shared as well. The distinct sculpted model
has substantially less residual work than the dragon; its 100K timing is not a
prediction for a 100K dragon. The full 721,510-face dragon is not GPU-benchmarked.

![GPU replay scaling and clipping cost by layer](timings.svg)

## Where the time goes

A separate, uncaptured 64K diagnostic records **{clipping / 1000:.3f} s clipping**,
{hierarchy:.2f} ms hierarchy work, {compact:.2f} ms residual compaction,
{profile["prepare_ms"]:.2f} ms preparation and {profile["mesh_ms"]:.2f} ms meshing.
Clipping accounts for **{100 * clipping / total:.1f}%** of that diagnostic. Its phase
sum is not the graph-replay median: event boundaries include host launch gaps
and the run has its own scheduling/load conditions. [Per-layer events](phases.json).

The next GPU work should target the long, divergent per-target clipping loops
and large fixed polygon temporaries: compact work queues, more parallel clipping,
and data layouts that reduce copies/register spills. Hierarchy polishing is not
supported as the priority by this profile. Later layers usually get cheaper, but
fragment distribution and work imbalance keep the decline irregular. Private
clipping scratch alone consumes about 12.2 GiB with the large default capacities;
a packed device pool would improve memory use as well as scheduling flexibility.

## Correctness and capture checks

**{validation["tests"]} integration checks pass**, including 35 accepted fixtures compared with the
original exact arrangement oracle. The source-labelled region and triangle unions
must agree to normalized image-area tolerance `1e-9`; this is a finite-tolerance
check, not a proof that sub-resolution features are preserved. Checks also cover
source-plane reconstruction, reprojection, selected T-junction cases, prefix
extraction, graph replay after device vertex/camera changes, multiple workspaces,
empty inputs, invalid device indices/cameras, explicit capacity errors and two
numerical regressions (a short clipping edge and a tiny backtracking edge).
[Validation summary](validation.json), [individual cases](validation.xml).

Capture checks prohibit host reads/synchronization during recording and inspect
the CUDA graph: conditional control flow is present, host/device transfers and
allocation nodes are absent. Device-to-device copies used by Warp primitives are
allowed. `warp_gpu` also passes through the optional CTest integration.

Independent rational ray/triangle tests compare the complete source-labelled
hit stack with downloaded GPU polygons, using a fixed grid, output-interior
samples and retained failing-ray probes:

| Input | Triangles | Nonempty rays checked | Rank mismatches |
| --- | ---: | ---: | ---: |
{audit_rows}

[Raw audits and mismatch examples](audits.json). A mismatch means a sampled ray's
layer/source sequence differs from the rational source intersections; this can
include omitted coverage, extra coverage or a wrong rank. Such failures remain
limitations of this approximate implementation. Passing samples do not certify
whole regions. Padded floating source bounds accelerate the rational tests;
queries within `1e-11` of an output boundary are excluded. The audit includes
[retained regression rays](../../tests/gpu/regression_rays.json), so historical
failing locations remain visible even when output subdivisions change.

Output order and some subdivisions vary across runs because atomics determine
append order, which can change candidate order and floating rounding. The
benchmark and audit therefore need not emit identical triangle counts. Interior
samples also depend on the resulting subdivision; mismatch counts across runs
are not a controlled comparison on an identical ray set. The fixed grid and
explicit regression probes are repeatable.

The two numerical fixes evaluate clipping edges relative to their own endpoint
and use a translation-stable polygon area/perimeter cleanup. No convex hull is
used to expand the visible set. Geometry is still binary64, with a spatial
resolution threshold near `1e-10`, and does not acquire the CPU solver's exact
predicates or guarantees. Positive-area depth ties use dense source indices;
CPU exact mode instead rejects them.

## Reproduce

From the repository root, with the CPU exact executable already built:

```bash
python3 -m pip install -r requirements-gpu-test.txt
python3 -m pytest tests/gpu -q --junitxml=results/gpu/validation.xml
python3 scripts/benchmark_gpu.py results/global/dragon-cache/{{1024,4096,16384,65536}}/resolved.json results/large/scenes/sculpted-102400.json --repetitions 3
python3 scripts/audit_gpu.py results/global/dragon-cache/{{1024,4096,16384,65536}}/resolved.json results/large/scenes/sculpted-102400.json
python3 scripts/profile_gpu.py results/global/dragon-cache/65536/resolved.json
python3 scripts/report_gpu.py
```

The profiler's host loop is deliberately confined to that uncaptured diagnostic;
the actual extractor uses a device conditional loop. See the
[device API and algorithm](../../warp_peel/README.md) for graph composition,
stream lifetime, input assumptions and mandatory device error-flag handling.
""")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    dragon = records[:4]
    axes[0].loglog(
        [c["sources"] for c in dragon],
        [c["median_ms"] / 1000 for c in dragon],
        "o-",
        label="Dragon (8–16 layers)",
        color="#246184",
    )
    c = records[4]
    axes[0].loglog(
        c["sources"],
        c["median_ms"] / 1000,
        "s",
        label="Sculpted mesh (10 layers)",
        color="#bd6428",
    )
    axes[0].set(
        xlabel="Input triangles",
        ylabel="All-layer graph replay (seconds)",
        title="L40 · medians of 3 · current host load",
    )
    axes[0].grid(alpha=0.2, which="both")
    axes[0].legend(fontsize=8)
    layers = profile["layers"]
    axes[1].bar(
        range(1, len(layers) + 1), [c["clip_ms"] for c in layers], color="#246184"
    )
    axes[1].set(
        xlabel="Peeling pass (1-based)",
        ylabel="Clipping kernel (ms)",
        title="64K dragon · separate uncaptured profile",
    )
    axes[1].grid(axis="y", alpha=0.2)
    path = OUT / "timings.svg"
    fig.savefig(path, metadata={"Date": None})
    path.write_text(
        "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
    )


if __name__ == "__main__":
    main()
