#!/usr/bin/env python3
"""Plot recorded large-mesh timings and replace the README's measured-results block."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path, nargs="?", default=ROOT / "results/large")
    a = p.parse_args()
    rows = list(csv.DictReader((a.directory / "summary.csv").open()))
    config = json.loads((a.directory / "config.json").read_text())
    colors = {"bvh-peel": "#2766b4", "vbvh-peel": "#c56d32"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    for method, color in colors.items():
        data = sorted(
            [r for r in rows if r["method"] == method],
            key=lambda r: int(r["input_triangles"]),
        )
        if not data:
            continue
        n = [int(r["input_triangles"]) for r in data]
        times = [float(r["median_seconds"]) for r in data]
        errors = [
            [float(r["median_seconds"]) - float(r["min_seconds"]) for r in data],
            [float(r["max_seconds"]) - float(r["median_seconds"]) for r in data],
        ]
        axes[0].errorbar(
            n,
            times,
            yerr=errors,
            marker="o",
            capsize=4,
            color=color,
            label=method,
        )
        axes[1].plot(
            n,
            [int(r["candidate_tests"]) / int(r["input_triangles"]) for r in data],
            "o-",
            color=color,
            label=method,
        )
    axes[0].axhline(
        0.1, color="#7b8492", linestyle="--", linewidth=1, label="100 ms reference"
    )
    axes[0].set(
        xscale="log",
        yscale="log",
        xlabel="Input triangles",
        ylabel="All-layer extraction + triangulation (s)",
        title="Complete extraction; median and range",
    )
    axes[1].set(
        xscale="log",
        xlabel="Input triangles",
        ylabel="Candidate tests / input triangle",
        title="Candidate work across all layers",
    )
    for ax in axes:
        ax.grid(True, which="major", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8)
    fig.suptitle(
        "Connected sculpted torus · exact constructions · four-core CPU quota",
        fontsize=12,
    )
    for ext in ("png", "svg", "pdf"):
        fig.savefig(a.directory / f"scaling.{ext}", dpi=180)
    svg = a.directory / "scaling.svg"
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n"
    )
    plt.close(fig)

    selected = [r for r in rows if int(r["input_triangles"]) in (10240, 102400)]
    table = [
        "| Input triangles | Solver | Layers | Extraction median (range) | Input setup | Output triangles |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for r in selected:
        table.append(
            f"| {int(r['input_triangles']):,} | `{r['method']}` | {r['layers']} | "
            f"{float(r['median_seconds']):.3f} s ({float(r['min_seconds']):.3f}–{float(r['max_seconds']):.3f}) | "
            f"{float(r['setup_seconds']):.3f} s | {int(r['output_triangles']):,} |"
        )
    statuses = [
        json.loads(x) for x in (a.directory / "status.jsonl").read_text().splitlines()
    ]
    failures = [s for s in statuses if s["status"] != "completed"]
    caveats = ""
    if failures:
        caveats = (
            "\nUnfinished configurations: "
            + "; ".join(
                f"{s['method']}, {s['input_triangles']:,} triangles: {s['status']}"
                for s in failures
            )
            + ". See the raw status log.\n"
        )
    block = (
        """## Large connected-mesh example

![Exploded view of the input mesh, perspective camera, and every extracted layer](results/large/demo/exploded.png)

The example is a connected, closed sculpted torus with **10,240 triangles**,
viewed through an oblique perspective camera. Each coloured layer is lifted onto
its original 3D source surfaces, then translated for this exploded view. All layers
are shown at the same scale; later ranks contain tiny silhouette fragments. The
renderer consumes the actual exported layer meshes. No backface culling or mesh
simplification is used. A **102,400-triangle** version tests the same shape at a
higher tessellation density.

"""
        + "\n".join(table)
        + caveats
        + """

**The near-real-time target has not been reached in exact mode.** These are measured complete
extractions, including conforming labelled triangulation, not extrapolated timings
or first-layer results. The plot's 100 ms line is a performance reference, not an
achieved guarantee. Different tessellations can have different maximum depth
complexity, including small extra layers near silhouettes.

Measurements use one warm-up and three subsequent extractions of the same
projected input. The table reports their median and range. Input parsing,
projection, and positive-area tie preflight are listed separately; validation and
file export are excluded. Thus these are not end-to-end animation-frame timings.
The environment has a four-core CPU quota on an Intel Xeon Platinum 8362 host.
Both methods use four threads for meshing; `vbvh-peel` updates its visibility map
sequentially. Interval filtering falls back to exact rational arithmetic, and all
constructed geometry remains exact.

Output triangle counts in the table come from the final timed repetition. Internal
convex subdivisions are not canonical: conservative interval bounds, lazy exact
evaluation and candidate traversal can change the triangulation while preserving
the source regions. The independently validated demonstration can therefore have
different face counts; both sets of counts are retained in `audits.json`.

![Measured scaling of complete extraction and candidate work](results/large/scaling.png)

The candidate-work plot measures this family, not a universal complexity bound.
The two solvers avoid the old per-insertion full-index rebuild and exhaustive
output-face label search. Fragmentation, repeated passes, and exact constructions
still contribute substantial cost. R3's finite-precision single-visibility results
are not directly comparable to these exact all-layer timings.

The current measurements point to visibility/clipping as the next optimization
target: it accounts for about 90% of the instrumented extraction stages in the
100K `bvh-peel` run. The next steps are stronger collective-coverage certificates,
less allocation and rational-expression work in clipping, and measured reuse
between layers with explicit invalidation. The exact solver remains the correctness
reference for any future finite-precision or GPU implementation. Neither the BVH
nor this mesh family establishes an output-independent subquadratic bound: the
number of fragments and depth layers can itself grow with the input.

[Raw measurements](results/large/summary.csv), [configuration and source hashes](results/large/config.json),
[attempts and timeouts](results/large/status.jsonl), and
[large-mesh validation](results/large/audits.json) are retained.
The [Blender scene](results/large/demo/exploded.blend) and layer OBJs in
`results/large/demo/` can be inspected separately (Git LFS is required for large files).

Reproduce the measurement and the exploded view:

```sh
python scripts/large_benchmark.py --threads 4 --timeout 360 --output results/large
python scripts/audit_large_results.py --threads 4 --directory results/large
python scripts/render_exploded.py \\
  --scene results/large/scenes/sculpted-10240.json --layers results/large/demo \\
  --output results/large/demo/exploded.png
python scripts/report_large_results.py results/large
```

Rendering requires Blender (the recorded figure uses 4.5.3) and Pillow; plotting
requires matplotlib. They are not part of the extraction timer.
"""
    )
    readme = ROOT / "README.md"
    s = readme.read_text()
    start, end = "<!-- LARGE_RESULTS_BEGIN -->", "<!-- LARGE_RESULTS_END -->"
    before, rest = s.split(start, 1)
    _, after = rest.split(end, 1)
    readme.write_text(before + start + "\n" + block + end + after)
    (a.directory / "README.md").write_text(
        "# Complete-layer mesh measurements\n\n"
        "The root README contains the figure, measured table, timing boundaries, and interpretation.\n"
        "`config.json` identifies the executable, source files, machine quota, and numerical mode.\n"
        "`summary.csv` contains three-repetition medians/ranges after one warm-up.\n"
        "`status.jsonl` retains every attempted configuration, including timeouts and errors.\n"
        "Per-configuration `metrics.json` files retain all repetitions and layer counters.\n"
        "`audits.json` and `demo/metrics.json` record exact invariants and sampled independent ray ranks.\n"
        "`demo/` contains the actual lifted layer meshes and the exploded Blender render.\n\n"
        f"Executable SHA-256: `{config['executable_sha256']}`.\n"
    )


if __name__ == "__main__":
    main()
