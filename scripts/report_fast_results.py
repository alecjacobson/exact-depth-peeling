#!/usr/bin/env python3
"""Publish the recorded fast/exact comparison without inventing missing measurements."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "results/fast"


def main():
    rows = list(csv.DictReader((DIR / "summary.csv").open()))
    lookup = {(r["variant"], int(r["input_triangles"])): r for r in rows}
    table = [
        "| Input | Layers | Exact BVH | Fast clipping | Fast + mesh contours | Speedup over exact |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    setup = []
    for n in (10240, 102400):
        exact, clipping, contours = [
            lookup[(m, n)] for m in ("exact-bvh", "fast-clipping", "fast-contours")
        ]
        table.append(
            f"| {n:,} | {contours['layers']} | {float(exact['median_seconds']) * 1000:,.1f} ms | "
            f"{float(clipping['median_seconds']) * 1000:,.1f} ms | **{float(contours['median_seconds']) * 1000:,.1f} ms** | "
            f"{float(exact['median_seconds']) / float(contours['median_seconds']):.1f}× |"
        )
        setup.append(f"{n:,}: {float(contours['setup_seconds']) * 1000:.1f} ms")
    fig, ax = plt.subplots(figsize=(8, 4.2), constrained_layout=True)
    for method, color in (
        ("exact-bvh", "#7E8796"),
        ("fast-clipping", "#CF8B36"),
        ("fast-contours", "#2677AD"),
    ):
        data = sorted(
            [r for r in rows if r["variant"] == method],
            key=lambda r: int(r["input_triangles"]),
        )
        ax.errorbar(
            [int(r["input_triangles"]) for r in data],
            [float(r["median_seconds"]) * 1000 for r in data],
            yerr=[
                [
                    1000 * (float(r["median_seconds"]) - float(r["min_seconds"]))
                    for r in data
                ],
                [
                    1000 * (float(r["max_seconds"]) - float(r["median_seconds"]))
                    for r in data
                ],
            ],
            marker="o",
            capsize=4,
            label=method,
            color=color,
        )
    ax.axhline(
        100, color="#9299A4", linestyle="--", linewidth=1, label="100 ms reference"
    )
    ax.axhline(
        1000 / 30,
        color="#B5BBC4",
        linestyle=":",
        linewidth=1,
        label="30 Hz extraction budget",
    )
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="Input triangles",
        ylabel="Complete extraction + triangulation (ms)",
        title="Same connected meshes · four CPU cores · median and range",
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(DIR / f"comparison.{ext}", dpi=180)
    plt.close(fig)
    svg = DIR / "comparison.svg"
    svg.write_text("\n".join(x.rstrip() for x in svg.read_text().splitlines()) + "\n")
    audits = json.loads((DIR / "audits.json").read_text())
    quality = [
        "| Mesh-contour mode | Exact ray ranks checked | Samples near edges skipped | Rank mismatches checked | Relative per-source area error |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in audits:
        if record["variant"] != "fast-contours":
            continue
        a = record["audit"]
        quality.append(
            f"| {record['input_triangles']:,} | {a['independent_exact_ray_ranks_checked']} | "
            f"{a['boundary_samples_skipped']} | {a['rank_mismatches_away_from_boundaries']} | "
            f"{a['relative_source_area_l1_error']:.3g} |"
        )
    control = next(
        r["audit"]
        for r in audits
        if r["variant"] == "fast-clipping" and r["input_triangles"] == 102400
    )
    control_note = (
        f"The clipping-only fast control had {control['rank_mismatches_away_from_boundaries']} rank mismatches "
        f"among {control['independent_exact_ray_ranks_checked']} checked 100K samples; "
        "its diagnostics are retained in the audit log.\n\n"
    )
    m = json.loads((DIR / "fast-contours-102400/metrics.json").read_text())
    topology, ranks = m["mesh_topology"], m["rank_classification"]
    validation = json.loads((DIR / "validation.json").read_text())
    block = (
        """## Fast mode with mesh contours

`fast-bvh-peel` is an explicitly **finite-precision** mode alongside the exact
reference solvers. It adopts R3's fixed-point orientation and convexity-repair
approach, and uses mesh connectivity to avoid most fragment clipping.

![Fast-mode exploded view: input, acquisition camera, and every extracted layer](results/fast/demo/exploded.png)

"""
        + "\n".join(table)
        + """

These are complete extractions on the same connected meshes, including labelled
triangulation. There is one warm-up, then nine timed fast runs or three exact runs.
The 10K case is within a 30 Hz **extraction** budget; the 100K case is still above
the 100 ms reference. This is a substantial improvement, but not a claim of
end-to-end real-time rendering or a reproduction of R3's benchmark results.

Parsing, projection and connectivity setup are separate from extraction
(fast contour mode: """
        + "; ".join(setup)
        + """), as are optional intersection
preprocessing, validation and file export. Repetitions reuse the projected input;
they do not simulate a moving camera. The four-thread runs use a four-core CPU
quota on an Intel Xeon Platinum 8362 host.

![Measured exact and finite-precision complete-extraction times](results/fast/comparison.png)

### Connectivity and intersection preprocessing

The `--resolved-mesh` contract is that surface intersections have already been
split into mesh edges. The supplied `resolve_intersections` tool performs this
camera-independent preprocessing with **exact** CGAL autorefinement. It preserves
`original_source_id`, `mesh_face_id`, and the original triangle records. It splits
intersections without deleting the intersecting surfaces. Its current importer
requires consistently oriented manifold connectivity, optionally with boundary;
the resulting geometrically non-manifold edges are supported by the fast extractor.
Already-resolved non-manifold input can also be passed directly to the extractor.
Positive-area coincident surfaces remain outside this preprocessing contract.

After projection, the extractor identifies edges by their imported 3D endpoints.
It retains open boundaries, front/back silhouette edges, all non-manifold or
inconsistently oriented edges, and conservative boundaries from camera clipping.
An edge with exactly two consistently oriented, same-facing incident faces is
excluded from the clipping-contour set. Source-triangle edges still define output
provenance and reconstruction; they have not been erased from the mesh.

For a triangle whose projected interior is clear of those contours (with a
four-grid-cell guard), the depth rank is constant. One interior ray classifies its
whole area directly into the corresponding layer. A static source BVH is reused
throughout extraction, with rank masks retiring these whole triangles at the right
layer. Only contour-near residual fragments undergo repeated clipping and index
rebuilding. The `--no-accelerate` option retains a direct-query correctness control.
Without `--resolved-mesh`, the same fast clipping machinery runs without the
constant-rank shortcut; it is retained as the comparison in the table.

"""
        + f"On the 100K example, **{topology['same_facing_internal_edges_ignored']:,}** internal edges are excluded\n"
        + f"from the contour set, leaving **{topology['contour_segments']:,}** contour segments.\n"
        + f"**{ranks['constant_rank_sources']:,} of {m['active_sources']:,}** triangles receive a whole-triangle rank.\n"
        + """
The remaining fragments still use local triangle clipping; full contour regions
are not yet assembled into clipping masks. Complexity depends on contour overlap,
ray candidates, generated fragments and output size. This is not a universal
linear-time bound, and intersection preprocessing can itself have large output.

### Numerical contract and measured error

The default is a **30-bit grid per image-window axis** (`--fast-grid-bits 12..30`).
Integer orientation tests are exact for the grid coordinates; projection, depth
planes and intersection construction use binary64. Constructed vertices are rounded
to the grid and non-convex results are repaired with a convex hull. This is not exact
geometry relative to the original input. Unresolvable depth comparisons use a
documented binary64-scale band with lower source ID winning ties.

To prevent rounded remnants from being peeled repeatedly, the fast path removes
polygons of area at most one grid cell squared and narrow polygons satisfying
`max cross(edge, vertex) <= 2*(abs(edge.dx)+abs(edge.dy))` for any edge. These losses,
convexity repairs and tie breaks are counted in the metrics. Small features or even
an entire tiny late layer can disappear: the `slivers` fixture has two exact layers
and one fast layer. Increase grid resolution or use the exact solver when those
features matter. The exact modes retain their original no-sliver-discard contract.

The fast mesh exporter nodes grid-collinear boundary vertices and preserves source
labels. It does **not** promise exact conforming geometry after rounding. Default
export is `approximate.json` plus OBJ meshes; `--mesh-only` and `--metrics-only`
still compute the triangulation. Fast output is never labelled `exact.json`.

"""
        + "\n".join(quality)
        + "\n\n"
        + control_note
        + """

Ray checks use independent exact camera-space intersections; samples within two
grid cells of source edges are reported separately and skipped. Area error is the
sum of absolute per-source differences in accumulated patch area, divided by total
exact source area. Neither sampling nor area equality is a full region certificate.
Small fixtures additionally compare every source-labelled region with the exact
arrangement using symmetric-difference area and an explicit tolerance budget.
`--validate` runs the large-scene audit; `--verify` adds the small-scene oracle.
An audit that exceeds its checks writes diagnostics and exits with code 3.

"""
        + f"The fast regression report contains {len(validation['records'])} checks, including prefix/thread consistency,\n"
        + """non-manifold junctions, and exact barycentric containment and area conservation
through intersection preprocessing. These numerical tests supplement the unchanged
exact solver tests; passing them does not make the fast mode exact.

### Run and reproduce

```sh
# Optional once per intersecting mesh; create the output directory first.
mkdir -p out
build/resolve_intersections input.json out/resolved.json
build/geometric_peel --scene out/resolved.json --method fast-bvh-peel \\
  --resolved-mesh --layers all --threads 4 --validate --mesh-only --out out/fast

# These generated connected meshes already have no self-intersections.
python scripts/benchmark_fast.py
python scripts/render_exploded.py \\
  --scene results/large/scenes/sculpted-10240.json --layers results/fast/demo \\
  --output results/fast/demo/exploded.png
python scripts/report_fast_results.py
```

The [raw timings](results/fast/summary.csv), [all attempts](results/fast/status.jsonl),
[configuration and source hashes](results/fast/config.json),
[large-scene audits](results/fast/audits.json), and
[regression report](results/fast/validation.json) are retained. The exploded figure
uses the actual fast-mode layer meshes; its [Blender scene](results/fast/demo/exploded.blend)
and large meshes are stored with Git LFS.
"""
    )
    readme = ROOT / "README.md"
    s = readme.read_text()
    start, end = "<!-- FAST_RESULTS_BEGIN -->", "<!-- FAST_RESULTS_END -->"
    if start not in s:
        s = s.replace(
            "<!-- LARGE_RESULTS_BEGIN -->",
            start + "\n" + end + "\n\n<!-- LARGE_RESULTS_BEGIN -->",
            1,
        )
    before, rest = s.split(start, 1)
    _, after = rest.split(end, 1)
    readme.write_text(before + start + "\n" + block + end + after)
    (DIR / "README.md").write_text(
        "# Finite-precision mesh-contour results\n\n"
        "See the root README for measured timings, numerical error, preprocessing and reproduction commands.\n"
        "`summary.csv` reports extraction medians/ranges; `status.jsonl` records every timed attempt.\n"
        "`audits.json` contains independent exact ray checks and area error; `validation.json` covers small-scene oracle checks and preprocessing provenance.\n"
        "`demo/` contains the actual fast-mode layer meshes and exploded render.\n"
        "The mode rounds constructions and drops grid-scale remnants. It is not an exact solver.\n"
    )


if __name__ == "__main__":
    main()
