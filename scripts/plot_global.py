#!/usr/bin/env python3
"""Plot measured exact global/peeling trends; retain censored runs in the table."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=Path("results/global/dragon"))
    a = p.parse_args()
    rows = list(csv.DictReader((a.results / "summary.csv").open()))
    colors = {"global-arrangement": "#a34a25", "bvh-peel": "#216c9a"}
    labels = {
        "global-arrangement": "Global contour arrangement",
        "bvh-peel": "Contour BVH peeling",
    }
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), layout="constrained")
    axes = axes.ravel()
    for method, color in colors.items():
        selected = sorted(
            (r for r in rows if r["method"] == method and r["status"] == "completed"),
            key=lambda r: int(r["input_faces"]),
        )
        x = [int(r["input_faces"]) for r in selected]
        y = [float(r["median_seconds"]) for r in selected]
        low = [m - float(r["min_seconds"]) for m, r in zip(y, selected)]
        high = [float(r["max_seconds"]) - m for m, r in zip(y, selected)]
        axes[0].errorbar(
            x,
            y,
            yerr=[low, high],
            marker="o",
            capsize=3,
            color=color,
            label=labels[method],
        )
        axes[2].plot(
            x,
            [int(r["output_triangles"]) for r in selected],
            marker="o",
            color=color,
            label=labels[method],
        )
        core = []
        meshing = []
        for row in selected:
            metrics = json.loads(
                (
                    a.results
                    / row["target_faces"]
                    / f"{method}-timing"
                    / "metrics.json"
                ).read_text()
            )
            if method == "global-arrangement":
                g = metrics["global_arrangement"]
                core.append(
                    sum(
                        g[k]
                        for k in (
                            "boundary_seconds",
                            "arrangement_seconds",
                            "labeling_seconds",
                        )
                    )
                )
            else:
                core.append(
                    sum(p["visibility_seconds"] for p in metrics["passes"])
                    + metrics["rank_classification"]["classification_seconds"]
                )
            meshing.append(sum(p["triangulation_seconds"] for p in metrics["passes"]))
        axes[1].plot(x, core, marker="o", color=color)
        axes[3].plot(x, meshing, marker="o", color=color)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Actual triangles after exact intersection splitting")
        ax.grid(True, which="both", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("All-layer extraction + triangulation (seconds)")
    axes[0].set_title("Median and range of three repetitions")
    axes[0].legend(fontsize=9)
    axes[1].set_ylabel("Subdivision / visibility phases (seconds)")
    axes[1].set_title("Before output meshing · last repetition")
    axes[2].set_ylabel("Exported triangles")
    axes[2].set_title("Exact labelled geometry; different triangulations")
    axes[3].set_ylabel("Conforming output meshing (seconds)")
    axes[3].set_title("Same mesher, different exact fragments · last repetition")
    fig.suptitle("Cached dragon sequence · fixed camera · four CPU threads")
    fig.savefig(a.results / "trends.svg")
    fig.savefig(a.results / "trends.png", dpi=160)
    lines = [
        "| Decimation target → resolved faces | Global exact | Peeling exact | Global / peeling |",
        "| --- | ---: | ---: | ---: |",
    ]
    for target in sorted({int(r["target_faces"]) for r in rows}):
        group = {r["method"]: r for r in rows if int(r["target_faces"]) == target}
        count = next(iter(group.values()))["input_faces"]

        def value(method, group=group):
            r = group.get(method)
            return (
                f"{float(r['median_seconds']):.3f} s"
                if r and r["status"] == "completed"
                else (r["status"] if r else "not run")
            )

        ratio = "—"
        if all(group.get(m, {}).get("status") == "completed" for m in colors):
            ratio = f"{float(group['global-arrangement']['median_seconds']) / float(group['bvh-peel']['median_seconds']):.2f}×"
        lines.append(
            f"| {target:,} → {int(count):,} | {value('global-arrangement')} | {value('bvh-peel')} | {ratio} |"
        )
    (a.results / "table.md").write_text("\n".join(lines) + "\n")
    config = json.loads((a.results / "config.json").read_text())
    print("\n".join(lines))
    print(
        f"Initial process cap: {config['process_timeout_seconds']} seconds. Retry caps are in runs.json; no timeout is a single-extraction bound."
    )


if __name__ == "__main__":
    main()
