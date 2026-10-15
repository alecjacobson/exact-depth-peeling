#!/usr/bin/env python3
"""Plot paired timing distributions and measured stages from the CPU experiments."""

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input", type=Path, default=Path("results/cpu-transfer/performance")
    )
    p.add_argument("--output", type=Path, default=Path("results/cpu-transfer"))
    a = p.parse_args()
    runs = json.loads((a.input / "runs.json").read_text())
    scenes = list(dict.fromkeys(r["scene"] for r in runs))
    variants = ["before", "control", "pruned", "merged"]
    labels = ["Before", "Instrumented control", "Rank pruning", "Pruning + merging"]
    colors = ["#64748b", "#a8b3c4", "#197eaa", "#d47724"]
    fig, axes = plt.subplots(
        1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.25, 1]}
    )
    ax = axes[0]
    x = np.arange(len(scenes))
    for i, (variant, label, color) in enumerate(zip(variants, labels, colors)):
        values = [
            [
                t * 1000
                for r in runs
                if r["scene"] == s and r["variant"] == variant
                for t in r["times"]
            ]
            for s in scenes
        ]
        medians = np.array([statistics.median(v) for v in values])
        low = medians - np.array([min(v) for v in values])
        high = np.array([max(v) for v in values]) - medians
        ax.errorbar(
            x + (i - 1.5) * 0.16,
            medians,
            yerr=[low, high],
            fmt="o",
            color=color,
            label=label,
            capsize=2,
            markersize=4,
        )
    ax.set_yscale("log")
    names = []
    for scene in scenes:
        count = next(r["input_sources"] for r in runs if r["scene"] == scene)
        names.append(
            ("Dragon" if scene.startswith("dragon") else "Torus") + f"\n{count:,}"
        )
    ax.set_xticks(x, names)
    ax.set_ylabel("All layers + meshing (ms, log scale)")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(fontsize=8)
    ax.set_title("Median and min–max of measured repetitions")
    ax = axes[1]
    scene = "dragon-65536"
    stages = [
        ("query_and_clip_seconds", "Query / clipping", "#197eaa"),
        ("residual_bounds_seconds", "Bounds", "#9dcce1"),
        ("residual_bvh_seconds", "Residual BVH", "#54a48c"),
        ("work_setup_seconds", "Work buffers", "#a8b3c4"),
        ("fragment_collection_seconds", "Collect fragments", "#64748b"),
        ("residual_merge_seconds", "Merge residuals", "#d47724"),
        ("boundary_index_seconds", "Boundary index", "#9976b5"),
        ("boundary_node_and_emit_seconds", "Node / triangulate", "#c9a9da"),
        ("face_collection_seconds", "Collect faces", "#cfa474"),
    ]
    bottom = np.zeros(3)
    for key, label, color in stages:
        values = [
            statistics.median(
                r["phases"][key]
                for r in runs
                if r["scene"] == scene and r["variant"] == v
            )
            for v in variants[1:]
        ]
        ax.bar(np.arange(3), values, bottom=bottom, label=label, color=color)
        bottom += values
    ax.set_xticks(np.arange(3), ["Control", "Pruned", "+ merged"])
    ax.set_ylabel("Seconds (sum of phase medians)")
    ax.set_title("64K dragon: last-repetition phase samples")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.suptitle(
        "Fast BVH, four threads; concurrent background load, not isolated scaling"
    )
    fig.tight_layout()
    a.output.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.output / "profile.png", dpi=160)
    svg = a.output / "profile.svg"
    fig.savefig(svg)
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n"
    )


if __name__ == "__main__":
    main()
