#!/usr/bin/env python3
"""Plot fast BVH timing ranges under the recorded shared CPU quota."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", type=Path, default=Path("results/fast/thread-scaling")
    )
    args = parser.parse_args()
    rows = json.loads((args.results / "runs.json").read_text())
    scenes = list(dict.fromkeys(r["scene"] for r in rows))
    fig, axes = plt.subplots(
        1, len(scenes), figsize=(12, 4), layout="constrained", squeeze=False
    )
    for ax, scene in zip(axes[0], scenes):
        data = sorted(
            (r for r in rows if r["scene"] == scene and r["status"] == "completed"),
            key=lambda r: r["threads"],
        )
        x = [r["threads"] for r in data]
        y = [r["median_seconds"] * 1000 for r in data]
        low = [(r["median_seconds"] - r["min_seconds"]) * 1000 for r in data]
        high = [(r["max_seconds"] - r["median_seconds"]) * 1000 for r in data]
        ax.errorbar(x, y, yerr=[low, high], marker="o", capsize=4, color="#216c9a")
        ax.axvline(4, color="#a34a25", linestyle="--", alpha=0.6)
        ax.set_xscale("log", base=2)
        ax.set_xticks(x, [str(n) for n in x])
        ax.set_xlabel("Worker threads (4-core CPU quota)")
        ax.set_ylabel("Complete extraction + triangulation (ms)")
        ax.set_title(f"{scene} · {data[0]['input_triangles']:,} triangles")
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Fast BVH · concurrent background load\nMedian and range of nine repetitions after one warm-up"
    )
    fig.savefig(args.results / "trends.png", dpi=160)
    path = args.results / "trends.svg"
    fig.savefig(path)
    path.write_text(
        "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
    )


if __name__ == "__main__":
    main()
