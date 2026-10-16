#!/usr/bin/env python3
"""Plot retained full-pipeline connectivity timings against a fresh baseline."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    root = Path("results/gpu-events-components")
    before = {
        c["sha256"]: c
        for c in json.loads((root / "baseline.json").read_text())["cases"]
    }
    after = json.loads((root / "benchmark.json").read_text())["cases"]
    dragons = [c for c in after if "dragon-cache" in c["scene"]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    ax = axes[0]
    for label, cases, color in (
        ("Previous event solver", [before[c["sha256"]] for c in dragons], "#7a7a7a"),
        ("Certified connectivity", dragons, "#0072b2"),
    ):
        x = [c["sources"] for c in cases]
        y = [c["median_ms"] for c in cases]
        ax.plot(x, y, "o-", label=label, color=color)
        ax.fill_between(
            x,
            [min(c["milliseconds"]) for c in cases],
            [max(c["milliseconds"]) for c in cases],
            alpha=0.15,
            color=color,
        )
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="Resolved dragon faces",
        ylabel="Full graph replay (ms)",
    )
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    ax = axes[1]
    cases = [dragons[-1], after[-1]]
    ax.bar(
        [0, 1],
        [before[c["sha256"]]["output"]["ray_candidate_tests"] for c in cases],
        width=0.35,
        label="Previous",
        color="#7a7a7a",
    )
    ax.bar(
        [0.38, 1.38],
        [c["output"]["ray_candidate_tests"] for c in cases],
        width=0.35,
        label="Connectivity",
        color="#0072b2",
    )
    ax.set(
        xticks=[0.19, 1.19],
        xticklabels=["64K dragon", "102K sculpted"],
        ylabel="Rank-ray candidate tests",
    )
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.legend(fontsize=8)
    fig.suptitle("Fewer rank queries do not guarantee faster full extraction")
    target = root / "comparison.svg"
    fig.savefig(target)
    target.write_text(
        "\n".join(line.rstrip() for line in target.read_text().splitlines()) + "\n"
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
