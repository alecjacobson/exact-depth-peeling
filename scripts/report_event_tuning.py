#!/usr/bin/env python3
"""Plot the fresh event-route performance comparison."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    root = Path("results/gpu-events-tuning")
    baseline = json.loads((root / "baseline.json").read_text())["cases"][:7]
    candidate = json.loads((root / "benchmark.json").read_text())["cases"][:7]
    x = [c["sources"] for c in baseline]
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    ax.plot(
        x,
        [c["median_ms"] for c in baseline],
        "o-",
        label="Previous event route (0639fe2)",
        color="#ad552d",
    )
    ax.plot(
        x,
        [c["median_ms"] for c in candidate],
        "o-",
        label="Flat frontier + parallel ranks",
        color="#176c8c",
    )
    ax.axhline(
        1000 / 30,
        color="#666666",
        linestyle="--",
        linewidth=1,
        label="30 Hz extraction budget",
    )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xticks(x, ["1K", "2K", "4K", "8K", "16K", "32K", "64K"])
    ax.set_yticks([2, 4, 8, 16, 32, 64, 128], ["2", "4", "8", "16", "32", "64", "128"])
    ax.set_xlabel("Dragon decimation target (resolved inputs)")
    ax.set_ylabel("All-layer extraction and conforming mesh (ms)")
    ax.set_title("Warp GPU event extraction · NVIDIA L40 · shared host")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.savefig(root / "comparison.svg", metadata={"Date": None})
    plt.close(fig)
    path = root / "comparison.svg"
    path.write_text(
        "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
    )


if __name__ == "__main__":
    main()
