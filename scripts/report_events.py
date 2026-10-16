#!/usr/bin/env python3
"""Generate the retained CPU/GPU comparison plot and table."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path("results/gpu-events")
new = json.loads((root / "benchmark.json").read_text())["cases"]
old = {c["sha256"]: c for c in json.loads((root / "old-gpu.json").read_text())["cases"]}
cpu = {
    (c["sha256"], c["threads"]): c
    for c in json.loads((root / "cpu/summary.json").read_text())["cases"]
}
rows = [
    "| Input | CPU 1 thread | CPU 4 threads | Old GPU | Contour GPU | CPU 4 / GPU |",
    "| --- | ---: | ---: | ---: | ---: | ---: |",
]
for c in new:
    h = c["sha256"]
    n = c["sources"]
    a = cpu[h, 1]["median_ms"]
    b = cpu[h, 4]["median_ms"]
    d = c["median_ms"]
    previous = f"{old[h]['median_ms']:.1f}" if h in old else "—"
    label = ("Dragon" if "dragon-cache" in c["scene"] else "Sculpted") + f" {n:,}"
    rows.append(
        f"| {label} | {a:.1f} | {b:.1f} | {previous} | **{d:.1f}** | {b / d:.1f}× |"
    )
(root / "table.md").write_text("\n".join(rows) + "\n")
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
dragons = [c for c in new if "dragon-cache" in c["scene"]]
for threads in [1, 4]:
    axes[0].plot(
        [c["sources"] for c in dragons],
        [cpu[c["sha256"], threads]["median_ms"] for c in dragons],
        "o-",
        label=f"CPU fast, {threads} threads",
    )
axes[0].plot(
    [c["sources"] for c in dragons],
    [c["median_ms"] for c in dragons],
    "o-",
    label="Contour GPU, full pipeline",
)
previous = [c for c in dragons if c["sha256"] in old]
axes[0].plot(
    [c["sources"] for c in previous],
    [old[c["sha256"]]["median_ms"] for c in previous],
    "o-",
    label="Old GPU, full pipeline",
)
axes[0].set(
    xscale="log",
    yscale="log",
    xlabel="Resolved dragon triangles",
    ylabel="Median milliseconds",
    title="All layers and conforming mesh",
)
axes[0].grid(True, alpha=0.2)
axes[0].legend(fontsize=8)
stack = json.loads((root / "stacks.json").read_text())["cases"]
axes[1].plot(
    [c["n"] for c in stack], [c["median_ms"] for c in stack], "o-", color="tab:green"
)
axes[1].set(
    xscale="log",
    yscale="log",
    xlabel="Stack surfaces / layers",
    ylabel="Median milliseconds",
    title="Shared-order identical-footprint control",
)
axes[1].grid(True, alpha=0.2)
fig.suptitle(
    "NVIDIA L40 · shared host, 4-core CPU quota · timing scopes differ", fontsize=11
)
fig.savefig(root / "comparison.svg")
svg = root / "comparison.svg"
svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
print("\n".join(rows))
