#!/usr/bin/env python3
"""Plot retained, separately captured Warp stage diagnostics."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    root = Path("results/gpu-events-profile")
    groups = [
        ("Preparation", ["prepare"], "#718096"),
        ("Subdivision", ["subdivide"], "#d69e2e"),
        ("Connectivity", ["rank_connectivity"], "#805ad5"),
        ("Primary rays", ["rank_ray_primary"], "#c53030"),
        ("Heavy rays", ["rank_ray_heavy"], "#ed8936"),
        (
            "Other rank work",
            [
                "rank_samples",
                "rank_query_sort",
                "rank_group_certify",
                "rank_hit_sort_scan",
                "rank_assign",
            ],
            "#319795",
        ),
        ("Boundary setup/BVH", ["mesh_vertices_scan", "mesh_bounds_bvh"], "#63b3ed"),
        ("Noding/emission/lift", ["mesh_node_emit_lift"], "#2b6cb0"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    for ax, case, title in zip(
        axes, ["64k", "full"], ["65,544-face dragon", "721,510-face dragon"]
    ):
        for row, version in enumerate(["before", "after"]):
            data = json.loads((root / f"{version}-{case}.json").read_text())[
                "median_ms"
            ]
            left = 0.0
            for label, keys, color in groups:
                width = sum(data[k] for k in keys)
                ax.barh(
                    row,
                    width,
                    left=left,
                    color=color,
                    label=label if row == 0 else None,
                )
                left += width
            ax.text(left + left * 0.01, row, f"{left:.1f}", va="center", fontsize=9)
        ax.set_yticks([0, 1], ["Before", "Flat sources"])
        ax.invert_yaxis()
        ax.set_xlabel("Sum of separately captured stage medians (ms)")
        ax.set_title(title)
        ax.set_xlim(0, ax.get_xlim()[1] * 1.08)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Ray containment storage removes most primary-ray cost")
    fig.tight_layout(rect=(0, 0.19, 1, 0.94))
    target = root / "stages.svg"
    fig.savefig(target, metadata={"Date": None})
    target.write_text(
        "\n".join(line.rstrip() for line in target.read_text().splitlines()) + "\n"
    )


if __name__ == "__main__":
    main()
