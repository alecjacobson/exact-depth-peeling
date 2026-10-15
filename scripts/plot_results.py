#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("directory", type=Path)
    p.add_argument("--include", type=Path, nargs="*", default=[])
    a = p.parse_args()
    rows = list(csv.DictReader((a.directory / "summary.csv").open()))
    for directory in a.include:
        with (directory / "summary.csv").open() as stream:
            rows.extend(csv.DictReader(stream))
    rows = list(
        {
            tuple(r[k] for k in ("suite", "size", "layers", "method", "order")): r
            for r in rows
        }.values()
    )
    for suite in ("foreground-cover", "no-foreground", "stacked"):
        if not any(row["suite"] == suite for row in rows):
            continue
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
        specs = [
            ("median_seconds", "Algorithm time (seconds)", 1),
            (
                "materialized_result_boundary_vertices",
                "Materialized Boolean-result boundary vertices",
                1,
            ),
            ("layer1_triangles", "Layer 1 exported triangles", 1),
            ("median_seconds", "All-layer algorithm time (seconds)", "all"),
        ]
        for ax, (field, label, k) in zip(axes.flat, specs):
            for method, order in [
                ("visible-peel", "near-first"),
                ("visible-peel", "random"),
                ("visible-peel", "reverse"),
                ("envelope-peel", "input"),
                ("arrangement", "input"),
            ]:
                rs = sorted(
                    [
                        r
                        for r in rows
                        if r["suite"] == suite
                        and r["layers"] == str(k)
                        and r["method"] == method
                        and r["order"] == order
                    ],
                    key=lambda r: int(r["size"]),
                )
                if not rs:
                    continue
                ax.plot(
                    [int(r["size"]) for r in rs],
                    [float(r[field]) for r in rs],
                    marker="o",
                    label=method + "/" + order,
                )
                if field == "median_seconds":
                    ax.fill_between(
                        [int(r["size"]) for r in rs],
                        [float(r["min_seconds"]) for r in rs],
                        [float(r["max_seconds"]) for r in rs],
                        alpha=0.12,
                    )
            ax.set(
                xlabel="g" if suite != "stacked" else "source count",
                ylabel=label,
                title=f"K={k}",
            )
            ax.set_xscale("log", base=2)
            ax.grid(alpha=0.25)
            if field == "median_seconds":
                ax.set_yscale("log")
            else:
                ax.set_ylim(bottom=0)
        axes[0, 0].legend(fontsize=7)
        if suite == "stacked":
            fig.suptitle(
                "Identical-footprint diagnostic: prototype overhead\nSort once, then emit layers; this is not representative mesh data",
                fontsize=12,
            )
        else:
            fig.suptitle(
                suite
                + " — successful measurements only; see source status logs for caps/timeouts"
            )
        for ext in ("png", "svg", "pdf"):
            fig.savefig(a.directory / f"{suite}.{ext}", dpi=160)
        plt.close(fig)

    cover_rows = [
        r
        for r in rows
        if r["suite"] == "foreground-cover"
        and r["layers"] == "1"
        and r["method"] == "visible-peel"
    ]
    if cover_rows:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
        for ax, field, title in zip(
            axes,
            ["max_visible_map_boundary_vertices", "polygon_intersections"],
            [
                "Maximum visible-map boundary vertices",
                "Visibility polygon-intersection calls",
            ],
        ):
            for order in ("near-first", "input", "random", "reverse"):
                rs = sorted(
                    [r for r in cover_rows if r["order"] == order],
                    key=lambda r: int(r["size"]),
                )
                ax.plot(
                    [int(r["size"]) for r in rs],
                    [int(r[field]) for r in rs],
                    marker="o",
                    label=order,
                )
            ax.set(xlabel="g", ylabel=title, title=title)
            ax.set_xscale("log", base=2)
            ax.set_yscale("symlog", linthresh=1)
            ax.grid(alpha=0.25)
        axes[0].legend()
        fig.suptitle(
            "Foreground cover, K=1: identical final output, different intermediate work"
        )
        for ext in ("png", "svg", "pdf"):
            fig.savefig(a.directory / f"insertion-order.{ext}", dpi=160)
        plt.close(fig)

    practical = [
        r
        for r in rows
        if r["suite"] in ("perspective-torus", "perspective-box")
        and r["layers"] in ("1", "2", "4")
    ]
    if practical:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
        for ax, suite in zip(axes, ("perspective-torus", "perspective-box")):
            scene_rows = [r for r in practical if r["suite"] == suite]
            for method in ("visible-peel", "envelope-peel"):
                rs = sorted(
                    [r for r in scene_rows if r["method"] == method],
                    key=lambda r: int(r["layers"]),
                )
                y = [float(r["median_seconds"]) for r in rs]
                ax.errorbar(
                    [int(r["layers"]) for r in rs],
                    y,
                    yerr=[
                        [v - float(r["min_seconds"]) for v, r in zip(y, rs)],
                        [float(r["max_seconds"]) - v for v, r in zip(y, rs)],
                    ],
                    marker="o",
                    capsize=3,
                    label=method,
                )
            faces = scene_rows[0]["input_sources"] if scene_rows else "?"
            ax.set(
                title=f"{suite}: {faces} source faces",
                xlabel="Requested K",
                ylabel="Algorithm seconds",
                xticks=[1, 2, 4],
                ylim=(0, None),
            )
            ax.grid(alpha=0.25)
            ax.legend()
        fig.suptitle(
            "Connected perspective scenes: median and range of three repetitions\nEmpty residuals terminate early; these are controlled synthetic workloads",
            fontsize=12,
        )
        for ext in ("png", "svg", "pdf"):
            fig.savefig(a.directory / f"connected-perspective.{ext}", dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    main()
