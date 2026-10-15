#!/usr/bin/env python3
"""Capped, sequential release benchmarks; warm-up + >=3 measured repetitions."""

import argparse
import csv
import hashlib
import itertools
import json
import resource
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

from generate_fixtures import (
    ROOT,
    cover,
    disjoint,
    fixtures,
    mesh,
    nested_mesh,
    perforated,
    perspective_view,
    stacked,
    torus,
    write,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exe", type=Path, default=ROOT / "build/geometric_peel")
    p.add_argument(
        "--suite",
        choices=["foreground-cover", "stacked", "other", "broader", "practical", "all"],
        default="all",
    )
    p.add_argument("--output", type=Path, default=ROOT / "results/benchmark")
    p.add_argument("--sizes", type=int, nargs="+", default=[2, 4, 8, 16, 32, 64])
    p.add_argument("--repetitions", type=int, default=3)
    p.add_argument("--timeout", type=float, default=5)
    p.add_argument("--budget", type=float, default=600)
    p.add_argument("--memory-mib", type=int, default=2048)
    p.add_argument(
        "--methods",
        nargs="+",
        choices=["visible-peel", "envelope-peel", "arrangement"],
        default=["visible-peel", "envelope-peel", "arrangement"],
    )
    p.add_argument(
        "--orders",
        nargs="+",
        choices=["near-first", "input", "random", "reverse"],
        default=["near-first", "input", "random", "reverse"],
    )
    p.add_argument(
        "--layers",
        nargs="+",
        choices=["1", "2", "4", "all"],
        default=["1", "2", "4", "all"],
    )
    p.add_argument("--only-workloads", nargs="+")
    a = p.parse_args()
    if a.repetitions < 3:
        p.error("At least three measured repetitions are required")
    a.output.mkdir(parents=True, exist_ok=True)
    exe = str(a.exe.resolve())
    start = time.monotonic()
    rows = []
    raw = (a.output / "runs.jsonl").open("w")
    statuses = (a.output / "status.jsonl").open("w")

    def limits():
        resource.setrlimit(resource.RLIMIT_AS, (a.memory_mib * 1024**2,) * 2)

    workloads = []
    if a.suite in ("foreground-cover", "all"):
        for g in a.sizes:
            for fg in (True, False):
                workloads.append(
                    ("foreground-cover" if fg else "no-foreground", g, cover(g, fg))
                )
    if a.suite in ("stacked", "all"):
        workloads += [("stacked", n, stacked(n)) for n in a.sizes]
    if a.suite in ("other", "all"):
        workloads += [("disjoint", n, disjoint(n)) for n in (8, 32, 64)]
        workloads += [
            (name, 0, fixtures()[name])
            for name in ("holes", "cyclic", "crossing", "random_07", "perspective_near")
        ]
        workloads += [("connected-mesh", n, mesh(n)) for n in (2, 10)]
    if a.suite == "broader":
        workloads += [("perforated", n, perforated(n)) for n in (4, 8)]
        workloads += [("nested-mesh", n, nested_mesh(n)) for n in (2, 7)]
    if a.suite == "practical":
        workloads += [
            ("perspective-torus", 4, torus(4, 3)),
            ("perspective-box", 10, perspective_view(mesh(10), 3)),
        ]
    if a.only_workloads:
        workloads = [w for w in workloads if w[0] in a.only_workloads]
        if not workloads:
            p.error("No workloads match --only-workloads")
    for name, size, data in workloads:
        write(a.output / "scenes" / f"{name}-{size}.json", data)
    config = {
        "executable_sha256": hashlib.sha256(a.exe.read_bytes()).hexdigest(),
        "suite": a.suite,
        "methods": a.methods,
        "orders": a.orders,
        "layers": a.layers,
        "only_workloads": a.only_workloads,
        "sizes": a.sizes,
        "repetitions": a.repetitions,
        "timeout_seconds": a.timeout,
        "total_budget_seconds": a.budget,
        "memory_cap_mib": a.memory_mib,
        "seed": 1729,
        "cpu": next(
            (
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
            "unknown",
        ),
        "dependencies": json.loads((ROOT / "dependencies.json").read_text()),
        "warmups": 1,
        "timing": "algorithm excludes validation, serialization and projection/preflight; latter separately recorded",
        "large_scene_validation": "exact invariants without arrangement oracle; all validations in warmup only",
    }
    write(a.output / "config.json", config)
    with tempfile.TemporaryDirectory(prefix="peel-bench-") as tmp:
        dest = Path(tmp) / "out"
        for name, size, data in workloads:
            path = a.output / "scenes" / f"{name}-{size}.json"
            n = len(data["triangles"])
            # Run all requested prefixes, including K exceeding maximum depth.
            methods = [
                (m, o)
                for m in a.methods
                for o in (a.orders if m == "visible-peel" else ["input"])
            ]
            requested_layers = [int(k) if k != "all" else k for k in a.layers]
            for k, (method, order) in itertools.product(requested_layers, methods):
                key = {
                    "suite": name,
                    "size": size,
                    "input_sources": n,
                    "layers": k,
                    "method": method,
                    "order": order,
                }
                status = dict(key)
                if method == "arrangement" and n > 64:
                    status["status"] = "oracle_cap"
                    statuses.write(json.dumps(status) + "\n")
                    continue
                if time.monotonic() - start > a.budget:
                    status["status"] = "total_budget_exhausted"
                    statuses.write(json.dumps(status) + "\n")
                    continue
                cmd = [
                    exe,
                    "--scene",
                    str(path),
                    "--method",
                    method,
                    "--order",
                    order,
                    "--layers",
                    str(k),
                    "--out",
                    str(dest),
                ]
                measurements = []
                validation_seconds = 0
                for rep in range(-1, a.repetitions):
                    args = cmd + (
                        ["--verify" if n <= 64 else "--validate"] if rep == -1 else []
                    )
                    try:
                        r = subprocess.run(
                            args,
                            check=False,
                            capture_output=True,
                            text=True,
                            timeout=a.timeout,
                            preexec_fn=limits,
                        )
                    except subprocess.TimeoutExpired:
                        status.update(
                            status="timeout",
                            phase="warmup_validation" if rep == -1 else "measurement",
                            repetition=rep,
                        )
                        break
                    if r.returncode:
                        status.update(
                            status="error",
                            phase="warmup_validation" if rep == -1 else "measurement",
                            stderr=r.stderr[-2000:],
                        )
                        break
                    m = json.loads((dest / "metrics.json").read_text())
                    if m["build_type"] != "Release":
                        raise RuntimeError("Benchmark requires release executable")
                    if rep == -1:
                        validation_seconds = m["validation_seconds"]
                        continue
                    raw.write(json.dumps(dict(key, repetition=rep, metrics=m)) + "\n")
                    raw.flush()
                    measurements.append(m)
                else:
                    status["status"] = "ok"
                status["completed_repetitions"] = len(measurements)
                statuses.write(json.dumps(status) + "\n")
                statuses.flush()
                if status["status"] != "ok":
                    print(
                        f"{name} {size} K={k} {method}/{order}: {status['status']}",
                        flush=True,
                    )
                    continue
                times = [m["algorithm_seconds"] for m in measurements]
                m = measurements[-1]
                passes = m["passes"]
                row = dict(
                    key,
                    status="ok",
                    median_seconds=statistics.median(times),
                    min_seconds=min(times),
                    max_seconds=max(times),
                    setup_median_seconds=statistics.median(
                        x["projection_preflight_seconds"] for x in measurements
                    ),
                    peak_rss_kib=max(x["peak_rss_kib"] for x in measurements),
                    warmup_validation_seconds=validation_seconds,
                    output_triangles=sum(x["exported_triangles"] for x in passes),
                    layer1_boundary_vertices=passes[0]["output"]["boundary_vertices"]
                    if passes
                    else 0,
                    layer1_triangles=passes[0]["exported_triangles"] if passes else 0,
                )
                for field in (
                    "candidate_tests",
                    "box_rejections",
                    "complete_occlusion_rejections",
                    "polygon_intersections",
                    "polygon_differences",
                    "depth_halfplane_clips",
                    "adapter_triangles",
                    "materialized_result_boundary_vertices",
                    "arrangement_vertices",
                    "arrangement_edges",
                ):
                    row[field] = (
                        sum(x[field] for x in passes) + m["eager_arrangement"][field]
                    )
                row["max_visible_map_boundary_vertices"] = max(
                    (x["max_visible_map_boundary_vertices"] for x in passes), default=0
                )
                row["raw_arrangement_triangles"] = m["eager_arrangement"][
                    "raw_triangles"
                ]
                rows.append(row)
                with (a.output / "summary.csv").open("w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                print(
                    f"{name} {size} K={k} {method}/{order}: {row['median_seconds']:.4f}s",
                    flush=True,
                )
    raw.close()
    statuses.close()
    write(
        a.output / "completion.json",
        {
            "elapsed_seconds": time.monotonic() - start,
            "successful_configurations": len(rows),
        },
    )


if __name__ == "__main__":
    main()
