#!/usr/bin/env python3
"""Paired CPU experiments under current load; run only one solver at a time."""

import argparse
import json
import os
import resource
import statistics
import subprocess
import time
from pathlib import Path

from benchmark_global import digest, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-exe", type=Path, required=True)
    p.add_argument("--exe", type=Path, default=Path("build/geometric_peel"))
    p.add_argument(
        "--output", type=Path, default=Path("results/cpu-transfer/performance")
    )
    p.add_argument("--targets", type=int, nargs="+", default=[1024, 4096, 16384, 65536])
    p.add_argument("--batches", type=int, default=3)
    p.add_argument("--repetitions", type=int, default=4)
    p.add_argument("--exact", action="store_true")
    p.add_argument("--torus", action="store_true")
    a = p.parse_args()
    if a.batches < 1 or a.repetitions < 2:
        p.error("Require at least one batch and two repetitions (one warm-up).")
    a.output.mkdir(parents=True, exist_ok=True)
    scenes = {
        f"dragon-{n}": Path(f"results/global/dragon-cache/{n}/resolved.json")
        for n in a.targets
    }
    if a.torus:
        scenes["torus-102400"] = Path("results/large/scenes/sculpted-102400.json")
    variants = {
        "before": (a.baseline_exe, []),
        "control": (a.exe, ["--no-rank-pruning"]),
        "pruned": (a.exe, []),
        "merged": (a.exe, ["--merge-residuals"]),
    }
    cgroup = Path("/sys/fs/cgroup") / Path(
        "/proc/self/cgroup"
    ).read_text().strip().split("::", 1)[1].lstrip("/")

    def stat():
        return {
            k: int(v)
            for k, v in (
                line.split() for line in (cgroup / "cpu.stat").read_text().splitlines()
            )
        }

    def cpu():
        r = resource.getrusage(resource.RUSAGE_CHILDREN)
        return r.ru_utime + r.ru_stime

    def limit():
        resource.setrlimit(resource.RLIMIT_AS, (24 * 1024**3,) * 2)

    config = {
        "before_sha256": digest(a.baseline_exe),
        "after_sha256": digest(a.exe),
        "scenes": {n: {"path": str(s), "sha256": digest(s)} for n, s in scenes.items()},
        "threads": 4,
        "warmups_per_process": 1,
        "repetitions": a.repetitions,
        "batches": a.batches,
        "exact": a.exact,
        "background_load_controlled": False,
        "cpu_max": (cgroup / "cpu.max").read_text().strip(),
        "timing_includes": ["classification", "all-layer extraction", "meshing"],
        "timing_excludes": ["import/projection/topology", "validation", "export"],
        "phase_metrics": "last repetition of each process",
        "max_ratio": 1.20,
        "timeout_seconds": 600,
        "address_space_cap_gib": 24,
    }
    write_json(a.output / "config.json", config)
    records = []
    for scene_name, scene in scenes.items():
        for batch in range(a.batches):
            # Alternate direction to reduce systematic bias from changing background load.
            names = list(variants) if batch % 2 == 0 else list(reversed(variants))
            for variant in names:
                exe, flags = variants[variant]
                out = a.output / scene_name / str(batch) / variant
                out.mkdir(parents=True, exist_ok=True)
                cmd = [
                    str(exe),
                    "--scene",
                    str(scene),
                    "--method",
                    "bvh-peel" if a.exact else "fast-bvh-peel",
                    "--resolved-mesh",
                    "--layers",
                    "all",
                    "--threads",
                    "4",
                    "--repeat",
                    str(a.repetitions),
                    "--metrics-only",
                    "--out",
                    str(out),
                ] + flags
                t, c, st = time.monotonic(), cpu(), stat()
                try:
                    r = subprocess.run(
                        cmd,
                        capture_output=True,
                        check=False,
                        text=True,
                        timeout=600,
                        preexec_fn=limit,
                        env=dict(os.environ, OMP_WAIT_POLICY="PASSIVE"),
                    )
                except subprocess.TimeoutExpired:
                    write_json(
                        out / "failure.json",
                        {"command": cmd, "status": "timeout", "timeout_seconds": 600},
                    )
                    raise SystemExit("Benchmark timed out; see failure.json") from None
                wall, used, end = time.monotonic() - t, cpu() - c, stat()
                if r.returncode:
                    write_json(
                        out / "failure.json",
                        {
                            "command": cmd,
                            "stderr": r.stderr,
                            "returncode": r.returncode,
                        },
                    )
                    raise RuntimeError(r.stderr)
                m = json.loads((out / "metrics.json").read_text())
                phases = {}
                for layer in m["passes"]:
                    for k, v in layer.get("phases", {}).items():
                        if isinstance(v, dict):
                            for mk, mv in v.items():
                                phases[mk] = phases.get(mk, 0) + mv
                        else:
                            phases[k] = phases.get(k, 0) + v
                rec = {
                    "scene": scene_name,
                    "batch": batch,
                    "variant": variant,
                    "command": cmd,
                    "wall_seconds": wall,
                    "process_cpu_seconds": used,
                    "estimated_other_cpu_equivalents": max(
                        0, (end["usage_usec"] - st["usage_usec"]) / 1e6 - used
                    )
                    / wall,
                    "times": m["algorithm_repetitions_seconds"][1:],
                    "phases": phases,
                    "layers": len(m["passes"]),
                    "input_sources": m["input_sources"],
                    "candidates": sum(x["candidate_tests"] for x in m["passes"]),
                    "boxes": sum(x["index_box_tests"] for x in m["passes"]),
                    "triangles": sum(x["exported_triangles"] for x in m["passes"]),
                    "peak_residual": max(
                        x["input_residual"]["convex_patches"] for x in m["passes"]
                    ),
                }
                records.append(rec)
                write_json(a.output / "runs.json", records)
                print(
                    scene_name,
                    batch,
                    variant,
                    round(statistics.median(rec["times"]), 4),
                    flush=True,
                )
    summary = []
    for scene in scenes:
        by_variant = {
            v: [r for r in records if r["scene"] == scene and r["variant"] == v]
            for v in variants
        }
        before = statistics.median(t for r in by_variant["before"] for t in r["times"])
        for v, rows in by_variant.items():
            times = [t for r in rows for t in r["times"]]
            median = statistics.median(times)
            row = dict(
                scene=scene,
                variant=v,
                median_seconds=median,
                min_seconds=min(times),
                max_seconds=max(times),
                ratio_to_before=median / before,
                timing_pass=median / before <= 1.20,
                **{
                    k: rows[0][k]
                    for k in [
                        "layers",
                        "candidates",
                        "boxes",
                        "triangles",
                        "peak_residual",
                    ]
                },
            )
            if v in ["control", "pruned"]:
                row["work_counts_preserved"] = all(
                    row[k] == by_variant["before"][0][k]
                    for k in ["layers", "candidates", "triangles", "peak_residual"]
                )
                assert row["work_counts_preserved"], row
            summary.append(row)
    write_json(a.output / "summary.json", summary)
    # Merging is an opt-in experiment, not an enabled-by-default performance gate.
    if any(
        not r["timing_pass"] for r in summary if r["variant"] in ["control", "pruned"]
    ):
        raise SystemExit(
            "Performance gate failed; retain and investigate these measurements."
        )


if __name__ == "__main__":
    main()
