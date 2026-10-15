#!/usr/bin/env python3
"""Measure CPU thread scaling on an already audited cached dragon level.

Run separately from other benchmarks. Reuse the four-thread result from the
doubling sweep; measure one and two threads with the same executable and input.
This measures extraction, including output meshing, with one warm-up and three
measured repetitions. It does not repeat the separate correctness audit.
"""

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", type=Path, default=Path("results/global/dragon-doubling")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/global/thread-scaling")
    )
    parser.add_argument("--target", type=int, default=4096)
    parser.add_argument("--exe", type=Path, default=Path("build/geometric_peel"))
    args = parser.parse_args()
    config = json.loads((args.results / "config.json").read_text())
    assert config["threads"] == 4 and config["executable_sha256"] == digest(args.exe)
    assert config["host"] == os.uname().nodename
    scene = Path("results/global/dragon-cache") / str(args.target) / "resolved.json"
    scene_hash = digest(scene)
    source = json.loads((args.results / "runs.json").read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output / "config.json",
        {
            "executable_sha256": digest(args.exe),
            "scene_sha256": scene_hash,
            "target_faces": args.target,
            "threads": [1, 2, 4],
            "host": config["host"],
            "warmups": 1,
            "measured_repetitions": 3,
            "process_timeout_seconds": 300,
            "address_space_cap_gib": config["address_space_cap_gib"],
            "four_thread_source": str(args.results),
            "source_runs_sha256": digest(args.results / "runs.json"),
            "algorithm_includes": config["algorithm_includes"],
            "algorithm_excludes": config["algorithm_excludes"],
        },
    )

    def limit():
        resource.setrlimit(
            resource.RLIMIT_AS, (config["address_space_cap_gib"] * 1024**3,) * 2
        )

    records = []
    for method in ("global-arrangement", "bvh-peel"):
        baseline = [
            r
            for r in source
            if r.get("target_faces") == args.target
            and r.get("method") == method
            and r.get("phase") == "timing"
        ][-1]
        assert (
            baseline["status"] == "completed" and baseline["scene_sha256"] == scene_hash
        )
        for threads in (1, 2):
            out = args.output / method / str(threads)
            command = list(
                map(
                    str,
                    [
                        args.exe,
                        "--scene",
                        scene,
                        "--method",
                        method,
                        "--resolved-mesh",
                        "--layers",
                        "all",
                        "--threads",
                        threads,
                        "--metrics-only",
                        "--repeat",
                        4,
                        "--out",
                        out,
                    ],
                )
            )
            start = time.monotonic()
            try:
                run = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=300,
                    env=dict(os.environ, OMP_WAIT_POLICY="PASSIVE"),
                    preexec_fn=limit,
                )
            except subprocess.TimeoutExpired:
                records.append(
                    {
                        "method": method,
                        "threads": threads,
                        "command": command,
                        "status": "timeout",
                        "wall_seconds": time.monotonic() - start,
                    }
                )
                write_json(args.output / "runs.json", records)
                raise
            record = {
                "method": method,
                "threads": threads,
                "command": command,
                "wall_seconds": time.monotonic() - start,
                "returncode": run.returncode,
                "status": "completed" if run.returncode == 0 else "error",
                "stdout": run.stdout,
                "stderr": run.stderr,
            }
            if run.returncode == 0:
                metrics = json.loads((out / "metrics.json").read_text())
                times = metrics["algorithm_repetitions_seconds"][1:]
                record.update(
                    median_seconds=statistics.median(times),
                    min_seconds=min(times),
                    max_seconds=max(times),
                    output_triangles=sum(
                        p["exported_triangles"] for p in metrics["passes"]
                    ),
                    layers=len(metrics["passes"]),
                )
            records.append(record)
            write_json(args.output / "runs.json", records)
            print(json.dumps(record), flush=True)
            run.check_returncode()
        record = dict(baseline, threads=4, reused_from=str(args.results))
        records.append(record)
        write_json(args.output / "runs.json", records)
    lines = [
        "| Exact method | 1 thread | 2 threads | 4 threads | 1→4 speedup |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for method in ("global-arrangement", "bvh-peel"):
        times = {
            r["threads"]: r["median_seconds"] for r in records if r["method"] == method
        }
        lines.append(
            f"| `{method}` | {times[1]:.3f} s | {times[2]:.3f} s | "
            f"{times[4]:.3f} s | {times[1] / times[4]:.2f}× |"
        )
    (args.output / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
