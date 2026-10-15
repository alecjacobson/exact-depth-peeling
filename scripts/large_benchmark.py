#!/usr/bin/env python3
"""Sequential complete-layer benchmarks; one warm-up and three measured extractions."""

import argparse
import csv
import hashlib
import json
import os
import platform
import resource
import statistics
import subprocess
import time
from pathlib import Path

from large_mesh_demo import make_mesh

ROOT = Path(__file__).resolve().parents[1]
SIZES = {768: (32, 12), 10240: (128, 40), 40960: (256, 80), 102400: (400, 128)}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exe", type=Path, default=ROOT / "build/geometric_peel")
    p.add_argument("--output", type=Path, default=ROOT / "results/large")
    p.add_argument("--sizes", nargs="+", type=int, default=list(SIZES))
    p.add_argument("--methods", nargs="+", default=["bvh-peel", "vbvh-peel"])
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--timeout", type=float, default=180)
    p.add_argument("--memory-mib", type=int, default=4096)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / "scenes").mkdir(exist_ok=True)
    cgroup = Path("/sys/fs/cgroup") / Path(
        "/proc/self/cgroup"
    ).read_text().strip().split(":")[-1].lstrip("/")
    config = {
        "command": __import__("sys").argv,
        "executable_sha256": sha(a.exe),
        "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_sha256": {
            str(f.relative_to(ROOT)): sha(f)
            for f in sorted((ROOT / "src").glob("*"))
            if f.is_file()
        },
        "platform": platform.platform(),
        "cpu": next(
            s.split(":", 1)[1].strip()
            for s in Path("/proc/cpuinfo").read_text().splitlines()
            if s.startswith("model name")
        ),
        "cpu_quota": (cgroup / "cpu.max").read_text().strip()
        if (cgroup / "cpu.max").exists()
        else None,
        "threads": a.threads,
        "omp_wait_policy": "PASSIVE",
        "warmup_repetitions": 1,
        "measured_repetitions": 3,
        "layers": "all",
        "arithmetic": "interval filters with exact rational fallback and exact constructions",
        "timing_includes": [
            "residual preparation",
            "visibility",
            "residual partition",
            "conforming labelled triangulation",
        ],
        "timing_excludes": [
            "input parsing/projection/tie preflight",
            "validation",
            "file serialization",
        ],
        "timeout_seconds": a.timeout,
        "memory_cap_mib": a.memory_mib,
    }
    (a.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    rows, statuses = [], []

    def limit():
        resource.setrlimit(resource.RLIMIT_AS, (a.memory_mib * 1024**2,) * 2)

    for n in a.sizes:
        scene = a.output / "scenes" / f"sculpted-{n}.json"
        scene.write_text(json.dumps(make_mesh(*SIZES[n]), separators=(",", ":")) + "\n")
        for method in a.methods:
            output = a.output / f"{method}-{n}"
            cmd = [
                str(a.exe.resolve()),
                "--scene",
                str(scene.resolve()),
                "--method",
                method,
                "--order",
                "near-first",
                "--layers",
                "all",
                "--threads",
                str(a.threads),
                "--repeat",
                "4",
                "--out",
                str(output.resolve()),
            ]
            if method in ("bvh-peel", "vbvh-peel"):
                cmd += ["--metrics-only"]
            env = dict(os.environ, OMP_WAIT_POLICY="PASSIVE")
            started = time.monotonic()
            status = {"method": method, "input_triangles": n, "command": cmd}
            print(f"START {method} {n}", flush=True)
            try:
                result = subprocess.run(
                    cmd,
                    env=env,
                    text=True,
                    capture_output=True,
                    timeout=a.timeout,
                    preexec_fn=limit,
                    check=False,
                )
                status.update(
                    status="completed" if result.returncode == 0 else "error",
                    returncode=result.returncode,
                    stdout=result.stdout,
                    stderr=result.stderr,
                )
                if result.returncode == 0:
                    m = json.loads((output / "metrics.json").read_text())
                    times = m["algorithm_repetitions_seconds"][1:]
                    assert len(times) == 3
                    row = {
                        "method": method,
                        "input_triangles": n,
                        "layers": len(m["passes"]),
                        "threads": a.threads,
                        "median_seconds": statistics.median(times),
                        "min_seconds": min(times),
                        "max_seconds": max(times),
                        "setup_seconds": m["projection_preflight_seconds"],
                        "output_triangles": sum(
                            x["exported_triangles"] for x in m["passes"]
                        ),
                        "candidate_tests": sum(
                            x["candidate_tests"] for x in m["passes"]
                        ),
                        "index_box_tests": sum(
                            x["index_box_tests"] for x in m["passes"]
                        ),
                        "peak_rss_kib": m["peak_rss_kib"],
                    }
                    rows.append(row)
                    status["measured_seconds"] = times
                    print(json.dumps(row), flush=True)
            except subprocess.TimeoutExpired:
                status["status"] = "timeout"
                print(f"TIMEOUT {method} {n}", flush=True)
            status["process_wall_seconds"] = time.monotonic() - started
            statuses.append(status)
            (a.output / "status.jsonl").write_text(
                "".join(json.dumps(s) + "\n" for s in statuses)
            )
            if rows:
                with (a.output / "summary.csv").open("w") as f:
                    writer = csv.DictWriter(
                        f, fieldnames=list(rows[0]), lineterminator="\n"
                    )
                    writer.writeheader()
                    writer.writerows(rows)
    (a.output / "completion.json").write_text(
        json.dumps(
            {
                "complete": True,
                "configurations": len(statuses),
                "successful": len(rows),
            },
            indent=2,
        )
        + "\n"
    )
    if any(s["status"] == "error" for s in statuses):
        raise SystemExit("A benchmark failed; inspect status.jsonl")


if __name__ == "__main__":
    main()
