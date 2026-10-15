#!/usr/bin/env python3
"""Sequential exact/fast comparison on identical inputs, followed by untimed exact audits."""

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

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exe", type=Path, default=ROOT / "build/geometric_peel")
    p.add_argument("--output", type=Path, default=ROOT / "results/fast")
    p.add_argument("--scenes", type=Path, default=ROOT / "results/large/scenes")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--timeout", type=float, default=180)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_WAIT_POLICY="PASSIVE")
    config = {
        "executable_sha256": sha(a.exe),
        "source_sha256": {
            str(f.relative_to(ROOT)): sha(f)
            for f in sorted((ROOT / "src").glob("*"))
            if f.is_file()
        },
        "git_base_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "platform": platform.platform(),
        "threads": a.threads,
        "omp_wait_policy": "PASSIVE",
        "cpu": next(
            s.split(":", 1)[1].strip()
            for s in Path("/proc/cpuinfo").read_text().splitlines()
            if s.startswith("model name")
        ),
        "warmup_repetitions": 1,
        "exact_measured_repetitions": 3,
        "fast_measured_repetitions": 9,
        "grid_bits": 30,
        "memory_cap_mib": 4096,
        "timeout_seconds": a.timeout,
        "timing_includes": [
            "BVH construction",
            "contour rank classification",
            "complete layer extraction",
            "labelled triangulation",
        ],
        "timing_excludes": [
            "input parse/projection/topology",
            "optional intersection preprocessing",
            "validation",
            "file export",
        ],
        "scene_sha256": {
            str(n): sha(a.scenes / f"sculpted-{n}.json") for n in (10240, 102400)
        },
    }
    cgroup = Path("/sys/fs/cgroup") / Path(
        "/proc/self/cgroup"
    ).read_text().strip().split(":")[-1].lstrip("/")
    config["cpu_quota"] = (
        (cgroup / "cpu.max").read_text().strip()
        if (cgroup / "cpu.max").exists()
        else None
    )
    (a.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    variants = [
        ("exact-bvh", "bvh-peel", []),
        ("fast-clipping", "fast-bvh-peel", []),
        ("fast-contours", "fast-bvh-peel", ["--resolved-mesh"]),
    ]
    rows, attempts, audits = [], [], []

    def limit():
        resource.setrlimit(resource.RLIMIT_AS, (4096 * 1024**2,) * 2)

    def execute(cmd):
        start = time.monotonic()
        try:
            r = subprocess.run(
                cmd,
                text=True,
                capture_output=True,
                timeout=a.timeout,
                env=env,
                preexec_fn=limit,
                check=False,
            )
            status = {
                "returncode": r.returncode,
                "stdout": r.stdout,
                "stderr": r.stderr,
                "status": "completed" if r.returncode == 0 else "error",
            }
        except subprocess.TimeoutExpired:
            status = {"status": "timeout"}
        status.update(command=cmd, wall_seconds=time.monotonic() - start)
        return status

    for n in (10240, 102400):
        for label, method, extra in variants:
            print(f"Measuring {label} {n}", flush=True)
            output = a.output / f"{label}-{n}"
            cmd = [
                str(a.exe),
                "--scene",
                str(a.scenes / f"sculpted-{n}.json"),
                "--method",
                method,
                "--layers",
                "all",
                "--threads",
                str(a.threads),
                "--repeat",
                "4" if label == "exact-bvh" else "10",
                "--metrics-only",
                "--out",
                str(output),
            ] + extra
            status = execute(cmd)
            status.update(variant=label, input_triangles=n)
            attempts.append(status)
            (a.output / "status.jsonl").write_text(
                "".join(json.dumps(s) + "\n" for s in attempts)
            )
            if status["status"] != "completed":
                continue
            m = json.loads((output / "metrics.json").read_text())
            times = m["algorithm_repetitions_seconds"][1:]
            row = {
                "variant": label,
                "input_triangles": n,
                "layers": len(m["passes"]),
                "median_seconds": statistics.median(times),
                "min_seconds": min(times),
                "max_seconds": max(times),
                "setup_seconds": m["projection_preflight_seconds"],
                "output_triangles": sum(p["exported_triangles"] for p in m["passes"]),
                "candidate_tests": sum(p["candidate_tests"] for p in m["passes"]),
                "peak_rss_kib": m["peak_rss_kib"],
            }
            rows.append(row)
            with (a.output / "summary.csv").open("w") as f:
                writer = csv.DictWriter(f, fieldnames=list(row), lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
            print(json.dumps(row), flush=True)
    for n in (10240, 102400):
        for label, method, extra in variants[1:]:
            print(f"Auditing {label} {n}", flush=True)
            demo = label == "fast-contours" and n == 10240
            output = a.output / ("demo" if demo else f"audit-{label}-{n}")
            cmd = [
                str(a.exe),
                "--scene",
                str(a.scenes / f"sculpted-{n}.json"),
                "--method",
                method,
                "--layers",
                "all",
                "--threads",
                str(a.threads),
                "--validate",
                "--mesh-only" if demo else "--metrics-only",
                "--out",
                str(output),
            ] + extra
            result = execute(cmd)
            record = {"variant": label, "input_triangles": n, "run": result}
            if (output / "metrics.json").exists():
                m = json.loads((output / "metrics.json").read_text())
                record["audit"] = m["fast_validation"]
            audits.append(record)
            (a.output / "audits.json").write_text(json.dumps(audits, indent=2) + "\n")
    (a.output / "completion.json").write_text(
        json.dumps(
            {
                "measured_configurations": len(rows),
                "attempted_configurations": len(attempts),
                "audited_configurations": len(audits),
            },
            indent=2,
        )
        + "\n"
    )
    if len(rows) != len(attempts):
        raise SystemExit("A timed run failed; see status.jsonl")


if __name__ == "__main__":
    main()
