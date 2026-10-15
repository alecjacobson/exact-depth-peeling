#!/usr/bin/env python3
"""Budgeted full-dragon extraction, then untimed independent exact-ray audit."""

import argparse
import hashlib
import json
import os
import resource
import statistics
import subprocess
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exe", type=Path, default=Path("build/geometric_peel"))
    p.add_argument("--output", type=Path, default=Path("results/dragon"))
    p.add_argument("--timeout", type=int, default=300)
    a = p.parse_args()
    env = dict(os.environ, OMP_WAIT_POLICY="PASSIVE")
    records = []
    config = {
        "executable_sha256": hashlib.sha256(a.exe.read_bytes()).hexdigest(),
        "scene_sha256": hashlib.sha256(
            (a.output / "scene.json").read_bytes()
        ).hexdigest(),
        "threads": 4,
        "warmups": 1,
        "measured_repetitions": 3,
        "process_timeout_seconds": a.timeout,
        "address_space_cap_gib": 24,
        "algorithm_includes": [
            "rank classification",
            "all layers",
            "labelled triangulation",
        ],
        "algorithm_excludes": [
            "JSON import/projection/topology",
            "intersection preprocessing",
            "audit",
            "export",
        ],
    }
    (a.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    def limit():
        resource.setrlimit(resource.RLIMIT_AS, (24 * 1024**3,) * 2)

    for label, method, extra in [
        ("fast", "fast-bvh-peel", ["--repeat", "4"]),
        ("exact", "bvh-peel", ["--repeat", "4"]),
        ("audit", "fast-bvh-peel", ["--validate"]),
    ]:
        out = a.output / label
        cmd = [
            str(a.exe),
            "--scene",
            str(a.output / "scene.json"),
            "--method",
            method,
            "--resolved-mesh",
            "--layers",
            "all",
            "--threads",
            "4",
            "--metrics-only",
            "--out",
            str(out),
        ] + extra
        t = time.monotonic()
        record = {"variant": label, "command": cmd}
        try:
            r = subprocess.run(
                cmd,
                text=True,
                capture_output=True,
                check=False,
                env=env,
                timeout=a.timeout,
                preexec_fn=limit,
            )
            record.update(
                returncode=r.returncode,
                stdout=r.stdout,
                stderr=r.stderr,
                status="completed" if r.returncode == 0 else "error",
            )
        except subprocess.TimeoutExpired as e:
            record.update(status="timeout", stdout=e.stdout, stderr=e.stderr)
        record["wall_seconds"] = time.monotonic() - t
        if (out / "metrics.json").exists():
            m = json.loads((out / "metrics.json").read_text())
            times = m["algorithm_repetitions_seconds"][1:]
            record.update(
                layers=len(m["passes"]),
                output_triangles=sum(p["exported_triangles"] for p in m["passes"]),
                setup_seconds=m["projection_preflight_seconds"],
                classification=m.get("rank_classification"),
            )
            if times:
                record.update(
                    median_seconds=statistics.median(times),
                    min_seconds=min(times),
                    max_seconds=max(times),
                )
            if label == "audit":
                record["audit"] = m["fast_validation"]
        records.append(record)
        print(json.dumps(record), flush=True)
        (a.output / "runs.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
