#!/usr/bin/env python3
"""Fresh CPU-fast comparison for the contour-event GPU experiment."""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scenes", type=Path, nargs="+")
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--out", type=Path, default=Path("results/gpu-events/cpu"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    quotas = {}
    for line in Path("/proc/self/cgroup").read_text().splitlines():
        if line.startswith("0::"):
            group = Path("/sys/fs/cgroup") / line.split("::")[1].lstrip("/")
            for ancestor in (group, *group.parents):
                quota = ancestor / "cpu.max"
                if quota.exists():
                    quotas[str(quota)] = quota.read_text().strip()
    report = {
        "scope": "CPU fast extraction and conforming meshing only; excludes import/projection/topology, validation/export; first of four repetitions discarded as warmup",
        "binary_sha256": hashlib.sha256(
            Path("build/geometric_peel").read_bytes()
        ).hexdigest(),
        "cpu_quotas": quotas,
        "load_before": os.getloadavg(),
        "cases": [],
    }
    for path in args.scenes:
        name = path.parent.name if path.name == "resolved.json" else path.stem
        for threads in args.threads:
            directory = args.out / f"{name}-t{threads}"
            subprocess.run(
                [
                    "build/geometric_peel",
                    "--scene",
                    str(path),
                    "--method",
                    "fast-bvh-peel",
                    "--resolved-mesh",
                    "--metrics-only",
                    "--repeat",
                    "4",
                    "--threads",
                    str(threads),
                    "--out",
                    str(directory),
                ],
                env=dict(os.environ, OMP_WAIT_POLICY="PASSIVE"),
                check=True,
                timeout=600,
            )
            m = json.loads((directory / "metrics.json").read_text())
            times = [1000 * x for x in m["algorithm_repetitions_seconds"][1:]]
            case = {
                "scene": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "threads": threads,
                "sources": m["input_sources"],
                "milliseconds": times,
                "median_ms": statistics.median(times),
                "metrics": str(directory / "metrics.json"),
            }
            report["cases"].append(case)
            (args.out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(case), flush=True)


if __name__ == "__main__":
    main()
