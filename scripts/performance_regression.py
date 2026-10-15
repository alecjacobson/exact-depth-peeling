#!/usr/bin/env python3
"""Compare a saved pre-refactor benchmark with fresh matched runs; fail on >20% slowdown.

Capture baseline with benchmark_fast.py before editing/building, then run this
script with no concurrent renderer/build/test workload. Timing is opt-in, not CTest.
"""

import argparse
import csv
import hashlib
import json
import os
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--baseline-exe", type=Path)
    p.add_argument("--exe", type=Path, default=ROOT / "build/geometric_peel")
    p.add_argument("--output", type=Path, default=ROOT / "results/contours/performance")
    p.add_argument("--max-ratio", type=float, default=1.20)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    baseline = list(csv.DictReader((a.baseline / "summary.csv").open()))
    env = dict(os.environ, OMP_WAIT_POLICY="PASSIVE")
    records = []
    config = {
        "baseline": json.loads((a.baseline / "config.json").read_text()),
        "baseline_rows": baseline,
        "maximum_allowed_ratio": a.max_ratio,
        "executable_sha256": hashlib.sha256(a.exe.read_bytes()).hexdigest(),
        "sources": {
            str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in sorted((ROOT / "src").glob("*"))
        },
        "paired_baseline_executable": str(a.baseline_exe) if a.baseline_exe else None,
        "paired_baseline_sha256": hashlib.sha256(
            a.baseline_exe.read_bytes()
        ).hexdigest()
        if a.baseline_exe
        else None,
        "threads": 4,
        "warmups": 1,
        "measured_repetitions": {"exact": 3, "fast": 9},
        "timing_excludes": ["projection/topology", "validation", "export"],
        "timing_includes": ["classification", "all-layer extraction", "triangulation"],
    }
    (a.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    for row in baseline + [
        dict(r, variant="exact-contours")
        for r in baseline
        if r["variant"] == "exact-bvh"
    ]:
        label = row["variant"]
        n = int(row["input_triangles"])
        fast = label.startswith("fast")
        out = a.output / f"{label}-{n}"
        cmd = [
            str(a.exe),
            "--scene",
            str(ROOT / f"results/large/scenes/sculpted-{n}.json"),
            "--method",
            "fast-bvh-peel" if fast else "bvh-peel",
            "--layers",
            "all",
            "--threads",
            "4",
            "--repeat",
            "10" if fast else "4",
            "--metrics-only",
            "--out",
            str(out),
        ]
        if label.endswith("contours"):
            cmd.append("--resolved-mesh")
        if a.baseline_exe:
            before_cmd = cmd.copy()
            before_cmd[0] = str(a.baseline_exe)
            before_dir = a.output / f"paired-before-{label}-{n}"
            before_cmd[before_cmd.index("--out") + 1] = str(before_dir)
            if label == "exact-contours":
                before_cmd.remove("--resolved-mesh")
            old = subprocess.run(
                before_cmd,
                text=True,
                capture_output=True,
                check=False,
                env=env,
                timeout=240,
            )
            if old.returncode:
                raise RuntimeError(old.stderr)
            bm = json.loads((before_dir / "metrics.json").read_text())
            row = dict(
                row,
                median_seconds=statistics.median(
                    bm["algorithm_repetitions_seconds"][1:]
                ),
            )
        r = subprocess.run(
            cmd, text=True, capture_output=True, check=False, env=env, timeout=240
        )
        if r.returncode:
            raise RuntimeError(r.stderr)
        m = json.loads((out / "metrics.json").read_text())
        times = m["algorithm_repetitions_seconds"][1:]
        t = statistics.median(times)
        ratio = t / float(row["median_seconds"])
        unchanged = label != "exact-contours"
        rec = {
            "variant": label,
            "input_triangles": n,
            "median_seconds": t,
            "min_seconds": min(times),
            "max_seconds": max(times),
            "baseline_seconds": float(row["median_seconds"]),
            "ratio": ratio,
            "timing_pass": ratio <= a.max_ratio,
            "layers": len(m["passes"]),
            "output_triangles": sum(p["exported_triangles"] for p in m["passes"]),
            "candidate_tests": sum(p["candidate_tests"] for p in m["passes"]),
            "classification": m.get("rank_classification"),
            "setup_seconds": m["projection_preflight_seconds"],
        }
        if unchanged:
            rec["work_counts_unchanged"] = all(
                rec[k] == int(row[k])
                for k in ["layers", "output_triangles", "candidate_tests"]
            )
            assert rec["work_counts_unchanged"], rec
        records.append(rec)
        print(json.dumps(rec), flush=True)
        (a.output / "comparison.json").write_text(json.dumps(records, indent=2) + "\n")
    if not all(r["timing_pass"] for r in records):
        raise SystemExit("Performance regression; see comparison.json")


if __name__ == "__main__":
    main()
