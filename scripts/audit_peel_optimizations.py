#!/usr/bin/env python3
"""Audit cached dragons separately from timing; retain approximate-mode mismatches."""

import argparse
import json
import os
import resource
import subprocess
from pathlib import Path

from benchmark_global import digest, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-exe", type=Path)
    p.add_argument("--exe", type=Path, default=Path("build/geometric_peel"))
    p.add_argument("--output", type=Path, default=Path("results/cpu-transfer/audits"))
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    write_json(
        a.output / "config.json",
        {
            "executable_sha256": digest(a.exe),
            "baseline_sha256": digest(a.baseline_exe) if a.baseline_exe else None,
            "threads": 4,
            "timeout_seconds": 240,
            "address_space_cap_gib": 24,
        },
    )

    def limit():
        resource.setrlimit(resource.RLIMIT_AS, (24 * 1024**3,) * 2)

    records = []
    for target, method, check in [
        (128, "bvh-peel", "--verify"),
        (512, "bvh-peel", "--verify-global"),
        (4096, "bvh-peel", "--validate"),
        (128, "fast-bvh-peel", "--verify"),
        (65536, "fast-bvh-peel", "--validate"),
    ]:
        scene = Path(f"results/global/dragon-cache/{target}/resolved.json")
        variants = [("pruned", a.exe, []), ("merged", a.exe, ["--merge-residuals"])]
        if a.baseline_exe and target == 65536 and method == "fast-bvh-peel":
            variants.append(("before", a.baseline_exe, []))
        for variant, exe, flags in variants:
            out = a.output / f"{target}-{method}-{variant}"
            out.mkdir(parents=True, exist_ok=True)
            cmd = [
                str(exe),
                "--scene",
                str(scene),
                "--method",
                method,
                "--resolved-mesh",
                "--layers",
                "all",
                "--threads",
                "4",
                "--metrics-only",
                "--oracle-cap",
                "256",
                check,
                "--out",
                str(out),
            ] + flags
            rec = {
                "target": target,
                "method": method,
                "variant": variant,
                "command": cmd,
                "scene_sha256": digest(scene),
                "check": check,
            }
            try:
                r = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=240,
                    preexec_fn=limit,
                    env=dict(os.environ, OMP_WAIT_POLICY="PASSIVE"),
                )
                rec.update(returncode=r.returncode, stderr=r.stderr)
                metrics = out / "metrics.json"
                if metrics.exists():
                    m = json.loads(metrics.read_text())
                    rec.update(
                        input_sources=m["input_sources"],
                        fast_validation=m.get("fast_validation"),
                        convex_audit=m.get("convex_audit"),
                        exact_invariants_validated=m["exact_invariants_validated"],
                        validated_against_oracle=m["validated_against_oracle"],
                        validated_against_global=m.get(
                            "validated_against_global", False
                        ),
                    )
                if r.returncode == 0:
                    rec["status"] = "passed"
                elif (
                    r.returncode == 3 and method == "fast-bvh-peel" and metrics.exists()
                ):
                    rec["status"] = "numerical_mismatch"
                else:
                    rec["status"] = "error"
            except subprocess.TimeoutExpired:
                rec["status"] = "timeout"
            records.append(rec)
            write_json(a.output / "runs.json", records)
            print(target, method, variant, rec["status"], flush=True)
            if rec["status"] in ["error", "timeout"]:
                raise SystemExit("Audit did not complete; see runs.json")


if __name__ == "__main__":
    main()
