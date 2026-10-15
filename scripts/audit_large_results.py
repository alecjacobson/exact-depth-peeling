#!/usr/bin/env python3
"""Validate the recorded large scenes and export the actual 10K demonstration meshes."""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exe", type=Path, default=ROOT / "build/geometric_peel")
    p.add_argument("--directory", type=Path, default=ROOT / "results/large")
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    config = json.loads((a.directory / "config.json").read_text())
    digest = sha(a.exe)
    if digest != config["executable_sha256"]:
        raise SystemExit(
            "Executable differs from the measured build; rerun benchmarks first"
        )
    records = []
    for n in (10240, 102400):
        scene = a.directory / "scenes" / f"sculpted-{n}.json"
        output = a.directory / ("demo" if n == 10240 else f"audit-{n}")
        cmd = [
            str(a.exe.resolve()),
            "--scene",
            str(scene.resolve()),
            "--method",
            "bvh-peel",
            "--order",
            "near-first",
            "--layers",
            "all",
            "--threads",
            str(a.threads),
            "--validate",
            "--mesh-only" if n == 10240 else "--metrics-only",
            "--out",
            str(output.resolve()),
        ]
        print(f"Auditing {n:,} triangles", flush=True)
        subprocess.run(cmd, env=dict(os.environ, OMP_WAIT_POLICY="PASSIVE"), check=True)
        m = json.loads((output / "metrics.json").read_text())
        measured = json.loads(
            (a.directory / f"bvh-peel-{n}" / "metrics.json").read_text()
        )
        assert len(m["passes"]) == len(measured["passes"])
        records.append(
            {
                "input_triangles": n,
                "scene_sha256": sha(scene),
                "command": cmd,
                "layers": len(m["passes"]),
                "audit": m["convex_audit"],
                "audited_output_triangles": [
                    p["exported_triangles"] for p in m["passes"]
                ],
                "timed_output_triangles": [
                    p["exported_triangles"] for p in measured["passes"]
                ],
                "metrics": str(output.relative_to(a.directory) / "metrics.json"),
            }
        )
        (a.directory / "audits.json").write_text(
            json.dumps(
                {
                    "executable_sha256": digest,
                    "scope": "Exact containment, conservation and disjointness; sampled independent exact ray ranks. Full arrangement rank certification is restricted to small fixtures.",
                    "records": records,
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
