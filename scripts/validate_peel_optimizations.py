#!/usr/bin/env python3
"""Check rank pruning and conservative residual merging against independent regions."""

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exe", type=Path, default=ROOT / "build/geometric_peel")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument(
        "--report", type=Path, default=ROOT / "results/cpu-transfer/validation.json"
    )
    a = p.parse_args()
    records = []
    with tempfile.TemporaryDirectory(prefix="peel-optimizations-") as tmp:
        tmp = Path(tmp)
        # A foreground rectangle splits the hidden part of a background triangle
        # along its internal diagonal. Its union is convex and must coalesce.
        scene = tmp / "convex-hidden-union.json"
        scene.write_text(
            json.dumps(
                {
                    "window": [-8, -8, 8, 8],
                    "triangles": [
                        {"id": 91, "vertices": [[0, 0, 3], [6, 0, 3], [0, 6, 3]]},
                        {"id": 17, "vertices": [[1, -1, 1], [3, -1, 1], [3, 7, 1]]},
                        {"id": 203, "vertices": [[1, -1, 1], [3, 7, 1], [1, 7, 1]]},
                    ],
                }
            )
        )
        scenes = [scene] + [
            ROOT / "tests/fixtures" / f"{n}.json"
            for n in [
                "holes",
                "components",
                "partial",
                "slivers",
                "cyclic",
                "perspective_torus",
                "perspective_near",
                "empty",
            ]
        ]
        for scene in scenes:
            for method in ("bvh-peel", "fast-bvh-peel"):
                baseline = None
                for label, flags in [
                    ("control", ["--no-rank-pruning"]),
                    ("pruned", []),
                    ("merged", ["--merge-residuals"]),
                    ("merged-unaccelerated", ["--merge-residuals", "--no-accelerate"]),
                ]:
                    for layers, order, threads in [
                        ("all", "input", 1),
                        ("all", "reverse", a.threads),
                        ("1", "input", a.threads),
                        ("2", "random", a.threads),
                    ]:
                        out = tmp / "output"
                        cmd = [
                            str(a.exe),
                            "--scene",
                            str(scene),
                            "--method",
                            method,
                            "--resolved-mesh",
                            "--verify",
                            "--oracle-cap",
                            "128",
                            "--layers",
                            layers,
                            "--order",
                            order,
                            "--threads",
                            str(threads),
                            "--out",
                            str(out),
                        ] + flags
                        r = subprocess.run(
                            cmd,
                            text=True,
                            capture_output=True,
                            check=False,
                            timeout=120,
                            env=dict(os.environ, OMP_WAIT_POLICY="PASSIVE"),
                        )
                        if r.returncode:
                            raise RuntimeError(f"{cmd}\n{r.stderr}")
                        m = json.loads((out / "metrics.json").read_text())
                        merged = sum(
                            x["phases"].get("merged_patches", 0) for x in m["passes"]
                        )
                        if scene.stem == "convex-hidden-union" and label == "merged":
                            assert merged > 0, (method, layers, m)
                        geometry = json.loads(
                            (
                                out
                                / (
                                    "exact.json"
                                    if method == "bvh-peel"
                                    else "approximate.json"
                                )
                            ).read_text()
                        )
                        if layers == "all" and order == "input":
                            if label == "control":
                                baseline = geometry["layers"]
                            elif label == "pruned":
                                assert baseline == geometry["layers"], (scene, method)
                        # Exact --verify certifies full labelled regions. Fast --verify
                        # checks its documented tolerance, not exact geometry equality.
                        records.append(
                            {
                                "scene": scene.stem,
                                "method": method,
                                "variant": label,
                                "layers": layers,
                                "order": order,
                                "threads": threads,
                                "merged_patches": merged,
                                "status": "passed",
                                "fast_validation": m.get("fast_validation"),
                            }
                        )
            print(scene.stem, flush=True)
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(
        json.dumps({"checks": len(records), "results": records}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
