#!/usr/bin/env python3
"""Cross-check global contour subdivision against the independent eager oracle."""

import argparse
import hashlib
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
        "--report", type=Path, default=ROOT / "results/global/validation.json"
    )
    a = p.parse_args()
    records = []
    env = dict(os.environ, OMP_WAIT_POLICY="PASSIVE")
    with tempfile.TemporaryDirectory(prefix="global-oracle-") as tmp:
        temp = Path(tmp)
        fixtures = sorted((ROOT / "tests/fixtures").glob("*.json"))
        fixtures = [f for f in fixtures if f.name != "manifest.json"]
        # Adjacent sheet triangles whose shared projected diagonal passes through
        # the centroid of a deeper triangle: rank sampling must avoid double hits.
        base = {
            "window": [-3, -3, 3, 3],
            "triangles": [
                {"id": 9, "vertices": [[-2, -2, 1], [2, -2, 1], [2, 2, 1]]},
                {"id": 81, "vertices": [[-2, -2, 1], [2, 2, 1], [-2, 2, 1]]},
                {"id": 702, "vertices": [[-1, -1, 2], [1, -1, 2], [0, 2, 2]]},
            ],
        }
        boundary = temp / "internal-edge-sample.json"
        boundary.write_text(json.dumps(base))
        fixtures.append(boundary)
        junction = temp / "nonmanifold.json"
        junction.write_text(
            json.dumps(
                {
                    "window": [-2, -2, 2, 2],
                    "triangles": [
                        {"id": 10 + i * 7, "vertices": [[0, -1, 2], [0, 1, 2], v]}
                        for i, v in enumerate(([-1, 0, 3], [1, 0, 3], ["1/2", 0, 1]))
                    ],
                }
            )
        )
        fixtures.append(junction)
        # A crossing can lie exactly on an omitted same-facing internal edge.
        # Its equality segment need not straddle both incident source polygons.
        crease = temp / "crossing-on-internal-edge.json"
        crease.write_text(
            json.dumps(
                {
                    "window": [-3, -3, 3, 3],
                    "triangles": [
                        {"id": 11, "vertices": [[-2, -2, 2], [2, -2, 1], [1, 0, 2]]},
                        {"id": 12, "vertices": [[-2, -2, 2], [1, 0, 2], [0, 2, 3]]},
                        {
                            "id": 13,
                            "vertices": [[-0.5, -0.5, 2], [0.5, -0.5, 2], [0, 0.5, 2]],
                        },
                    ],
                }
            )
        )
        fixtures.append(crease)
        for scene in fixtures:
            reject = scene.stem in ("tie", "hidden_tie")
            for label, layers, threads, extra in [
                ("contours", "all", 1, []),
                ("parallel", "all", a.threads, []),
                ("unaccelerated", "all", 1, ["--no-accelerate"]),
                ("all-edges", "all", a.threads, ["--global-all-edges"]),
                ("prefix", "1", a.threads, []),
                ("prefix-2", "2", 1, []),
            ]:
                cmd = [
                    str(a.exe),
                    "--scene",
                    str(scene),
                    "--method",
                    "global-arrangement",
                    "--layers",
                    layers,
                    "--threads",
                    str(threads),
                    "--verify",
                    "--oracle-cap",
                    "128",
                    "--metrics-only",
                    "--out",
                    str(temp / "result"),
                ] + extra
                r = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=False,
                    env=env,
                    timeout=90,
                )
                if reject:
                    assert r.returncode and "positive-area depth tie" in r.stderr, (
                        r.stderr
                    )
                    record = {
                        "scene": scene.stem,
                        "variant": label,
                        "status": "expected_rejection",
                    }
                else:
                    if r.returncode:
                        raise RuntimeError(f"{cmd}\n{r.stderr}")
                    m = json.loads((temp / "result/metrics.json").read_text())
                    record = {
                        "scene": scene.stem,
                        "variant": label,
                        "status": "passed",
                        "global_arrangement": m["global_arrangement"],
                    }
                    if scene.stem in (
                        "crossing",
                        "perspective_crossing",
                        "crossing-on-internal-edge",
                    ):
                        assert m["global_arrangement"]["depth_equality_segments"] > 0
                if scene.stem == "internal-edge-sample" and label != "all-edges":
                    assert m["global_arrangement"]["generic_sample_retries"] > 0
                records.append(record)
            print(scene.stem, flush=True)
        # Exercise the new oracle through the peeling verification API as well.
        for name in ("crossing", "perspective_torus", "holes"):
            cmd = [
                str(a.exe),
                "--scene",
                str(ROOT / f"tests/fixtures/{name}.json"),
                "--method",
                "bvh-peel",
                "--verify-global",
                "--metrics-only",
                "--out",
                str(temp / "crosscheck"),
            ]
            r = subprocess.run(
                cmd, capture_output=True, text=True, check=False, env=env, timeout=90
            )
            if r.returncode:
                raise RuntimeError(r.stderr)
            records.append(
                {"scene": name, "variant": "peeling-verify-global", "status": "passed"}
            )
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(
        json.dumps(
            {
                "checks": len(records),
                "executable_sha256": hashlib.sha256(a.exe.read_bytes()).hexdigest(),
                "results": records,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"{len(records)} global oracle checks passed")


if __name__ == "__main__":
    main()
