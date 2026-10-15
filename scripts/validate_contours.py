#!/usr/bin/env python3
"""Exact contour acceleration: oracle, prefix, ordering and threading regressions."""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exe", type=Path, default=ROOT / "build/geometric_peel")
    p.add_argument(
        "--resolver", type=Path, default=ROOT / "build/resolve_intersections"
    )
    p.add_argument(
        "--report", type=Path, default=ROOT / "results/contours/validation.json"
    )
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    names = [
        "single",
        "window_clipped",
        "disjoint",
        "stacked",
        "partial",
        "holes",
        "components",
        "coplanar_adjacent",
        "shared_vertex",
        "edge_on",
        "closed_object",
        "cover_2",
        "empty",
        "binary_float",
        "cyclic",
        "perspective_partial",
        "perspective_near",
        "perspective_behind",
        "camera_transform",
        "perspective_torus",
        "slivers",
    ]
    records = []
    with tempfile.TemporaryDirectory(prefix="exact-contours-") as temp:
        tmp = Path(temp)
        scenes = [ROOT / "tests/fixtures" / f"{n}.json" for n in names]
        for n in ["crossing", "perspective_crossing"]:
            scene = tmp / f"{n}-resolved.json"
            subprocess.run(
                [
                    str(a.resolver),
                    str(ROOT / "tests/fixtures" / f"{n}.json"),
                    str(scene),
                ],
                check=True,
                capture_output=True,
            )
            scenes.append(scene)
        junction = {
            "window": [-2, -2, 2, 2],
            "triangles": [
                {"id": 10 + i * 7, "vertices": [[0, -1, 2], [0, 1, 2], c]}
                for i, c in enumerate(([-1, 0, 3], [1, 0, 3], ["1/2", 0, 1]))
            ],
        }
        scene = tmp / "nonmanifold.json"
        scene.write_text(json.dumps(junction))
        scenes.append(scene)
        # A deeply covered triangle must cross omitted internal mesh edges while
        # remaining clear of the silhouette; exercise direct ranks > 0 and sparse IDs.
        triangles = []
        for z in (1, 2, 3):
            for x in (-3, -1, 1):
                for y in (-3, -1, 1):
                    for face in (
                        ((x, y), (x + 2, y), (x + 2, y + 2)),
                        ((x, y), (x + 2, y + 2), (x, y + 2)),
                    ):
                        triangles.append(
                            {
                                "id": 100 + len(triangles) * 11,
                                "vertices": [
                                    [u + (0.25 if z == 2 else 0), v, z] for u, v in face
                                ],
                            }
                        )
        scene = tmp / "internal-edge-ranks.json"
        scene.write_text(json.dumps({"window": [-4, -4, 4, 4], "triangles": triangles}))
        scenes.append(scene)
        for scene in scenes:
            for order, threads, k, accelerated in [
                ("input", 1, "all", True),
                ("reverse", a.threads, "all", True),
                ("random", a.threads, "all", True),
                ("near-first", 1, "all", False),
                ("input", a.threads, "1", True),
                ("input", a.threads, "2", True),
                ("input", 1, "4", True),
            ]:
                out = tmp / "result"
                cmd = [
                    str(a.exe),
                    "--scene",
                    str(scene),
                    "--method",
                    "bvh-peel",
                    "--resolved-mesh",
                    "--verify",
                    "--oracle-cap",
                    "128",
                    "--order",
                    order,
                    "--layers",
                    k,
                    "--threads",
                    str(threads),
                    "--out",
                    str(out),
                ]
                if not accelerated:
                    cmd.append("--no-accelerate")
                r = subprocess.run(
                    cmd, text=True, capture_output=True, check=False, timeout=90
                )
                if r.returncode:
                    raise RuntimeError(f"{scene.name}: {cmd}\n{r.stderr}")
                m = json.loads((out / "metrics.json").read_text())
                if scene.stem == "internal-edge-ranks":
                    assert m["rank_classification"]["constant_rank_sources"] >= 1
                records.append(
                    {
                        "scene": scene.stem,
                        "order": order,
                        "threads": threads,
                        "layers": k,
                        "accelerated": accelerated,
                        "classification": m["rank_classification"],
                        "status": "passed",
                    }
                )
            print(scene.stem, flush=True)
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(
        json.dumps({"checks": len(records), "results": records}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
