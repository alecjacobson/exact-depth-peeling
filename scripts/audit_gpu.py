#!/usr/bin/env python3
"""Independent rational-ray audit of downloaded GPU regions (outside extraction)."""

import argparse
import hashlib
import json
import sys
from fractions import Fraction as Q
from itertools import pairwise
from pathlib import Path

import numpy as np
from shapely.geometry import Point, Polygon
from shapely.strtree import STRtree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.check_rays import cross, dot, subtract
from warp_peel import DevicePeeler
from warp_peel.io import load_scene


def audit(p, labels, data, samples=512, regression_queries=()):
    info = p.read_status()
    count = info["polygons"]
    raw = p.output[:count].numpy()
    polygons = [Polygon(r["v"][: r["n"]]) for r in raw]
    tree = STRtree(polygons)
    sources = labels[p.output_sources[:count].numpy()]
    layers = p.output_layers[:count].numpy()
    q = lambda x: Q(str(x))
    matrix = [
        [q(x) for x in row]
        for row in data.get("camera", {}).get(
            "world_to_camera", [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]
        )
    ]
    perspective = data.get("camera", {}).get("type") == "perspective"
    near = q(data.get("camera", {}).get("near", "0.01"))
    window = list(map(q, data["window"]))
    originals, lower, upper = [], [], []
    for t in data["triangles"]:
        camera = [
            tuple(dot(row[:3], tuple(map(q, v))) + row[3] for row in matrix)
            for v in t["vertices"]
        ]
        if perspective and min(v[2] for v in camera) < near:
            raise ValueError(
                "This large-scene audit requires all source vertices beyond near; near clipping is covered by fixture tests"
            )
        uv = [
            (float(v[0] / v[2]), float(v[1] / v[2]))
            if perspective
            else (float(v[0]), float(v[1]))
            for v in camera
        ]
        originals.append(
            (
                t["id"],
                camera[0],
                subtract(camera[1], camera[0]),
                subtract(camera[2], camera[0]),
            )
        )
        lower.append(np.min(uv, axis=0))
        upper.append(np.max(uv, axis=0))
    lower, upper = np.asarray(lower), np.asarray(upper)
    queries = [
        ((x + 0.371) / 32, (y + 0.619) / 32) for x in range(32) for y in range(32)
    ]
    if count:
        for i in np.linspace(0, count - 1, min(samples, count), dtype=int):
            queries.append(tuple(polygons[i].representative_point().coords[0]))
    queries.extend(regression_queries)
    checked = skipped = mismatch = max_hits = 0
    examples = []
    for uv in queries:
        point = Point(uv)
        candidates = tree.query(point)
        if any(polygons[i].boundary.distance(point) < 1e-11 for i in candidates):
            skipped += 1
            continue
        actual = sorted(
            (int(layers[i]), int(sources[i]))
            for i in candidates
            if polygons[i].contains(point)
        )
        x = tuple(window[a] + q(uv[a]) * (window[a + 2] - window[a]) for a in range(2))
        xf = np.array(list(map(float, x)))
        # Conservative floating bounds select candidates; hit tests and ranks use rationals.
        selected = np.where(
            np.all(lower <= xf + 1e-9, axis=1) & np.all(upper >= xf - 1e-9, axis=1)
        )[0]
        origin = (Q(0), Q(0), Q(0)) if perspective else (*x, Q(0))
        direction = (*x, Q(1)) if perspective else (Q(0), Q(0), Q(1))
        hits = []
        boundary = False
        for i in selected:
            ident, v, e1, e2 = originals[i]
            h = cross(direction, e2)
            det = dot(e1, h)
            if det == 0:
                continue
            s = subtract(origin, v)
            b = dot(s, h) / det
            qq = cross(s, e1)
            c = dot(direction, qq) / det
            depth = dot(e2, qq) / det
            if b < 0 or c < 0 or b + c > 1 or (perspective and depth < near):
                continue
            if b == 0 or c == 0 or b + c == 1:
                boundary = True
                break
            hits.append((depth, ident))
        hits.sort()
        if boundary or any(a[0] == b[0] for a, b in pairwise(hits)):
            skipped += 1
            continue
        expected = [(i, ident) for i, (_, ident) in enumerate(hits)]
        if not hits and not actual:
            continue
        checked += 1
        max_hits = max(max_hits, len(hits))
        if actual != expected:
            mismatch += 1
            if len(examples) < 20:
                examples.append(
                    {"uv_normalized": uv, "expected": expected, "actual": actual}
                )
    return {
        "scope": "Sampled source-labelled ranks using independent rational ray/triangle tests, padded floating input bounds, and downloaded GPU polygons; not a region certificate. Grid plus output interior samples; boundary-near samples skipped.",
        "checked_rays": checked,
        "skipped_boundary_rays": skipped,
        "rank_mismatches": mismatch,
        "maximum_hits": max_hits,
        "examples": examples,
        "output": info,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenes", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, default=Path("results/gpu/audits.json"))
    args = parser.parse_args()
    kernel_sha = hashlib.sha256(Path("warp_peel/kernels.py").read_bytes()).hexdigest()
    report = []
    for scene in args.scenes:
        print(scene, flush=True)
        arrays, labels, data = load_scene(scene)
        p = DevicePeeler(*arrays)
        p.run()
        sha = hashlib.sha256(scene.read_bytes()).hexdigest()
        regressions = json.loads(
            (
                Path(__file__).resolve().parents[1] / "tests/gpu/regression_rays.json"
            ).read_text()
        )
        result = audit(p, labels, data, regression_queries=regressions.get(sha, ()))
        result["regression_rays"] = regressions.get(sha, [])
        result.update(
            scene=str(scene),
            sha256=hashlib.sha256(scene.read_bytes()).hexdigest(),
            kernel_sha256=kernel_sha,
        )
        print(
            {k: v for k, v in result.items() if k not in ("examples", "scope")},
            flush=True,
        )
        report.append(result)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        del p, arrays


if __name__ == "__main__":
    main()
