#!/usr/bin/env python3
"""Complete labelled-region comparison with the contour-aware exact global oracle."""

import argparse
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shapely import make_valid, set_precision
from shapely.geometry import Polygon
from shapely.ops import unary_union

from warp_peel.events import EventPeeler
from warp_peel.io import load_scene


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scene", type=Path)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument(
        "--out", type=Path, default=Path("results/gpu-events/full-region-audit.json")
    )
    ap.add_argument("--merge-regions", action="store_true")
    args = ap.parse_args()
    subprocess.run(
        [
            "build/geometric_peel",
            "--scene",
            str(args.scene),
            "--method",
            "global-arrangement",
            "--threads",
            "4",
            "--out",
            str(args.work),
        ],
        check=True,
        timeout=600,
    )
    exact = json.loads((args.work / "exact.json").read_text())
    window = [float(Fraction(x)) for x in exact["window"]]

    def ring(points):
        return [
            tuple(
                (float(Fraction(v)) - window[i]) / (window[i + 2] - window[i])
                for i, v in enumerate(p)
            )
            for p in points
        ]

    # Rational constructions can collapse distinct vertices after conversion to
    # binary64. Apply the same explicit comparison grid to both outputs. This
    # affects validation only; it never repairs the GPU solver's output buffers.
    grid = 1e-12
    repairs = {"oracle": 0, "gpu": 0}

    def normalize(poly, side):
        if not poly.is_valid:
            repairs[side] += 1
            poly = make_valid(poly)
        return set_precision(poly, grid)

    expected = {}
    for layer in exact["layers"]:
        for region in layer["regions"]:
            expected[(layer["layer_index"] - 1, region["source_id"])] = unary_union(
                [
                    normalize(
                        Polygon(ring(d["outer"]), [ring(h) for h in d["holes"]]),
                        "oracle",
                    )
                    for d in region["domain"]
                ]
            )
    arrays, labels, _ = load_scene(args.scene)
    p = EventPeeler(*arrays, merge_regions=args.merge_regions)
    p.run()
    info = p.read_status()
    n = info["polygons"]
    polys = p.output.numpy()[:n]
    sources = p.output_sources.numpy()
    layers = p.output_layers.numpy()
    grouped = defaultdict(list)
    for i, poly in enumerate(polys):
        grouped[(int(layers[i]), int(labels[sources[i]]))].append(
            normalize(Polygon(poly["v"][: poly["n"]]), "gpu")
        )
    actual = {k: unary_union(v) for k, v in grouped.items()}
    errors = []
    for key in sorted(actual.keys() | expected.keys()):
        area = (
            actual.get(key, Polygon())
            .symmetric_difference(expected.get(key, Polygon()))
            .area
        )
        errors.append((key, area))
    tolerance = 1e-9
    failures = [
        {"layer_source": k, "symmetric_difference_area": a}
        for k, a in errors
        if a > tolerance
    ]
    report = {
        "scene": str(args.scene),
        "merge_regions": args.merge_regions,
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path("warp_peel").glob("*.py")
        },
        "scene_sha256": hashlib.sha256(args.scene.read_bytes()).hexdigest(),
        "oracle_binary_sha256": hashlib.sha256(
            Path("build/geometric_peel").read_bytes()
        ).hexdigest(),
        "oracle_output_sha256": hashlib.sha256(
            (args.work / "exact.json").read_bytes()
        ).hexdigest(),
        "oracle": "global-arrangement exact regions, independently validated against arrangement on repository fixtures",
        "scope": "Complete source-labelled polygon unions converted to binary64, made valid when needed, and snapped to the stated comparison grid on both sides; not an exact arithmetic certificate",
        "comparison_grid": grid,
        "invalid_polygons_after_float_conversion": repairs,
        "tolerance_normalized_area": tolerance,
        "compared_regions": len(errors),
        "maximum_symmetric_difference_area": max((a for _, a in errors), default=0),
        "sum_symmetric_difference_area": sum(a for _, a in errors),
        "failures": failures,
        "output": info,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
