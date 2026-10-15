#!/usr/bin/env python3
"""Generate connected closed curved meshes for complete depth-peeling measurements.

Coordinates are quantized once on generation, then stored as exact rationals.
No approximation or simplification takes place in the geometry solver.
"""

import argparse
import json
import math
from collections import Counter
from fractions import Fraction
from pathlib import Path


def make_mesh(nu, nv, shape="sculpted-torus"):
    verts = []
    for i in range(nu):
        u = 2 * math.pi * i / nu
        for j in range(nv):
            v = 2 * math.pi * j / nv
            major = 1.6 + (0.16 * math.cos(3 * u) if shape == "sculpted-torus" else 0)
            minor = 0.5 + (0.08 * math.sin(5 * u) if shape == "sculpted-torus" else 0)
            p = (
                (major + minor * math.cos(v)) * math.cos(u),
                (major + minor * math.cos(v)) * math.sin(u),
                minor * math.sin(v)
                + (0.22 * math.sin(3 * u) if shape == "sculpted-torus" else 0),
            )
            verts.append([str(Fraction(round(x * 1_000_000), 1_000_000)) for x in p])
    faces = []
    for i in range(nu):
        for j in range(nv):
            a = i * nv + j
            b = ((i + 1) % nu) * nv + j
            c = ((i + 1) % nu) * nv + (j + 1) % nv
            d = i * nv + (j + 1) % nv
            faces.extend(((a, b, c), (a, c, d)))
    counts = Counter(
        tuple(sorted((f[j], f[(j + 1) % 3]))) for f in faces for j in range(3)
    )
    assert all(v == 2 for v in counts.values()), "Mesh must be closed and manifold"
    return {
        "window": ["-1", "-1", "1", "1"],
        "camera": {
            "type": "perspective",
            "near": "1/10",
            "world_to_camera": [
                ["1", "0", "0", "0"],
                ["0", "3/5", "-4/5", "0"],
                ["0", "4/5", "3/5", "7"],
            ],
        },
        "triangles": [
            {"id": i, "vertices": [verts[j] for j in f]} for i, f in enumerate(faces)
        ],
        "metadata": {
            "generator": "scripts/large_mesh_demo.py",
            "shape": shape,
            "nu": nu,
            "nv": nv,
            "unique_vertices": len(verts),
            "edges": len(counts),
            "triangles": len(faces),
            "closed_two_manifold": True,
            "connected": True,
            "quantization_denominator": 1_000_000,
            "backface_culling": False,
            "license": "CC0-1.0",
        },
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nu", type=int, default=128)
    p.add_argument("--nv", type=int, default=40)
    p.add_argument(
        "--shape", choices=["torus", "sculpted-torus"], default="sculpted-torus"
    )
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if min(a.nu, a.nv) < 3:
        p.error("Both grid dimensions must be at least 3")
    s = make_mesh(a.nu, a.nv, a.shape)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(s, separators=(",", ":")) + "\n")
    print(json.dumps(s["metadata"]))


if __name__ == "__main__":
    main()
