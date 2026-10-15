#!/usr/bin/env python3
"""Losslessly pack exported OBJ coordinates and source IDs for Blender/review."""

import argparse
import json
from pathlib import Path

import numpy as np
from dragon_scene import read_ply


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--layers", type=Path, required=True)
    p.add_argument(
        "--mesh",
        type=Path,
        default=Path("../sparse-solver-benchmark/xyzrgb_dragon-720K.ply"),
    )
    p.add_argument("--output", type=Path, default=Path("results/dragon/meshes.npz"))
    a = p.parse_args()
    v, f = read_ply(a.mesh)
    data = {"input_vertices": v, "input_faces": f}
    files = sorted(
        a.layers.glob("layer-*.obj"), key=lambda p: int(p.stem.split("-")[-1])
    )
    for k, path in enumerate(files):
        vertices = []
        faces = []
        sources = []
        source = -1
        with path.open() as stream:
            for line in stream:
                if line.startswith("v "):
                    vertices.append(tuple(map(float, line.split()[1:4])))
                elif line.startswith("g source_"):
                    source = int(line.strip().split("_")[-1])
                elif line.startswith("f "):
                    faces.append(tuple(int(x) - 1 for x in line.split()[1:4]))
                    sources.append(source)
        vv = np.array(vertices, dtype=np.float64).reshape(-1, 3)
        ff = np.array(faces, dtype=np.int32).reshape(-1, 3)
        del vertices, faces
        vv, inv = np.unique(vv, axis=0, return_inverse=True)
        data[f"vertices_{k}"] = vv
        data[f"faces_{k}"] = inv[ff].astype(np.int32)
        data[f"sources_{k}"] = np.array(sources, dtype=np.int32)
        print(k + 1, len(ff), len(vv), flush=True)
    data["layer_count"] = len(files)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.output, **data)
    metrics = json.loads((a.layers / "metrics.json").read_text())
    assert len(files) == len(metrics["passes"])
    assert all(
        len(data[f"faces_{i}"]) == m["exported_triangles"]
        for i, m in enumerate(metrics["passes"])
    )


if __name__ == "__main__":
    main()
