#!/usr/bin/env python3
"""Import the full supplied binary PLY without decimation; retain source face IDs."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def read_ply(path):
    with path.open("rb") as f:
        header = []
        while True:
            line = f.readline().decode("ascii").strip()
            header.append(line)
            if line == "end_header":
                break
        assert "format binary_little_endian 1.0" in header
        nv = int(next(x.split()[-1] for x in header if x.startswith("element vertex")))
        nf = int(next(x.split()[-1] for x in header if x.startswith("element face")))
        assert header[
            header.index(f"element vertex {nv}") + 1 : header.index(
                f"element face {nf}"
            )
        ] == ["property double x", "property double y", "property double z"]
        assert "property list int int vertex_indices" in header
        v = np.fromfile(f, dtype="<f8", count=nv * 3).reshape(-1, 3)
        faces = np.fromfile(f, dtype="<i4", count=nf * 4).reshape(-1, 4)
        assert (faces[:, 0] == 3).all()
        return v, faces[:, 1:]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mesh",
        type=Path,
        default=Path("../sparse-solver-benchmark/xyzrgb_dragon-720K.ply"),
    )
    p.add_argument("--output", type=Path, default=Path("results/dragon/scene.json"))
    a = p.parse_args()
    v, f = read_ply(a.mesh)
    # Model's Z is up. Oblique front view gives a readable dragon silhouette.
    center = (v.min(0) + v.max(0)) / 2
    direction = np.array([0.18, -1.0, 0.12])
    direction /= np.linalg.norm(direction)
    eye = center + direction * 370
    forward = (center - eye) / np.linalg.norm(center - eye)
    right = np.cross(forward, [0, 0, 1])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    rotation = np.stack([right, up, forward])
    matrix = np.column_stack([rotation, -rotation @ eye])
    camera = v @ rotation.T + matrix[:, 3]
    uv = camera[:, :2] / camera[:, 2:]
    mid = (uv.min(0) + uv.max(0)) / 2
    half = (uv.max(0) - uv.min(0)) * 0.57
    window = [*(mid - half), *(mid + half)]
    meta = {
        "input_path": str(a.mesh),
        "input_sha256": hashlib.sha256(a.mesh.read_bytes()).hexdigest(),
        "input_vertices": len(v),
        "input_faces": len(f),
        "decimated": False,
        "world_bounds": [v.min(0).tolist(), v.max(0).tolist()],
        "camera_position": eye.tolist(),
        "camera_target": center.tolist(),
        "camera": {
            "type": "perspective",
            "near": 1,
            "world_to_camera": matrix.tolist(),
        },
        "window": window,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    # Stream triangle records so conversion needs no large Python JSON object graph.
    with a.output.open("w") as out:
        out.write(
            '{"camera":'
            + json.dumps(meta["camera"])
            + ',"window":'
            + json.dumps(window)
            + ',"triangles":['
        )
        for i, face in enumerate(f):
            if i:
                out.write(",")
            out.write(
                json.dumps(
                    {"id": i, "mesh_face_id": i, "vertices": v[face].tolist()},
                    separators=(",", ":"),
                )
            )
        out.write("]}\n")
    (a.output.parent / "input.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
