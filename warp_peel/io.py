"""Optional host serialization adapters. All geometric processing is in Warp kernels."""

import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import warp as wp


def load_scene(path, device="cuda:0"):
    data = json.loads(Path(path).read_text())
    number = lambda x: float(Fraction(x)) if isinstance(x, str) else float(x)
    triangles = data["triangles"]
    vertices = np.array(
        [[[number(x) for x in p] for p in t["vertices"]] for t in triangles],
        dtype=np.float64,
    ).reshape((-1, 3))
    faces = np.arange(3 * len(triangles), dtype=np.int32).reshape((-1, 3))
    camera = np.eye(4, dtype=np.float64)
    cam = data.get("camera", {})
    if cam.get("type", "orthographic") not in ("orthographic", "perspective"):
        raise ValueError("Unknown camera type")
    if "world_to_camera" in cam:
        camera[:3] = [[number(x) for x in row] for row in cam["world_to_camera"]]
    settings = np.array(
        [number(x) for x in data["window"]]
        + [int(cam.get("type") == "perspective"), number(cam.get("near", 0.01))],
        dtype=np.float64,
    )
    if (
        not np.isfinite(vertices).all()
        or not np.isfinite(camera).all()
        or not np.isfinite(settings).all()
    ):
        raise ValueError("Input must be finite")
    if settings[2] <= settings[0] or settings[3] <= settings[1] or settings[5] <= 0:
        raise ValueError("Invalid image window or near plane")
    labels = np.array([t.get("id", i) for i, t in enumerate(triangles)], dtype=np.int64)
    if len(set(labels)) != len(labels):
        raise ValueError("Source IDs must be unique")
    arrays = [
        wp.array(vertices, dtype=wp.vec3d, device=device),
        wp.array(faces, dtype=wp.vec3i, device=device),
        wp.array(camera[None], dtype=wp.mat44d, device=device),
        wp.array(settings, dtype=wp.float64, device=device),
    ]
    return arrays, labels, data


def download_mesh(peeler, labels=None):
    """Explicit host transfer for serialization; not part of the GPU pipeline."""
    info = peeler.read_status()
    n = info["triangles"]
    if n == 0:
        return {
            "uv": np.empty((0, 3, 2)),
            "world": np.empty((0, 3, 3)),
            "sources": np.empty(0, dtype=np.int64),
            "layers": np.empty(0, dtype=np.int32),
            "status": info,
        }
    sources = peeler.triangle_sources[:n].numpy()
    if labels is not None:
        sources = labels[sources]
    return {
        "uv": peeler.triangles_uv[:n].numpy(),
        "world": peeler.triangles_world[:n].numpy(),
        "sources": sources,
        "layers": peeler.triangle_layers[:n].numpy(),
        "status": info,
    }
