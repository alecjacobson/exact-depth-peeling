#!/usr/bin/env python3
"""Display-resolution coverage outlines; never modify authoritative solver meshes."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assets", type=Path, default=Path("results/dragon"))
    p.add_argument("--width", type=int, default=2800)
    a = p.parse_args()
    data = np.load(a.assets / "meshes.npz")
    meta = json.loads((a.assets / "input.json").read_text())
    matrix = np.array(meta["camera"]["world_to_camera"])
    r = matrix[:, :3]
    t = matrix[:, 3]
    inverse = np.linalg.inv(r)
    window = np.array(meta["window"])
    extent = window[2:] - window[:2]
    height = round(a.width * extent[1] / extent[0])
    factor = np.array([a.width - 1, height - 1]) / extent
    input_cam = data["input_vertices"] @ r.T + t
    inputs = data["input_faces"]
    result = {"raster_size": np.array([a.width, height])}
    for k in range(int(data["layer_count"])):
        world = data[f"vertices_{k}"]
        faces = data[f"faces_{k}"]
        sources = data[f"sources_{k}"]
        cam = world @ r.T + t
        uv = cam[:, :2] / cam[:, 2:]
        pixels = np.rint((uv - window[:2]) * factor).astype(np.int32)
        mask = np.zeros((height, a.width), np.uint8)
        # Union in the coverage raster, so subpixel cracks are not enlarged into thick lines.
        for start in range(0, len(faces), 20000):
            for poly in pixels[faces[start : start + 20000]]:
                cv2.fillConvexPoly(mask, poly, 255)
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        rings = []
        edges = []
        offset = 0
        for contour in contours:
            ring = cv2.approxPolyDP(contour, 0.65, True)[:, 0, :]
            if len(ring) < 3:
                continue
            rings.append(ring)
            edges.extend(
                (offset + i, offset + (i + 1) % len(ring)) for i in range(len(ring))
            )
            offset += len(ring)
        xy = np.concatenate(rings) if rings else np.empty((0, 2))
        boundary_uv = xy / factor + window[:2]
        nearest = cKDTree(uv).query(boundary_uv)[1]
        vertex_source = np.empty(len(world), np.int32)
        vertex_source[faces.ravel()] = np.repeat(sources, 3)
        tri = input_cam[inputs[vertex_source[nearest]]]
        normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        d = np.einsum("ij,ij->i", normal, tri[:, 0])
        rays = np.column_stack([boundary_uv, np.ones(len(xy))])
        z = d / np.einsum("ij,ij->i", normal, rays)
        boundary_world = (rays * z[:, None] - t) @ inverse.T
        result[f"world_{k}"] = boundary_world
        result[f"edges_{k}"] = np.array(edges, dtype=np.int32).reshape(-1, 2)
        print(k + 1, len(xy), flush=True)
    np.savez_compressed(a.assets / "outlines.npz", **result)


if __name__ == "__main__":
    main()
