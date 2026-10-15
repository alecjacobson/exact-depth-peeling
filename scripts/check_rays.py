#!/usr/bin/env python3
"""Independent rational ray/3D-triangle spot checks; not a region certificate."""

import argparse
import json
from fractions import Fraction as Q
from itertools import pairwise
from pathlib import Path


def subtract(a, b):
    return tuple(x - y for x, y in zip(a, b))


def cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("exact", type=Path)
    parser.add_argument("--minimum-layer", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.exact.read_text())
    matrix = [[Q(x) for x in row] for row in data["camera"]["world_to_camera"]]
    perspective = data["camera"]["type"] == "perspective"
    near = Q(data["camera"]["near"])
    sources = []
    for source in data["sources"]:
        world = [tuple(map(Q, point)) for point in source["world_vertices"]]
        camera = [
            tuple(dot(row[:3], point) + row[3] for row in matrix) for point in world
        ]
        sources.append(
            (
                source["source_id"],
                camera[0],
                subtract(camera[1], camera[0]),
                subtract(camera[2], camera[0]),
            )
        )
    queries = []
    for layer in data["layers"]:
        rank = layer["layer_index"]
        if rank < args.minimum_layer:
            continue
        for face in layer["faces"]:
            points = [tuple(map(Q, point)) for point in face["uv"]]
            for weight in range(1, 129):
                weights = (1, weight, weight * weight + 1)
                point = tuple(
                    sum(weights[i] * points[i][axis] for i in range(3)) / sum(weights)
                    for axis in range(2)
                )
                origin = (Q(0), Q(0), Q(0)) if perspective else (*point, Q(0))
                direction = (*point, Q(1)) if perspective else (Q(0), Q(0), Q(1))
                hits, boundary = [], False
                for ident, vertex, edge1, edge2 in sources:
                    h = cross(direction, edge2)
                    determinant = dot(edge1, h)
                    if determinant == 0:
                        continue
                    s = subtract(origin, vertex)
                    b = dot(s, h) / determinant
                    q = cross(s, edge1)
                    c = dot(direction, q) / determinant
                    t = dot(edge2, q) / determinant
                    if b < 0 or c < 0 or b + c > 1 or (perspective and t < near):
                        continue
                    if b == 0 or c == 0 or b + c == 1:
                        boundary = True
                        break
                    hits.append((t, ident))
                hits.sort()
                if boundary or any(a[0] == b[0] for a, b in pairwise(hits)):
                    continue
                if len(hits) < rank or hits[rank - 1][1] != face["source_id"]:
                    raise AssertionError((rank, face["source_id"], hits))
                queries.append(
                    {
                        "layer": rank,
                        "source_id": face["source_id"],
                        "uv": list(map(str, point)),
                        "ray_source_order": [ident for _, ident in hits],
                        "camera_depths": [str(t) for t, _ in hits],
                    }
                )
                break
            else:
                raise RuntimeError(
                    "Could not find a sample away from source edges and depth ties"
                )
    if not queries:
        raise RuntimeError("No faces at the requested minimum layer")
    report = {
        "status": "passed",
        "exact_input": str(args.exact),
        "minimum_layer": args.minimum_layer,
        "query_count": len(queries),
        "maximum_observed_hits": max(len(q["ray_source_order"]) for q in queries),
        "scope": "One interior exact ray per selected exported face; samples do not certify entire regions",
        "queries": queries,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"Passed {len(queries)} independent exact ray queries; maximum observed hits {report['maximum_observed_hits']}"
    )


if __name__ == "__main__":
    main()
