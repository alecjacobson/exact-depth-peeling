#!/usr/bin/env python3
"""Deterministic exact rational scenes. Original synthetic data: CC0-1.0."""

import argparse
import json
import random
from fractions import Fraction as F
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def q(x):
    f = F(x)
    return str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"


def tri(vertices, ident=None):
    d = {"vertices": [[q(x) for x in v] for v in vertices]}
    if ident is not None:
        d["id"] = ident
    return d


def flat(points, z):
    return tri([(x, y, z) for x, y in points])


def rectangle(x0, y0, x1, y1, z):
    return [
        flat([(x0, y0), (x1, y0), (x1, y1)], z),
        flat([(x0, y0), (x1, y1), (x0, y1)], z),
    ]


def scene(triangles, window=(-5, -5, 5, 5), **extra):
    for i, t in enumerate(triangles):
        t.setdefault("id", i)
    return {
        "window": list(map(q, window)),
        "camera": {"type": "orthographic"},
        "triangles": triangles,
        "metadata": {
            "source": "scripts/generate_fixtures.py",
            "license": "CC0-1.0",
            "backface_culling": False,
        },
        **extra,
    }


def cover(g, foreground=True):
    ts = [flat([(-4, -3), (4, -3), (0, 4)], 1)] if foreground else []
    # [-1,1]^2 lies strictly inside the foreground triangle.
    for i in range(g):
        a = -1 + F(2 * i, g)
        b = a + F(1, 2 * g)
        ts += rectangle(-1, a, 1, b, 2)
        ts += rectangle(a, -1, b, 1, 3)
    return scene(ts)


def stacked(n):
    return scene([flat([(-3, -3), (3, -3), (0, 3)], i + 1) for i in range(n)])


def disjoint(n):
    return scene(
        [flat([(2 * i, 0), (2 * i + 1, 0), (2 * i, 1)], 1) for i in range(n)],
        (-1, -1, 2 * n + 1, 2),
    )


def mesh(n=16):
    # Closed triangulated box with rational gridded faces, 12*n*n triangles.
    ts = []
    for axis in range(3):
        other = [i for i in range(3) if i != axis]
        for side in (-1, 1):
            for i in range(n):
                for j in range(n):
                    vs = []
                    for a, b in ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)):
                        v = [F(0)] * 3
                        v[axis] = F(side)
                        v[other[0]] = -1 + F(2 * a, n)
                        v[other[1]] = -1 + F(2 * b, n)
                        # Invertible rational shear avoids edge-on side faces.
                        x, y, z = v
                        vs.append((x + z / 4, y + z / 5, z + 3 + x / 7 + y / 9))
                    ts += [tri([vs[0], vs[1], vs[2]]), tri([vs[0], vs[2], vs[3]])]
    return scene(ts, (-3, -3, 3, 3))


def perforated(g):
    triangles = [flat([(-4, -3), (4, -3), (0, 4)], 4)]
    for i in range(g):
        for j in range(g):
            x, y = -1 + F(2 * i, g), -1 + F(2 * j, g)
            triangles.append(flat([(x, y), (x + F(1, g), y), (x, y + F(1, g))], 1))
    return scene(triangles)


def nested_mesh(n):
    outer = mesh(n)
    inner = mesh(n)
    for triangle in inner["triangles"]:
        triangle.pop("id", None)
        for vertex in triangle["vertices"]:
            x, y, z = map(F, vertex)
            vertex[:] = list(map(q, (x / 2, y / 2, 3 + (z - 3) / 2)))
    triangles = outer["triangles"] + inner["triangles"]
    for t in triangles:
        t.pop("id", None)
    return scene(triangles, (-3, -3, 3, 3))


def perspective_view(data, center_z, distance=6):
    # Exact orthonormal rotation (5,12,13), then a positive camera-space depth.
    c, s = F(5, 13), F(12, 13)
    data["camera"] = {
        "type": "perspective",
        "near": "1/2",
        "world_to_camera": [
            list(map(q, (c, 0, s, -s * center_z))),
            list(map(q, (0, 1, 0, 0))),
            list(map(q, (-s, 0, c, distance - c * center_z))),
        ],
    }
    data["window"] = ["-1", "-1", "1", "1"]
    return data


def torus(major_quadrant=4, minor_quadrant=3):
    # Rational unit-circle samples; no trigonometric rounding enters the fixture.
    def circle(n):
        result = []
        for quadrant in range(4):
            for k in range(n):
                t = F(k, n)
                x, y = (1 - t * t) / (1 + t * t), 2 * t / (1 + t * t)
                for _ in range(quadrant):
                    x, y = -y, x
                result.append((x, y))
        return result

    major, minor = circle(major_quadrant), circle(minor_quadrant)
    vertices = [
        [
            ((2 + F(3, 4) * cp) * ct, (2 + F(3, 4) * cp) * st, 5 + F(3, 4) * sp)
            for cp, sp in minor
        ]
        for ct, st in major
    ]
    triangles = []
    for i in range(len(major)):
        for j in range(len(minor)):
            a, b, c, d = (
                vertices[i][j],
                vertices[(i + 1) % len(major)][j],
                vertices[(i + 1) % len(major)][(j + 1) % len(minor)],
                vertices[i][(j + 1) % len(minor)],
            )
            triangles.extend((tri([a, b, c]), tri([a, c, d])))
    return perspective_view(scene(triangles), 5)


def fixtures():
    base = [(-2, -2), (2, -2), (0, 2)]
    fs = {
        "single": scene([flat(base, 1)]),
        "window_clipped": scene([flat(base, 1)], (-1, -1, 1, 1)),
        "disjoint": disjoint(3),
        "stacked": stacked(4),
        "partial": scene([flat(base, 3), flat([(-1, -3), (3, -1), (-1, 1)], 1)]),
        "holes": scene(
            [
                flat([(-4, -3), (4, -3), (0, 4)], 4),
                flat([(-1, -1), (0, -1), (-1, 0)], 1),
                flat([(0, 0), (1, 0), (0, 1)], 2),
            ]
        ),
        "components": scene(
            [flat(base, 4)]
            + rectangle(-3, -1, 3, F(-3, 4), 1)
            + rectangle(-3, 0, 3, F(1, 4), 2)
        ),
        "crossing": scene([tri([(-2, -2, 1), (2, -2, 5), (0, 2, 3)]), flat(base, 3)]),
        "coplanar_adjacent": scene(rectangle(-2, -2, 2, 2, 1)),
        "shared_vertex": scene(
            [
                flat([(0, 0), (2, 0), (0, 2)], 1),
                flat([(0, 0), (0, 2), (-2, 0)], 1),
                flat([(0, 0), (-2, 0), (0, -2)], 1),
                flat([(0, 0), (0, -2), (2, 0)], 1),
            ]
        ),
        "tie": scene([flat(base, 1), flat(base, 1)]),
        "hidden_tie": scene(
            [flat([(-4, -4), (4, -4), (0, 4)], 0), flat(base, 2), flat(base, 2)]
        ),
        "slivers": scene(
            [
                flat([(0, 0), (2, 0), (2, F(1, 10**12))], 1),
                flat([(0, 0), (2, F(1, 10**13)), (0, 1)], 1 + F(1, 10**15)),
            ]
        ),
        "edge_on": scene([tri([(0, 0, 1), (0, 0, 2), (0, 1, 2)]), flat(base, 3)]),
        "closed_object": mesh(1),
        "cover_2": cover(2),
        "empty": scene([]),
        "binary_float": scene(
            [{"vertices": [[0.1, 0.2, 1.1], [1.3, 0.2, 1.1], [0.1, 1.4, 1.1]]}]
        ),
    }
    a = [(-3, -2), (3, -2), (0, F(-8, 5))]
    b = [(-3, F(-5, 2)), (F(-5, 2), F(-5, 2)), (F(1, 2), 3)]
    c = [(F(5, 2), F(-5, 2)), (3, F(-5, 2)), (F(-1, 2), 3)]
    fs["cyclic"] = scene(
        [flat(a, 3), tri([(x, y, 3 - y / 2) for x, y in b]), flat(c, 2)],
        expected_cycle=[[0, 1], [1, 2], [2, 0]],
    )
    for name in ("partial", "crossing"):
        src = fs[name]
        ts = []
        for t in src["triangles"]:
            vs = []
            for x, y, z in t["vertices"]:
                x, y, z = map(F, (x, y, z))
                depth = 1 / (1 - z / 10)
                vs.append((x * depth, y * depth, depth))
            ts.append(tri(vs))
        fs["perspective_" + name] = scene(
            ts, camera={"type": "perspective", "near": "1/10"}
        )
    fs["perspective_near"] = scene(
        [
            tri([(-1, -1, F(1, 4)), (2, -1, 2), (0, 2, 2)]),
            tri([(-2, -2, 3), (2, -2, 3), (0, 2, 3)]),
        ],
        camera={"type": "perspective", "near": "1/2"},
    )
    fs["perspective_behind"] = scene(
        [tri([(-1, -1, -1), (2, -1, 2), (0, 2, 2)])],
        camera={"type": "perspective", "near": "1/2"},
    )
    fs["camera_transform"] = scene(
        [tri([(0, 0, 2), (2, 0, 3), (0, 2, 4)])],
        camera={
            "type": "perspective",
            "near": "1/5",
            "world_to_camera": [[1, 0, "1/3", 1], [0, 1, 0, -1], [0, 0, 1, 1]],
        },
    )
    fs["perspective_torus"] = torus(1, 1)
    for seed in range(12):
        rng = random.Random(seed)
        ts = [
            tri(
                [
                    (
                        rng.randrange(-4, 5),
                        rng.randrange(-4, 5),
                        F(rng.randrange(1, 40), 7) + F(i, 101),
                    )
                    for _ in range(3)
                ]
            )
            for i in range(5)
        ]
        fs[f"random_{seed:02}"] = scene(ts, seed=seed)
    return fs


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=ROOT / "tests/fixtures")
    args = p.parse_args()
    fs = fixtures()
    for name, data in fs.items():
        write(args.output / (name + ".json"), data)
    write(
        args.output / "manifest.json",
        {
            "generator": "scripts/generate_fixtures.py",
            "license": "CC0-1.0",
            "fixtures": [
                {
                    "name": n,
                    "expected": "reject_positive_area_tie"
                    if n in ("tie", "hidden_tie")
                    else "accept",
                }
                for n in fs
            ],
        },
    )
    print(f"Generated {len(fs)} fixtures in {args.output}")


if __name__ == "__main__":
    main()
