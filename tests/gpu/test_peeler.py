"""CUDA integration checks against the independent exact arrangement oracle."""

import json
import os
import subprocess
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
import warp as wp
from shapely.geometry import Polygon
from shapely.ops import unary_union

from warp_peel import Capacities, DevicePeeler
from warp_peel import kernels as geometry
from warp_peel.io import load_scene

ROOT = Path(__file__).resolve().parents[2]
EXE = Path(os.environ.get("PEEL_EXE", ROOT / "build/geometric_peel"))


def regions(peeler, labels):
    info = peeler.read_status()
    polys = peeler.output.numpy()[: info["polygons"]]
    sources = peeler.output_sources.numpy()
    layers = peeler.output_layers.numpy()
    result = defaultdict(list)
    for i, p in enumerate(polys):
        result[(int(layers[i]), int(labels[sources[i]]))].append(
            Polygon(p["v"][: p["n"]])
        )
    return {k: unary_union(v) for k, v in result.items()}


def oracle(path, directory, layers=0):
    args = [
        str(EXE),
        "--scene",
        str(path),
        "--method",
        "arrangement",
        "--verify",
        "--out",
        str(directory),
    ]
    if layers:
        args += ["--layers", str(layers)]
    subprocess.run(args, check=True, capture_output=True, timeout=120)
    data = json.loads((directory / "exact.json").read_text())
    w = [float(Fraction(x)) for x in data["window"]]

    def ring(r):
        return [
            (
                (float(Fraction(x)) - w[0]) / (w[2] - w[0]),
                (float(Fraction(y)) - w[1]) / (w[3] - w[1]),
            )
            for x, y in r
        ]

    result = {}
    for layer in data["layers"]:
        for r in layer["regions"]:
            result[(layer["layer_index"] - 1, r["source_id"])] = unary_union(
                [
                    Polygon(ring(d["outer"]), [ring(h) for h in d["holes"]])
                    for d in r["domain"]
                ]
            )
    return result


def assert_regions(actual, expected, tolerance=1e-9):
    for key in actual.keys() | expected.keys():
        a, b = actual.get(key, Polygon()), expected.get(key, Polygon())
        assert a.symmetric_difference(b).area <= tolerance, (
            key,
            a.area,
            b.area,
            a.symmetric_difference(b).area,
        )


CASES = [
    f["name"]
    for f in json.loads((ROOT / "tests/fixtures/manifest.json").read_text())["fixtures"]
    if f["expected"] == "accept"
]


@pytest.mark.parametrize("name", CASES)
def test_oracle(name, tmp_path):
    path = ROOT / "tests/fixtures" / f"{name}.json"
    arrays, labels, _ = load_scene(path)
    p = DevicePeeler(*arrays)
    p.run()
    expected = oracle(path, tmp_path / "oracle")
    assert_regions(regions(p, labels), expected)
    # Independent triangle union and area checks catch holes/overlapping fans.
    info = p.read_status()
    tris = p.triangles_uv.numpy()[: info["triangles"]]
    sources = p.triangle_sources.numpy()
    layers = p.triangle_layers.numpy()
    grouped = defaultdict(list)
    for i, t in enumerate(tris):
        a, b = t[1] - t[0], t[2] - t[0]
        assert a[0] * b[1] - a[1] * b[0] > 0
        grouped[(int(layers[i]), int(labels[sources[i]]))].append(Polygon(t))
    mesh = {key: unary_union(v) for key, v in grouped.items()}
    assert_regions(mesh, expected)
    for key, v in grouped.items():
        assert abs(sum(t.area for t in v) - mesh[key].area) < 1e-9
    world = (
        p.triangles_world[: info["triangles"]].numpy()
        if info["triangles"]
        else np.empty((0, 3, 3))
    )
    if len(world):
        camera = p.camera.numpy()[0]
        settings = p.settings.numpy()
        transformed = world @ camera[:3, :3].T + camera[:3, 3]
        uv = transformed[:, :, :2].copy()
        if settings[4]:
            uv /= transformed[:, :, 2, None]
        uv = (uv - settings[:2]) / (settings[2:4] - settings[:2])
        np.testing.assert_allclose(uv, tris, atol=2e-9, rtol=0)
        inputs = p.vertices.numpy()[p.faces.numpy()[sources[: len(world)]]]
        normals = np.cross(inputs[:, 1] - inputs[:, 0], inputs[:, 2] - inputs[:, 0])
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        distances = np.sum((world - inputs[:, :1]) * normals[:, None], axis=2)
        np.testing.assert_allclose(distances, 0, atol=2e-8, rtol=0)
    # Replay resets all counters and emits the same labelled sets.
    p.run()
    assert_regions(regions(p, labels), expected)


@wp.kernel
def translate(vertices: wp.array(dtype=wp.vec3d), shift: wp.vec3d):
    i = wp.tid()
    vertices[i] = vertices[i] + shift


@wp.kernel
def bad_face(faces: wp.array(dtype=wp.vec3i)):
    faces[0] = wp.vec3i(-1, 0, 1000000)


@wp.kernel
def zero_camera(camera: wp.array(dtype=wp.mat44d)):
    camera[0] = wp.mat44d()


def test_external_capture_and_changed_input(tmp_path, monkeypatch):
    arrays, labels, data = load_scene(ROOT / "tests/fixtures/partial.json")
    p = DevicePeeler(*arrays)
    p.warmup()
    with monkeypatch.context() as m:
        m.setattr(
            wp.array, "numpy", lambda *a, **kw: pytest.fail("Host read during capture")
        )
        m.setattr(
            wp,
            "synchronize",
            lambda *a, **kw: pytest.fail("Host synchronization during capture"),
        )
        m.setattr(
            wp,
            "synchronize_stream",
            lambda *a, **kw: pytest.fail("Host synchronization during capture"),
        )
        with wp.ScopedStream(p.stream), wp.ScopedCapture(stream=p.stream) as capture:
            p.enqueue()
    graph = capture.graph
    wp.capture_launch(graph, stream=p.stream)
    before = regions(p, labels)
    wp.launch(
        translate, p.nv, inputs=[p.vertices, wp.vec3d(0.5, 0.25, 0.0)], stream=p.stream
    )
    wp.capture_launch(graph, stream=p.stream)
    after = regions(p, labels)
    assert any(before[k].symmetric_difference(after[k]).area > 1e-4 for k in before)
    for t in data["triangles"]:
        for v in t["vertices"]:
            v[0] = str(Fraction(v[0]) + Fraction(1, 2))
            v[1] = str(Fraction(v[1]) + Fraction(1, 4))
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(data))
    assert_regions(after, oracle(path, tmp_path / "oracle"))
    # Another workspace can grow Warp's scratch without invalidating this graph.
    q = DevicePeeler(*load_scene(ROOT / "tests/fixtures/perspective_torus.json")[0])
    q.run()
    q.read_status()
    wp.capture_launch(graph, stream=p.stream)
    assert_regions(regions(p, labels), after)
    dot = tmp_path / "graph.dot"
    wp.capture_debug_dot_print(graph, str(dot))
    text = dot.read_text()
    assert "CONDITIONAL" in text.upper()
    assert "DTOH" not in text.upper() and "HTOD" not in text.upper()
    assert "ALLOC" not in text.upper()


@pytest.mark.parametrize("limit", [1, 2])
def test_prefix(limit, tmp_path):
    arrays, labels, _ = load_scene(ROOT / "tests/fixtures/holes.json")
    p = DevicePeeler(*arrays, layers=limit)
    p.run()
    expected = oracle(ROOT / "tests/fixtures/holes.json", tmp_path / "oracle")
    assert_regions(
        regions(p, labels), {k: v for k, v in expected.items() if k[0] < limit}
    )


@pytest.mark.parametrize(
    "field,bit",
    [
        ("residual", 4),
        ("polygons", 8),
        ("vertices", 16),
        ("triangles", 32),
        ("layers", 64),
        ("pieces_per_worker", 2),
        ("edge_vertices", 128),
    ],
)
def test_overflow(field, bit):
    from dataclasses import replace

    arrays = load_scene(ROOT / "tests/fixtures/holes.json")[0]
    cap = replace(Capacities.for_mesh(arrays[1].size), **{field: 1})
    p = DevicePeeler(*arrays, capacities=cap)
    for _ in range(2):
        p.run()
        with pytest.raises(RuntimeError, match="GPU extraction failed"):
            p.read_status()
        assert p.status.numpy()[9] & bit


@pytest.mark.parametrize("invalid", ["index", "camera"])
def test_invalid_device_input(invalid):
    p = DevicePeeler(*load_scene(ROOT / "tests/fixtures/single.json")[0])
    p.run()
    p.read_status()
    if invalid == "index":
        wp.launch(bad_face, 1, inputs=[p.faces], stream=p.stream)
    else:
        wp.launch(zero_camera, 1, inputs=[p.camera], stream=p.stream)
    p.run()
    with pytest.raises(RuntimeError, match="invalid input"):
        p.read_status()


@pytest.mark.parametrize("name", ["partial", "holes", "closed_object"])
def test_mesh_has_no_t_junctions(name):
    from shapely.geometry import Point, box
    from shapely.strtree import STRtree

    p = DevicePeeler(*load_scene(ROOT / "tests/fixtures" / f"{name}.json")[0])
    p.run()
    info = p.read_status()
    triangles = p.triangles_uv[: info["triangles"]].numpy()
    layers = p.triangle_layers[: info["triangles"]].numpy()
    for layer in np.unique(layers):
        ts = triangles[layers == layer]
        points = np.unique(ts.reshape((-1, 2)), axis=0)
        tree = STRtree([Point(x) for x in points])
        for t in ts:
            for a, b in zip(t, np.roll(t, -1, axis=0)):
                lo, hi = np.minimum(a, b) - 1e-10, np.maximum(a, b) + 1e-10
                candidates = points[tree.query(box(*lo, *hi))]
                d = b - a
                tval = (candidates - a) @ d / (d @ d)
                distance = np.abs(
                    d[0] * (candidates[:, 1] - a[1]) - d[1] * (candidates[:, 0] - a[0])
                ) / np.linalg.norm(d)
                assert not np.any(
                    (tval > 1e-8) & (tval < 1 - 1e-8) & (distance < 1e-11)
                )


@wp.kernel
def move_camera(camera: wp.array(dtype=wp.mat44d)):
    m = camera[0]
    m[0, 3] += wp.float64(0.5)
    camera[0] = m


def test_device_camera_update(tmp_path):
    arrays, labels, data = load_scene(ROOT / "tests/fixtures/partial.json")
    p = DevicePeeler(*arrays)
    p.run()
    before = regions(p, labels)
    wp.launch(move_camera, 1, inputs=[p.camera], stream=p.stream)
    p.run()
    after = regions(p, labels)
    assert any(before[k].symmetric_difference(after[k]).area > 1e-4 for k in before)
    data["camera"]["world_to_camera"] = [[1, 0, 0, 0.5], [0, 1, 0, 0], [0, 0, 1, 0]]
    path = tmp_path / "camera.json"
    path.write_text(json.dumps(data))
    assert_regions(after, oracle(path, tmp_path / "oracle"))


def test_device_welding_signed_zero():
    # Sorting must agree with equality even when the input contains signed zeros.
    vertices = np.array(
        [[0, 2, 1], [-0.0, 1, 1], [0, 1, 1], [-0.0, 2, 1]], dtype=np.float64
    )
    p = DevicePeeler(
        wp.array(vertices, dtype=wp.vec3d, device="cuda:0"),
        wp.empty(0, dtype=wp.vec3i, device="cuda:0"),
        wp.array(np.eye(4)[None], dtype=wp.mat44d, device="cuda:0"),
        wp.array([-5.0, -5.0, 5.0, 5.0, 0.0, 1.0], dtype=wp.float64, device="cuda:0"),
    )
    p.run()
    p.read_status()
    ids = p.canonical.numpy()
    assert ids[0] == ids[3] and ids[1] == ids[2] and ids[0] != ids[1]


def test_empty_download():
    from warp_peel.io import download_mesh

    p = DevicePeeler(*load_scene(ROOT / "tests/fixtures/empty.json")[0])
    p.run()
    data = download_mesh(p)
    assert data["world"].shape == (0, 3, 3)


@wp.kernel
def clean_backtracking_edge(output: wp.array(dtype=geometry.Polygon)):
    p = geometry.Polygon()
    p = geometry.append(p, wp.vec2d(0.0, 0.0))
    p = geometry.append(p, wp.vec2d(1.0, 0.0))
    p = geometry.append(
        p, wp.vec2d(wp.float64(1.0) - wp.float64(2e-12), wp.float64(0.0))
    )
    p = geometry.append(p, wp.vec2d(1.0, 1.0))
    p = geometry.append(p, wp.vec2d(0.0, 1.0))
    output[0] = geometry.clean(p)


def test_roundoff_edge_does_not_erase_polygon():
    output = wp.zeros(1, dtype=geometry.Polygon, device="cuda:0")
    wp.launch(clean_backtracking_edge, 1, inputs=[output], device="cuda:0")
    wp.synchronize()
    p = output.numpy()[0]
    assert p["n"] >= 3
    assert abs(Polygon(p["v"][: p["n"]]).area - 1.0) < 1e-10


@wp.kernel
def split_short_edge(output: wp.array(dtype=geometry.Polygon)):
    a = wp.vec2d(0.3, 0.3)
    h = wp.float64(1e-6)
    p = geometry.Polygon()
    p = geometry.append(p, a + wp.vec2d(-h, -h))
    p = geometry.append(p, a + wp.vec2d(h, -h))
    p = geometry.append(p, a + wp.vec2d(h, h))
    p = geometry.append(p, a + wp.vec2d(-h, h))
    b = a + wp.vec2d(1e-12, 2e-12)
    inside, outside = geometry.split_edge(p, a, b)
    output[0] = inside
    output[1] = outside


def test_short_clipping_edge_keeps_its_position():
    output = wp.zeros(2, dtype=geometry.Polygon, device="cuda:0")
    wp.launch(split_short_edge, 1, inputs=[output], device="cuda:0")
    wp.synchronize()
    polygons = output.numpy()
    areas = [Polygon(p["v"][: p["n"]]).area for p in polygons]
    assert abs(areas[0] - areas[1]) < 1e-20
    assert abs(sum(areas) - 4e-12) < 1e-20
