"""Checks for the experimental device contour/shared-rank route."""

import json
import subprocess
from dataclasses import replace

import numpy as np
import pytest
import test_peeler as reference_tests
import warp as wp
from test_peeler import CASES, ROOT, assert_regions, oracle, regions
from test_peeler import test_external_capture_and_changed_input as check_capture
from test_peeler import test_oracle as check_complete_output
from test_peeler import test_prefix as check_prefix

from warp_peel.events import EventCapacities, EventPeeler
from warp_peel.io import load_scene

# These contain unresolved 3D crossings, outside the contour input contract.
CROSSING = {"crossing", "perspective_crossing"} | {f"random_{i:02}" for i in range(12)}


@pytest.mark.parametrize("name", [n for n in CASES if n not in CROSSING])
def test_complete_oracle(name, tmp_path, monkeypatch):
    monkeypatch.setattr(reference_tests, "DevicePeeler", EventPeeler)
    check_complete_output(name, tmp_path)


@pytest.mark.parametrize("name", sorted(CROSSING))
def test_resolved_oracle(name, tmp_path):
    path = tmp_path / "resolved.json"
    subprocess.run(
        [
            str(ROOT / "build/resolve_intersections"),
            str(ROOT / "tests/fixtures" / f"{name}.json"),
            str(path),
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )
    arrays, labels, _ = load_scene(path)
    p = EventPeeler(*arrays)
    p.run()
    assert_regions(regions(p, labels), oracle(path, tmp_path / "oracle"))


def test_capture(tmp_path, monkeypatch):
    monkeypatch.setattr(reference_tests, "DevicePeeler", EventPeeler)
    check_capture(tmp_path, monkeypatch)


@pytest.mark.parametrize("limit", [1, 2])
def test_prefix(limit, tmp_path, monkeypatch):
    monkeypatch.setattr(reference_tests, "DevicePeeler", EventPeeler)
    check_prefix(limit, tmp_path)


def stack(n):
    # Cyclic vertex permutations must still share the sample and contour records.
    vertices = np.array(
        [[-0.75, -0.75, 0], [0.75, -0.75, 0], [0, 0.75, 0]], dtype=np.float64
    )
    mesh = np.stack([np.roll(vertices, i % 3, axis=0) for i in range(n)])
    mesh[:, :, 2] = np.arange(1, n + 1)[:, None]
    return [
        wp.array(mesh.reshape(-1, 3), dtype=wp.vec3d, device="cuda"),
        wp.array(np.arange(3 * n).reshape(-1, 3), dtype=wp.vec3i, device="cuda"),
        wp.array(np.eye(4)[None], dtype=wp.mat44d, device="cuda"),
        wp.array([-1.0, -1.0, 1.0, 1.0, 0.0, 0.01], dtype=wp.float64, device="cuda"),
    ]


@pytest.mark.parametrize("n", [16, 128, 1024])
def test_shared_stack(n):
    p = EventPeeler(*stack(n))
    p.run()
    info = p.read_status()
    assert info["polygons"] == n
    assert info["splits"] == 0
    assert info["unique_contours"] == 3
    assert info["ray_groups"] == 1
    assert info["ray_candidate_tests"] == 2 * n
    assert info["ray_hits"] == n
    assert info["contour_tests"] <= 3 * n
    np.testing.assert_array_equal(
        p.output_layers.numpy()[:n], p.output_sources.numpy()[:n]
    )


@pytest.mark.parametrize(
    "field,match",
    [
        ("hits", "shared ray hit"),
        ("rounds", "subdivision round"),
        ("frontier", "residual pool"),
        ("regions", "output polygon"),
    ],
)
def test_capacity(field, match):
    arrays, _, _ = load_scene(ROOT / "tests/fixtures/holes.json")
    cap = replace(EventCapacities.for_mesh(arrays[1].size), **{field: 1})
    p = EventPeeler(*arrays, capacities=cap)
    p.run()
    with pytest.raises(RuntimeError, match=match):
        p.read_status()


@pytest.mark.parametrize("invalid", ["index", "camera"])
def test_invalid_input(invalid, monkeypatch):
    monkeypatch.setattr(reference_tests, "DevicePeeler", EventPeeler)
    reference_tests.test_invalid_device_input(invalid)


@pytest.mark.parametrize("name", ["partial", "holes", "closed_object"])
def test_seams(name, monkeypatch):
    monkeypatch.setattr(reference_tests, "DevicePeeler", EventPeeler)
    reference_tests.test_mesh_has_no_t_junctions(name)


def test_camera_update(tmp_path, monkeypatch):
    monkeypatch.setattr(reference_tests, "DevicePeeler", EventPeeler)
    reference_tests.test_device_camera_update(tmp_path)


def test_foreground_unsplit():
    arrays, _labels, _ = load_scene(ROOT / "tests/fixtures/cover_2.json")
    p = EventPeeler(*arrays)
    p.run()
    info = p.read_status()
    ids = p.output_sources.numpy()[: info["polygons"]]
    ranks = p.output_layers.numpy()[: info["polygons"]]
    assert np.count_nonzero((ids == 0) & (ranks == 0)) == 1


def test_nonmanifold_edge(tmp_path):
    path = tmp_path / "nonmanifold.json"
    a, b = [-1, 0, 2], [1, 0, 2]
    data = {
        "window": [-2, -2, 2, 2],
        "camera": {"type": "orthographic"},
        "triangles": [
            {"id": i, "vertices": [a, b, c]}
            for i, c in enumerate([[0, 1, 1], [0, -1, 1], [0, 1, 3]])
        ],
    }
    path.write_text(json.dumps(data))
    arrays, labels, _ = load_scene(path)
    p = EventPeeler(*arrays)
    p.run()
    assert_regions(regions(p, labels), oracle(path, tmp_path / "oracle"))
