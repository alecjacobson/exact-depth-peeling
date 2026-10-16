"""Certified connectivity must preserve complete source-labelled regions."""

import json

import numpy as np
import pytest
from test_peeler import ROOT, assert_regions, oracle, regions

from warp_peel.events import EventPeeler
from warp_peel.io import load_scene


def grid_scene(path, half_cover=False):
    triangles = []
    for z in (1, 2):
        for y in range(4):
            for x in range(4):
                if half_cover and z == 1 and x >= 2:
                    continue
                a, b, c, d = [x, y, z], [x + 1, y, z], [x + 1, y + 1, z], [x, y + 1, z]
                for vertices in ([a, b, c], [a, c, d]):
                    triangles.append({"id": len(triangles), "vertices": vertices})
    path.write_text(
        json.dumps(
            {
                "window": [-1, -1, 5, 5],
                "camera": {"type": "orthographic"},
                "triangles": triangles,
            }
        )
    )
    return load_scene(path)


@pytest.mark.parametrize("half_cover", [False, True])
def test_mesh_rank_components(tmp_path, half_cover):
    path = tmp_path / "grid.json"
    arrays, labels, _ = grid_scene(path, half_cover)
    p = EventPeeler(*arrays, share_regions=False)
    p.run()
    info = p.read_status()
    assert info["rank_connections"] > 0
    assert info["rank_components"] <= 3
    assert info["ray_groups"] <= 3
    assert info["merged_polygons"] == 0  # Provenance prevents merging source faces.
    assert info["polygons"] == len(labels)
    assert_regions(regions(p, labels), oracle(path, tmp_path / "oracle"))
    baseline = EventPeeler(
        *arrays, share_regions=False, propagate_ranks=False, merge_regions=False
    )
    baseline.run()
    assert_regions(regions(p, labels), regions(baseline, labels))
    assert baseline.read_status()["ray_groups"] >= 16
    # Reuse exactly the captured graph after deforming both surfaces. The old
    # adjacency state must be cleared; this also exercises non-coplanar patches.
    xyz = arrays[0].numpy()
    xyz[:, 2] += 0.05 * xyz[:, 0] * xyz[:, 1]
    arrays[0].assign(xyz)
    p.run()
    baseline.run()
    assert_regions(regions(p, labels), regions(baseline, labels))


@pytest.mark.parametrize(
    "propagate,merge", [(True, False), (False, True), (True, True)]
)
@pytest.mark.parametrize(
    "name", ["holes", "partial", "closed_object", "coplanar_adjacent"]
)
def test_component_switches(tmp_path, name, propagate, merge):
    path = ROOT / "tests/fixtures" / f"{name}.json"
    if not path.exists():
        pytest.fail(f"Missing fixture {name}")
    arrays, labels, _ = load_scene(path)
    p = EventPeeler(*arrays, propagate_ranks=propagate, merge_regions=merge)
    p.run()
    assert_regions(regions(p, labels), oracle(path, tmp_path / "oracle"))
    assert p.read_status()["edge_hash_misses"] == 0


def test_stack_hash_is_bounded():
    from test_events import stack

    p = EventPeeler(*stack(4096))
    p.run()
    info = p.read_status()
    assert info["edge_hash_misses"] == 0
    assert info["matched_patch_edges"] == 0
    assert info["rank_components"] == 4096
    assert info["ray_groups"] == 1
    np.testing.assert_array_equal(
        p.output_layers.numpy()[:4096], p.output_sources.numpy()[:4096]
    )


def test_query_sort_bucket_replay():
    from test_events import stack

    arrays = stack(1025)
    xyz = arrays[0].numpy()
    p = EventPeeler(*arrays)
    p.capture()
    for count in (1025, 127, 0, 257, 256, 1025):
        changed = xyz.copy().reshape(1025, 3, 3)
        changed[count:, :, 0] += 10.0
        arrays[0].assign(changed.reshape(-1, 3))
        p.run()
        info = p.read_status()
        assert info["polygons"] == count
        assert info["layers"] == count
        assert info["ray_groups"] == int(count > 0)
        np.testing.assert_array_equal(
            p.output_sources.numpy()[:count], p.output_layers.numpy()[:count]
        )


def test_merge_replay_bypasses_empty_adjacency():
    from test_events import stack

    arrays, labels, _ = load_scene(ROOT / "tests/fixtures/partial.json")
    original = arrays[0].numpy()
    separated = stack(2)[0].numpy()
    p = EventPeeler(*arrays, merge_regions=True)
    control = EventPeeler(*arrays, propagate_ranks=False, merge_regions=False)
    for xyz in (original, separated, original, separated):
        arrays[0].assign(xyz)
        p.run()
        control.run()
        assert_regions(regions(p, labels), regions(control, labels))
        if xyz is separated:
            info = p.read_status()
            assert info["splits"] == 0
            assert info["matched_patch_edges"] == 0
            assert info["polygons"] == 2
            assert info["rank_components"] == 2


@pytest.mark.parametrize("name", ["holes", "partial", "closed_object"])
def test_merged_mesh_oracle(name, tmp_path, monkeypatch):
    from functools import partial

    import test_peeler as reference_tests

    # Includes triangle unions, positive orientation, no overlapping fans and
    # reconstruction on each original source plane, beyond polygon-rank checks.
    monkeypatch.setattr(
        reference_tests, "DevicePeeler", partial(EventPeeler, merge_regions=True)
    )
    reference_tests.test_oracle(name, tmp_path)
