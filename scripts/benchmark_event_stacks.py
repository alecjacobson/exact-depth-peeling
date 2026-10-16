#!/usr/bin/env python3
"""Scaling control: coincident footprints at distinct depths, all layers."""

import argparse
import gc
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import warp as wp

from warp_peel.events import EventPeeler


def stack_arrays(n):
    xyz = np.tile(
        np.array([[-0.75, -0.75, 0], [0.75, -0.75, 0], [0, 0.75, 0]]), (n, 1, 1)
    )
    xyz[:, :, 2] = np.arange(1, n + 1)[:, None]
    return [
        wp.array(xyz.reshape(-1, 3), dtype=wp.vec3d, device="cuda"),
        wp.array(np.arange(3 * n).reshape(-1, 3), dtype=wp.vec3i, device="cuda"),
        wp.array(np.eye(4)[None], dtype=wp.mat44d, device="cuda"),
        wp.array([-1.0, -1.0, 1.0, 1.0, 0.0, 0.01], dtype=wp.float64, device="cuda"),
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--sizes", type=int, nargs="+", default=[16, 64, 256, 1024, 4096, 16384, 65536]
    )
    ap.add_argument("--out", type=Path, default=Path("results/gpu-events/stacks.json"))
    args = ap.parse_args()
    report = {
        "cases": [],
        "scope": "Complete captured pipeline; two warmups then five event timings; upload/JIT/capture/readback excluded",
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path("warp_peel").glob("*.py")
        },
    }
    for n in args.sizes:
        arrays = stack_arrays(n)
        p = EventPeeler(*arrays)
        p.capture()
        for _ in range(2):
            p.run()
            p.read_status()
        times = []
        for _ in range(5):
            with wp.ScopedStream(p.stream):
                begin, end = wp.Event(enable_timing=True), wp.Event(enable_timing=True)
                wp.record_event(begin)
                p.run()
                wp.record_event(end)
            times.append(wp.get_event_elapsed_time(begin, end))
        info = p.read_status()
        assert info["polygons"] == n and info["splits"] == 0 and info["layers"] == n
        assert info["unique_contours"] == 3 and info["ray_groups"] == 1
        assert info["ray_candidate_tests"] == 2 * n and info["ray_hits"] == n
        assert info["contour_tests"] <= 3 * n
        np.testing.assert_array_equal(
            p.output_sources.numpy()[:n], p.output_layers.numpy()[:n]
        )
        case = {
            "n": n,
            "milliseconds": times,
            "median_ms": statistics.median(times),
            "output": info,
            "load": subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used",
                    "--format=csv,noheader",
                ],
                text=True,
            ).strip(),
        }
        report["cases"].append(case)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(case), flush=True)
        del p, arrays
        gc.collect()


if __name__ == "__main__":
    main()
