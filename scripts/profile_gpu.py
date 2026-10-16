#!/usr/bin/env python3
"""Diagnostic uncaptured phase events; never used by DevicePeeler extraction."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import warp as wp

from warp_peel import DevicePeeler
from warp_peel import kernels as k
from warp_peel.io import load_scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path)
    parser.add_argument("--out", type=Path, default=Path("results/gpu/phases.json"))
    args = parser.parse_args()
    p = DevicePeeler(*load_scene(args.scene)[0])
    p.warmup()
    records = []
    launch = p._launch
    with wp.ScopedStream(p.stream):
        before = wp.Event(enable_timing=True)
        begin_peel = wp.Event(enable_timing=True)
        end_peel = wp.Event(enable_timing=True)
        after = wp.Event(enable_timing=True)

        def instrument(kernel, dim, *inputs):
            if kernel is k.peel:
                wp.record_event(begin_peel)
            launch(kernel, dim, *inputs)
            if kernel is k.peel:
                wp.record_event(end_peel)

        p._launch = instrument
        wp.record_event(before)
        p._prepare()
        wp.record_event(after)
        prepare = wp.get_event_elapsed_time(before, after)
        # Host reads only in this explicitly uncaptured diagnostic, never in solver.
        while p.read_status()["residual_patches"]:
            wp.record_event(before)
            p._layer()
            wp.record_event(after)
            wp.synchronize_event(after)
            records.append(
                {
                    "hierarchies_ms": wp.get_event_elapsed_time(before, begin_peel),
                    "clip_ms": wp.get_event_elapsed_time(begin_peel, end_peel),
                    "compact_ms": wp.get_event_elapsed_time(end_peel, after),
                }
            )
        wp.record_event(before)
        p._mesh()
        wp.record_event(after)
        mesh = wp.get_event_elapsed_time(before, after)
    info = p.read_status()
    report = {
        "scene": str(args.scene),
        "kernel_sha256": hashlib.sha256(
            Path("warp_peel/kernels.py").read_bytes()
        ).hexdigest(),
        "scope": "Uncaptured diagnostic CUDA events; includes launch gaps and current GPU contention. Not the graph-replay timing.",
        "prepare_ms": prepare,
        "mesh_ms": mesh,
        "layers": records,
        "output": info,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
