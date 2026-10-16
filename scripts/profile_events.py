#!/usr/bin/env python3
"""Captured stage diagnostics, separate from complete graph replay timings."""

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import warp as wp

from scripts.benchmark_event_stacks import stack_arrays
from warp_peel.events import EventPeeler
from warp_peel.io import load_scene


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scene", type=Path, nargs="?")
    ap.add_argument("--stack", type=int)
    ap.add_argument("--out", type=Path, default=Path("results/gpu-events/phases.json"))
    args = ap.parse_args()
    if (args.scene is None) == (args.stack is None):
        ap.error("Specify one scene or --stack N")
    arrays = (
        stack_arrays(args.stack)
        if args.stack is not None
        else load_scene(args.scene)[0]
    )
    p = EventPeeler(*arrays)
    p.warmup()
    phases = [
        ("prepare", p._prepare),
        ("subdivide", lambda: wp.capture_while(p.condition, p._step)),
        ("rank", p._rank),
        ("mesh", p._mesh),
    ]
    graphs = []
    for name, fn in phases:
        with wp.ScopedStream(p.stream), wp.ScopedCapture(stream=p.stream) as capture:
            fn()
        graphs.append((name, capture.graph))
    times = {name: [] for name, _ in graphs}
    for rep in range(7):
        for name, graph in graphs:
            with wp.ScopedStream(p.stream):
                begin, end = wp.Event(enable_timing=True), wp.Event(enable_timing=True)
                wp.record_event(begin)
                wp.capture_launch(graph, stream=p.stream)
                wp.record_event(end)
            dt = wp.get_event_elapsed_time(begin, end)
            if rep >= 2:
                times[name].append(dt)
        p.read_status()
    result = {
        "scene": str(args.scene) if args.scene else f"stack:{args.stack}",
        "scope": "Four separately captured stage graphs; two warmup sequences then five measured sequences. Diagnostic medians are not the full-graph median.",
        "milliseconds": times,
        "median_ms": {k: statistics.median(v) for k, v in times.items()},
        "output": p.read_status(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
