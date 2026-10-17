#!/usr/bin/env python3
"""Captured stage diagnostics, separate from complete graph replay timings."""

import argparse
import hashlib
import json
import statistics
import subprocess
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
    ap.add_argument("--detailed", action="store_true")
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
    if args.detailed:
        phases = phases[:2] + [
            ("rank_connectivity", p._connections),
            ("rank_samples", p._rank_samples),
            ("rank_query_sort", p._rank_query_sort),
            ("rank_group_certify", p._rank_group),
            ("rank_ray_primary", p._rank_ray_primary),
            ("rank_ray_heavy", p._rank_ray_heavy),
            ("rank_hit_sort_scan", p._rank_hit_sort),
            ("rank_assign", p._rank_assign),
            ("mesh_vertices_scan", p._mesh_vertices),
            ("mesh_bounds_bvh", p._mesh_index),
            ("mesh_node_emit_lift", p._mesh_emit),
        ]
    load_before = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,utilization.gpu,memory.used",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
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
        "scope": "Separately captured stage graphs; two warmup sequences then five measured sequences. Diagnostic medians are not the full-graph median. Noding, emission and world lifting remain fused.",
        "warp": wp.__version__,
        "load_before": load_before,
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path("warp_peel").glob("*.py")
        },
        "input_sha256": hashlib.sha256(args.scene.read_bytes()).hexdigest()
        if args.scene
        else None,
        "milliseconds": times,
        "median_ms": {k: statistics.median(v) for k, v in times.items()},
        "output": p.read_status(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
