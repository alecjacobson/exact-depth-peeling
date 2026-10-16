#!/usr/bin/env python3
"""Measure full device-resident graph replay; uploads, JIT, capture and readback excluded."""

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import warp as wp

from warp_peel import DevicePeeler
from warp_peel.io import load_scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenes", nargs="+", type=Path)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--out", type=Path, default=Path("results/gpu/benchmark.json"))
    args = parser.parse_args()
    report = {
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path("warp_peel").glob("*.py"))
        },
        "load_at_start": subprocess.check_output(
            ["nvidia-smi", "pmon", "-c", "1"], text=True
        ),
        "warp": wp.__version__,
        "gpu": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            text=True,
        ).strip(),
        "timing": "CUDA events around full graph replay; includes device input validation, welding, projection, contours, classification, all peeling, boundary noding and world-space mesh generation; excludes upload, allocation, warmup, capture, download and audits",
        "cases": [],
    }
    for path in args.scenes:
        case = {
            "scene": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        print(path, flush=True)
        start = time.perf_counter()
        arrays, labels, _data = load_scene(path)
        case["upload_seconds"] = time.perf_counter() - start
        p = DevicePeeler(*arrays)
        case["sources"] = len(labels)
        case["capacities"] = vars(p.cap)
        case["gpu_processes_before"] = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
        start = time.perf_counter()
        p.capture()
        case["warmup_capture_seconds"] = time.perf_counter() - start
        try:
            for _ in range(2):
                p.run()
                p.read_status()
            times = []
            for _ in range(args.repetitions):
                with wp.ScopedStream(p.stream):
                    begin = wp.Event(enable_timing=True)
                    end = wp.Event(enable_timing=True)
                    wp.record_event(begin)
                    p.run()
                    wp.record_event(end)
                times.append(wp.get_event_elapsed_time(begin, end))
                case["output"] = p.read_status()
            case["milliseconds"] = times
            case["median_ms"] = statistics.median(times)
            case["per_layer_counts"] = p.stats.numpy()[
                : case["output"]["peeling_passes"]
            ].tolist()
            # Diagnostic phase timings outside capture; includes launch gaps.
            stages = []
            with wp.ScopedStream(p.stream):
                begin = wp.Event(enable_timing=True)
                end = wp.Event(enable_timing=True)
                wp.record_event(begin)
                p._prepare()
                wp.record_event(end)
                stages.append(
                    {
                        "phase": "prepare",
                        "milliseconds": wp.get_event_elapsed_time(begin, end),
                    }
                )
                while p.status.numpy()[0] and not p.status.numpy()[9]:
                    wp.record_event(begin)
                    p._layer()
                    wp.record_event(end)
                    stages.append(
                        {
                            "phase": "layer",
                            "milliseconds": wp.get_event_elapsed_time(begin, end),
                        }
                    )
                wp.record_event(begin)
                p._mesh()
                wp.record_event(end)
                stages.append(
                    {
                        "phase": "mesh",
                        "milliseconds": wp.get_event_elapsed_time(begin, end),
                    }
                )
            case["uncaptured_diagnostic_stages"] = stages
        except RuntimeError as exc:
            case["error"] = str(exc)
        print(
            json.dumps(
                {
                    k: v
                    for k, v in case.items()
                    if k not in ("per_layer_counts", "uncaptured_diagnostic_stages")
                }
            ),
            flush=True,
        )
        report["cases"].append(case)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        del p, arrays
        import gc

        gc.collect()
    if any("error" in c for c in report["cases"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
