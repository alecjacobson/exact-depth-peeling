#!/usr/bin/env python3
"""Full graph replay timings for contour events; optional independent ray audit."""

import argparse
import gc
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import warp as wp

from scripts.audit_gpu import audit
from warp_peel.events import EventPeeler
from warp_peel.io import load_scene


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scenes", nargs="+", type=Path)
    ap.add_argument(
        "--out", type=Path, default=Path("results/gpu-events/benchmark.json")
    )
    ap.add_argument("--repetitions", type=int, default=5)
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--no-share-regions", action="store_true")
    args = ap.parse_args()
    report = {
        "warp": wp.__version__,
        "share_regions": not args.no_share_regions,
        "scope": "Full graph replay, including projection/topology, contour subdivision, shared rank queries and conforming world mesh; excludes upload/allocation/JIT/capture/readback/audit",
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path("warp_peel").glob("*.py")
        },
        "cases": [],
    }
    for path in args.scenes:
        print(path, flush=True)
        arrays, labels, data = load_scene(path)
        p = EventPeeler(*arrays, share_regions=not args.no_share_regions)
        case = {
            "scene": str(path),
            "sources": len(labels),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "capacities": vars(p.event_cap),
        }
        case["load_before"] = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,utilization.gpu,memory.used",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
        case["processes_before"] = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
        try:
            p.capture()
            for _ in range(2):
                p.run()
                p.read_status()
            times = []
            for _ in range(args.repetitions):
                with wp.ScopedStream(p.stream):
                    begin, end = (
                        wp.Event(enable_timing=True),
                        wp.Event(enable_timing=True),
                    )
                    wp.record_event(begin)
                    p.run()
                    wp.record_event(end)
                times.append(wp.get_event_elapsed_time(begin, end))
            case["milliseconds"] = times
            case["median_ms"] = statistics.median(times)
            case["output"] = p.read_status()
            if args.audit:
                regressions = json.loads(
                    Path("tests/gpu/regression_rays.json").read_text()
                )
                case["audit"] = audit(
                    p,
                    labels,
                    data,
                    samples=512,
                    regression_queries=regressions.get(case["sha256"], []),
                )
        except RuntimeError as exc:
            case["error"] = str(exc)
        report["cases"].append(case)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(case), flush=True)
        del p, arrays, data
        gc.collect()
    if any(
        "error" in c or c.get("audit", {}).get("rank_mismatches", 0)
        for c in report["cases"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
