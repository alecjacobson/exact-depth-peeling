#!/usr/bin/env python3
"""Compare retained event benchmarks; timing gate requires comparable GPU/load."""

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("baseline", type=Path)
    ap.add_argument("candidate", type=Path)
    ap.add_argument("--maximum-ratio", type=float, default=1.15)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    old = json.loads(args.baseline.read_text())["cases"]
    new = json.loads(args.candidate.read_text())["cases"]
    before = {c["sha256"]: c for c in old}
    after = {c["sha256"]: c for c in new}
    failures = []
    if (
        not before
        or before.keys() != after.keys()
        or len(before) != len(old)
        or len(after) != len(new)
    ):
        failures.append(
            "Cases must be nonempty, unique, and have matching input checksums"
        )
    rows = []
    for sha in before.keys() & after.keys():
        a, b = before[sha], after[sha]
        if "error" in a or "error" in b:
            failures.append(f"{b['scene']}: extraction error")
            continue
        ratio = b["median_ms"] / a["median_ms"]
        if ratio > args.maximum_ratio:
            failures.append(f"{b['scene']}: timing ratio {ratio:.3f}")
        for field in ("polygons", "triangles", "layers"):
            if a["output"][field] != b["output"][field]:
                failures.append(
                    f"{b['scene']}: changed {field}; review geometry before accepting"
                )
        if b.get("audit", {}).get("rank_mismatches") != 0:
            failures.append(f"{b['scene']}: missing or failed ray audit")
        rows.append(
            {
                "sources": b["sources"],
                "scene": b["scene"],
                "baseline_ms": a["median_ms"],
                "candidate_ms": b["median_ms"],
                "candidate_over_baseline": ratio,
            }
        )
    report = {
        "maximum_ratio": args.maximum_ratio,
        "scope": "Timing and output-count regression gate, not a geometric certificate; use comparable hardware/load and separate full-region validation",
        "cases": sorted(rows, key=lambda c: c["sources"]),
        "failures": failures,
    }
    encoded = json.dumps(report, indent=2) + "\n"
    if args.out:
        args.out.write_text(encoded)
    print(encoded)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
