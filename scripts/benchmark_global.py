#!/usr/bin/env python3
"""Matched exact global/peeling timings on cached, intersection-resolved dragon LODs.

Each timed process performs one warm-up and three measured extractions. Audits
and exact region comparisons run separately, with their own caps. A timeout is
for the entire process, never a claimed per-extraction lower bound. Failed and
partial runs remain in runs.json; only completed timings enter summary.csv.
"""

import argparse
import csv
import hashlib
import json
import os
import resource
import statistics
import subprocess
import time
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exe", type=Path, default=Path("build/geometric_peel"))
    p.add_argument("--resolver", type=Path, default=Path("build/resolve_intersections"))
    p.add_argument("--cache", type=Path, default=Path("results/global/dragon-cache"))
    p.add_argument("--output", type=Path, default=Path("results/global/dragon"))
    p.add_argument(
        "--sizes", type=int, nargs="+", default=[64, 128, 512, 2048, 8192, 32768]
    )
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--timeout", type=float, default=180)
    p.add_argument("--validation-timeout", type=float, default=180)
    p.add_argument("--compare-max", type=int, default=512)
    p.add_argument("--oracle-max", type=int, default=128)
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--retry-timeouts",
        type=float,
        metavar="SECONDS",
        help="With --resume, retry only timed-out solver jobs with this process cap",
    )
    a = p.parse_args()
    if a.retry_timeouts is not None and (not a.resume or a.retry_timeouts <= 0):
        p.error("--retry-timeouts requires --resume and a positive time limit")
    a.output.mkdir(parents=True, exist_ok=True)
    config = {
        "executable_sha256": digest(a.exe),
        "resolver_sha256": digest(a.resolver),
        "threads": a.threads,
        "sizes": a.sizes,
        "process_timeout_seconds": a.timeout,
        "validation_timeout_seconds": a.validation_timeout,
        "warmups": 1,
        "measured_repetitions": 3,
        "address_space_cap_gib": 24,
        "compare_max": a.compare_max,
        "oracle_max": a.oracle_max,
        "arithmetic": "exact predicates and constructions, both methods",
        "algorithm_includes": [
            "all layers",
            "BVH construction",
            "labelled conforming triangulation",
        ],
        "algorithm_excludes": [
            "import/projection/topology",
            "intersection preprocessing",
            "validation",
            "serialization",
        ],
        "host": os.uname().nodename,
    }
    config_path = a.output / "config.json"
    if (
        a.resume
        and config_path.exists()
        and json.loads(config_path.read_text()) != config
    ):
        raise RuntimeError(
            "Cannot resume measurements with a changed configuration/binary"
        )
    write_json(config_path, config)
    records = (
        json.loads((a.output / "runs.json").read_text())
        if a.resume and (a.output / "runs.json").exists()
        else []
    )
    env = dict(os.environ, OMP_WAIT_POLICY="PASSIVE")

    def limit():
        resource.setrlimit(resource.RLIMIT_AS, (24 * 1024**3,) * 2)

    def execute(cmd, timeout):
        start = time.monotonic()
        record = {"command": list(map(str, cmd)), "process_timeout_seconds": timeout}
        try:
            r = subprocess.run(
                record["command"],
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout,
                env=env,
                preexec_fn=limit,
            )
            record.update(
                status="completed" if r.returncode == 0 else "error",
                returncode=r.returncode,
                stdout=r.stdout,
                stderr=r.stderr,
            )
        except subprocess.TimeoutExpired as error:

            def text(v):
                return v.decode(errors="replace") if isinstance(v, bytes) else v

            record.update(
                status="timeout", stdout=text(error.stdout), stderr=text(error.stderr)
            )
        record["wall_seconds"] = time.monotonic() - start
        return record

    def save(record):
        records.append(record)
        write_json(a.output / "runs.json", records)
        print(
            json.dumps(
                {
                    k: v
                    for k, v in record.items()
                    if k not in ("stdout", "stderr", "command")
                }
            ),
            flush=True,
        )
        columns = [
            "target_faces",
            "input_faces",
            "method",
            "status",
            "median_seconds",
            "min_seconds",
            "max_seconds",
            "setup_seconds",
            "layers",
            "output_triangles",
            "peak_rss_kib",
            "scene_sha256",
        ]
        with (a.output / "summary.csv").open("w") as out:
            writer = csv.DictWriter(
                out, fieldnames=columns, extrasaction="ignore", lineterminator="\n"
            )
            writer.writeheader()
            latest = {
                (r["target_faces"], r["method"]): r
                for r in records
                if r.get("phase") == "timing"
            }
            writer.writerows(latest.values())

    for target in a.sizes:
        directory = a.cache / str(target)
        raw = directory / "raw.json"
        manifest = json.loads((directory / "manifest.json").read_text())
        if digest(raw) != manifest["scene_sha256"]:
            raise RuntimeError(f"Corrupt raw cache: {raw}")
        scene = directory / "resolved.json"
        preprocessing = directory / "preprocessing.json"
        identity = {"input_sha256": digest(raw), "resolver_sha256": digest(a.resolver)}
        if preprocessing.exists():
            previous = json.loads(preprocessing.read_text())
            if previous["identity"] != identity or previous["output_sha256"] != digest(
                scene
            ):
                raise RuntimeError(f"Preprocessing cache identity changed: {directory}")
        else:
            record = execute([a.resolver, raw, scene], a.validation_timeout)
            record.update(phase="preprocessing", target_faces=target)
            save(record)
            if record["status"] != "completed":
                continue
            data = json.loads(scene.read_text())
            previous = {
                "identity": identity,
                "output_sha256": digest(scene),
                "preprocessing": data["intersection_preprocessing"],
            }
            write_json(preprocessing, previous)
        count = previous["preprocessing"]["output_triangles"]
        source_hash = digest(scene)
        for method in ("global-arrangement", "bvh-peel"):
            for phase in ("timing", "audit", "comparison"):
                if phase == "comparison" and (
                    method != "bvh-peel" or target > a.compare_max
                ):
                    continue
                previous = (
                    [
                        i
                        for i, r in enumerate(records)
                        if r.get("target_faces") == target
                        and r.get("method") == method
                        and r.get("phase") == phase
                    ]
                    if a.resume
                    else []
                )
                retry = bool(
                    previous
                    and a.retry_timeouts
                    and records[previous[-1]]["status"] == "timeout"
                )
                if previous and not retry:
                    continue
                out = a.output / str(target) / f"{method}-{phase}"
                out.mkdir(parents=True, exist_ok=True)
                metrics = out / "metrics.json"
                metrics.unlink(missing_ok=True)  # Never accept a stale successful run.
                cmd = [
                    a.exe,
                    "--scene",
                    scene,
                    "--method",
                    method,
                    "--resolved-mesh",
                    "--threads",
                    a.threads,
                    "--layers",
                    "all",
                    "--metrics-only",
                    "--out",
                    out,
                ]
                if phase == "timing":
                    cmd += ["--repeat", 4]
                elif phase == "comparison":
                    cmd += ["--verify-global"]
                else:
                    cmd += (
                        ["--verify", "--oracle-cap", count]
                        if target <= a.oracle_max
                        else ["--validate"]
                    )
                record = execute(
                    cmd,
                    a.retry_timeouts
                    if retry
                    else (a.timeout if phase == "timing" else a.validation_timeout),
                )
                if retry:
                    record["retry_of_record"] = previous[-1]
                record.update(
                    phase=phase,
                    method=method,
                    target_faces=target,
                    input_faces=count,
                    scene_sha256=source_hash,
                )
                if record["status"] == "completed":
                    m = json.loads(metrics.read_text())
                    record.update(
                        setup_seconds=m["projection_preflight_seconds"],
                        layers=len(m["passes"]),
                        peak_rss_kib=m["peak_rss_kib"],
                        output_triangles=sum(
                            p["exported_triangles"] for p in m["passes"]
                        ),
                    )
                    if phase == "timing":
                        times = m["algorithm_repetitions_seconds"][1:]
                        record.update(
                            median_seconds=statistics.median(times),
                            min_seconds=min(times),
                            max_seconds=max(times),
                        )
                    else:
                        record.update(
                            validation_seconds=m["validation_seconds"],
                            validated_against_oracle=m["validated_against_oracle"],
                            validated_against_global=m["validated_against_global"],
                            convex_audit=m["convex_audit"],
                        )
                save(record)


if __name__ == "__main__":
    main()
