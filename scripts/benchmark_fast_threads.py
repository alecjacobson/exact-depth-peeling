#!/usr/bin/env python3
"""Sequential fast BVH thread scaling, including oversubscription of the CPU quota."""

import argparse
import csv
import json
import os
import resource
import statistics
import subprocess
import time
from pathlib import Path

from benchmark_global import digest, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exe", type=Path, default=Path("build/geometric_peel"))
    p.add_argument("--output", type=Path, default=Path("results/fast/thread-scaling"))
    p.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8])
    a = p.parse_args()
    scenes = {
        "dragon-4k": Path("results/global/dragon-cache/4096/resolved.json"),
        "dragon-64k": Path("results/global/dragon-cache/65536/resolved.json"),
        "torus-100k": Path("results/large/scenes/sculpted-102400.json"),
    }
    cgroup = Path("/sys/fs/cgroup") / Path(
        "/proc/self/cgroup"
    ).read_text().strip().split("::", 1)[1].lstrip("/")
    limits = {}
    directory = cgroup
    while directory != Path("/sys/fs"):
        limits[str(directory)] = {
            name: (directory / name).read_text().strip()
            for name in ("cpu.max", "cpuset.cpus.effective", "memory.max", "pids.max")
            if (directory / name).exists()
        }
        if directory == Path("/sys/fs/cgroup"):
            break
        directory = directory.parent
    a.output.mkdir(parents=True, exist_ok=True)
    config = {
        "executable_sha256": digest(a.exe),
        "scene_sha256": {name: digest(path) for name, path in scenes.items()},
        "host": os.uname().nodename,
        "logical_cpus": os.cpu_count(),
        "affinity": sorted(os.sched_getaffinity(0)),
        "cgroup_limits": limits,
        "lscpu": json.loads(subprocess.check_output(["lscpu", "-J"], text=True)),
        "threads": a.threads,
        "warmups": 1,
        "measured_repetitions": 9,
        "grid_bits": 30,
        "omp_wait_policy": "PASSIVE",
        "process_timeout_seconds": 300,
        "address_space_cap_gib": 24,
        "timing_includes": [
            "BVHs",
            "contour classification",
            "all layers",
            "conforming triangulation",
        ],
        "timing_excludes": [
            "import/projection/topology",
            "intersection preprocessing",
            "validation",
            "export",
        ],
        "exact_geometry": False,
        "new_accuracy_audits": False,
        "background_cpu_load_controlled": False,
        "other_cpu_estimate": "cgroup CPU usage minus solver process CPU time, divided by process wall time; includes orchestration overhead",
    }
    write_json(a.output / "config.json", config)

    def limit():
        resource.setrlimit(resource.RLIMIT_AS, (24 * 1024**3,) * 2)

    def cpu_seconds():
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        return usage.ru_utime + usage.ru_stime

    def cpu_stats():
        return {
            key: int(value)
            for key, value in (
                line.split() for line in (cgroup / "cpu.stat").read_text().splitlines()
            )
        }

    records = []
    for name, scene in scenes.items():
        for threads in a.threads:
            out = a.output / name / str(threads)
            cmd = list(
                map(
                    str,
                    [
                        a.exe,
                        "--scene",
                        scene,
                        "--method",
                        "fast-bvh-peel",
                        "--resolved-mesh",
                        "--fast-grid-bits",
                        30,
                        "--layers",
                        "all",
                        "--threads",
                        threads,
                        "--repeat",
                        10,
                        "--metrics-only",
                        "--out",
                        out,
                    ],
                )
            )
            start = time.monotonic()
            cpu_before, stats_before = cpu_seconds(), cpu_stats()
            record = {"scene": name, "threads": threads, "command": cmd}
            try:
                run = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=300,
                    preexec_fn=limit,
                    env=dict(os.environ, OMP_WAIT_POLICY="PASSIVE"),
                )
                record.update(
                    status="completed" if run.returncode == 0 else "error",
                    returncode=run.returncode,
                    stdout=run.stdout,
                    stderr=run.stderr,
                )
            except subprocess.TimeoutExpired:
                record.update(status="timeout")
            record["wall_seconds"] = time.monotonic() - start
            record["process_cpu_seconds"] = cpu_seconds() - cpu_before
            after = cpu_stats()
            record["cgroup_cpu_stat_delta"] = {
                k: after[k] - v for k, v in stats_before.items()
            }
            record["estimated_other_cpu_equivalents"] = (
                max(
                    0,
                    record["cgroup_cpu_stat_delta"]["usage_usec"] / 1e6
                    - record["process_cpu_seconds"],
                )
                / record["wall_seconds"]
            )
            if record["status"] == "completed":
                m = json.loads((out / "metrics.json").read_text())
                times = m["algorithm_repetitions_seconds"][1:]
                record.update(
                    input_triangles=m["input_sources"],
                    layers=len(m["passes"]),
                    output_triangles=sum(x["exported_triangles"] for x in m["passes"]),
                    candidate_tests=sum(x["candidate_tests"] for x in m["passes"]),
                    median_seconds=statistics.median(times),
                    min_seconds=min(times),
                    max_seconds=max(times),
                    setup_seconds=m["projection_preflight_seconds"],
                    classification_seconds=m["rank_classification"][
                        "classification_seconds"
                    ],
                    visibility_seconds=sum(
                        x["visibility_seconds"] for x in m["passes"]
                    ),
                    meshing_seconds=sum(
                        x["triangulation_seconds"] for x in m["passes"]
                    ),
                )
            records.append(record)
            write_json(a.output / "runs.json", records)
            print(
                json.dumps(
                    {
                        k: v
                        for k, v in record.items()
                        if k not in ("command", "stdout", "stderr")
                    }
                ),
                flush=True,
            )
    columns = [
        "scene",
        "input_triangles",
        "threads",
        "status",
        "median_seconds",
        "min_seconds",
        "max_seconds",
        "setup_seconds",
        "layers",
        "output_triangles",
        "candidate_tests",
    ]
    with (a.output / "summary.csv").open("w") as f:
        writer = csv.DictWriter(
            f, fieldnames=columns, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(records)
    if any(r["status"] != "completed" for r in records):
        raise RuntimeError("Incomplete runs; see runs.json")
    lines = [
        "| Scene | Input triangles | "
        + " | ".join(f"{n} thread" + ("s" if n != 1 else "") for n in a.threads)
        + " | Observed 1→4 speedup |",
        "| --- | ---: | " + " | ".join("---:" for _ in a.threads) + " | ---: |",
    ]
    invariants = {}
    for name in scenes:
        rows = {r["threads"]: r for r in records if r["scene"] == name}
        invariants[name] = {
            key: len({r[key] for r in rows.values()}) == 1
            for key in ("layers", "output_triangles", "candidate_tests")
        }
        ratio = (
            f"{rows[1]['median_seconds'] / rows[4]['median_seconds']:.2f}×"
            if 1 in rows and 4 in rows
            else "—"
        )
        lines.append(
            f"| {name} | {next(iter(rows.values()))['input_triangles']:,} | "
            + " | ".join(
                f"{rows[n]['median_seconds'] * 1000:.1f} ms" for n in a.threads
            )
            + f" | {ratio} |"
        )
    write_json(a.output / "work-count-invariants.json", invariants)
    (a.output / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
