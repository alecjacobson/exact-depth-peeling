#!/usr/bin/env python3
"""Check the finite-precision path against exact regions/rays and preprocessing provenance."""

import argparse
import hashlib
import json
import subprocess
import tempfile
from fractions import Fraction as F
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exe", type=Path, default=ROOT / "build/geometric_peel")
    p.add_argument(
        "--resolver", type=Path, default=ROOT / "build/resolve_intersections"
    )
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--report", type=Path, default=ROOT / "results/fast/validation.json")
    a = p.parse_args()
    records, failures = [], []
    manifest = json.loads((ROOT / "tests/fixtures/manifest.json").read_text())
    resolved = {
        "single",
        "window_clipped",
        "disjoint",
        "stacked",
        "partial",
        "holes",
        "components",
        "coplanar_adjacent",
        "shared_vertex",
        "edge_on",
        "closed_object",
        "cover_2",
        "empty",
        "binary_float",
        "cyclic",
        "perspective_partial",
        "perspective_near",
        "perspective_behind",
        "camera_transform",
        "perspective_torus",
    }
    with tempfile.TemporaryDirectory(prefix="fast-validation-") as tmp:
        tmp = Path(tmp)

        def run(scene, name, mesh=False, extra=(), verify=True):
            out = tmp / name
            cmd = [
                str(a.exe),
                "--scene",
                str(scene),
                "--method",
                "fast-bvh-peel",
                "--layers",
                "all",
                "--threads",
                str(a.threads),
                "--out",
                str(out),
            ]
            if verify:
                cmd += ["--verify", "--oracle-cap", "128"]
            if mesh:
                cmd += ["--resolved-mesh"]
            cmd += list(extra)
            result = subprocess.run(
                cmd, text=True, capture_output=True, timeout=60, check=False
            )
            if result.returncode:
                records.append(
                    {"case": name, "status": "error", "stderr": result.stderr}
                )
                failures.append(name)
                return None
            m = json.loads((out / "metrics.json").read_text())
            assert (
                m["exact_geometry"] is False
                and m["exact_invariants_validated"] is False
            )
            assert not (out / "exact.json").exists()
            audit = m["fast_validation"]
            records.append(
                {
                    "case": name,
                    "audit": audit,
                    "topology": m["mesh_topology"],
                    "classification": m["rank_classification"],
                    "layers": len(m["passes"]),
                }
            )
            if audit and audit["status"] != "passed" and name != "slivers":
                failures.append(name)
            return m

        for fixture in manifest["fixtures"]:
            name = fixture["name"]
            scene = ROOT / "tests/fixtures" / f"{name}.json"
            if fixture["expected"].startswith("reject"):
                m = run(scene, name, verify=False)
                if m:
                    assert sum(x["depth_tie_breaks"] for x in m["passes"]) > 0
                continue
            run(scene, name)
            if name in resolved:
                run(scene, name + "-mesh", mesh=True)
            print(name, flush=True)

        for name in ("closed_object", "perspective_torus"):
            scene = ROOT / "tests/fixtures" / f"{name}.json"
            full = json.loads((tmp / (name + "-mesh") / "approximate.json").read_text())
            for k in (1, 2, 4):
                tag = f"{name}-prefix-{k}"
                run(scene, tag, mesh=True, extra=("--layers", str(k), "--threads", "1"))
                prefix = json.loads((tmp / tag / "approximate.json").read_text())
                assert prefix["layers"] == full["layers"][:k]

        # This three-page junction is already resolved, but not manifold.
        junction = {
            "window": [-2, -2, 2, 2],
            "triangles": [
                {"id": i, "vertices": [[0, -1, 2], [0, 1, 2], c]}
                for i, c in enumerate(([-1, 0, 3], [1, 0, 3], ["1/2", 0, 1]))
            ],
        }
        scene = tmp / "junction.json"
        scene.write_text(json.dumps(junction))
        m = run(scene, "nonmanifold-junction", mesh=True)
        if m:
            assert m["mesh_topology"]["nonmanifold_or_inconsistent_edges"] >= 1

        for name in ("crossing", "perspective_crossing"):
            original = ROOT / "tests/fixtures" / f"{name}.json"
            processed = tmp / (name + "-resolved.json")
            subprocess.run(
                [str(a.resolver), str(original), str(processed)],
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
            data = json.loads(processed.read_text())
            inputs = {
                t.get("id", i): t for i, t in enumerate(data["original_triangles"])
            }
            sums = {i: [F(0)] * 3 for i in inputs}
            sub = lambda a, b: [x - y for x, y in zip(a, b)]
            dot = lambda a, b: sum(x * y for x, y in zip(a, b))
            cross = lambda a, b: [
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ]
            for t in data["triangles"]:
                source = t["original_source_id"]
                base = [[F(v) for v in p] for p in inputs[source]["vertices"]]
                vertices = [[F(v) for v in p] for p in t["vertices"]]
                u, v = sub(base[1], base[0]), sub(base[2], base[0])
                uu, uv, vv = dot(u, u), dot(u, v), dot(v, v)
                for point in vertices:
                    w = sub(point, base[0])
                    assert dot(w, cross(u, v)) == 0
                    b = (dot(w, u) * vv - dot(w, v) * uv) / (uu * vv - uv * uv)
                    c = (dot(w, v) * uu - dot(w, u) * uv) / (uu * vv - uv * uv)
                    assert b >= 0 and c >= 0 and b + c <= 1
                area = cross(
                    sub(vertices[1], vertices[0]), sub(vertices[2], vertices[0])
                )
                sums[source] = [x + y for x, y in zip(sums[source], area)]
            for source, t in inputs.items():
                v = [[F(x) for x in p] for p in t["vertices"]]
                assert sums[source] == cross(sub(v[1], v[0]), sub(v[2], v[0]))
            m = run(processed, name + "-preprocessed", mesh=True)
            if m:
                assert m["mesh_topology"]["nonmanifold_or_inconsistent_edges"] > 0
            records.append(
                {
                    "case": name + "-preprocessing-provenance",
                    "status": "passed",
                    "details": data["intersection_preprocessing"],
                }
            )
    report = {
        "status": "passed" if not failures else "failed",
        "failures": failures,
        "expected_precision_loss_fixture": "slivers",
        "executable_sha256": hashlib.sha256(a.exe.read_bytes()).hexdigest(),
        "resolver_sha256": hashlib.sha256(a.resolver.read_bytes()).hexdigest(),
        "records": records,
    }
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(report, indent=2) + "\n")
    if failures:
        raise SystemExit("Unexpected fast-mode failures: " + ", ".join(failures))


if __name__ == "__main__":
    main()
