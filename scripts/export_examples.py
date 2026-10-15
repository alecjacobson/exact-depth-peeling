#!/usr/bin/env python3
"""Retain small exact, SVG and lifted mesh artifacts after a validation run."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for name in (
    "single",
    "holes",
    "components",
    "cyclic",
    "crossing",
    "perspective_near",
    "cover_2",
    "closed_object",
    "perspective_torus",
):
    for method in ("visible-peel", "envelope-peel", "arrangement"):
        subprocess.run(
            [
                str(ROOT / "build/geometric_peel"),
                "--scene",
                str(ROOT / "tests/fixtures" / f"{name}.json"),
                "--method",
                method,
                "--layers",
                "all",
                "--verify",
                "--out",
                str(ROOT / "results/examples" / name / method),
            ],
            check=True,
        )

practical = ROOT / "results/practical/scenes/perspective-torus-4.json"
if practical.exists():
    for method in ("visible-peel", "envelope-peel"):
        for layers in ("4", "all"):
            destination = (
                ROOT
                / "results/examples/perspective-torus-384"
                / (method if layers == "4" else method + "-all")
            )
            subprocess.run(
                [
                    str(ROOT / "build/geometric_peel"),
                    "--scene",
                    str(practical),
                    "--method",
                    method,
                    "--layers",
                    layers,
                    "--validate",
                    "--out",
                    str(destination),
                ],
                check=True,
            )
    subtree = ROOT / "results/examples/perspective-torus-384"
    subprocess.run(
        [
            str(ROOT / "build/geometric_peel"),
            "--compare",
            str(subtree / "visible-peel-all/exact.json"),
            str(subtree / "envelope-peel-all/exact.json"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/check_rays.py"),
            str(subtree / "visible-peel-all/exact.json"),
            "--minimum-layer",
            "5",
            "--out",
            str(ROOT / "results/practical/late-layer-ray-checks.json"),
        ],
        check=True,
    )
