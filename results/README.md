# Recorded artifacts

- `contours/`: exact contour correctness checks and before/after performance regressions, including the first failed timing attempt.
- `dragon/`: full 721,510-face input, all-layer measurements and audits, lossless mesh archive, Blender rendering, and layer-stack animation.

- `fast/`: matched exact/finite-precision measurements, mesh-contour acceleration,
  numerical error audits, and an exploded rendering of the fast-mode output.
- `validation.json`: exact fixture checks, insertion orders, and prefix tests.
- `findings.md`: interpretation of the measurements and next steps.
- `benchmark/`: the budgeted cover/no-cover sweep and any remaining default-suite cases.
- `stacks/`, `other/`, `broader/`: separate budgets for stacked, disjoint, cyclic,
  crossing, perforated, and generated connected scenes.
- `practical/`: connected perspective box/torus measurements at K=1,2,4.
- `large-first/`: the separately budgeted g=64 first-layer rerun.
- `connected/`, `nested/`: focused connected-mesh runs with a longer process cap.
- `examples/`: retained exact JSON, SVG, OBJ, and metrics for small validated scenes.
- `large/`: accelerated complete-layer measurements up to 102,400 triangles,
  scalable exact audits, and the exploded 10,240-triangle Blender demonstration.
  Its separate README describes the compact metrics format and timing boundaries.

The original benchmark directories have configuration, exact scene/camera manifests,
`runs.jsonl` (individual successful measured repetitions), `summary.csv` (median
and range for configurations completing all three repetitions), `status.jsonl`
(all attempted or explicitly skipped configurations), and `completion.json`.
The cover and stack directories also contain standalone PNG/SVG/PDF figures.

The process timeout includes setup, validation during warm-up, and export. A
`warmup_validation` timeout is therefore **not** an algorithm-only runtime bound.
Partial repetitions are retained in the raw log but excluded from the summary.
An `oracle_cap` means the configured 64-input-source benchmark limit; it is not a
timeout. `total_budget_exhausted` means that configuration was never attempted.

All timing is from sequential subprocess runs on a shared host. There is one
warm-up and three measured repetitions per successful configuration. Algorithm
time excludes setup/preflight, validation, and serialization, each reported
separately. Polygon-operation counters cover the algorithm; mandatory preflight
can itself perform exact overlap tests. CGAL internal construction counts are
unavailable, and no total internal-construction claim is inferred from a small
output or the adapter counters.
