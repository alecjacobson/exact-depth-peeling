# Finite-precision mesh-contour results

See the root README for measured timings, numerical error, preprocessing and reproduction commands.
`summary.csv` reports extraction medians/ranges; `status.jsonl` records every timed attempt.
`audits.json` contains independent exact ray checks and area error; `validation.json` covers small-scene oracle checks and preprocessing provenance.
`demo/` contains the actual fast-mode layer meshes and exploded render.
The mode rounds constructions and drops grid-scale remnants. It is not an exact solver.

`thread-scaling/` records the 1/2/4/8-thread fast BVH comparison on the cached
4K/64K dragons and 100K torus under explicitly accepted concurrent load. The host
has 128 logical CPUs but this session shares a four-core quota. See its
[report](thread-scaling/README.md) for timing boundaries and contention estimates.
`thread-scaling-loaded/` preserves the initially paused batch, which is not mixed
into the final table.
