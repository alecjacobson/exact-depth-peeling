# Finite-precision mesh-contour results

See the root README for measured timings, numerical error, preprocessing and reproduction commands.
`summary.csv` reports extraction medians/ranges; `status.jsonl` records every timed attempt.
`audits.json` contains independent exact ray checks and area error; `validation.json` covers small-scene oracle checks and preprocessing provenance.
`demo/` contains the actual fast-mode layer meshes and exploded render.
The mode rounds constructions and drops grid-scale remnants. It is not an exact solver.
