# Full XYZ RGB dragon example

The input is the requested `sparse-solver-benchmark/xyzrgb_dragon-720K.ply`:
360,757 vertices and 721,510 triangles, without further decimation. `input.json`
records its SHA-256, bounds, acquisition camera, and image window. The supplied
file is a reduced version of the XYZ RGB Asian Dragon, credited to XYZ RGB Inc.
and distributed through the [Stanford 3D Scanning Repository](https://graphics.stanford.edu/data/3Dscanrep/).
The upstream model-use terms apply; this mesh is not one of this project's CC0
synthetic fixtures. The derivation history of the supplied 720K file is not recorded
in its PLY header.

`scene.json` retains the imported binary64 vertex values in round-tripping JSON,
unique input-face IDs, and the camera. Exactness refers to those represented
vertices and camera values. `preprocessing.json` records exact, camera-independent
CGAL autorefinement: it found zero intersection edges and retained all 721,510
faces. Therefore the original scene is used directly with `--resolved-mesh`.
Fast import reports one collapsed source (degenerate or below its grid resolution);
721,509 source domains enter extraction.

`config.json` and `runs.json` record complete-extraction attempts, timing boundaries,
process limits, errors/timeouts, and the independent exact-ray audit. A timed-out
exact run is not an exact performance result or a validation of the fast result.
The fixed 30-bit grid can change tiny features and late layers. On this input the audit fails: 40 / 2,779 checked ray ranks disagree; 301 near-boundary samples are skipped. Relative per-source area L1 error is 4.54e-7. The visualization is explicitly approximate. The small-scene
exact oracle regressions and the large-scene sampled audit have different scopes.

`meshes.npz` is a lossless, compressed packing of the exported float64 OBJ vertices,
triangle indices, and source IDs. Duplicate coordinates are welded only when all
three float64 values are identical. Arrays `vertices_k`, `faces_k`, `sources_k`
use zero-based layer indices; `input_vertices`, `input_faces` contain the original
PLY mesh. It is the authoritative rendering input, and is not a decimated proxy.
The meshes are finite-precision solver outputs, not exact certificates.

The Blender script creates a separate rendering copy. Only that copy welds positions
at 1e-8 model units. Decorative outlines come from a 2,800-pixel-wide coverage raster of each layer,
not every triangulation edge; they fade before the 3D reconstruction. The raster
union prevents subpixel cracks from becoming thick lines. These display outlines
(`outlines.npz`) are separate from the authoritative geometric output. Render colors identify ranks. Layer opacity is 40%.
Billboards are placed at separate camera depths and their vertices are positioned
on the original acquisition rays. Surface shape keys interpolate each vertex along that
same ray back to its exported 3D location. Decorative outlines fade at the start
of that morph; they are display-resolution guides rather than exported geometry. Thus all morph stages preserve the
acquisition projection. The camera orbits to expose depth and returns to the
acquisition location before the sequence fades and restarts. Small late layers
are retained even when they are too small to see at GIF resolution.

The scene and mesh archive are already versioned. The first conversion command
requires the original supplied PLY; extraction and rendering can start directly
from the tracked scene/archive. Reproduce from the repository root:

```sh
python scripts/dragon_scene.py
build/resolve_intersections results/dragon/scene.json out/dragon-resolved.json
python scripts/benchmark_dragon.py
build/geometric_peel --scene results/dragon/scene.json --method fast-bvh-peel \
  --resolved-mesh --layers all --threads 4 --mesh-only --out out/dragon-layers
python scripts/pack_dragon_layers.py --layers out/dragon-layers
python scripts/dragon_outlines.py
blender -b -t 4 --python scripts/render_dragon.py -- --assets results/dragon \
  --output out/dragon-render
python scripts/compose_dragon.py --frames out/dragon-render
```

The renderer uses Blender 4.5.3, Cycles with an NVIDIA L40, transparent colored
surface materials, coverage outlines, soft studio lights, and denoising. Labels
and GIF palette conversion are applied after rendering. Large final assets use
Git LFS; full intermediate OBJ files and PNG frame sequences are reproducible scratch
outputs and are not committed.
