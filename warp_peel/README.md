# Device-resident Warp peeling

`DevicePeeler` extracts analytic polygon layers and reconstructs their 3D triangle
meshes using Warp kernels. It ports the CPU contour-aware BVH peeling algorithm;
it does not call the C++ solvers, rasterize a depth buffer, or construct a global
arrangement. The CPU exact routes remain the correctness references.

This is a finite-precision GPU implementation, not an exact solver. Its first
priority is a complete, replayable device pipeline. See the
[measured results and limitations](../results/gpu/README.md) before choosing it
for performance. The near-real-time goal is not yet achieved.

## Run

Requires Python 3.10+, an NVIDIA CUDA GPU, CUDA conditional graph support
(driver CUDA 12.4+), and Warp 1.15.0. Tested on an L40 with driver 570.158.01
(CUDA driver 12.8). Dependencies are separate from the CPU build:

```bash
python3 -m pip install -r requirements-gpu.txt
python3 -m warp_peel tests/fixtures/partial.json --out /tmp/partial.npz
python3 -m warp_peel results/global/dragon-cache/1024/resolved.json --out /tmp/dragon.npz
```

The CLI performs explicit upload and download outside extraction. NPZ fields
are `uv` (normalized image coordinates), `world`, `sources` (input source IDs),
and `layers` (**zero-based**). UV/world arrays have shape `(triangle_count,3,2/3)`.
Each output triangle retains its input source and lies on that source plane,
within floating-point error. The JSON sidecar identifies the approximate format.
Use `--layers 1` for a prefix; zero requests all layers.

## Device API and graph capture

```python
import warp as wp
from warp_peel import DevicePeeler

# Existing CUDA arrays, already populated by the caller:
# vertices: wp.vec3d[V]; faces: wp.vec3i[N]
# camera: wp.mat44d[1], affine world-to-camera transform
# settings: wp.float64[6] = xmin, ymin, xmax, ymax, perspective_0_or_1, near
peeler = DevicePeeler(vertices, faces, camera, settings)
peeler.capture()  # allocation/JIT/warm-up happen before capture
peeler.run()  # asynchronous graph replay on peeler.stream
# Consume triangle buffers and status on that stream, or wait on it explicitly.
```

For composition into another graph:

```python
peeler.warmup()
with wp.ScopedStream(peeler.stream), wp.ScopedCapture(stream=peeler.stream) as cap:
    peeler.enqueue()
    # Launch downstream device consumers here; use status[3] as triangle count
    # and require status[9] == 0 before reading valid output.
wp.capture_launch(cap.graph, stream=peeler.stream)
```

After upload, validation, welding, topology, projection, contour classification,
BVH construction/refitting, every peeling pass, output compaction, boundary
noding, triangulation and lifting stay on the device. A CUDA conditional `while`
node evaluates the device continuation flag. There is no host layer-count read,
host geometry fallback, or allocation during extraction. The Python loops that
record kernels depend only on fixed array capacities and tree levels.

Allocation, JIT and primitive scratch warm-up precede capture. Standard Warp GPU
radix-sort primitives are used alongside the Warp geometry kernels; there
is no project-specific CUDA/C++ extension. Prefix sums use a preallocated Warp
upsweep/downsweep tree because Warp 1.15’s native scan records temporary allocation
nodes in the graph. The native Warp BVH rebuild was not
used because its temporary allocations fail inside a conditional graph body in
the tested version. The hierarchy here uses preallocated Warp arrays throughout.

Each workspace has its own stream to isolate Warp's cached radix-sort scratch.
Capture on that stream and keep the workspace alive as long as any graph uses
it. Update inputs on the same stream, or establish an event/stream dependency.
Positions, topology indices and camera **values** can change without recapture;
array addresses, sizes, capacities and layer limit remain fixed. Workspaces may
run concurrently on their own streams, but one workspace must not be used by
overlapping extractions. Atomic appends make buffer order and some floating-point
subdivision choices nondeterministic; outputs are identified by source and layer.

`read_status()` synchronizes and downloads a small status array for debugging or
serialization. It is optional and never called by `enqueue()` or graph replay.
`warp_peel.io.download_mesh()` is likewise an explicit host consumer.

## Algorithm

1. Validate device inputs, weld exactly equal vertex positions with stable GPU
   radix sorts, and sort undirected edge keys. An edge is internal only when its
   two oppositely directed incident faces have the same projected facing.
   Open, silhouette and non-manifold edges become conservative contours.
2. Transform vertices, clip against the perspective near plane and image window,
   orient projected convex polygons, and compute affine camera-depth planes
   (negative reciprocal depth for perspective). Clipped polygons expose all
   their boundary edges conservatively.
3. Build Morton-ordered balanced BVHs for source polygons and contour-bearing
   polygons. If no contour touches a source footprint, determine its whole-face
   depth rank with one source ray query. Ambiguous samples enter ordinary peeling.
   Whole faces with certified ranks can be emitted directly.
4. In each device-controlled pass, refit source bounds to retire known ranks
   below the current layer and rebuild the residual BVH. A worker owns one target
   patch at a time, queries overlapping candidates that can be in front, and
   subtracts them using convex half-plane clipping evaluated relative to each
   clipping edge (to avoid cancellation for short edges). It saves visible pieces in
   this layer and hidden pieces for the next pass. When fully hidden, it carries
   the original incoming patch instead of its temporary subdivision. Candidate
   geometry is frozen throughout the pass; targets write into separate pools.
5. Scan output polygon sizes with a preallocated prefix-sum tree, gather layer-labelled boundary vertices, and build
   a point hierarchy. Each polygon edge queries nearby same-layer vertices once,
   sorts its collinear parameters with a bounded heapsort, and splits at those
   points. Centroid fans triangulate the noded boundaries. Affine depth and the
   inverse camera lift the triangles back to their original scene positions.

The hierarchy is a Morton-sorted complete binary tree, with fixed-capacity
padding and stackless traversal. It is not an SAH tree. Rebuilds sort all allocated
leaves; later layers still pay that capacity cost. Clipping avoids all-pairs
candidate arrangements, but **does not provide a general subquadratic guarantee**:
candidate traversal, fragment growth and repeated passes can still be costly.
Edge noding uses one query plus `O(K log K)` parameter sorting, not repeated
searches for each next split point. The fixed polygon arrays and long per-target
loops are the present GPU implementation's main performance concern.

## Numerical contract and capacity

Resolve surface self-intersections before supplying a mesh, as with the CPU
contour path. Preprocessing is outside this graph; the repository's exact CPU
preprocessor and cached resolved dragons can provide those inputs. Non-manifold
edges are conservatively retained as contours. No automatic intersection repair
or exactness certification runs on the GPU.

Geometry uses binary64. BVH bounds use padded binary32 coordinates. The image
window is normalized to `[0,1]^2`; remnants narrower than approximately `1e-10`
in that domain are discarded, and nearly coincident boundary points are merged
at that scale. Strict signs determine clipping depth order; ambiguous whole-face samples
fall back to clipping. Polygon cleanup uses an anchored area/perimeter test
rather than rejecting an entire polygon because of a tiny reversed edge. Exactly
zero depth-difference planes use dense source index as a deterministic tie rule,
which differs from the CPU exact solver's positive-area-tie rejection. Whole-face
classification falls back to peeling near depth ties or source boundaries.
Very thin features and near-coincident surfaces can still lose or change ranks;
the floating route is not a correctness oracle.

The affine camera must be finite and invertible, with last row `[0,0,0,1]`.
The window has positive dimensions and the near distance is positive. Computed
depth-plane coefficients must be finite with magnitude at most `1e28`. Invalid
indices, camera settings or unsupported numeric range set the device error flag.
Degenerate projected triangles contribute no area.

All storage is bounded and allocated up front. `Capacities.for_mesh(N)` uses
`4N` residual patches, `16N` output polygons, `96N` boundary vertices and `64N`
output triangles (with a minimum sizing base of 16). Each polygon holds at most
32 vertices. Up to 4,096 workers have three private piece pools each; defaults
are 512 pieces through 8,192 input faces and 2,048 above that. Maximum layers default to 128 (configurable); edge candidate storage has a
hard limit of 128, and `edge_vertices` may lower that limit. These are empirical defaults,
not geometric upper bounds. Large defaults consume substantial GPU memory:
private clipping scratch alone is about **12.2 GiB** at 4,096 × 2,048 pieces.
Use `dataclasses.replace(Capacities.for_mesh(N), ...)` to choose capacities;
`--pieces` and `--pool-scale` expose common CLI overrides.

Capacity exhaustion sets a sticky error flag. **All output is invalid when
`status[9] != 0`**, even if some triangles were written. There is no silent
truncation, host retry, or CPU fallback. Reallocate/recapture explicitly with
larger capacities if needed. `read_status()` raises with the failing pool name.
A replay resets counters and errors. Explicit prefixes are valid partial
extractions; unresolved patches remain in the residual pool, while directly
ranked sources beyond the prefix remain represented by `original` and `ranks`.

| Status slot | Meaning |
| --- | --- |
| 0 | Current residual count |
| 1 | Next-pass residual count (zero after completed pass) |
| 2 / 3 | Output polygon / triangle count |
| 4 / 5 | Number of output layers / peeling passes |
| 6 | Boundary vertex count |
| 8 | Exactly zero depth comparisons that used the source tie rule |
| 9 | Error bits: 1 polygon, 2 worker pieces, 4 residual, 8 output polygons, 16 boundary vertices, 32 triangles, 64 layers, 128 edge candidates, 256 input |
| 10 / 12 | Maximum accumulated hidden pieces per target / peak residual count |

`stats[layer]` contains targets processed, tested candidates, and fully hidden
targets restored whole. Device output buffers are `output`, `output_sources`,
`output_layers`, `triangles_uv`, `triangles_world`, `triangle_sources` and
`triangle_layers`. Counts delimit valid prefixes; unused capacity is unspecified.

## Validation and reproduction

```bash
# Requires the existing exact executable and optional test dependencies.
python3 -m pip install -r requirements-gpu-test.txt
python3 -m pytest tests/gpu -q
python3 scripts/benchmark_gpu.py results/global/dragon-cache/{1024,4096,16384,65536}/resolved.json
python3 scripts/audit_gpu.py results/global/dragon-cache/{1024,4096,16384,65536}/resolved.json
```

The optional CMake switch `-DPEEL_WARP_TESTS=ON` also registers `warp_gpu`
with CTest; CPU-only builds leave it off.

Fixture tests compare complete source-labelled layer unions with the independent
exact arrangement oracle, then check triangle union/area, source-plane lifting,
reprojection, selected T-junction cases, prefixes, failure flags and changed-input
replay. Capture tests prohibit host readbacks/synchronization during recording
and inspect the CUDA graph for conditional control flow and absence of host
transfers/allocation nodes. Large-scene audits use independent rational rays;
passing samples do not certify every output region.

The implementation follows Warp's documented
[`capture_while` interface](https://nvidia.github.io/warp/v1.15/api_reference/_generated/warp.capture_while.html).
