# Experimental contour subdivision and shared depth ranks

`EventPeeler` is a separate pure-Warp GPU route. It computes all ranks without
iterative depth peeling and keeps the existing exact CPU solvers as references.
The [measurements](../results/gpu-events/README.md) compare complete GPU extraction
with the old GPU route and CPU fast mode. It remains finite precision.

```bash
python3 -m warp_peel scene.json --method events --out layers.npz
python3 scripts/benchmark_events.py results/global/dragon-cache/65536/resolved.json --audit
python3 scripts/benchmark_event_stacks.py
```

```python
from warp_peel import EventPeeler

p = EventPeeler(vertices, faces, camera, settings)  # same device arrays as DevicePeeler
p.capture()
p.run()  # asynchronous; outputs remain on p.stream
```

Input/output array formats, source provenance, stream ownership, external capture
and explicit host download follow [the original device API](README.md).
`warmup()` precedes an external `ScopedCapture`; call `enqueue()` on the workspace
stream within that capture. There is no host geometry fallback or host loop over
layers/tasks. Update input values on the workspace stream without recapturing;
shapes, capacities and addresses must stay fixed.

## Implemented algorithm

1. Reuse the existing device validation, projection, welding and mesh contour
   classification. Input surface intersections must already be resolved; retain
   open, silhouette, non-manifold and camera-clipping boundaries. Both surface
   entry and exit triangles participate.
2. Extract projected contour segments and canonicalize endpoint order. Stable
   radix sorts group exactly coincident segments. For each group, take the
   minimum depth score at each endpoint; their interpolant is a conservative
   lower bound on the nearest segment. Build a BVH over these unique segments.
   This grouping removes duplicate clipping candidates for coincident-footprint
   stacks. It is only a clipping acceleration: it does not discard surface hits
   or reduce silhouette/non-manifold multiplicity during rank assignment.
3. Seed a flat frontier with the projected source polygons. Each thread queries
   at most 32 contour candidates for its current convex patch. Reject candidates
   whose finite segment does not cross the patch interior or cannot be nearer.
   A relevant segment's supporting line splits the patch into two convex pieces;
   append both children with their BVH traversal continuation to the next
   frontier. Otherwise continue or emit the finished patch. A captured device
   conditional loop processes the frontier until empty. A task owns one polygon,
   never a private collection of fragments; work passes to parallel children.
4. Choose a canonical interior sample per completed convex patch. Sort sample
   coordinates and group exactly equal locations. Count then gather all source
   hits once per group, using half-open edge ownership to count a tessellated
   sheet once on a shared edge. Sort hit records by `(group, depth, source)`.
   Binary-search each patch's source depth within its group's ordered hits to
   obtain its absolute rank. No separate construction pass for each layer occurs.
5. Reuse device boundary noding, conforming triangulation and lifting to the
   original source plane. The resulting regions can represent holes and multiple
   components as unions of convex patches; they are not triangulated as a single
   fan across an unrepresented hole.

A resolved mesh has locally constant source rank away from retained contours:
coverage changes at a contour, and depth swaps would require a surface
intersection. Subdivision isolates those changes; point queries then assign the
constant rank on each open patch. This is the mathematical rationale, not an
exactness guarantee for the floating-point implementation.

Coincident contour consolidation is conservative: the segment between the
endpoint minima lies at least as near as the minimum of the individual affine
scores. It may introduce extra cuts when different segments supply the two
minima. It cannot justify dropping hits from rank queries.

## What this prototype does and does not establish

For K identical footprints at distinct depths, the measured work is three unique
contours, at most 3K contour tests, one ray group, K stored hits and 2K ray
candidate tests (count and fill). All K ranks come from one sorted list.
Sorts and grouping replace quadratic independent rank counts. This is a tested
control, not a general subquadratic guarantee.

This is **contour-guided supporting-line subdivision**, not a full implementation
of R2's boundary-event scan or a minimal arrangement of finite segments. Extending
lines can create extra subdivisions, and fragmented descendants may revisit
candidates. A long subdivision path requires many frontier rounds. Exact sample
sharing helps identical footprints but does not share order between nearby,
different sample locations: Q distinct samples can still inspect QK overlapping
sources. General shared ordering across changing footprints remains unfinished.

The code avoids a common subdivision imposed on every source. In particular,
farther contours do not cut a foreground source, and the foreground-cover test
requires that source to remain one polygon. It does not promise minimal output
for every partially occluded source, and does not merge all equal-rank pieces.

Prefixes use the same all-rank construction and suppress output beyond the
requested rank. They have no early-out performance advantage. Empty suppressed
polygons remain within the output polygon prefix with `n == 0`; triangle outputs
contain only requested layers. Full all-layer benchmark outputs have no such
suppressed polygons.

Numerics are binary64 with the existing padded binary32 bounds and approximately
`1e-10` normalized-domain cleanup. No rational or adaptive-exact fallback runs on
the GPU. Source index resolves exact depth ties. Very thin regions and nearly
coincident surfaces remain numerical risks; passing sampled rays is not a full
region certificate. Unresolved crossings are outside this route's contract;
fixture tests preprocess them with the exact intersection resolver.

## Bounded storage and counters

`EventCapacities.for_mesh(N)` uses a sizing base of `max(N,16)`: 4N frontier
patches, 8N output regions, 32N shared ray hits, 32N boundary vertices and 32N mesh
triangles. Up to eight contour slots per clipped source are sorted; duplicate
slots become inactive hierarchy leaves. Layer capacity is the sizing base, and
there are at most 4,096 subdivision rounds. Polygon and edge-noding limits remain
32 and 128 vertices respectively. Defaults are empirical, not geometric bounds.
Use `dataclasses.replace` to customize the capacities, or CLI `--pool-scale` to
scale the five pools. No worker-private fragment collections participate in extraction.

Overflow invalidates **all** output via sticky `status[9]`, with inherited error
bits plus 1024 for shared-hit capacity and 2048 for subdivision rounds. Check
that device flag before consuming output; `read_status()` is an optional explicit
host check. Replay resets errors and counters. Allocation and sort scratch
warmup occur before capture; the graph has no host transfers or allocation nodes.

`read_status()` reports frontier task visits, unique contours, contour tests,
splits, shared ray groups, ray candidate tests across both count/fill traversals,
stored ray hits and subdivision rounds. These are not peeling passes or pixel
samples. Counts exclude BVH internal-node tests. The inherited `depth_ties`
counter is not instrumented in this experimental rank sorter and must not be
interpreted as proving that no ties occurred.

## Validation and reproduction

```bash
python3 -m pytest tests/gpu -q
python3 scripts/benchmark_events.py results/global/dragon-cache/{1024,2048,4096,8192,16384,32768,65536}/resolved.json results/large/scenes/sculpted-102400.json --audit
python3 scripts/benchmark_event_cpu.py results/global/dragon-cache/{1024,2048,4096,8192,16384,32768,65536}/resolved.json results/large/scenes/sculpted-102400.json
python3 scripts/profile_events.py results/global/dragon-cache/65536/resolved.json
```

Tests compare source-labelled region unions with the independent exact
arrangement solver, verify triangulated output/reconstruction on direct fixtures,
and cover holes, closed surfaces, non-manifold edges, camera clipping, prefixes,
capacity failures, changed-input replay and absence of allocation/transfer nodes
in captured graphs. Crossing fixtures are resolved before the region comparison.
The separate stack test verifies work counts as well as every assigned rank.

A high-depth shared ray is still counted/filled serially by one thread. The
65,536-surface control demonstrates linear geometric work but takes about 477 ms;
linear work alone does not ensure GPU utilization. The separate stack profile
places about 453 ms in rank assignment. Cooperative traversal of heavy
ray groups is a remaining performance experiment. The main dragon workload has
many independent groups instead.
