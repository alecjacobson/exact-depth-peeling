# Experimental contour subdivision and shared depth ranks

`EventPeeler` is a separate pure-Warp GPU route. It computes all ranks without
iterative depth peeling and keeps the existing exact CPU solvers as references.
The [latest measurements](../results/gpu-events-tuning/README.md) compare this
implementation with the first event prototype. The [original comparison](../results/gpu-events/README.md)
also includes iterative GPU and CPU fast mode. It remains finite precision.

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
3. Seed a flat frontier with projected source polygons. Vertices and sizes live
   in separate device arrays. A thread queries at most 32 contour candidates for
   one patch, rejecting segments that do not cross its interior or cannot be
   nearer. A relevant segment's supporting line splits the patch directly into
   the next frontier's vertex arrays. Both children retain the BVH continuation.
   Cleanup uses the reference GPU splitter's duplicate, area and perimeter policy.
   If cleanup collapses either child, carry the original patch forward instead.
   A captured device conditional loop processes the frontier until empty. There
   are no private growing fragment collections or by-value polygon matrices in
   the splitting loop. Finished patches convert to the common output format.
4. Choose a canonical interior sample per completed patch and group identical
   coordinates. Also propose groups using 32-by-32 image-space buckets. The grid
   only proposes sharing; it never clips geometry or approximates coverage.
   For each bucket with distinct samples, query the source BVH over their actual
   bounding rectangle. Accept sharing only if **every candidate polygon strictly
   contains all four corners**, with a conservative edge margin. Any failure
   retains the separate exact-coordinate groups. On intersection-resolved input,
   accepted rectangles have constant source membership and depth order. Internal
   tessellation boundaries can prevent certification even on a continuous sheet.
   This is deliberately conservative. Use `share_regions=False` in the device
   constructor to disable this optimization for comparison.
5. Gather hits once per accepted group. Each thread handles up to 32 candidates;
   longer queries enqueue their unvisited DFS suffix. A persistent worker grid
   partitions those suffixes into disjoint BVH subtrees. Partition count adapts
   to the number of heavy queries. Hits append to a bounded global pool while
   per-group counts accumulate; a prefix scan determines sorted group offsets.
   Half-open edge ownership counts a tessellated sheet once on a shared edge.
   This visits each candidate once, without separate count/fill traversals.
   Certification of heavy rectangles uses the same subtree scheduling principle.
6. Stable source/depth/group sorts order hits. Device conditional graph branches
   select a sufficient power-of-two sort length (or the final pool capacity),
   starting at 256 entries. Scratch is warmed at full capacity before capture;
   no counts are read on the host. Each polygon binary-searches its group's
   shared order, evaluating candidate depth planes at **that polygon's sample**.
   Using depths evaluated at a different sample would be wrong for tilted planes.
7. Reuse boundary noding, conforming triangulation and original-plane lifting.
   A compact list assigns one task per actual polygon edge instead of reserving
   32 lanes for every polygon. The geometric meshing function is shared with the
   original GPU route. Holes and disconnected regions remain unions of convex
   patches, not fans spanning unrepresented holes.

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
contours, at most 3K contour tests, one ray group, K stored hits and K ray
candidate tests. All K ranks come from one sorted list.
Sorts and grouping replace quadratic independent rank counts. This is a tested
control, not a general subquadratic guarantee.

This is **contour-guided supporting-line subdivision**, not a full implementation
of R2's boundary-event scan or a minimal arrangement of finite segments. Extending
lines can create extra subdivisions, and fragmented descendants may revisit
candidates. A long subdivision path requires many frontier rounds. Certified rectangles also share distinct sample locations when the entire
rectangle lies inside every candidate source. A nested-footprint control verifies
K certification tests plus K ray candidates instead of K independent K-hit queries.
However, that control still has quadratic contour broad-phase work. On the 64K
dragon almost all proposed rectangles fail certification. Q unshared samples can
still inspect QK overlapping sources. General shared ordering across changing
footprints, including across source tessellation boundaries, remains unfinished.

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
A collapsed-child cut carries its parent plus an empty reserved sibling slot to
the next round. Thus peak frontier occupancy and task visits can differ from the
original prototype despite identical output geometry. Other padded BVH builds,
query sorts and scans still process allocated capacity.

Overflow invalidates **all** output via sticky `status[9]`, with inherited error
bits plus 1024 for shared-hit capacity and 2048 for subdivision rounds. Check
that device flag before consuming output; `read_status()` is an optional explicit
host check. Replay resets errors and counters. Allocation and sort scratch
warmup occur before capture; the graph has no host transfers or allocation nodes.

`read_status()` reports frontier task visits, unique contours, contour tests,
splits, shared ray groups, ray candidate tests, stored ray hits and subdivision rounds. Additional counters
report heavy ray groups, attempted/certified rectangles, certificate candidate
tests and heavy certificate groups. These are not peeling passes or pixel
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

The [tuning report](../results/gpu-events-tuning/README.md) retains full-pipeline
and stage timings, controls with certification enabled/disabled, fresh old/new
comparisons, full-dragon results and validation. The old serial heavy-ray
bottleneck is removed; finite supporting-line cuts, conservative sharing,
capacity-dependent work and finite-precision robustness remain open issues.
