# GPU strategy review, 16 October 2026

Status: proposed redesign and experiment gates, not an implemented speedup.
The current device pipeline remains the working baseline. Its measured CPU/GPU
gap warrants changing the visibility algorithm, not just tuning its BVH.

Implementation update: the separate [EventPeeler prototype](EVENTS.md) now passes
the fixture and identical-stack gates and has [measured results](../results/gpu-events/README.md).
It implements contour-guided supporting-line subdivision and exactly shared ray
locations, rather than the full finite-segment event arrangement or general
shared ordering proposed below. The original review is retained to distinguish
those remaining requirements from the completed prototype. The
[connectivity follow-up](../results/gpu-events-components/README.md) now propagates
ranks across certified patch adjacencies, including internal mesh edges. It
reduces ray work but has mixed total-time results; finite-boundary construction
and general shared ordering remain incomplete.

## Diagnosis

`kernels.classify` uses contours to certify whole-source depth ranks. For
uncertain sources, `peel` and `cut` still subtract candidate polygons serially
from each target's growing list of pieces, repeating on subsequent layers.
Contours reduce the number of targets but are not the clipping representation
for the expensive cases.

The default clipping launch has at most 4,096 threads (128 CUDA warps). A thread
owns traversal, candidates, intermediate pieces and all polygon edges for one
target, then advances by a fixed stride. Slow targets cannot distribute their
work among idle threads. Each polygon contains 32 binary64 coordinate pairs;
clipping passes these large values through several temporaries and copies.
Three worker-private pools reserve about 12.2 GiB at the large-mesh defaults.
Register spills, occupancy and actual memory traffic still need hardware
profiling; their individual costs have not been measured.

The retained [64K dragon benchmark](../results/gpu/benchmark.json) records:

| Quantity | Value |
| --- | ---: |
| Input faces / peeling passes | 65,544 / 16 |
| Target visits across all passes | 662,385 |
| Candidate tests across all passes | 6,654,454 |
| Fully hidden target visits restored whole | 374,806 |
| Output polygons / triangles | 354,632 / 1,654,986 |
| GPU full graph median | 9.767 s |
| CPU fast, four threads, recorded median | 2.680 s |

Visits/tests are sums of `per_layer_counts`, not unique pairs or polygon splits.
The [separate diagnostic](../results/gpu/phases.json) attributes 99.1% of its time
to clipping; it cannot identify the hardware cause. CPU/GPU records have
different load and setup boundaries, documented in their reports.

There are two problems: expensive execution per visit, and repeated geometric
work. Queuing the existing subtraction operations addresses only the first.
Even contour-certified targets independently count all nearer sources: K targets
sharing a footprint can incur quadratic rank-query work before peeling. BVH
pruning cannot reject genuine overlapping candidates merely to avoid that cost.

## Preferred experiment: boundary events with shared depth ordering

Keep polygonal, source-labelled output, original-plane reconstruction, pure Warp
execution, device-only control and allocation-free graph replay. Keep the exact
CPU oracle. Pixel depth peeling motivates small parallel work units but does not
establish a matching complexity bound for analytic polygon output.

For **all layers**, investigate computing rank changes once at analytic boundary
events, replacing repeated subtraction:

1. Retain open, silhouette and non-manifold edges, plus clipping-generated
   boundaries. Preserve incident-face multiplicity and orientation. Internal
   edges can cancel for coverage counting; source boundaries still matter for
   final provenance. Do not backface-cull exit surfaces.
2. Generate compact edge/triangle and boundary-intersection tasks using
   conservative spatial/depth bounds. Process short records in parallel, with
   flat endpoint, parameter, source-ID and offset arrays. Subdivide overloaded
   tasks instead of assigning unbounded polygon collections to one thread.
   Spatial bins or beams partition work; they do not approximate the image.
3. Sort/scan events and share depth order across compatible regions. For K
   nonintersecting surfaces with an identical footprint, collect and sort hits
   once, assigning all K ranks in O(K log K) comparison work plus output.
   Avoid K independent counts of those same hits. Generalizing shared order to
   changing footprints is a central prototype requirement, not a solved detail.
4. Construct source-local rank boundaries where nearer contours change that
   source's rank. Reject farther contours for that target. Node actual boundary
   intersections, assemble labelled regions, triangulate and lift once. Avoid
   overlaying every scene contour on every source or exporting arbitrary beam
   partitions as visibility boundaries.

For an intersection-resolved mesh, the number of nearer hits is locally constant
inside a source footprint away from retained contours: coverage changes at a
boundary, and depth order cannot swap without a surface intersection. This
motivates the representation; it is not a complete construction algorithm or a
finite-precision robustness proof.

The prototype must resolve these issues explicitly:

- **Rank seeds:** contour jumps give rank differences, not absolute ranks. Seed
  queries need shared sorting. One query per source or cell can reintroduce
  quadratic work. One sample does not certify constant coverage/order across a
  beam; geometric certificates and event processing/subdivision are required.
- **Active order:** sorting endpoints by edge parameter alone does not order
  overlapping depth intervals. Specify an active-order merge/sweep without
  pairwise comparisons on the identical-stack control. Cyclic occlusion rules
  out sorting all scene triangles once into a single depth order.
- **Multiplicity:** closed silhouettes can change nearer surface-hit counts by
  two; open edges by one. Non-manifold incidences need signed sums. A binary
  coverage mask or a uniform unit jump is insufficient.
- **Topology:** retain closed loops wholly inside a target, holes, disconnected
  regions and thin features. Group coincident events consistently before rank
  propagation. Include target boundaries, propagate ranks across faces, assemble
  holes and mesh them; triangle fans alone cannot mesh arbitrary holed regions.
- **Seams:** canonical event identities and consistent constructions must make
  neighboring tasks agree. Remove artificial cuts or bound and disclose their
  cost. Global stitching/noding must not become the new dominant stage.

Keep bounded cooperative/wavefront clipping as an experimental control. If event
ordering and assembly cost more, change direction based on that evidence.

## What the papers establish

[R2, Sections 3–4](https://www.cg.tuwien.ac.at/research/publications/2013/Auzinger_2013_AnaVis/Auzinger_2013_AnaVis-Paper.pdf)
decomposes analytic visibility into edge/triangle intersections, sorted events,
scans and boundary completion, using cooperation within each edge. This is a
better GPU work model than one thread owning a growing polygon set. However,
boundary completion includes pairwise depth tests; the output is boundary
segments, and analytic peeling is future work. It supplies neither a ready-made
all-layer mesher nor the desired dense-overlap complexity guarantee.

[R3, Sections 4, 6 and 7](https://cseweb.ucsd.edu/~ravir/lifan_vectorizer.pdf)
maintains visible regions through incremental insertion and a visibility BVH.
Its reported implementation/timings are CPU based; GPU adaptation is discussed
as future work. Its selective construction remains useful, but its speed does
not validate the current per-target GPU subtraction architecture.

The existing global baseline uses a common subdivision and repeated rank
queries. Source-local depth filtering and shared ordering must demonstrate that
they avoid those costs. Calling a common arrangement tiled does not improve its
asymptotics. All-layer and prefix workloads also differ: processing deep events
is useful for all layers but can waste work for a short prefix. Retain peeling
for prefix comparisons.

## Gates before replacing the baseline

Prototype event generation, shared ranks and region assembly on exact-oracle
fixtures before a production rewrite. Measure stage times, candidate records,
actual intersections, rank-query visits, peak storage and output size. Hardware
profile the old clipping kernel to separate execution cost from geometric work.

| Control | Required evidence |
| --- | --- |
| K identical footprints, distinct depths | Shared ordering; no quadratic rank queries or pairwise interval ordering |
| Foreground over crossing background strips | Foreground stays whole; measure prefix work separately |
| Disjoint occluders over a large face | Correct holes; no all-pairs construction just because K candidates exist |
| Crossing nearer contours | Construct necessary intersections and account for actual output growth |
| Silhouettes, open/non-manifold edges | Correct rank jumps and all entry/exit surfaces |
| Cycles, slivers, near-plane cuts | Correct labels/topology within the numerical contract |

With K relevant boundaries and I actual intersections, sorting/construction
should depend on events and I, not gratuitous K-squared pair resolution. This is
a design target, not a proven pipeline bound: I and required output can be
quadratic, broad phases can degenerate, and repeated seeds or duplicated beam
work can dominate.

Then run cached 1K, 2K, 4K, 8K, 16K, 32K and 64K dragons and the 102K sculpted
mesh. Compare complete source-labelled unions against the exact oracle where
tractable; check conservation, prefixes, seams and reconstruction; retain
independent large-scene ray audits. Do not hide current numerical failures or
treat equal areas/sample agreement as a complete certificate.

Report both complete-pipeline and matching extraction-only scopes, CPU one/four
threads, old/new GPU, warmups, repetitions, memory and concurrent load. Disclose
numerical-policy differences. Preserve graph replay, changed-input and explicit
capacity-failure checks. Beating CPU fast mode is the first performance gate;
near-real-time all-layer extraction remains a separate, unachieved target.


## Frontier and shared-rank follow-up

The follow-up [implementation](EVENTS.md) moves frontier vertices into flat device
arrays, splits long ray queries over disjoint BVH subtrees, gathers hits in one
pass, chooses hit-sort capacity on device and schedules only actual output edges.
It also certifies shared depth order for distinct samples whose whole bounding
rectangle lies inside every source candidate. Failed certificates retain separate
queries; buckets do not approximate coverage. This is a useful special case of
shared ordering, not a general active-order sweep. Supporting-line subdivision
and the general QK query bound remain. See the [new measurements](../results/gpu-events-tuning/README.md)
for regression gates and remaining bottlenecks.
