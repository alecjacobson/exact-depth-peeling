# Exact geometric depth peeling prototype design

Build a CPU prototype that turns a triangle mesh and a fixed camera into an ordered sequence of polygonal visibility layers, then triangulates each layer in the image plane. Preserve the source triangle and the exact mapping back to its original 3D plane for every output face. The experiment should determine whether peeling avoids unnecessary geometric work on foreground regions compared with building an arrangement for all depth layers at once.

The immediate deliverable is a working research prototype, reproducible fixtures, and measured results. Use exact arithmetic for the reference implementation. GPU acceleration is the next step but not needed now. Differentiability, and production packaging are later work.

Prepared 14 October 2026. References and code links appear at the end and use labels such as [R1] and [C1].

# 1 Research question and success criteria

A foreground triangle may cover a complicated mesh while remaining entirely visible. Its layer should contain the original triangle, even if the projections of hidden edges have thousands of intersections. A global arrangement of projected edges can unnecessarily cut that foreground triangle at every background event.

Test two separate hypotheses:

- **Output selectivity:** a layer contains only visibility boundaries and required source-triangle boundaries, followed by triangulation edges. Hidden geometry does not impose additional output boundaries.
- **Computational selectivity:** computing the first K layers avoids constructing some intersections and subdivisions needed only for later layers. K equal to 1, 2, or 4 is especially informative.

Do not infer the second claim from a small final output. An envelope algorithm may compute complicated intermediate geometry and discard it. Likewise, dissolving subdivisions after a global arrangement has been built improves output size but does not recover the work already spent.

Build three methods sharing geometry types, projection, validation, and output normalization:

| Method | Role | Main risk |
| --- | --- | --- |
| `arrangement` | Small-scene exact oracle and intentionally eager comparison | Constructs a common subdivision for all ranks |
| `envelope-peel` | Straightforward exact peeling baseline using CGAL envelopes | Repeated envelope construction and residual triangulation can be expensive |
| `visible-peel` | Main experiment using incremental maintenance of the current visible map | Performance depends on traversal order, coverage rejection, and intermediate fragments |

Success requires exact agreement of positive-area regions with the oracle, complete provenance, valid exported triangulations, and instrumentation that separates tests from constructions. A speedup is a hypothesis to measure, not a condition to manufacture. Report unfavorable cases as well as favorable ones.

# 2 Relation to existing work

Weiler and Atherton [R1] establish polygon-output hidden-surface removal using clipping, including polygons with holes. Auzinger, Wimmer, and Jeschke [R2] compute visible polygon boundaries on a GPU and explicitly propose analytic depth peeling as future work. Their implementation assumes nonintersecting triangles and uses an epsilon-based numerical treatment rather than exact constructions.

Zhou and colleagues [R3] maintain visible polygonal regions under incremental triangle insertion using a Visibility BVH. Their public code archive [C6] is a useful implementation reference. Their fixed-point rounding and convexity repair are not the numerical contract for this prototype. The arrangement-level literature [R4] supplies the mathematical depth-rank interpretation, but does not by itself supply this selective peeling implementation.

The experiment combines established visibility operations with a specific residual representation and performance comparison. Do not claim that geometric visibility, polygon clipping, or the notion of analytic depth peeling is new.

# 3 Geometry contract

## Initial scope

Use C++ with CGAL and CMake for the exact geometry core. Add small Python scripts for generating fixtures, running benchmarks, and plotting results. Start with an orthographic camera. Add a pinhole perspective camera after the orthographic methods agree.

Accept triangle soups as well as connected meshes. Mesh connectivity is useful metadata but is not required by the algorithm. Do not backface-cull: the requested layers represent all surface hits along a ray, including exits from closed objects.

Initial supported inputs have finite coordinates, a bounded image window, and no positive-area coincident surface overlap along the view. Shared edges, shared vertices, coplanar adjacent triangles, holes, and disconnected components are valid. Support noncoplanar triangle intersections once the depth-comparison half-plane implementation is present. Test cyclic occlusion even when triangles do not intersect in 3D.

Reject positive-area depth ties with a clear diagnostic in version 1. Do not silently perturb geometry, pick a triangle ID, or treat duplicate faces as separate depth layers. Distinct-depth peeling would remove all tied hits together; provenance for such groups can be added later. Nvdiffrast likewise skips further hits at an already peeled depth [C7].

Perform the tie preflight even on fully hidden geometry. One approach groups sources by identical exact affine-score coefficients, then tests domain overlaps only within each group. Adjacent coplanar faces sharing an edge have no positive-area overlap and pass. Include preflight time in the reported setup cost.

## Exactness and boundaries

Use exact predicates and exact constructions throughout projection, clipping, visibility decisions, Boolean operations, and triangulation. A suitable initial type is `CGAL::Cartesian<CGAL::Exact_rational>`, as used in the envelope example [C1]. A lazy exact kernel can be evaluated later with identical semantics.

Integer/rational synthetic inputs define exact fixtures. For imported floating-point meshes and cameras, convert their stored finite binary values to exact numbers before geometric transformations. Exactness then refers to those represented inputs. Do not project in floating point first and describe the result as exact with respect to the original mesh and camera.

Version 1 uses regularized planar sets: isolated points and line-only remnants are not standalone surface patches. CGAL's regularized polygon operations have this behavior [C2]. All region equality, partition, and ordering guarantees apply to open two-dimensional cells, with their closures retained for meshing. Shared boundary closures may belong to multiple adjacent patches.

Skip exactly zero projected-area triangles from the area representation and count them in diagnostics. Do not discard positive-area slivers using a tolerance. An application requiring ownership of every boundary ray or edge-on surface needs a separate lower-dimensional representation. The prototype does not promise bit-for-bit agreement with a GPU's fill convention or quantized depth buffer.

## Mathematical definition

For source triangle i, let P_i be its projected domain after camera and window clipping. Let d_i(u) be increasing depth along the ray through image position u. Away from ties and boundaries:

```text
rank_i(u) = number of j with u in P_j and d_j(u) < d_i(u)
P_i,k     = { u in P_i : rank_i(u) = k - 1 }
layer k   = the source-labelled collection of all nonempty P_i,k
```

A source triangle can contribute to several layers in different image regions. Never remove an entire triangle merely because some portion was visible.

Maintain a residual domain R_i for every original source triangle. At the beginning, R_i = P_i. At a peel:

```text
V_i = the region of R_i winning the nearest-depth comparison
      against all other current residual domains
emit V_i with source i
R_i = regularized_difference(R_i, V_i)
```

Previously emitted patches are absent from the next pass. Resolve visibility among the remaining patches; do not retain the previous layers as occluders.

Correctness follows ray by ray: initially the nearest remaining hit is the first hit. Removing exactly that hit makes the nearest remaining hit the second, and induction gives every subsequent rank. The argument requires neither a global triangle order nor the absence of cyclic occlusion. Source-domain clipping implements those removals simultaneously over open image regions.

## Camera representation

Represent every projected source surface by an affine ordering score s_i(u,v), with smaller scores nearer. This separates visibility comparisons from reconstruction.

```text
Orthographic camera:
    u = x, v = y, s = z
    s_i(u,v) = a_i*u + b_i*v + c_i

Normalized pinhole camera, with z > 0:
    u = x/z, v = y/z, q = 1/z, s = -q
    q_i(u,v) is affine on the projected source triangle
    (x,y,z) = (u/q, v/q, 1/q)
```

These formulas use camera coordinates; retain the inverse camera transform to reconstruct world positions. Incorporate intrinsics by working in normalized image coordinates or by composing the corresponding affine image transform. Clip to a strictly positive near plane before perspective division, then to the finite image window. Perform the clipping exactly. Camera clipping may create a polygon with more than three vertices; preserve its original source ID.

For the perspective envelope baseline, construct graph triangles in `(u,v,-1/z)` space and use their lower envelope. They are planar triangles in that transformed space. Do not feed `(u,v,z)` to an affine-depth envelope algorithm under perspective projection.

For every nondegenerate source, store the original 3D triangle and either the inverse projection-to-plane map or a projectively correct barycentric map. Image-space barycentric coordinates are not world-space barycentric coordinates under perspective projection.

# 4 Data model and output

Use these conceptual records; adapt the concrete containers to the chosen CGAL types.

```text
SourceTriangle
    source_id
    original_world_vertices[3]
    original_mesh_face_id
    affine_score_coefficients[3]
    projection_to_world_mapping
    initial_domain                  exact polygon set

Residual
    source_id
    domain                          exact polygon set, with holes/components
    conservative_projected_bounds
    conservative_score_interval

VisibleRegion
    source_id
    domain                          exact polygon set

Layer
    layer_index                     one-based
    regions[]                       authoritative geometry
    triangulation                   derived output
    face_source_ids[]
    corner_world_positions[]        or corner source barycentrics
```

Keep residual geometry grouped by original source ID. Dissolve boundaries between pieces belonging to the same source after each pass. Internal triangulation edges used by a solver must not become permanent residual boundaries.

Polygon sets and source plane maps are authoritative. Triangulations are adapters or exports. This avoids repeated interpolation through newly created 3D triangles and allows identical source regions from different algorithms to be compared exactly.

Export one conforming 2D triangulation per layer. Preserve source boundaries as constraints, even across coplanar neighboring input triangles, so each output face has an unambiguous source ID. A vertex shared in 2D across an occlusion boundary can have different lifted positions on its two sides. Store depth/3D data per face corner, or duplicate vertices when exporting a lifted 3D mesh. Never weld across a depth jump based only on equal image coordinates.

Provide an exact JSON representation with rational coordinates serialized as numerator/denominator strings, plus convenient floating-point SVG and mesh previews. Floating-point exports must never be read back into exact validation.

# 5 Exact peeling baseline

Use the CGAL envelope example and source-data traits [C1] to build `envelope-peel`.

```text
residual[i] = initial_domain[i]
for k in 1 .. requested_layers_or_until_empty:
    graph_triangles = []
    for each nonempty residual[i]:
        triangulate its domain exactly
        lift each 2D triangle to the graph of score_i
        attach original source_id i
        append to graph_triangles

    diagram = lower_envelope(graph_triangles)
    visible = extract bounded positive-area faces, with all boundary cycles
    visible = union faces having the same original source_id

    validate current layer and source containment
    emit visible
    for each source i appearing in visible:
        residual[i] = difference(residual[i], visible[i])
    assert progress or all residuals are empty
```

The example printer walks outer boundaries; the prototype must also extract every hole cycle. Handle cells with no inducing source as empty background. Multiple inducing surfaces with distinct original sources over a positive-area face are an unsupported tie. Multiple graph triangles belonging to one original source should be consolidated by that source ID.

This method establishes correctness and integration. It does not establish optimal avoidance of hidden work: the library can build intermediate envelope subdivisions for background subsets, and every pass may retriangulate complicated residuals. Record these costs separately. Do not spend the first milestone modifying CGAL's envelope internals.

# 6 Incremental visibility method

Implement `visible-peel` with the same outer peeling loop, replacing the envelope subroutine with incremental maintenance of visible regions. The proposed subroutine is motivated by incremental vectorized visibility [R3], but the following exact polygon-set implementation is a separate prototype design.

## Correct insertion before acceleration

At the start of a pass, the visible map M is empty. Insert each residual source domain once. The map has disjoint region interiors and stores the winning source in every covered region.

When inserting domain C of source i, use a snapshot of the previously visible regions. Let H_ij be the half-plane where `score_i < score_j`. In the pseudocode, operations are regularized and clip unbounded half-planes to the finite image window first.

```text
pending = C
snapshot = candidate regions of M whose bounds overlap C
updates = []

for old_region (j, D) in snapshot:
    if pending is empty:
        break

    overlap = intersection(pending, D)
    if overlap is empty:
        continue

    # Equality on an entire positive-area overlap is an unsupported tie.
    new_wins = intersection(overlap, H_ij)
    old_wins = difference(overlap, new_wins)

    if new_wins is nonempty:
        updates.append(replace D by difference(D, new_wins), source j)

    pending = difference(pending, old_wins)

apply updates to M
if pending is nonempty:
    add pending to M with source i
```

The old map regions are interior-disjoint. Therefore subtracting an old winner from `pending` cannot invalidate a replacement made in another old region. Boundaries are handled under the regularized-set convention. Begin with a simple vector of source-labelled polygon sets and rebuild the spatial index after an insertion if necessary; this is easier to validate than simultaneous graph and BVH mutation.

If the old region wins everywhere in an overlap, leave the old region geometrically unchanged. Do not split it along the incoming hidden boundary and later hope to clean it up. If the incoming source wins everywhere, preserve its pending domain while subtracting the overwritten part from the old region. Resolve true crossings with the affine half-plane comparison.

The stored residual domains remain untouched during insertion. Discard temporary rejected pieces of `pending` for this pass only. After the entire visible map is finalized, subtract its visible regions from the original residuals. Updating persistent residuals during insertion would incorrectly remove surfaces that are later occluded by another insertion.

## Order and conservative rejection

Compare at least three insertion orders: input order, deterministic random order, and increasing conservative minimum score. The last is a heuristic to encounter foreground geometry early; it is not an assumed global depth order. Output regions must be identical for all orders, including cyclic-overlap and intersecting examples.

Add accelerations only after the unaccelerated version passes exact comparisons:

1. **Bounds rejection:** skip regions whose conservative projected boxes do not overlap.
2. **Depth separation:** if score intervals certify one source is nearer throughout the relevant box, skip construction of the depth-equality cut. Coverage clipping may still be required.
3. **Complete occlusion certificate:** if the incoming domain's bounding rectangle is contained in one convex visible cell and that cell is strictly nearer throughout the rectangle, reject the incoming domain without constructing polygon intersections.
4. **Spatial index:** index current visible components or convex cells in 2D; use conservative bounds and update them after changes.

For the complete-occlusion certificate, test rectangle corners against every half-plane of the convex visible cell and evaluate affine score differences at the corners. Checking corners against an arbitrary polygon with holes is insufficient. If no single-cell certificate applies, fall back to exact polygon operations. Coverage by several visible cells can be an optional later certificate.

Use exact rational bounds first. If floating-point broad-phase bounds are added, round them outward and prove that rejection cannot discard a true overlap; ambiguous predicates fall back to exact evaluation. An uncertain query is a candidate, not a rejection.

Cache immutable source projection and plane data across peels. Rebuild indexes of residual or visible domains initially. Add cross-pass caches only after profiles identify a concrete benefit, with explicit invalidation when domains shrink. Do not reuse old occlusion coverage after removing the surfaces that supplied it.

## Expected behavior on hidden backgrounds

After a foreground covering triangle enters M, a background triangle strictly inside its projection should leave that visible triangle unchanged. With the convex coverage and score certificate, the background insertion should also avoid polygon intersection construction. Processing N background inputs still costs at least reading/classifying them; this design does not promise constant total runtime.

With reverse insertion order, background regions can be constructed before the foreground arrives and removes them. The contrast between insertion orders is part of the experiment. Even requesting all layers must not propagate later-layer cuts backward into completed earlier layers.

# 7 Arrangement oracle and fair comparisons

Implement `arrangement` only for small inputs initially, for example N at most 64. Increase the cap only after measuring memory use.

Construct an exact image-plane arrangement from the window boundary, every clipped source-domain boundary, and every pairwise depth-equality segment restricted to the overlap of the corresponding domains. For nonintersecting triangles the equality segments do not cut overlap interiors. For intersecting triangles they are essential. If two affine scores coincide identically, test positive-area domain overlap and reject that unsupported input.

Within every open arrangement cell, both domain membership and depth ordering are constant. Obtain a guaranteed interior rational point, for example by triangulating the cell with its holes and taking a nondegenerate interior triangle's barycenter. Do not use the arithmetic mean of polygon vertices as an interior-point algorithm.

Evaluate all covering sources and sort them by exact score at that point. Assign the cell to the corresponding source and rank. Retain the complete depth list for diagnostic queries. CGAL segment arrangements and overlay examples provide the supporting data structures [C3].

For comparison, union cells by `(layer, source_id)` and compare those sets with the peeling result by exact symmetric difference. Compare both raw arrangement output and this normalized output. It would be misleading to benchmark only an undissolved output against a cleaned peeling mesh.

Optionally add a per-source occlusion-count arrangement after the three required methods work. This is a stronger all-layer comparison than the global arrangement because it excludes triangles certified behind each source. It still resolves all coverage counts eagerly. A win over the global arrangement alone does not prove superiority over all possible all-layer algorithms.

# 8 Triangulation and reconstruction

For each finalized layer, collect only its visible-region boundaries and required source boundaries into one planar subdivision. Node shared endpoints and collinear overlaps exactly. Dissolve internal solver edges within each source region before assembling these constraints.

Use a constrained triangulation with exact geometry, adapted from CGAL's polygon-domain example [C4]. The published example uses an inexact-construction kernel; replace it with the exact kernel used by the prototype. Select an intersection policy consistent with exact constraint handling. Prefer to normalize constraints into a planar graph before insertion.

Classify each triangulation face with an interior point against the labelled visibility subdivision. Marking a single polygon with nested holes by parity, as in the example, is not enough to label a full layer containing adjacent source regions. Retain only covered faces and copy their source IDs. Verify that each triangle interior belongs to exactly one visible source region.

Test that the whole 2D layer has no T-junctions. Triangulating each region independently can produce nonconforming boundaries and is acceptable only as an explicitly intermediate adapter. Do not add quality refinement or Steiner vertices in the initial exporter unless required for validity.

For each emitted triangle corner, recover the exact point on the original source plane and source barycentric coordinates. Check barycentric nonnegativity and sum-to-one exactly, and verify that reprojection returns the same image coordinate. Use the source ID to preserve original attributes. For perspective cameras, use perspective-correct interpolation or the exact plane inverse.

# 9 Validation

## Exact checks

Run the following checks on small fixtures for every method and requested layer count:

| Check | Required result |
| --- | --- |
| Source containment | Every output region is contained in its clipped original source domain |
| Per-source conservation | Union of emitted regions and current residual equals the initial domain by regularized symmetric difference |
| Per-source disjointness | Emitted ranks and the residual have disjoint interiors |
| Layer uniqueness | At most one source owns any open image cell in a layer |
| Rank correctness | Overlay output and oracle boundaries; every open verification cell has the correct source and exact rank |
| Prefix consistency | First K layers equal the prefix of an independent all-layer run |
| Order invariance | All tested insertion orders give identical labelled region sets |
| Triangulation | Positive triangle areas, no interior overlaps, no T-junctions, and exact equality to visible regions |
| Reconstruction | Exact reprojection and valid original-source barycentrics |
| Completion | All-layer extraction leaves no positive-area residual; maximum layer count is at most N |

Area equality alone is not sufficient: a missing region and an extra region can have equal area. Use set comparisons and rank checks. On the all-layer result, additionally verify that the sum of projected layer areas equals the sum of clipped source triangle areas, counting overlap multiplicity.

Independent exact ray queries are useful spot checks, but samples alone cannot certify that a small patch is absent. On small scenes, use the full verification arrangement to check all open cells.

## Required fixtures

1. One triangle, with no clipping and then with window clipping.
2. Disjoint triangles and fully overlapping triangles at distinct depths.
3. Partial overlap where one source contributes to several ranks.
4. One large far triangle with several disjoint near triangles, producing holes in its first visible region.
5. Near strips splitting a far triangle into multiple components.
6. Three nonintersecting triangles with cyclic occlusion; verify the intended cycle geometrically.
7. Two noncoplanar triangles intersecting across their common projected interior, requiring an interior depth-equality cut.
8. Coplanar adjacent triangles with a shared edge, and several triangles meeting at a shared vertex.
9. Positive-area coincident triangles: deterministic explicit rejection in version 1.
10. Rational slivers, nearly parallel projected edges, and very close but unequal depths.
11. A closed triangulated object, retaining both entry and exit surfaces.
12. Perspective versions of overlap and crossing fixtures, including near-plane clipping.

Random small triangle soups with fixed seeds should supplement these cases. Record rejected unsupported cases separately from failures. Use exact rational generation so tiny features and depth separations are intentional rather than accidental floating-point noise.

## Raster comparison

Optionally rasterize the exported layers and compare with `nvdiffrast.torch.DepthPeeler` [C7]. Match camera, window, backface policy, and requested layer count. Compare source IDs and depth at sample locations away from boundaries and depth ties. Classify disagreements caused by finite depth precision separately. GPU comparison is a familiar visual and sampling check, not the exact oracle.

# 10 Performance experiments

## Foreground cover stress test

Place one triangle at z = 1 entirely inside the viewport. Place g horizontal thin strips at z = 2 and g vertical thin strips at z = 3, all strictly inside the foreground triangle's projection. Each strip is a rectangle represented by two triangles. Strips in the same family do not overlap. All vertices are rational. Use the pair of strip families to generate a quadratic number of projected background crossings as g grows.

Layer 1 must remain the original foreground triangle, with three boundary vertices and one output triangle. Background crossings can become relevant in deeper layers; they are irrelevant to constructing the first visible region.

Use g in {2, 4, 8, 16, 32, 64}, then increase only within the time and memory budget. Compare K in {1, 2, 4, all}; skip nonexistent ranks naturally. Compare all insertion orders. Add a variant with the foreground removed to expose the work that deeper layers genuinely require.

For `visible-peel` with foreground-first ordering and the complete-occlusion certificate, target zero background-induced subdivisions of the foreground and zero polygon-intersection constructions for certified fully hidden candidates. Broad-phase, containment, and depth tests still count. If the certificate implementation does construct intersections, explain and measure it rather than claiming this target was met.

## Other workloads

- **Stacked identical footprints:** increasing numbers of parallel surfaces. Shows repeated-pass cost when output per layer is simple.
- **Perforated foreground:** many small occluders over a far surface. Exercises holes, fragmentation, and collective coverage.
- **Mostly disjoint projections:** checks overhead when little occlusion exists.
- **Cyclic and crossing soups:** tests insertion-order sensitivity and exact depth cuts.
- **Connected meshes:** several modest meshes with different depth complexity; record the data source, license, camera, and backface policy. Begin around 1k faces and scale toward 10k only after profiling.

Run the exact arrangement oracle only at tractable sizes. Use explicit time/memory caps; report timeouts rather than omitting them. Do not extrapolate a speedup beyond the measured range.

## Required instrumentation

Record per pass and in total:

- Input sources, active sources, residual components, residual boundary vertices, and solver-adapter triangles.
- Candidate pair/region tests, box rejections, depth-separated cases, complete-occlusion rejections, polygon intersections, polygon differences, and depth-half-plane clips.
- Constructed vertices and edges, peak live geometry, and maximum intermediate visible-map complexity.
- Final source regions, holes, boundary vertices, and exported triangles before and after normalization.
- Time for projection/preflight, visibility, residual updates, triangulation, validation, and serialization separately.
- Peak resident memory, backend, insertion order, camera, exact number type, compiler, dependency revision, seed, and requested K.

For the library envelope baseline, expose the counters available at the adapter level and label unavailable internal counters as unavailable. Do not equate the number of adapter triangles with the number of internal constructions. Instrumenting CGAL internals is optional and should be justified by a specific remaining question.

Use a release build, a warm-up, and at least three measured repetitions for tractable cases. Report median and range. Keep exact validation outside the timed algorithm interval but include it in the reproducible run. Plot time and constructed geometry versus g or input size, and show foreground output complexity separately.

# 11 Implementation milestones and agent deliverables

Implement in this order and keep a working executable after each milestone:

1. **Geometry foundation:** exact import/projection, source maps, polygon sets, and a single-triangle export with round-trip checks.
2. **Exact reference:** small-scene arrangement oracle, `envelope-peel`, holes, and per-source conservation checks. Verify orthographic fixtures before performance work.
3. **Selective method:** unaccelerated `visible-peel`; establish oracle agreement and insertion-order invariance. Then add conservative ordering and full-occlusion certificates.
4. **Experiments:** foreground-cover and stacked-footprint sweeps, with construction counters and intermediate complexity.
5. **Perspective and broader inputs:** pinhole score transform, near clipping, noncoplanar intersections, modest connected meshes, and optional raster comparison.

Suggested layout:

```text
CMakeLists.txt
src/exact_geometry.*
src/camera.*
src/scene.*
src/arrangement_oracle.*
src/envelope_peel.*
src/visible_peel.*
src/triangulate_export.*
src/main.cpp
tests/fixtures/
scripts/generate_fixtures.py
scripts/benchmark.py
scripts/plot_results.py
results/
README.md
```

Provide a CLI along these lines; it is a proposed interface, not an existing executable:

```bash
geometric_peel --scene scene.json --method envelope-peel --layers 4 --out out/envelope
geometric_peel --scene scene.json --method visible-peel --order near-first --layers all --out out/selective
geometric_peel --scene scene.json --method arrangement --layers all --out out/oracle
geometric_peel --compare out/selective/exact.json out/oracle/exact.json
python scripts/benchmark.py --suite foreground-cover --output results/cover
```

Deliver source and build instructions; fixture/camera manifests; exact outputs and SVG layer previews; machine-readable benchmark tables; plotting scripts and standalone plots; and a short findings report. The report must answer whether foreground output stayed minimal, whether construction work was avoided, how sensitive performance was to order, where repeated peeling lost, and whether the next investment should be indexing, coverage certificates, residual representation, or a different algorithm.

Record dependency versions and the commit used for external code. Use the existing project environment if an agent receives this document inside a repository. Do not replace project build conventions unnecessarily. No GPU implementation is required to complete this experiment, but decisions should not be made that would preclude it later as it is the immediate next step.

# 12 Existing code to start from

**[C1] CGAL 3D Envelopes.** Start with the versioned [triangle envelope example](https://doc.cgal.org/6.2/Envelope_3/Envelope_3_2envelope_triangles_8cpp-example.html) and [manual](https://doc.cgal.org/6.2/Envelope_3/index.html). Relevant types are `Env_triangle_traits_3`, `Env_surface_data_traits_3`, and `Envelope_diagram_2`; the operation is `lower_envelope_3`. The example attaches source data and uses an exact rational Cartesian kernel. Extend extraction to include holes and original source IDs. The package is an envelope building block, not a ready-made multilayer exporter.

**[C2] CGAL planar Boolean operations.** The [manual and source examples](https://doc.cgal.org/6.2/Boolean_set_operations_2/index.html) include `Boolean_set_operations_2/sequence.cpp`, using `Polygon_set_2`, `Polygon_with_holes_2`, and an exact-construction kernel. Use this for residual domains and visible-region unions/differences. Its operations regularize away lower-dimensional remnants; the geometry contract above deliberately matches that behavior.

**[C3] CGAL arrangements.** See the [2D Arrangements manual](https://doc.cgal.org/6.2/Arrangement_on_surface_2/index.html), particularly `Arrangement_on_surface_2/overlay.cpp`, `face_extension_overlay.cpp`, and segment-traits examples. Use arrangements for the small oracle, labelled output normalization, and verification overlays. Do not use a full projected arrangement as the selective production visibility subroutine.

**[C4] CGAL triangulation.** Adapt [Triangulation_2/polygon_triangulation.cpp](https://doc.cgal.org/6.2/Triangulation_2/Triangulation_2_2polygon_triangulation_8cpp-example.html) for constraints and polygonal domains. Replace the example's inexact-construction kernel, and extend face classification from a single domain to source-labelled adjacent regions. The [manual](https://doc.cgal.org/6.2/Triangulation_2/index.html) documents constraint intersection policies.

**[C5] CGAL source repository.** [CGAL/cgal](https://github.com/CGAL/cgal). Use the package paths above to find the corresponding examples and headers. The links in this document target documentation version 6.2; pin the actual source revision used by the agent and report any API adaptations. Check package license files before redistributing incorporated source.

**[C6] Vectorized visibility source archive.** [MangoSister/vectorizer2-archive](https://github.com/MangoSister/vectorizer2-archive), associated with [R3]. The repository contains `src`, `deps`, `test`, `tool`, and [CMakeLists.txt](https://github.com/MangoSister/vectorizer2-archive/blob/main/CMakeLists.txt). Inspect its visibility representation, splitting/merging, and acceleration design. Treat it as algorithmic reference code; do not depend on building its full rendering stack for the first prototype or assume its rounding policy provides exact constructions.

**[C7] Nvdiffrast depth peeling.** [Official documentation](https://nvlabs.github.io/nvdiffrast/) and [NVlabs/nvdiffrast](https://github.com/NVlabs/nvdiffrast). The documented API is `nvdiffrast.torch.DepthPeeler` with `rasterize_next_layer()`. Use only for optional raster comparisons. Equal-depth hits are not separate peeled layers, and sampled agreement is not an exact geometric certificate.

# 13 Research references

**[R1]** Kevin Weiler and Peter Atherton. *Hidden Surface Removal Using Polygon Area Sorting*. SIGGRAPH, 1977, pp. 214–222. [Paper](https://www.cs.drexel.edu/~david/Classes/CS430/HWs/p214-weiler.pdf), [DOI](https://doi.org/10.1145/965141.563896). Polygon clipping and polygon-output hidden-surface removal are prior art.

**[R2]** Thomas Auzinger, Michael Wimmer, and Stefan Jeschke. *Analytic Visibility on the GPU*. Computer Graphics Forum 32(2), 2013, pp. 409–418. [Author page](https://www.cg.tuwien.ac.at/research/publications/2013/Auzinger_2013_AnaVis/), [paper](https://www.cg.tuwien.ac.at/research/publications/2013/Auzinger_2013_AnaVis/Auzinger_2013_AnaVis-Paper.pdf), [DOI](https://doi.org/10.1111/cgf.12061). See Section 3 for visible boundaries and numerical treatment, and Section 6 for the proposed analytic depth-peeling extension.

**[R3]** Yang Zhou, Lifan Wu, Ravi Ramamoorthi, and Ling-Qi Yan. *Vectorization for Fast, Analytic, and Differentiable Visibility*. ACM Transactions on Graphics 40(3), Article 27, 2021. [Research page](https://research.nvidia.com/labs/rtr/publication/zhou2021vectorization/), [paper](https://cseweb.ucsd.edu/~ravir/lifan_vectorizer.pdf), [DOI](https://doi.org/10.1145/3452097). The relevant parts are incremental visible-region maintenance, the Visibility BVH, and the fixed-point robustness discussion in Section 4.

**[R4]** Pankaj K. Agarwal, Boris Aronov, Timothy M. Chan, and Micha Sharir. *On Levels in Arrangements of Lines, Segments, Planes, and Triangles*. Discrete & Computational Geometry 19, 1998, pp. 315–331. [DOI](https://doi.org/10.1007/PL00009348), [author institution record](https://cris.tau.ac.il/en/publications/on-levels-in-arrangements-of-lines-segments-planes-and-triangles-2/). Use for terminology and the theoretical setting of bounded triangle-patch levels; do not transplant its general complexity bounds into a measured claim about this mesh prototype.

**[R5]** Micha Sharir and Mark H. Overmars. *A Simple Output-Sensitive Algorithm for Hidden Surface Removal*. ACM Transactions on Graphics 11(1), 1992, pp. 1–11. [DOI](https://doi.org/10.1145/102377.112141). Its supplied partial depth-order assumption matters. The prototype must also handle cyclic occlusion and therefore cannot assume that sorting the original triangles produces such an order.
