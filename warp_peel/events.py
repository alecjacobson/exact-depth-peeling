"""Experimental contour subdivision and shared-ray rank assignment in Warp.

This is a bounded wavefront prototype, not a complete R2 boundary arrangement.
Supporting-line subdivisions remain; see EVENTS.md for costs and limitations.
"""

from dataclasses import dataclass

import warp as wp

from . import beams as b
from . import components as c
from . import event_kernels as e
from . import frontier as f
from . import kernels as g
from .solver import Capacities, DeviceBVH, DevicePeeler


@dataclass(frozen=True)
class EventCapacities:
    frontier: int
    regions: int
    hits: int
    vertices: int
    triangles: int
    rounds: int = 4096
    layers: int = 4096
    edge_vertices: int = 128

    @classmethod
    def for_mesh(cls, n):
        n = max(16, n)
        return cls(4 * n, 8 * n, 32 * n, 32 * n, 32 * n, layers=n)


class EventPeeler(DevicePeeler):
    """All-layer contour subdivision, followed by shared sorted ray hits.

    Inputs/outputs and graph lifetime match DevicePeeler. Requires resolved surface
    intersections. No layer-by-layer peeling and no private fragment collections.
    """

    def __init__(
        self,
        vertices,
        faces,
        camera,
        settings,
        *,
        capacities=None,
        layers=0,
        share_regions=True,
        propagate_ranks=True,
        merge_regions=False,
    ):
        ec = capacities or EventCapacities.for_mesh(faces.size)
        if min(vars(ec).values()) < 1:
            raise ValueError("Event capacities must be positive")
        self.event_cap = ec
        self.share_regions = share_regions
        self.propagate_ranks = propagate_ranks
        self.merge_regions = merge_regions
        super().__init__(
            vertices,
            faces,
            camera,
            settings,
            capacities=Capacities(
                ec.frontier,
                ec.regions,
                ec.vertices,
                ec.triangles,
                workers=1,
                pieces_per_worker=1,
                layers=ec.layers,
                edge_vertices=ec.edge_vertices,
            ),
            layers=layers,
            _iterative_storage=False,
        )

        def zeros(n, dtype):
            return wp.zeros(n, dtype=dtype, device=self.device)

        self.edge_count = 8 * max(1, self.n)
        self.edges = zeros((self.edge_count, 2), wp.vec2d)
        self.edge_depths = zeros((self.edge_count, 2), wp.float64)
        self.edge_sort_keys = zeros(2 * self.edge_count, wp.float64)
        self.edge_sort_ids = zeros(2 * self.edge_count, int)
        self.merged_edges = zeros((self.edge_count, 2), wp.vec2d)
        self.near_depths = zeros((self.edge_count, 2), wp.float64)
        self.edge_lower, self.edge_upper = (
            zeros(self.edge_count, wp.vec3),
            zeros(self.edge_count, wp.vec3),
        )
        self.edge_bvh = DeviceBVH(self.edge_lower, self.edge_upper)
        self.frontier_vertices = zeros((ec.frontier, 32), wp.vec2d)
        self.frontier_sizes = zeros(ec.frontier, int)
        self.next_vertices = zeros((ec.frontier, 32), wp.vec2d)
        self.next_sizes = zeros(ec.frontier, int)
        self.cursors = zeros(ec.frontier, wp.vec3i)
        self.next_cursors = zeros(ec.frontier, wp.vec3i)
        self.metrics = zeros(19, wp.int64)
        self.beam_roots = zeros(ec.regions, int)
        self.beam_bounds = zeros((ec.regions, 4), wp.float64)
        self.beam_valid = zeros(ec.regions, int)
        self.heavy_beams = zeros(ec.regions, wp.vec2i)
        self.heavy_queries = zeros(ec.regions, wp.vec2i)
        self.samples = zeros(ec.regions, wp.vec2d)
        self.query_keys = zeros(2 * ec.regions, wp.float64)
        self.query_ids = zeros(2 * ec.regions, int)
        self.groups = zeros(ec.regions, int)
        self.hit_depths = zeros(ec.hits, wp.float64)
        self.hit_sources = zeros(ec.hits, int)
        self.hit_groups = zeros(ec.hits, int)
        self.sort_depths = zeros(2 * ec.hits, wp.float64)
        self.sort_keys = zeros(2 * ec.hits, int)
        self.hit_ids = zeros(2 * ec.hits, int)
        self.sort_condition = zeros(1, int)
        self.mesh_tasks = zeros(ec.vertices, wp.vec2i)
        self.representatives = zeros(ec.regions, int)
        self.connection_condition = zeros(1, int)
        if propagate_ranks or merge_regions:
            self.parents = zeros(ec.regions, int)
            self.merge_choices = zeros(ec.regions, int)
            self.partners = zeros(ec.vertices, int)
            self.patch_owners = zeros(ec.vertices, wp.uint64)
            self.patch_starts = zeros(ec.vertices, wp.vec2d)
            self.patch_ends = zeros(ec.vertices, wp.vec2d)
            self.topology_owners = zeros(3 * max(1, self.n), wp.uint64)
            self.patch_edge_count = zeros(1, int)
            self.edge_slots = zeros(1 << (2 * ec.vertices - 1).bit_length(), int)
        if merge_regions:
            self.compact_output = zeros(ec.regions, g.Polygon)
            self.compact_sources = zeros(ec.regions, int)
            self.compact_layers = zeros(ec.regions, int)

    def _launch(self, kernel, dim, *args):
        block = 128 if kernel.module.name in (e.__name__, c.__name__) else 32
        if kernel in (c.edge_owners, c.propose, c.merge, c.compact, c.restore):
            block = 32
        wp.launch(
            kernel, dim=dim, inputs=list(args), device=self.device, block_dim=block
        )

    def _prepare(self):
        super()._prepare(classify=False)
        self._launch(
            f.seed,
            max(self.n, self.metrics.size),
            self.original,
            self.frontier_vertices,
            self.frontier_sizes,
            self.residual_sources,
            self.cursors,
            self.status,
            self.condition,
            self.metrics,
            self.n,
        )
        self._launch(
            e.make_edges,
            self.edge_count,
            self.original,
            self.scores,
            self.masks,
            self.facing,
            self.clipped,
            self.edges,
            self.edge_depths,
            self.status,
        )
        for axis in (3, 2, 1, 0):
            self._launch(
                e.edge_sort_keys,
                self.edge_count,
                self.edges,
                self.edge_sort_keys,
                self.edge_sort_ids,
                axis,
            )
            wp.utils.radix_sort_pairs(
                self.edge_sort_keys, self.edge_sort_ids, count=self.edge_count
            )
        self._launch(
            e.edge_reset,
            self.edge_count,
            self.near_depths,
            self.edge_lower,
            self.edge_upper,
        )
        self._launch(
            e.reduce_edges,
            self.edge_count,
            self.edges,
            self.edge_depths,
            self.edge_sort_ids,
            self.merged_edges,
            self.near_depths,
            self.metrics,
        )
        self._launch(
            e.edge_bounds,
            self.edge_count,
            self.merged_edges,
            self.near_depths,
            self.edge_lower,
            self.edge_upper,
        )
        self.edge_bvh.rebuild()

    def _step(self):
        self._launch(
            f.subdivide,
            self.cap.residual,
            self.frontier_vertices,
            self.frontier_sizes,
            self.residual_sources,
            self.cursors,
            self.scores,
            self.merged_edges,
            self.near_depths,
            self.edge_bvh.tree,
            self.next_vertices,
            self.next_sizes,
            self.next_sources,
            self.next_cursors,
            self.output,
            self.output_sources,
            self.output_layers,
            self.status,
            self.metrics,
        )
        self._launch(
            f.copy_frontier,
            self.cap.residual,
            self.next_vertices,
            self.next_sizes,
            self.next_sources,
            self.next_cursors,
            self.frontier_vertices,
            self.frontier_sizes,
            self.residual_sources,
            self.cursors,
            self.status,
        )
        self._launch(
            e.advance,
            1,
            self.status,
            self.condition,
            self.metrics,
            self.event_cap.rounds,
        )

    def _scan(self):
        self._launch(
            g.scan_leaves,
            self.scan_size,
            self.counts,
            self.scan_sums,
            self.scan_prefixes,
            self.scan_size,
        )
        level = self.scan_size // 2
        while level:
            self._launch(g.scan_up, level, self.scan_sums, level)
            level //= 2
        level = 1
        while level < self.scan_size:
            self._launch(g.scan_down, level, self.scan_sums, self.scan_prefixes, level)
            level *= 2
        self._launch(
            g.scan_result,
            self.cap.polygons,
            self.scan_prefixes,
            self.offsets,
            self.scan_size,
        )

    def _connections(self):
        if not (self.propagate_ranks or self.merge_regions):
            self._identity_components()
            return
        self._launch(c.connection_condition, 1, self.metrics, self.connection_condition)
        if self.n:
            self._launch(
                c.topology_keys,
                3 * self.n,
                self.edge_keys,
                self.edge_ids,
                self.topology_owners,
                self.masks,
                self.connection_condition,
            )
        if self.device.is_capturing:
            wp.capture_if(
                self.connection_condition,
                self._connect_patches,
                self._identity_components,
            )
        else:
            self._connect_patches()

    def _identity_components(self):
        self._launch(
            c.identity,
            self.cap.polygons,
            self.representatives,
            self.status,
            self.metrics,
            int(self.propagate_ranks),
        )

    def _connect_patches(self):
        self._launch(
            c.reset,
            max(self.edge_slots.size, self.cap.polygons, self.cap.vertices),
            self.edge_slots,
            self.parents,
            self.partners,
            self.merge_choices,
        )
        self._launch(
            g.vertex_counts, self.cap.polygons, self.output, self.status, self.counts
        )
        self._scan()
        self._launch(
            c.check_edges,
            1,
            self.scan_sums,
            self.patch_edge_count,
            self.status,
            self.cap.vertices,
        )
        self._launch(
            e.mesh_tasks,
            self.cap.polygons,
            self.counts,
            self.offsets,
            self.mesh_tasks,
            self.status,
        )
        self._launch(
            c.edge_owners,
            self.cap.polygons,
            self.offsets,
            self.output,
            self.output_sources,
            self.original,
            self.masks,
            self.facing,
            self.clipped,
            self.topology_owners,
            self.patch_owners,
            self.patch_starts,
            self.patch_ends,
            self.status,
        )
        self._launch(
            c.match_edges,
            self.cap.vertices,
            self.mesh_tasks,
            self.patch_starts,
            self.patch_ends,
            self.patch_owners,
            self.edge_slots,
            self.partners,
            self.patch_edge_count,
            self.status,
            self.metrics,
        )
        if self.propagate_ranks:
            self._launch(
                c.connect,
                self.cap.vertices,
                self.mesh_tasks,
                self.patch_starts,
                self.patch_ends,
                self.output_sources,
                self.scores,
                self.partners,
                self.parents,
                self.merged_edges,
                self.near_depths,
                self.edge_bvh.tree,
                self.patch_edge_count,
                self.status,
                self.metrics,
            )
            self._launch(
                c.roots,
                self.cap.polygons,
                self.parents,
                self.representatives,
                self.status,
                self.metrics,
            )
        else:
            self._identity_components()

    def _coalesce(self):
        self._launch(
            c.propose,
            self.cap.vertices,
            self.mesh_tasks,
            self.output,
            self.output_sources,
            self.output_layers,
            self.partners,
            self.merge_choices,
            self.patch_edge_count,
            self.status,
        )
        self._launch(
            c.merge,
            self.cap.vertices,
            self.mesh_tasks,
            self.output,
            self.partners,
            self.merge_choices,
            self.patch_edge_count,
            self.status,
            self.metrics,
        )
        self._launch(
            c.live_counts, self.cap.polygons, self.output, self.counts, self.status
        )
        self._scan()
        self._launch(
            c.compact,
            self.cap.polygons,
            self.output,
            self.output_sources,
            self.output_layers,
            self.offsets,
            self.compact_output,
            self.compact_sources,
            self.compact_layers,
            self.status,
        )
        self._launch(
            c.restore,
            self.cap.polygons,
            self.output,
            self.output_sources,
            self.output_layers,
            self.compact_output,
            self.compact_sources,
            self.compact_layers,
            self.scan_sums,
            self.status,
        )

    def _rank(self):
        self._connections()
        self._launch(
            e.samples,
            self.cap.polygons,
            self.output,
            self.representatives,
            self.samples,
            self.status,
        )
        if self.device.is_capturing:
            self._query_sort_bucket(0, len(self._query_buckets()) - 1)
        else:
            self._sort_queries(self.cap.polygons)
        if self.share_regions:
            self._launch(b.reset, self.cap.polygons, self.beam_bounds, self.beam_valid)
            self._launch(
                b.group,
                self.cap.polygons,
                self.samples,
                self.query_ids,
                self.groups,
                self.beam_roots,
                self.counts,
                self.beam_bounds,
                self.status,
            )
            self._launch(
                b.certify,
                self.cap.polygons,
                self.query_ids,
                self.beam_roots,
                self.beam_bounds,
                self.beam_valid,
                self.original,
                self.source_bvh.tree,
                self.heavy_beams,
                self.status,
                self.metrics,
            )
            self._launch(
                b.certify_heavy,
                8192,
                self.beam_bounds,
                self.beam_valid,
                self.original,
                self.source_bvh.tree,
                self.heavy_beams,
                self.status,
                self.metrics,
                8192,
            )
            self._launch(
                b.apply,
                self.cap.polygons,
                self.query_ids,
                self.beam_roots,
                self.groups,
                self.beam_valid,
                self.status,
                self.metrics,
            )
        else:
            self._launch(
                e.group_queries,
                self.cap.polygons,
                self.samples,
                self.query_ids,
                self.groups,
                self.counts,
                self.status,
            )
        self._launch(
            e.ray_hits,
            self.cap.polygons,
            self.samples,
            self.query_ids,
            self.groups,
            self.original,
            self.scores,
            self.source_bvh.tree,
            self.counts,
            self.heavy_queries,
            self.hit_depths,
            self.hit_sources,
            self.hit_groups,
            self.status,
            self.metrics,
        )
        self._launch(
            e.heavy_ray_hits,
            8192,
            self.samples,
            self.query_ids,
            self.original,
            self.scores,
            self.source_bvh.tree,
            self.counts,
            self.heavy_queries,
            self.hit_depths,
            self.hit_sources,
            self.hit_groups,
            self.status,
            self.metrics,
            8192,
        )
        self._scan()
        self._launch(
            e.check_hits,
            1,
            self.scan_sums,
            self.status,
            self.metrics,
            self.event_cap.hits,
        )
        if self.device.is_capturing:
            self._sort_bucket(0, len(self._hit_buckets()) - 1)
        else:
            # Prime the largest native scratch allocation before capture.
            self._sort_hits(self.event_cap.hits)
        self._launch(
            e.assign_ranks,
            self.cap.polygons,
            self.samples,
            self.representatives,
            self.groups,
            self.counts,
            self.offsets,
            self.hit_depths,
            self.hit_sources,
            self.hit_ids,
            self.scores,
            self.output,
            self.output_sources,
            self.output_layers,
            self.status,
            self.limit,
            self.all_layers,
        )

        if self.merge_regions:
            if self.device.is_capturing:
                wp.capture_if(self.connection_condition, self._coalesce)
            else:
                self._coalesce()

    def _query_buckets(self):
        sizes = []
        size = 256
        while size < self.cap.polygons:
            sizes.append(size)
            size *= 2
        return [*sizes, self.cap.polygons]

    def _query_sort_bucket(self, low, high):
        sizes = self._query_buckets()
        if low == high:
            self._sort_queries(sizes[low])
            return
        mid = (low + high) // 2
        self._launch(e.query_bucket, 1, self.status, self.sort_condition, sizes[mid])
        wp.capture_if(
            self.sort_condition,
            lambda: self._query_sort_bucket(low, mid),
            lambda: self._query_sort_bucket(mid + 1, high),
        )

    def _sort_queries(self, size):
        for axis in (1, 0):
            self._launch(
                e.query_keys,
                size,
                self.samples,
                self.query_keys,
                self.query_ids,
                axis,
            )
            wp.utils.radix_sort_pairs(self.query_keys, self.query_ids, count=size)
        if self.share_regions:
            self._launch(
                b.keys,
                size,
                self.samples,
                self.query_keys,
                self.query_ids,
                self.status,
            )
            wp.utils.radix_sort_pairs(self.query_keys, self.query_ids, count=size)

    def _hit_buckets(self):
        sizes = []
        size = 256
        while size < self.event_cap.hits:
            sizes.append(size)
            size *= 2
        return [*sizes, self.event_cap.hits]

    def _sort_bucket(self, low, high):
        sizes = self._hit_buckets()
        if low == high:
            self._sort_hits(sizes[low])
            return
        mid = (low + high) // 2
        self._launch(e.hit_bucket, 1, self.metrics, self.sort_condition, sizes[mid])
        wp.capture_if(
            self.sort_condition,
            lambda: self._sort_bucket(low, mid),
            lambda: self._sort_bucket(mid + 1, high),
        )

    def _sort_hits(self, size):
        # Stable source -> depth -> group sorting gives lexicographic
        # (group, depth, source), including deterministic equal-depth ties.
        for stage in range(3):
            self._launch(
                e.hit_keys,
                size,
                self.hit_depths,
                self.hit_sources,
                self.hit_groups,
                self.sort_depths,
                self.sort_keys,
                self.hit_ids,
                self.metrics,
                stage,
            )
            keys = self.sort_depths if stage == 1 else self.sort_keys
            wp.utils.radix_sort_pairs(keys, self.hit_ids, count=size)

    def _mesh(self):
        self._launch(
            g.vertex_counts, self.cap.polygons, self.output, self.status, self.counts
        )
        self._scan()
        self._launch(
            g.scatter_vertices,
            self.cap.polygons,
            self.output,
            self.output_layers,
            self.offsets,
            self.status,
            self.boundary_vertices,
            self.boundary_layers,
        )
        self._launch(
            e.mesh_tasks,
            self.cap.polygons,
            self.counts,
            self.offsets,
            self.mesh_tasks,
            self.status,
        )
        self._launch(
            g.point_bounds,
            self.cap.vertices,
            self.boundary_vertices,
            self.boundary_layers,
            self.status,
            self.point_lower,
            self.point_upper,
        )
        self.point_bvh.rebuild()
        self._launch(
            g.mesh_edge_tasks,
            self.cap.vertices,
            self.mesh_tasks,
            self.output,
            self.output_sources,
            self.output_layers,
            self.scores,
            self.boundary_vertices,
            self.point_bvh.tree,
            self.camera,
            self.settings,
            self.status,
            self.triangles_uv,
            self.triangles_world,
            self.triangle_sources,
            self.triangle_layers,
            self.cap.edge_vertices,
        )

    def warmup(self):
        if self.device.is_capturing:
            raise RuntimeError("Warm-up must run before graph capture")
        wp.load_module(module=g, device=self.device)
        wp.load_module(module=e, device=self.device)
        wp.load_module(module=f, device=self.device)
        wp.load_module(module=b, device=self.device)
        wp.load_module(module=c, device=self.device)
        with wp.ScopedStream(self.stream):
            self._prepare()
            self._step()
            self._rank()
            self._mesh()
        wp.synchronize_stream(self.stream)
        self.warmed = True

    def enqueue(self):
        if not self.warmed or not self.device.is_capturing:
            raise RuntimeError("Call warmup(), then enqueue() inside ScopedCapture")
        if wp.get_stream(self.device) != self.stream:
            raise RuntimeError("Capture on the workspace stream")
        self._prepare()
        wp.capture_while(self.condition, self._step)
        self._rank()
        self._mesh()

    def read_status(self):
        wp.synchronize_stream(self.stream)
        status = self.status.numpy()
        if status[9] & 1024:
            raise RuntimeError("GPU extraction failed: shared ray hit capacity")
        if status[9] & 2048:
            raise RuntimeError("GPU extraction failed: subdivision round capacity")
        info = super().read_status()
        m = self.metrics.numpy()
        for name in ("peeling_passes", "max_hidden_pieces"):
            info.pop(name)
        info.update(
            zip(
                (
                    "target_visits",
                    "contour_tests",
                    "splits",
                    "ray_groups",
                    "ray_candidate_tests",
                    "ray_hits",
                    "rounds",
                    "unique_contours",
                    "heavy_ray_groups",
                    "beam_attempts",
                    "certified_beams",
                    "beam_candidate_tests",
                    "heavy_beams",
                    "rank_components",
                    "rank_connections",
                    "connection_contour_tests",
                    "merged_polygons",
                    "matched_patch_edges",
                    "edge_hash_misses",
                ),
                map(int, m),
            )
        )
        return info
