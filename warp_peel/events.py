"""Experimental contour subdivision and shared-ray rank assignment in Warp.

This is a bounded wavefront prototype, not a complete R2 boundary arrangement.
Supporting-line subdivisions remain; see EVENTS.md for costs and limitations.
"""

from dataclasses import dataclass

import warp as wp

from . import event_kernels as e
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

    def __init__(self, vertices, faces, camera, settings, *, capacities=None, layers=0):
        ec = capacities or EventCapacities.for_mesh(faces.size)
        if min(vars(ec).values()) < 1:
            raise ValueError("Event capacities must be positive")
        self.event_cap = ec
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
        self.cursors = zeros(ec.frontier, wp.vec3i)
        self.next_cursors = zeros(ec.frontier, wp.vec3i)
        self.metrics = zeros(8, wp.int64)
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

    def _launch(self, kernel, dim, *args):
        block = 128 if kernel.module.name == e.__name__ else 32
        wp.launch(
            kernel, dim=dim, inputs=list(args), device=self.device, block_dim=block
        )

    def _prepare(self):
        super()._prepare(classify=False)
        self._launch(
            e.initialize,
            max(self.n, 8),
            self.original,
            self.residual,
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
            e.subdivide,
            self.cap.residual,
            self.residual,
            self.residual_sources,
            self.cursors,
            self.scores,
            self.merged_edges,
            self.near_depths,
            self.edge_bvh.tree,
            self.next,
            self.next_sources,
            self.next_cursors,
            self.output,
            self.output_sources,
            self.output_layers,
            self.status,
            self.metrics,
        )
        self._launch(
            e.copy_frontier,
            self.cap.residual,
            self.next,
            self.next_sources,
            self.next_cursors,
            self.residual,
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

    def _rank(self):
        self._launch(
            e.samples, self.cap.polygons, self.output, self.samples, self.status
        )
        for axis in (1, 0):
            self._launch(
                e.query_keys,
                self.cap.polygons,
                self.samples,
                self.query_keys,
                self.query_ids,
                axis,
            )
            wp.utils.radix_sort_pairs(
                self.query_keys, self.query_ids, count=self.cap.polygons
            )
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
            self.offsets,
            self.hit_depths,
            self.hit_sources,
            self.hit_groups,
            self.status,
            self.metrics,
            0,
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
            self.offsets,
            self.hit_depths,
            self.hit_sources,
            self.hit_groups,
            self.status,
            self.metrics,
            1,
        )
        # Stable source -> depth -> group sorting gives lexicographic
        # (group, depth, source), including deterministic equal-depth ties.
        for stage in range(3):
            self._launch(
                e.hit_keys,
                self.event_cap.hits,
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
            wp.utils.radix_sort_pairs(keys, self.hit_ids, count=self.event_cap.hits)
        self._launch(
            e.assign_ranks,
            self.cap.polygons,
            self.samples,
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

    def warmup(self):
        if self.device.is_capturing:
            raise RuntimeError("Warm-up must run before graph capture")
        wp.load_module(module=g, device=self.device)
        wp.load_module(module=e, device=self.device)
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
                ),
                map(int, m),
            )
        )
        return info
