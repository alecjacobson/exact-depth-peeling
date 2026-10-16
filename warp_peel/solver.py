"""Preallocated, device-resident, CUDA-graph-capturable analytic depth peeling."""

from dataclasses import dataclass

import warp as wp

from . import kernels as k


@dataclass(frozen=True)
class Capacities:
    residual: int
    polygons: int
    vertices: int
    triangles: int
    workers: int = 4096
    pieces_per_worker: int = 512
    layers: int = 128
    edge_vertices: int = 128

    @classmethod
    def for_mesh(cls, triangles: int):
        n = max(triangles, 16)
        return cls(
            4 * n,
            16 * n,
            96 * n,
            64 * n,
            workers=min(4096, n),
            pieces_per_worker=512 if n <= 8192 else 2048,
        )


class DeviceBVH:
    """Morton-ordered balanced tree, with all hierarchy storage preallocated.

    Sorting uses Warp's GPU radix primitive; leaf and parent construction are
    Warp kernels. No native BVH builder or temporary allocation in the loop.
    """

    def __init__(self, lower, upper, layered=False):
        self.lower, self.upper = lower, upper
        self.device = lower.device
        self.layered = int(layered)
        self.size = 1 << (max(1, lower.size) - 1).bit_length()
        self.keys = wp.zeros(2 * self.size, dtype=wp.uint64, device=self.device)
        self.ids = wp.zeros(2 * self.size, dtype=int, device=self.device)
        self.tree = k.Tree()
        self.tree.lower = wp.zeros(2 * self.size, dtype=wp.vec3, device=self.device)
        self.tree.upper = wp.zeros(2 * self.size, dtype=wp.vec3, device=self.device)
        self.tree.ids = self.ids
        self.tree.leaves = self.size

    def rebuild(self):
        wp.launch(
            k.tree_keys,
            self.size,
            inputs=[
                self.lower,
                self.upper,
                self.keys,
                self.ids,
                self.lower.size,
                self.layered,
            ],
            device=self.device,
            block_dim=32,
        )
        wp.utils.radix_sort_pairs(self.keys, self.ids, count=self.size)
        self.refit()

    def refit(self):
        wp.launch(
            k.tree_leaves,
            self.size,
            inputs=[self.tree, self.lower, self.upper, self.lower.size],
            device=self.device,
            block_dim=32,
        )
        start = self.size // 2
        while start:
            wp.launch(
                k.tree_level,
                start,
                inputs=[self.tree, start],
                device=self.device,
                block_dim=32,
            )
            start //= 2


class DevicePeeler:
    """Accept device arrays; all geometry, topology and loop decisions stay on CUDA.

    Vertices are float64 vec3, faces int32 vec3, camera a one-element mat44d,
    settings float64 [xmin, ymin, xmax, ymax, perspective(0/1), near].
    Input surfaces must have self-intersections resolved before upload.
    Outputs use dense source indices into faces; external labels remain caller-owned.
    Allocation/JIT/warm-up precede capture. ``enqueue`` supports external capture.
    Readback is an explicit, optional consumer operation, never used by extraction.
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
        _iterative_storage=True,
    ):
        self.device = vertices.device
        if not self.device.is_cuda:
            raise ValueError(
                "DevicePeeler requires a CUDA device and conditional graph support"
            )
        for a, dtype in [
            (vertices, wp.vec3d),
            (faces, wp.vec3i),
            (camera, wp.mat44d),
            (settings, wp.float64),
        ]:
            if a.device != self.device or a.dtype != dtype or a.ndim != 1:
                raise ValueError(f"Expected a 1-D {dtype} array on {self.device}")
        if camera.size != 1 or settings.size != 6:
            raise ValueError("Expected one camera matrix and six camera settings")
        self.vertices, self.faces, self.camera, self.settings = (
            vertices,
            faces,
            camera,
            settings,
        )
        self.stream = wp.Stream(self.device)
        self.stream.wait_stream(wp.get_stream(self.device))
        self.n = faces.size
        self.nv = vertices.size
        self.cap = capacities or Capacities.for_mesh(self.n)
        if min(vars(self.cap).values()) < 1 or self.cap.edge_vertices > 128:
            raise ValueError(
                "Capacities must be positive; edge_vertices must be <= 128"
            )
        if layers < 0 or layers > self.cap.layers:
            raise ValueError("layers must be zero (all), or fit in the layer capacity")
        self.limit = layers or self.cap.layers
        self.all_layers = int(layers == 0)
        self.graph = None
        self.warmed = False

        def zeros(shape, dtype):
            return wp.zeros(shape, dtype=dtype, device=self.device)

        self.status = zeros(16, int)
        self.condition = zeros(1, int)
        self.stats = zeros((self.cap.layers, 3), wp.int64)
        self.canonical = zeros(max(1, self.nv), int)
        self.vertex_keys = zeros(2 * max(1, self.nv), wp.float64)
        self.vertex_ids = zeros(2 * max(1, self.nv), int)
        self.edge_keys = zeros(6 * max(1, self.n), wp.uint64)
        self.edge_ids = zeros(6 * max(1, self.n), int)
        self.masks = zeros(3 * max(1, self.n), int)
        self.original = zeros(max(1, self.n), k.Polygon)
        self.scores = zeros(max(1, self.n), wp.vec3d)
        self.facing = zeros(max(1, self.n), int)
        self.clipped = zeros(max(1, self.n), int)
        self.ranks = zeros(max(1, self.n), int)
        residual_storage = self.cap.residual if _iterative_storage else 1
        self.residual = zeros(residual_storage, k.Polygon)
        self.residual_sources = zeros(self.cap.residual, int)
        self.next = zeros(residual_storage, k.Polygon)
        self.next_sources = zeros(self.cap.residual, int)
        self.output = zeros(self.cap.polygons, k.Polygon)
        self.output_sources = zeros(self.cap.polygons, int)
        self.output_layers = zeros(self.cap.polygons, int)
        self.visible = zeros((self.cap.workers, self.cap.pieces_per_worker), k.Polygon)
        self.temporary = zeros(
            (self.cap.workers, self.cap.pieces_per_worker), k.Polygon
        )
        self.hidden = zeros((self.cap.workers, self.cap.pieces_per_worker), k.Polygon)
        self.counts = zeros(self.cap.polygons, int)
        self.offsets = zeros(self.cap.polygons, int)
        self.scan_size = 1 << (self.cap.polygons - 1).bit_length()
        self.scan_sums = zeros(2 * self.scan_size, int)
        self.scan_prefixes = zeros(2 * self.scan_size, int)
        self.boundary_vertices = zeros(self.cap.vertices, wp.vec2d)
        self.boundary_layers = zeros(self.cap.vertices, int)
        self.triangles_uv = zeros((self.cap.triangles, 3), wp.vec2d)
        self.triangles_world = zeros((self.cap.triangles, 3), wp.vec3d)
        self.triangle_sources = zeros(self.cap.triangles, int)
        self.triangle_layers = zeros(self.cap.triangles, int)
        self.source_lower, self.source_upper = (
            zeros(max(1, self.n), wp.vec3),
            zeros(max(1, self.n), wp.vec3),
        )
        self.contour_lower, self.contour_upper = (
            zeros(max(1, self.n), wp.vec3),
            zeros(max(1, self.n), wp.vec3),
        )
        self.residual_lower, self.residual_upper = (
            zeros(residual_storage, wp.vec3),
            zeros(residual_storage, wp.vec3),
        )
        self.point_lower, self.point_upper = (
            zeros(self.cap.vertices, wp.vec3),
            zeros(self.cap.vertices, wp.vec3),
        )
        # All hierarchy construction is in Warp kernels and GPU radix primitives.
        self.source_bvh = DeviceBVH(self.source_lower, self.source_upper)
        self.contour_bvh = DeviceBVH(self.contour_lower, self.contour_upper)
        self.residual_bvh = DeviceBVH(self.residual_lower, self.residual_upper)
        self.point_bvh = DeviceBVH(self.point_lower, self.point_upper, layered=True)

    def _launch(self, kernel, dim, *args):
        wp.launch(kernel, dim=dim, inputs=list(args), device=self.device, block_dim=32)

    def _prepare(self, classify=True):
        self._launch(
            k.reset, max(16, self.cap.layers), self.status, self.condition, self.stats
        )
        self._launch(
            k.validate_input,
            max(1, self.nv, self.n),
            self.vertices,
            self.faces,
            self.camera,
            self.settings,
            self.status,
        )
        if self.nv:
            for axis in (2, 1, 0):
                self._launch(
                    k.vertex_keys,
                    self.nv,
                    self.vertices,
                    self.vertex_keys,
                    self.vertex_ids,
                    axis,
                    int(axis == 2),
                )
                wp.utils.radix_sort_pairs(
                    self.vertex_keys, self.vertex_ids, count=self.nv
                )
            self._launch(
                k.weld, self.nv, self.vertices, self.vertex_ids, self.canonical
            )
        if self.n:
            self._launch(
                k.project,
                self.n,
                self.vertices,
                self.faces,
                self.camera,
                self.settings,
                self.original,
                self.scores,
                self.facing,
                self.clipped,
                self.source_lower,
                self.source_upper,
                self.status,
            )
            self._launch(
                k.edge_keys,
                3 * self.n,
                self.faces,
                self.canonical,
                self.edge_keys,
                self.edge_ids,
                self.masks,
                self.status,
            )
            wp.utils.radix_sort_pairs(self.edge_keys, self.edge_ids, count=3 * self.n)
            self._launch(
                k.mark_edges,
                3 * self.n,
                self.edge_keys,
                self.edge_ids,
                self.canonical,
                self.faces,
                self.facing,
                self.masks,
                3 * self.n,
            )
            if classify:
                self._launch(
                    k.contour_bounds,
                    self.n,
                    self.original,
                    self.masks,
                    self.facing,
                    self.clipped,
                    self.contour_lower,
                    self.contour_upper,
                )
        self.source_bvh.rebuild()
        if classify:
            self.contour_bvh.rebuild()
        if self.n and classify:
            self._launch(
                k.classify,
                self.n,
                self.original,
                self.scores,
                self.masks,
                self.facing,
                self.clipped,
                self.source_bvh.tree,
                self.contour_bvh.tree,
                self.ranks,
                self.residual,
                self.residual_sources,
                self.output,
                self.output_sources,
                self.output_layers,
                self.status,
                self.limit,
                self.all_layers,
            )
        if classify:
            self._launch(k.start_loop, 1, self.status, self.condition)

    def _layer(self):
        self._launch(
            k.retire_bounds,
            max(1, self.n),
            self.original,
            self.scores,
            self.ranks,
            self.status,
            self.source_lower,
            self.source_upper,
        )
        self.source_bvh.refit()
        self._launch(
            k.active_bounds,
            self.cap.residual,
            self.residual,
            self.residual_sources,
            self.scores,
            self.status,
            self.residual_lower,
            self.residual_upper,
        )
        self.residual_bvh.rebuild()
        self._launch(
            k.peel,
            self.cap.workers,
            self.residual,
            self.residual_sources,
            self.original,
            self.scores,
            self.ranks,
            self.residual_bvh.tree,
            self.source_bvh.tree,
            self.visible,
            self.temporary,
            self.hidden,
            self.next,
            self.next_sources,
            self.output,
            self.output_sources,
            self.output_layers,
            self.status,
            self.stats,
            self.cap.workers,
        )
        self._launch(
            k.copy_residual,
            self.cap.residual,
            self.next,
            self.next_sources,
            self.residual,
            self.residual_sources,
            self.status,
        )
        self._launch(
            k.advance, 1, self.status, self.condition, self.limit, self.all_layers
        )

    def _mesh(self):
        self._launch(
            k.vertex_counts, self.cap.polygons, self.output, self.status, self.counts
        )
        self._launch(
            k.scan_leaves,
            self.scan_size,
            self.counts,
            self.scan_sums,
            self.scan_prefixes,
            self.scan_size,
        )
        level = self.scan_size // 2
        while level:
            self._launch(k.scan_up, level, self.scan_sums, level)
            level //= 2
        level = 1
        while level < self.scan_size:
            self._launch(k.scan_down, level, self.scan_sums, self.scan_prefixes, level)
            level *= 2
        self._launch(
            k.scan_result,
            self.cap.polygons,
            self.scan_prefixes,
            self.offsets,
            self.scan_size,
        )
        self._launch(
            k.scatter_vertices,
            self.cap.polygons,
            self.output,
            self.output_layers,
            self.offsets,
            self.status,
            self.boundary_vertices,
            self.boundary_layers,
        )
        self._launch(
            k.point_bounds,
            self.cap.vertices,
            self.boundary_vertices,
            self.boundary_layers,
            self.status,
            self.point_lower,
            self.point_upper,
        )
        self.point_bvh.rebuild()
        self._launch(
            k.mesh_edges,
            (self.cap.polygons, 32),
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
        """Compile and prime Warp radix scratch storage before external capture."""
        if self.device.is_capturing:
            raise RuntimeError("Warm-up must run before graph capture")
        wp.load_module(module=k, device=self.device)
        with wp.ScopedStream(self.stream):
            self._prepare()
            self._layer()
            self._mesh()
        wp.synchronize_stream(self.stream)
        self.warmed = True

    def enqueue(self):
        """Record the complete pipeline in the caller's CUDA graph; no host reads."""
        if not self.warmed or not self.device.is_capturing:
            raise RuntimeError("Call warmup(), then enqueue() inside wp.ScopedCapture")
        if wp.get_stream(self.device) != self.stream:
            raise RuntimeError(
                "Capture on the workspace stream (peeler.stream) used for warm-up"
            )
        self._prepare()
        wp.capture_while(self.condition, self._layer)
        self._mesh()

    def capture(self):
        if not self.warmed:
            self.warmup()
        with (
            wp.ScopedStream(self.stream),
            wp.ScopedCapture(stream=self.stream) as capture,
        ):
            self.enqueue()
        self.graph = capture.graph
        return self.graph

    def run(self):
        """Asynchronously replay; results and the error flag remain on device."""
        if self.graph is None:
            self.capture()
        wp.capture_launch(self.graph, stream=self.stream)

    def read_status(self):
        """Explicit host inspection, outside capture and extraction timings."""
        wp.synchronize_stream(self.stream)
        values = self.status.numpy()
        if values[9]:
            labels = {
                1: "polygon vertices",
                2: "per-worker pieces",
                4: "residual pool",
                8: "output polygon pool",
                16: "boundary vertex pool",
                32: "triangle pool",
                64: "layer limit",
                128: "edge noding capacity",
                256: "invalid input",
            }
            raise RuntimeError(
                "GPU extraction failed: "
                + ", ".join(v for b, v in labels.items() if values[9] & b)
            )
        return {
            "residual_patches": int(values[0]),
            "polygons": int(values[2]),
            "triangles": int(values[3]),
            "layers": int(values[4]),
            "peeling_passes": int(values[5]),
            "boundary_vertices": int(values[6]),
            "depth_ties": int(values[8]),
            "error": int(values[9]),
            "max_hidden_pieces": int(values[10]),
            "peak_residual_patches": int(values[12]),
        }
