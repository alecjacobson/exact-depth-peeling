"""Flat convex frontier storage for contour subdivision.

The predicates and cleanup policy match kernels.py; vertices live in device
arrays, avoiding dynamically indexed, by-value 32-vertex matrices in clipping.
"""

# ruff: noqa: UP018, RUF046
import warp as wp

from . import kernels as g

wp.set_module_options({"enable_backward": False, "block_dim": 32})


@wp.kernel
def seed(
    original: wp.array(dtype=g.Polygon),
    vertices: wp.array2d(dtype=wp.vec2d),
    sizes: wp.array(dtype=int),
    sources: wp.array(dtype=int),
    cursors: wp.array(dtype=wp.vec3i),
    status: wp.array(dtype=int),
    condition: wp.array(dtype=int),
    metrics: wp.array(dtype=wp.int64),
    n: int,
):
    i = wp.tid()
    if i < metrics.shape[0]:
        metrics[i] = wp.int64(0)
    if i < n and original[i].n > 0 and status[9] == 0:
        slot = wp.atomic_add(status, 0, 1)
        if slot < sizes.shape[0]:
            p = original[i]
            sizes[slot] = p.n
            for j in range(p.n):
                vertices[slot, j] = g.point(p, j)
            sources[slot] = i
            cursors[slot] = wp.vec3i(1, -1, 0)
        else:
            wp.atomic_or(status, 9, 4)
    if i == 0:
        condition[0] = int(n > 0)


@wp.func
def clean(vertices: wp.array2d(dtype=wp.vec2d), slot: int, n: int):
    if n > g.MAX_VERTICES:
        return n
    count = int(0)
    for j in range(n):
        x = vertices[slot, j]
        if count == 0 or wp.length(x - vertices[slot, count - 1]) > (
            wp.float64(g.EPS) * wp.float64(0.01)
        ):
            vertices[slot, count] = x
            count += 1
    if count > 1 and wp.length(vertices[slot, 0] - vertices[slot, count - 1]) <= (
        wp.float64(g.EPS) * wp.float64(0.01)
    ):
        count -= 1
    if count < 3:
        return 0
    origin = vertices[slot, 0]
    area2, perimeter = wp.float64(0.0), wp.float64(0.0)
    for j in range(count):
        a, b = vertices[slot, j], vertices[slot, (j + 1) % count]
        area2 += g.cross(origin, a, b)
        perimeter += wp.abs(b[0] - a[0]) + wp.abs(b[1] - a[1])
    if area2 <= wp.float64(g.EPS) * perimeter:
        return 0
    return count


@wp.func
def append(vertices: wp.array2d(dtype=wp.vec2d), slot: int, n: int, x: wp.vec2d):
    if n < g.MAX_VERTICES:
        vertices[slot, n] = x
    return n + 1


@wp.func
def front_segment(
    vertices: wp.array2d(dtype=wp.vec2d),
    slot: int,
    n: int,
    a: wp.vec2d,
    b: wp.vec2d,
    za: wp.float64,
    zb: wp.float64,
):
    low, high = wp.float64(0.0), wp.float64(1.0)
    for j in range(n + 1):
        da, db = za, zb
        if j < n:
            u, v = vertices[slot, j], vertices[slot, (j + 1) % n]
            da, db = g.cross(u, v, a), g.cross(u, v, b)
        if da <= wp.float64(0.0) and db <= wp.float64(0.0):
            return False
        if da < wp.float64(0.0):
            low = wp.max(low, da / (da - db))
        if db < wp.float64(0.0):
            high = wp.min(high, da / (da - db))
    return high > low


@wp.kernel
def subdivide(
    vertices: wp.array2d(dtype=wp.vec2d),
    sizes: wp.array(dtype=int),
    sources: wp.array(dtype=int),
    cursors: wp.array(dtype=wp.vec3i),
    scores: wp.array(dtype=wp.vec3d),
    edges: wp.array2d(dtype=wp.vec2d),
    edge_depths: wp.array2d(dtype=wp.float64),
    tree: g.Tree,
    next_vertices: wp.array2d(dtype=wp.vec2d),
    next_sizes: wp.array(dtype=int),
    next_sources: wp.array(dtype=int),
    next_cursors: wp.array(dtype=wp.vec3i),
    output: wp.array(dtype=g.Polygon),
    output_sources: wp.array(dtype=int),
    output_layers: wp.array(dtype=int),
    status: wp.array(dtype=int),
    metrics: wp.array(dtype=wp.int64),
):
    i = wp.tid()
    if i >= status[0] or status[9] != 0:
        return
    n = sizes[i]
    if n == 0:
        return
    wp.atomic_add(metrics, 0, wp.int64(1))
    source = sources[i]
    node = cursors[i][0]
    lower, upper = wp.vec3d(1.0e30), wp.vec3d(-1.0e30)
    for j in range(n):
        x = vertices[i, j]
        v3 = wp.vec3d(x[0], x[1], g.value(scores[source], x))
        lower, upper = wp.min(lower, v3), wp.max(upper, v3)
    pad = wp.vec3d(
        2.0e-7,
        2.0e-7,
        wp.float64(2.0e-7)
        * wp.max(wp.float64(1.0), wp.max(wp.abs(lower[2]), wp.abs(upper[2]))),
    )
    lo, hi = wp.vec3(lower - pad), wp.vec3(upper + pad)
    lo[2] = -1.0e30
    done = bool(False)
    for task in range(32):
        node, candidate = g.query_next(tree, lo, hi, node)
        if candidate < 0:
            done = True
            break
        wp.atomic_add(metrics, 1, wp.int64(1))
        a, b = edges[candidate, 0], edges[candidate, 1]
        positive, negative = bool(False), bool(False)
        for j in range(n):
            side = g.cross(a, b, vertices[i, j])
            positive = positive or side > wp.float64(0.0)
            negative = negative or side < wp.float64(0.0)
        if positive and negative:
            za = g.value(scores[source], a) - edge_depths[candidate, 0]
            zb = g.value(scores[source], b) - edge_depths[candidate, 1]
            if front_segment(vertices, i, n, a, b, za, zb):
                slot = wp.atomic_add(status, 1, 2)
                if slot + 1 >= next_sizes.shape[0]:
                    wp.atomic_or(status, 9, 4)
                    return
                left, right = int(0), int(0)
                d = b - a
                score = wp.vec3d(-d[1], d[0], wp.float64(0.0))
                for j in range(n):
                    u, v = vertices[i, j], vertices[i, (j + 1) % n]
                    du, dv = g.value(score, u - a), g.value(score, v - a)
                    if du >= wp.float64(0.0):
                        left = append(next_vertices, slot, left, u)
                    if du <= wp.float64(0.0):
                        right = append(next_vertices, slot + 1, right, u)
                    if (du > wp.float64(0.0) and dv < wp.float64(0.0)) or (
                        du < wp.float64(0.0) and dv > wp.float64(0.0)
                    ):
                        x = u + (du / (du - dv)) * (v - u)
                        left = append(next_vertices, slot, left, x)
                        right = append(next_vertices, slot + 1, right, x)
                left, right = (
                    clean(next_vertices, slot, left),
                    clean(next_vertices, slot + 1, right),
                )
                if left > g.MAX_VERTICES or right > g.MAX_VERTICES:
                    wp.atomic_or(status, 9, 1)
                    return
                if left > 0 and right > 0:
                    wp.atomic_add(metrics, 2, wp.int64(1))
                else:
                    # Match the original splitter: a collapsed child cancels
                    # this cut. Continue the parent on the next frontier round.
                    left, right = n, int(0)
                    for j in range(n):
                        next_vertices[slot, j] = vertices[i, j]
                next_sizes[slot] = left
                next_sizes[slot + 1] = right
                for j in range(2):
                    next_sources[slot + j] = source
                    next_cursors[slot + j] = wp.vec3i(node, -1, 0)
                return
    if done:
        p = g.Polygon()
        p.n = n
        for j in range(n):
            x = vertices[i, j]
            p.v[j, 0] = x[0]
            p.v[j, 1] = x[1]
        g.save_polygon(p, source, 0, output, output_sources, output_layers, status)
    else:
        slot = wp.atomic_add(status, 1, 1)
        if slot < next_sizes.shape[0]:
            next_sizes[slot] = n
            next_sources[slot] = source
            next_cursors[slot] = wp.vec3i(node, -1, 0)
            for j in range(n):
                next_vertices[slot, j] = vertices[i, j]
        else:
            wp.atomic_or(status, 9, 4)


@wp.kernel
def copy_frontier(
    next_vertices: wp.array2d(dtype=wp.vec2d),
    next_sizes: wp.array(dtype=int),
    next_sources: wp.array(dtype=int),
    next_cursors: wp.array(dtype=wp.vec3i),
    vertices: wp.array2d(dtype=wp.vec2d),
    sizes: wp.array(dtype=int),
    sources: wp.array(dtype=int),
    cursors: wp.array(dtype=wp.vec3i),
    status: wp.array(dtype=int),
):
    i = wp.tid()
    if i < status[1] and status[9] == 0:
        n = next_sizes[i]
        sizes[i] = n
        sources[i] = next_sources[i]
        cursors[i] = next_cursors[i]
        for j in range(n):
            vertices[i, j] = next_vertices[i, j]
