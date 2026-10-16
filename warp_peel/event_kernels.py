"""Bounded contour tasks and shared ray ordering; all data remain on CUDA."""

# Warp dynamic loop variables need explicit constructor casts.
# ruff: noqa: UP018, RUF046
import warp as wp

from . import kernels as g

wp.set_module_options({"enable_backward": False, "block_dim": 128})


@wp.kernel
def initialize(
    original: wp.array(dtype=g.Polygon),
    residual: wp.array(dtype=g.Polygon),
    sources: wp.array(dtype=int),
    cursors: wp.array(dtype=wp.vec3i),
    status: wp.array(dtype=int),
    condition: wp.array(dtype=int),
    metrics: wp.array(dtype=wp.int64),
    n: int,
):
    i = wp.tid()
    if i < 8:
        metrics[i] = wp.int64(0)
    if i < n and original[i].n > 0 and status[9] == 0:
        slot = wp.atomic_add(status, 0, 1)
        if slot < residual.shape[0]:
            residual[slot] = original[i]
            sources[slot] = i
            cursors[slot] = wp.vec3i(1, -1, 0)
        else:
            wp.atomic_or(status, 9, 4)
    if i == 0:
        condition[0] = int(n > 0)


@wp.func
def front_segment(
    p: g.Polygon, a: wp.vec2d, b: wp.vec2d, za: wp.float64, zb: wp.float64
):
    low = wp.float64(0.0)
    high = wp.float64(1.0)
    # Clip the finite contour to this source patch, then to its front halfspace.
    for j in range(p.n + 1):
        da = wp.float64(0.0)
        db = wp.float64(0.0)
        if j < p.n:
            u = g.point(p, j)
            v = g.point(p, (j + 1) % p.n)
            da = g.cross(u, v, a)
            db = g.cross(u, v, b)
        else:
            da = za
            db = zb
        if da <= wp.float64(0.0) and db <= wp.float64(0.0):
            return False
        if da < wp.float64(0.0):
            low = wp.max(low, da / (da - db))
        if db < wp.float64(0.0):
            high = wp.min(high, da / (da - db))
    return high > low


@wp.func
def put(
    p: g.Polygon,
    source: int,
    cursor: wp.vec3i,
    out: wp.array(dtype=g.Polygon),
    ids: wp.array(dtype=int),
    cursors: wp.array(dtype=wp.vec3i),
    status: wp.array(dtype=int),
):
    slot = wp.atomic_add(status, 1, 1)
    if slot < out.shape[0]:
        out[slot] = p
        ids[slot] = source
        cursors[slot] = cursor
    else:
        wp.atomic_or(status, 9, 4)


@wp.kernel
def subdivide(
    residual: wp.array(dtype=g.Polygon),
    sources: wp.array(dtype=int),
    cursors: wp.array(dtype=wp.vec3i),
    scores: wp.array(dtype=wp.vec3d),
    edges: wp.array2d(dtype=wp.vec2d),
    edge_depths: wp.array2d(dtype=wp.float64),
    tree: g.Tree,
    next_polys: wp.array(dtype=g.Polygon),
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
    wp.atomic_add(metrics, 0, wp.int64(1))
    p = residual[i]
    source = sources[i]
    node = cursors[i][0]
    lo, hi = g.bounds(p, scores[source])
    lo[2] = -1.0e30
    done = bool(False)
    for task in range(32):
        candidate = int(-1)
        node, candidate = g.query_next(tree, lo, hi, node)
        if candidate < 0:
            done = True
            break
        wp.atomic_add(metrics, 1, wp.int64(1))
        a, b = edges[candidate, 0], edges[candidate, 1]
        positive = bool(False)
        negative = bool(False)
        for j in range(p.n):
            side = g.cross(a, b, g.point(p, j))
            positive = positive or side > wp.float64(0.0)
            negative = negative or side < wp.float64(0.0)
        if positive and negative:
            za = g.value(scores[source], a) - edge_depths[candidate, 0]
            zb = g.value(scores[source], b) - edge_depths[candidate, 1]
            if front_segment(p, a, b, za, zb):
                left, right = g.split_edge(p, a, b)
                if left.n > g.MAX_VERTICES or right.n > g.MAX_VERTICES:
                    wp.atomic_or(status, 9, 1)
                    return
                if left.n > 0 and right.n > 0:
                    wp.atomic_add(metrics, 2, wp.int64(1))
                    cursor = wp.vec3i(node, -1, 0)
                    put(
                        left,
                        source,
                        cursor,
                        next_polys,
                        next_sources,
                        next_cursors,
                        status,
                    )
                    put(
                        right,
                        source,
                        cursor,
                        next_polys,
                        next_sources,
                        next_cursors,
                        status,
                    )
                    return
    if done:
        g.save_polygon(p, source, 0, output, output_sources, output_layers, status)
    else:
        put(
            p,
            source,
            wp.vec3i(node, -1, 0),
            next_polys,
            next_sources,
            next_cursors,
            status,
        )


@wp.kernel
def copy_frontier(
    next_polys: wp.array(dtype=g.Polygon),
    next_sources: wp.array(dtype=int),
    next_cursors: wp.array(dtype=wp.vec3i),
    polygons: wp.array(dtype=g.Polygon),
    sources: wp.array(dtype=int),
    cursors: wp.array(dtype=wp.vec3i),
    status: wp.array(dtype=int),
):
    i = wp.tid()
    if i < status[1] and status[9] == 0:
        polygons[i] = next_polys[i]
        sources[i] = next_sources[i]
        cursors[i] = next_cursors[i]


@wp.kernel
def advance(
    status: wp.array(dtype=int),
    condition: wp.array(dtype=int),
    metrics: wp.array(dtype=wp.int64),
    limit: int,
):
    status[0] = status[1]
    status[1] = 0
    metrics[6] += wp.int64(1)
    if metrics[6] >= wp.int64(limit) and status[0] > 0:
        wp.atomic_or(status, 9, 2048)
    status[12] = wp.max(status[12], status[0])
    condition[0] = int(status[0] > 0 and status[9] == 0)


@wp.kernel
def samples(
    polygons: wp.array(dtype=g.Polygon),
    points: wp.array(dtype=wp.vec2d),
    status: wp.array(dtype=int),
):
    i = wp.tid()
    x = wp.vec2d(1.0e30)
    if i < status[2] and status[9] == 0:
        p = polygons[i]
        anchor = int(0)
        for j in range(1, p.n):
            a, b = g.point(p, j), g.point(p, anchor)
            if a[0] < b[0] or (a[0] == b[0] and a[1] < b[1]):
                anchor = j
        # Canonical cyclic sum shares samples for rotated identical footprints.
        x = wp.vec2d()
        for j in range(p.n):
            x += g.point(p, (anchor + j) % p.n) / wp.float64(p.n)
    points[i] = x


@wp.kernel
def query_keys(
    points: wp.array(dtype=wp.vec2d),
    keys: wp.array(dtype=wp.float64),
    ids: wp.array(dtype=int),
    axis: int,
):
    i = wp.tid()
    index = i
    if axis == 0:
        index = ids[i]
    keys[i] = points[index][axis]
    ids[i] = index


@wp.func
def less(a: wp.vec2d, b: wp.vec2d):
    return a[0] < b[0] or (a[0] == b[0] and a[1] < b[1])


@wp.kernel
def group_queries(
    points: wp.array(dtype=wp.vec2d),
    ids: wp.array(dtype=int),
    groups: wp.array(dtype=int),
    counts: wp.array(dtype=int),
    status: wp.array(dtype=int),
):
    i = wp.tid()
    counts[i] = 0
    if i < status[2] and status[9] == 0:
        index = ids[i]
        x = points[index]
        low = int(0)
        high = i + 1
        while low < high:
            mid = (low + high) // 2
            if less(points[ids[mid]], x):
                low = mid + 1
            else:
                high = mid
        groups[index] = low


@wp.func
def contains(p: g.Polygon, x: wp.vec2d):
    inside = bool(True)
    for j in range(p.n):
        a = g.point(p, j)
        b = g.point(p, (j + 1) % p.n)
        s = g.cross(a, b, x)
        # Half-open ownership counts a tessellated sheet once on a shared edge.
        owned = b[1] > a[1] or (b[1] == a[1] and b[0] < a[0])
        if s < wp.float64(0.0) or (s == wp.float64(0.0) and not owned):
            inside = False
            break
    return inside and p.n > 0


@wp.kernel
def ray_hits(
    points: wp.array(dtype=wp.vec2d),
    query_ids: wp.array(dtype=int),
    groups: wp.array(dtype=int),
    original: wp.array(dtype=g.Polygon),
    scores: wp.array(dtype=wp.vec3d),
    tree: g.Tree,
    counts: wp.array(dtype=int),
    offsets: wp.array(dtype=int),
    depths: wp.array(dtype=wp.float64),
    sources: wp.array(dtype=int),
    hit_groups: wp.array(dtype=int),
    status: wp.array(dtype=int),
    metrics: wp.array(dtype=wp.int64),
    write: int,
):
    i = wp.tid()
    if i >= status[2] or status[9] != 0:
        return
    query = query_ids[i]
    if groups[query] != i:
        return
    x = points[query]
    lo = wp.vec3(float(x[0]) - 2.0e-7, float(x[1]) - 2.0e-7, -1.0e30)
    hi = wp.vec3(float(x[0]) + 2.0e-7, float(x[1]) + 2.0e-7, 1.0e30)
    node = int(1)
    source = int(-1)
    count = int(0)
    visits = int(0)
    while node > 0:
        node, source = g.query_next(tree, lo, hi, node)
        if source < 0:
            break
        visits += 1
        if contains(original[source], x):
            if write != 0:
                slot = offsets[i] + count
                if slot < depths.shape[0]:
                    depths[slot] = g.value(scores[source], x)
                    sources[slot] = source
                    hit_groups[slot] = i
                else:
                    wp.atomic_or(status, 9, 1024)
            count += 1
    if write == 0:
        counts[i] = count
        wp.atomic_add(metrics, 3, wp.int64(1))
    wp.atomic_add(metrics, 4, wp.int64(visits))


@wp.kernel
def check_hits(
    sums: wp.array(dtype=int),
    status: wp.array(dtype=int),
    metrics: wp.array(dtype=wp.int64),
    capacity: int,
):
    metrics[5] = wp.int64(sums[1])
    if sums[1] > capacity:
        wp.atomic_or(status, 9, 1024)


@wp.kernel
def hit_keys(
    depths: wp.array(dtype=wp.float64),
    sources: wp.array(dtype=int),
    groups: wp.array(dtype=int),
    float_keys: wp.array(dtype=wp.float64),
    int_keys: wp.array(dtype=int),
    ids: wp.array(dtype=int),
    metrics: wp.array(dtype=wp.int64),
    stage: int,
):
    i = wp.tid()
    index = i
    if stage != 0:
        index = ids[i]
    key = int(2147483647)
    depth = wp.float64(1.0e30)
    if wp.int64(index) < metrics[5] and index < depths.shape[0]:
        if stage == 0:
            key = sources[index]
        elif stage == 1:
            depth = depths[index]
        else:
            key = groups[index]
    ids[i] = index
    int_keys[i] = key
    float_keys[i] = depth


@wp.kernel
def assign_ranks(
    points: wp.array(dtype=wp.vec2d),
    groups: wp.array(dtype=int),
    counts: wp.array(dtype=int),
    offsets: wp.array(dtype=int),
    depths: wp.array(dtype=wp.float64),
    hit_sources: wp.array(dtype=int),
    hit_ids: wp.array(dtype=int),
    scores: wp.array(dtype=wp.vec3d),
    polygons: wp.array(dtype=g.Polygon),
    sources: wp.array(dtype=int),
    layers: wp.array(dtype=int),
    status: wp.array(dtype=int),
    limit: int,
    all_layers: int,
):
    i = wp.tid()
    if i >= status[2] or status[9] != 0:
        return
    source = sources[i]
    z = g.value(scores[source], points[i])
    group = groups[i]
    low = int(0)
    high = counts[group]
    while low < high:
        mid = (low + high) // 2
        h = hit_ids[offsets[group] + mid]
        if depths[h] < z or (depths[h] == z and hit_sources[h] < source):
            low = mid + 1
        else:
            high = mid
    layers[i] = low
    if low >= limit:
        if all_layers != 0:
            wp.atomic_or(status, 9, 64)
        else:
            p = polygons[i]
            p.n = 0
            polygons[i] = p
    else:
        wp.atomic_max(status, 4, low + 1)


@wp.kernel
def make_edges(
    original: wp.array(dtype=g.Polygon),
    scores: wp.array(dtype=wp.vec3d),
    masks: wp.array(dtype=int),
    facing: wp.array(dtype=int),
    clipped: wp.array(dtype=int),
    edges: wp.array2d(dtype=wp.vec2d),
    depths: wp.array2d(dtype=wp.float64),
    status: wp.array(dtype=int),
):
    i = wp.tid()
    source, edge = i // 8, i % 8
    a, b = wp.vec2d(1.0e30), wp.vec2d(1.0e30)
    za, zb = wp.float64(1.0e30), wp.float64(1.0e30)
    if source < original.shape[0] and status[9] == 0:
        p = original[source]
        if p.n > 8:
            wp.atomic_or(status, 9, 1)
        if edge < p.n and g.contour_edge(source, edge, masks, facing, clipped):
            a, b = g.point(p, edge), g.point(p, (edge + 1) % p.n)
            if less(b, a):
                temp = a
                a = b
                b = temp
            za, zb = g.value(scores[source], a), g.value(scores[source], b)
    edges[i, 0] = a
    edges[i, 1] = b
    depths[i, 0] = za
    depths[i, 1] = zb


@wp.kernel
def edge_sort_keys(
    edges: wp.array2d(dtype=wp.vec2d),
    keys: wp.array(dtype=wp.float64),
    ids: wp.array(dtype=int),
    axis: int,
):
    i = wp.tid()
    index = i
    if axis != 3:
        index = ids[i]
    keys[i] = edges[index, axis // 2][axis % 2]
    ids[i] = index


@wp.func
def edge_less(edges: wp.array2d(dtype=wp.vec2d), i: int, j: int):
    return less(edges[i, 0], edges[j, 0]) or (
        edges[i, 0] == edges[j, 0] and less(edges[i, 1], edges[j, 1])
    )


@wp.kernel
def edge_reset(
    depths: wp.array2d(dtype=wp.float64),
    lower: wp.array(dtype=wp.vec3),
    upper: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    depths[i, 0] = wp.float64(1.0e30)
    depths[i, 1] = wp.float64(1.0e30)
    lower[i] = wp.vec3(1.0e30)
    upper[i] = wp.vec3(-1.0e30)


@wp.kernel
def reduce_edges(
    edges: wp.array2d(dtype=wp.vec2d),
    depths: wp.array2d(dtype=wp.float64),
    ids: wp.array(dtype=int),
    merged: wp.array2d(dtype=wp.vec2d),
    near: wp.array2d(dtype=wp.float64),
    metrics: wp.array(dtype=wp.int64),
):
    i = wp.tid()
    index = ids[i]
    if edges[index, 0][0] >= wp.float64(1.0e29):
        return
    low, high = int(0), i + 1
    while low < high:
        mid = (low + high) // 2
        if edge_less(edges, ids[mid], index):
            low = mid + 1
        else:
            high = mid
    wp.atomic_min(near, low, 0, depths[index, 0])
    wp.atomic_min(near, low, 1, depths[index, 1])
    if low == i:
        merged[low, 0] = edges[index, 0]
        merged[low, 1] = edges[index, 1]
        wp.atomic_add(metrics, 7, wp.int64(1))


@wp.kernel
def edge_bounds(
    edges: wp.array2d(dtype=wp.vec2d),
    depths: wp.array2d(dtype=wp.float64),
    lower: wp.array(dtype=wp.vec3),
    upper: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    if depths[i, 0] < wp.float64(1.0e29):
        a = wp.vec3d(edges[i, 0][0], edges[i, 0][1], depths[i, 0])
        b = wp.vec3d(edges[i, 1][0], edges[i, 1][1], depths[i, 1])
        pad = wp.vec3d(
            2.0e-7,
            2.0e-7,
            wp.float64(2.0e-7)
            * wp.max(wp.float64(1.0), wp.max(wp.abs(a[2]), wp.abs(b[2]))),
        )
        lower[i] = wp.vec3(wp.min(a, b) - pad)
        upper[i] = wp.vec3(wp.max(a, b) + pad)
