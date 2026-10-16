"""Bounded contour tasks and shared ray ordering; all data remain on CUDA."""

# Warp dynamic loop variables need explicit constructor casts.
# ruff: noqa: UP018, RUF046
import warp as wp

from . import kernels as g

wp.set_module_options({"enable_backward": False, "block_dim": 128})


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
    representatives: wp.array(dtype=int),
    points: wp.array(dtype=wp.vec2d),
    status: wp.array(dtype=int),
):
    i = wp.tid()
    x = wp.vec2d(1.0e30)
    if i < status[2] and status[9] == 0:
        p = polygons[representatives[i]]
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


@wp.func
def store_hit(
    source: int,
    group: int,
    x: wp.vec2d,
    scores: wp.array(dtype=wp.vec3d),
    depths: wp.array(dtype=wp.float64),
    sources: wp.array(dtype=int),
    hit_groups: wp.array(dtype=int),
    status: wp.array(dtype=int),
    metrics: wp.array(dtype=wp.int64),
):
    slot = wp.atomic_add(metrics, 5, wp.int64(1))
    if slot < wp.int64(depths.shape[0]):
        depths[int(slot)] = g.value(scores[source], x)
        sources[int(slot)] = source
        hit_groups[int(slot)] = group
    else:
        wp.atomic_or(status, 9, 1024)


@wp.func
def query_subtree(tree: g.Tree, lo: wp.vec3, hi: wp.vec3, node: int, end: int):
    found = int(-1)
    while node > 0 and node != end:
        a, b = tree.lower[node], tree.upper[node]
        if (
            a[0] <= hi[0]
            and a[1] <= hi[1]
            and a[2] <= hi[2]
            and b[0] >= lo[0]
            and b[1] >= lo[1]
            and b[2] >= lo[2]
        ):
            if node < tree.leaves:
                node *= 2
            else:
                found = tree.ids[node - tree.leaves]
                node = g.successor(node)
                break
        else:
            node = g.successor(node)
    return node, found


@wp.kernel
def ray_hits(
    points: wp.array(dtype=wp.vec2d),
    query_ids: wp.array(dtype=int),
    groups: wp.array(dtype=int),
    original: wp.array(dtype=g.Polygon),
    scores: wp.array(dtype=wp.vec3d),
    tree: g.Tree,
    counts: wp.array(dtype=int),
    heavy: wp.array(dtype=wp.vec2i),
    depths: wp.array(dtype=wp.float64),
    sources: wp.array(dtype=int),
    hit_groups: wp.array(dtype=int),
    status: wp.array(dtype=int),
    metrics: wp.array(dtype=wp.int64),
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
    count, visits = int(0), int(0)
    for step in range(32):
        node, source = g.query_next(tree, lo, hi, node)
        if source < 0:
            break
        visits += 1
        if contains(original[source], x):
            store_hit(
                source, i, x, scores, depths, sources, hit_groups, status, metrics
            )
            count += 1
        if node == 0:
            break
    counts[i] = count
    wp.atomic_add(metrics, 3, wp.int64(1))
    wp.atomic_add(metrics, 4, wp.int64(visits))
    if node > 0:
        # The cursor denotes the unvisited DFS suffix. Store its first leaf;
        # disjoint subtrees can now finish that suffix without revisiting hits.
        while node < tree.leaves:
            node *= 2
        slot = int(wp.atomic_add(metrics, 8, wp.int64(1)))
        heavy[slot] = wp.vec2i(i, node)


@wp.kernel
def heavy_ray_hits(
    points: wp.array(dtype=wp.vec2d),
    query_ids: wp.array(dtype=int),
    original: wp.array(dtype=g.Polygon),
    scores: wp.array(dtype=wp.vec3d),
    tree: g.Tree,
    counts: wp.array(dtype=int),
    heavy: wp.array(dtype=wp.vec2i),
    depths: wp.array(dtype=wp.float64),
    sources: wp.array(dtype=int),
    hit_groups: wp.array(dtype=int),
    status: wp.array(dtype=int),
    metrics: wp.array(dtype=wp.int64),
    workers: int,
):
    if status[9] != 0:
        return
    parts = wp.min(8192, tree.leaves)
    while parts > 16 and wp.int64(parts) * metrics[8] > wp.int64(65536):
        parts //= 2
    width = tree.leaves // parts
    for task in range(wp.tid(), int(metrics[8]) * parts, workers):
        job = heavy[task // parts]
        part = task % parts
        first = tree.leaves + part * width
        if job[1] >= first + width:
            continue
        root = parts + part
        node = root
        if job[1] > first:
            node = job[1]
        end = g.successor(root)
        i = job[0]
        x = points[query_ids[i]]
        lo = wp.vec3(float(x[0]) - 2.0e-7, float(x[1]) - 2.0e-7, -1.0e30)
        hi = wp.vec3(float(x[0]) + 2.0e-7, float(x[1]) + 2.0e-7, 1.0e30)
        count, visits = int(0), int(0)
        while node > 0 and node != end:
            node, source = query_subtree(tree, lo, hi, node, end)
            if source < 0:
                break
            visits += 1
            if contains(original[source], x):
                store_hit(
                    source, i, x, scores, depths, sources, hit_groups, status, metrics
                )
                count += 1
        wp.atomic_add(counts, i, count)
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
    representatives: wp.array(dtype=int),
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
    source = sources[representatives[i]]
    z = g.value(scores[source], points[i])
    group = groups[i]
    low = int(0)
    high = counts[group]
    while low < high:
        mid = (low + high) // 2
        h = hit_ids[offsets[group] + mid]
        hz = g.value(scores[hit_sources[h]], points[i])
        if hz < z or (hz == z and hit_sources[h] < source):
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


@wp.kernel
def mesh_tasks(
    counts: wp.array(dtype=int),
    offsets: wp.array(dtype=int),
    tasks: wp.array(dtype=wp.vec2i),
    status: wp.array(dtype=int),
):
    i = wp.tid()
    if i < status[2] and status[9] == 0:
        for edge in range(counts[i]):
            j = offsets[i] + edge
            if j < tasks.shape[0]:
                tasks[j] = wp.vec2i(i, edge)


@wp.kernel
def hit_bucket(
    metrics: wp.array(dtype=wp.int64),
    condition: wp.array(dtype=int),
    maximum: int,
):
    condition[0] = int(metrics[5] <= wp.int64(maximum))


@wp.kernel
def query_bucket(
    status: wp.array(dtype=int), condition: wp.array(dtype=int), size: int
):
    condition[0] = int(status[2] <= size)
