"""Rank propagation across certified patch adjacencies and convex coalescing.

Hashes only propose exact edge matches. Ownership distinguishes a source-local
cut from a shared manifold mesh edge, avoiding projected-stack collision chains.
"""

# ruff: noqa: UP018, RUF046
import warp as wp

from . import kernels as g

wp.set_module_options({"enable_backward": False, "block_dim": 128})


@wp.kernel
def reset(
    slots: wp.array(dtype=int),
    parents: wp.array(dtype=int),
    partners: wp.array(dtype=int),
    choices: wp.array(dtype=int),
):
    i = wp.tid()
    if i < slots.shape[0]:
        slots[i] = -1
    if i < parents.shape[0]:
        parents[i] = i
        choices[i] = 2147483647
    if i < partners.shape[0]:
        partners[i] = -1


@wp.kernel
def topology_keys(
    keys: wp.array(dtype=wp.uint64),
    ids: wp.array(dtype=int),
    owners: wp.array(dtype=wp.uint64),
    masks: wp.array(dtype=int),
    condition: wp.array(dtype=int),
):
    i = wp.tid()
    owners[ids[i]] = keys[i]
    if masks[ids[i]] == 0:
        wp.atomic_or(condition, 0, 1)


@wp.kernel
def connection_condition(
    metrics: wp.array(dtype=wp.int64), condition: wp.array(dtype=int)
):
    condition[0] = int(metrics[2] > wp.int64(0))


@wp.kernel
def check_edges(
    sums: wp.array(dtype=int),
    edge_count: wp.array(dtype=int),
    status: wp.array(dtype=int),
    capacity: int,
):
    edge_count[0] = sums[1]
    if sums[1] > capacity:
        wp.atomic_or(status, 9, 16)


@wp.func
def mix(x: wp.uint64):
    x = (x ^ (x >> wp.uint64(30))) * wp.uint64(13787848793156543929)
    x = (x ^ (x >> wp.uint64(27))) * wp.uint64(10723151780598845931)
    return x ^ (x >> wp.uint64(31))


@wp.func
def point_key(x: wp.vec2d):
    # Quantization affects hash placement only. Full coordinates are compared.
    return mix(wp.uint64(x[0] * wp.float64(1.0e12))) ^ mix(
        wp.uint64(x[1] * wp.float64(1.0e12)) + wp.uint64(19)
    )


@wp.kernel
def edge_owners(
    offsets: wp.array(dtype=int),
    polygons: wp.array(dtype=g.Polygon),
    sources: wp.array(dtype=int),
    original: wp.array(dtype=g.Polygon),
    masks: wp.array(dtype=int),
    facing: wp.array(dtype=int),
    clipped: wp.array(dtype=int),
    topology: wp.array(dtype=wp.uint64),
    owners: wp.array(dtype=wp.uint64),
    starts: wp.array(dtype=wp.vec2d),
    ends: wp.array(dtype=wp.vec2d),
    status: wp.array(dtype=int),
):
    i = wp.tid()
    if i >= status[2] or status[9] != 0:
        return
    p = polygons[i]
    source = sources[i]
    q = original[source]
    for k in range(p.n):
        a, b = g.point(p, k), g.point(p, (k + 1) % p.n)
        owner = wp.uint64(9223372036854775808) | wp.uint64(source)
        if clipped[source] == 0:
            for j in range(q.n):
                edge = j
                if facing[source] < 0:
                    edge = (4 - j) % 3
                if masks[3 * source + edge] == 0:
                    u, v = g.point(q, j), g.point(q, (j + 1) % q.n)
                    # Identity checks also avoid FMA cancellation at endpoints.
                    if (a == u or a == v or g.cross(u, v, a) == wp.float64(0.0)) and (
                        b == u or b == v or g.cross(u, v, b) == wp.float64(0.0)
                    ):
                        owner = topology[3 * source + edge]
                        break
        slot = offsets[i] + k
        owners[slot] = owner
        starts[slot] = a
        ends[slot] = b


@wp.kernel
def match_edges(
    tasks: wp.array(dtype=wp.vec2i),
    starts: wp.array(dtype=wp.vec2d),
    ends: wp.array(dtype=wp.vec2d),
    owners: wp.array(dtype=wp.uint64),
    slots: wp.array(dtype=int),
    partners: wp.array(dtype=int),
    edge_count: wp.array(dtype=int),
    status: wp.array(dtype=int),
    metrics: wp.array(dtype=wp.int64),
):
    i = wp.tid()
    if i >= edge_count[0] or status[9] != 0:
        return
    a, b = starts[i], ends[i]
    if a == b:
        return
    key = mix(owners[i]) ^ point_key(a) ^ point_key(b)
    slot = int(key & wp.uint64(slots.shape[0] - 1))
    for attempt in range(64):
        other = wp.atomic_cas(slots, slot, -1, i)
        if other < 0:
            return
        if (
            owners[i] == owners[other]
            and tasks[i][0] != tasks[other][0]
            and a == ends[other]
            and b == starts[other]
        ):
            partners[i] = other
            partners[other] = i
            wp.atomic_add(metrics, 17, wp.int64(1))
            return
        slot = (slot + 1) & (slots.shape[0] - 1)
    # A bounded hash miss is an optimization miss, never lost geometry.
    wp.atomic_add(metrics, 18, wp.int64(1))


@wp.func
def root(parents: wp.array(dtype=int), i: int):
    node = i
    while True:
        parent = wp.atomic_add(parents, node, 0)
        if parent == node:
            break
        grand = wp.atomic_add(parents, parent, 0)
        wp.atomic_min(parents, node, grand)
        node = grand
    return node


@wp.func
def unite(parents: wp.array(dtype=int), a: int, b: int):
    ra, rb = root(parents, a), root(parents, b)
    while ra != rb:
        high, low = wp.max(ra, rb), wp.min(ra, rb)
        old = wp.atomic_min(parents, high, low)
        if old == high:
            break
        ra, rb = root(parents, old), root(parents, low)


@wp.kernel
def connect(
    tasks: wp.array(dtype=wp.vec2i),
    starts: wp.array(dtype=wp.vec2d),
    ends: wp.array(dtype=wp.vec2d),
    sources: wp.array(dtype=int),
    scores: wp.array(dtype=wp.vec3d),
    partners: wp.array(dtype=int),
    parents: wp.array(dtype=int),
    edges: wp.array2d(dtype=wp.vec2d),
    depths: wp.array2d(dtype=wp.float64),
    tree: g.Tree,
    edge_count: wp.array(dtype=int),
    status: wp.array(dtype=int),
    metrics: wp.array(dtype=wp.int64),
):
    i = wp.tid()
    if i >= edge_count[0] or status[9] != 0:
        return
    other = partners[i]
    if other <= i:
        return
    task = tasks[i]
    a, b = starts[i], ends[i]
    if wp.length(b - a) <= wp.float64(16.0 * g.EPS):
        return
    x = (a + b) * wp.float64(0.5)
    z = g.value(scores[sources[task[0]]], x)
    pad = wp.float64(2.0e-7) * wp.max(wp.float64(1.0), wp.abs(z))
    lo = wp.vec3(float(x[0]) - 2.0e-7, float(x[1]) - 2.0e-7, -1.0e30)
    hi = wp.vec3(float(x[0]) + 2.0e-7, float(x[1]) + 2.0e-7, float(z + pad))
    node, visits = int(1), int(0)
    safe = bool(True)
    while node > 0:
        node, candidate = g.query_next(tree, lo, hi, node)
        if candidate < 0:
            break
        visits += 1
        u, v = edges[candidate, 0], edges[candidate, 1]
        d = v - u
        length = wp.length(d)
        if length > wp.float64(0.0):
            t = wp.dot(x - u, d) / wp.dot(d, d)
            if (
                wp.abs(g.cross(u, v, x)) <= wp.float64(g.EPS) * length
                and t >= -wp.float64(g.EPS) / length
                and t <= wp.float64(1.0) + wp.float64(g.EPS) / length
            ):
                depth = depths[candidate, 0] + t * (
                    depths[candidate, 1] - depths[candidate, 0]
                )
                if depth <= z + pad:
                    safe = False
                    break
    wp.atomic_add(metrics, 15, wp.int64(visits))
    if safe:
        unite(parents, task[0], tasks[other][0])
        wp.atomic_add(metrics, 14, wp.int64(1))


@wp.kernel
def roots(
    parents: wp.array(dtype=int),
    representatives: wp.array(dtype=int),
    status: wp.array(dtype=int),
    metrics: wp.array(dtype=wp.int64),
):
    i = wp.tid()
    if i < status[2] and status[9] == 0:
        r = root(parents, i)
        representatives[i] = r
        if i == r:
            wp.atomic_add(metrics, 13, wp.int64(1))


@wp.kernel
def identity(
    representatives: wp.array(dtype=int),
    status: wp.array(dtype=int),
    metrics: wp.array(dtype=wp.int64),
    enabled: int,
):
    i = wp.tid()
    representatives[i] = i
    if i == 0:
        metrics[13] = wp.int64(status[2] * enabled)


@wp.kernel
def propose(
    tasks: wp.array(dtype=wp.vec2i),
    polygons: wp.array(dtype=g.Polygon),
    sources: wp.array(dtype=int),
    layers: wp.array(dtype=int),
    partners: wp.array(dtype=int),
    choices: wp.array(dtype=int),
    edge_count: wp.array(dtype=int),
    status: wp.array(dtype=int),
):
    i = wp.tid()
    if i >= edge_count[0] or status[9] != 0:
        return
    other = partners[i]
    if other <= i:
        return
    ta, tb = tasks[i], tasks[other]
    if sources[ta[0]] != sources[tb[0]] or layers[ta[0]] != layers[tb[0]]:
        return
    a, b = polygons[ta[0]], polygons[tb[0]]
    if a.n < 3 or b.n < 3 or a.n + b.n - 2 > g.MAX_VERTICES:
        return
    u, v = g.point(a, ta[1]), g.point(b, tb[1])
    if g.cross(
        g.point(a, (ta[1] + a.n - 1) % a.n), u, g.point(b, (tb[1] + 2) % b.n)
    ) >= wp.float64(0.0) and g.cross(
        g.point(b, (tb[1] + b.n - 1) % b.n), v, g.point(a, (ta[1] + 2) % a.n)
    ) >= wp.float64(0.0):
        wp.atomic_min(choices, ta[0], i)
        wp.atomic_min(choices, tb[0], i)


@wp.kernel
def merge(
    tasks: wp.array(dtype=wp.vec2i),
    polygons: wp.array(dtype=g.Polygon),
    partners: wp.array(dtype=int),
    choices: wp.array(dtype=int),
    edge_count: wp.array(dtype=int),
    status: wp.array(dtype=int),
    metrics: wp.array(dtype=wp.int64),
):
    i = wp.tid()
    if i >= edge_count[0] or status[9] != 0:
        return
    other = partners[i]
    if other <= i:
        return
    ta, tb = tasks[i], tasks[other]
    if choices[ta[0]] != i or choices[tb[0]] != i:
        return
    a, b = polygons[ta[0]], polygons[tb[0]]
    joined = g.Polygon()
    for j in range(a.n):
        joined = g.append(joined, g.point(a, (ta[1] + 1 + j) % a.n))
    for j in range(b.n - 2):
        joined = g.append(joined, g.point(b, (tb[1] + 2 + j) % b.n))
    q = g.Polygon()
    for j in range(joined.n):
        prev = g.point(joined, (j + joined.n - 1) % joined.n)
        x = g.point(joined, j)
        following = g.point(joined, (j + 1) % joined.n)
        if g.cross(prev, x, following) != wp.float64(0.0):
            q = g.append(q, x)
    if q.n >= 3:
        polygons[ta[0]] = q
        b.n = 0
        polygons[tb[0]] = b
        wp.atomic_add(metrics, 16, wp.int64(1))


@wp.kernel
def live_counts(
    polygons: wp.array(dtype=g.Polygon),
    counts: wp.array(dtype=int),
    status: wp.array(dtype=int),
):
    i = wp.tid()
    counts[i] = 0
    if i < status[2] and status[9] == 0:
        counts[i] = int(polygons[i].n > 0)


@wp.kernel
def compact(
    polygons: wp.array(dtype=g.Polygon),
    sources: wp.array(dtype=int),
    layers: wp.array(dtype=int),
    offsets: wp.array(dtype=int),
    out: wp.array(dtype=g.Polygon),
    out_sources: wp.array(dtype=int),
    out_layers: wp.array(dtype=int),
    status: wp.array(dtype=int),
):
    i = wp.tid()
    if i < status[2] and status[9] == 0 and polygons[i].n > 0:
        slot = offsets[i]
        out[slot] = polygons[i]
        out_sources[slot] = sources[i]
        out_layers[slot] = layers[i]


@wp.kernel
def restore(
    polygons: wp.array(dtype=g.Polygon),
    sources: wp.array(dtype=int),
    layers: wp.array(dtype=int),
    out: wp.array(dtype=g.Polygon),
    out_sources: wp.array(dtype=int),
    out_layers: wp.array(dtype=int),
    sums: wp.array(dtype=int),
    status: wp.array(dtype=int),
):
    i = wp.tid()
    if status[9] == 0:
        if i < sums[1]:
            polygons[i] = out[i]
            sources[i] = out_sources[i]
            layers[i] = out_layers[i]
        if i == 0:
            status[2] = sums[1]
