"""Conservative sharing of rank queries over sample bounding rectangles.

A bucket only proposes a group. Every source-BVH candidate must strictly cover
all four corners of the samples' actual bounding rectangle. Otherwise retain
individual queries. On intersection-resolved input, that certificate makes the
source set and depth order constant throughout the rectangle.
"""

# ruff: noqa: UP018, RUF046
import warp as wp

from . import event_kernels as e
from . import kernels as g

wp.set_module_options({"enable_backward": False, "block_dim": 128})


@wp.func
def cell(x: wp.vec2d):
    return int(
        wp.clamp(x[0] * wp.float64(32.0), wp.float64(0.0), wp.float64(31.0))
    ) * 32 + int(wp.clamp(x[1] * wp.float64(32.0), wp.float64(0.0), wp.float64(31.0)))


@wp.kernel
def keys(
    points: wp.array(dtype=wp.vec2d),
    keys: wp.array(dtype=wp.float64),
    ids: wp.array(dtype=int),
    status: wp.array(dtype=int),
):
    i = wp.tid()
    key = wp.float64(1.0e30)
    if i < status[2]:
        key = wp.float64(cell(points[ids[i]]))
    keys[i] = key


@wp.kernel
def reset(bounds: wp.array2d(dtype=wp.float64), valid: wp.array(dtype=int)):
    i = wp.tid()
    for j in range(4):
        bounds[i, j] = wp.float64(1.0e30)
        if j >= 2:
            bounds[i, j] = wp.float64(-1.0e30)
    valid[i] = 0


@wp.kernel
def group(
    points: wp.array(dtype=wp.vec2d),
    ids: wp.array(dtype=int),
    groups: wp.array(dtype=int),
    roots: wp.array(dtype=int),
    counts: wp.array(dtype=int),
    bounds: wp.array2d(dtype=wp.float64),
    status: wp.array(dtype=int),
):
    i = wp.tid()
    counts[i] = 0
    if i >= status[2] or status[9] != 0:
        return
    index = ids[i]
    x = points[index]
    key = cell(x)
    low, high = int(0), i + 1
    while low < high:
        mid = (low + high) // 2
        if cell(points[ids[mid]]) < key:
            low = mid + 1
        else:
            high = mid
    root = low
    roots[index] = root
    wp.atomic_min(bounds, root, 0, x[0])
    wp.atomic_min(bounds, root, 1, x[1])
    wp.atomic_max(bounds, root, 2, x[0])
    wp.atomic_max(bounds, root, 3, x[1])
    high = i + 1
    while low < high:
        mid = (low + high) // 2
        if e.less(points[ids[mid]], x):
            low = mid + 1
        else:
            high = mid
    groups[index] = low


@wp.func
def covers(p: g.Polygon, bounds: wp.array2d(dtype=wp.float64), i: int):
    for j in range(p.n):
        a, b = g.point(p, j), g.point(p, (j + 1) % p.n)
        margin = wp.float64(g.EPS) * wp.length(b - a)
        for c in range(4):
            x = wp.vec2d(bounds[i, 2 * (c % 2)], bounds[i, 1 + 2 * (c // 2)])
            if g.cross(a, b, x) <= margin:
                return False
    return p.n > 0


@wp.func
def box(bounds: wp.array2d(dtype=wp.float64), i: int):
    return (
        wp.vec3(float(bounds[i, 0]) - 2.0e-7, float(bounds[i, 1]) - 2.0e-7, -1.0e30),
        wp.vec3(float(bounds[i, 2]) + 2.0e-7, float(bounds[i, 3]) + 2.0e-7, 1.0e30),
    )


@wp.kernel
def certify(
    ids: wp.array(dtype=int),
    roots: wp.array(dtype=int),
    bounds: wp.array2d(dtype=wp.float64),
    valid: wp.array(dtype=int),
    original: wp.array(dtype=g.Polygon),
    tree: g.Tree,
    heavy: wp.array(dtype=wp.vec2i),
    status: wp.array(dtype=int),
    metrics: wp.array(dtype=wp.int64),
):
    i = wp.tid()
    if i >= status[2] or status[9] != 0 or roots[ids[i]] != i:
        return
    if bounds[i, 0] == bounds[i, 2] and bounds[i, 1] == bounds[i, 3]:
        return  # Identical samples already share a query without certification.
    wp.atomic_add(metrics, 9, wp.int64(1))
    lo, hi = box(bounds, i)
    node, visits = int(1), int(0)
    accepted = int(1)
    for step in range(32):
        node, source = g.query_next(tree, lo, hi, node)
        if source < 0:
            break
        visits += 1
        if not covers(original[source], bounds, i):
            accepted = 0
            break
        if node == 0:
            break
    wp.atomic_add(metrics, 11, wp.int64(visits))
    valid[i] = accepted
    if accepted != 0 and node > 0:
        while node < tree.leaves:
            node *= 2
        slot = int(wp.atomic_add(metrics, 12, wp.int64(1)))
        heavy[slot] = wp.vec2i(i, node)


@wp.kernel
def certify_heavy(
    bounds: wp.array2d(dtype=wp.float64),
    valid: wp.array(dtype=int),
    original: wp.array(dtype=g.Polygon),
    tree: g.Tree,
    heavy: wp.array(dtype=wp.vec2i),
    status: wp.array(dtype=int),
    metrics: wp.array(dtype=wp.int64),
    workers: int,
):
    if status[9] != 0:
        return
    parts = wp.min(8192, tree.leaves)
    while parts > 16 and wp.int64(parts) * metrics[12] > wp.int64(65536):
        parts //= 2
    width = tree.leaves // parts
    for task in range(wp.tid(), int(metrics[12]) * parts, workers):
        job = heavy[task // parts]
        i = job[0]
        part = task % parts
        first = tree.leaves + part * width
        if job[1] >= first + width or wp.atomic_add(valid, i, 0) == 0:
            continue
        root = parts + part
        node = root
        if job[1] > first:
            node = job[1]
        end = g.successor(root)
        lo, hi = box(bounds, i)
        visits = int(0)
        while node > 0 and node != end:
            node, source = e.query_subtree(tree, lo, hi, node, end)
            if source < 0:
                break
            visits += 1
            if not covers(original[source], bounds, i):
                wp.atomic_min(valid, i, 0)
                break
        wp.atomic_add(metrics, 11, wp.int64(visits))


@wp.kernel
def apply(
    ids: wp.array(dtype=int),
    roots: wp.array(dtype=int),
    groups: wp.array(dtype=int),
    valid: wp.array(dtype=int),
    status: wp.array(dtype=int),
    metrics: wp.array(dtype=wp.int64),
):
    i = wp.tid()
    if i < status[2] and status[9] == 0:
        index = ids[i]
        root = roots[index]
        if valid[root] != 0:
            groups[index] = root
            if root == i:
                wp.atomic_add(metrics, 10, wp.int64(1))
