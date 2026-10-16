"""Device geometry for analytic depth peeling. No host geometry or native extensions."""

# Warp requires constructor casts for variables mutated in dynamic loops.
# ruff: noqa: UP018, RUF046
import warp as wp

wp.set_module_options({"enable_backward": False, "block_dim": 32})

# Fixed-size per-polygon registers; pool and per-worker piece capacities are runtime choices.
MAX_VERTICES = wp.constant(32)
EPS = wp.constant(1.0e-10)
Matrix = wp.types.matrix(shape=(32, 2), dtype=wp.float64)
CameraRing = wp.types.matrix(shape=(8, 3), dtype=wp.float64)
Depths = wp.types.vector(length=8, dtype=wp.float64)
EdgeParameters = wp.types.vector(length=128, dtype=wp.float64)

# Status: current residual, next residual, output polygons, output triangles,
# layer count, current layer, boundary vertices, reserved, depth ties,
# error bitmask, maximum hidden pieces per target, reserved, peak residual.
# Error bits: polygon=1, worker=2, residual=4, output=8,
# vertex=16, triangle=32, layer=64, edge noding=128, invalid input=256.


@wp.struct
class Tree:
    lower: wp.array(dtype=wp.vec3)
    upper: wp.array(dtype=wp.vec3)
    ids: wp.array(dtype=int)
    leaves: int


@wp.func
def successor(node: int):
    while node > 0 and (node & 1) != 0:
        node = node >> 1
    if node > 0:
        node += 1
    return node


@wp.func
def query_next(tree: Tree, low: wp.vec3, high: wp.vec3, node: int):
    found = int(-1)
    while node > 0:
        a = tree.lower[node]
        b = tree.upper[node]
        overlap = (
            a[0] <= high[0]
            and a[1] <= high[1]
            and a[2] <= high[2]
            and b[0] >= low[0]
            and b[1] >= low[1]
            and b[2] >= low[2]
        )
        if overlap:
            if node < tree.leaves:
                node = 2 * node
            else:
                found = tree.ids[node - tree.leaves]
                node = successor(node)
                break
        else:
            node = successor(node)
    return node, found


@wp.kernel
def tree_keys(
    lower: wp.array(dtype=wp.vec3),
    upper: wp.array(dtype=wp.vec3),
    keys: wp.array(dtype=wp.uint64),
    ids: wp.array(dtype=int),
    count: int,
    layered: int,
):
    i = wp.tid()
    ids[i] = i
    key = wp.uint64(18446744073709551615)
    if i < count:
        lo = lower[i]
        hi = upper[i]
        if lo[0] <= hi[0] and lo[1] <= hi[1]:
            x = int(wp.clamp((lo[0] + hi[0]) * 0.5, 0.0, 1.0) * 32767.0)
            y = int(wp.clamp((lo[1] + hi[1]) * 0.5, 0.0, 1.0) * 32767.0)
            key = wp.uint64(0)
            for b in range(15):
                key = key | (wp.uint64((x >> b) & 1) << wp.uint64(2 * b))
                key = key | (wp.uint64((y >> b) & 1) << wp.uint64(2 * b + 1))
            if layered != 0:
                key = key | (wp.uint64(int(lo[2])) << wp.uint64(32))
    keys[i] = key


@wp.kernel
def tree_leaves(
    tree: Tree,
    lower: wp.array(dtype=wp.vec3),
    upper: wp.array(dtype=wp.vec3),
    count: int,
):
    i = wp.tid()
    j = tree.ids[i]
    lo = wp.vec3(1.0e30)
    hi = wp.vec3(-1.0e30)
    if j < count:
        lo = lower[j]
        hi = upper[j]
    tree.lower[tree.leaves + i] = lo
    tree.upper[tree.leaves + i] = hi


@wp.kernel
def tree_level(tree: Tree, start: int):
    i = start + wp.tid()
    tree.lower[i] = wp.min(tree.lower[2 * i], tree.lower[2 * i + 1])
    tree.upper[i] = wp.max(tree.upper[2 * i], tree.upper[2 * i + 1])


@wp.struct
class Polygon:
    v: Matrix
    n: int


@wp.func
def point(p: Polygon, i: int):
    return wp.vec2d(p.v[i, 0], p.v[i, 1])


@wp.func
def append(p: Polygon, x: wp.vec2d):
    if p.n < MAX_VERTICES:
        p.v[p.n, 0] = x[0]
        p.v[p.n, 1] = x[1]
    p.n += 1
    return p


@wp.func
def cross(a: wp.vec2d, b: wp.vec2d, c: wp.vec2d):
    u = b - a
    v = c - a
    return u[0] * v[1] - u[1] * v[0]


@wp.func
def clean(p: Polygon):
    q = Polygon()
    if p.n > MAX_VERTICES:
        return p
    for j in range(p.n):
        x = point(p, j)
        if q.n == 0 or wp.length(x - point(q, q.n - 1)) > (
            wp.float64(EPS) * wp.float64(0.01)
        ):
            q = append(q, x)
    if q.n > 1 and wp.length(point(q, 0) - point(q, q.n - 1)) <= (
        wp.float64(EPS) * wp.float64(0.01)
    ):
        q.n -= 1
    if q.n < 3:
        q.n = 0
    else:
        # Use a translation-stable area/perimeter test. A tiny reversed edge
        # caused by roundoff must not erase an otherwise positive-area polygon.
        origin = point(q, 0)
        area2 = wp.float64(0.0)
        perimeter = wp.float64(0.0)
        for j in range(q.n):
            a = point(q, j)
            b = point(q, (j + 1) % q.n)
            area2 += cross(origin, a, b)
            perimeter += wp.abs(b[0] - a[0]) + wp.abs(b[1] - a[1])
        if area2 <= wp.float64(EPS) * perimeter:
            q.n = 0
    return q


@wp.func
def value(s: wp.vec3d, x: wp.vec2d):
    return s[0] * x[0] + s[1] * x[1] + s[2]


@wp.func
def split_at(p: Polygon, s: wp.vec3d, anchor: wp.vec2d):
    inside = Polygon()
    outside = Polygon()
    for j in range(p.n):
        a = point(p, j)
        b = point(p, (j + 1) % p.n)
        da = value(s, a - anchor)
        db = value(s, b - anchor)
        if da >= wp.float64(0.0):
            inside = append(inside, a)
        if da <= wp.float64(0.0):
            outside = append(outside, a)
        if (da > wp.float64(0.0) and db < wp.float64(0.0)) or (
            da < wp.float64(0.0) and db > wp.float64(0.0)
        ):
            x = a + (da / (da - db)) * (b - a)
            inside = append(inside, x)
            outside = append(outside, x)
    return clean(inside), clean(outside)


@wp.func
def split(p: Polygon, s: wp.vec3d):
    return split_at(p, s, wp.vec2d())


@wp.func
def split_edge(p: Polygon, a: wp.vec2d, b: wp.vec2d):
    # Evaluate in the edge's local frame. Forming ax+by+c at the global
    # origin loses the location of very short edges through cancellation.
    d = b - a
    return split_at(p, wp.vec3d(-d[1], d[0], wp.float64(0.0)), a)


@wp.func
def separated(a: Polygon, b: Polygon):
    result = bool(False)
    for j in range(a.n):
        positive = bool(False)
        for k in range(b.n):
            if cross(point(a, j), point(a, (j + 1) % a.n), point(b, k)) > wp.float64(
                0.0
            ):
                positive = bool(True)
        if not positive:
            result = bool(True)
            break
    return result


@wp.func
def bounds(p: Polygon, score: wp.vec3d):
    lo = wp.vec3d(1.0e30)
    hi = wp.vec3d(-1.0e30)
    for j in range(p.n):
        x = point(p, j)
        v = wp.vec3d(x[0], x[1], value(score, x))
        lo = wp.min(lo, v)
        hi = wp.max(hi, v)
    pad = wp.vec3d(
        2.0e-7,
        2.0e-7,
        wp.float64(2.0e-7)
        * wp.max(wp.float64(1.0), wp.max(wp.abs(lo[2]), wp.abs(hi[2]))),
    )
    return wp.vec3(lo - pad), wp.vec3(hi + pad)


@wp.kernel
def reset(
    status: wp.array(dtype=int),
    condition: wp.array(dtype=int),
    stats: wp.array2d(dtype=wp.int64),
):
    i = wp.tid()
    if i < status.shape[0]:
        status[i] = 0
    if i == 0:
        condition[0] = 0
    if i < stats.shape[0]:
        for k in range(stats.shape[1]):
            stats[i, k] = wp.int64(0)


@wp.kernel
def vertex_keys(
    vertices: wp.array(dtype=wp.vec3d),
    keys: wp.array(dtype=wp.float64),
    ids: wp.array(dtype=int),
    axis: int,
    initial: int,
):
    i = wp.tid()
    if initial != 0:
        ids[i] = i
    keys[i] = vertices[ids[i]][axis]


@wp.func
def less(a: wp.vec3d, b: wp.vec3d):
    return a[0] < b[0] or (
        a[0] == b[0] and (a[1] < b[1] or (a[1] == b[1] and a[2] < b[2]))
    )


@wp.kernel
def weld(
    vertices: wp.array(dtype=wp.vec3d),
    ids: wp.array(dtype=int),
    canonical: wp.array(dtype=int),
):
    i = wp.tid()
    v = vertices[ids[i]]
    low = int(0)
    high = i
    while low < high:
        mid = (low + high) // 2
        if less(vertices[ids[mid]], v):
            low = mid + 1
        else:
            high = mid
    canonical[ids[i]] = ids[low]


@wp.kernel
def validate_input(
    vertices: wp.array(dtype=wp.vec3d),
    faces: wp.array(dtype=wp.vec3i),
    camera: wp.array(dtype=wp.mat44d),
    settings: wp.array(dtype=wp.float64),
    status: wp.array(dtype=int),
):
    i = wp.tid()
    if i < vertices.shape[0]:
        v = vertices[i]
        for j in range(3):
            if not wp.isfinite(v[j]):
                wp.atomic_or(status, 9, 256)
    if i < faces.shape[0]:
        for j in range(3):
            if faces[i][j] < 0 or faces[i][j] >= vertices.shape[0]:
                wp.atomic_or(status, 9, 256)
    if i == 0:
        m = camera[0]
        for r in range(4):
            for c in range(4):
                if not wp.isfinite(m[r, c]):
                    wp.atomic_or(status, 9, 256)
        for j in range(6):
            if not wp.isfinite(settings[j]):
                wp.atomic_or(status, 9, 256)
        if (
            wp.determinant(m) == wp.float64(0.0)
            or m[3, 0] != wp.float64(0.0)
            or m[3, 1] != wp.float64(0.0)
            or m[3, 2] != wp.float64(0.0)
            or m[3, 3] != wp.float64(1.0)
        ):
            wp.atomic_or(status, 9, 256)
        if (
            settings[2] <= settings[0]
            or settings[3] <= settings[1]
            or settings[5] <= wp.float64(0.0)
            or (settings[4] != wp.float64(0.0) and settings[4] != wp.float64(1.0))
        ):
            wp.atomic_or(status, 9, 256)


@wp.kernel
def project(
    vertices: wp.array(dtype=wp.vec3d),
    faces: wp.array(dtype=wp.vec3i),
    camera: wp.array(dtype=wp.mat44d),
    settings: wp.array(dtype=wp.float64),
    polygons: wp.array(dtype=Polygon),
    scores: wp.array(dtype=wp.vec3d),
    facing: wp.array(dtype=int),
    clipped: wp.array(dtype=int),
    lower: wp.array(dtype=wp.vec3),
    upper: wp.array(dtype=wp.vec3),
    status: wp.array(dtype=int),
):
    i = wp.tid()
    if status[9] != 0:
        polygons[i] = Polygon()
        lower[i] = wp.vec3(1.0e30)
        upper[i] = wp.vec3(-1.0e30)
        return
    m = camera[0]
    r = CameraRing()
    count = int(3)
    for j in range(3):
        world = vertices[faces[i][j]]
        cam = m * wp.vec4d(world[0], world[1], world[2], 1.0)
        for k in range(3):
            r[j, k] = cam[k]
        if (
            not wp.isfinite(cam[0])
            or not wp.isfinite(cam[1])
            or not wp.isfinite(cam[2])
        ):
            wp.atomic_or(status, 9, 256)
    was_clipped = int(0)
    if settings[4] != wp.float64(0.0):
        near_ring = CameraRing()
        count = int(0)
        for j in range(3):
            a = wp.vec3d(r[j, 0], r[j, 1], r[j, 2])
            b = wp.vec3d(r[(j + 1) % 3, 0], r[(j + 1) % 3, 1], r[(j + 1) % 3, 2])
            da = a[2] - settings[5]
            db = b[2] - settings[5]
            if da >= wp.float64(0.0):
                for k in range(3):
                    near_ring[count, k] = a[k]
                count += 1
            else:
                was_clipped = 1
            if (da > wp.float64(0.0) and db < wp.float64(0.0)) or (
                da < wp.float64(0.0) and db > wp.float64(0.0)
            ):
                near_x = a + da / (da - db) * (b - a)
                for k in range(3):
                    near_ring[count, k] = near_x[k]
                count += 1
        r = near_ring
    p = Polygon()
    depth = Depths()
    for j in range(count):
        invz = wp.float64(1.0)
        if settings[4] != wp.float64(0.0):
            invz = wp.float64(1.0) / r[j, 2]
        x = wp.vec2d(
            (r[j, 0] * invz - settings[0]) / (settings[2] - settings[0]),
            (r[j, 1] * invz - settings[1]) / (settings[3] - settings[1]),
        )
        p = append(p, x)
        depth[j] = r[j, 2]
        if settings[4] != wp.float64(0.0):
            depth[j] = -invz
        if (
            x[0] < wp.float64(0.0)
            or x[0] > wp.float64(1.0)
            or x[1] < wp.float64(0.0)
            or x[1] > wp.float64(1.0)
        ):
            was_clipped = 1
    score = wp.vec3d()
    det = wp.float64(0.0)
    index = int(2)
    if count >= 3:
        for j in range(2, count):
            det = cross(point(p, 0), point(p, 1), point(p, j))
            if det != wp.float64(0.0):
                index = j
                break
    facing[i] = 0
    if det != wp.float64(0.0):
        pa = point(p, 1) - point(p, 0)
        pb = point(p, index) - point(p, 0)
        s0 = ((depth[1] - depth[0]) * pb[1] - (depth[index] - depth[0]) * pa[1]) / det
        s1 = (pa[0] * (depth[index] - depth[0]) - pb[0] * (depth[1] - depth[0])) / det
        score = wp.vec3d(s0, s1, depth[0] - s0 * p.v[0, 0] - s1 * p.v[0, 1])
        facing[i] = 1
        if det < wp.float64(0.0):
            facing[i] = -1
            q = Polygon()
            for j in range(p.n):
                q = append(q, point(p, p.n - 1 - j))
            p = q
        p, _unused = split(p, wp.vec3d(1.0, 0.0, 0.0))
        p, _unused = split(p, wp.vec3d(-1.0, 0.0, 1.0))
        p, _unused = split(p, wp.vec3d(0.0, 1.0, 0.0))
        p, _unused = split(p, wp.vec3d(0.0, -1.0, 1.0))
    else:
        p.n = 0
    for j in range(3):
        if not wp.isfinite(score[j]) or wp.abs(score[j]) > wp.float64(1.0e28):
            wp.atomic_or(status, 9, 256)
    polygons[i] = p
    scores[i] = score
    clipped[i] = was_clipped
    lo, hi = bounds(p, score)
    lower[i] = lo
    upper[i] = hi


@wp.kernel
def edge_keys(
    faces: wp.array(dtype=wp.vec3i),
    canonical: wp.array(dtype=int),
    keys: wp.array(dtype=wp.uint64),
    ids: wp.array(dtype=int),
    masks: wp.array(dtype=int),
    status: wp.array(dtype=int),
):
    e = wp.tid()
    if status[9] != 0:
        keys[e] = wp.uint64(e)
        ids[e] = e
        masks[e] = 0
        return
    f = e // 3
    j = e % 3
    a = canonical[faces[f][j]]
    b = canonical[faces[f][(j + 1) % 3]]
    keys[e] = (wp.uint64(wp.min(a, b)) << wp.uint64(32)) | wp.uint64(wp.max(a, b))
    ids[e] = e
    masks[e] = 0


@wp.kernel
def mark_edges(
    keys: wp.array(dtype=wp.uint64),
    ids: wp.array(dtype=int),
    canonical: wp.array(dtype=int),
    faces: wp.array(dtype=wp.vec3i),
    facing: wp.array(dtype=int),
    masks: wp.array(dtype=int),
    count: int,
):
    i = wp.tid()
    if i == 0 or keys[i - 1] != keys[i]:
        end = i + 1
        while end < count and keys[end] == keys[i]:
            end += 1
        contour = bool(True)
        if end - i == 2:
            e = ids[i]
            f = ids[i + 1]
            a = canonical[faces[e // 3][e % 3]]
            b = canonical[faces[f // 3][f % 3]]
            if a != b and facing[e // 3] == facing[f // 3]:
                contour = bool(False)
        if contour:
            for k in range(i, end):
                masks[ids[k]] = 1


@wp.func
def contour_edge(
    source: int,
    edge: int,
    masks: wp.array(dtype=int),
    facing: wp.array(dtype=int),
    clipped: wp.array(dtype=int),
):
    j = edge
    if clipped[source] != 0:
        return True
    if facing[source] < 0:
        j = (4 - edge) % 3
    return masks[3 * source + j] != 0


@wp.kernel
def contour_bounds(
    polygons: wp.array(dtype=Polygon),
    masks: wp.array(dtype=int),
    facing: wp.array(dtype=int),
    clipped: wp.array(dtype=int),
    lower: wp.array(dtype=wp.vec3),
    upper: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    p = polygons[i]
    lo = wp.vec3d(1.0e30)
    hi = wp.vec3d(-1.0e30)
    for j in range(p.n):
        if contour_edge(i, j, masks, facing, clipped):
            a = point(p, j)
            b = point(p, (j + 1) % p.n)
            lo = wp.min(lo, wp.vec3d(wp.min(a[0], b[0]), wp.min(a[1], b[1]), 0.0))
            hi = wp.max(hi, wp.vec3d(wp.max(a[0], b[0]), wp.max(a[1], b[1]), 0.0))
    lower[i] = wp.vec3(lo - wp.vec3d(2.0e-7))
    upper[i] = wp.vec3(hi + wp.vec3d(2.0e-7))


@wp.func
def touches(p: Polygon, a: wp.vec2d, b: wp.vec2d):
    hit = bool(True)
    for j in range(p.n):
        if cross(point(p, j), point(p, (j + 1) % p.n), a) < -wp.float64(EPS) and cross(
            point(p, j), point(p, (j + 1) % p.n), b
        ) < -wp.float64(EPS):
            hit = bool(False)
    positive = bool(False)
    negative = bool(False)
    for j in range(p.n):
        v = cross(a, b, point(p, j))
        positive = positive or v >= -wp.float64(EPS)
        negative = negative or v <= wp.float64(EPS)
    return hit and positive and negative


@wp.func
def save_polygon(
    p: Polygon,
    source: int,
    layer: int,
    output: wp.array(dtype=Polygon),
    sources: wp.array(dtype=int),
    layers: wp.array(dtype=int),
    status: wp.array(dtype=int),
):
    slot = wp.atomic_add(status, 2, 1)
    if slot < output.shape[0]:
        output[slot] = p
        sources[slot] = source
        layers[slot] = layer
        wp.atomic_max(status, 4, layer + 1)
    else:
        wp.atomic_or(status, 9, 8)


@wp.kernel
def classify(
    polygons: wp.array(dtype=Polygon),
    scores: wp.array(dtype=wp.vec3d),
    masks: wp.array(dtype=int),
    facing: wp.array(dtype=int),
    clipped: wp.array(dtype=int),
    source_bvh: Tree,
    contour_bvh: Tree,
    ranks: wp.array(dtype=int),
    residual: wp.array(dtype=Polygon),
    residual_sources: wp.array(dtype=int),
    output: wp.array(dtype=Polygon),
    output_sources: wp.array(dtype=int),
    output_layers: wp.array(dtype=int),
    status: wp.array(dtype=int),
    limit: int,
    all_layers: int,
):
    i = wp.tid()
    p = polygons[i]
    rank = int(-1)
    if p.n > 0:
        lo, hi = bounds(p, wp.vec3d())
        lo[2] = -1.0e-6
        hi[2] = 1.0e-6
        node = int(1)
        j = int(0)
        boundary = bool(False)
        while node > 0:
            node, j = query_next(contour_bvh, lo, hi, node)
            if j < 0:
                break
            other = polygons[j]
            for e in range(other.n):
                if contour_edge(j, e, masks, facing, clipped) and touches(
                    p, point(other, e), point(other, (e + 1) % other.n)
                ):
                    boundary = bool(True)
                    break
            if boundary:
                break
        if not boundary:
            sample = wp.vec2d()
            for e in range(p.n):
                sample += point(p, e) / wp.float64(p.n)
            z = value(scores[i], sample)
            qlo = wp.vec3(float(sample[0]) - 2.0e-7, float(sample[1]) - 2.0e-7, -1.0e30)
            qhi = wp.vec3(
                float(sample[0]) + 2.0e-7,
                float(sample[1]) + 2.0e-7,
                float(z + wp.float64(2.0e-7) * wp.max(wp.float64(1.0), wp.abs(z))),
            )
            node2 = int(1)
            rank = int(0)
            ambiguous = bool(False)
            while node2 > 0:
                node2, j = query_next(source_bvh, qlo, qhi, node2)
                if j < 0:
                    break
                if j != i:
                    other = polygons[j]
                    inside = bool(True)
                    edge = bool(False)
                    for e in range(other.n):
                        s = cross(
                            point(other, e), point(other, (e + 1) % other.n), sample
                        )
                        inside = inside and s >= wp.float64(0.0)
                        edge = edge or wp.abs(s) <= wp.float64(EPS)
                    if inside:
                        delta = z - value(scores[j], sample)
                        if edge or wp.abs(delta) <= wp.float64(1.0e-12) * wp.max(
                            wp.float64(1.0), wp.abs(z)
                        ):
                            ambiguous = bool(True)
                        elif delta > wp.float64(0.0):
                            rank += 1
            if ambiguous:
                rank = -1
        if rank < 0:
            slot = wp.atomic_add(status, 0, 1)
            if slot < residual.shape[0]:
                residual[slot] = p
                residual_sources[slot] = i
            else:
                wp.atomic_or(status, 9, 4)
        elif rank < limit:
            save_polygon(p, i, rank, output, output_sources, output_layers, status)
        elif all_layers != 0:
            wp.atomic_or(status, 9, 64)
    ranks[i] = rank


@wp.kernel
def start_loop(status: wp.array(dtype=int), condition: wp.array(dtype=int)):
    status[12] = status[0]
    condition[0] = int(status[0] > 0 and status[9] == 0)


@wp.kernel
def active_bounds(
    polygons: wp.array(dtype=Polygon),
    sources: wp.array(dtype=int),
    scores: wp.array(dtype=wp.vec3d),
    status: wp.array(dtype=int),
    lower: wp.array(dtype=wp.vec3),
    upper: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    lo = wp.vec3(1.0e30)
    hi = wp.vec3(-1.0e30)
    if i < status[0] and status[9] == 0:
        lo, hi = bounds(polygons[i], scores[sources[i]])
    lower[i] = lo
    upper[i] = hi


@wp.kernel
def retire_bounds(
    polygons: wp.array(dtype=Polygon),
    scores: wp.array(dtype=wp.vec3d),
    ranks: wp.array(dtype=int),
    status: wp.array(dtype=int),
    lower: wp.array(dtype=wp.vec3),
    upper: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    lo = wp.vec3(1.0e30)
    hi = wp.vec3(-1.0e30)
    if ranks[i] >= status[5]:
        lo, hi = bounds(polygons[i], scores[i])
    lower[i] = lo
    upper[i] = hi


@wp.func
def cut(
    worker: int,
    nv: int,
    nh: int,
    candidate: Polygon,
    delta: wp.vec3d,
    target_source: int,
    other_source: int,
    visible: wp.array2d(dtype=Polygon),
    temporary: wp.array2d(dtype=Polygon),
    hidden: wp.array2d(dtype=Polygon),
    status: wp.array(dtype=int),
):
    count = int(0)
    cap = visible.shape[1]
    for v in range(nv):
        p = visible[worker, v]
        if separated(p, candidate) or separated(candidate, p):
            if count < cap:
                temporary[worker, count] = p
            count += 1
        else:
            positive = bool(False)
            negative = bool(False)
            for e in range(p.n):
                d = value(delta, point(p, e))
                positive = positive or d > wp.float64(0.0)
                negative = negative or d < wp.float64(0.0)
            if not positive and not negative:
                positive = target_source > other_source
                wp.atomic_add(status, 8, 1)
            if not positive:
                if count < cap:
                    temporary[worker, count] = p
                count += 1
            else:
                inside = p
                base = count
                if negative:
                    inside, outside = split(inside, delta)
                    if inside.n > MAX_VERTICES or outside.n > MAX_VERTICES:
                        wp.atomic_or(status, 9, 1)
                        return -1, nh
                    if outside.n > 0:
                        if count < cap:
                            temporary[worker, count] = outside
                        count += 1
                for e in range(candidate.n):
                    if inside.n == 0:
                        break
                    inside, outside = split_edge(
                        inside,
                        point(candidate, e),
                        point(candidate, (e + 1) % candidate.n),
                    )
                    if inside.n > MAX_VERTICES or outside.n > MAX_VERTICES:
                        wp.atomic_or(status, 9, 1)
                        return -1, nh
                    if outside.n > 0:
                        if count < cap:
                            temporary[worker, count] = outside
                        count += 1
                if inside.n == 0:
                    count = base
                    if count < cap:
                        temporary[worker, count] = p
                    count += 1
                else:
                    if nh < cap:
                        hidden[worker, nh] = inside
                    nh += 1
        if count > cap or nh > cap:
            wp.atomic_or(status, 9, 2)
            return -1, nh
    for j in range(count):
        visible[worker, j] = temporary[worker, j]
    return count, nh


@wp.kernel
def peel(
    residual: wp.array(dtype=Polygon),
    residual_sources: wp.array(dtype=int),
    original: wp.array(dtype=Polygon),
    scores: wp.array(dtype=wp.vec3d),
    ranks: wp.array(dtype=int),
    residual_bvh: Tree,
    source_bvh: Tree,
    visible: wp.array2d(dtype=Polygon),
    temporary: wp.array2d(dtype=Polygon),
    hidden: wp.array2d(dtype=Polygon),
    next_polygons: wp.array(dtype=Polygon),
    next_sources: wp.array(dtype=int),
    output: wp.array(dtype=Polygon),
    output_sources: wp.array(dtype=int),
    output_layers: wp.array(dtype=int),
    status: wp.array(dtype=int),
    stats: wp.array2d(dtype=wp.int64),
    workers: int,
):
    worker = wp.tid()
    target = worker
    while target < status[0] and status[9] == 0:
        p = residual[target]
        source = residual_sources[target]
        nv = int(1)
        nh = int(0)
        visible[worker, 0] = p
        lo, hi = bounds(p, scores[source])
        lo[2] = -1.0e30
        tested = int(0)
        for tree_index in range(2):
            tree = source_bvh
            if tree_index == 1:
                tree = residual_bvh
            node = int(1)
            c = int(0)
            while node > 0 and nv > 0:
                node, c = query_next(tree, lo, hi, node)
                if c < 0:
                    break
                other = c
                candidate = Polygon()
                eligible = bool(True)
                if tree_index == 0:
                    candidate = original[c]
                    eligible = ranks[c] >= status[5]
                else:
                    candidate = residual[c]
                    other = residual_sources[c]
                if eligible and other != source:
                    tested += 1
                    nv, nh = cut(
                        worker,
                        nv,
                        nh,
                        candidate,
                        scores[source] - scores[other],
                        source,
                        other,
                        visible,
                        temporary,
                        hidden,
                        status,
                    )
            if nv <= 0:
                break
        wp.atomic_add(stats, status[5], 0, wp.int64(1))
        wp.atomic_add(stats, status[5], 1, wp.int64(tested))
        wp.atomic_max(status, 10, nh)
        if nv == 0:
            hidden[worker, 0] = p
            nh = 1
            wp.atomic_add(stats, status[5], 2, wp.int64(1))
        if nv >= 0 and status[9] == 0:
            for j in range(nv):
                save_polygon(
                    visible[worker, j],
                    source,
                    status[5],
                    output,
                    output_sources,
                    output_layers,
                    status,
                )
            offset = wp.atomic_add(status, 1, nh)
            if offset + nh > next_polygons.shape[0]:
                wp.atomic_or(status, 9, 4)
            else:
                for j in range(nh):
                    next_polygons[offset + j] = hidden[worker, j]
                    next_sources[offset + j] = source
        target += workers


@wp.kernel
def copy_residual(
    next_polygons: wp.array(dtype=Polygon),
    next_sources: wp.array(dtype=int),
    residual: wp.array(dtype=Polygon),
    sources: wp.array(dtype=int),
    status: wp.array(dtype=int),
):
    i = wp.tid()
    if i < status[1] and status[9] == 0:
        residual[i] = next_polygons[i]
        sources[i] = next_sources[i]


@wp.kernel
def advance(
    status: wp.array(dtype=int),
    condition: wp.array(dtype=int),
    limit: int,
    all_layers: int,
):
    status[0] = status[1]
    wp.atomic_max(status, 12, status[0])
    status[1] = 0
    status[5] += 1
    if status[0] > 0 and status[5] >= limit and all_layers != 0:
        wp.atomic_or(status, 9, 64)
    condition[0] = int(status[0] > 0 and status[5] < limit and status[9] == 0)


@wp.kernel
def vertex_counts(
    output: wp.array(dtype=Polygon),
    status: wp.array(dtype=int),
    counts: wp.array(dtype=int),
):
    i = wp.tid()
    counts[i] = 0
    if i < status[2] and status[9] == 0:
        counts[i] = output[i].n


@wp.kernel
def scatter_vertices(
    output: wp.array(dtype=Polygon),
    layers: wp.array(dtype=int),
    offsets: wp.array(dtype=int),
    status: wp.array(dtype=int),
    vertices: wp.array(dtype=wp.vec2d),
    vertex_layers: wp.array(dtype=int),
):
    i = wp.tid()
    if i < status[2] and status[9] == 0:
        p = output[i]
        base = offsets[i]
        if base + p.n > vertices.shape[0]:
            wp.atomic_or(status, 9, 16)
        else:
            for j in range(p.n):
                vertices[base + j] = point(p, j)
                vertex_layers[base + j] = layers[i]
            wp.atomic_max(status, 6, base + p.n)


@wp.kernel
def point_bounds(
    vertices: wp.array(dtype=wp.vec2d),
    layers: wp.array(dtype=int),
    status: wp.array(dtype=int),
    lower: wp.array(dtype=wp.vec3),
    upper: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    lo = wp.vec3(1.0e30)
    hi = wp.vec3(-1.0e30)
    if i < status[6] and status[9] == 0:
        p = vertices[i]
        lo = wp.vec3(float(p[0]) - 2.0e-7, float(p[1]) - 2.0e-7, float(layers[i]))
        hi = wp.vec3(float(p[0]) + 2.0e-7, float(p[1]) + 2.0e-7, float(layers[i]))
    lower[i] = lo
    upper[i] = hi


@wp.func
def lift(
    x: wp.vec2d, s: wp.vec3d, settings: wp.array(dtype=wp.float64), inverse: wp.mat44d
):
    u = settings[0] + x[0] * (settings[2] - settings[0])
    v = settings[1] + x[1] * (settings[3] - settings[1])
    z = value(s, x)
    if settings[4] != wp.float64(0.0):
        z = -wp.float64(1.0) / z
        u *= z
        v *= z
    w = inverse * wp.vec4d(u, v, z, 1.0)
    return wp.vec3d(w[0], w[1], w[2])


# One traversal per edge followed by a bounded heapsort: O(K log K),
# including duplicate endpoints. Capacity exhaustion is an explicit failure.
@wp.func
def mesh_one_edge(
    i: int,
    edge: int,
    output: wp.array(dtype=Polygon),
    sources: wp.array(dtype=int),
    layers: wp.array(dtype=int),
    scores: wp.array(dtype=wp.vec3d),
    vertices: wp.array(dtype=wp.vec2d),
    bvh: Tree,
    camera: wp.array(dtype=wp.mat44d),
    settings: wp.array(dtype=wp.float64),
    status: wp.array(dtype=int),
    triangles_uv: wp.array2d(dtype=wp.vec2d),
    triangles_world: wp.array2d(dtype=wp.vec3d),
    triangle_sources: wp.array(dtype=int),
    triangle_layers: wp.array(dtype=int),
    edge_capacity: int,
):
    if i < status[2] and status[9] == 0:
        p = output[i]
        if edge < p.n:
            a = point(p, edge)
            b = point(p, (edge + 1) % p.n)
            d = b - a
            length2 = wp.dot(d, d)
            center = wp.vec2d()
            for j in range(p.n):
                center += point(p, j) / wp.float64(p.n)
            lower = wp.vec3(
                float(wp.min(a[0], b[0])) - 2.0e-7,
                float(wp.min(a[1], b[1])) - 2.0e-7,
                float(layers[i]),
            )
            upper = wp.vec3(
                float(wp.max(a[0], b[0])) + 2.0e-7,
                float(wp.max(a[1], b[1])) + 2.0e-7,
                float(layers[i]),
            )
            parameters = EdgeParameters()
            count = int(0)
            node = int(1)
            j = int(0)
            while node > 0 and length2 > wp.float64(0.0):
                node, j = query_next(bvh, lower, upper, node)
                if j < 0:
                    break
                x = vertices[j]
                t = wp.dot(x - a, d) / length2
                if (
                    t > wp.float64(EPS)
                    and t < wp.float64(1.0) - wp.float64(EPS)
                    and wp.abs(cross(a, b, x)) <= wp.float64(EPS) * wp.sqrt(length2)
                ):
                    if count < edge_capacity:
                        parameters[count] = t
                        count += 1
                    else:
                        wp.atomic_or(status, 9, 128)
                        return
            # Build a max heap, then extract in ascending order in place.
            start = count // 2 - 1
            end = count - 1
            while end > 0:
                root = int(0)
                if start >= 0:
                    root = start
                    start -= 1
                else:
                    swap = parameters[end]
                    parameters[end] = parameters[0]
                    parameters[0] = swap
                    end -= 1
                while root * 2 + 1 <= end:
                    child = root * 2 + 1
                    if child + 1 <= end and parameters[child] < parameters[child + 1]:
                        child += 1
                    if parameters[root] < parameters[child]:
                        swap = parameters[root]
                        parameters[root] = parameters[child]
                        parameters[child] = swap
                        root = child
                    else:
                        break
            previous = wp.float64(0.0)
            cursor = a
            inverse = wp.inverse(camera[0])
            for step in range(count + 1):
                following = wp.float64(1.0)
                if step < count:
                    following = parameters[step]
                if following <= previous + wp.float64(EPS):
                    continue
                next_point = a + following * d
                if wp.abs(cross(center, cursor, next_point)) > (
                    wp.float64(EPS) * wp.float64(EPS)
                ):
                    slot = wp.atomic_add(status, 3, 1)
                    if slot < triangle_sources.shape[0]:
                        triangles_uv[slot, 0] = center
                        triangles_uv[slot, 1] = cursor
                        triangles_uv[slot, 2] = next_point
                        for k in range(3):
                            triangles_world[slot, k] = lift(
                                triangles_uv[slot, k],
                                scores[sources[i]],
                                settings,
                                inverse,
                            )
                        triangle_sources[slot] = sources[i]
                        triangle_layers[slot] = layers[i]
                    else:
                        wp.atomic_or(status, 9, 32)
                cursor = next_point
                previous = following


@wp.kernel
def mesh_edges(
    output: wp.array(dtype=Polygon),
    sources: wp.array(dtype=int),
    layers: wp.array(dtype=int),
    scores: wp.array(dtype=wp.vec3d),
    vertices: wp.array(dtype=wp.vec2d),
    bvh: Tree,
    camera: wp.array(dtype=wp.mat44d),
    settings: wp.array(dtype=wp.float64),
    status: wp.array(dtype=int),
    triangles_uv: wp.array2d(dtype=wp.vec2d),
    triangles_world: wp.array2d(dtype=wp.vec3d),
    triangle_sources: wp.array(dtype=int),
    triangle_layers: wp.array(dtype=int),
    edge_capacity: int,
):
    i, edge = wp.tid()
    mesh_one_edge(
        i,
        edge,
        output,
        sources,
        layers,
        scores,
        vertices,
        bvh,
        camera,
        settings,
        status,
        triangles_uv,
        triangles_world,
        triangle_sources,
        triangle_layers,
        edge_capacity,
    )


@wp.kernel
def mesh_edge_tasks(
    tasks: wp.array(dtype=wp.vec2i),
    output: wp.array(dtype=Polygon),
    sources: wp.array(dtype=int),
    layers: wp.array(dtype=int),
    scores: wp.array(dtype=wp.vec3d),
    vertices: wp.array(dtype=wp.vec2d),
    bvh: Tree,
    camera: wp.array(dtype=wp.mat44d),
    settings: wp.array(dtype=wp.float64),
    status: wp.array(dtype=int),
    triangles_uv: wp.array2d(dtype=wp.vec2d),
    triangles_world: wp.array2d(dtype=wp.vec3d),
    triangle_sources: wp.array(dtype=int),
    triangle_layers: wp.array(dtype=int),
    edge_capacity: int,
):
    j = wp.tid()
    if j < status[6] and status[9] == 0:
        task = tasks[j]
        mesh_one_edge(
            task[0],
            task[1],
            output,
            sources,
            layers,
            scores,
            vertices,
            bvh,
            camera,
            settings,
            status,
            triangles_uv,
            triangles_world,
            triangle_sources,
            triangle_layers,
            edge_capacity,
        )


@wp.kernel
def scan_leaves(
    counts: wp.array(dtype=int),
    sums: wp.array(dtype=int),
    prefixes: wp.array(dtype=int),
    size: int,
):
    i = wp.tid()
    count = int(0)
    if i < counts.shape[0]:
        count = counts[i]
    sums[size + i] = count
    if i == 0:
        prefixes[1] = 0


@wp.kernel
def scan_up(sums: wp.array(dtype=int), start: int):
    i = start + wp.tid()
    sums[i] = sums[2 * i] + sums[2 * i + 1]


@wp.kernel
def scan_down(sums: wp.array(dtype=int), prefixes: wp.array(dtype=int), start: int):
    i = start + wp.tid()
    prefixes[2 * i] = prefixes[i]
    prefixes[2 * i + 1] = prefixes[i] + sums[2 * i]


@wp.kernel
def scan_result(prefixes: wp.array(dtype=int), offsets: wp.array(dtype=int), size: int):
    i = wp.tid()
    offsets[i] = prefixes[size + i]
