# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
"""Ray and overlap queries against meshes: what a ray hits, and what a sphere, capsule or box touches.

Colliders holds a set of objects to ask about. Each mesh gets a bounding volume hierarchy (a tree of boxes around
its triangles, built once in its own space and shared by every object showing it and every Colliders set), and
update() reads where the objects are, once a frame, so that the queries after it cost only the kernels:

    walls = Colliders([*level, door, *crates])
    ...                                     # move things
    walls.update()
    hit = walls.raycast(eye, forward, max_distance=20)
    ground = walls.raycast(feet + (0, 0.5, 0), (0, -1, 0))
    walker += walls.push_out(walker, 0.3)   # out of anything a sphere of radius 0.3 there would be inside

Being in a set is what makes an object solid: `visible` and `opacity` play no part, so an invisible box can stand in
for a detailed model, and holes in textures count as solid. Rays hit faces from either side. Separate sets work as
layers (walls, enemies, pickups); the trees are shared, so another set costs little.

What a game does with a contact (sliding along walls, gravity, steps) is up to the game.
"""
import functools
import math
import weakref
from dataclasses import dataclass

import numpy as np
from numba import njit, prange

from .renderer import _rotations
from .threads import kernel_lock
from .transforms import quat_to_matrix, scale3, world_matrix

LEAF_SIZE = 4     # triangles in a leaf of a tree, at most (unless they can't be split)
BINS = 16         # candidate split planes a node tries along its longest axis
MAX_DEPTH = 48    # deepest a tree goes (a node this deep is a leaf, however many triangles it holds)
STACK = MAX_DEPTH + 2  # nodes a walk through a tree keeps waiting at once, at most
RAY_CHUNK = 64    # rays each iteration of raycast_many's parallel loop casts
MERGE_COS = 0.9   # contacts with one object whose directions are closer than this (about 25 degrees) are one
FLAT = 1e-12      # a triangle whose corners are this close to a line (relative to its sides) has no area to meet, and
                  # a ray this close to a triangle's plane (relative to its sides) runs along it rather than through it
SLACK = 1e-9      # a ray passes over boxes and spheres around triangles only where they start this much (relative)
                  # beyond the distance it looks to: a hit right at its max_distance is found whatever the rounding


@dataclass(repr=False)
class Hit:
    """Where a ray met a surface: the object, the point in the world, the normal there (unit length, on the side
    the ray came from), how far along the ray that is, the face (a row of the object's mesh.faces), and whether
    the ray met its front (the outside of a closed mesh)."""
    object: object
    position: np.ndarray
    normal: np.ndarray
    distance: float
    face: int
    front: bool


@dataclass(repr=False)
class Contact:
    """Where a shape touches or cuts into a surface: the object, the point on its face nearest the shape, the
    direction to move the shape to get it out (unit length), how far (depth), and the face (a row of the object's
    mesh.faces)."""
    object: object
    point: np.ndarray
    normal: np.ndarray
    depth: float
    face: int


def _describe(record):
    """Hit and Contact as text, with the object named by its type rather than printed whole."""
    fields = ", ".join(f"{k}={np.round(v, 4).tolist() if isinstance(v, np.ndarray) else v!r}"
                       for k, v in vars(record).items() if k != "object")
    return f"{type(record).__name__}({type(record.object).__name__}, {fields})"


Hit.__repr__ = Contact.__repr__ = _describe


# ----- trees -------------------------------------------------------------------------------------------------------

@njit(cache=True, error_model="numpy")
def _bin(value, low, extent):
    """The bin (0..BINS-1) a centroid coordinate falls in, along an axis spanning low..low+extent; 0 for NaN."""
    f = (value - low) / extent * BINS
    if f >= BINS - 1:
        return BINS - 1
    if f > 0.0:
        return int(f)
    return 0


@njit(cache=True, error_model="numpy")
def _half_area(lo, hi):
    """Half the surface area of a box (what the chance of a ray passing through it goes with)."""
    dx, dy, dz = hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]
    return dx * dy + dy * dz + dz * dx


@njit(cache=True, error_model="numpy")
def build_tree(tri):
    """A bounding volume hierarchy over triangles tri (F, 3, 3), F > 0, all corners finite: (box minima (N, 3),
    maxima (N, 3), first (N,), count (N,), order (F,), depth). A node with count > 0 is a leaf of triangles
    order[first:first + count]; one with count 0 has children first and first + 1. Split by the surface area
    heuristic, binned along each node's longest axis of centroids."""
    n_tri = len(tri)
    lo_t = np.empty((n_tri, 3))
    hi_t = np.empty((n_tri, 3))
    cen = np.empty((n_tri, 3))
    for t in range(n_tri):
        for a in range(3):
            lo_t[t, a] = min(tri[t, 0, a], tri[t, 1, a], tri[t, 2, a])
            hi_t[t, a] = max(tri[t, 0, a], tri[t, 1, a], tri[t, 2, a])
            cen[t, a] = (tri[t, 0, a] + tri[t, 1, a] + tri[t, 2, a]) / 3.0
    order = np.arange(n_tri)
    size = 2 * n_tri
    lo = np.empty((size, 3))
    hi = np.empty((size, 3))
    first = np.zeros(size, np.int64)
    count = np.zeros(size, np.int64)
    # Nodes waiting to be split: (node, start, end, depth). Each pop pushes at most two, one level deeper.
    stack = np.empty((2 * MAX_DEPTH + 4, 4), np.int64)
    stack[0, 0], stack[0, 1], stack[0, 2], stack[0, 3] = 0, 0, n_tri, 0
    sp, n_nodes, deepest = 1, 1, 0
    bin_count = np.zeros(BINS, np.int64)
    bin_lo = np.empty((BINS, 3))
    bin_hi = np.empty((BINS, 3))
    right_area = np.empty(BINS)
    right_count = np.zeros(BINS, np.int64)
    c_lo = np.empty(3)
    c_hi = np.empty(3)
    acc_lo = np.empty(3)
    acc_hi = np.empty(3)
    while sp > 0:
        sp -= 1
        node, start, end, depth = stack[sp, 0], stack[sp, 1], stack[sp, 2], stack[sp, 3]
        deepest = max(deepest, depth)
        for a in range(3):
            lo[node, a], hi[node, a] = np.inf, -np.inf
            c_lo[a], c_hi[a] = np.inf, -np.inf
        for i in range(start, end):
            t = order[i]
            for a in range(3):
                lo[node, a] = min(lo[node, a], lo_t[t, a])
                hi[node, a] = max(hi[node, a], hi_t[t, a])
                c_lo[a] = min(c_lo[a], cen[t, a])
                c_hi[a] = max(c_hi[a], cen[t, a])
        n = end - start
        first[node], count[node] = start, n
        if n <= LEAF_SIZE or depth >= MAX_DEPTH:
            continue
        axis = 0
        for a in range(1, 3):
            if c_hi[a] - c_lo[a] > c_hi[axis] - c_lo[axis]:
                axis = a
        extent = c_hi[axis] - c_lo[axis]
        if not extent > 0.0:  # every centroid in one place (or out of range): no split helps
            continue
        bin_count[:] = 0
        bin_lo[:] = np.inf
        bin_hi[:] = -np.inf
        for i in range(start, end):
            t = order[i]
            b = _bin(cen[t, axis], c_lo[axis], extent)
            bin_count[b] += 1
            for a in range(3):
                bin_lo[b, a] = min(bin_lo[b, a], lo_t[t, a])
                bin_hi[b, a] = max(bin_hi[b, a], hi_t[t, a])
        # Sweep from the right for the boxes right of each plane, then from the left for the cost of each.
        acc_lo[:], acc_hi[:] = np.inf, -np.inf
        total = 0
        for b in range(BINS - 1, 0, -1):
            total += bin_count[b]
            for a in range(3):
                acc_lo[a] = min(acc_lo[a], bin_lo[b, a])
                acc_hi[a] = max(acc_hi[a], bin_hi[b, a])
            right_count[b] = total
            right_area[b] = _half_area(acc_lo, acc_hi) if total else 0.0
        acc_lo[:], acc_hi[:] = np.inf, -np.inf
        total, best, split = 0, np.inf, -1
        for b in range(BINS - 1):
            total += bin_count[b]
            for a in range(3):
                acc_lo[a] = min(acc_lo[a], bin_lo[b, a])
                acc_hi[a] = max(acc_hi[a], bin_hi[b, a])
            if total == 0 or right_count[b + 1] == 0:
                continue
            cost = total * _half_area(acc_lo, acc_hi) + right_count[b + 1] * right_area[b + 1]
            if cost < best:
                best, split = cost, b + 1
        if split < 0:  # (costs not finite: coordinates near the largest floats) split in the middle instead
            mid = start + n // 2
        else:
            i, j = start, end - 1
            while i <= j:
                if _bin(cen[order[i], axis], c_lo[axis], extent) < split:
                    i += 1
                else:
                    order[i], order[j] = order[j], order[i]
                    j -= 1
            mid = i
            if mid == start or mid == end:
                mid = start + n // 2
        left = n_nodes
        n_nodes += 2
        first[node], count[node] = left, 0
        stack[sp, 0], stack[sp, 1], stack[sp, 2], stack[sp, 3] = left + 1, mid, end, depth + 1
        stack[sp + 1, 0], stack[sp + 1, 1], stack[sp + 1, 2], stack[sp + 1, 3] = left, start, mid, depth + 1
        sp += 2
    return lo[:n_nodes].copy(), hi[:n_nodes].copy(), first[:n_nodes].copy(), count[:n_nodes].copy(), order, deepest


def _has_area(tri):
    """Whether each triangle of tri (F, 3, 3) has an area: its corners are not all on one line (to within FLAT).
    (A ray meeting one that hasn't would find a distance that is only rounding error, and a shape touching it no
    direction to be pushed out in.) Call under np.errstate."""
    e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    size = np.maximum(np.abs(e1).max(axis=1), np.abs(e2).max(axis=1))[:, None]  # (scaled to 1: no overflow)
    e1, e2 = e1 / size, e2 / size
    n = np.cross(e1, e2)
    return (n * n).sum(axis=1) > FLAT * FLAT * (e1 * e1).sum(axis=1) * (e2 * e2).sum(axis=1)  # (NaN fails)


class _Tree:
    """A mesh's tree, in the mesh's own space: its triangles in the tree's order and the face each came from."""

    def __init__(self, mesh):
        mesh.check()
        vertices = np.asarray(mesh.vertices, dtype=np.float64).reshape(-1, 3)
        faces = np.asarray(mesh.faces, dtype=np.int64).reshape(-1, 3)
        tri = vertices[faces] if len(faces) else np.zeros((0, 3, 3))
        with np.errstate(all="ignore"):
            # (a corner at NaN or infinity, or no area: never hit)
            keep = np.flatnonzero(np.isfinite(tri).all(axis=(1, 2)) & _has_area(tri))
            tri = np.ascontiguousarray(tri[keep])
            if len(tri):
                self.lo, self.hi, self.first, self.count, order, self.depth = build_tree(tri)
            else:
                self.lo = self.hi = np.zeros((0, 3))
                self.first = self.count = order = np.zeros(0, np.int64)
                self.depth = 0
            self.tri = np.ascontiguousarray(tri[order])
            self.face = keep[order]
            if len(tri):
                centre = (self.lo[0] + self.hi[0]) / 2
                self.sphere = (centre, float(np.sqrt(((tri.reshape(-1, 3) - centre) ** 2).sum(axis=1).max())))
            else:
                self.sphere = (np.zeros(3), 0.0)


_TREES = {}  # id(mesh): ((vertices, faces) as the tree was built from them, _Tree), or None after invalidate()


def _forget(key):
    """Drop a mesh's tree, once the mesh is gone."""
    _TREES.pop(key, None)


def mesh_tree(mesh):
    """The mesh's tree, built on first use and kept while the mesh lives (shared by every Colliders); built afresh
    when mesh.vertices or mesh.faces are replaced, but not when they are edited in place (Colliders.invalidate)."""
    entry = _TREES.get(id(mesh), False)
    if entry and entry[0][0] is mesh.vertices and entry[0][1] is mesh.faces:
        return entry[1]
    if entry is False:  # (once a mesh: invalidate() leaves None, so that calling it each frame adds nothing)
        weakref.finalize(mesh, _forget, id(mesh))
    tree = _Tree(mesh)
    _TREES[id(mesh)] = ((mesh.vertices, mesh.faces), tree)
    return tree


# ----- the walk through the trees ----------------------------------------------------------------------------------

@njit(cache=True, error_model="numpy")
def _ray_sphere(ox, oy, oz, dx, dy, dz, sphere, best):
    """Whether a ray (unit direction) passes within a sphere (cx, cy, cz, r) before distance best."""
    px, py, pz = ox - sphere[0], oy - sphere[1], oz - sphere[2]
    b = px * dx + py * dy + pz * dz
    c = px * px + py * py + pz * pz - sphere[3] * sphere[3]
    disc = b * b - c
    if disc < 0.0:
        return False
    root = math.sqrt(disc)
    return not (-b + root < 0.0 or -b - root > best * (1.0 + SLACK))  # (NaN: tested anyway)


@njit(cache=True, error_model="numpy")
def _slab(l, inverse, lo, hi, near, far):
    """near and far narrowed to where a ray (origin l, inverse direction along one axis) is between lo and hi on
    that axis; a ray parallel to the axis's planes (inverse infinite) narrows nothing if it is between them, and
    everything (far below near) if not."""
    if not abs(inverse) < np.inf:
        return (near, far) if lo <= l <= hi else (np.inf, -np.inf)
    t1, t2 = (lo - l) * inverse, (hi - l) * inverse
    return max(near, min(t1, t2)), min(far, max(t1, t2))


@njit(cache=True, error_model="numpy")
def _ray_box(lx, ly, lz, ix, iy, iz, lo, hi, node, best):
    """Whether a ray (origin l, inverse direction i) passes through box `node` before distance best."""
    near, far = _slab(lx, ix, lo[node, 0], hi[node, 0], -np.inf, np.inf)
    near, far = _slab(ly, iy, lo[node, 1], hi[node, 1], near, far)
    near, far = _slab(lz, iz, lo[node, 2], hi[node, 2], near, far)
    return not (near > far or far < 0.0 or near > best * (1.0 + SLACK))  # (NaN: tested anyway)


@njit(cache=True, error_model="numpy")
def _ray_triangle(lx, ly, lz, dx, dy, dz, tri, t):
    """Distance along a ray (origin l, direction d) to triangle t of tri, or inf if it misses (Moller-Trumbore)."""
    ax, ay, az = tri[t, 0, 0], tri[t, 0, 1], tri[t, 0, 2]
    e1x, e1y, e1z = tri[t, 1, 0] - ax, tri[t, 1, 1] - ay, tri[t, 1, 2] - az
    e2x, e2y, e2z = tri[t, 2, 0] - ax, tri[t, 2, 1] - ay, tri[t, 2, 2] - az
    px, py, pz = dy * e2z - dz * e2y, dz * e2x - dx * e2z, dx * e2y - dy * e2x
    det = e1x * px + e1y * py + e1z * pz
    # (along the triangle's plane, to within rounding: a miss, rather than a distance made of rounding errors)
    if not abs(det) > FLAT * ((abs(e1x) + abs(e1y) + abs(e1z)) * (abs(e2x) + abs(e2y) + abs(e2z))
                              * (abs(dx) + abs(dy) + abs(dz))):
        return np.inf
    inv = 1.0 / det
    sx, sy, sz = lx - ax, ly - ay, lz - az
    u = (sx * px + sy * py + sz * pz) * inv
    if not (u >= 0.0 and u <= 1.0):
        return np.inf
    qx, qy, qz = sy * e1z - sz * e1y, sz * e1x - sx * e1z, sx * e1y - sy * e1x
    v = (dx * qx + dy * qy + dz * qz) * inv
    if not (v >= 0.0 and u + v <= 1.0):
        return np.inf
    d = (e2x * qx + e2y * qy + e2z * qz) * inv
    return d if d >= 0.0 else np.inf


@njit(cache=True, error_model="numpy")
def _ray_instance(i, ox, oy, oz, dx, dy, dz, best, inv, pos, inst_tree, node_start, tri_start, lo, hi, first,
                  count, tri, stack):
    """The nearest triangle (an index into tri, or -1) instance i has along a ray before distance best, and the
    distance to it."""
    rx, ry, rz = ox - pos[i, 0], oy - pos[i, 1], oz - pos[i, 2]
    lx = inv[i, 0, 0] * rx + inv[i, 0, 1] * ry + inv[i, 0, 2] * rz
    ly = inv[i, 1, 0] * rx + inv[i, 1, 1] * ry + inv[i, 1, 2] * rz
    lz = inv[i, 2, 0] * rx + inv[i, 2, 1] * ry + inv[i, 2, 2] * rz
    ldx = inv[i, 0, 0] * dx + inv[i, 0, 1] * dy + inv[i, 0, 2] * dz  # (not unit length: distances stay the world's)
    ldy = inv[i, 1, 0] * dx + inv[i, 1, 1] * dy + inv[i, 1, 2] * dz
    ldz = inv[i, 2, 0] * dx + inv[i, 2, 1] * dy + inv[i, 2, 2] * dz
    ix, iy, iz = 1.0 / ldx, 1.0 / ldy, 1.0 / ldz
    m = inst_tree[i]
    base, tbase = node_start[m], tri_start[m]
    found = -1
    stack[0] = base
    sp = 1
    while sp > 0:
        sp -= 1
        node = stack[sp]
        if not _ray_box(lx, ly, lz, ix, iy, iz, lo, hi, node, best):
            continue
        if count[node] > 0:
            for k in range(tbase + first[node], tbase + first[node] + count[node]):
                d = _ray_triangle(lx, ly, lz, ldx, ldy, ldz, tri, k)
                if d <= best and d < np.inf:
                    best, found = d, k
        else:
            stack[sp], stack[sp + 1] = base + first[node], base + first[node] + 1
            sp += 2
    return found, best


@njit(cache=True, error_model="numpy")
def _face_normal(i, k, dx, dy, dz, inv, tri):
    """The unit normal in the world of triangle k of instance i on the side a ray along d meets it, and whether that
    is its front (the side its corners go round counter-clockwise)."""
    e1x, e1y, e1z = tri[k, 1, 0] - tri[k, 0, 0], tri[k, 1, 1] - tri[k, 0, 1], tri[k, 1, 2] - tri[k, 0, 2]
    e2x, e2y, e2z = tri[k, 2, 0] - tri[k, 0, 0], tri[k, 2, 1] - tri[k, 0, 1], tri[k, 2, 2] - tri[k, 0, 2]
    nx, ny, nz = e1y * e2z - e1z * e2y, e1z * e2x - e1x * e2z, e1x * e2y - e1y * e2x
    # Normals go into the world by the inverse transpose, which keeps them on the outside even through a mirroring.
    wx = inv[i, 0, 0] * nx + inv[i, 1, 0] * ny + inv[i, 2, 0] * nz
    wy = inv[i, 0, 1] * nx + inv[i, 1, 1] * ny + inv[i, 2, 1] * nz
    wz = inv[i, 0, 2] * nx + inv[i, 1, 2] * ny + inv[i, 2, 2] * nz
    length = math.sqrt(wx * wx + wy * wy + wz * wz)
    if length > 0.0:
        wx, wy, wz = wx / length, wy / length, wz / length
    front = wx * dx + wy * dy + wz * dz <= 0.0
    if not front:
        wx, wy, wz = -wx, -wy, -wz
    return wx, wy, wz, front


@njit(cache=True, error_model="numpy")
def _cast(ox, oy, oz, dx, dy, dz, best, skip, sphere, inv, pos, inst_tree, node_start, tri_start, lo, hi, first,
          count, tri, stack):
    """The nearest hit of a ray (unit direction) before distance best: (instance or -1, triangle, distance)."""
    hit_inst, hit_tri = -1, -1
    for i in range(len(skip)):
        if skip[i] or not _ray_sphere(ox, oy, oz, dx, dy, dz, sphere[i], best):
            continue
        k, d = _ray_instance(i, ox, oy, oz, dx, dy, dz, best, inv, pos, inst_tree, node_start, tri_start, lo, hi,
                             first, count, tri, stack)
        if k >= 0:
            hit_inst, hit_tri, best = i, k, d
    return hit_inst, hit_tri, best


@njit(cache=True, error_model="numpy")
def cast_ray(origin, direction, max_distance, skip, sphere, inv, pos, inst_tree, node_start, tri_start, lo, hi,
             first, count, tri, out):
    """The nearest hit of one ray (unit direction) within max_distance: its instance (-1 for none) and triangle;
    out gets (distance, normal x, y, z, front)."""
    stack = np.empty(STACK, np.int64)
    ox, oy, oz, dx, dy, dz = origin[0], origin[1], origin[2], direction[0], direction[1], direction[2]
    i, k, d = _cast(ox, oy, oz, dx, dy, dz, max_distance, skip, sphere, inv, pos, inst_tree, node_start, tri_start,
                    lo, hi, first, count, tri, stack)
    if i >= 0:
        out[1], out[2], out[3], front = _face_normal(i, k, dx, dy, dz, inv, tri)
        out[0], out[4] = d, 1.0 if front else 0.0
    return i, k


@njit(cache=True, error_model="numpy")
def cast_all(origin, direction, max_distance, skip, sphere, inv, pos, inst_tree, node_start, tri_start, lo, hi,
             first, count, tri, out_inst, out_tri, out_dist):
    """Every hit of one ray (unit direction) within max_distance, in no order, into out_* while they have room;
    returns how many there are (more than the room: call again with more)."""
    stack = np.empty(STACK, np.int64)
    ox, oy, oz, dx, dy, dz = origin[0], origin[1], origin[2], direction[0], direction[1], direction[2]
    n = 0
    for i in range(len(skip)):
        if skip[i] or not _ray_sphere(ox, oy, oz, dx, dy, dz, sphere[i], max_distance):
            continue
        rx, ry, rz = ox - pos[i, 0], oy - pos[i, 1], oz - pos[i, 2]
        lx = inv[i, 0, 0] * rx + inv[i, 0, 1] * ry + inv[i, 0, 2] * rz
        ly = inv[i, 1, 0] * rx + inv[i, 1, 1] * ry + inv[i, 1, 2] * rz
        lz = inv[i, 2, 0] * rx + inv[i, 2, 1] * ry + inv[i, 2, 2] * rz
        ldx = inv[i, 0, 0] * dx + inv[i, 0, 1] * dy + inv[i, 0, 2] * dz
        ldy = inv[i, 1, 0] * dx + inv[i, 1, 1] * dy + inv[i, 1, 2] * dz
        ldz = inv[i, 2, 0] * dx + inv[i, 2, 1] * dy + inv[i, 2, 2] * dz
        ix, iy, iz = 1.0 / ldx, 1.0 / ldy, 1.0 / ldz
        m = inst_tree[i]
        base, tbase = node_start[m], tri_start[m]
        stack[0] = base
        sp = 1
        while sp > 0:
            sp -= 1
            node = stack[sp]
            if not _ray_box(lx, ly, lz, ix, iy, iz, lo, hi, node, max_distance):
                continue
            if count[node] > 0:
                for k in range(tbase + first[node], tbase + first[node] + count[node]):
                    d = _ray_triangle(lx, ly, lz, ldx, ldy, ldz, tri, k)
                    if d <= max_distance and d < np.inf:
                        if n < len(out_inst):
                            out_inst[n], out_tri[n], out_dist[n] = i, k, d
                        n += 1
            else:
                stack[sp], stack[sp + 1] = base + first[node], base + first[node] + 1
                sp += 2
    return n


@njit(cache=True, error_model="numpy", parallel=True)
def cast_rays(origins, directions, max_distance, skip, sphere, inv, pos, inst_tree, node_start, tri_start, lo, hi,
              first, count, tri, out_inst, out_tri, out_dist, out_normal):
    """The nearest hit of each ray (unit directions) within max_distance[r]: out_inst[r] (-1 for none), out_tri[r],
    out_dist[r] (inf for none) and out_normal[r]. Parallel over chunks of rays; each writes only its own rays'
    slots."""
    n_rays = len(origins)
    for c in prange((n_rays + RAY_CHUNK - 1) // RAY_CHUNK):
        stack = np.empty(STACK, np.int64)
        for r in range(c * RAY_CHUNK, min((c + 1) * RAY_CHUNK, n_rays)):
            ox, oy, oz = origins[r, 0], origins[r, 1], origins[r, 2]
            dx, dy, dz = directions[r, 0], directions[r, 1], directions[r, 2]
            i, k, d = _cast(ox, oy, oz, dx, dy, dz, max_distance[r], skip, sphere, inv, pos, inst_tree, node_start,
                            tri_start, lo, hi, first, count, tri, stack)
            out_inst[r], out_tri[r] = i, k
            if i >= 0:
                out_dist[r] = d
                out_normal[r, 0], out_normal[r, 1], out_normal[r, 2], _ = _face_normal(i, k, dx, dy, dz, inv, tri)
            else:
                out_dist[r] = np.inf
                out_normal[r, 0] = out_normal[r, 1] = out_normal[r, 2] = 0.0


# ----- overlaps ----------------------------------------------------------------------------------------------------

@njit(cache=True, error_model="numpy")
def _closest_on_triangle(px, py, pz, a, b, c):
    """The point of triangle a, b, c nearest p (Ericson, Real-Time Collision Detection, 5.1.5)."""
    abx, aby, abz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    acx, acy, acz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
    apx, apy, apz = px - a[0], py - a[1], pz - a[2]
    d1 = abx * apx + aby * apy + abz * apz
    d2 = acx * apx + acy * apy + acz * apz
    if d1 <= 0.0 and d2 <= 0.0:
        return a[0], a[1], a[2]
    bpx, bpy, bpz = px - b[0], py - b[1], pz - b[2]
    d3 = abx * bpx + aby * bpy + abz * bpz
    d4 = acx * bpx + acy * bpy + acz * bpz
    if d3 >= 0.0 and d4 <= d3:
        return b[0], b[1], b[2]
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return a[0] + v * abx, a[1] + v * aby, a[2] + v * abz
    cpx, cpy, cpz = px - c[0], py - c[1], pz - c[2]
    d5 = abx * cpx + aby * cpy + abz * cpz
    d6 = acx * cpx + acy * cpy + acz * cpz
    if d6 >= 0.0 and d5 <= d6:
        return c[0], c[1], c[2]
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return a[0] + w * acx, a[1] + w * acy, a[2] + w * acz
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and d4 - d3 >= 0.0 and d5 - d6 >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return b[0] + w * (c[0] - b[0]), b[1] + w * (c[1] - b[1]), b[2] + w * (c[2] - b[2])
    denom = va + vb + vc
    if not abs(denom) > 0.0:  # (a triangle with no area: its first corner will do)
        return a[0], a[1], a[2]
    v, w = vb / denom, vc / denom
    return a[0] + abx * v + acx * w, a[1] + aby * v + acy * w, a[2] + abz * v + acz * w


@njit(cache=True, error_model="numpy")
def _clamp01(x):
    return min(1.0, max(0.0, x))  # (NaN as 0)


@njit(cache=True, error_model="numpy")
def _closest_segments(p, q, r, s):
    """Fractions (a, b) along segments p-q and r-s of their nearest points (Ericson 5.1.9)."""
    d1x, d1y, d1z = q[0] - p[0], q[1] - p[1], q[2] - p[2]
    d2x, d2y, d2z = s[0] - r[0], s[1] - r[1], s[2] - r[2]
    rx, ry, rz = p[0] - r[0], p[1] - r[1], p[2] - r[2]
    a = d1x * d1x + d1y * d1y + d1z * d1z
    e = d2x * d2x + d2y * d2y + d2z * d2z
    f = d2x * rx + d2y * ry + d2z * rz
    if not a > 1e-300 and not e > 1e-300:
        return 0.0, 0.0
    if not a > 1e-300:
        return 0.0, _clamp01(f / e)
    c = d1x * rx + d1y * ry + d1z * rz
    if not e > 1e-300:
        return _clamp01(-c / a), 0.0
    b = d1x * d2x + d1y * d2y + d1z * d2z
    denom = a * e - b * b
    sa = _clamp01((b * f - c * e) / denom) if denom > 0.0 else 0.0
    tb = (b * sa + f) / e
    if tb < 0.0:
        tb, sa = 0.0, _clamp01(-c / a)
    elif tb > 1.0:
        tb, sa = 1.0, _clamp01((b - c) / a)
    return sa, tb


@njit(cache=True, error_model="numpy")
def _capsule_triangle(a, b, radius, v, w):
    """Whether the capsule a-b (radius) touches triangle v (3, 3), and the contact: point on the triangle, the
    unit direction to push the capsule out, and how far. w is scratch space (2, 3), for an edge."""
    e1x, e1y, e1z = v[1, 0] - v[0, 0], v[1, 1] - v[0, 1], v[1, 2] - v[0, 2]
    e2x, e2y, e2z = v[2, 0] - v[0, 0], v[2, 1] - v[0, 1], v[2, 2] - v[0, 2]
    nx, ny, nz = e1y * e2z - e1z * e2y, e1z * e2x - e1x * e2z, e1x * e2y - e1y * e2x
    length = math.sqrt(nx * nx + ny * ny + nz * nz)
    if length > 0.0:
        nx, ny, nz = nx / length, ny / length, nz / length
    # The segment crossing the triangle: push out along the normal, towards the end farther from its plane.
    da = (a[0] - v[0, 0]) * nx + (a[1] - v[0, 1]) * ny + (a[2] - v[0, 2]) * nz
    db = (b[0] - v[0, 0]) * nx + (b[1] - v[0, 1]) * ny + (b[2] - v[0, 2]) * nz
    if length > 0.0 and ((da > radius and db > radius) or (da < -radius and db < -radius)):
        return False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # both ends farther from its plane than radius, one side
    if length > 0.0 and da * db < 0.0:
        f = da / (da - db)
        x, y, z = a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]), a[2] + f * (b[2] - a[2])
        cx, cy, cz = _closest_on_triangle(x, y, z, v[0], v[1], v[2])
        if (cx - x) ** 2 + (cy - y) ** 2 + (cz - z) ** 2 <= 1e-24 * (1.0 + x * x + y * y + z * z):
            if abs(da) > abs(db):
                return True, x, y, z, nx * (1.0 if da > 0.0 else -1.0), ny * (1.0 if da > 0.0 else -1.0), \
                    nz * (1.0 if da > 0.0 else -1.0), radius + abs(db)
            return True, x, y, z, nx * (1.0 if db > 0.0 else -1.0), ny * (1.0 if db > 0.0 else -1.0), \
                nz * (1.0 if db > 0.0 else -1.0), radius + abs(da)
    # Otherwise the nearest points are from an end to the face, or from the segment to an edge.
    best = np.inf
    px = py = pz = sx = sy = sz = 0.0
    for end in range(1 if a[0] == b[0] and a[1] == b[1] and a[2] == b[2] else 2):
        e = a if end == 0 else b
        cx, cy, cz = _closest_on_triangle(e[0], e[1], e[2], v[0], v[1], v[2])
        d = (cx - e[0]) ** 2 + (cy - e[1]) ** 2 + (cz - e[2]) ** 2
        if d < best:
            best, px, py, pz, sx, sy, sz = d, cx, cy, cz, e[0], e[1], e[2]
    for edge in range(0 if a[0] == b[0] and a[1] == b[1] and a[2] == b[2] else 3):  # (a sphere's nearest point is its centre's)
        for k in range(3):
            w[0, k], w[1, k] = v[edge, k], v[(edge + 1) % 3, k]
        fs, ft = _closest_segments(a, b, w[0], w[1])
        qx, qy, qz = a[0] + fs * (b[0] - a[0]), a[1] + fs * (b[1] - a[1]), a[2] + fs * (b[2] - a[2])
        cx, cy, cz = w[0, 0] + ft * (w[1, 0] - w[0, 0]), w[0, 1] + ft * (w[1, 1] - w[0, 1]), \
            w[0, 2] + ft * (w[1, 2] - w[0, 2])
        d = (cx - qx) ** 2 + (cy - qy) ** 2 + (cz - qz) ** 2
        if d < best:
            best, px, py, pz, sx, sy, sz = d, cx, cy, cz, qx, qy, qz
    if not best < radius * radius:
        return False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    dist = math.sqrt(best)
    if dist > 1e-12 * (1.0 + radius):
        return True, px, py, pz, (sx - px) / dist, (sy - py) / dist, (sz - pz) / dist, radius - dist
    side = 1.0 if (sx - v[0, 0]) * nx + (sy - v[0, 1]) * ny + (sz - v[0, 2]) * nz >= 0.0 else -1.0
    return True, px, py, pz, side * nx, side * ny, side * nz, radius  # (on the face: out along its normal)


@njit(cache=True, error_model="numpy")
def _box_triangle(centre, axes, half, v, axis_out):
    """Whether a box (centre, unit axes as rows, half sizes) overlaps triangle v (3, 3), by separating axes: the
    box's three, the triangle's normal and the nine crossings of their edges. Returns (overlaps, depth) with the
    direction to push the box out by the least, depth, in axis_out."""
    best = np.inf
    for test in range(13):
        if test < 3:
            lx, ly, lz = axes[test, 0], axes[test, 1], axes[test, 2]
        else:
            e1x, e1y, e1z = v[1, 0] - v[0, 0], v[1, 1] - v[0, 1], v[1, 2] - v[0, 2]
            if test == 3:
                e2x, e2y, e2z = v[2, 0] - v[0, 0], v[2, 1] - v[0, 1], v[2, 2] - v[0, 2]
                lx, ly, lz = e1y * e2z - e1z * e2y, e1z * e2x - e1x * e2z, e1x * e2y - e1y * e2x
            else:
                j, edge = (test - 4) // 3, (test - 4) % 3
                ex = v[(edge + 1) % 3, 0] - v[edge, 0]
                ey = v[(edge + 1) % 3, 1] - v[edge, 1]
                ez = v[(edge + 1) % 3, 2] - v[edge, 2]
                ax, ay, az = axes[j, 0], axes[j, 1], axes[j, 2]
                lx, ly, lz = ay * ez - az * ey, az * ex - ax * ez, ax * ey - ay * ex
        length = math.sqrt(lx * lx + ly * ly + lz * lz)
        if not length > 1e-12:
            continue  # (parallel edges, or a triangle with no area: another axis covers it)
        lx, ly, lz = lx / length, ly / length, lz / length
        t0 = v[0, 0] * lx + v[0, 1] * ly + v[0, 2] * lz
        t1 = v[1, 0] * lx + v[1, 1] * ly + v[1, 2] * lz
        t2 = v[2, 0] * lx + v[2, 1] * ly + v[2, 2] * lz
        tmin, tmax = min(t0, t1, t2), max(t0, t1, t2)
        pc = centre[0] * lx + centre[1] * ly + centre[2] * lz
        rb = 0.0
        for k in range(3):
            rb += half[k] * abs(axes[k, 0] * lx + axes[k, 1] * ly + axes[k, 2] * lz)
        overlap = min(tmax - (pc - rb), (pc + rb) - tmin)
        if not overlap > 0.0:
            return False, 0.0
        if overlap < best:
            best = overlap
            sign = 1.0 if pc >= (tmin + tmax) / 2 else -1.0
            axis_out[0], axis_out[1], axis_out[2] = sign * lx, sign * ly, sign * lz
    return best < np.inf, best


@njit(cache=True, error_model="numpy")
def overlap(kind, a, b, radius, axes, half, skip, sphere, lin, inv, pos, inst_tree, node_start, tri_start, lo, hi,
            first, count, tri, out_inst, out_tri, out_point, out_normal, out_depth):
    """The contacts of a shape with the instances, into out_*: for each instance, one for each way out (contacts
    whose directions are within MERGE_COS of each other are one, the deepest); returns how many, or -1 if out_*
    have too little room (call again with more). kind 0 is a capsule a-b (a sphere where a == b) of `radius`; kind 1 a box with
    centre a, unit axes (rows of axes) and half sizes `half`. Triangles are tested in the world, after their
    instance's bounding sphere and tree have been narrowed down in the instance's own space."""
    stack = np.empty(STACK, np.int64)
    v = np.empty((3, 3))
    w = np.empty((2, 3))
    axis_out = np.empty(3)
    q_lo, q_hi = np.empty(3), np.empty(3)
    # A box around the shape in the world, and a sphere around that.
    for k in range(3):
        if kind == 0:
            q_lo[k], q_hi[k] = min(a[k], b[k]) - radius, max(a[k], b[k]) + radius
        else:
            reach = abs(axes[0, k]) * half[0] + abs(axes[1, k]) * half[1] + abs(axes[2, k]) * half[2]
            q_lo[k], q_hi[k] = a[k] - reach, a[k] + reach
    cx, cy, cz = (q_lo[0] + q_hi[0]) / 2, (q_lo[1] + q_hi[1]) / 2, (q_lo[2] + q_hi[2]) / 2
    hx, hy, hz = (q_hi[0] - q_lo[0]) / 2, (q_hi[1] - q_lo[1]) / 2, (q_hi[2] - q_lo[2]) / 2
    q_radius = math.sqrt(hx * hx + hy * hy + hz * hz)
    n = 0
    for i in range(len(skip)):
        if skip[i]:
            continue
        sx, sy, sz = cx - sphere[i, 0], cy - sphere[i, 1], cz - sphere[i, 2]
        if not sx * sx + sy * sy + sz * sz <= (q_radius + sphere[i, 3]) ** 2:
            continue
        # The world box in the instance's space: a box around its image there.
        rx, ry, rz = cx - pos[i, 0], cy - pos[i, 1], cz - pos[i, 2]
        lc0 = inv[i, 0, 0] * rx + inv[i, 0, 1] * ry + inv[i, 0, 2] * rz
        lc1 = inv[i, 1, 0] * rx + inv[i, 1, 1] * ry + inv[i, 1, 2] * rz
        lc2 = inv[i, 2, 0] * rx + inv[i, 2, 1] * ry + inv[i, 2, 2] * rz
        lh0 = abs(inv[i, 0, 0]) * hx + abs(inv[i, 0, 1]) * hy + abs(inv[i, 0, 2]) * hz
        lh1 = abs(inv[i, 1, 0]) * hx + abs(inv[i, 1, 1]) * hy + abs(inv[i, 1, 2]) * hz
        lh2 = abs(inv[i, 2, 0]) * hx + abs(inv[i, 2, 1]) * hy + abs(inv[i, 2, 2]) * hz
        m = inst_tree[i]
        base, tbase = node_start[m], tri_start[m]
        mine = n  # where this instance's contacts start
        stack[0] = base
        sp = 1
        while sp > 0:
            sp -= 1
            node = stack[sp]
            if (lo[node, 0] > lc0 + lh0 or hi[node, 0] < lc0 - lh0 or lo[node, 1] > lc1 + lh1
                    or hi[node, 1] < lc1 - lh1 or lo[node, 2] > lc2 + lh2 or hi[node, 2] < lc2 - lh2):
                continue
            if count[node] == 0:
                stack[sp], stack[sp + 1] = base + first[node], base + first[node] + 1
                sp += 2
                continue
            for t in range(tbase + first[node], tbase + first[node] + count[node]):
                for c in range(3):
                    for k in range(3):
                        v[c, k] = (lin[i, k, 0] * tri[t, c, 0] + lin[i, k, 1] * tri[t, c, 1]
                                   + lin[i, k, 2] * tri[t, c, 2] + pos[i, k])
                px = py = pz = nx = ny = nz = 0.0
                if kind == 0:
                    touch, px, py, pz, nx, ny, nz, depth = _capsule_triangle(a, b, radius, v, w)
                else:
                    touch, depth = _box_triangle(a, axes, half, v, axis_out)
                    if touch:
                        px, py, pz = _closest_on_triangle(a[0], a[1], a[2], v[0], v[1], v[2])
                        nx, ny, nz = axis_out[0], axis_out[1], axis_out[2]
                if not touch:
                    continue
                # One contact for each way out: merged with this instance's contact pushing nearly the same way.
                j = mine
                while j < n and (out_normal[j, 0] * nx + out_normal[j, 1] * ny + out_normal[j, 2] * nz
                                 < MERGE_COS):
                    j += 1
                if j == n:
                    if n == len(out_inst):
                        return -1  # no room: call again with more
                    n += 1
                elif not depth > out_depth[j]:
                    continue
                out_inst[j], out_tri[j], out_depth[j] = i, t, depth
                out_point[j, 0], out_point[j, 1], out_point[j, 2] = px, py, pz
                out_normal[j, 0], out_normal[j, 1], out_normal[j, 2] = nx, ny, nz
    return n


# ----- sets of objects ---------------------------------------------------------------------------------------------

def _parts(objects):
    """The Object3Ds in objects (an Object3D, a Model, or a list of either), with each Model unpacked into its
    parts."""
    if hasattr(objects, "mesh"):  # one Object3D
        yield objects
        return
    for obj in objects:
        if hasattr(obj, "root") and hasattr(obj, "__iter__"):  # a Model
            yield from obj
        else:
            yield obj


def _point(value, name):
    point = np.asarray(value, dtype=np.float64).reshape(-1)
    if point.shape != (3,):
        raise ValueError(f"{name} must be three numbers, not {np.shape(value)}")
    return np.ascontiguousarray(point)


def _lengths(directions):
    """The lengths of directions (N, 3), added up one coordinate at a time: the same bits on any machine, so a ray
    gives the same answer from raycast and raycast_many (numpy's sums and dot products round differently with the
    processor's vector instructions)."""
    d = np.asarray(directions, dtype=np.float64)
    return np.sqrt(d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1] + d[:, 2] * d[:, 2])


def _limit(value):
    """A distance limit as a float: None as no limit, NaN and anything below 0 as 0."""
    value = np.inf if value is None else float(value)
    return value if value > 0.0 else 0.0


def _quiet(method):
    """method, run under np.errstate(all="ignore"): bad numbers give no hit or contact rather than a warning (which
    would be printed over the picture), as in Renderer.render()."""
    @functools.wraps(method)
    def quiet(*args, **kwargs):
        with np.errstate(all="ignore"):
            return method(*args, **kwargs)
    return quiet


class Colliders:
    """A set of objects (Object3Ds, or Models for all their parts) to ask what a ray hits or what a shape touches.

    The set reads where its objects are when made and on each update(): call that after moving them, once a frame
    for any number of queries. Objects can be added to or taken out of `objects` (a list) at any time, before an
    update(). A mesh's tree is built the first time a set holding it updates (a fraction of a second for a hundred
    thousand triangles) and shared with every other set; after editing a mesh's arrays in place, call invalidate().

    Each object counts with the pose it has in the world through its parents, whether or not it is visible. An
    object scaled to nothing in some direction, or with a pose that isn't finite, is never hit.
    """

    def __init__(self, objects=()):
        self.objects = list(_parts(objects))
        self._packed = None   # (trees, arrays): the trees of the meshes in the set, packed together
        self._instances = []  # the objects the arrays hold, in their order
        self.update()

    def invalidate(self):
        """Build the trees of the set's meshes afresh (after their arrays were edited in place), and update()."""
        for obj in _parts(self.objects):
            mesh = getattr(obj, "mesh", None)
            if mesh is not None and id(mesh) in _TREES:
                _TREES[id(mesh)] = None
        self._packed = None
        self.update()

    @_quiet
    def update(self):
        """Read where the objects are now (and pick up objects added to or taken out of `objects`)."""
        self.objects = list(_parts(self.objects))
        objects, trees, tree_of, inst_tree, where, turn, scale, placed_by = [], [], {}, [], [], [], [], []
        parents = {}  # id(node): its pose in the world (world_matrix), worked out once for all its children
        for obj in self.objects:
            mesh = getattr(obj, "mesh", None)
            if mesh is None:
                continue
            tree = mesh_tree(mesh)
            if not len(tree.tri):
                continue
            if obj.parent is not None:
                placed_by.append((len(where), world_matrix(obj.parent, parents)[:2]))
            where.append(obj.position)
            turn.append(obj.rotation)
            sc = obj.scale
            scale.append((sc, sc, sc) if isinstance(sc, (int, float, np.number)) else scale3(sc))
            m = tree_of.get(id(tree))
            if m is None:
                m = tree_of[id(tree)] = len(trees)
                trees.append(tree)
            objects.append(obj)
            inst_tree.append(m)
        self._instances = objects
        n = len(objects)
        pos = np.array(where, np.float64).reshape(n, 3)
        quat = np.array(turn, np.float64).reshape(n, 4)
        quat = quat / np.maximum(np.linalg.norm(quat, axis=1, keepdims=True), 1e-300)
        lin = _rotations(quat) * np.array(scale, np.float64).reshape(n, 1, 3)
        for row, (parent_lin, parent_pos) in placed_by:  # (as Object3D.world_matrix places them)
            lin[row], pos[row] = parent_lin @ lin[row], parent_pos + parent_lin @ pos[row]
        # The inverse from the adjugate: no exception for a singular matrix, which is skipped instead.
        c0, c1, c2 = lin[:, :, 0], lin[:, :, 1], lin[:, :, 2]
        rows = np.stack([np.cross(c1, c2), np.cross(c2, c0), np.cross(c0, c1)], axis=1)
        det = np.einsum("ij,ij->i", c0, rows[:, 0])
        inv = rows / det[:, None, None]
        size = np.abs(lin).max(axis=(1, 2))
        ok = (np.isfinite(lin).all(axis=(1, 2)) & np.isfinite(pos).all(axis=1) & np.isfinite(inv).all(axis=(1, 2))
              & (np.abs(det) > 1e-12 * size ** 3))
        # Bounding spheres in the world: the stretch of lin is at most the square root of the largest row sum
        # of |lin^T lin| (Gershgorin), exact for a rotation and an even scale.
        radius = np.array([trees[m].sphere[1] for m in inst_tree], np.float64)
        centre = np.array([trees[m].sphere[0] for m in inst_tree], np.float64).reshape(n, 3)
        stretch = np.sqrt(np.abs(np.einsum("nki,nkj->nij", lin, lin)).sum(axis=2).max(axis=1)) if n else radius
        sphere = np.concatenate([np.einsum("nij,nj->ni", lin, centre) + pos, (radius * stretch)[:, None]], axis=1)
        inv[~ok] = 0.0
        sphere[~ok] = 0.0
        if self._packed is None or len(self._packed[0]) != len(trees) or any(
                a is not b for a, b in zip(self._packed[0], trees)):
            self._packed = (trees, self._pack(trees))
        c = np.ascontiguousarray
        self._arrays = {"sphere": c(sphere), "lin": c(np.where(ok[:, None, None], lin, 0.0)), "inv": c(inv),
                        "pos": c(np.where(ok[:, None], pos, 0.0)), "inst_tree": np.array(inst_tree, np.int64),
                        **self._packed[1]}
        self._skip_base = (~ok).astype(np.uint8)

    @staticmethod
    def _pack(trees):
        """The trees' arrays end to end, with where each one's nodes and triangles start."""
        node_start = np.cumsum([0] + [len(t.lo) for t in trees])[:-1].astype(np.int64)
        tri_start = np.cumsum([0] + [len(t.tri) for t in trees])[:-1].astype(np.int64)
        c = np.ascontiguousarray
        cat = lambda parts, shape, dtype: c(np.concatenate(parts) if parts else np.zeros(shape, dtype), dtype)
        return {"node_start": node_start, "tri_start": tri_start,
                "lo": cat([t.lo for t in trees], (0, 3), np.float64), "hi": cat([t.hi for t in trees], (0, 3), np.float64),
                "first": cat([t.first for t in trees], 0, np.int64), "count": cat([t.count for t in trees], 0, np.int64),
                "tri": cat([t.tri for t in trees], (0, 3, 3), np.float64),
                "face": cat([t.face for t in trees], 0, np.int64)}

    def _skip(self, ignore):
        """Which instances a query passes over: those that can't be hit, and those in `ignore` (objects or Models)."""
        skip = self._skip_base.copy()
        if ignore:
            ids = {id(obj) for obj in _parts(ignore)}
            for i, obj in enumerate(self._instances):
                if id(obj) in ids:
                    skip[i] = 1
        return skip

    def _tree_args(self):
        a = self._arrays
        return (a["inv"], a["pos"], a["inst_tree"], a["node_start"], a["tri_start"], a["lo"], a["hi"], a["first"],
                a["count"], a["tri"])

    @_quiet
    def raycast(self, origin, direction, max_distance=np.inf, ignore=(), all=False):
        """The nearest Hit of the ray from origin along direction within max_distance, or None. ignore: objects (or
        Models) to pass through, such as the one casting it. With all=True, every Hit along it, nearest first (a
        ray through a closed mesh hits it going in and coming out)."""
        origin, direction = _point(origin, "origin"), _point(direction, "direction")
        length = float(_lengths(direction[None])[0])
        if not (length > 0.0 and np.isfinite(length) and np.isfinite(origin).all()):
            return [] if all else None
        direction = direction / length
        skip, a = self._skip(ignore), self._arrays
        limit = _limit(max_distance)
        if all:
            room = 16
            while True:
                inst, tri, dist = np.empty(room, np.int64), np.empty(room, np.int64), np.empty(room)
                n = cast_all(origin, direction, limit, skip, a["sphere"], *self._tree_args(), inst, tri, dist)
                if n <= room:
                    break
                room = n
            hits = []
            for j in np.argsort(dist[:n], kind="stable"):
                hit = self._hit(inst[j], tri[j], dist[j], origin, direction)
                last = hits[-1] if hits else None
                if (last is not None and last.object is hit.object and last.front == hit.front
                        and hit.distance - last.distance <= 1e-9 * (1.0 + hit.distance)):
                    continue  # (through an edge or a corner: the faces sharing it are one surface)
                hits.append(hit)
            return hits
        out = np.zeros(5)
        i, k = cast_ray(origin, direction, limit, skip, a["sphere"], *self._tree_args(), out)
        if i < 0:
            return None
        return Hit(self._instances[i], origin + direction * out[0], out[1:4].copy(), float(out[0]),
                   int(a["face"][k]), bool(out[4]))

    def _hit(self, i, k, distance, origin, direction):
        """A Hit from what a kernel found: instance i, triangle k (in the packed trees) at distance."""
        a = self._arrays
        tri = a["tri"][k]
        normal = np.cross(tri[1] - tri[0], tri[2] - tri[0]) @ a["inv"][i]  # (inverse transpose: see _face_normal)
        length = np.linalg.norm(normal)
        normal = normal / length if length > 0.0 else normal
        front = bool(normal @ direction <= 0.0)
        return Hit(self._instances[i], origin + direction * distance, normal if front else -normal, float(distance),
                   int(a["face"][k]), front)

    @_quiet
    def raycast_many(self, origins, directions, max_distance=np.inf, ignore=()):
        """The nearest hit of each of many rays at once, on all cores, as arrays: (objects, a list with None where
        a ray hits nothing; distances (N,), inf for none; positions (N, 3); normals (N, 3), zero for none; faces
        (N,), -1 for none). origins and directions are (N, 3), or (3,) for the same one for every ray; max_distance
        one number or (N,)."""
        origins = np.asarray(origins, dtype=np.float64)
        directions = np.asarray(directions, dtype=np.float64)
        rays = np.broadcast_shapes(origins.shape[:-1], directions.shape[:-1])  # (N,), or () for one ray
        n = rays[0] if rays else 1
        origins = np.ascontiguousarray(np.broadcast_to(origins, (n, 3)), np.float64)
        directions = np.broadcast_to(directions, (n, 3))
        lengths = _lengths(directions)
        usable = (lengths > 0.0) & np.isfinite(lengths) & np.isfinite(origins).all(axis=1)
        directions = np.ascontiguousarray(np.where(usable[:, None], directions / lengths[:, None], 0.0), np.float64)
        limit = np.broadcast_to(np.asarray(np.inf if max_distance is None else max_distance, dtype=np.float64), (n,))
        limit = np.ascontiguousarray(np.where(usable & (limit > 0.0), limit, 0.0), np.float64)  # (NaN as 0)
        origins = np.ascontiguousarray(np.where(usable[:, None], origins, 0.0), np.float64)
        inst, tri = np.empty(n, np.int64), np.empty(n, np.int64)
        dist, normal = np.empty(n), np.empty((n, 3))
        a = self._arrays
        with kernel_lock():
            cast_rays(origins, directions, limit, self._skip(ignore), a["sphere"], *self._tree_args(), inst, tri, dist,
                      normal)
        hit = inst >= 0
        dist[~usable] = np.inf
        objects = [self._instances[i] if i >= 0 else None for i in inst]
        positions = np.where(hit[:, None], origins + directions * np.where(hit, dist, 0.0)[:, None], np.nan)
        faces = np.where(hit, a["face"][np.where(hit, tri, 0)] if len(a["face"]) else -1, -1)
        return objects, dist, positions, normal, faces

    def _overlap(self, kind, a, b, radius, axes, half, ignore):
        arrays, skip = self._arrays, self._skip(ignore)
        room = 16
        while True:
            inst, tri = np.empty(room, np.int64), np.empty(room, np.int64)
            point, normal, depth = np.empty((room, 3)), np.empty((room, 3)), np.empty(room)
            n = overlap(kind, a, b, radius, axes, half, skip, arrays["sphere"], arrays["lin"], *self._tree_args(),
                        inst, tri, point, normal, depth)
            if n >= 0:
                break
            room *= 4
        order = np.argsort(-depth[:n], kind="stable")
        return [Contact(self._instances[inst[j]], point[j].copy(), normal[j].copy(), float(depth[j]),
                        int(arrays["face"][tri[j]])) for j in order]

    def overlap_sphere(self, centre, radius, ignore=()):
        """Where a sphere touches or cuts into the objects: Contacts, deepest first, [] if none. One for each way out
        of each object: the faces pushing it out nearly the same way (within about 25 degrees) give one Contact, the
        deepest, so a corner of a wall and a floor gives two and a curved surface a few."""
        return self.overlap_capsule(centre, centre, radius, ignore)

    @_quiet
    def overlap_capsule(self, a, b, radius, ignore=()):
        """Where a capsule (the points within radius of the segment a-b: the usual shape for a character, a and b at
        the centres of its round ends) touches or cuts into the objects: Contacts as overlap_sphere gives them."""
        a, b = _point(a, "a"), _point(b, "b")
        radius = _limit(radius)
        if not (np.isfinite(a).all() and np.isfinite(b).all() and np.isfinite(radius)):
            return []
        return self._overlap(0, a, b, radius, np.eye(3), np.zeros(3), ignore)

    @_quiet
    def overlap_box(self, centre, size, rotation=None, ignore=()):
        """Where a box touches or cuts into the objects: Contacts as overlap_sphere gives them. size: its full size along
        its x, y and z (one number for a cube); rotation: a quaternion (w first) turning it, as Object3D takes.
        A contact's point is the one on the face nearest the box's centre."""
        centre = _point(centre, "centre")
        half = np.broadcast_to(np.asarray(size, dtype=np.float64), (3,)) / 2
        axes = np.eye(3) if rotation is None else quat_to_matrix(np.asarray(rotation, dtype=np.float64)).T
        axes = axes / np.linalg.norm(axes, axis=1, keepdims=True)
        if not (np.isfinite(centre).all() and np.isfinite(half).all() and np.isfinite(axes).all()):
            return []
        return self._overlap(1, centre, centre, 0.0, np.ascontiguousarray(axes), np.ascontiguousarray(np.abs(half)),
                             ignore)

    @_quiet
    def push_out(self, centre, radius, end=None, ignore=(), iterations=4):
        """How far to move a sphere at centre (or a capsule from centre to `end`) to get it out of every face it
        cuts into: a vector (3,), zero if it is clear. Moves out of the deepest contact, then looks again, up to
        `iterations` times (enough for a corner where walls and floor meet)."""
        start = _point(centre, "centre")
        end = start if end is None else _point(end, "end")
        moved, slack = np.zeros(3), 1e-9 * (1.0 + _limit(radius))
        for _ in range(int(min(100.0, max(1.0, float(iterations))))):  # (NaN as 1)
            contacts = self.overlap_capsule(start + moved, end + moved, radius, ignore)
            if not contacts or contacts[0].depth <= 2 * slack:  # (out but for rounding: touching)
                break
            moved += contacts[0].normal * (contacts[0].depth + slack)
        return moved
