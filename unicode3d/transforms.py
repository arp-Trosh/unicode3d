# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
"""Matrix and quaternion helpers.

Conventions: right-handed world, +Y is up, the camera looks down -Z in view
space, and points are column vectors (p' = M @ p). Quaternions are numpy
arrays ordered [w, x, y, z].
"""
import math

import numpy as np
from numba import njit

UP = np.array([0.0, 1.0, 0.0])


def normalize(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def perspective(fov_y, aspect, near, far):
    """OpenGL-style projection matrix. fov_y is in radians. Degenerate settings (near == far, a fov of 0) give
    infinities or NaN in it, which the renderer draws around, rather than an exception."""
    fov_y, aspect, near, far = (np.float64(v) for v in (fov_y, aspect, near, far))  # (numpy's division by zero)
    f = 1.0 / np.tan(fov_y / 2.0)
    m = np.zeros((4, 4))
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2.0 * far * near / (near - far)
    m[3, 2] = -1.0
    return m


def _cross(a, b):
    """np.cross of two 3-vectors, worked out the same way (so to the same bits) in a tenth of the time."""
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]], dtype=float)


def look_at(eye, target, up=UP):
    """View matrix for a camera at `eye` looking at `target`."""
    eye = np.asarray(eye, dtype=float)
    fwd = normalize(np.asarray(target, dtype=float) - eye)
    up = np.asarray(up, dtype=float)
    if np.linalg.norm(_cross(fwd, up)) < 1e-6:
        up = np.array([0.0, 0.0, -1.0])  # looking straight up/down
    right = normalize(_cross(fwd, up))
    true_up = _cross(right, fwd)
    m = np.eye(4)
    m[0, :3] = right
    m[1, :3] = true_up
    m[2, :3] = -fwd
    m[:3, 3] = -m[:3, :3] @ eye
    return m


def view_axes(view, fov_y, aspect):
    """Which way each pixel of a perspective view (`view` from look_at(), and perspective(), fov_y in radians)
    looks: (3, 3) rows a, so that the pixel at (nx, ny) in normalized device coordinates (-1..1, y up) looks along
    a[0] + nx * a[1] + ny * a[2]: straight ahead, and the offsets to the right and top edges of the view one unit
    ahead. Needs no inverse of the view, so a degenerate one gives NaN rather than an exception."""
    tan_y = np.tan(np.float64(fov_y) / 2.0)
    return np.stack([-view[2, :3], view[0, :3] * tan_y * aspect, view[1, :3] * tan_y])


def quat_identity():
    return np.array([1.0, 0.0, 0.0, 0.0])


def quat_axis_angle(axis, angle):
    axis = normalize(axis)
    s = np.sin(angle / 2.0)
    return np.array([np.cos(angle / 2.0), *(axis * s)])


def quat_mul(a, b):
    """Hamilton product: rotating by the result applies b first, then a."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_to_matrix(q):
    w, x, y, z = normalize(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def quat_from_matrix(m):
    """The rotation quaternion (w first) of a 3x3 rotation matrix (orthonormal, determinant 1)."""
    m = np.asarray(m, dtype=float)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = 2.0 * np.sqrt(trace + 1.0)
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(max(1.0 + m[0, 0] - m[1, 1] - m[2, 2], 1e-12))
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(max(1.0 + m[1, 1] - m[0, 0] - m[2, 2], 1e-12))
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = 2.0 * np.sqrt(max(1.0 + m[2, 2] - m[0, 0] - m[1, 1], 1e-12))
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    return normalize(np.array(q))


def quat_slerp(a, b, t):
    """The rotation a fraction t (0..1) of the way from quaternion a to b, turning at a steady rate about one
    axis (spherical linear interpolation), the short way round."""
    a, b = normalize(a), normalize(b)
    d = float(np.dot(a, b))
    if d < 0.0:  # q and -q are the same rotation: go the shorter way
        b, d = -b, -d
    if d > 0.9995:  # nearly the same: a straight line, normalized, is as good and avoids dividing by ~0
        return normalize(a + t * (b - a))
    angle = np.arccos(min(d, 1.0))
    return (np.sin((1.0 - t) * angle) * a + np.sin(t * angle) * b) / np.sin(angle)

def quat_between(u, v):
    """Shortest rotation taking direction u onto direction v."""
    u, v = normalize(u), normalize(v)
    d = float(np.dot(u, v))
    if d < -1.0 + 1e-9:
        axis = np.cross(u, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(u, [0.0, 1.0, 0.0])
        return quat_axis_angle(axis, np.pi)
    return normalize(np.array([1.0 + d, *np.cross(u, v)]))


def scale3(scale):
    """A scale, one number or one per axis (x, y, z), as three numbers (3,)."""
    s = np.asarray(scale, dtype=float)
    if s.shape == ():
        return np.full(3, float(s))
    if s.shape != (3,):
        raise ValueError(f"scale must be a number or three numbers (x, y, z), not {scale!r}")
    return s


def scene_poses(nodes, top=None):
    """The poses in the world of nodes (Nodes and Object3Ds) and all their parents, worked out in one pass:
    (rows, linear (N, 3, 3), position (N, 3), visible (N,) bool), where rows maps id(node) to its row. A point p
    in a node's own space is at linear @ p + position (see Node.world_matrix); a node is visible only if it and
    all its parents are. Each parent is worked out once however many children it has. With top (a node above
    them), poses are in top's own space instead, and top and what is above it are left out."""
    rows, order = {}, []
    for node in nodes:
        chain = []
        while node is not None and node is not top and id(node) not in rows:
            chain.append(node)
            node = node.parent
            if len(chain) > 1000:
                raise ValueError("scene graph has a cycle: an object is its own ancestor")
        for node in reversed(chain):
            rows[id(node)] = len(order)
            order.append(node)
    n = len(order)
    parent = np.array([-1 if node.parent is None or node.parent is top else rows[id(node.parent)] for node in order],
                      np.int64)
    position = np.array([node.position for node in order], np.float64).reshape(n, 3)
    rotation = np.array([node.rotation for node in order], np.float64).reshape(n, 4)
    scales = [(s, s, s) if isinstance(s, (int, float, np.number)) else s for s in (node.scale for node in order)]
    try:
        scale = np.array(scales, np.float64)
    except (TypeError, ValueError):
        scale = None
    if scale is None or scale.shape != (n, 3):  # (something that isn't one number or three: scale3 says what)
        scale = np.array([scale3(node.scale) for node in order], np.float64).reshape(n, 3)
    visible = np.array([bool(node.visible) for node in order], np.bool_)
    linear, place = np.empty((n, 3, 3)), np.empty((n, 3))
    compose_poses(parent, position, rotation, scale, visible, linear, place)
    return rows, linear, place, visible


@njit(cache=True, error_model="numpy")
def compose_poses(parent, position, rotation, scale, visible, linear, place):
    """World poses of nodes listed parents first (scene_poses): parent[i] is the row of node i's parent, or -1 for
    none; position (N, 3), rotation (N, 4, quaternions, w first, normalized here), scale (N, 3) are each node's
    own. Writes linear (N, 3, 3) and place (N, 3), and makes visible[i] false if a parent is hidden. Serial: each
    row needs its parent's, written before it."""
    m = np.empty((3, 3))
    for i in range(len(parent)):
        w, x, y, z = rotation[i, 0], rotation[i, 1], rotation[i, 2], rotation[i, 3]
        n = math.sqrt(w * w + x * x + y * y + z * z)
        if n > 1e-12:  # (as normalize())
            w, x, y, z = w / n, x / n, y / n, z / n
        sx, sy, sz = scale[i, 0], scale[i, 1], scale[i, 2]
        m[0, 0], m[0, 1], m[0, 2] = (1 - 2 * (y * y + z * z)) * sx, 2 * (x * y - w * z) * sy, 2 * (x * z + w * y) * sz
        m[1, 0], m[1, 1], m[1, 2] = 2 * (x * y + w * z) * sx, (1 - 2 * (x * x + z * z)) * sy, 2 * (y * z - w * x) * sz
        m[2, 0], m[2, 1], m[2, 2] = 2 * (x * z - w * y) * sx, 2 * (y * z + w * x) * sy, (1 - 2 * (x * x + y * y)) * sz
        p = parent[i]
        if p < 0:
            for r in range(3):
                place[i, r] = position[i, r]
                for c in range(3):
                    linear[i, r, c] = m[r, c]
            continue
        visible[i] = visible[i] and visible[p]
        for r in range(3):
            place[i, r] = place[p, r] + (linear[p, r, 0] * position[i, 0] + linear[p, r, 1] * position[i, 1]
                                         + linear[p, r, 2] * position[i, 2])
            for c in range(3):
                linear[i, r, c] = linear[p, r, 0] * m[0, c] + linear[p, r, 1] * m[1, c] + linear[p, r, 2] * m[2, c]


@njit(cache=True, error_model="numpy")
def place_boxes(vertices, first, count, row, linear, place, lo, hi):
    """The box (lo, hi: (K, 3)) around each of K runs of vertices (count[k] of them from first[k] in vertices
    (V, 3)) placed by pose row[k] of linear (N, 3, 3) and place (N, 3): p -> linear @ p + place. A coordinate
    that comes out NaN counts as infinite, so that the box isn't finite either."""
    for k in range(len(first)):
        r = row[k]
        for a in range(3):
            lo[k, a] = np.inf
            hi[k, a] = -np.inf
        for i in range(first[k], first[k] + count[k]):
            for a in range(3):
                x = (linear[r, a, 0] * vertices[i, 0] + linear[r, a, 1] * vertices[i, 1]
                     + linear[r, a, 2] * vertices[i, 2]) + place[r, a]
                if x != x:
                    lo[k, a] = -np.inf
                    hi[k, a] = np.inf
                if x < lo[k, a]:
                    lo[k, a] = x
                if x > hi[k, a]:
                    hi[k, a] = x


def world_matrix(node):
    """(linear (3, 3), position (3,), visible) of a node in the world, through all its parents (see
    Node.world_matrix). For many nodes at once, scene_poses() works out each shared parent once."""
    rows, linear, place, visible = scene_poses([node])
    row = rows[id(node)]
    return linear[row], place[row], bool(visible[row])
