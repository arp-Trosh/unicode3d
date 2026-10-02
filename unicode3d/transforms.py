# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
"""Matrix and quaternion helpers.

Conventions: right-handed world, +Y is up, the camera looks down -Z in view
space, and points are column vectors (p' = M @ p). Quaternions are numpy
arrays ordered [w, x, y, z].
"""
import numpy as np

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
