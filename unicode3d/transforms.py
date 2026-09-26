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
    """OpenGL-style projection matrix. fov_y is in radians."""
    f = 1.0 / np.tan(fov_y / 2.0)
    m = np.zeros((4, 4))
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2.0 * far * near / (near - far)
    m[3, 2] = -1.0
    return m


def look_at(eye, target, up=UP):
    """View matrix for a camera at `eye` looking at `target`."""
    eye = np.asarray(eye, dtype=float)
    fwd = normalize(np.asarray(target, dtype=float) - eye)
    if np.linalg.norm(np.cross(fwd, up)) < 1e-6:
        up = np.array([0.0, 0.0, -1.0])  # looking straight up/down
    right = normalize(np.cross(fwd, up))
    true_up = np.cross(right, fwd)
    m = np.eye(4)
    m[0, :3] = right
    m[1, :3] = true_up
    m[2, :3] = -fwd
    m[:3, 3] = -m[:3, :3] @ eye
    return m


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
