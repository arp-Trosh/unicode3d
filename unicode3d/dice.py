"""Dice meshes and roll animation."""
import numpy as np

from .mesh import BOX_FACE_NORMALS, make_box
from .transforms import UP, normalize, quat_axis_angle, quat_between, quat_mul, quat_to_matrix

# Pip value on each box face, in BOX_FACES order (+X, -X, +Y, -Y, +Z, -Z); opposite faces sum to 7.
DIE_VALUES = (3, 4, 2, 5, 1, 6)

PIP_LAYOUTS = {
    1: [(0.5, 0.5)],
    2: [(0.25, 0.75), (0.75, 0.25)],
    3: [(0.25, 0.75), (0.5, 0.5), (0.75, 0.25)],
    4: [(0.25, 0.25), (0.25, 0.75), (0.75, 0.25), (0.75, 0.75)],
    5: [(0.25, 0.25), (0.25, 0.75), (0.5, 0.5), (0.75, 0.25), (0.75, 0.75)],
    6: [(0.25, 0.25), (0.25, 0.5), (0.25, 0.75), (0.75, 0.25), (0.75, 0.5), (0.75, 0.75)],
}


def face_texture(res=48, border=0.07, border_shade=0.55):
    """Blank face texture with a darker rim so the cube's edges read clearly.

    Returns the texture plus the (u, v) coordinates of each texel for drawing on it.
    """
    c = (np.arange(res) + 0.5) / res
    u, v = np.meshgrid(c, 1.0 - c)  # texture row 0 is the top of the face (v = 1)
    tex = np.ones((res, res))
    tex[np.minimum.reduce([u, 1 - u, v, 1 - v]) < border] = border_shade
    return tex, u, v


def pip_texture(value, res=48, radius=0.12, pip_shade=0.0):
    tex, u, v = face_texture(res)
    for pu, pv in PIP_LAYOUTS[value]:
        tex[(u - pu) ** 2 + (v - pv) ** 2 < radius ** 2] = pip_shade
    return tex


def make_die(size=1.0):
    """Standard six-sided die; face i shows DIE_VALUES[i]."""
    return make_box(size, [pip_texture(v) for v in DIE_VALUES])


def orientation_showing(face, yaw=0.0):
    """Rotation that turns box face `face` to point straight up, then spins it by `yaw` around +Y."""
    return quat_mul(quat_axis_angle(UP, yaw), quat_between(BOX_FACE_NORMALS[face], UP))


def top_face(rotation):
    """Index of the box face currently pointing most nearly up."""
    local_up = quat_to_matrix(rotation).T @ UP
    return int(np.argmax(BOX_FACE_NORMALS @ local_up))


class RollAnimation:
    """A die dropped from above that bounces, tumbles, and settles showing `face`.

    The result is picked up front; the animation only makes it look random.
    `rest_position` is the die's centre once settled (y = half its size to sit on y = 0).
    """

    def __init__(self, face, rest_position, rng=None, drop_height=5.0, gravity=30.0,
                 restitution=0.35, turns=(1.0, 2.0), scatter=1.0):
        rng = rng if rng is not None else np.random.default_rng()
        self.face = face
        self.final_rotation = orientation_showing(face, rng.uniform(0, 2 * np.pi))
        self.rest = np.asarray(rest_position, dtype=float)
        self.start = self.rest + [rng.uniform(-scatter, scatter), 0.0, -scatter - rng.uniform(0, scatter)]
        self.spin_axis = normalize(rng.normal(size=3))
        self.spin_angle = 2 * np.pi * rng.uniform(*turns)

        # Pre-simulate the drop so the tumble can be timed to stop exactly at rest.
        times, heights = [0.0], [drop_height]
        t, y, vy, dt = 0.0, drop_height, 0.0, 1.0 / 240
        while True:
            vy -= gravity * dt
            y += vy * dt
            t += dt
            if y <= 0.0:
                y, vy = 0.0, -vy * restitution
                if vy < 1.0:
                    times.append(t)
                    heights.append(0.0)
                    break
            times.append(t)
            heights.append(y)
        self.times, self.heights = np.array(times), np.array(heights)
        self.duration = t

    def done(self, t):
        return t >= self.duration

    def pose(self, t):
        """(position, rotation) at time t seconds after the roll started."""
        t = min(max(t, 0.0), self.duration)
        p = t / self.duration
        slide = 1.0 - (1.0 - p) ** 2
        position = self.start + (self.rest - self.start) * slide
        position[1] = self.rest[1] + np.interp(t, self.times, self.heights)
        spin = quat_axis_angle(self.spin_axis, self.spin_angle * (1.0 - p) ** 2)
        return position, quat_mul(spin, self.final_rotation)
