# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
"""Particles: many small copies of one mesh (sparks, smoke, dust, debris, rain), kept in arrays and moved in a
kernel, drawn by passing the Particles to Renderer.render() among the objects. Each particle is drawn as an
instance of the mesh, as an Object3D would be, but none of them costs Python work a frame."""
import math

import numpy as np
from numba import njit

from .color import to_linear_rgb
from .mesh import Mesh
from .shapes import blob_mesh
from .transforms import quat_identity

BALL = blob_mesh((0.5, 0.5, 0.5), rings=3, segments=4)  # (1 across: `size` is how wide a particle is; 16 faces)


def _disk(sides=8):
    """A flat polygon 1 across in the xy plane, facing +z (towards the camera, with face_camera)."""
    a = 2.0 * np.pi * (np.arange(sides) + 0.5) / sides
    vertices = np.r_[[(0.0, 0.0, 0.0)], np.c_[0.5 * np.cos(a), 0.5 * np.sin(a), np.zeros(sides)]]
    faces = np.array([(0, 1 + k, 1 + (k + 1) % sides) for k in range(sides)])
    return Mesh(vertices, faces)


DISK = _disk()


def _range(value, name):
    """A number, or (low, high) to pick each particle's at random between, as (low, high) floats."""
    a = np.asarray(value, dtype=np.float64).reshape(-1)
    if a.shape == (1,):
        return float(a[0]), float(a[0])
    if a.shape == (2,):
        return float(a[0]), float(a[1])
    raise ValueError(f"{name} must be a number or (low, high), not {value!r}")


class Particles:
    """Many small copies of one mesh, each with its own place, velocity, age, size, colour and opacity, moved by
    update() in one kernel and drawn by passing this to Renderer.render() among the objects (each particle drawn
    as an instance of the mesh). A burst of sparks or a column of smoke is one Particles, however many particles
    it holds:

        sparks = Particles(gravity=(0, -9.8, 0), emissive=1.0)
        sparks.emit(at, count=30, speed=(1.0, 3.0), life=(0.4, 0.7), size=0.06, color=(255, 120, 60))
        ...
        sparks.update(dt)                      # each frame: move them on, drop those whose life is over
        renderer.render(objects + [sparks], camera, lights)

    mesh: what each particle is (1 across is `size` across); by default a small ball, or with face_camera a flat
        disk. Any mesh works: a box for chips of stone, a textured square for sprites (cut-out alpha included).
    face_camera: turn each particle to face the camera (the mesh's +z towards it, its +y up the picture): flat
        sprites, smoke puffs, glints. `spin` then turns it in the picture's plane.
    gravity: acceleration (3,), in units a second a second. drag: the share of speed lost a second (0: none).
    ground: a height (y) the particles bounce on and come to rest on, or None: a particle's centre stays half its
        size above it (the underside of a mesh 1 across). bounce is the share of speed kept upwards, friction
        along the ground.
    limit: the most particles held at once; beyond it, emit() makes room by dropping the earliest emitted.
    seed: for the random numbers emit() draws (the same seed and calls give the same particles).

    Like an Object3D, a Particles has position, rotation, scale, visible and parent: particles live in its own
    space (the world's, by default), so a Particles with a parent moves with it. emissive, cast_shadows (False
    by default: a thousand sparks in the shadow maps cost more than they show), opacity (times each particle's),
    color (times each particle's), specular, shininess, double_sided and outline work as on an Object3D (outline is
    off by default: dark edges would ring every spark and give smoke a hard rim). pick() on a
    particle gives the Particles. Ray and overlap queries (Colliders) and union_bounds() leave Particles out.

    The live particles' arrays are open to programs moving them their own way: positions, velocities, ages and
    lives, each a view of the first len(self) rows, to change in place (they change with emit() and update()).
    """

    billboard = False  # (Object3D's: a Particles faces the camera with face_camera)
    reflectivity = 0.0
    simplify = False

    def __init__(self, mesh=None, face_camera=False, gravity=(0.0, 0.0, 0.0), drag=0.0, ground=None, bounce=0.3,
                 friction=0.6, limit=10000, emissive=0.0, cast_shadows=False, opacity=1.0, color=(255, 255, 255),
                 specular=0.0, shininess=None, double_sided=False, outline=False, seed=None, position=(0.0, 0.0, 0.0),
                 rotation=None, scale=1.0, visible=True, parent=None):
        self.face_camera = bool(face_camera)
        self.mesh = mesh if mesh is not None else (DISK if self.face_camera else BALL)
        self.gravity, self.drag, self.ground = gravity, drag, ground
        self.bounce, self.friction = bounce, friction
        self.emissive, self.cast_shadows, self.opacity, self.color = emissive, cast_shadows, opacity, color
        self.specular, self.shininess, self.double_sided = specular, shininess, double_sided
        self.outline = outline
        self.position = np.array(position, dtype=float)
        self.rotation = quat_identity() if rotation is None else np.array(rotation, dtype=float)
        self.scale, self.visible, self.parent = scale, visible, parent
        self.rng = np.random.default_rng(seed)
        self.limit = max(int(limit), 0)
        n = self.limit
        self._count = 0
        self._pos, self._vel = np.zeros((n, 3)), np.zeros((n, 3))
        self._age, self._life = np.zeros(n), np.zeros(n)
        self._size = np.zeros((n, 2))       # at birth, at the end of life
        self._rgb = np.zeros((n, 2, 3))     # linear light, at birth and at the end
        self._alpha = np.zeros((n, 2))
        self._axis = np.zeros((n, 3))       # what each turns about (unit), how far it has turned, how fast
        self._angle, self._spin = np.zeros(n), np.zeros(n)

    def __len__(self):
        return self._count

    def __repr__(self):
        return f"Particles({self._count} of {self.limit}, mesh with {len(self.mesh.faces)} faces)"

    @property
    def positions(self):
        return self._pos[:self._count]

    @property
    def velocities(self):
        return self._vel[:self._count]

    @property
    def ages(self):
        return self._age[:self._count]

    @property
    def lives(self):
        return self._life[:self._count]

    def world_matrix(self):
        """(linear (3, 3), position (3,), visible) of the Particles' own space in the world (see Node)."""
        from .transforms import world_matrix
        return world_matrix(self)

    def emit(self, at, count=1, velocity=(0.0, 0.0, 0.0), speed=0.0, radius=0.0, life=1.0, size=0.1,
             size_end=None, color=(255, 255, 255), color_end=None, opacity=1.0, opacity_end=None, spin=0.0,
             angle=None):
        """Add count particles (rounded down) at `at` (a point in the Particles' space), spread through a ball of `radius`.
        Returns how many were added (all of them, up to limit).

        Each moves at velocity (3,) plus `speed` in a random direction. size, its width, goes to size_end over
        its life (None: stays), as color does to color_end and opacity to opacity_end: a spark dimming, smoke
        growing and thinning. spin is how fast it turns (radians a second) about a random axis (with
        face_camera, in the picture's plane), from `angle` (None: a random one; 0 with face_camera).
        speed, radius, life, size, size_end and spin may each be (low, high), each particle drawing its own at
        random between them."""
        with np.errstate(all="ignore"):  # (bad numbers make particles left out of the picture, not warnings over it)
            return self._emit(at, count, velocity, speed, radius, life, size, size_end, color, color_end, opacity,
                              opacity_end, spin, angle)

    def _emit(self, at, count, velocity, speed, radius, life, size, size_end, color, color_end, opacity, opacity_end,
              spin, angle):
        count = int(min(max(count, 0), self.limit)) if count == count else 0
        if count == 0:
            return 0
        rng = self.rng

        def pick(value, name):  # (low + spread * random: no exception for a spread beyond floats, as uniform() has)
            low, high = _range(value, name)
            return low + (high - low) * rng.random(count)
        free = self.limit - self._count
        if count > free:  # (room made by dropping the earliest emitted)
            self._drop_first(count - free)
        k, n = self._count, self._count + count
        at = np.asarray(at, dtype=np.float64).reshape(3)
        direction = rng.normal(size=(count, 3))
        length = np.sqrt((direction * direction).sum(axis=1))[:, None]
        direction = direction / np.where(length > 0.0, length, 1.0)
        ball = rng.normal(size=(count, 3))
        ball *= (pick(radius, "radius") * rng.random(count) ** (1.0 / 3.0)
                 / np.maximum(np.sqrt((ball * ball).sum(axis=1)), 1e-300))[:, None]
        self._pos[k:n] = at + ball
        self._vel[k:n] = np.asarray(velocity, dtype=np.float64).reshape(3) + direction * pick(speed, "speed")[:, None]
        self._age[k:n] = 0.0
        self._life[k:n] = pick(life, "life")
        self._size[k:n, 0] = pick(size, "size")
        self._size[k:n, 1] = self._size[k:n, 0] if size_end is None else pick(size_end, "size_end")
        start = to_linear_rgb(color)
        self._rgb[k:n, 0] = start
        self._rgb[k:n, 1] = start if color_end is None else to_linear_rgb(color_end)
        self._alpha[k:n, 0] = float(opacity)
        self._alpha[k:n, 1] = float(opacity if opacity_end is None else opacity_end)
        axis = rng.normal(size=(count, 3))
        self._axis[k:n] = axis / np.maximum(np.sqrt((axis * axis).sum(axis=1)), 1e-300)[:, None]
        if angle is None:
            self._angle[k:n] = 0.0 if self.face_camera else rng.uniform(0.0, 2.0 * np.pi, count)
        else:
            self._angle[k:n] = float(angle)
        self._spin[k:n] = pick(spin, "spin")
        self._count = n
        return count

    def _drop_first(self, k):
        n = self._count
        for a in (self._pos, self._vel, self._age, self._life, self._size, self._rgb, self._alpha, self._axis,
                  self._angle, self._spin):
            a[:n - k] = a[k:n]
        self._count = n - k

    def clear(self):
        """Drop every particle."""
        self._count = 0

    def update(self, dt):
        """Move the particles on by dt seconds (gravity, drag, the ground), turn them, and drop those whose life
        is over."""
        dt = float(dt)
        dt = dt if 0.0 < dt < math.inf else 0.0
        gravity = np.nan_to_num(np.asarray(self.gravity, dtype=np.float64).reshape(3), posinf=0.0, neginf=0.0)
        number = lambda x: float(x) if math.isfinite(float(x)) else 0.0
        ground = math.nan if self.ground is None else float(self.ground)  # (NaN: no ground, as NaN compares false)
        self._count = step(self._count, dt, self._pos, self._vel, self._age, self._life, self._size, self._rgb,
                           self._alpha, self._axis, self._angle, self._spin, gravity,
                           min(max(number(self.drag), 0.0), 1e300), ground, number(self.bounce),
                           number(self.friction))

    def _rows(self, linear, position, view):
        """Each live particle drawn: (count, its pose (linear (n, 3, 3) and position (n, 3)) in the world, linear
        rgb (n, 3), opacity (n,)), from the Particles' own pose in the world (linear, position) and the camera's
        view (see Lines._rows), of which only the (right, up, back) rows (3, 3) of its basis count here.
        Particles that aren't finite, or wholly clear, are left out."""
        n, basis = self._count, view[0]
        lin, pos = np.empty((n, 3, 3)), np.empty((n, 3))
        rgb, alpha = np.empty((n, 3)), np.empty(n)
        kept = particle_rows(n, self._pos, self._age, self._life, self._size, self._rgb, self._alpha, self._axis,
                             self._angle, self.face_camera, basis, np.ascontiguousarray(linear, np.float64),
                             np.ascontiguousarray(position, np.float64), lin, pos, rgb, alpha)
        return kept, lin, pos, rgb, alpha


@njit(cache=True, error_model="numpy")
def step(n, dt, pos, vel, age, life, size, rgb, alpha, axis, angle, spin, gravity, drag, ground, bounce, friction):
    """Move particles 0..n on by dt (see Particles.update) and keep only those still alive, in their order, at the
    front of the arrays. Returns how many are kept. Serial: each one moves a row towards the front."""
    keep = max(0.0, 1.0 - drag * dt)
    w = 0
    for i in range(n):
        a = age[i] + dt
        if not (a < life[i]):  # (over, or a NaN life)
            continue
        for c in range(3):
            vel[i, c] = (vel[i, c] + gravity[c] * dt) * keep
            pos[i, c] += vel[i, c] * dt
        s = spin[i]
        u = min(1.0, max(0.0, a / life[i]))
        floor = ground + 0.5 * abs(size[i, 0] + (size[i, 1] - size[i, 0]) * u)  # (its underside on the ground)
        if pos[i, 1] < floor and vel[i, 1] < 0.0:  # (false for no ground: NaN)
            pos[i, 1] = floor
            vel[i, 1] *= -bounce
            vel[i, 0] *= friction
            vel[i, 2] *= friction
            s *= 0.5
        if w != i:
            for c in range(3):
                pos[w, c] = pos[i, c]
                vel[w, c] = vel[i, c]
                axis[w, c] = axis[i, c]
                for e in range(2):
                    rgb[w, e, c] = rgb[i, e, c]
            for e in range(2):
                size[w, e] = size[i, e]
                alpha[w, e] = alpha[i, e]
            life[w] = life[i]
        age[w] = a
        spin[w] = s
        angle[w] = angle[i] + s * dt
        w += 1
    return w


@njit(cache=True, error_model="numpy")
def particle_rows(n, pos, age, life, size, rgb, alpha, axis, angle, face, basis, linear, position, out_lin,
                  out_pos, out_rgb, out_alpha):
    """Each of particles 0..n as an instance (see Particles._rows): its size, colour and opacity at its age, its
    turn, and its pose through the Particles' own (linear, position). Writes the kept ones to the front of the
    out_ arrays and returns how many. A face-on particle's axes are the camera's, times the Particles' size (the
    cube root of linear's determinant), whatever linear's turn."""
    d = (linear[0, 0] * (linear[1, 1] * linear[2, 2] - linear[1, 2] * linear[2, 1])
         - linear[0, 1] * (linear[1, 0] * linear[2, 2] - linear[1, 2] * linear[2, 0])
         + linear[0, 2] * (linear[1, 0] * linear[2, 1] - linear[1, 1] * linear[2, 0]))
    grow = abs(d) ** (1.0 / 3.0)
    turn = np.empty((3, 3))
    w = 0
    for i in range(n):
        u = age[i] / life[i]
        u = min(1.0, max(0.0, u))
        s = size[i, 0] + (size[i, 1] - size[i, 0]) * u
        a = alpha[i, 0] + (alpha[i, 1] - alpha[i, 0]) * u
        a = min(1.0, max(0.0, a))
        if not (a > 0.0) or not (abs(s) < 1e300):
            continue
        c, sn = math.cos(angle[i]), math.sin(angle[i])
        if face:  # columns: right and up turned by the angle in the picture's plane, and back (towards the camera)
            for r in range(3):
                turn[r, 0] = (c * basis[0, r] + sn * basis[1, r]) * s * grow
                turn[r, 1] = (c * basis[1, r] - sn * basis[0, r]) * s * grow
                turn[r, 2] = basis[2, r] * s * grow
        else:  # the turn about its axis (Rodrigues), times the size, then the Particles' own pose
            x, y, z = axis[i, 0], axis[i, 1], axis[i, 2]
            t = 1.0 - c
            m00, m01, m02 = c + x * x * t, x * y * t - z * sn, x * z * t + y * sn
            m10, m11, m12 = y * x * t + z * sn, c + y * y * t, y * z * t - x * sn
            m20, m21, m22 = z * x * t - y * sn, z * y * t + x * sn, c + z * z * t
            for r in range(3):
                l0, l1, l2 = linear[r, 0], linear[r, 1], linear[r, 2]
                turn[r, 0] = (l0 * m00 + l1 * m10 + l2 * m20) * s
                turn[r, 1] = (l0 * m01 + l1 * m11 + l2 * m21) * s
                turn[r, 2] = (l0 * m02 + l1 * m12 + l2 * m22) * s
        finite = True
        for r in range(3):
            p = linear[r, 0] * pos[i, 0] + linear[r, 1] * pos[i, 1] + linear[r, 2] * pos[i, 2] + position[r]
            out_pos[w, r] = p
            finite = finite and abs(p) < 1e300
            for k in range(3):
                out_lin[w, r, k] = turn[r, k]
                finite = finite and abs(turn[r, k]) < 1e300
        if not finite:  # (NaN or infinite somewhere: left out, as objects with such poses are)
            continue
        for k in range(3):
            out_rgb[w, k] = max(0.0, rgb[i, 0, k] + (rgb[i, 1, k] - rgb[i, 0, k]) * u)
        out_alpha[w] = a
        w += 1
    return w
