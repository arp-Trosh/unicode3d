# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
"""Lines: straight segments (grids, axes, laser beams, paths, debug drawing), each drawn as a flat strip turned
to face the camera, as an instance of one small square, through the same instance rows Particles use."""
import math

import numpy as np
from numba import njit

from .mesh import Mesh
from .transforms import quat_identity

# Each segment is drawn as one of these strips, stretched along it (y from 0 to 1), as wide as the line across it
# (x), and turned to face the camera (z): STRIPS[k] is 1 wide at y = 0 and TAPER ** k wide at y = 1. A line a
# number of pixels wide is that much wider in the world at its far end as that end is further off (the width it
# needs grows in step with distance), so a strip tapered as much draws it exactly; the one taken is at most TAPER
# times too wide at its near end.
TAPER = 1.1
N_STRIPS = 64
STRIPS = [Mesh(np.array([(-0.5, 0.0, 0.0), (0.5, 0.0, 0.0), (0.5 * r, 1.0, 0.0), (-0.5 * r, 1.0, 0.0)]),
               np.array([(0, 1, 2), (0, 2, 3)])) for r in TAPER ** np.arange(N_STRIPS)]
MOST_TAPER = TAPER ** (N_STRIPS - 1)  # (a segment one end of which is further off than that: in pieces)


def _segments(points, name):
    a = np.asarray(points, dtype=np.float64)
    if a.size == 0:
        return np.zeros((0, 3))
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(f"{name} must be (n, 3) points, not of shape {a.shape}")
    return a


class Lines:
    """Straight segments from starts[i] to ends[i], each drawn as a flat strip facing the camera, `width` wide:
    a grid, axes, a laser beam, the path a unit will take, a ray query's hits. All the segments are one object,
    however many there are, drawn by passing this to Renderer.render() among the objects:

        grid = Lines(starts, ends, color=(90, 90, 90))         # 1 pixel wide, wherever it is
        beam = Lines([gun], [target], width=0.05, units="world", color=(255, 40, 40))
        renderer.render(objects + [grid, beam], camera, lights)

    starts, ends: (n, 3) points, in the Lines' own space (the world's, by default). colors: None (each the Lines'
        `color`), or (n, 3) 0..255 a segment, times `color`.
    width: how wide each line is, in pixels of the picture (units="pixels", the default: the same however far
        off, for grids, axes and debug drawing) or in units of the world (units="world": thinner further off, for
        beams and wires). A line in world units is still drawn at least min_pixels wide, so far-off lines don't
        break up into dashes (0: as thin as the world says).

    starts, ends and colors are open to change between frames (assign new arrays, or change them in place), or
    set() them all at once. emissive is 1 by default: lines show their colour as it is, unshaded (0: lit as a
    strip facing the camera). Like an Object3D, a Lines has position, rotation, scale, visible and parent, and
    opacity, color, cast_shadows (False by default), double_sided and outline (False by default). pick() on a
    line gives the Lines. Ray and overlap queries (Colliders) and union_bounds() leave Lines out.
    """

    mesh = STRIPS[0]  # (and the others, as each segment needs)
    billboard = False  # (Object3D's: each strip faces the camera of itself)
    reflectivity = 0.0
    simplify = False
    skin = None
    specular = 0.0
    shininess = None

    def __init__(self, starts=(), ends=(), colors=None, color=(255, 255, 255), width=1.0, units="pixels",
                 min_pixels=1.0, emissive=1.0, opacity=1.0, cast_shadows=False, double_sided=True, outline=False,
                 position=(0.0, 0.0, 0.0), rotation=None, scale=1.0, visible=True, parent=None):
        if units not in ("pixels", "world"):
            raise ValueError(f'units must be "pixels" or "world", not {units!r}')
        self.set(starts, ends, colors)
        self.color, self.width, self.units, self.min_pixels = color, width, units, min_pixels
        self.emissive, self.opacity, self.cast_shadows = emissive, opacity, cast_shadows
        self.double_sided, self.outline = double_sided, outline
        self.position = np.array(position, dtype=float)
        self.rotation = quat_identity() if rotation is None else np.array(rotation, dtype=float)
        self.scale, self.visible, self.parent = scale, visible, parent
        self._decoded = (None, None)

    def set(self, starts, ends, colors=None):
        """Replace the segments: starts and ends (n, 3), colors None or (n, 3) 0..255."""
        self.starts, self.ends = _segments(starts, "starts"), _segments(ends, "ends")
        if len(self.starts) != len(self.ends):
            raise ValueError(f"{len(self.starts)} starts but {len(self.ends)} ends")
        self.colors = None if colors is None else np.asarray(colors)

    def __len__(self):
        return min(len(self.starts), len(self.ends))

    def __repr__(self):
        return f"Lines({len(self)} segments, {self.width} {self.units} wide)"

    def world_matrix(self):
        """(linear (3, 3), position (3,), visible) of the Lines' own space in the world (see Node)."""
        from .transforms import world_matrix
        return world_matrix(self)

    def _linear_colors(self, n):
        """Each segment's colour, linear (n, 3); ones without colors. Remembered while colors stays the same."""
        if self.colors is None:
            return np.ones((n, 3))
        c = np.asarray(self.colors, dtype=np.float64)
        key = c.tobytes()
        if self._decoded[0] != key:
            s = np.nan_to_num(np.broadcast_to(c, (len(c), 3)) / 255.0, posinf=0.0, neginf=0.0).clip(0.0, 1.0)
            self._decoded = (key, np.where(s <= 0.04045, s / 12.92, ((s + 0.055) / 1.055) ** 2.4))
        rgb = self._decoded[1]
        if len(rgb) < n:
            raise ValueError(f"{len(rgb)} colors for {n} segments")
        return rgb[:n]

    def _rows(self, linear, position, view):
        """Each segment drawn, as one or more pieces: (count, each piece's pose (linear (n, 3, 3) and position
        (n, 3)) in the world, linear rgb (n, 3), opacity (n,), which of STRIPS it is (n,)), from the Lines' own pose
        in the world (linear, position) and the camera's view: (basis (rows right, up, back), eye, near, the width
        in the world of a pixel 1 unit ahead)."""
        n = len(self)
        basis, eye, near, pixel = view
        starts = np.ascontiguousarray(_segments(self.starts, "starts")[:n], np.float64)  # (checked: (n, 3) each
        ends = np.ascontiguousarray(_segments(self.ends, "ends")[:n], np.float64)        # frame, as kernels index)
        rgb = self._linear_colors(n)
        number = lambda x: float(x) if math.isfinite(float(x)) else 0.0
        width = max(number(self.width), 0.0)
        world, pixels = (width, max(number(self.min_pixels), 0.0)) if self.units == "world" else (0.0, width)
        args = (n, starts, ends, np.ascontiguousarray(linear, np.float64), np.ascontiguousarray(position, np.float64),
                np.ascontiguousarray(eye, np.float64), np.ascontiguousarray(-basis[2], np.float64), number(near),
                pixels * number(pixel), world)
        pieces = np.empty(n, np.int64)
        total = line_pieces(*args, pieces)
        lin, pos = np.empty((total, 3, 3)), np.empty((total, 3))
        seg, strip = np.empty(total, np.int64), np.empty(total, np.int64)
        kept = line_rows(*args, pieces, lin, pos, seg, strip)
        return kept, lin, pos, rgb[seg[:kept]], np.ones(kept), strip


@njit(cache=True, error_model="numpy")
def _span(i, starts, ends, linear, position, eye, forward, near, a, b):
    """Segment i in the world, from its near end (into a) to its far end (b), cut where it passes behind the
    near plane; returns how far ahead of the eye each end is, (0, 0) if it is all behind (or not finite)."""
    za, zb = 0.0, 0.0
    for r in range(3):
        a[r] = linear[r, 0] * starts[i, 0] + linear[r, 1] * starts[i, 1] + linear[r, 2] * starts[i, 2] + position[r]
        b[r] = linear[r, 0] * ends[i, 0] + linear[r, 1] * ends[i, 1] + linear[r, 2] * ends[i, 2] + position[r]
        za += (a[r] - eye[r]) * forward[r]
        zb += (b[r] - eye[r]) * forward[r]
    if not (near > 0.0) or not (max(za, zb) >= near) or not (abs(za) < 1e300 and abs(zb) < 1e300):
        return 0.0, 0.0
    if zb < za:  # (the near end first)
        for r in range(3):
            a[r], b[r] = b[r], a[r]
        za, zb = zb, za
    if za < near:  # (cut at the near plane)
        t = (near - za) / (zb - za)
        for r in range(3):
            a[r] += (b[r] - a[r]) * t
        za = near
    return za, zb


@njit(cache=True, error_model="numpy")
def _scale(linear):
    """How much linear stretches things, on average: the cube root of its determinant (its size)."""
    d = (linear[0, 0] * (linear[1, 1] * linear[2, 2] - linear[1, 2] * linear[2, 1])
         - linear[0, 1] * (linear[1, 0] * linear[2, 2] - linear[1, 2] * linear[2, 0])
         + linear[0, 2] * (linear[1, 0] * linear[2, 1] - linear[1, 1] * linear[2, 0]))
    return abs(d) ** (1.0 / 3.0)


@njit(cache=True, error_model="numpy")
def _cuts(za, zb, world, grow):
    """How a segment from depth za to zb (za <= zb) is drawn, the width it needs at depth z being
    max(world, grow * z): (where a part of the constant width `world` ends (a depth: za for none), how many
    tapered pieces follow from there to zb)."""
    if not (grow * zb > world):  # (the world width all along: one strip of it)
        return zb, 0
    z = min(max(world / grow, za), zb) if grow > 0.0 else za  # (nearer than z, the world width is wider)
    if not (zb > z):  # (the same distance all along: one strip as wide as it needs there)
        return z, 1
    return z, int(max(1.0, min(1e6, math.ceil(math.log(zb / z) / math.log(MOST_TAPER) - 1e-9))))


@njit(cache=True, error_model="numpy")
def line_pieces(n, starts, ends, linear, position, eye, forward, near, grow, world, pieces):
    """How many pieces each of segments 0..n is drawn in (into pieces (n,)); returns their sum. See line_rows."""
    world = world * _scale(linear)
    a, b = np.empty(3), np.empty(3)
    total = 0
    for i in range(n):
        za, zb = _span(i, starts, ends, linear, position, eye, forward, near, a, b)
        k = 0
        if za > 0.0:
            z, tapered = _cuts(za, zb, world, grow)
            k = tapered + (1 if z > za or tapered == 0 else 0)
        pieces[i] = k
        total += k
    return total


@njit(cache=True, error_model="numpy")
def line_rows(n, starts, ends, linear, position, eye, forward, near, grow, world, pieces, out_lin, out_pos,
              out_seg, out_strip):
    """Segments 0..n as instances of STRIPS (see Lines._rows; how many pieces each from line_pieces): the width
    a segment needs at depth z ahead of the eye is max(world (times the Lines' size), grow * z) (grow: the width
    of `pixels` pixels 1 unit ahead), so from its near end it is a strip of the world width as far as that is the
    wider, then pieces tapered as they go further off (each at most MOST_TAPER times as far off at its far end).
    Each piece runs from one end to the other along y, its width along x is across the segment and the way to
    the eye, and its face (z) is towards the eye. Writes the kept pieces (finite, not seen end on, of some length
    and width) to the front of the out_ arrays, with the segment each belongs to and its strip, and returns how
    many."""
    world = world * _scale(linear)
    a, b, p, q = np.empty(3), np.empty(3), np.empty(3), np.empty(3)
    w = 0
    for i in range(n):
        if pieces[i] == 0:
            continue
        za, zb = _span(i, starts, ends, linear, position, eye, forward, near, a, b)
        z, tapered = _cuts(za, zb, world, grow)
        # Across the segment and the way to the eye (the same anywhere along it), and facing the eye.
        dx, dy, dz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
        vx, vy, vz = eye[0] - a[0], eye[1] - a[1], eye[2] - a[2]
        sx, sy, sz = dy * vz - dz * vy, dz * vx - dx * vz, dx * vy - dy * vx
        ls = math.sqrt(sx * sx + sy * sy + sz * sz)
        ld = math.sqrt(dx * dx + dy * dy + dz * dz)
        if not (ls > 1e-12 * ld * math.sqrt(vx * vx + vy * vy + vz * vz)) or not (ld < 1e300):
            continue  # (seen end on: a dot at most; of no length; or not finite)
        sx, sy, sz = sx / ls, sy / ls, sz / ls
        nx, ny, nz = (sy * dz - sz * dy) / ld, (sz * dx - sx * dz) / ld, (sx * dy - sy * dx) / ld
        constant = z > za or tapered == 0  # (a first piece of the world width)
        for j in range(pieces[i]):
            # This piece: from depth z0 to z1, and which strip with what width at its near end.
            if constant and j == 0:
                z0, z1, k, near_width = za, z, 0, world
            else:
                m = j - (1 if constant else 0)
                z0 = z * (zb / z) ** (m / tapered)
                z1 = zb if m == tapered - 1 else z * (zb / z) ** ((m + 1) / tapered)
                k = int(min(N_STRIPS - 1.0, max(0.0, math.floor(math.log(z1 / z0) / math.log(TAPER) + 1e-9))))
                near_width = grow * z1 / TAPER ** k  # (so that its far end is grow * z1 wide, its near at least)
            t0, t1 = (z0 - za) / (zb - za), (z1 - za) / (zb - za)
            if not (zb > za):
                t0, t1 = 0.0, 1.0
            if not (0.0 < near_width < 1e300):
                continue
            for r in range(3):
                p[r] = a[r] + (b[r] - a[r]) * t0
                q[r] = a[r] + (b[r] - a[r]) * t1
            out_lin[w, 0, 0], out_lin[w, 1, 0], out_lin[w, 2, 0] = sx * near_width, sy * near_width, sz * near_width
            out_lin[w, 0, 1], out_lin[w, 1, 1], out_lin[w, 2, 1] = q[0] - p[0], q[1] - p[1], q[2] - p[2]
            out_lin[w, 0, 2], out_lin[w, 1, 2], out_lin[w, 2, 2] = nx * near_width, ny * near_width, nz * near_width
            out_pos[w, 0], out_pos[w, 1], out_pos[w, 2] = p[0], p[1], p[2]
            out_seg[w] = i
            out_strip[w] = k
            w += 1
    return w
