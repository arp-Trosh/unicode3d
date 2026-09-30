# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
"""What a Renderer draws behind the scene: a colour, a gradient, a sky or a sky box.

Set Renderer.background to one of these (or to a plain colour); None leaves the
empty parts of the frame transparent, so the screen's own background shows.

  Gradient(top, bottom)          fixed on screen, top row to bottom row
  Sky(zenith, horizon, ground)   by the direction each pixel looks in: it moves as the camera turns
  SkyBox(textures)               six pictures on the inside of a box around the camera, which never gets nearer

Fog, set as Renderer.fog, fades distant surfaces into the background (or into a colour of its own).

Colours are named Colors or (r, g, b) sRGB, as for objects, and blend in linear light.
"""
from dataclasses import dataclass, field

import numpy as np
from numba import njit, prange

from .color import to_linear_rgb
from .mesh import _srgb01
from .texture import build_mipmaps, pack as pack_textures, sample as sample_texture
from .transforms import normalize

NONE, COLOR, GRADIENT, SKY, SKYBOX = range(5)

# Each sky box face as seen from inside: its outward normal, and the directions of the texture's
# right (u) and up (v) edges, in BOX_FACES order (+X, -X, +Y, -Y, +Z, -Z). The sides are upright;
# the top face continues upward from the top edge of the -Z face, and the bottom face downward
# from its bottom edge, as in the usual unfolded cross.
SKYBOX_FACES = np.array([
    ((1, 0, 0), (0, 0, 1), (0, 1, 0)),     # +X
    ((-1, 0, 0), (0, 0, -1), (0, 1, 0)),   # -X
    ((0, 1, 0), (1, 0, 0), (0, 0, 1)),     # +Y (up)
    ((0, -1, 0), (1, 0, 0), (0, 0, -1)),   # -Y (down)
    ((0, 0, 1), (-1, 0, 0), (0, 1, 0)),    # +Z
    ((0, 0, -1), (1, 0, 0), (0, 1, 0)),    # -Z
], dtype=float)


@dataclass
class Gradient:
    """A vertical gradient across the frame, from `top` to `bottom`."""
    top: object = (40, 50, 80)
    bottom: object = (10, 10, 20)


@dataclass
class Sky:
    """A sky that follows the view: `horizon` at eye level, fading up to `zenith` overhead and quickly
    down to `ground` below the horizon."""
    zenith: object = (60, 110, 200)
    horizon: object = (190, 210, 230)
    ground: object = (70, 65, 60)


@dataclass
class SkyBox:
    """Six textures (in BOX_FACES order: +X, -X, +Y, -Y, +Z, -Z), each (H, W) brightness or (H, W, 3)
    colour, 0..1 sRGB or 0..255 ints, pictured on the inside of a box that moves with the camera.
    See SKYBOX_FACES for which way up each one is."""
    textures: list = field(default_factory=list)

    def packed(self):
        """pack() of the faces' mipmap chains, built once and again if a texture is replaced."""
        cached = self.__dict__.get("_packed")
        if cached is None or len(cached[0]) != len(self.textures) or any(
                a is not b for a, b in zip(cached[0], self.textures)):
            if len(self.textures) != 6:
                raise ValueError("a SkyBox needs exactly six textures")
            chains = [build_mipmaps(_srgb01(t)) for t in self.textures]
            cached = self._packed = (list(self.textures), pack_textures(chains))
        return cached[1]


@dataclass
class Fog:
    """Fog in the world, for Renderer.fog: surfaces nearer the camera than `start` are clear, and farther ones
    fade until, at `end` and beyond, they are gone: into `color` (a named Color or (r, g, b)), or, with None,
    into whatever is behind them, the background (a sky's horizon behind a distant hill), or the terminal's own
    background if there is none. Distances are in world units, from the eye, so surfaces keep their fog as the
    camera turns or other things come into view."""
    start: float = 10.0
    end: float = 50.0
    color: object = None


def fog_args(fog):
    """(depth cueing, start, end, linear rgb (3,), into the background) for shading.post_effects, from
    Renderer.fog: a Fog, or a number (depth cueing), or None."""
    if isinstance(fog, Fog):
        clear = fog.color is None
        rgb = np.zeros(3) if clear else np.asarray(to_linear_rgb(fog.color), np.float64)
        return 0.0, float(fog.start), max(float(fog.end), 1e-9), rgb, clear
    cue = float(fog or 0.0)
    return (cue if np.isfinite(cue) else 0.0), 0.0, 0.0, np.zeros(3), False


_NO_TEXTURES = pack_textures([[np.zeros((1, 1, 3))]])


def background_args(background, camera, aspect, height):
    """(kind, colours (3, 3) linear, basis (3, 3), texels, levels, first, lod) for fill_background, and
    (values, refs) for the renderer's check for an unchanged scene."""
    colors = np.zeros((3, 3))
    basis = np.zeros((3, 3))
    texels, levels, first = _NO_TEXTURES
    lod = 0.0
    if background is None:
        return (NONE, colors, basis, texels, levels, first, lod), ((), ())
    if isinstance(background, (Gradient, Sky, SkyBox)):
        kind = {Gradient: GRADIENT, Sky: SKY, SkyBox: SKYBOX}[type(background)]
    else:
        kind = COLOR
    if kind == COLOR:
        colors[0] = to_linear_rgb(background)
    elif kind == GRADIENT:
        colors[0], colors[1] = to_linear_rgb(background.top), to_linear_rgb(background.bottom)
    elif kind == SKY:
        colors[:] = [to_linear_rgb(c) for c in (background.zenith, background.horizon, background.ground)]
    if kind in (SKY, SKYBOX):
        # A pixel at (nx, ny) in normalized device coordinates looks along forward + nx * right + ny * up.
        forward = normalize(np.asarray(camera.target, float) - np.asarray(camera.position, float))
        right = normalize(np.cross(forward, camera.up))
        up = np.cross(right, forward)
        tan_y = np.tan(np.radians(camera.fov) / 2)
        basis[:] = forward, right * tan_y * aspect, up * tan_y
    refs = ()
    if kind == SKYBOX:
        texels, levels, first = background.packed()
        refs = tuple(background.textures)
        # About how many texels of a face span one pixel, looking straight at it.
        size = levels[0, 1]
        lod = float(np.log2(max(size * tan_y / max(height, 1), 1e-9)))
    values = (kind, colors.tobytes(), basis.tobytes(), lod)
    return (kind, colors, basis, texels, levels, first, lod), (values, refs)


@njit(cache=True, error_model="numpy")
def sky_colour(kind, colors, faces, texels, levels, first, lod, dx, dy, dz):
    """The background seen looking along unit direction (dx, dy, dz) in the world, linear rgb, for a sky
    or sky box; a plain colour for COLOR; for GRADIENT, the gradient from top (straight up) to bottom
    (straight down); black for NONE (and for kind -1: reflections switched off)."""
    if kind == SKY:
        angle = np.arcsin(min(max(dy, -1.0), 1.0))
        if angle >= 0.0:  # horizon to zenith, turning quickly at first
            t, top = np.sqrt(angle / (np.pi / 2)), 0
        else:  # horizon to ground within about ten degrees
            t, top = min(-angle / (np.pi / 18), 1.0), 2
        return (colors[1, 0] + (colors[top, 0] - colors[1, 0]) * t,
                colors[1, 1] + (colors[top, 1] - colors[1, 1]) * t,
                colors[1, 2] + (colors[top, 2] - colors[1, 2]) * t)
    if kind == SKYBOX:
        ax, ay, az = abs(dx), abs(dy), abs(dz)
        if ax >= ay and ax >= az:
            face, m = (0 if dx > 0 else 1), ax
        elif ay >= az:
            face, m = (2 if dy > 0 else 3), ay
        else:
            face, m = (4 if dz > 0 else 5), az
        px, py, pz = dx / m, dy / m, dz / m
        u = 0.5 * (px * faces[face, 1, 0] + py * faces[face, 1, 1] + pz * faces[face, 1, 2] + 1.0)
        v = 0.5 * (px * faces[face, 2, 0] + py * faces[face, 2, 1] + pz * faces[face, 2, 2] + 1.0)
        r, g, b, _ = sample_texture(texels, levels, first, face, u, v, lod)
        return r, g, b
    if kind == GRADIENT:
        t = 0.5 - 0.5 * dy
        return (colors[0, 0] + (colors[1, 0] - colors[0, 0]) * t, colors[0, 1] + (colors[1, 1] - colors[0, 1]) * t,
                colors[0, 2] + (colors[1, 2] - colors[0, 2]) * t)
    if kind == COLOR:
        return colors[0, 0], colors[0, 1], colors[0, 2]
    return 0.0, 0.0, 0.0


@njit(cache=True, error_model="numpy", parallel=True)
def fill_background(rgb, alpha, kind, colors, basis, faces, texels, levels, first, lod):
    """Fill what the scene leaves uncovered (alpha < 1) with the background, in place: each pixel gets
    the background in proportion to how much of it is uncovered, and becomes opaque."""
    h, w = alpha.shape
    if kind == NONE:
        return
    for y in prange(h):
        for x in range(w):
            a = alpha[y, x]
            if a >= 1.0:
                continue
            r, g, b = colors[0, 0], colors[0, 1], colors[0, 2]
            if kind == GRADIENT:
                t = y / max(h - 1, 1)
                r = colors[0, 0] + (colors[1, 0] - colors[0, 0]) * t
                g = colors[0, 1] + (colors[1, 1] - colors[0, 1]) * t
                b = colors[0, 2] + (colors[1, 2] - colors[0, 2]) * t
            elif kind == SKY or kind == SKYBOX:
                nx, ny = (x + 0.5) / w * 2.0 - 1.0, 1.0 - (y + 0.5) / h * 2.0
                dx = basis[0, 0] + nx * basis[1, 0] + ny * basis[2, 0]
                dy = basis[0, 1] + nx * basis[1, 1] + ny * basis[2, 1]
                dz = basis[0, 2] + nx * basis[1, 2] + ny * basis[2, 2]
                dl = np.sqrt(dx * dx + dy * dy + dz * dz)
                r, g, b = sky_colour(kind, colors, faces, texels, levels, first, lod, dx / dl, dy / dl, dz / dl)
            uncovered = 1.0 - a
            rgb[y, x, 0] += uncovered * r
            rgb[y, x, 1] += uncovered * g
            rgb[y, x, 2] += uncovered * b
            alpha[y, x] = 1.0
