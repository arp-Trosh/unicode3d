# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
"""Camera, lights, scene objects and the render pipeline."""
from dataclasses import dataclass, field

import numpy as np
from numba import njit, prange

from .background import SKYBOX_FACES, background_args, fill_background
from .color import to_linear_rgb, to_srgb
from .mesh import Mesh
from .raster import (ATTRS, FACE_CHUNK, ROW_BAND, VERTEX_CHUNK, FrameBuffer, barycentric, bin_bands, count_bands,
                     project, rasterize, transform)
from .texture import pack as pack_textures, sample as sample_texture
from .transforms import look_at, normalize, perspective, quat_identity, quat_mul, quat_to_matrix


def _vec3(*v):
    return field(default_factory=lambda: np.array(v, dtype=float))


@dataclass
class Camera:
    position: np.ndarray = _vec3(0.0, 0.0, 5.0)
    target: np.ndarray = _vec3(0.0, 0.0, 0.0)
    up: np.ndarray = _vec3(0.0, 1.0, 0.0)
    fov: float = 50.0  # vertical, degrees
    near: float = 0.1
    far: float = 100.0

    def view_matrix(self):
        return look_at(self.position, self.target, self.up)


@dataclass
class Light:
    """Light from far away, arriving from one direction everywhere (like the sun).

    Light levels are perceived brightness, 0..1: a level of 0.5 looks half as bright. A surface gets
    ambient everywhere plus diffuse where it faces the light, from every light in the scene, summed.
    """
    direction: np.ndarray = _vec3(0.3, -1.0, -0.5)  # the way the light travels
    ambient: float = 0.3    # light levels are perceived brightness, 0..1
    diffuse: float = 0.7
    specular: float = 0.35  # strength of the Blinn-Phong highlight, in the light's colour whatever the surface's
    shininess: float = 24.0
    color: object = (255, 255, 255)  # a named Color, or (r, g, b) sRGB: it scales the light's levels channel by channel


@dataclass
class PointLight:
    """Light spreading from a point (a lamp, a torch), fading to nothing at `range` from it.

    Its levels (see Light) are full at the light and fall off smoothly with distance. ambient is
    light it gives surfaces nearby whichever way they face.
    """
    position: np.ndarray = _vec3(0.0, 2.0, 0.0)
    color: object = (255, 255, 255)
    diffuse: float = 0.8
    specular: float = 0.35
    shininess: float = 24.0
    range: float = 10.0
    ambient: float = 0.0


LIGHT_COLUMNS = 15  # kind, direction or position (3), colour as levels (3), linear colour (3), ambient, diffuse,
                    # specular, shininess, range


def _light_rows(lights):
    """The lights as a (L, LIGHT_COLUMNS) array for _shade."""
    rows = np.zeros((len(lights), LIGHT_COLUMNS))
    for i, light in enumerate(lights):
        lin = _linear_color(light.color)
        if isinstance(light, PointLight):
            rows[i, 0], rows[i, 1:4], rows[i, 14] = 1, np.asarray(light.position, float), float(light.range)
        else:
            rows[i, 1:4] = -normalize(light.direction)
        rows[i, 4:7], rows[i, 7:10] = to_srgb(light.color), lin
        rows[i, 10:14] = light.ambient, light.diffuse, light.specular, light.shininess
    return rows


def _as_lights(lights):
    """A light or a sequence of lights, as a list."""
    return [lights] if isinstance(lights, (Light, PointLight)) else list(lights)


@dataclass
class Pick:
    """What Renderer.pick found: the object, the point on its surface, and how far that is from the camera."""
    object: object
    position: np.ndarray
    distance: float


class _Placed:
    """Placement in a scene graph, shared by Node and Object3D: position, rotation and scale are
    relative to `parent` (a Node or Object3D), or to the world if there is none."""

    def world_transform(self):
        """(position (3,), rotation quaternion (4,), scale, visible) in the world, through all parents.

        A parent's scale scales its children's offsets and sizes; an object is visible
        only if it and all its parents are.
        """
        position = np.asarray(self.position, dtype=float)
        rotation, scale, visible = np.asarray(self.rotation, dtype=float), float(self.scale), bool(self.visible)
        node, seen = self.parent, 0
        while node is not None:
            position = np.asarray(node.position, dtype=float) + quat_to_matrix(node.rotation) @ (node.scale * position)
            rotation = quat_mul(node.rotation, rotation)
            scale *= float(node.scale)
            visible = visible and bool(node.visible)
            node, seen = node.parent, seen + 1
            if seen > 1000:
                raise ValueError("scene graph has a cycle: an object is its own ancestor")
        return position, rotation, scale, visible

    def to_world(self, point):
        """A point given in this node's own space, in world coordinates."""
        position, rotation, scale, _ = self.world_transform()
        return position + quat_to_matrix(rotation) @ (scale * np.asarray(point, dtype=float))


@dataclass
class Node(_Placed):
    """A transform with no mesh, for grouping: objects whose parent is a Node move, turn, scale and
    hide with it. Nodes can have Nodes as parents, and are not passed to render()."""
    position: np.ndarray = _vec3(0.0, 0.0, 0.0)
    rotation: np.ndarray = field(default_factory=quat_identity)
    scale: float = 1.0
    visible: bool = True
    parent: object = None


@dataclass
class Object3D(_Placed):
    """A mesh placed in the scene. position, rotation (a quaternion, w first) and scale are relative to
    `parent` if it has one (a Node or another Object3D), otherwise to the world."""
    mesh: Mesh
    position: np.ndarray = _vec3(0.0, 0.0, 0.0)
    rotation: np.ndarray = field(default_factory=quat_identity)
    scale: float = 1.0
    color: object = 0       # a named Color, or (r, g, b) sRGB as 0..255 ints or 0..1 floats
    visible: bool = True
    double_sided: bool = False  # draw back faces too (for meshes with inconsistent winding)
    parent: object = None
    emissive: float = 0.0   # light level the surface gives off itself, added to the lights': 1.0 shows its
                            # colour at full brightness whatever the lighting (a lamp, a screen, a glowing marker)


_COLOR_CACHE = {}


def _linear_color(color):
    """to_linear_rgb(color), remembered for colours seen before (objects mostly keep theirs)."""
    if isinstance(color, (int, np.integer)):
        key = int(color)
    else:
        a = np.asarray(color)
        key = (a.dtype.kind, tuple(a.ravel().tolist()))
    c = _COLOR_CACHE.get(key)
    if c is None:
        if len(_COLOR_CACHE) > 4096:
            _COLOR_CACHE.clear()
        c = _COLOR_CACHE[key] = to_linear_rgb(color)
    return c


def _mesh_key(mesh):
    """What a packed mesh depends on, compared by identity (see Renderer._scene_state)."""
    return (mesh, mesh.vertices, mesh.faces, mesh.uvs, mesh.materials, mesh.vertex_colors, mesh.face_colors,
            *mesh.textures)


def _pack_meshes(meshes):
    """The meshes' arrays packed together for raster.project, and their textures for sampling."""
    chains, chain_of = [], {}
    verts, normals, faces, uvs, colors, face_chain, face_texels = [], [], [], [], [], [], []
    mesh_vertex, mesh_face, spheres = [0], [0], []
    for mesh in meshes:
        n_faces = len(mesh.faces)
        verts.append(np.asarray(mesh.vertices, dtype=float))
        normals.append(mesh.vertex_normals())
        faces.append(np.asarray(mesh.faces, dtype=np.int64) + mesh_vertex[-1])
        if mesh.materials is not None and mesh.textures:
            table = np.empty(len(mesh.textures), np.int64)
            for m in range(len(mesh.textures)):
                levels = mesh.mipmaps(m)
                if id(levels) not in chain_of:
                    chain_of[id(levels)] = len(chains)
                    chains.append(levels)
                table[m] = chain_of[id(levels)]
            materials = np.asarray(mesh.materials)
            face_chain.append(table[materials])
            sizes = np.array([np.shape(t)[0] * np.shape(t)[1] for t in mesh.textures], dtype=float)
            face_texels.append(sizes[materials])
            uvs.append(np.asarray(mesh.uvs, dtype=float))
        else:
            face_chain.append(np.full(n_faces, -1, np.int64))
            face_texels.append(np.zeros(n_faces))
            uvs.append(np.zeros((n_faces, 3, 2)))
        corner = mesh.corner_colors()
        colors.append(np.ones((n_faces, 3, 3)) if corner is None else corner)
        centre, radius = mesh.bounds()
        spheres.append((*centre, radius))
        mesh_vertex.append(mesh_vertex[-1] + len(mesh.vertices))
        mesh_face.append(mesh_face[-1] + n_faces)
    c = np.ascontiguousarray
    return {"vertices": c(np.concatenate(verts), np.float64), "normals": c(np.concatenate(normals), np.float64),
            "faces": c(np.concatenate(faces), np.int64), "uvs": c(np.concatenate(uvs), np.float64),
            "colors": c(np.concatenate(colors), np.float64), "face_chain": c(np.concatenate(face_chain), np.int64),
            "face_texels": c(np.concatenate(face_texels), np.float64),
            "mesh_vertex": np.array(mesh_vertex, np.int64), "mesh_face": np.array(mesh_face, np.int64),
            "spheres": np.array(spheres, np.float64).reshape(-1, 4), "textures": pack_textures(chains)}




# Sample positions within a pixel (x, y from its top-left corner): the standard
# multisample patterns, where no two samples share a row or column, so near-vertical
# and near-horizontal edges get as many coverage steps as there are samples.
SAMPLE_PATTERNS = {
    1: ((0.5, 0.5),),
    4: ((0.375, 0.125), (0.875, 0.375), (0.125, 0.625), (0.625, 0.875)),
    8: ((0.5625, 0.3125), (0.4375, 0.6875), (0.8125, 0.5625), (0.3125, 0.1875),
        (0.1875, 0.8125), (0.0625, 0.4375), (0.6875, 0.9375), (0.9375, 0.0625)),
    16: ((0.5625, 0.5625), (0.4375, 0.3125), (0.3125, 0.625), (0.75, 0.4375),
         (0.1875, 0.375), (0.625, 0.8125), (0.8125, 0.6875), (0.6875, 0.1875),
         (0.375, 0.875), (0.5, 0.0625), (0.25, 0.125), (0.125, 0.75),
         (0.0, 0.5), (0.9375, 0.25), (0.875, 0.9375), (0.0625, 0.0)),
}
_NO_UVS = np.zeros((1, 3, 2))  # stands in for the uvs of untextured meshes
EDGE_CONTRAST = 0.03  # linear-light spread among a pixel's first samples that marks it for more


@njit(cache=True)
def _decode(level):
    """A perceived light level (0..1, like an sRGB value) as linear light."""
    return level / 12.92 if level <= 0.04045 else ((level + 0.055) / 1.055) ** 2.4


@njit(cache=True)
def _shade(t, b0, b1, b2, inv_w, attrs, chain, lod, texels, levels, first, lights, eye, emissive, out):
    """Linear RGB, into out (3,), of triangle t at barycentric weights b: Blinn-Phong lighting from
    every light (rows of _light_rows()), plus `emissive`, on a surface of the colour interpolated
    from its corners, textured if the triangle has a mipmap chain. Highlights take the light's colour."""
    # Perspective-correct interpolation of world position, normal, uv and colour.
    w0, w1, w2 = b0 * inv_w[t, 0], b1 * inv_w[t, 1], b2 * inv_w[t, 2]
    ws = w0 + w1 + w2
    at = attrs[t]

    def lerp(k):
        return (w0 * at[0, k] + w1 * at[1, k] + w2 * at[2, k]) / ws

    nx, ny, nz = lerp(3), lerp(4), lerp(5)
    nl = max(np.sqrt(nx * nx + ny * ny + nz * nz), 1e-12)
    nx, ny, nz = nx / nl, ny / nl, nz / nl
    px, py, pz = lerp(0), lerp(1), lerp(2)
    ex, ey, ez = eye[0] - px, eye[1] - py, eye[2] - pz
    el = max(np.sqrt(ex * ex + ey * ey + ez * ez), 1e-12)
    ex, ey, ez = ex / el, ey / el, ez / el
    level_r = level_g = level_b = emissive
    spec_r = spec_g = spec_b = 0.0
    for i in range(lights.shape[0]):
        light = lights[i]
        lx, ly, lz, fade = light[1], light[2], light[3], 1.0
        if light[0] != 0:  # a point light: towards it, fading out smoothly by its range
            lx, ly, lz = lx - px, ly - py, lz - pz
            d = max(np.sqrt(lx * lx + ly * ly + lz * lz), 1e-12)
            lx, ly, lz = lx / d, ly / d, lz / d
            f = max(1.0 - (d / max(light[14], 1e-12)) ** 2, 0.0)
            fade = f * f
            if fade <= 0.0:
                continue
        level = (light[10] + light[11] * max(nx * lx + ny * ly + nz * lz, 0.0)) * fade
        level_r, level_g, level_b = level_r + light[4] * level, level_g + light[5] * level, level_b + light[6] * level
        if light[12] > 0.0:
            hx, hy, hz = lx + ex, ly + ey, lz + ez
            hl = max(np.sqrt(hx * hx + hy * hy + hz * hz), 1e-12)
            spec = light[12] * max((nx * hx + ny * hy + nz * hz) / hl, 0.0) ** light[13] * fade
            spec_r, spec_g, spec_b = spec_r + light[7] * spec, spec_g + light[8] * spec, spec_b + light[9] * spec
    r, g, b = lerp(8), lerp(9), lerp(10)
    if chain[t] >= 0:
        tr, tg, tb = sample_texture(texels, levels, first, chain[t], lerp(6), lerp(7), lod[t])
        r, g, b = r * tr, g * tg, b * tb
    # Light levels are perceived brightness (0.5 looks half as bright), as artists tune them, so they
    # are decoded like any sRGB value; everything after this point works in linear light.
    kr = _decode(min(max(level_r, 0.0), 1.0))
    kg = kr if level_g == level_r else _decode(min(max(level_g, 0.0), 1.0))
    kb = kr if level_b == level_r else _decode(min(max(level_b, 0.0), 1.0))
    out[0] = min(max(r * kr + spec_r, 0.0), 1.0)
    out[1] = min(max(g * kg + spec_g, 0.0), 1.0)
    out[2] = min(max(b * kb + spec_b, 0.0), 1.0)


@njit(cache=True, parallel=True)
def _resolve(tris, depth, pixels, width, xs, ys, inv_w, attrs, tri_inst, ident, emissive, chain, lod, texels, levels,
             first, lights, eye, contrast, sample_rgb, rgb, cover, near_depth, near_id, more):
    """Shade the rasterized samples and sum them up per pixel.

    Each pixel is shaded once per triangle covering it, at the pixel centre, like
    hardware multisampling: the samples only decide coverage. (Texture detail is
    smoothed by mipmapping instead.)

    Writes, per pixel: rgb (summed colour of the covered samples), cover (how many
    samples were covered), the depth and object id of the nearest sample, and
    more: whether the samples disagree (some covered and some not, different
    objects, or colours further apart than `contrast`), so the pixel is worth more
    samples. sample_rgb (M, S, 3) is scratch space.
    """
    m, n = tris.shape
    for c in prange(m):
        cx, cy = pixels[c] % width + 0.5, pixels[c] // width + 0.5
        for s in range(n):
            t = tris[c, s]
            if t < 0:
                sample_rgb[c, s, :] = 0.0
                continue
            prev = s
            for s2 in range(s):
                if tris[c, s2] == t:
                    prev = s2
                    break
            if prev < s:
                sample_rgb[c, s, :] = sample_rgb[c, prev, :]
            else:
                b0, b1, b2 = barycentric(xs, ys, t, cx, cy)
                _shade(t, b0, b1, b2, inv_w, attrs, chain, lod, texels, levels, first, lights, eye,
                       emissive[tri_inst[t]], sample_rgb[c, s])
        r = g = b = 0.0
        lr = lg = lb = np.inf
        hr = hg = hb = -np.inf
        covered, mixed, best, best_id = 0, False, -1.0, 0
        first_id = ident[tri_inst[tris[c, 0]]] if tris[c, 0] >= 0 else 0
        for s in range(n):
            t = tris[c, s]
            sid = ident[tri_inst[t]] if t >= 0 else 0
            covered += t >= 0
            mixed |= sid != first_id
            if depth[c, s] > best:
                best, best_id = depth[c, s], sid
            vr, vg, vb = sample_rgb[c, s, 0], sample_rgb[c, s, 1], sample_rgb[c, s, 2]
            r, g, b = r + vr, g + vg, b + vb
            lr, lg, lb = min(lr, vr), min(lg, vg), min(lb, vb)
            hr, hg, hb = max(hr, vr), max(hg, vg), max(hb, vb)
        rgb[c, 0], rgb[c, 1], rgb[c, 2] = r, g, b
        cover[c], near_depth[c], near_id[c] = covered, best, best_id
        more[c] = 0 < covered < n or mixed or max(hr - lr, hg - lg, hb - lb) > contrast


@njit(cache=True)
def _combine(rgb, alpha, base_rgb, base_cover, n, pixels, extra_rgb, extra_cover, n_extra):
    """Each pixel's colour and coverage (flat framebuffer arrays) from its n base samples, plus the
    n_extra samples taken in `pixels`."""
    for c in range(base_cover.shape[0]):
        alpha[c] = base_cover[c] / n
        for k in range(3):
            rgb[c, k] = base_rgb[c, k] / n
    total = n + n_extra
    for j in range(pixels.shape[0]):
        c = pixels[j]
        alpha[c] = (base_cover[c] + extra_cover[j]) / total
        for k in range(3):
            rgb[c, k] = (base_rgb[c, k] + extra_rgb[j, k]) / total


@njit(cache=True, parallel=True)
def _post_effects(rgb, alpha, depth, fog, outline):
    """Fog, then outlines, applied in place to a framebuffer's arrays (see Renderer for both)."""
    h, w = depth.shape
    if fog:
        # Dim pixels in proportion to how far back they sit within the scene's depth range.
        near, far = np.inf, -np.inf
        for y in range(h):
            for x in range(w):
                if alpha[y, x] > 0:
                    near, far = min(near, 1.0 / depth[y, x]), max(far, 1.0 / depth[y, x])
        span = max(far - near, 0.25 * near)
        for y in prange(h):
            for x in range(w):
                if alpha[y, x] > 0:
                    rgb[y, x, :] *= 1.0 - fog * (1.0 / depth[y, x] - near) / span
    if outline:
        # Darken pixels just behind a depth edge. 1/w is linear across any plane on screen, so its
        # second difference is ~0 on flat and gently curved surfaces and large where one surface
        # passes in front of another. It is positive on the far side of the edge, which is the side
        # darkened, so shapes in front keep their full size.
        for y in prange(h):
            for x in range(w):
                d = depth[y, x]
                if alpha[y, x] <= 0:
                    continue
                edge = (0 < y < h - 1 and alpha[y - 1, x] > 0 and alpha[y + 1, x] > 0
                        and depth[y - 1, x] + depth[y + 1, x] - 2 * d > 0.08 * d)
                edge = edge or (0 < x < w - 1 and alpha[y, x - 1] > 0 and alpha[y, x + 1] > 0
                                and depth[y, x - 1] + depth[y, x + 1] - 2 * d > 0.08 * d)
                if edge:
                    rgb[y, x, :] *= 1.0 - outline


def _chunks(counts, starts, size):
    """Split each instance's run of counts[i] items, starting at starts[i], into chunks of at most `size`:
    (instance, first, end) arrays, in instance order."""
    per = (counts + size - 1) // size
    inst = np.repeat(np.arange(len(counts)), per)
    within = np.arange(len(inst)) - np.repeat(np.cumsum(per) - per, per)
    first = starts[inst] + within * size
    end = np.minimum(first + size, starts[inst] + counts[inst])
    return inst.astype(np.int64), first.astype(np.int64), end.astype(np.int64)


class _Buffers:
    """Arrays kept from one render to the next, so drawing a frame allocates (and page-faults in) no new memory."""

    def __init__(self):
        self._arrays = {}

    def get(self, name, shape, dtype=np.float64):
        """An array of this shape with undefined contents, in the memory of the last one of this name if it fits."""
        size = int(np.prod(shape))
        a = self._arrays.get(name)
        if a is None or a.dtype != dtype or a.size < size:
            a = self._arrays[name] = np.empty(size + size // 4, dtype)  # room to grow a little
        return a[:size].reshape(shape)

    def arange(self, n):
        """0..n-1 (not to be modified)."""
        a = self._arrays.get("arange")
        if a is None or len(a) != n:
            a = self._arrays["arange"] = np.arange(n)
        return a


class Renderer:
    """Renders objects into a FrameBuffer whose pixels are finer than terminal cells.

    width and height are in terminal cells; cell_pixels is the (columns, rows)
    of pixels in a cell, which comes from the screen's glyph set
    (Screen.cell_pixels). cell_aspect is a cell's width divided by its height
    (terminal cells are roughly twice as tall as they are wide).

    samples: samples per pixel (1, 4, 8 or 16), averaged for smooth edges. Each
    pixel is shaded once per triangle covering it, like hardware multisampling;
    samples only measure coverage, and texture detail is smoothed by mipmaps.
    edge_samples: extra samples (0, 4, 8 or 16) taken only in pixels whose first
    samples disagree (silhouettes, creases, overlaps), where they matter.
    fog: how much the farthest surfaces are dimmed relative to the nearest (depth cueing).
    outline: how much to darken the far side of depth edges, where one surface
    passes in front of another, so overlapping shapes stay distinct.
    lod_bias: added to every texture's mip level; negative is sharper, positive softer.
    background: drawn behind the scene: None (leave it empty, so the screen's background
    shows), a colour, or a Gradient, Sky or SkyBox (see background.py).
    """

    def __init__(self, width, height, cell_pixels=(1, 2), cell_aspect=0.5, samples=4, edge_samples=8,
                 fog=0.3, outline=0.55, lod_bias=-0.5, background=None):
        for n in (samples, edge_samples):
            if n not in SAMPLE_PATTERNS and n != 0:
                raise ValueError(f"sample counts must be one of {sorted(SAMPLE_PATTERNS)}, not {n}")
        self.cell_aspect = cell_aspect
        self.samples = samples
        self.edge_samples = edge_samples
        self.fog = fog
        self.outline = outline
        self.lod_bias = lod_bias
        self.background = background
        self.width = self.height = 0
        self.cell_pixels = tuple(cell_pixels)
        self.framebuffer = FrameBuffer(0, 0, cell_pixels)
        self.view_proj = None
        self.draws = 0     # renders that rasterized the scene, rather than reusing an unchanged frame
        self._last = None  # what the framebuffer shows, as _scene_state() gives it
        self._pack = None  # (mesh keys, _pack_meshes() of those meshes) as last drawn
        self._objects = []  # the render list as last drawn, for pick()
        self._eye = self._forward = None  # the camera's position and view direction then
        self._buffers = _Buffers()
        self.resize(width, height)

    def resize(self, width, height, cell_pixels=None):
        """Set the size in cells, and optionally the pixels per cell (e.g. Screen.cell_pixels)."""
        cell_pixels = self.cell_pixels if cell_pixels is None else tuple(cell_pixels)
        if (width, height, cell_pixels) == (self.width, self.height, self.cell_pixels):
            return
        self.width, self.height, self.cell_pixels = width, height, cell_pixels
        self.framebuffer.cell_pixels = cell_pixels
        self.framebuffer.resize(width * cell_pixels[0], height * cell_pixels[1])

    def project(self, point):
        """Cell coordinates (x, y) of a world point as of the last render, or None if behind the camera."""
        if self.view_proj is None:
            return None
        clip = self.view_proj @ np.array([*point, 1.0])
        if clip[3] <= 1e-6:
            return None
        return (clip[0] / clip[3] + 1.0) * 0.5 * self.width, (1.0 - clip[1] / clip[3]) * 0.5 * self.height

    def ray(self, x, y):
        """(origin, unit direction) in the world of the line of sight through cell (x, y) as of the last
        render (cells count from the frame's top-left; fractions are fine, and x + 0.5, y + 0.5 is a
        cell's centre); None before the first render."""
        if self.view_proj is None or self.width < 1 or self.height < 1:
            return None
        nx, ny = x / self.width * 2.0 - 1.0, 1.0 - y / self.height * 2.0
        inv = np.linalg.inv(self.view_proj)
        near, far = inv @ np.array([nx, ny, -1.0, 1.0]), inv @ np.array([nx, ny, 1.0, 1.0])
        near, far = near[:3] / near[3], far[:3] / far[3]
        return self._eye.copy(), normalize(far - near)

    def pick(self, x, y):
        """What the last render drew in cell (x, y) of its frame: a Pick (object, position in the world,
        distance from the camera), taking the nearest surface among the cell's pixels; None if nothing."""
        fb = self.framebuffer
        pw, ph = self.cell_pixels
        x, y = int(x), int(y)
        if self.view_proj is None or not (0 <= x < self.width and 0 <= y < self.height):
            return None
        ids = fb.ids[y * ph:(y + 1) * ph, x * pw:(x + 1) * pw]
        depth = np.where(ids > 0, fb.depth[y * ph:(y + 1) * ph, x * pw:(x + 1) * pw], 0.0)
        if not depth.any():
            return None
        py, px = np.unravel_index(np.argmax(depth), depth.shape)
        obj = self._objects[ids[py, px] - 1]
        # The pixel's centre, at the depth drawn there: 1/w is the distance along the view direction.
        origin, direction = self.ray((x * pw + px + 0.5) / pw, (y * ph + py + 0.5) / ph)
        forward = self._forward
        distance = 1.0 / depth[py, px] / max(float(direction @ forward), 1e-12)
        return Pick(obj, origin + direction * distance, distance)

    def invalidate(self):
        """Make the next render draw afresh, e.g. after editing a mesh's arrays in place."""
        self._last = None
        self._pack = None

    def _instances(self, objects):
        """The objects to draw, as arrays: their meshes' place in the pack, world poses, colours, flags and
        ids (render-list index + 1). Packs the meshes afresh if the set of meshes has changed."""
        meshes, mesh_of, keys = [], {}, []
        mesh_idx, pos, quat, scale, rgb, double, ident, emissive = [], [], [], [], [], [], [], []
        for i, obj in enumerate(objects):
            mesh = obj.mesh
            if mesh is None or not len(mesh.faces):
                continue
            if obj.parent is None:
                if not obj.visible:
                    continue
                p, q, sc = obj.position, obj.rotation, obj.scale
            else:
                p, q, sc, visible = obj.world_transform()
                if not visible:
                    continue
            m = mesh_of.get(id(mesh))
            if m is None:
                m = mesh_of[id(mesh)] = len(meshes)
                meshes.append(mesh)
                keys.append(_mesh_key(mesh))
            mesh_idx.append(m)
            pos.append(p)
            quat.append(q)
            scale.append(sc)
            rgb.append(_linear_color(obj.color))
            double.append(obj.double_sided)
            ident.append(i + 1)
            emissive.append(obj.emissive)
        if not meshes:
            return None
        packed = self._pack
        if packed is None or len(packed[0]) != len(keys) or any(
                len(a) != len(b) or any(x is not y for x, y in zip(a, b)) for a, b in zip(packed[0], keys)):
            packed = self._pack = (keys, _pack_meshes(meshes))
        quat = np.array(quat, dtype=np.float64).reshape(-1, 4)
        quat /= np.maximum(np.linalg.norm(quat, axis=1, keepdims=True), 1e-300)
        return {"mesh": np.array(mesh_idx, np.int64), "pos": np.array(pos, np.float64).reshape(-1, 3), "quat": quat,
                "scale": np.array(scale, np.float64), "rgb": np.array(rgb, np.float64).reshape(-1, 3),
                "double": np.array(double, np.bool_), "ident": np.array(ident, np.int32),
                "emissive": np.array(emissive, np.float64), "pack": packed[1],
                "keys": keys}

    def _scene_state(self, inst, camera, lights):
        """Everything a render depends on: (values compared by equality, objects compared by identity).

        Meshes, their arrays and textures count as changed when replaced, as in Mesh's own caches;
        edits made inside them are not seen (call invalidate() after those).
        """
        values = [self.width, self.height, self.cell_pixels, self.cell_aspect, self.samples, self.edge_samples,
                  self.fog, self.outline, self.lod_bias,
                  np.asarray(camera.position, float).tobytes(), np.asarray(camera.target, float).tobytes(),
                  np.asarray(camera.up, float).tobytes(), camera.fov, camera.near, camera.far,
                  _light_rows(lights).tobytes()]
        if inst is None:
            return values, []
        values += [inst[k].tobytes() for k in ("mesh", "pos", "quat", "scale", "rgb", "double", "ident", "emissive")]
        return values, [x for key in inst["keys"] for x in (*key, None)]

    def render(self, objects, camera, lights):
        """Draw the objects; returns the framebuffer (reused by the next render: copy it to keep it).

        lights: a Light or PointLight, or a list of them (their light adds up).
        Objects with a parent are placed through it (see Node). When nothing has
        changed since the last render (see _scene_state), the framebuffer is
        returned as it is, so a still scene costs almost nothing.
        """
        lights = _as_lights(lights)
        self._objects = list(objects)
        inst = self._instances(objects)
        aspect = self.width * self.cell_aspect / max(self.height, 1)
        bg_args, bg_state = background_args(self.background, camera, aspect, self.framebuffer.height)
        state = self._scene_state(inst, camera, lights)
        state = (state[0] + list(bg_state[0]), state[1] + list(bg_state[1]))
        last = self._last
        if (last is not None and last[0] == state[0] and len(last[1]) == len(state[1])
                and all(a is b for a, b in zip(last[1], state[1]))):
            return self.framebuffer
        self._last = None
        fb = self._draw(inst, camera, lights, bg_args)
        self.draws += 1
        self._last = state
        return fb

    def _draw(self, inst, camera, lights, bg_args):
        fb = self.framebuffer
        fb.clear()
        if self.width < 1 or self.height < 1:
            return fb
        aspect = self.width * self.cell_aspect / self.height
        self.view_proj = perspective(np.radians(camera.fov), aspect, camera.near, camera.far) @ camera.view_matrix()
        self._eye = np.asarray(camera.position, dtype=float).copy()
        self._forward = normalize(np.asarray(camera.target, float) - self._eye)
        scene = self._geometry(inst, camera, lights) if inst is not None else None
        if scene is not None:
            self._shade_scene(scene)
        kind, colors, basis, texels, levels, first, lod = bg_args
        fill_background(fb.rgb, fb.alpha, kind, colors, basis, SKYBOX_FACES, texels, levels, first, lod)
        return fb

    def _shade_scene(self, scene):
        """Rasterize and shade the scene's triangles into the framebuffer, with extra samples at edges,
        then fog and outlines."""
        fb, buf = self.framebuffer, self._buffers
        # Which triangles may touch each band of rows, for both rasterizing passes.
        n_bands = (fb.height + ROW_BAND - 1) // ROW_BAND
        band_count = buf.get("band_count", (n_bands,), np.int64)
        per_chunk = count_bands(scene["ys"], fb.height, band_count)
        band_start = np.zeros(n_bands + 1, np.int64)
        np.cumsum(band_count, out=band_start[1:])
        band_tris = buf.get("band_tris", (int(band_start[-1]),), np.int64)
        bin_bands(scene["ys"], fb.height, per_chunk, band_start, band_tris)
        scene["bands"] = (band_start, band_tris)
        n = self.samples
        base = self._accumulate(scene, SAMPLE_PATTERNS[n], "base", depth=fb.depth.reshape(-1), ids=fb.ids.reshape(-1))
        pixels, extra_rgb, extra_cover = np.zeros(0, np.int64), np.zeros((0, 3)), np.zeros(0, np.int64)
        if self.edge_samples and n > 1:
            pixels = np.flatnonzero(base["more"])
            if len(pixels):
                # A different pattern from the base one: the same one turned a quarter.
                pattern = tuple((1.0 - y, x) for x, y in SAMPLE_PATTERNS[self.edge_samples])
                extra = self._accumulate(scene, pattern, "edge", pixels)
                extra_rgb, extra_cover = extra["rgb"], extra["cover"]
        _combine(fb.rgb.reshape(-1, 3), fb.alpha.reshape(-1), base["rgb"], base["cover"], n,
                 pixels, extra_rgb, extra_cover, self.edge_samples)
        _post_effects(fb.rgb, fb.alpha, fb.depth, self.fog, self.outline)

    # ----- geometry --------------------------------------------------------------------

    def _geometry(self, inst, camera, lights):
        """Every visible object's triangles in screen space, in one set of arrays in render-list order,
        with what shading them needs; None if there are none."""
        fb, buf, pack = self.framebuffer, self._buffers, inst["pack"]
        eye = np.asarray(camera.position, dtype=float)
        n_inst = len(inst["mesh"])
        vertex_counts = np.diff(pack["mesh_vertex"])[inst["mesh"]]
        face_counts = np.diff(pack["mesh_face"])[inst["mesh"]]
        inst_vertex = np.zeros(n_inst + 1, np.int64)
        np.cumsum(vertex_counts, out=inst_vertex[1:])
        # The work, in pieces that run in parallel: chunks of each instance's vertices, then of its faces.
        vchunk_inst, vchunk_first, vchunk_end = _chunks(vertex_counts, np.zeros(n_inst, np.int64), VERTEX_CHUNK)
        chunk_inst, chunk_first, chunk_end = _chunks(face_counts, pack["mesh_face"][inst["mesh"]], FACE_CHUNK)
        n_chunks = len(chunk_inst)
        whole, cut = buf.get("whole", (n_chunks,), np.int64), buf.get("cut", (n_chunks,), np.int64)
        off_whole, off_cut = buf.get("off_whole", (n_chunks,), np.int64), buf.get("off_cut", (n_chunks,), np.int64)
        world = buf.get("world", (int(inst_vertex[-1]), 10))
        k = transform(pack["vertices"], pack["normals"], pack["faces"], pack["mesh_vertex"], pack["spheres"],
                      inst["mesh"], inst["quat"], inst["scale"], inst["pos"], inst["double"], inst_vertex,
                      self.view_proj, eye, float(camera.near), vchunk_inst, vchunk_first, vchunk_end,
                      chunk_inst, chunk_first, chunk_end, world, buf.get("visible", (n_inst,), np.bool_),
                      whole, cut, off_whole, off_cut)
        xs, ys, inv_w = buf.get("xs", (k, 3)), buf.get("ys", (k, 3)), buf.get("inv_w", (k, 3))
        attrs = buf.get("attrs", (k, 3, ATTRS))
        tri_inst, chain, lod = buf.get("tri_inst", (k,), np.int32), buf.get("chain", (k,), np.int64), buf.get("lod", (k,))
        project(pack["faces"], pack["uvs"], pack["colors"], pack["face_chain"], pack["face_texels"],
                pack["mesh_vertex"], inst["mesh"], inst["rgb"], inst["double"], inst_vertex, eye, float(camera.near),
                fb.width, fb.height, float(self.lod_bias), world, chunk_inst, chunk_first, chunk_end, whole, cut,
                off_whole, off_cut, xs, ys, inv_w, attrs, tri_inst, chain, lod)
        if not k:
            return None
        return {"xs": xs[:k], "ys": ys[:k], "inv_w": inv_w[:k], "attrs": attrs[:k], "tri_inst": tri_inst[:k],
                "ident": inst["ident"], "emissive": inst["emissive"], "chain": chain[:k], "lod": lod[:k],
                "textures": pack["textures"], "lights": np.ascontiguousarray(_light_rows(lights).reshape(-1, LIGHT_COLUMNS)),
                "eye": eye}

    # ----- sampling and shading ---------------------------------------------------------

    def _accumulate(self, scene, pattern, name, pixels=None, depth=None, ids=None):
        """Render every sample position in `pattern`, in all pixels or just the given flat pixel indices.

        Returns _resolve()'s per-pixel outputs (rgb, cover, more), in buffers named
        after `name`; the depth and object id of each pixel's nearest sample go into
        `depth` and `ids` if given.
        """
        fb, buf = self.framebuffer, self._buffers
        if pixels is None:
            pixels = slots = buf.arange(fb.width * fb.height)
        else:
            slots = buf.get("slots", (fb.width * fb.height,), np.int64)
            slots.fill(-1)
            slots[pixels] = buf.arange(fb.width * fb.height)[:len(pixels)]
        n, m = len(pattern), len(pixels)
        sample_depth = buf.get("sample_depth", (m, n))
        sample_depth.fill(0.0)
        tris = buf.get("tris", (m, n), np.int32)
        tris.fill(-1)
        rasterize(sample_depth, tris, fb.width, fb.height, scene["xs"], scene["ys"], scene["inv_w"],
                  np.array(pattern, dtype=float), slots, *scene["bands"])
        out = {"rgb": buf.get(name + "_rgb", (m, 3)), "cover": buf.get(name + "_cover", (m,), np.int64),
               "more": buf.get(name + "_more", (m,), np.bool_)}
        depth = buf.get("near_depth", (m,)) if depth is None else depth
        ids = buf.get("near_id", (m,), np.int32) if ids is None else ids
        _resolve(tris, sample_depth, pixels, fb.width, scene["xs"], scene["ys"], scene["inv_w"], scene["attrs"],
                 scene["tri_inst"], scene["ident"], scene["emissive"], scene["chain"], scene["lod"], *scene["textures"],
                 scene["lights"], scene["eye"], EDGE_CONTRAST, buf.get("sample_rgb", (m, n, 3)), out["rgb"], out["cover"],
                 depth, ids, out["more"])
        return out
