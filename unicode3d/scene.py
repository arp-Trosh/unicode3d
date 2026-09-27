# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
"""Camera, lights, scene objects and the render pipeline."""
from dataclasses import dataclass, field

import numpy as np
from numba import njit, prange

from .color import to_linear_rgb
from .mesh import Mesh
from .raster import FrameBuffer, barycentric, project, rasterize
from .texture import pack as pack_textures, sample as sample_texture
from .transforms import look_at, normalize, perspective, quat_identity, quat_to_matrix


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
    direction: np.ndarray = _vec3(0.3, -1.0, -0.5)  # the way the light travels
    ambient: float = 0.3    # light levels are perceived brightness, 0..1
    diffuse: float = 0.7
    specular: float = 0.35  # strength of the Blinn-Phong highlight, which is white whatever the surface colour
    shininess: float = 24.0


@dataclass
class Object3D:
    mesh: Mesh
    position: np.ndarray = _vec3(0.0, 0.0, 0.0)
    rotation: np.ndarray = field(default_factory=quat_identity)
    scale: float = 1.0
    color: object = 0       # a named Color, or (r, g, b) sRGB as 0..255 ints or 0..1 floats
    visible: bool = True
    double_sided: bool = False  # draw back faces too (for meshes with inconsistent winding)




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
def _shade(t, b0, b1, b2, inv_w, attrs, albedo, chain, lod, texels, levels, first, lighting, out):
    """Linear RGB, into out (3,), of triangle t at barycentric weights b: Blinn-Phong with a white
    highlight on a surface of colour `albedo`, textured if the triangle has a mipmap chain."""
    # Perspective-correct interpolation of world position, normal and uv.
    w0, w1, w2 = b0 * inv_w[t, 0], b1 * inv_w[t, 1], b2 * inv_w[t, 2]
    ws = w0 + w1 + w2
    at = attrs[t]

    def lerp(k):
        return (w0 * at[0, k] + w1 * at[1, k] + w2 * at[2, k]) / ws

    nx, ny, nz = lerp(3), lerp(4), lerp(5)
    nl = max(np.sqrt(nx * nx + ny * ny + nz * nz), 1e-12)
    nx, ny, nz = nx / nl, ny / nl, nz / nl
    lx, ly, lz = lighting[4], lighting[5], lighting[6]
    ex, ey, ez = lighting[7] - lerp(0), lighting[8] - lerp(1), lighting[9] - lerp(2)
    el = max(np.sqrt(ex * ex + ey * ey + ez * ez), 1e-12)
    hx, hy, hz = lx + ex / el, ly + ey / el, lz + ez / el
    hl = max(np.sqrt(hx * hx + hy * hy + hz * hz), 1e-12)
    diffuse = lighting[0] + lighting[1] * max(nx * lx + ny * ly + nz * lz, 0.0)
    spec = lighting[2] * max((nx * hx + ny * hy + nz * hz) / hl, 0.0) ** lighting[3]
    r, g, b = albedo[0], albedo[1], albedo[2]
    if chain[t] >= 0:
        tr, tg, tb = sample_texture(texels, levels, first, chain[t], lerp(6), lerp(7), lod[t])
        r, g, b = r * tr, g * tg, b * tb
    # Light levels are perceived brightness (0.5 looks half as bright), as artists tune them, so they
    # are decoded like any sRGB value; everything after this point works in linear light.
    k = _decode(min(max(diffuse, 0.0), 1.0))
    out[0] = min(max(r * k + spec, 0.0), 1.0)
    out[1] = min(max(g * k + spec, 0.0), 1.0)
    out[2] = min(max(b * k + spec, 0.0), 1.0)


@njit(cache=True, parallel=True)
def _resolve(tris, depth, pixels, width, xs, ys, inv_w, attrs, tri_obj, ident, albedo, chain, lod, texels, levels,
             first, lighting, contrast, sample_rgb, rgb, cover, near_depth, near_id, more):
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
                _shade(t, b0, b1, b2, inv_w, attrs, albedo[tri_obj[t]], chain, lod, texels, levels, first, lighting,
                       sample_rgb[c, s])
        r = g = b = 0.0
        lr = lg = lb = np.inf
        hr = hg = hb = -np.inf
        covered, mixed, best, best_id = 0, False, -1.0, 0
        first_id = ident[tri_obj[tris[c, 0]]] if tris[c, 0] >= 0 else 0
        for s in range(n):
            t = tris[c, s]
            sid = ident[tri_obj[t]] if t >= 0 else 0
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
    """

    def __init__(self, width, height, cell_pixels=(1, 2), cell_aspect=0.5, samples=4, edge_samples=8,
                 fog=0.3, outline=0.55, lod_bias=-0.5):
        for n in (samples, edge_samples):
            if n not in SAMPLE_PATTERNS and n != 0:
                raise ValueError(f"sample counts must be one of {sorted(SAMPLE_PATTERNS)}, not {n}")
        self.cell_aspect = cell_aspect
        self.samples = samples
        self.edge_samples = edge_samples
        self.fog = fog
        self.outline = outline
        self.lod_bias = lod_bias
        self.width = self.height = 0
        self.cell_pixels = tuple(cell_pixels)
        self.framebuffer = FrameBuffer(0, 0, cell_pixels)
        self.view_proj = None
        self.draws = 0     # renders that rasterized the scene, rather than reusing an unchanged frame
        self._last = None  # what the framebuffer shows, as _scene_state() gives it
        self._textures = None  # the mipmap chains in use, and pack_textures() of them
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

    def invalidate(self):
        """Make the next render draw afresh, e.g. after editing a mesh's arrays in place."""
        self._last = None

    def _scene_state(self, objects, camera, light):
        """Everything a render depends on: (values compared by equality, objects compared by identity).

        Meshes, their arrays and textures count as changed when replaced, as in Mesh's own caches;
        edits made inside them are not seen (call invalidate() after those).
        """
        values = [self.width, self.height, self.cell_pixels, self.cell_aspect, self.samples, self.edge_samples,
                  self.fog, self.outline, self.lod_bias,
                  np.asarray(camera.position, float).tobytes(), np.asarray(camera.target, float).tobytes(),
                  np.asarray(camera.up, float).tobytes(), camera.fov, camera.near, camera.far,
                  np.asarray(light.direction, float).tobytes(), light.ambient, light.diffuse, light.specular,
                  light.shininess]
        refs = []
        for obj in objects:
            m = obj.mesh
            values += [obj.visible, obj.double_sided, float(obj.scale), to_linear_rgb(obj.color).tobytes(),
                       np.asarray(obj.position, float).tobytes(), np.asarray(obj.rotation, float).tobytes()]
            refs += [m, m.vertices, m.faces, m.uvs, m.materials, *m.textures] if m is not None else [None]
        return values, refs

    def render(self, objects, camera, light):
        """Draw the objects; returns the framebuffer (reused by the next render: copy it to keep it).

        When nothing has changed since the last render (see _scene_state), the
        framebuffer is returned as it is, so a still scene costs almost nothing.
        """
        state = self._scene_state(objects, camera, light)
        last = self._last
        if (last is not None and last[0] == state[0] and len(last[1]) == len(state[1])
                and all(a is b for a, b in zip(last[1], state[1]))):
            return self.framebuffer
        self._last = None
        fb = self._draw(objects, camera, light)
        self.draws += 1
        self._last = state
        return fb

    def _draw(self, objects, camera, light):
        fb = self.framebuffer
        fb.clear()
        if self.width < 1 or self.height < 1:
            return fb
        aspect = self.width * self.cell_aspect / self.height
        self.view_proj = perspective(np.radians(camera.fov), aspect, camera.near, camera.far) @ camera.view_matrix()
        scene = self._geometry(objects, camera, light)
        if scene is None:
            return fb

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
        return fb

    # ----- geometry --------------------------------------------------------------------

    def _geometry(self, objects, camera, light):
        """Every visible object's triangles in screen space, in one set of arrays in render-list order,
        with what shading them needs; None if there are none."""
        fb, buf = self.framebuffer, self._buffers
        drawn = [(i, obj) for i, obj in enumerate(objects) if obj.visible and len(obj.mesh.faces)]
        room = 2 * sum(len(obj.mesh.faces) for _, obj in drawn)  # clipping can split a face in two
        xs, ys, inv_w = buf.get("xs", (room, 3)), buf.get("ys", (room, 3)), buf.get("inv_w", (room, 3))
        attrs, src = buf.get("attrs", (room, 3, 8)), buf.get("src", (room,), np.int64)
        tri_obj, chain, lod = buf.get("tri_obj", (room,), np.int32), buf.get("chain", (room,), np.int64), buf.get("lod", (room,))
        eye = np.asarray(camera.position, dtype=float)
        ident, albedo, chains, chain_of = [], [], [], {}
        k = 0
        for i, obj in drawn:
            mesh = obj.mesh
            textured = mesh.materials is not None and bool(mesh.textures)
            # One set of argument types and layouts, so the kernel compiled by compile_kernels() serves every
            # mesh (numpy returns the arrays themselves when they already fit).
            end = project(np.ascontiguousarray(mesh.vertices, np.float64),
                          np.ascontiguousarray(mesh.vertex_normals(), np.float64),
                          np.ascontiguousarray(mesh.faces, np.int64),
                          np.ascontiguousarray(mesh.uvs, np.float64) if textured else _NO_UVS, textured, quat_to_matrix(obj.rotation), float(obj.scale),
                          np.asarray(obj.position, dtype=float), self.view_proj, eye, obj.double_sided, camera.near,
                          fb.width, fb.height, buf.get("vertices", (len(mesh.vertices), 10)),
                          xs, ys, inv_w, attrs, src, k)
            if end == k:
                continue
            part = slice(k, end)
            tri_obj[part] = len(ident)
            ident.append(i + 1)
            albedo.append(to_linear_rgb(obj.color))
            if textured:
                table = np.empty(len(mesh.textures), np.int64)
                for m in range(len(mesh.textures)):
                    levels = mesh.mipmaps(m)
                    if id(levels) not in chain_of:
                        chain_of[id(levels)] = len(chains)
                        chains.append(levels)
                    table[m] = chain_of[id(levels)]
                chain[part] = table[mesh.materials[src[part]]]
                lod[part] = self._mip_levels(mesh, src[part], xs[part], ys[part], attrs[part, :, 6:8])
            else:
                chain[part], lod[part] = -1, 0.0
            k = end
        if not k:
            return None
        # Packing copies every texel, so it is kept for as long as the same mipmaps are in use.
        packed = self._textures
        if packed is None or len(packed[0]) != len(chains) or any(a is not b for a, b in zip(packed[0], chains)):
            packed = self._textures = (chains, pack_textures(chains))
        lighting = np.array([light.ambient, light.diffuse, light.specular, light.shininess,
                             *-normalize(light.direction), *eye])
        return {"xs": xs[:k], "ys": ys[:k], "inv_w": inv_w[:k], "attrs": attrs[:k], "tri_obj": tri_obj[:k],
                "ident": np.array(ident, np.int32), "albedo": np.array(albedo), "chain": chain[:k], "lod": lod[:k],
                "textures": packed[1], "lighting": lighting}

    def _mip_levels(self, mesh, src, xs, ys, uv):
        """Mip level for each triangle: log2 of how many texels span one pixel across it."""
        px_area = np.abs((xs[:, 1] - xs[:, 0]) * (ys[:, 2] - ys[:, 0]) - (ys[:, 1] - ys[:, 0]) * (xs[:, 2] - xs[:, 0]))
        d1, d2 = uv[:, 1] - uv[:, 0], uv[:, 2] - uv[:, 0]
        uv_area = np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])
        texels = np.array([t.shape[0] * t.shape[1] for t in mesh.textures], dtype=float)[mesh.materials[src]]
        ratio = uv_area * texels / np.maximum(px_area, 1e-9)
        return 0.5 * np.log2(np.maximum(ratio, 1e-9)) + self.lod_bias

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
                  np.array(pattern, dtype=float), slots)
        out = {"rgb": buf.get(name + "_rgb", (m, 3)), "cover": buf.get(name + "_cover", (m,), np.int64),
               "more": buf.get(name + "_more", (m,), np.bool_)}
        depth = buf.get("near_depth", (m,)) if depth is None else depth
        ids = buf.get("near_id", (m,), np.int32) if ids is None else ids
        _resolve(tris, sample_depth, pixels, fb.width, scene["xs"], scene["ys"], scene["inv_w"], scene["attrs"],
                 scene["tri_obj"], scene["ident"], scene["albedo"], scene["chain"], scene["lod"], *scene["textures"],
                 scene["lighting"], EDGE_CONTRAST, buf.get("sample_rgb", (m, n, 3)), out["rgb"], out["cover"],
                 depth, ids, out["more"])
        return out
