# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
"""The Renderer: draws a scene's objects (scene.Object3D) into a framebuffer of pixels finer than terminal cells.

It packs the meshes and poses into arrays, projects every triangle, rasterizes several samples a pixel, shades
them (shading.py), blends see-through surfaces over the solid ones, and adds what mirrors show (mirrors.py),
shadows (shadows.py), fog and outlines, and the background.
"""
import math
from dataclasses import dataclass

import numpy as np

from .background import SKYBOX_FACES, background_args, fill_background, fog_args
from .color import cached_linear_rgb
from .lights import LIGHT_COLUMNS, as_lights, light_rows
from .mirrors import Mirrors
from .raster import (ATTRS, CLEAR, CUT, FACE_CHUNK, ROW_BAND, SAMPLE_PATTERNS, SOLID, VERTEX_CHUNK, FrameBuffer,
                     bin_bands, count_bands, project, rasterize, rasterize_layers, transform, upscale, NO_CLIP)
from .shading import blend, post_effects, resolve
from .shadows import ShadowMaps
from .texture import alpha_kind, pack as pack_textures
from .threads import kernel_lock
from .transforms import normalize, perspective, scale3, view_axes

MAX_PIXELS = 1920 * 1080  # the default Renderer.max_pixels
EDGE_CONTRAST = 0.03  # linear-light spread among a pixel's first samples that marks it for more
HAZE_TEXELS = 4       # how many texels across a sky box face fog fades into (see shading.post_effects)


@dataclass
class Pick:
    """What Renderer.pick found: the object, the point on its surface, and how far that is from the camera."""
    object: object
    position: np.ndarray
    distance: float


@dataclass
class Anchor:
    """Where Renderer.anchor found a point in the frame: the cell (x, y, counted from the frame's top-left), how far
    the point is from the camera, whether something drawn is in front of it, and whether it was off the frame (or
    behind the camera) and moved to the frame's edge."""
    x: int
    y: int
    distance: float
    hidden: bool
    edge: bool


LABEL_MARGIN = 0.01  # how much nearer than a point (a fraction of its distance) a surface must be to hide it


def _objects_of(owner):
    """The ids of the objects in owner: an Object3D, a Model, or a list of either (None: none)."""
    if owner is None:
        return set()
    if hasattr(owner, "mesh"):
        return {id(owner)}
    return {id(obj) for part in owner for obj in ([part] if hasattr(part, "mesh") else part)}


def _mesh_key(mesh):
    """What a packed mesh depends on, compared by identity (see Renderer._scene_state)."""
    return (mesh, mesh.vertices, mesh.faces, mesh.uvs, mesh.materials, mesh.vertex_colors, mesh.face_colors,
            mesh.normals, *mesh.textures)


def _flat_plane(vertices, faces, radius):
    """The plane (unit normal, d: n . x + d = 0) all of a mesh's faces lie on, to within a millionth of its
    size; NaNs if they don't (it is not flat)."""
    vertices = vertices[np.unique(faces)] if len(faces) else vertices[:0]
    if len(vertices) < 3 or not np.isfinite(vertices).all():  # (a corner at NaN or infinity: no plane)
        return (np.nan,) * 4
    centre = vertices.mean(axis=0)
    offsets = vertices - centre
    if not np.isfinite(offsets).all():  # (corners so far out that their sum overflows: an SVD of infinities can
        return (np.nan,) * 4            # loop forever)
    _, spread, axes = np.linalg.svd(offsets, full_matrices=False)
    normal = axes[-1]
    if np.abs((vertices - centre) @ normal).max() > 1e-6 * max(radius, 1e-12):
        return (np.nan,) * 4
    return (*normal, -float(normal @ centre))


def _pack_meshes(meshes):
    """The meshes' arrays packed together for raster.project, and their textures for sampling."""
    chains, chain_of, kinds = [], {}, []
    verts, normals, faces, uvs, colors, face_chain, face_texels, face_kind = [], [], [], [], [], [], [], []
    mesh_vertex, mesh_face, spheres, clear, tints, planes = [0], [0], [], [], [], []
    for mesh in meshes:
        mesh.check()  # (the kernels index these arrays by each other's sizes without checking)
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
                    kinds.append(alpha_kind(levels))
                table[m] = chain_of[id(levels)]
            materials = np.asarray(mesh.materials)
            face_chain.append(table[materials])
            face_kind.append(np.array(kinds, np.int8)[table[materials]])
            sizes = np.array([np.shape(t)[0] * np.shape(t)[1] for t in mesh.textures], dtype=float)
            face_texels.append(sizes[materials])
            uvs.append(np.asarray(mesh.uvs, dtype=float))
        else:
            face_chain.append(np.full(n_faces, -1, np.int64))
            face_kind.append(np.zeros(n_faces, np.int8))
            face_texels.append(np.zeros(n_faces))
            uvs.append(np.zeros((n_faces, 3, 2)))
        corner = mesh.corner_colors()
        colors.append(np.ones((n_faces, 3, 4)) if corner is None else corner)
        # Whether light goes through any of it (see-through colours or textures), and whether its far side
        # shows through anything (those, or holes in its textures).
        tints.append((corner is not None and bool((corner[:, :, 3] < 1.0).any())) or bool((face_kind[-1] == 2).any()))
        clear.append(tints[-1] or bool(face_kind[-1].any()))
        centre, radius = mesh.bounds()
        spheres.append((*centre, radius))
        planes.append(_flat_plane(verts[-1], np.asarray(mesh.faces), radius))
        mesh_vertex.append(mesh_vertex[-1] + len(mesh.vertices))
        mesh_face.append(mesh_face[-1] + n_faces)
    c = np.ascontiguousarray
    return {"vertices": c(np.concatenate(verts), np.float64), "normals": c(np.concatenate(normals), np.float64),
            "faces": c(np.concatenate(faces), np.int64), "uvs": c(np.concatenate(uvs), np.float64),
            "colors": c(np.concatenate(colors), np.float64), "face_chain": c(np.concatenate(face_chain), np.int64),
            "face_kind": c(np.concatenate(face_kind), np.int8),
            "face_texels": c(np.concatenate(face_texels), np.float64),
            "mesh_vertex": np.array(mesh_vertex, np.int64), "mesh_face": np.array(mesh_face, np.int64),
            "spheres": np.array(spheres, np.float64).reshape(-1, 4), "clear": np.array(clear, np.bool_),
            "tints": np.array(tints, np.bool_), "planes": np.array(planes, np.float64).reshape(-1, 4),
            "textures": pack_textures(chains)}


def _chunks(counts, starts, size):
    """Split each instance's run of counts[i] items, starting at starts[i], into chunks of at most `size`:
    (instance, first, end) arrays, in instance order."""
    per = (counts + size - 1) // size
    inst = np.repeat(np.arange(len(counts)), per)
    within = np.arange(len(inst)) - np.repeat(np.cumsum(per) - per, per)
    first = starts[inst] + within * size
    end = np.minimum(first + size, starts[inst] + counts[inst])
    return inst.astype(np.int64), first.astype(np.int64), end.astype(np.int64)


_NO_TURN = np.array([1.0, 0.0, 0.0, 0.0])


def _rotations(quat):
    """Rotation matrices (N, 3, 3) of unit quaternions (N, 4), w first."""
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                     2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                     2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], axis=1).reshape(-1, 3, 3)


class _Buffers:
    """Arrays kept from one render to the next, so drawing a frame allocates (and page-faults in) no new memory."""

    def __init__(self):
        self._arrays = {}

    def get(self, name, shape, dtype=np.float64):
        """An array of this shape with undefined contents, in the memory of the last one of this name if it fits."""
        size = int(math.prod(shape))  # (np.prod takes microseconds, and this runs a hundred times a frame)
        a = self._arrays.get(name)
        if a is None or a.dtype != dtype or a.size < size:
            a = self._arrays[name] = np.empty(size + size // 4, dtype)  # room to grow a little
        return a[:size].reshape(shape)

    def arange(self, n):
        """0..n-1 (not to be modified)."""
        a = self._arrays.get(("arange", n))
        if a is None:
            if len(self._arrays) > 64:  # after many resizes
                for key in [key for key in self._arrays if isinstance(key, tuple)]:
                    del self._arrays[key]
            a = self._arrays[("arange", n)] = np.arange(n)
        return a


class Renderer(ShadowMaps, Mirrors):
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
    fog: a Fog (see background.py), fading surfaces with their distance into the background or a colour;
    or a number, how much the farthest surfaces are dimmed relative to the nearest (depth cueing: it
    shifts as things come into view or leave it, which a Fog doesn't).
    outline: how much to darken the far side of depth edges, where one surface
    passes in front of another, so overlapping shapes stay distinct.
    lod_bias: added to every texture's mip level; negative is sharper, positive softer.
    background: drawn behind the scene: None (leave it empty, so the screen's background
    shows), a colour, or a Gradient, Sky or SkyBox (see background.py).
    shadow_size: texels across the shadow map of each Light with shadows, which covers
    everything in the scene as seen from the light. Larger is sharper and slower to draw.
    point_shadow_size: texels across each of the six faces of the cube shadow map of each
    PointLight with shadows, which together see all round it.
    shadow_softness: how far shadow edges are blurred, in shadow-map texels either way (at
    least over a pixel on screen, so that they look as smooth as the edges of shapes).
    shadows: False draws no shadows whatever the lights say (a graphics setting, e.g. for
    slow machines); True draws those of the lights that have them.
    transparency_layers: how many see-through surfaces (Object3D.opacity, or alpha in mesh
    colours) each pixel shows in front of the solid ones; nearer ones win when there are more.
    reflections: False draws no reflections whatever the objects' reflectivity (a graphics setting).
    mirror_bounces: how deep mirrors show each other (1: a mirror seen in a mirror shows the
    background reflected in it, not what is in front of it; at most 4). Deeper images stop early
    once they are too small or dim to see.
    max_pixels: the most pixels drawn (about 1080p by default). A bigger framebuffer (a
    full-screen terminal with a tiny font) is drawn at a lower resolution that fits, and
    stretched to size, so that time and memory stay bounded however big the terminal is;
    None or 0 for no limit. Working memory is a few hundred bytes a pixel.
    """

    def __init__(self, width, height, cell_pixels=(1, 2), cell_aspect=0.5, samples=4, edge_samples=8,
                 fog=0.3, outline=0.55, lod_bias=-0.5, background=None, shadow_size=1024, point_shadow_size=256,
                 shadow_softness=1.5, shadows=True, transparency_layers=4, reflections=True, mirror_bounces=1,
                 max_pixels=MAX_PIXELS):
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
        self.shadow_size = shadow_size
        self.point_shadow_size = point_shadow_size
        self.shadow_softness = shadow_softness
        self.shadows = shadows
        self.transparency_layers = transparency_layers
        self.reflections = reflections
        self.mirror_bounces = mirror_bounces
        self.width = self.height = 0
        self.cell_pixels = tuple(cell_pixels)
        self.max_pixels = max_pixels
        self.framebuffer = FrameBuffer(0, 0, cell_pixels)
        self._fb = self.framebuffer  # what is drawn into: the framebuffer itself, or a smaller one within max_pixels
        self._fb_for = None  # the (width, height, cell_pixels, max_pixels) _fb was made for
        self.view_proj = None
        self.draws = 0     # renders that rasterized the scene, rather than reusing an unchanged frame
        self._last = None  # what the framebuffer shows, as _scene_state() gives it
        self._pack = None  # (mesh keys, _pack_meshes() of those meshes) as last drawn
        self._objects = []  # the render list as last drawn, for pick()
        self._eye = self._axes = None  # the camera's position and its pixels' directions (view_axes) then
        self._ortho = 0.0  # for an orthographic camera, how far behind it the eye it is drawn from is (Camera.drawn_as)
        self._buffers = _Buffers()
        self._shadows = None  # (what the shadow maps depend on, the maps as _shadow_maps() returns them)
        self.resize(width, height)

    def resize(self, width, height, cell_pixels=None):
        """Set the size in cells, and optionally the pixels per cell (e.g. Screen.cell_pixels)."""
        cell_pixels = self.cell_pixels if cell_pixels is None else tuple(cell_pixels)
        width, height = max(int(width), 0), max(int(height), 0)  # (rows - 2 for a status line, in a 1-row terminal)
        if (width, height, cell_pixels) == (self.width, self.height, self.cell_pixels):
            return
        self.width, self.height, self.cell_pixels = width, height, cell_pixels
        self.framebuffer.cell_pixels = cell_pixels
        self.framebuffer.resize(width * cell_pixels[0], height * cell_pixels[1])
        self._fit()

    @property
    def drawn_size(self):
        """(width, height) in pixels of what render() draws: the framebuffer's size, or less where that is more
        than max_pixels (the picture is then stretched to the framebuffer's size)."""
        self._fit()
        return self._fb.width, self._fb.height

    def _fit(self):
        """Make the framebuffer drawn into: the framebuffer itself if it has at most max_pixels pixels, else a
        smaller one of about the same shape with about max_pixels (see Renderer)."""
        wanted = (self.width, self.height, self.cell_pixels, self.max_pixels)
        if wanted == self._fb_for:
            return
        self._fb_for = wanted
        out = self.framebuffer
        budget = int(self.max_pixels) if self.max_pixels else 0
        if budget <= 0 or out.width * out.height <= budget:
            self._fb = out
            return
        shrink = np.sqrt(budget / (out.width * out.height))
        w, h = max(int(out.width * shrink), 1), max(int(out.height * shrink), 1)
        if self._fb is out or (self._fb.width, self._fb.height) != (w, h):
            self._fb = FrameBuffer(w, h, out.cell_pixels)

    def project(self, point):
        """Cell coordinates (x, y) of a world point as of the last render, or None if behind the camera."""
        if self.view_proj is None:
            return None
        with np.errstate(all="ignore"):
            if self._ortho:  # (exactly orthographic, rather than as drawn from far back: see Camera.drawn_as)
                nx, ny, ahead = self._in_view(np.asarray(point, dtype=float))[:3]
                if not ahead > 0.0:
                    return None
            else:
                clip = self.view_proj @ np.array([*point, 1.0])
                if not clip[3] > 1e-6:  # (also NaN)
                    return None
                nx, ny = clip[0] / clip[3], clip[1] / clip[3]
            return (nx + 1.0) * 0.5 * self.width, (1.0 - ny) * 0.5 * self.height

    def _in_view(self, p):
        """Where world point p is in the last render's view: (x, y in normalized device coordinates, how far ahead
        of the camera it is, x and y unscaled by distance, distance from the camera). Call under np.errstate."""
        a = self._axes
        # (numpy's numbers, not Python's: a division by zero gives inf or NaN rather than an exception)
        if self._ortho:  # (the view is _ortho from the eye to the camera's plane, a[1] and a[2] wide there)
            v = p - self._camera_at
            ahead = v @ a[0]
            sx, sy = (v @ a[1]) / (a[1] @ a[1]) / self._ortho, (v @ a[2]) / (a[2] @ a[2]) / self._ortho
            return sx, sy, ahead, sx, sy, ahead
        v = p - self._eye
        ahead = v @ a[0]
        sx, sy = (v @ a[1]) / (a[1] @ a[1]), (v @ a[2]) / (a[2] @ a[2])
        return sx / ahead, sy / ahead, ahead, sx, sy, np.sqrt(v @ v)

    def anchor(self, point, owner=None, clamp=False):
        """Where a world point is in the last render's frame, for drawing a label or a bar there (Screen.label): an
        Anchor, or None before the first render, and for a point behind the camera or outside the frame unless
        clamp is set, which moves it to the frame's edge instead, in the direction it lies (for markers pointing at
        things out of view).

        hidden tells whether a surface drawn there is in front of the point (by more than LABEL_MARGIN of its
        distance and a pixel or two); owner, an object or objects (or a Model) the point belongs to, doesn't count,
        so a label on an object's own surface isn't hidden by it."""
        if self.view_proj is None or self.width < 1 or self.height < 1:
            return None
        p = np.asarray(point, dtype=float).reshape(-1)
        if p.shape != (3,):
            raise ValueError(f"point must be three numbers, not {np.shape(point)}")
        with np.errstate(all="ignore"):
            nx, ny, ahead, sx, sy, distance = self._in_view(p)
            ahead_ok = ahead > 0.0  # (NaN fails)
            inside = ahead_ok and -1.0 <= nx <= 1.0 and -1.0 <= ny <= 1.0
            if not inside:
                if not clamp:
                    return None
                if not ahead_ok:
                    nx, ny = sx, sy  # behind: towards where it lies
                most = max(abs(nx), abs(ny))
                if not (most > 0.0 and most < np.inf):
                    nx, ny, most = 0.0, -1.0, 1.0  # (straight behind, or numbers that aren't finite: the bottom)
                nx, ny = nx / most, ny / most
            x = min(max((nx + 1.0) * 0.5 * self.width, 0.0), self.width - 1.0)
            y = min(max((1.0 - ny) * 0.5 * self.height, 0.0), self.height - 1.0)
            hidden = False
            if inside:
                fb = self.framebuffer
                pw, ph = self.cell_pixels
                px = int(min(max((nx + 1.0) * 0.5 * fb.width, 0.0), fb.width - 1.0))
                py = int(min(max((1.0 - ny) * 0.5 * fb.height, 0.0), fb.height - 1.0))
                depth = float(fb.depth[py, px])
                if depth > 0.0 and int(fb.ids[py, px]) > 0:
                    surface = 1.0 / np.float64(depth) - self._ortho  # how far ahead the surface drawn there is
                    pixel = 2.0 * abs(self._view_tan[1]) * (ahead + self._ortho) / fb.height  # (in the world, there)
                    if surface < ahead - LABEL_MARGIN * ahead - 2.0 * pixel:
                        drawn = self._objects[int(fb.ids[py, px]) - 1]
                        hidden = id(drawn) not in _objects_of(owner)
        return Anchor(int(x), int(y), float(distance), bool(hidden), not inside)

    def ray(self, x, y):
        """(origin, unit direction) in the world of the line of sight through cell (x, y) as of the last
        render (cells count from the frame's top-left; fractions are fine, and x + 0.5, y + 0.5 is a
        cell's centre); None before the first render."""
        if self.view_proj is None or self.width < 1 or self.height < 1:
            return None
        nx, ny = x / self.width * 2.0 - 1.0, 1.0 - y / self.height * 2.0
        with np.errstate(all="ignore"):  # (a degenerate camera gives NaN, not an exception)
            a = self._axes
            if self._ortho:  # parallel rays, from the camera's plane (the eye is _ortho behind it)
                forward = normalize(a[0])
                return self._camera_at + self._ortho * (nx * a[1] + ny * a[2]), forward
            return self._eye.copy(), normalize(a[0] + nx * a[1] + ny * a[2])

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
        with np.errstate(all="ignore"):
            if self._ortho:  # 1/w less what is behind the camera's plane: the distance from that plane
                distance = 1.0 / depth[py, px] - self._ortho
            else:
                distance = 1.0 / depth[py, px] / max(float(direction @ self._axes[0]), 1e-12)
            return Pick(obj, origin + direction * distance, distance)

    def invalidate(self):
        """Make the next render draw afresh, e.g. after editing a mesh's arrays in place."""
        self._last = None
        self._pack = None
        self._shadows = None

    def _instances(self, objects):
        """The objects to draw, as arrays: their meshes' place in the pack, world poses (linear (3, 3) and
        position), colours, materials, flags and ids (render-list index + 1). Packs the meshes afresh if the set
        of meshes has changed."""
        meshes, mesh_of, keys = [], {}, []
        mesh_idx, pos, quat, scale, placed_by, rgb, double, ident, emissive, cast, alpha, shine, spec, shiny = (
            [], [], [], [], [], [], [], [], [], [], [], [], [], [])
        for i, obj in enumerate(objects):
            mesh = obj.mesh
            if mesh is None or not len(mesh.faces):
                continue
            if not obj.opacity > 0.0:
                continue  # wholly clear (or NaN): nothing to draw, nor any shadow
            if obj.parent is None:
                if not obj.visible:
                    continue
                p, q, sc = obj.position, obj.rotation, obj.scale
                sc = (sc, sc, sc) if isinstance(sc, (int, float, np.number)) else tuple(scale3(sc))
            else:
                linear, p, visible = obj.world_matrix()
                if not visible:
                    continue
                q, sc = _NO_TURN, (1.0, 1.0, 1.0)
                placed_by.append((len(pos), linear))
            m = mesh_of.get(id(mesh))
            if m is None:
                m = mesh_of[id(mesh)] = len(meshes)
                meshes.append(mesh)
                keys.append(_mesh_key(mesh))
            mesh_idx.append(m)
            pos.append(p)
            quat.append(q)
            scale.append(sc)
            rgb.append(cached_linear_rgb(obj.color))
            double.append(obj.double_sided)
            ident.append(i + 1)
            emissive.append(obj.emissive)
            cast.append(obj.cast_shadows)
            alpha.append(obj.opacity)
            shine.append(obj.reflectivity)
            spec.append(obj.specular)
            shiny.append(0.0 if obj.shininess is None else obj.shininess)
        if not meshes:
            return None
        where = np.array(pos, np.float64).reshape(-1, 3)
        turn = np.array(quat, dtype=np.float64).reshape(-1, 4)
        turn = turn / np.maximum(np.linalg.norm(turn, axis=1, keepdims=True), 1e-300)
        linear = _rotations(turn) * np.array(scale, np.float64).reshape(-1, 1, 3)
        for row, lin in placed_by:  # through parents: worked out whole by world_matrix()
            linear[row] = lin
        placed = np.isfinite(where).all(axis=1) & np.isfinite(linear).all(axis=(1, 2))
        if not placed.all():  # NaN or infinite in a pose (a physics blow-up, say): nowhere to draw it
            keep = np.flatnonzero(placed)
            if not len(keep):
                return None
            mesh_idx, rgb, double, ident, emissive, cast, alpha, shine, spec, shiny = (
                [values[i] for i in keep]
                for values in (mesh_idx, rgb, double, ident, emissive, cast, alpha, shine, spec, shiny))
            where, linear = where[keep], linear[keep]
            used = sorted(set(mesh_idx))
            renumber = {m: j for j, m in enumerate(used)}
            meshes, keys = [meshes[m] for m in used], [keys[m] for m in used]
            mesh_idx = [renumber[m] for m in mesh_idx]
        packed = self._pack
        if packed is None or len(packed[0]) != len(keys) or any(
                len(a) != len(b) or any(x is not y for x, y in zip(a, b)) for a, b in zip(packed[0], keys)):
            packed = self._pack = (keys, _pack_meshes(meshes))
        mesh_idx = np.array(mesh_idx, np.int64)
        alpha = np.clip(np.array(alpha, np.float64), 0.0, 1.0)
        # See-through objects, and those with holes, always show their far side, through their near one.
        double = np.array(double, np.bool_) | (alpha < 1.0) | packed[1]["clear"][mesh_idx]

        def numbers(values):  # (NaN as 0: a NaN colour or glow would reach the picture)
            a = np.array(values, np.float64)
            return np.where(np.isnan(a), 0.0, a) if np.isnan(a).any() else a

        return {"mesh": mesh_idx, "pos": where, "lin": np.ascontiguousarray(linear),
                "flip": np.linalg.det(linear) < 0.0,
                "rgb": numbers(rgb).reshape(-1, 3), "double": double, "alpha": alpha,
                "shine": np.clip(numbers(shine), 0.0, 1.0),
                "specular": np.maximum(numbers(spec), 0.0),
                "shininess": np.maximum(numbers(shiny), 0.0),
                "ident": np.array(ident, np.int32),
                "emissive": numbers(emissive), "cast": np.array(cast, np.bool_), "pack": packed[1],
                "keys": keys}

    def _scene_state(self, inst, camera):
        """Everything a render depends on: (values compared by equality, objects compared by identity).

        Meshes, their arrays and textures count as changed when replaced, as in Mesh's own caches;
        edits made inside them are not seen (call invalidate() after those).
        """
        fog = fog_args(self.fog)
        values = [self.width, self.height, self.cell_pixels, self.max_pixels, self.cell_aspect, self.samples,
                  self.edge_samples, self.transparency_layers, self.reflections, self.mirror_bounces,
                  fog[:3] + (fog[3].tobytes(), fog[4]), self.outline, self.lod_bias, self.shadow_size,
                  self.point_shadow_size, self.shadow_softness,
                  np.asarray(camera.position, float).tobytes(), np.asarray(camera.target, float).tobytes(),
                  np.asarray(camera.up, float).tobytes(), camera.fov, camera.near, camera.far,
                  self._light_rows.tobytes()]
        if inst is None:
            return values, []
        values += [inst[k].tobytes()
                   for k in ("mesh", "pos", "lin", "rgb", "double", "ident", "emissive", "cast", "alpha",
                             "shine", "specular", "shininess")]
        return values, [x for key in inst["keys"] for x in (*key, None)]

    def render(self, objects, camera, lights):
        """Draw the objects; returns the framebuffer (reused by the next render: copy it to keep it).

        lights: a Light or PointLight, or a list of them (their light adds up).
        Objects with a parent are placed through it (see Node). When nothing has
        changed since the last render (see _scene_state), the framebuffer is
        returned as it is, so a still scene costs almost nothing. Floating-point trouble (NaN or infinite
        coordinates) is drawn around rather than warned about: a warning would be printed over the picture.
        """
        with np.errstate(all="ignore"), kernel_lock():
            return self._render(objects, camera, lights)

    def _render(self, objects, camera, lights):
        self._camera_at = np.asarray(camera.position, dtype=float).copy()  # (where an orthographic camera's rays start)
        camera, self._ortho = camera.drawn_as()  # (an orthographic camera is drawn as a perspective one, far back)
        lights = as_lights(lights)
        self._fit()
        self._objects = objects = list(objects)  # (once: objects may be a generator)
        inst = self._instances(objects)
        self._light_rows = light_rows(lights, self.shadows)
        aspect = self.width * self.cell_aspect / max(self.height, 1)
        bg_args, bg_state = background_args(self.background, camera, aspect, self._fb.height)
        state = self._scene_state(inst, camera)
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
        fb = self._fb
        if self.width < 1 or self.height < 1:
            self.framebuffer.clear()
            return self.framebuffer
        aspect = self.width * self.cell_aspect / self.height
        view = camera.view_matrix()
        self.view_proj = perspective(np.radians(camera.fov), aspect, camera.near, camera.far) @ view
        self._eye = np.asarray(camera.position, dtype=float).copy()
        self._axes = view_axes(view, np.radians(camera.fov), aspect)  # (for ray(), and what mirrors show of the sky)
        tan_y = float(np.tan(np.radians(camera.fov) / 2))
        self._view_tan = (tan_y * aspect, tan_y)  # for fog: the direction each pixel looks in
        kind, colors, basis, texels, levels, first, lod = bg_args
        # What shiny surfaces reflect: the background (kind -1 when reflections are off).
        self._sky = (int(kind) if self.reflections else -1, colors, SKYBOX_FACES, texels, levels, first, float(lod))
        # What fog fades into, behind a sky box: the sky blurred to about HAZE_TEXELS across a face.
        haze_lod = float(np.log2(max(levels[0, 1] / HAZE_TEXELS, 1.0)))
        self._haze = (int(kind), colors, basis, SKYBOX_FACES, texels, levels, first, haze_lod)
        scene = self._geometry(inst, camera, lights) if inst is not None else None
        if scene is not None:
            self._shade_scene(scene)  # (which writes every pixel)
        else:
            fb.clear()
        fill_background(fb.rgb, fb.alpha, kind, colors, basis, SKYBOX_FACES, texels, levels, first, lod)
        out = self.framebuffer
        if fb is not out:  # drawn smaller, within max_pixels: stretched to the size the screen needs
            upscale(fb.rgb, fb.alpha, fb.depth, fb.ids, out.rgb, out.alpha, out.depth, out.ids)
        return out

    def _shade_scene(self, scene):
        """Rasterize and shade the scene's triangles into the framebuffer, with extra samples at edges,
        then fog and outlines."""
        fb = self._fb
        # Which solid triangles (cut-outs too) may touch each band of rows, for both rasterizing passes.
        scene["bands"] = self._bins(scene["xs"], scene["ys"], scene["see"], SOLID | CUT, fb.height, "")
        n = self.samples
        # The base samples give every pixel its colour, coverage, depth and id; edge samples then add to some.
        base = self._accumulate(scene, SAMPLE_PATTERNS[n], "base", depth=fb.depth.reshape(-1), ids=fb.ids.reshape(-1),
                                frame_samples=0)
        layers = self._layers(scene, SAMPLE_PATTERNS[n], base["sample_depth"]) if (scene["see"] == 1).any() else None
        if self.edge_samples and n > 1:
            pixels = np.flatnonzero(base["more"])
            if len(pixels):
                # A different pattern from the base one: the same one turned a quarter.
                pattern = tuple((1.0 - y, x) for x, y in SAMPLE_PATTERNS[self.edge_samples])
                extra = self._accumulate(scene, pattern, "edge", pixels, frame_samples=n)
                rows = self._buffers.get("edge_rows", (fb.width * fb.height,), np.int64)
                rows.fill(-1)
                rows[pixels] = np.arange(len(pixels))
                base["extra"] = (rows, extra["tris"], extra["sample_rgb"])  # for mirrors to replace them too
        if self.reflections and self.mirror_bounces > 0:  # what mirrors show (the base samples drew every pixel)
            every = self._buffers.arange(fb.width * fb.height)
            self._reflect(scene, base, every, every, fb.rgb.reshape(-1, 3), 0, 1.0)
        if layers is not None:
            self._blend_layers(scene, layers, n)
        outline = float(self.outline) if math.isfinite(self.outline) else 0.0  # (NaN would darken edges into NaN)
        post_effects(fb.rgb, fb.alpha, fb.depth, *fog_args(self.fog), *self._view_tan, float(self._ortho), outline,
                     *self._haze)

    def _bins(self, xs, ys, select, want, height, prefix, span=None):
        """The triangles whose kind (select: 0 solid, 1 see-through, 2 cut-out) is among `want` (SOLID, CLEAR
        and CUT, or-ed together) that may touch each band of ROW_BAND rows of a picture `height` rows tall,
        as (band_start, band_tris) for rasterize() and friends, in buffers named from `prefix`. span: the
        columns (first, last) to be drawn in each band (see _spans); all of them if None."""
        buf = self._buffers
        n_bands = (height + ROW_BAND - 1) // ROW_BAND
        if span is None:
            span = self._full_span(n_bands)
        band_count = buf.get(prefix + "band_count", (n_bands,), np.int64)
        per_chunk = count_bands(xs, ys, select, want, height, span, band_count)
        band_start = np.zeros(n_bands + 1, np.int64)
        np.cumsum(band_count, out=band_start[1:])
        band_tris = buf.get(prefix + "band_tris", (int(band_start[-1]),), np.int64)
        bin_bands(xs, ys, select, want, height, span, per_chunk, band_start, band_tris)
        return band_start, band_tris

    def _full_span(self, n_bands):
        """Spans (see _bins) covering every column of every band, however wide the picture."""
        key = ("span", n_bands)
        span = self._buffers._arrays.get(key)
        if span is None:
            span = self._buffers._arrays[key] = np.tile([-np.inf, np.inf], (n_bands, 1))
        return span

    def _spans(self, pixels):
        """The columns (first, last) of each band of rows that `pixels` (flat indices) reach; empty
        (first > last) for bands they don't."""
        width, n_bands = self._fb.width, (self._fb.height + ROW_BAND - 1) // ROW_BAND
        band, x = pixels // width // ROW_BAND, (pixels % width).astype(float)
        span = np.empty((n_bands, 2))
        span[:, 0], span[:, 1] = np.inf, -np.inf
        np.minimum.at(span[:, 0], band, x)
        np.maximum.at(span[:, 1], band, x + 1.0)
        return span

    # ----- see-through surfaces ---------------------------------------------------------

    def _layers(self, scene, pattern, solid, prefix="", span=None):
        """The see-through surfaces in front of the solid ones in each pixel, nearest first (see
        raster.rasterize_layers); solid is the solid surfaces' depth at each of the `pattern` positions
        (infinite in pixels to leave alone). In buffers named from `prefix`; None if there are none."""
        fb, buf = self._fb, self._buffers
        m, k = fb.width * fb.height, max(int(self.transparency_layers), 1)
        layers = {"depth": buf.get(prefix + "layer_depth", (m, k)),
                  "tri": buf.get(prefix + "layer_tri", (m, k), np.int32),
                  "cover": buf.get(prefix + "layer_cover", (m, k), np.int32),
                  "count": buf.get(prefix + "layer_count", (m,), np.int32)}
        band_start, band_tris = self._bins(scene["xs"], scene["ys"], scene["see"], CLEAR, fb.height, prefix + "clear_",
                                           span)
        if not band_start[-1]:
            return None  # no see-through surface where it is drawn
        rasterize_layers(solid, fb.width, fb.height, scene["xs"], scene["ys"], scene["inv_w"], scene["tri_inst"],
                         np.array(pattern, dtype=float), band_start, band_tris, layers["depth"], layers["tri"],
                         layers["cover"], layers["count"])
        return layers

    def _blend_layers(self, scene, layers, n_samples, target=None):
        """Shade the see-through layers and blend them over the framebuffer, or `target` (whole-frame rgb,
        alpha, depth, ids, flat), farthest first (see _blend)."""
        fb, buf = self._fb, self._buffers
        pixels = np.flatnonzero(layers["count"])
        target = target or (fb.rgb.reshape(-1, 3), fb.alpha.reshape(-1), fb.depth.reshape(-1), fb.ids.reshape(-1))
        blend(pixels, *target,
              layers["count"], layers["depth"], layers["tri"], layers["cover"], n_samples, fb.width, scene["xs"],
              scene["ys"], scene["inv_w"], scene["attrs"], scene["tri_inst"], scene["ident"], scene["emissive"],
              scene["specular"], scene["shininess"], scene["shine"], scene["chain"], scene["lod"], *scene["textures"],
              scene["lights"], *scene["shadows"], scene["pixel_size"], scene["eye"], *scene["sky"],
              buf.get("layer_scratch", (len(pixels), 9)))

    # ----- geometry --------------------------------------------------------------------

    def _geometry(self, inst, camera, lights):
        """Every visible object's triangles in screen space, in one set of arrays in render-list order,
        with what shading them needs; None if there are none."""
        eye = np.asarray(camera.position, dtype=float)
        shading = {"shadows": self._shadow_maps(inst, lights), "pixel_size": self._pixel_size(camera),
                   "lights": np.ascontiguousarray(self._light_rows.reshape(-1, LIGHT_COLUMNS))}
        return self._view(inst, self.view_proj, eye, float(camera.near), NO_CLIP, self._pass_shine(inst, 0), shading,
                          "", self._axes)

    def _view(self, inst, view_proj, eye, near, clip, shine, shading, prefix, axes):
        """The instances' triangles seen through view_proj from `eye` (clipped against the plane `clip`), in
        buffers named from `prefix`, as a scene for _accumulate() and friends; None if there are none.
        shine: each instance's reflectivity of the background (see _shade); shading: its lights, shadow
        maps and pixel size; axes: the directions its pixels look in (see transforms.view_axes)."""
        fb, buf, pack = self._fb, self._buffers, inst["pack"]
        k, world, inst_vertex, (chunk_inst, chunk_first, chunk_end, whole, cut, off_whole, off_cut) = self._transform(
            pack, inst, inst["double"], view_proj, eye, near, prefix, clip)
        xs, ys = buf.get(prefix + "xs", (k, 3)), buf.get(prefix + "ys", (k, 3))
        inv_w = buf.get(prefix + "inv_w", (k, 3))
        attrs = buf.get(prefix + "attrs", (k, 3, ATTRS))
        tri_inst, chain = buf.get(prefix + "tri_inst", (k,), np.int32), buf.get(prefix + "chain", (k,), np.int64)
        lod, see = buf.get(prefix + "lod", (k,)), buf.get(prefix + "see", (k,), np.int8)
        project(pack["faces"], pack["uvs"], pack["colors"], pack["face_chain"], pack["face_kind"],
                pack["mesh_vertex"], inst["mesh"], inst["rgb"], inst["alpha"], inst["double"], inst["flip"],
                inst_vertex, eye, near, clip, fb.width, fb.height, float(self.lod_bias), world, chunk_inst,
                chunk_first, chunk_end, whole, cut, off_whole, off_cut, xs, ys, inv_w, attrs, tri_inst, chain, lod,
                see)
        if not k:
            return None
        return {"xs": xs[:k], "ys": ys[:k], "inv_w": inv_w[:k], "attrs": attrs[:k], "tri_inst": tri_inst[:k],
                "see": see[:k], "ident": inst["ident"], "emissive": inst["emissive"], "specular": inst["specular"],
                "shininess": inst["shininess"], "chain": chain[:k],
                "lod": lod[:k], "shine": shine if self.reflections else np.zeros_like(shine), "sky": self._sky,
                "textures": pack["textures"], "eye": eye, "inst": inst, "view_proj": view_proj, "near": near,
                "axes": axes, "shading": shading, **shading}

    def _pixel_size(self, camera):
        """The width in the world of a pixel one unit in front of the camera: the larger of its width and
        height, where pixels are not square."""
        fb = self._fb
        tall = 2.0 * np.tan(np.radians(camera.fov) / 2)
        wide = tall * self.width * self.cell_aspect / self.height
        return float(max(tall / fb.height, wide / fb.width))

    def _transform(self, pack, inst, double, view_proj, eye, near, prefix, clip=NO_CLIP):
        """raster.transform() of the instances `inst` (a dict of arrays like _instances() gives) seen through
        view_proj, in buffers whose names start with `prefix`: (triangle count, world, inst_vertex, the face
        chunks and their plan (chunk_inst, chunk_first, chunk_end, whole, cut, off_whole, off_cut))."""
        buf = self._buffers
        n_inst = len(inst["mesh"])
        vertex_counts = np.diff(pack["mesh_vertex"])[inst["mesh"]]
        face_counts = np.diff(pack["mesh_face"])[inst["mesh"]]
        inst_vertex = np.zeros(n_inst + 1, np.int64)
        np.cumsum(vertex_counts, out=inst_vertex[1:])
        # The work, in pieces that run in parallel: chunks of each instance's vertices, then of its faces.
        vchunk_inst, vchunk_first, vchunk_end = _chunks(vertex_counts, np.zeros(n_inst, np.int64), VERTEX_CHUNK)
        chunk_inst, chunk_first, chunk_end = _chunks(face_counts, pack["mesh_face"][inst["mesh"]], FACE_CHUNK)
        n_chunks = len(chunk_inst)
        whole, cut = buf.get(prefix + "whole", (n_chunks,), np.int64), buf.get(prefix + "cut", (n_chunks,), np.int64)
        off_whole = buf.get(prefix + "off_whole", (n_chunks,), np.int64)
        off_cut = buf.get(prefix + "off_cut", (n_chunks,), np.int64)
        world = buf.get(prefix + "world", (int(inst_vertex[-1]), 10))
        k = transform(pack["vertices"], pack["normals"], pack["faces"], pack["mesh_vertex"], pack["spheres"],
                      inst["mesh"], inst["lin"], inst["pos"], double, inst["flip"], inst_vertex,
                      view_proj, eye, near, clip, vchunk_inst, vchunk_first, vchunk_end,
                      chunk_inst, chunk_first, chunk_end, world, buf.get(prefix + "visible", (n_inst,), np.bool_),
                      whole, cut, off_whole, off_cut)
        return k, world, inst_vertex, (chunk_inst, chunk_first, chunk_end, whole, cut, off_whole, off_cut)

    # ----- sampling and shading ---------------------------------------------------------

    def _accumulate(self, scene, pattern, name, pixels=None, depth=None, ids=None, frame_samples=None):
        """Render every sample position in `pattern`, in all pixels or just the given flat pixel indices.

        Returns _resolve()'s per-pixel outputs (rgb, cover, more) and, at each sample, its depth,
        triangle and colour (sample_depth, tris, sample_rgb), in buffers named after `name`; the depth
        and object id of each pixel's nearest sample go into `depth` and `ids` if given. With frame_samples, each
        pixel's colour and coverage go into the framebuffer instead of rgb and cover (left empty): averaged with
        the frame_samples samples already there (0: none), as resolve() does.
        """
        fb, buf = self._fb, self._buffers
        n = len(pattern)
        whole = pixels is None
        if whole:
            pixels = slots = buf.arange(fb.width * fb.height)
        else:
            slots = buf.get("slots", (fb.width * fb.height,), np.int64)
            slots.fill(-1)
            slots[pixels] = buf.arange(fb.width * fb.height)[:len(pixels)]
        m = len(pixels)
        sample_depth = buf.get(name + "_sample_depth", (m, n))
        tris = buf.get(name + "_tris", (m, n), np.int32)
        if not whole:  # (a whole frame's are emptied by rasterize, band by band)
            sample_depth.fill(0.0)
            tris.fill(-1)
        rasterize(sample_depth, tris, fb.width, fb.height, scene["xs"], scene["ys"], scene["inv_w"],
                  np.array(pattern, dtype=float), slots, *scene["bands"], scene["see"], scene["attrs"], scene["chain"],
                  scene["lod"], *scene["textures"], whole)
        if frame_samples is None:
            sums = buf.get(name + "_rgb", (m, 3)), buf.get(name + "_cover", (m,), np.int64)
            frame = np.zeros((0, 3)), np.zeros(0)
        else:
            sums = np.zeros((0, 3)), np.zeros(0, np.int64)
            frame = fb.rgb.reshape(-1, 3), fb.alpha.reshape(-1)
        out = {"rgb": sums[0], "cover": sums[1],
               "more": buf.get(name + "_more", (m,), np.bool_), "sample_depth": sample_depth, "tris": tris,
               "sample_rgb": buf.get(name + "_sample_rgb", (m, n, 3))}
        depth = buf.get("near_depth", (m,)) if depth is None else depth
        ids = buf.get("near_id", (m,), np.int32) if ids is None else ids
        resolve(tris, sample_depth, pixels, fb.width, scene["xs"], scene["ys"], scene["inv_w"], scene["attrs"],
                scene["tri_inst"], scene["ident"], scene["emissive"], scene["specular"], scene["shininess"],
                scene["shine"], scene["chain"], scene["lod"], *scene["textures"], scene["lights"], *scene["shadows"],
                scene["pixel_size"], scene["eye"], *scene["sky"], EDGE_CONTRAST, out["sample_rgb"],
                buf.get(name + "_spec", (m, 3)), out["rgb"], out["cover"], depth, ids, out["more"], *frame,
                frame_samples or 0)
        return out
