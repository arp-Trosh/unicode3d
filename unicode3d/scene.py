# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
"""Camera, lights, scene objects and the render pipeline."""
from dataclasses import dataclass, field

import numpy as np
from numba import njit, prange

from .background import SKYBOX_FACES, background_args, fill_background, sky_colour
from .color import to_linear_rgb, to_srgb
from .mesh import Mesh
from .raster import (ATTRS, FACE_CHUNK, ROW_BAND, VERTEX_CHUNK, FrameBuffer, barycentric, bin_bands, count_bands,
                     count_cube, project, project_cube, project_depth, rasterize, rasterize_depth, rasterize_layers,
                     rasterize_tint, transform, upscale, bit_count, NO_CLIP, SURFACE)
from .texture import alpha_kind, pack as pack_textures, sample as sample_texture
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
    shadows: bool = False   # objects block this light from what lies behind them (see Renderer for the settings)


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
    shadows: bool = False   # objects block this light, in every direction (see Renderer for the settings)


LIGHT_COLUMNS = 16  # kind, direction or position (3), colour as levels (3), linear colour (3), ambient, diffuse,
                    # specular, shininess, range, first shadow map (-1 for none)


def _light_rows(lights, shadows=True):
    """The lights as a (L, LIGHT_COLUMNS) array for _shade; shadows=False leaves out every light's shadows."""
    rows = np.zeros((len(lights), LIGHT_COLUMNS))
    rows[:, 15] = -1
    maps = 0
    for i, light in enumerate(lights):
        lin = _linear_color(light.color)
        if isinstance(light, PointLight):
            rows[i, 0], rows[i, 1:4], rows[i, 14] = 1, np.asarray(light.position, float), float(light.range)
        else:
            rows[i, 1:4] = -normalize(light.direction)
        if light.shadows and shadows:  # its first shadow map: one for a Light, six (a cube) for a PointLight
            rows[i, 15], maps = maps, maps + (6 if isinstance(light, PointLight) else 1)
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
    cast_shadows: bool = True  # whether it blocks lights that have shadows (False for a lamp's own bulb, say)
    opacity: float = 1.0    # 1 is solid, less lets what is behind show through (0 is invisible); multiplies the
                            # alpha of the mesh's colours. See-through objects show their far side too.
    reflectivity: float = 0.0  # how much it reflects, 0..1 (1: a perfect mirror): a flat mesh reflects the
                               # scene, a curved one the background (see Renderer)


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


def _flat_plane(vertices, faces, radius):
    """The plane (unit normal, d: n . x + d = 0) all of a mesh's faces lie on, to within a millionth of its
    size; NaNs if they don't (it is not flat)."""
    vertices = vertices[np.unique(faces)] if len(faces) else vertices[:0]
    if len(vertices) < 3 or not np.isfinite(vertices).all():  # (a corner at NaN or infinity: no plane)
        return (np.nan,) * 4
    centre = vertices.mean(axis=0)
    _, spread, axes = np.linalg.svd(vertices - centre, full_matrices=False)
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
MAX_PIXELS = 1920 * 1080  # the default Renderer.max_pixels
MAX_BOUNCES = 4          # the deepest Renderer.mirror_bounces goes
MIRROR_MIN_PIXELS = 50   # a mirror seen in a mirror smaller than this shows no reflection
MIRROR_MIN_WEIGHT = 0.05  # nor one whose reflection would count for less of the frame's colour than this
SOLID, CLEAR, CUT = 1, 2, 4  # kinds of triangle (raster.project's `see`: 0, 1, 2) as bits, for Renderer._bins
_NO_UVS = np.zeros((1, 3, 2))  # stands in for the uvs of untextured meshes
EDGE_CONTRAST = 0.03  # linear-light spread among a pixel's first samples that marks it for more


@njit(cache=True, error_model="numpy")
def _decode(level):
    """A perceived light level (0..1, like an sRGB value) as linear light."""
    return level / 12.92 if level <= 0.04045 else ((level + 0.055) / 1.055) ** 2.4


SHADOW_COLUMNS = 10  # per shadow map: kind (0: a Light's, 1: a face of a PointLight's cube), offset of its texels,
                     # size, texel width (a Light's: in the world; a face's: per unit of distance from the light),
                     # depth bias (a Light's: in depth; a face's: in texels), softness, the PointLight's position
                     # (3), whether see-through things are in it (1: its texels in trans are filled in)
CUBE_MARGIN = 8  # texels around each face of a cube map beyond its 90 degrees, so that filtering stays inside it
_NO_SHADOWS = (np.zeros(0, np.float32), np.zeros((4, 0), np.float32), np.zeros((0, 4, 4)),
               np.zeros((0, SHADOW_COLUMNS)))  # when no light has any


@njit(cache=True, error_model="numpy")
def _lit_texel(texels, offset, size, x, y, d):
    """Whether texel (x, y) of a shadow map (size x size texels from texels[offset]) lets light through to
    depth d: nothing nearer the light is there (outside the map there is nothing to cast a shadow)."""
    return x < 0 or y < 0 or x >= size or y >= size or texels[offset + y * size + x] <= d


@njit(cache=True, error_model="numpy")
def _through(trans, offset, size, x, y, d):
    """The light (rgb) that see-through things let through to depth d at texel (x, y) of a shadow map: what
    trans holds there if the nearest of them is nearer the light than d, else all of it."""
    if x < 0 or y < 0 or x >= size or y >= size:
        return 1.0, 1.0, 1.0
    k = offset + y * size + x
    if trans[0, k] <= d:
        return 1.0, 1.0, 1.0
    return trans[1, k], trans[2, k], trans[3, k]


@njit(cache=True, error_model="numpy")
def _shadow(texels, trans, mats, params, first, px, py, pz, nx, ny, nz, ndl, footprint):
    """How much of a light reaches point p on a surface with normal n, 0 (in shadow) to 1 (lit), from its
    shadow map `first`, or for a PointLight the face of its cube (maps first to first + 5: +x, -x, +y, -y,
    +z, -z) that p lies in (see Renderer._shadow_maps). ndl is the cosine of the angle between n and the
    light, and footprint the width in the world of the screen pixel being shaded.

    The map is averaged over a square around the point (percentage-closer filtering): softness texels
    either way, or half the pixel's footprint if that is more, so that shadow edges are smoothed at
    least like the edges of shapes are. Small squares weight each texel by how much of it they cover;
    large ones take 4 x 4 samples, each blending its 4 nearest texels. The point is first moved off the
    surface along its normal, more the more the surface slopes away from the light and the larger the
    square, so that the surface does not shadow itself.

    Returns (how much light reaches the point, 0..1, and what see-through things in the way let through
    of it, rgb: all of it, 1, where there are none), both averaged over the same square.
    """
    m, cube = first, params[first, 0] != 0
    texel = params[m, 3]
    if cube:
        dx, dy, dz = px - params[m, 6], py - params[m, 7], pz - params[m, 8]
        ax, ay, az = abs(dx), abs(dy), abs(dz)
        if ax >= ay and ax >= az:
            m, dist = m + (0 if dx > 0 else 1), ax
        elif ay >= az:
            m, dist = m + (2 if dy > 0 else 3), ay
        else:
            m, dist = m + (4 if dz > 0 else 5), az
        texel *= max(dist, 1e-9)  # a face's texels widen with distance from the light
    offset, size, bias, soft = int(params[m, 1]), int(params[m, 2]), params[m, 4], params[m, 5]
    tinted = params[m, 9] != 0
    radius = max(soft, 0.5 * footprint / texel)
    if cube:
        radius = min(radius, CUBE_MARGIN - 1.0)
    shift = (radius + 1.0) * texel * np.sqrt(max(1.0 - ndl * ndl, 0.0))
    qx, qy, qz = px + nx * shift, py + ny * shift, pz + nz * shift
    mat = mats[m]
    cx = mat[0, 0] * qx + mat[0, 1] * qy + mat[0, 2] * qz + mat[0, 3]
    cy = mat[1, 0] * qx + mat[1, 1] * qy + mat[1, 2] * qz + mat[1, 3]
    if cube:  # depth is 1/w; the point counts as bias texels nearer the light than it is
        w = mat[3, 0] * qx + mat[3, 1] * qy + mat[3, 2] * qz + mat[3, 3]
        cx, cy, d = cx / w, cy / w, 1.0 / max(w - bias * texel, 1e-9)
    else:
        cz = mat[2, 0] * qx + mat[2, 1] * qy + mat[2, 2] * qz + mat[2, 3]
        d = (1.0 - cz) * 0.5 + bias
    # Texel (x, y) has its centre at u = x, v = y.
    u, v = (cx + 1.0) * 0.5 * size - 0.5, (1.0 - cy) * 0.5 * size - 0.5
    if not (abs(u) < 1e9 and abs(v) < 1e9):  # far outside the map (or NaN): nothing there casts a shadow
        return 1.0, 1.0, 1.0, 1.0
    lit = tr = tg = tb = 0.0
    if radius <= 3.0:  # at most 8 x 8 texels
        for y in range(int(np.floor(v - radius + 0.5)), int(np.floor(v + radius + 0.5)) + 1):
            wy = min(v + radius, y + 0.5) - max(v - radius, y - 0.5)
            if wy <= 0.0:
                continue
            for x in range(int(np.floor(u - radius + 0.5)), int(np.floor(u + radius + 0.5)) + 1):
                wx = min(u + radius, x + 0.5) - max(u - radius, x - 0.5)
                if wx <= 0.0:
                    continue
                if _lit_texel(texels, offset, size, x, y, d):
                    lit += wx * wy
                if tinted:
                    r, g, b = _through(trans, offset, size, x, y, d)
                    tr, tg, tb = tr + wx * wy * r, tg + wx * wy * g, tb + wx * wy * b
        total = 4.0 * radius * radius
    else:
        step = radius / 2.0
        for j in range(4):
            sv = v + (j - 1.5) * step
            y0 = int(np.floor(sv))
            fy = sv - y0
            for i in range(4):
                su = u + (i - 1.5) * step
                x0 = int(np.floor(su))
                fx = su - x0
                for y, x, w in ((y0, x0, (1.0 - fy) * (1.0 - fx)), (y0, x0 + 1, (1.0 - fy) * fx),
                                (y0 + 1, x0, fy * (1.0 - fx)), (y0 + 1, x0 + 1, fy * fx)):
                    if _lit_texel(texels, offset, size, x, y, d):
                        lit += w
                    if tinted:
                        r, g, b = _through(trans, offset, size, x, y, d)
                        tr, tg, tb = tr + w * r, tg + w * g, tb + w * b
        total = 16.0
    if not tinted:
        return lit / total, 1.0, 1.0, 1.0
    return lit / total, min(tr / total, 1.0), min(tg / total, 1.0), min(tb / total, 1.0)


@njit(cache=True, error_model="numpy")
def _shade(t, b0, b1, b2, inv_w, attrs, chain, lod, texels, levels, first, lights, shadow_texels, shadow_trans,
           shadow_mats, shadow_params, pixel_size, eye, emissive, out, split, spec_out):
    """Linear RGB, into out (3,), of triangle t at barycentric weights b: Blinn-Phong lighting from
    every light (rows of _light_rows()), plus `emissive`, on a surface of the colour interpolated
    from its corners, textured if the triangle has a mipmap chain. Highlights take the light's colour.
    Lights with a shadow map light only what they reach (see _shadow), tinted by see-through things in
    the way; their ambient light is everywhere. pixel_size is the width of a pixel one unit from the
    eye, in the world. With split, out gets the surface's own lit colour and spec_out (3,) the
    highlight, apart (for see-through and shiny surfaces, see _reflect_sky).

    Returns the surface's alpha there (its texture's included), how squarely it faces the eye (the cosine
    of the angle), and the direction a view from the eye is reflected in (rx, ry, rz)."""
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
        ndl = nx * lx + ny * ly + nz * lz
        reach = tr = tg = tb = 1.0
        if light[15] >= 0:  # a surface facing away is in its own shadow
            if ndl > 0.0:
                reach, tr, tg, tb = _shadow(shadow_texels, shadow_trans, shadow_mats, shadow_params, int(light[15]),
                                            px, py, pz, nx, ny, nz, ndl, pixel_size * el)
            else:
                reach = 0.0
        if tr == 1.0 and tg == 1.0 and tb == 1.0:
            level = (light[10] + light[11] * max(ndl, 0.0) * reach) * fade
            level_r, level_g, level_b = (level_r + light[4] * level, level_g + light[5] * level,
                                         level_b + light[6] * level)
        else:  # light through coloured glass: tinted, channel by channel
            lit = light[11] * max(ndl, 0.0) * reach
            level_r += light[4] * (light[10] + lit * tr) * fade
            level_g += light[5] * (light[10] + lit * tg) * fade
            level_b += light[6] * (light[10] + lit * tb) * fade
        if light[12] > 0.0 and reach > 0.0:
            hx, hy, hz = lx + ex, ly + ey, lz + ez
            hl = max(np.sqrt(hx * hx + hy * hy + hz * hz), 1e-12)
            spec = light[12] * max((nx * hx + ny * hy + nz * hz) / hl, 0.0) ** light[13] * fade * reach
            spec_r, spec_g, spec_b = (spec_r + light[7] * spec * tr, spec_g + light[8] * spec * tg,
                                      spec_b + light[9] * spec * tb)
    r, g, b, a = lerp(8), lerp(9), lerp(10), lerp(11)
    if chain[t] >= 0:
        tr, tg, tb, ta = sample_texture(texels, levels, first, chain[t], lerp(6), lerp(7), lod[t])
        r, g, b, a = r * tr, g * tg, b * tb, a * ta
    # Light levels are perceived brightness (0.5 looks half as bright), as artists tune them, so they
    # are decoded like any sRGB value; everything after this point works in linear light.
    kr = _decode(min(max(level_r, 0.0), 1.0))
    kg = kr if level_g == level_r else _decode(min(max(level_g, 0.0), 1.0))
    kb = kr if level_b == level_r else _decode(min(max(level_b, 0.0), 1.0))
    if split:
        out[0], out[1], out[2] = min(max(r * kr, 0.0), 1.0), min(max(g * kg, 0.0), 1.0), min(max(b * kb, 0.0), 1.0)
        spec_out[0], spec_out[1], spec_out[2] = spec_r, spec_g, spec_b
    else:
        out[0] = min(max(r * kr + spec_r, 0.0), 1.0)
        out[1] = min(max(g * kg + spec_g, 0.0), 1.0)
        out[2] = min(max(b * kb + spec_b, 0.0), 1.0)
    ne = nx * ex + ny * ey + nz * ez  # the view reflected about the surface: 2 (n . e) n - e
    return a, abs(ne), 2 * ne * nx - ex, 2 * ne * ny - ey, 2 * ne * nz - ez


@njit(cache=True, error_model="numpy", parallel=True)
def _resolve(tris, depth, pixels, width, xs, ys, inv_w, attrs, tri_inst, ident, emissive, shine, chain, lod, texels,
             levels, first, lights, shadow_texels, shadow_trans, shadow_mats, shadow_params, pixel_size, eye, sky,
             sky_colors, sky_faces, sky_texels, sky_levels, sky_first, sky_lod, contrast, sample_rgb, spec_rgb, rgb,
             cover, near_depth, near_id, more):
    """Shade the rasterized samples and sum them up per pixel.

    Each pixel is shaded once per triangle covering it, at the pixel centre, like
    hardware multisampling: the samples only decide coverage. (Texture detail is
    smoothed by mipmapping instead.)

    Writes, per pixel: rgb (summed colour of the covered samples), cover (how many
    samples were covered), the depth and object id of the nearest sample, and
    more: whether the samples disagree (some covered and some not, different
    objects, or colours further apart than `contrast`), so the pixel is worth more
    samples. sample_rgb (M, S, 3) and spec_rgb (M, 3) are scratch space.

    Shiny surfaces (shine, per instance) mix that much of the background seen reflected in
    them (sky and its colours etc.: background.sky_colour's arguments) into their colour.
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
                gloss = shine[tri_inst[t]]
                _, _, rx, ry, rz = _shade(t, b0, b1, b2, inv_w, attrs, chain, lod, texels, levels, first, lights,
                                          shadow_texels, shadow_trans, shadow_mats, shadow_params, pixel_size, eye,
                                          emissive[tri_inst[t]], sample_rgb[c, s], gloss > 0.0, spec_rgb[c])
                if gloss > 0.0:  # polished: part the background reflected in it, and its highlight on top
                    sr, sg, sb = sky_colour(sky, sky_colors, sky_faces, sky_texels, sky_levels, sky_first, sky_lod,
                                            rx, ry, rz)
                    for k, reflected in ((0, sr), (1, sg), (2, sb)):
                        sample_rgb[c, s, k] = min(sample_rgb[c, s, k] * (1.0 - gloss) + gloss * reflected
                                                  + spec_rgb[c, k], 1.0)
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


@njit(cache=True, error_model="numpy")
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


@njit(cache=True, error_model="numpy", parallel=True)
def _blend(pixels, rgb, alpha, depth, ids, layer_count, layer_depth, layer_tri, layer_cover, n_samples, width, xs, ys,
           inv_w, attrs, tri_inst, ident, emissive, shine, chain, lod, texels, levels, first, lights, shadow_texels,
           shadow_trans, shadow_mats, shadow_params, pixel_size, eye, sky, sky_colors, sky_faces, sky_texels,
           sky_levels, sky_first, sky_lod, scratch):
    """Blend the see-through layers (raster.rasterize_layers) of each of `pixels` (flat indices of those that
    have any: shared out evenly among threads, where the pixels themselves may be bunched together) over
    its colour and coverage (flat framebuffer arrays, premultiplied by coverage), farthest first, and give
    the pixel the depth and object id of the nearest.

    Each layer is shaded at the pixel centre and covers the fraction of the pixel's n_samples it
    covered (layer_cover: a bit per sample). Its opacity is its interpolated alpha, raised towards 1
    where the surface is seen edge-on (Fresnel's law, as glass reflects more at a slant), the extra
    showing the background reflected in it; its highlights show at full strength however clear it
    is, as on glass; both fade out below an alpha of 0.25, so that something faded to nothing
    disappears. sky < 0: reflections are off, and the extra shows the surface's own colour. The
    nearest layer that shows at all gives the pixel its depth and id. A layer that is shiny as well
    (shine, per instance) takes that much of the reflected background into its own colour. scratch
    (len(pixels), 9) is room for the shading, a row for each.
    """
    for p in prange(pixels.shape[0]):
        c = pixels[p]
        n = layer_count[c]
        cx, cy = c % width + 0.5, c // width + 0.5
        r, g, b, a = rgb[c, 0], rgb[c, 1], rgb[c, 2], alpha[c]
        shows = -1  # the nearest layer that shows so far
        for layer in range(n - 1, -1, -1):
            t = layer_tri[c, layer]
            b0, b1, b2 = barycentric(xs, ys, t, cx, cy)
            clear, facing, rx, ry, rz = _shade(t, b0, b1, b2, inv_w, attrs, chain, lod, texels, levels, first,
                                               lights, shadow_texels, shadow_trans, shadow_mats, shadow_params,
                                               pixel_size, eye, emissive[tri_inst[t]], scratch[p, 0:3], True,
                                               scratch[p, 3:6])
            scratch[p, 6], scratch[p, 7], scratch[p, 8] = sky_colour(sky, sky_colors, sky_faces, sky_texels,
                                                                     sky_levels, sky_first, sky_lod, rx, ry, rz)
            polish = shine[tri_inst[t]]  # a shiny see-through surface: part of its colour the background reflected
            if polish > 0.0:
                for k in range(3):
                    scratch[p, k] = scratch[p, k] * (1.0 - polish) + polish * scratch[p, 6 + k]
            clear = min(max(clear, 0.0), 1.0)
            gloss = min(4.0 * clear, 1.0)
            fresnel = 0.04 + 0.96 * (1.0 - min(facing, 1.0)) ** 5
            # What it gains in opacity at a slant is the background reflected in it (unless reflections are off).
            mirrored = (1.0 - clear) * fresnel * gloss
            opacity = clear + mirrored
            if sky < 0:
                mirrored = 0.0
            part = bit_count(layer_cover[c, layer]) / n_samples
            over = opacity * part
            r = r * (1.0 - over) + part * ((opacity - mirrored) * scratch[p, 0] + mirrored * scratch[p, 6]
                                            + gloss * scratch[p, 3])
            g = g * (1.0 - over) + part * ((opacity - mirrored) * scratch[p, 1] + mirrored * scratch[p, 7]
                                            + gloss * scratch[p, 4])
            b = b * (1.0 - over) + part * ((opacity - mirrored) * scratch[p, 2] + mirrored * scratch[p, 8]
                                            + gloss * scratch[p, 5])
            a = a * (1.0 - over) + over
            if over > 0.0:
                shows = layer
        rgb[c, 0], rgb[c, 1], rgb[c, 2] = min(r, a), min(g, a), min(b, a)
        alpha[c] = a
        if shows >= 0:  # what the pixel shows first (a layer faded to nothing in front of it shows nothing)
            depth[c], ids[c] = layer_depth[c, shows], ident[tri_inst[layer_tri[c, shows]]]


@njit(cache=True, error_model="numpy", parallel=True)
def _mirror_cover(tris, tri_inst, ident, mirror, cover):
    """How many of each pixel's samples (rows of tris: rasterize()'s triangle at each) landed on the object
    whose id is `mirror`, into cover."""
    for c in prange(tris.shape[0]):
        n = 0
        for s in range(tris.shape[1]):
            t = tris[c, s]
            n += t >= 0 and ident[tri_inst[t]] == mirror
        cover[c] = n


@njit(cache=True, error_model="numpy", parallel=True)
def _gather_pass(pixels, rgb, cover, depth, ids, n_samples, out_rgb, out_alpha, out_depth, out_ids):
    """A pass drawn in only some pixels (rows of rgb, cover, depth, ids, one for each of `pixels`, flat indices)
    into whole-frame arrays: colour premultiplied by coverage, coverage, depth and id."""
    for j in prange(pixels.shape[0]):
        c = pixels[j]
        for k in range(3):
            out_rgb[c, k] = rgb[j, k] / n_samples
        out_alpha[c], out_depth[c], out_ids[c] = cover[j] / n_samples, depth[j], ids[j]


@njit(cache=True, error_model="numpy", parallel=True)
def _fill_sky(pixels, rgb, alpha, width, height, inv_view_proj, sky, sky_colors, sky_faces, sky_texels, sky_levels,
              sky_first, sky_lod):
    """Fill what the scene leaves uncovered in each of `pixels` (flat indices of whole-frame rgb, alpha) with
    the background seen along the ray through the pixel of the view whose inverse is inv_view_proj (a
    reflected one: what a mirror shows beyond everything), and make the pixel opaque."""
    for j in prange(pixels.shape[0]):
        c = pixels[j]
        a = alpha[c]
        if a >= 1.0:
            continue
        nx, ny = (c % width + 0.5) / width * 2.0 - 1.0, 1.0 - (c // width + 0.5) / height * 2.0
        m = inv_view_proj
        near_w = m[3, 0] * nx + m[3, 1] * ny - m[3, 2] + m[3, 3]
        far_w = m[3, 0] * nx + m[3, 1] * ny + m[3, 2] + m[3, 3]
        dx = (m[0, 0] * nx + m[0, 1] * ny + m[0, 2] + m[0, 3]) / far_w - (m[0, 0] * nx + m[0, 1] * ny - m[0, 2]
                                                                           + m[0, 3]) / near_w
        dy = (m[1, 0] * nx + m[1, 1] * ny + m[1, 2] + m[1, 3]) / far_w - (m[1, 0] * nx + m[1, 1] * ny - m[1, 2]
                                                                           + m[1, 3]) / near_w
        dz = (m[2, 0] * nx + m[2, 1] * ny + m[2, 2] + m[2, 3]) / far_w - (m[2, 0] * nx + m[2, 1] * ny - m[2, 2]
                                                                           + m[2, 3]) / near_w
        dl = max(np.sqrt(dx * dx + dy * dy + dz * dz), 1e-30)
        r, g, b = sky_colour(sky, sky_colors, sky_faces, sky_texels, sky_levels, sky_first, sky_lod, dx / dl, dy / dl,
                             dz / dl)
        rgb[c, 0] += (1.0 - a) * r
        rgb[c, 1] += (1.0 - a) * g
        rgb[c, 2] += (1.0 - a) * b
        alpha[c] = 1.0


@njit(cache=True, error_model="numpy", parallel=True)
def _mix_mirror(target, pixels, rows, tris, sample_rgb, extra_rows, extra_tris, extra_rgb, tri_inst, ident, mirror,
                reflectivity, seen):
    """Show what a mirror reflects: in each of `pixels` (flat indices of whole-frame colours `target`,
    premultiplied by coverage), the share of the pixel's samples that fell on the mirror (id `mirror`)
    changes from the mirror's own colour towards `seen` (whole-frame, opaque) by `reflectivity`.

    The samples are row rows[j] of tris and sample_rgb (the pass the pixel was drawn in), and, where the
    pixel took extra samples at an edge (extra_rows[c] >= 0, indexed by flat pixel), that row of
    extra_tris and extra_rgb too, as in _combine."""
    n_base, n_extra = tris.shape[1], extra_tris.shape[1]
    for j in prange(pixels.shape[0]):
        c, row, extra = pixels[j], rows[j], extra_rows[pixels[j]]
        own_r = own_g = own_b = 0.0
        n = 0
        for s in range(n_base):
            t = tris[row, s]
            if t >= 0 and ident[tri_inst[t]] == mirror:
                n += 1
                own_r, own_g, own_b = own_r + sample_rgb[row, s, 0], own_g + sample_rgb[row, s, 1], own_b + sample_rgb[
                    row, s, 2]
        total = n_base
        if extra >= 0:
            total += n_extra
            for s in range(n_extra):
                t = extra_tris[extra, s]
                if t >= 0 and ident[tri_inst[t]] == mirror:
                    n += 1
                    own_r, own_g, own_b = (own_r + extra_rgb[extra, s, 0], own_g + extra_rgb[extra, s, 1],
                                           own_b + extra_rgb[extra, s, 2])
        part = n / total
        target[c, 0] += reflectivity * (part * seen[c, 0] - own_r / total)
        target[c, 1] += reflectivity * (part * seen[c, 1] - own_g / total)
        target[c, 2] += reflectivity * (part * seen[c, 2] - own_b / total)


@njit(cache=True, error_model="numpy", parallel=True)
def _post_effects(rgb, alpha, depth, fog, outline):
    """Fog, then outlines, applied in place to a framebuffer's arrays (see Renderer for both)."""
    h, w = depth.shape
    if fog:
        # Dim pixels in proportion to how far back they sit within the scene's depth range (of the pixels
        # drawn that have a depth).
        near, far = np.inf, -np.inf
        for y in range(h):
            for x in range(w):
                if alpha[y, x] > 0 and depth[y, x] > 0:
                    near, far = min(near, 1.0 / depth[y, x]), max(far, 1.0 / depth[y, x])
        span = max(far - near, 0.25 * near)
        for y in prange(h):
            for x in range(w):
                if alpha[y, x] > 0 and depth[y, x] > 0:
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


def _world_spheres(spheres, quat, scale, pos):
    """Bounding spheres (centres (N, 3), radii (N,)) in the world of instances whose meshes have bounding
    spheres `spheres` (N, 4: centre, radius), with rotations `quat` (N, 4), scales and positions."""
    w, q, v = quat[:, :1], quat[:, 1:], spheres[:, :3] * scale[:, None]
    t = 2.0 * np.cross(q, v)
    return v + w * t + np.cross(q, t) + pos, spheres[:, 3] * np.abs(scale)


def _light_view(direction, centres, radii, size):
    """The view of a Light travelling in `direction` for its shadow map of size x size texels: an
    orthographic projection of a box around the spheres (centres, radii), looking along the light.
    Returns (world to normalized device coordinates (4, 4), a texel's width in the world, and the change in
    the map's depth per unit of distance along the light).

    The box moves in steps of whole texels and grows in steps of about 9%, so that shadows keep still as
    the objects in it move, rather than crawling as the texels shift under them.
    """
    d = normalize(direction)
    up = np.array([0.0, 1.0, 0.0]) if abs(d[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    right = normalize(np.cross(d, up))
    basis = np.stack([right, np.cross(right, d), -d])  # x, y across the light; z towards where it comes from
    p = centres @ basis.T
    lo, hi = (p - radii[:, None]).min(axis=0), (p + radii[:, None]).max(axis=0)
    half = 2.0 ** (np.ceil(8.0 * np.log2(max((hi[0] - lo[0]) / 2, (hi[1] - lo[1]) / 2, 1e-9))) / 8)
    texel = 2.0 * half / size
    cx, cy = (np.round((lo[:2] + hi[:2]) / 2 / texel) * texel).tolist()
    # Depth runs from 0 (empty) to 1 (nearest the light), with room at the far end so nothing sits at 0.
    margin = 0.01 * (hi[2] - lo[2]) + texel
    z_lo, span = lo[2] - margin, hi[2] - lo[2] + 2 * margin
    mat = np.zeros((4, 4))
    mat[0, :3], mat[0, 3] = basis[0] / half, -cx / half
    mat[1, :3], mat[1, 3] = basis[1] / half, -cy / half
    mat[2, :3], mat[2, 3] = -2.0 * basis[2] / span, 1.0 + 2.0 * z_lo / span
    mat[3, 3] = 1.0
    return mat, texel, 1.0 / span


# The faces of a cube map: the way each looks from the light, and its up.
CUBE_FACES = (((1.0, 0.0, 0.0), (0.0, -1.0, 0.0)), ((-1.0, 0.0, 0.0), (0.0, -1.0, 0.0)),
              ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), ((0.0, -1.0, 0.0), (0.0, 0.0, -1.0)),
              ((0.0, 0.0, 1.0), (0.0, -1.0, 0.0)), ((0.0, 0.0, -1.0), (0.0, -1.0, 0.0)))


_CUBE_BASES = {}  # (reach, size): _cube_views() for a light at the origin


def _cube_views(position, reach, size):
    """The views of a PointLight at `position` for the six faces of its cube shadow map, size x size
    texels each: perspectives out to `reach`, a little wider than 90 degrees (CUBE_MARGIN texels either
    side), so that what a face shows overlaps its neighbours. Returns (world to clip space of each face
    (6, 4, 4), texel width per unit of distance from the light, near plane).
    """
    tan_half = 1.0 / (1.0 - 2.0 * CUBE_MARGIN / size)
    near = max(0.002 * reach, 1e-4)
    base = _CUBE_BASES.get((reach, size))
    if base is None:
        if len(_CUBE_BASES) > 64:
            _CUBE_BASES.clear()
        proj = perspective(2.0 * np.arctan(tan_half), 1.0, near, reach)
        base = _CUBE_BASES[reach, size] = np.array([proj @ look_at(np.zeros(3), np.array(ahead), np.array(up))
                                                    for ahead, up in CUBE_FACES])
    mats = base.copy()
    mats[:, :, 3] -= base[:, :, :3] @ position  # the same views, moved to the light
    return mats, 2.0 * tan_half / size, near


_EVERYWHERE = np.diag([0.0, 0.0, 0.0, 1.0])  # a view for transform() that culls nothing (w = 1, no sides)


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
        a = self._arrays.get(("arange", n))
        if a is None:
            if len(self._arrays) > 64:  # after many resizes
                for key in [key for key in self._arrays if isinstance(key, tuple)]:
                    del self._arrays[key]
            a = self._arrays[("arange", n)] = np.arange(n)
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
        self._eye = self._forward = None  # the camera's position and view direction then
        self._buffers = _Buffers()
        self._shadows = None  # (what the shadow maps depend on, the maps as _shadow_maps() returns them)
        self.resize(width, height)

    def resize(self, width, height, cell_pixels=None):
        """Set the size in cells, and optionally the pixels per cell (e.g. Screen.cell_pixels)."""
        cell_pixels = self.cell_pixels if cell_pixels is None else tuple(cell_pixels)
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
        self._shadows = None

    def _instances(self, objects):
        """The objects to draw, as arrays: their meshes' place in the pack, world poses, colours, flags and
        ids (render-list index + 1). Packs the meshes afresh if the set of meshes has changed."""
        meshes, mesh_of, keys = [], {}, []
        mesh_idx, pos, quat, scale, rgb, double, ident, emissive, cast, alpha, shine = ([], [], [], [], [], [], [], [],
                                                                                        [], [], [])
        for i, obj in enumerate(objects):
            mesh = obj.mesh
            if mesh is None or not len(mesh.faces):
                continue
            if obj.opacity <= 0.0:
                continue  # wholly clear: nothing to draw, nor any shadow
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
            cast.append(obj.cast_shadows)
            alpha.append(obj.opacity)
            shine.append(obj.reflectivity)
        if not meshes:
            return None
        where = np.array(pos, np.float64).reshape(-1, 3)
        turn = np.array(quat, dtype=np.float64).reshape(-1, 4)
        size = np.array(scale, np.float64)
        placed = np.isfinite(where).all(axis=1) & np.isfinite(turn).all(axis=1) & np.isfinite(size)
        if not placed.all():  # NaN or infinite in a pose (a physics blow-up, say): nowhere to draw it
            keep = np.flatnonzero(placed)
            if not len(keep):
                return None
            mesh_idx, rgb, double, ident, emissive, cast, alpha, shine = (
                [values[i] for i in keep] for values in (mesh_idx, rgb, double, ident, emissive, cast, alpha, shine))
            where, turn, size = where[keep], turn[keep], size[keep]
            used = sorted(set(mesh_idx))
            renumber = {m: j for j, m in enumerate(used)}
            meshes, keys = [meshes[m] for m in used], [keys[m] for m in used]
            mesh_idx = [renumber[m] for m in mesh_idx]
        packed = self._pack
        if packed is None or len(packed[0]) != len(keys) or any(
                len(a) != len(b) or any(x is not y for x, y in zip(a, b)) for a, b in zip(packed[0], keys)):
            packed = self._pack = (keys, _pack_meshes(meshes))
        quat = turn / np.maximum(np.linalg.norm(turn, axis=1, keepdims=True), 1e-300)
        mesh_idx = np.array(mesh_idx, np.int64)
        alpha = np.clip(np.array(alpha, np.float64), 0.0, 1.0)
        # See-through objects, and those with holes, always show their far side, through their near one.
        double = np.array(double, np.bool_) | (alpha < 1.0) | packed[1]["clear"][mesh_idx]
        return {"mesh": mesh_idx, "pos": where, "quat": quat, "scale": size,
                "rgb": np.array(rgb, np.float64).reshape(-1, 3), "double": double, "alpha": alpha,
                "shine": np.clip(np.array(shine, np.float64), 0.0, 1.0),
                "ident": np.array(ident, np.int32),
                "emissive": np.array(emissive, np.float64), "cast": np.array(cast, np.bool_), "pack": packed[1],
                "keys": keys}

    def _scene_state(self, inst, camera, lights):
        """Everything a render depends on: (values compared by equality, objects compared by identity).

        Meshes, their arrays and textures count as changed when replaced, as in Mesh's own caches;
        edits made inside them are not seen (call invalidate() after those).
        """
        values = [self.width, self.height, self.cell_pixels, self.max_pixels, self.cell_aspect, self.samples,
                  self.edge_samples,
                  self.transparency_layers, self.reflections, self.mirror_bounces,
                  self.fog, self.outline, self.lod_bias, self.shadow_size, self.point_shadow_size,
                  self.shadow_softness,
                  np.asarray(camera.position, float).tobytes(), np.asarray(camera.target, float).tobytes(),
                  np.asarray(camera.up, float).tobytes(), camera.fov, camera.near, camera.far,
                  _light_rows(lights, self.shadows).tobytes()]
        if inst is None:
            return values, []
        values += [inst[k].tobytes()
                   for k in ("mesh", "pos", "quat", "scale", "rgb", "double", "ident", "emissive", "cast", "alpha",
                             "shine")]
        return values, [x for key in inst["keys"] for x in (*key, None)]

    def render(self, objects, camera, lights):
        """Draw the objects; returns the framebuffer (reused by the next render: copy it to keep it).

        lights: a Light or PointLight, or a list of them (their light adds up).
        Objects with a parent are placed through it (see Node). When nothing has
        changed since the last render (see _scene_state), the framebuffer is
        returned as it is, so a still scene costs almost nothing. Floating-point trouble (NaN or infinite
        coordinates) is drawn around rather than warned about: a warning would be printed over the picture.
        """
        with np.errstate(all="ignore"):
            return self._render(objects, camera, lights)

    def _render(self, objects, camera, lights):
        lights = _as_lights(lights)
        self._fit()
        self._objects = list(objects)
        inst = self._instances(objects)
        aspect = self.width * self.cell_aspect / max(self.height, 1)
        bg_args, bg_state = background_args(self.background, camera, aspect, self._fb.height)
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
        fb = self._fb
        fb.clear()
        if self.width < 1 or self.height < 1:
            self.framebuffer.clear()
            return self.framebuffer
        aspect = self.width * self.cell_aspect / self.height
        self.view_proj = perspective(np.radians(camera.fov), aspect, camera.near, camera.far) @ camera.view_matrix()
        self._eye = np.asarray(camera.position, dtype=float).copy()
        self._forward = normalize(np.asarray(camera.target, float) - self._eye)
        kind, colors, _, texels, levels, first, lod = bg_args
        # What shiny surfaces reflect: the background (kind -1 when reflections are off).
        self._sky = (int(kind) if self.reflections else -1, colors, SKYBOX_FACES, texels, levels, first, float(lod))
        scene = self._geometry(inst, camera, lights) if inst is not None else None
        if scene is not None:
            self._shade_scene(scene)
        kind, colors, basis, texels, levels, first, lod = bg_args
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
        base = self._accumulate(scene, SAMPLE_PATTERNS[n], "base", depth=fb.depth.reshape(-1), ids=fb.ids.reshape(-1))
        layers = self._layers(scene, SAMPLE_PATTERNS[n], base["sample_depth"]) if (scene["see"] == 1).any() else None
        pixels, extra_rgb, extra_cover = np.zeros(0, np.int64), np.zeros((0, 3)), np.zeros(0, np.int64)
        if self.edge_samples and n > 1:
            pixels = np.flatnonzero(base["more"])
            if len(pixels):
                # A different pattern from the base one: the same one turned a quarter.
                pattern = tuple((1.0 - y, x) for x, y in SAMPLE_PATTERNS[self.edge_samples])
                extra = self._accumulate(scene, pattern, "edge", pixels)
                extra_rgb, extra_cover = extra["rgb"], extra["cover"]
                rows = self._buffers.get("edge_rows", (fb.width * fb.height,), np.int64)
                rows.fill(-1)
                rows[pixels] = np.arange(len(pixels))
                base["extra"] = (rows, extra["tris"], extra["sample_rgb"])  # for mirrors to replace them too
        _combine(fb.rgb.reshape(-1, 3), fb.alpha.reshape(-1), base["rgb"], base["cover"], n,
                 pixels, extra_rgb, extra_cover, self.edge_samples)
        if self.reflections and self.mirror_bounces > 0:  # what mirrors show (the base samples drew every pixel)
            every = self._buffers.arange(fb.width * fb.height)
            self._reflect(scene, base, every, every, fb.rgb.reshape(-1, 3), 0, 1.0)
        if layers is not None:
            self._blend_layers(scene, layers, n)
        _post_effects(fb.rgb, fb.alpha, fb.depth, self.fog, self.outline)

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
        _blend(pixels, *target,
               layers["count"], layers["depth"], layers["tri"], layers["cover"], n_samples, fb.width, scene["xs"],
               scene["ys"], scene["inv_w"], scene["attrs"], scene["tri_inst"], scene["ident"], scene["emissive"],
               scene["shine"], scene["chain"], scene["lod"], *scene["textures"], scene["lights"], *scene["shadows"],
               scene["pixel_size"], scene["eye"], *scene["sky"], buf.get("layer_scratch", (len(pixels), 9)))

    # ----- geometry --------------------------------------------------------------------

    def _geometry(self, inst, camera, lights):
        """Every visible object's triangles in screen space, in one set of arrays in render-list order,
        with what shading them needs; None if there are none."""
        eye = np.asarray(camera.position, dtype=float)
        shading = {"shadows": self._shadow_maps(inst, lights), "pixel_size": self._pixel_size(camera),
                   "lights": np.ascontiguousarray(_light_rows(lights, self.shadows).reshape(-1, LIGHT_COLUMNS))}
        return self._view(inst, self.view_proj, eye, float(camera.near), NO_CLIP, self._pass_shine(inst, 0), shading,
                          "")

    def _view(self, inst, view_proj, eye, near, clip, shine, shading, prefix):
        """The instances' triangles seen through view_proj from `eye` (clipped against the plane `clip`), in
        buffers named from `prefix`, as a scene for _accumulate() and friends; None if there are none.
        shine: each instance's reflectivity of the background (see _shade); shading: its lights, shadow
        maps and pixel size."""
        fb, buf, pack = self._fb, self._buffers, inst["pack"]
        k, world, inst_vertex, (chunk_inst, chunk_first, chunk_end, whole, cut, off_whole, off_cut) = self._transform(
            pack, inst, inst["double"], view_proj, eye, near, prefix, clip)
        xs, ys = buf.get(prefix + "xs", (k, 3)), buf.get(prefix + "ys", (k, 3))
        inv_w = buf.get(prefix + "inv_w", (k, 3))
        attrs = buf.get(prefix + "attrs", (k, 3, ATTRS))
        tri_inst, chain = buf.get(prefix + "tri_inst", (k,), np.int32), buf.get(prefix + "chain", (k,), np.int64)
        lod, see = buf.get(prefix + "lod", (k,)), buf.get(prefix + "see", (k,), np.int8)
        project(pack["faces"], pack["uvs"], pack["colors"], pack["face_chain"], pack["face_texels"], pack["face_kind"],
                pack["mesh_vertex"], inst["mesh"], inst["rgb"], inst["alpha"], inst["double"], inst_vertex, eye,
                near, clip, fb.width, fb.height, float(self.lod_bias), world, chunk_inst, chunk_first, chunk_end,
                whole, cut, off_whole, off_cut, xs, ys, inv_w, attrs, tri_inst, chain, lod, see)
        if not k:
            return None
        return {"xs": xs[:k], "ys": ys[:k], "inv_w": inv_w[:k], "attrs": attrs[:k], "tri_inst": tri_inst[:k],
                "see": see[:k], "ident": inst["ident"], "emissive": inst["emissive"], "chain": chain[:k],
                "lod": lod[:k], "shine": shine if self.reflections else np.zeros_like(shine), "sky": self._sky,
                "textures": pack["textures"], "eye": eye, "inst": inst, "view_proj": view_proj, "near": near,
                "shading": shading, **shading}

    # ----- mirrors -----------------------------------------------------------------------

    def _mirrors(self, inst):
        """Which instances are mirrors: flat, solid (not see-through, no holes) and reflective (Object3D
        .reflectivity), as a boolean array; each shows the scene reflected in its plane (see _reflect)."""
        pack = inst["pack"]
        return ((inst["shine"] > 0.0) & (inst["alpha"] >= 1.0) & ~pack["clear"][inst["mesh"]]
                & np.isfinite(pack["planes"][inst["mesh"], 0]))

    def _pass_shine(self, inst, level):
        """Each instance's reflectivity of the background (see _shade) in a pass `level` mirrors deep (0: the
        frame itself): mirrors reflect the scene instead (their own pass) until mirror_bounces runs out,
        and then the background."""
        shine = inst["shine"].copy()
        if self.reflections and level < min(int(self.mirror_bounces), MAX_BOUNCES):
            shine[self._mirrors(inst)] = 0.0
        return shine

    def _plane(self, inst, i):
        """Instance i's mirror plane in the world: (unit normal (3,), d), n . x + d = 0 on it."""
        normal, d = inst["pack"]["planes"][inst["mesh"][i], :3], inst["pack"]["planes"][inst["mesh"][i], 3]
        rot = quat_to_matrix(inst["quat"][i])
        point = inst["pos"][i] + rot @ (inst["scale"][i] * -d * normal)
        n = rot @ normal
        return n, -float(n @ point)

    def _reflect(self, scene, drawn, pixels, rows, target, level, weight):
        """Show what each mirror drawn in a pass reflects. scene, drawn: the pass (its scene and _accumulate()
        output), which covered `pixels` (flat indices; drawn's rows[j] is pixel pixels[j]); target: its
        whole-frame colours (premultiplied), which the mirrors' share of each pixel is changed in. level:
        how many mirrors deep the pass is; weight: how much of the frame's colour it makes up (the
        reflectivities multiplied along the way), so that faint images deep down are skipped.
        """
        inst, fb, buf = scene["inst"], self._fb, self._buffers
        mirrors = np.flatnonzero(self._mirrors(inst))
        if level >= min(int(self.mirror_bounces), MAX_BOUNCES) or not len(mirrors):
            return
        n = len(SAMPLE_PATTERNS[self.samples])
        cover = buf.get(f"mirror{level}_cover", (len(rows),), np.int64)
        for i in mirrors:
            ident, reflectivity = int(inst["ident"][i]), float(inst["shine"][i])
            if weight * reflectivity < MIRROR_MIN_WEIGHT:
                continue
            _mirror_cover(drawn["tris"], scene["tri_inst"], scene["ident"], ident, cover)
            on = np.flatnonzero(cover)
            if not len(on) or (level > 0 and len(on) < MIRROR_MIN_PIXELS):
                continue
            # The camera reflected in the mirror's plane, which faces it; what is behind the plane is cut away.
            normal, d = self._plane(inst, i)
            eye = scene["eye"]
            if normal @ eye + d < 0.0:
                normal, d = -normal, -d
            mirror = np.eye(4)
            mirror[:3, :3] -= 2.0 * np.outer(normal, normal)
            mirror[:3, 3] = -2.0 * d * normal
            view_proj = scene["view_proj"] @ mirror
            keep = np.arange(len(inst["mesh"])) != i
            sub = {k: (v[keep] if isinstance(v, np.ndarray) and len(v) == len(keep) else v) for k, v in inst.items()}
            prefix = f"mirror{level}_"
            child = self._view(sub, view_proj, (mirror @ np.r_[eye, 1.0])[:3], scene["near"], np.r_[normal, d],
                               self._pass_shine(sub, level + 1), scene["shading"], prefix)
            shown = pixels[on]
            seen = (buf.get(prefix + "rgb_frame", (fb.width * fb.height, 3)),
                    buf.get(prefix + "alpha_frame", (fb.width * fb.height,)),
                    buf.get(prefix + "depth_frame", (fb.width * fb.height,)),
                    buf.get(prefix + "ids_frame", (fb.width * fb.height,), np.int32))
            seen[0][shown], seen[1][shown] = 0.0, 0.0
            if child is not None:
                span = self._spans(shown)
                child["bands"] = self._bins(child["xs"], child["ys"], child["see"], SOLID | CUT, fb.height, prefix,
                                            span)
                pattern = SAMPLE_PATTERNS[self.samples]
                image = self._accumulate(child, pattern, prefix + "pass", shown)
                _gather_pass(shown, image["rgb"], image["cover"], buf.get("near_depth", (len(shown),)),
                             buf.get("near_id", (len(shown),), np.int32), n, *seen)
                # As in the frame itself: mirrors in the reflection, then glass in front, then the background.
                self._reflect(child, image, shown, np.arange(len(shown)), seen[0], level + 1, weight * reflectivity)
                if (child["see"] == 1).any():
                    solid = buf.get(prefix + "solid", (fb.width * fb.height, n))
                    solid.fill(np.inf)  # (nothing is drawn outside the mirror)
                    solid[shown] = image["sample_depth"]
                    layers = self._layers(child, pattern, solid, prefix, span)
                    if layers is not None:
                        self._blend_layers(child, layers, n, seen)
            _fill_sky(shown, seen[0], seen[1], fb.width, fb.height, np.linalg.inv(view_proj), *self._sky)
            extra_rows, extra_tris, extra_rgb = drawn.get("extra") or self._no_extra()
            _mix_mirror(target, shown, rows[on], drawn["tris"], drawn["sample_rgb"], extra_rows, extra_tris, extra_rgb,
                        scene["tri_inst"], scene["ident"], ident, reflectivity, seen[0])

    def _no_extra(self):
        """What _mix_mirror takes for a pass without extra samples at edges: no pixel has any."""
        buf, m = self._buffers, self._fb.width * self._fb.height
        rows = buf.get("no_extra_rows", (m,), np.int64)
        rows.fill(-1)
        return rows, np.zeros((0, 1), np.int32), np.zeros((0, 1, 3))

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
                      inst["mesh"], inst["quat"], inst["scale"], inst["pos"], double, inst_vertex,
                      view_proj, eye, near, clip, vchunk_inst, vchunk_first, vchunk_end,
                      chunk_inst, chunk_first, chunk_end, world, buf.get(prefix + "visible", (n_inst,), np.bool_),
                      whole, cut, off_whole, off_cut)
        return k, world, inst_vertex, (chunk_inst, chunk_first, chunk_end, whole, cut, off_whole, off_cut)

    # ----- shadows -----------------------------------------------------------------------

    def _shadow_maps(self, inst, lights):
        """The shadow maps of every light with shadows, for _shadow(): one for a Light, six (a cube) for a
        PointLight, in the order of the lights' first maps (column 15 of _light_rows()). Returns (texels:
        every map's depth nearest the light, as rasterize_depth() leaves it, one after another; trans:
        what see-through things do to the light at each of those texels, as rasterize_tint() leaves it
        (only filled in for maps whose params say so); mats (N, 4, 4): world to each map's clip space;
        params (N, SHADOW_COLUMNS)). Solid things go into texels and see-through ones into trans.

        The maps are kept while nothing they depend on changes, so moving only the camera costs nothing.
        """
        shadowed = [light for light in lights if light.shadows and self.shadows]
        if not shadowed:
            return _NO_SHADOWS
        size, cube_size = int(self.shadow_size), int(self.point_shadow_size)
        soft = max(float(self.shadow_softness), 0.5)
        # What the maps depend on: where the lights are (not how bright), and what is in the scene.
        key = ([np.asarray(light.position, float).tobytes() + np.float64(light.range).tobytes()
                if isinstance(light, PointLight) else np.asarray(light.direction, float).tobytes()
                for light in shadowed] + [size, cube_size, soft]
               + [inst[k].tobytes() for k in ("mesh", "pos", "quat", "scale", "cast", "rgb", "alpha")], inst["pack"])
        last = self._shadows
        if last is not None and last[0][0] == key[0] and last[0][1] is key[1]:
            return last[1]
        buf, pack = self._buffers, inst["pack"]
        centres, radii = _world_spheres(pack["spheres"][inst["mesh"]], inst["quat"], inst["scale"], inst["pos"])
        cast = np.flatnonzero(inst["cast"])
        # Each light's maps: (world to clip space (maps, 4, 4), params rows, casters).
        views = []
        for light in shadowed:
            if isinstance(light, PointLight):
                position, reach = np.asarray(light.position, float), float(light.range)
                near_enough = cast[np.linalg.norm(centres[cast] - position, axis=1) - radii[cast] < reach]
                face_mats, texel, near = _cube_views(position, reach, cube_size)
                views.append((face_mats, [(1, 0, cube_size, texel, 1.0, soft, *position, 0)] * 6, near_enough))
            else:
                mat, texel, depth_scale = _light_view(np.asarray(light.direction, float), centres, radii, size)
                views.append((mat[None], [(0, 0, size, texel, depth_scale * texel, soft, 0, 0, 0, 0)], cast))
        mats = np.concatenate([view[0] for view in views])
        params = np.array([row for view in views for row in view[1]], dtype=float)
        areas = params[:, 2].astype(np.int64) ** 2
        params[1:, 1] = np.cumsum(areas)[:-1]
        texels = buf.get("shadow_texels", (int(areas.sum()),), np.float32)
        clear = (inst["alpha"] < 1.0) | pack["tints"][inst["mesh"]]  # which objects light goes through
        trans = buf.get("shadow_trans", (4, int(areas.sum())), np.float32) if clear[cast].any() else _NO_SHADOWS[1]
        m = 0
        for view_mats, rows, casters in views:
            offset, n, sides, first = int(params[m, 1]), int(params[m, 2]), len(rows), m
            depth = texels[offset:offset + sides * n * n].reshape(sides * n, n)  # a cube's faces one above another
            m += sides
            if not len(casters):
                depth.fill(0.0)
                continue
            sub_inst = {k: inst[k][casters] for k in ("mesh", "quat", "scale", "pos", "rgb", "alpha")}
            double = np.ones(len(casters), np.bool_)
            if sides == 1:
                k, world, inst_vertex, (chunk_inst, chunk_first, chunk_end, whole, cut, off_whole, off_cut) = \
                    self._transform(pack, sub_inst, double, view_mats[0], np.zeros(3), 0.5, "shadow_")
                xs, ys, dep, surf, see, chain, lod = self._shadow_triangles(k)
                project_depth(pack["faces"], pack["uvs"], pack["colors"], pack["face_chain"], pack["face_texels"],
                              pack["face_kind"], pack["mesh_vertex"], sub_inst["mesh"], sub_inst["rgb"],
                              sub_inst["alpha"], inst_vertex, world, n, n, 0.5, False, chunk_inst, chunk_first,
                              chunk_end, whole, cut, off_whole, off_cut, xs, ys, dep, surf, see, chain, lod)
                tri_side = buf.get("shadow_side", (k,), np.int64)
                tri_side.fill(0)
            else:
                # The casters' vertices in the world, then their triangles in each cube face they reach.
                _, world, inst_vertex, (chunk_inst, chunk_first, chunk_end, *_) = self._transform(
                    pack, sub_inst, double, _EVERYWHERE, np.zeros(3), 0.5, "shadow_")
                counts = buf.get("shadow_counts", (len(chunk_inst), 6), np.int64)
                count_cube(pack["faces"], pack["mesh_vertex"], sub_inst["mesh"], inst_vertex, world, view_mats, near,
                           chunk_inst, chunk_first, chunk_end, counts)
                offsets = np.zeros(counts.shape, np.int64)
                offsets.reshape(-1)[1:] = np.cumsum(counts)[:-1]
                k = int(counts.sum())
                xs, ys, dep, surf, see, chain, lod = self._shadow_triangles(k)
                tri_side = buf.get("shadow_side", (k,), np.int64)
                project_cube(pack["faces"], pack["uvs"], pack["colors"], pack["face_chain"], pack["face_texels"],
                             pack["face_kind"], pack["mesh_vertex"], sub_inst["mesh"], sub_inst["rgb"],
                             sub_inst["alpha"], inst_vertex, world, view_mats, near, n, chunk_inst, chunk_first,
                             chunk_end, offsets, xs, ys, dep, surf, see, chain, lod, tri_side)
            band_start, band_tris = self._bins(xs, ys, see, SOLID | CUT, sides * n, "shadow_")
            rasterize_depth(depth, xs, ys, dep, tri_side, n, band_start, band_tris, see, surf, chain, lod,
                            *pack["textures"])
            if clear[casters].any():
                band_start, band_tris = self._bins(xs, ys, see, CLEAR, sides * n, "shadow_")
                rasterize_tint(trans, offset, n, sides * n, xs, ys, dep, tri_side, n, band_start, band_tris, surf,
                               chain, lod, *pack["textures"])
                params[first:first + sides, 9] = 1
        shadows = (texels, trans, mats, params)
        self._shadows = (key, shadows)
        return shadows

    def _shadow_triangles(self, k):
        """Buffers for k triangles of a shadow map: pixel coordinates (xs, ys), depth and surface (see
        raster.project_depth) of each corner, and each triangle's kind, mipmap chain and mip level."""
        buf = self._buffers
        return (buf.get("shadow_xs", (k, 3)), buf.get("shadow_ys", (k, 3)), buf.get("shadow_depth", (k, 3)),
                buf.get("shadow_surf", (k, 3, SURFACE)), buf.get("shadow_see", (k,), np.int8),
                buf.get("shadow_chain", (k,), np.int64), buf.get("shadow_lod", (k,)))

    # ----- sampling and shading ---------------------------------------------------------

    def _accumulate(self, scene, pattern, name, pixels=None, depth=None, ids=None):
        """Render every sample position in `pattern`, in all pixels or just the given flat pixel indices.

        Returns _resolve()'s per-pixel outputs (rgb, cover, more) and, at each sample, its depth,
        triangle and colour (sample_depth, tris, sample_rgb), in buffers named after `name`; the depth
        and object id of each pixel's nearest sample go into `depth` and `ids` if given.
        """
        fb, buf = self._fb, self._buffers
        if pixels is None:
            pixels = slots = buf.arange(fb.width * fb.height)
        else:
            slots = buf.get("slots", (fb.width * fb.height,), np.int64)
            slots.fill(-1)
            slots[pixels] = buf.arange(fb.width * fb.height)[:len(pixels)]
        n, m = len(pattern), len(pixels)
        sample_depth = buf.get(name + "_sample_depth", (m, n))
        sample_depth.fill(0.0)
        tris = buf.get(name + "_tris", (m, n), np.int32)
        tris.fill(-1)
        rasterize(sample_depth, tris, fb.width, fb.height, scene["xs"], scene["ys"], scene["inv_w"],
                  np.array(pattern, dtype=float), slots, *scene["bands"], scene["see"], scene["attrs"], scene["chain"],
                  scene["lod"], *scene["textures"])
        out = {"rgb": buf.get(name + "_rgb", (m, 3)), "cover": buf.get(name + "_cover", (m,), np.int64),
               "more": buf.get(name + "_more", (m,), np.bool_), "sample_depth": sample_depth, "tris": tris,
               "sample_rgb": buf.get(name + "_sample_rgb", (m, n, 3))}
        depth = buf.get("near_depth", (m,)) if depth is None else depth
        ids = buf.get("near_id", (m,), np.int32) if ids is None else ids
        _resolve(tris, sample_depth, pixels, fb.width, scene["xs"], scene["ys"], scene["inv_w"], scene["attrs"],
                 scene["tri_inst"], scene["ident"], scene["emissive"], scene["shine"], scene["chain"], scene["lod"],
                 *scene["textures"], scene["lights"], *scene["shadows"], scene["pixel_size"], scene["eye"],
                 *scene["sky"], EDGE_CONTRAST,
                 out["sample_rgb"], buf.get(name + "_spec", (m, 3)), out["rgb"], out["cover"], depth, ids, out["more"])
        return out
