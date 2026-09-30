# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
"""Shadow maps: drawing them (Renderer's ShadowMaps part) and looking them up while shading (shadow_lookup).

A Light's shadow map is one orthographic view along the light of a box around everything in the scene; a
PointLight's is a cube of six perspective views out from it. Each texel holds the depth of the nearest thing
blocking the light there, and see-through things add what they let through (tinted shadows).
"""
import numpy as np
from numba import njit

from .lights import PointLight
from .raster import (CLEAR, CUT, SOLID, SURFACE, count_cube, project_cube, project_depth, rasterize_depth,
                     rasterize_tint)
from .transforms import look_at, normalize, perspective


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
def shadow_lookup(texels, trans, mats, params, first, px, py, pz, nx, ny, nz, ndl, footprint):
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


def _world_spheres(spheres, linear, pos):
    """Bounding spheres (centres (N, 3), radii (N,)) in the world of instances whose meshes have bounding
    spheres `spheres` (N, 4: centre, radius), placed by `linear` (N, 3, 3) and positions (see raster.transform)."""
    gram = np.abs(np.einsum("nki,nkj->nij", linear, linear)).sum(axis=2).max(axis=1)  # as raster.stretch
    return np.einsum("nij,nj->ni", linear, spheres[:, :3]) + pos, spheres[:, 3] * np.sqrt(gram)


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


class ShadowMaps:
    """The Renderer's shadow maps (a part of Renderer, in its own module)."""

    def _shadow_maps(self, inst, lights):
        """The shadow maps of every light with shadows, for shadow_lookup(): one for a Light, six (a cube) for a
        PointLight, in the order of the lights' first maps (column 15 of lights.light_rows()). Returns (texels:
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
               + [inst[k].tobytes() for k in ("mesh", "pos", "lin", "cast", "rgb", "alpha")], inst["pack"])
        last = self._shadows
        if last is not None and last[0][0] == key[0] and last[0][1] is key[1]:
            return last[1]
        buf, pack = self._buffers, inst["pack"]
        centres, radii = _world_spheres(pack["spheres"][inst["mesh"]], inst["lin"], inst["pos"])
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
            sub_inst = {k: inst[k][casters] for k in ("mesh", "lin", "flip", "pos", "rgb", "alpha")}
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
