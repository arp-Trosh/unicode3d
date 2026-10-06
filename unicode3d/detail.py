# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
"""Levels of detail: simpler copies of a mesh, for the Renderer to draw in its place where it is small on screen
(see Renderer.simplify).

A level is made by vertex clustering: the mesh's box is cut into cubes `cell` across, the vertices in each cube
are moved to their mean, and faces left with two corners in one cube are dropped. The furthest a vertex moved
(at most a cube's diagonal) is the level's error: the Renderer picks, for each object and frame, the coarsest level
whose error is under Renderer.simplify pixels on screen. Creases are kept: vertices in one cube whose
normals point different ways (the sides of a box) become vertices of their own at the same place, each with its
side's normal. Faces keep their texture coordinates, materials and colours; vertex colours are averaged.
"""
import math

import numpy as np
from numba import njit

from .mesh import Mesh


def _mesh_key(mesh):
    """What a level depends on: the mesh's arrays and textures, compared by identity (as renderer._mesh_key)."""
    return (mesh, mesh.vertices, mesh.faces, mesh.uvs, mesh.materials, mesh.vertex_colors, mesh.face_colors,
            mesh.normals, *mesh.textures)


def _same(a, b):
    return len(a) == len(b) and all(x is y for x, y in zip(a, b))

FINEST = 1 / 256  # the finest cell tried, as a fraction of the mesh's bounding radius
SHRINK = 0.75     # a level is kept only with at most this share of the faces of the next finer one kept
STEP = 2 ** 0.5   # how much larger each cell tried is than the one before
_forgotten = [0]  # how many times forget_levels() was called: levels made before the last call are made afresh


def forget_levels():
    """Make every mesh's levels afresh when next wanted (Renderer.invalidate calls it: arrays edited in place)."""
    _forgotten[0] += 1


def simplified(mesh, cell):
    """(a copy of mesh with its vertices clustered in cubes `cell` across (see the module's notes), or None if no
    faces are left; the furthest a vertex moved). The mesh's vertices must be finite."""
    vertices = np.asarray(mesh.vertices, dtype=float).reshape(-1, 3)
    faces = np.asarray(mesh.faces, dtype=np.int64).reshape(-1, 3)
    normals = mesh.vertex_normals()
    lo = vertices.min(axis=0)
    q = np.floor((vertices - lo) / cell).astype(np.int64)
    dims = q.max(axis=0) + 1
    cube = q[:, 0] + dims[0] * (q[:, 1] + dims[1] * q[:, 2])
    cubes, cube = np.unique(cube, return_inverse=True)
    cube = cube.reshape(-1)
    corner_cube = cube[faces]
    keep = ((corner_cube[:, 0] != corner_cube[:, 1]) & (corner_cube[:, 1] != corner_cube[:, 2])
            & (corner_cube[:, 0] != corner_cube[:, 2]))
    if not keep.any():
        return None, 0.0
    # Each cube's place: its vertices' mean.
    n_cubes = len(cubes)
    count = np.bincount(cube, minlength=n_cubes).astype(float)
    place = np.stack([np.bincount(cube, vertices[:, a], n_cubes) for a in range(3)], axis=1) / count[:, None]
    moved = float(np.sqrt(((vertices - place[cube]) ** 2).sum(axis=1)).max())
    # A vertex of its own for each way normals point in a cube (each normal component rounded to -1, 0 or 1).
    side = np.rint(normals).astype(np.int64) + 1  # (0..2 each)
    keys, of = np.unique(cube * 27 + side[:, 0] * 9 + side[:, 1] * 3 + side[:, 2], return_inverse=True)
    of = of.reshape(-1)
    faces = of[faces[keep]]
    used = np.unique(faces)  # (only what the faces left use)
    renumber = np.full(len(keys), -1, np.int64)
    renumber[used] = np.arange(len(used))
    summed = np.zeros((len(keys), 3))
    np.add.at(summed, of, normals)
    length = np.linalg.norm(summed[used], axis=1, keepdims=True)
    out_n = np.where(length > 1e-12, summed[used] / np.maximum(length, 1e-12), 0.0)  # (zero: worked out)
    level = Mesh(place[keys[used] // 27], renumber[faces], textures=mesh.textures, normals=out_n)
    if mesh.uvs is not None and mesh.materials is not None and mesh.textures:
        level.uvs = np.asarray(mesh.uvs, dtype=float).reshape(-1, 3, 2)[keep]
        level.materials = np.asarray(mesh.materials).reshape(-1)[keep]
    if mesh.vertex_colors is not None:
        colors = np.asarray(mesh.vertex_colors)
        colors = colors.reshape(-1, 4 if colors.ndim > 1 and colors.shape[-1] == 4 else 3)[:len(vertices)]
        summed = np.zeros((len(keys), colors.shape[1]))
        np.add.at(summed, of, colors.astype(float))
        mean = summed[used] / np.bincount(of, minlength=len(keys))[used, None]
        # (0..255 ints stay ints: floats would be read as 0..1)
        level.vertex_colors = np.rint(mean).astype(colors.dtype) if np.issubdtype(colors.dtype, np.integer) else mean
    elif mesh.face_colors is not None:
        colors = np.asarray(mesh.face_colors)
        level.face_colors = colors.reshape(len(keep), -1)[keep]
    return level, moved


def known_levels(mesh):
    """detail_levels(mesh) if they are made and current, else None (without making them)."""
    cached = mesh.__dict__.get("_detail")
    if cached is not None and cached[0][0] == _forgotten[0] and _same(cached[0][1:], _mesh_key(mesh)[1:]):
        return cached[1]
    return None


def detail_levels(mesh):
    """The mesh's levels of detail, finest first, as [(mesh, error)]: the mesh itself (error 0), then simpler
    copies, each with at most SHRINK of the faces of the one before, and the most a vertex moved making it (in the
    mesh's own units). Just the mesh where it can't be made simpler (too few faces, or vertices that aren't
    finite). Kept on the mesh, and made afresh when its arrays are replaced. Calling it ahead of time (while loading
    models, say, on a thread of its own) saves the Renderer making them while it draws."""
    known = known_levels(mesh)
    if known is not None:
        return known
    key = (_forgotten[0], *_mesh_key(mesh)[1:])
    levels = [(mesh, 0.0)]
    try:
        mesh.check()
        vertices = np.asarray(mesh.vertices, dtype=float).reshape(-1, 3)
        faces = len(np.asarray(mesh.faces).reshape(-1, 3))
        ok = faces >= 8 and np.isfinite(vertices).all()
    except (ValueError, TypeError):
        ok = False
    if ok:
        _, radius = mesh.bounds()
        cell, last = radius * FINEST, faces
        while cell <= radius and last >= 8:
            level, moved = simplified(mesh, cell)
            if level is None:
                break
            if len(level.faces) <= SHRINK * last:
                levels.append((level, max(moved, levels[-1][1])))  # (errors only grow, finer to coarser)
                last = len(level.faces)
            cell *= STEP
    from .renderer import _mesh_piece  # (renderer.py imports this module)
    for level, _ in levels[1:]:  # (what the Renderer works out from each level, worked out now too)
        _mesh_piece(level, _mesh_key(level))
    mesh._detail = (key, levels)
    return levels


@njit(cache=True, error_model="numpy")
def allowed_errors(linear, where, mesh_idx, spheres, eye, per_unit, near, simplified, allowed):
    """How far each instance's vertices may move (allowed (N,), in its mesh's own units) to stay within per_unit
    times its distance from the eye (the size of `simplify` pixels there): its mesh's bounding sphere (spheres
    (M, 4)) placed by linear (N, 3, 3) and where (N, 3), the scale its longest column. 0 where that isn't finite,
    where the sphere reaches nearer than `near`, or where not simplified (N,) bool. Serial: about a microsecond for
    a thousand instances; the same numbers as working it out with numpy (Renderer._detail did)."""
    for n in range(len(mesh_idx)):
        m = mesh_idx[n]
        stretch = -np.inf
        for c in range(3):
            s = math.sqrt(linear[n, 0, c] ** 2 + linear[n, 1, c] ** 2 + linear[n, 2, c] ** 2)
            if s > stretch or s != s:  # (a NaN wins, as in numpy's max)
                stretch = s
            if stretch != stretch:
                break
        d = 0.0
        for r in range(3):
            centre = (linear[n, r, 0] * spheres[m, 0] + linear[n, r, 1] * spheres[m, 1]
                      + linear[n, r, 2] * spheres[m, 2]) + where[n, r]
            d += (centre - eye[r]) ** 2
        nearest = math.sqrt(d) - spheres[m, 3] * stretch
        a = per_unit * nearest / stretch
        allowed[n] = a if abs(a) < np.inf and nearest > near and simplified[n] else 0.0


@njit(cache=True, error_model="numpy")
def pick_levels(mesh_idx, allowed, shiny, error, mirrorless, first, level):
    """Each instance's level (level (N,), an index into the pack's levels): the coarsest of its mesh's whose error
    (error (M, L), or mirrorless for shiny (N,) bool instances: levels a shiny one may take, inf beyond) is within
    allowed (N,), its mesh's levels starting at first (M,) (its finest, the mesh itself, if none is)."""
    for n in range(len(mesh_idx)):
        m = mesh_idx[n]
        table = mirrorless if shiny[n] else error
        k = 0
        for j in range(table.shape[1]):
            k += table[m, j] <= allowed[n]
        level[n] = first[m] + max(k - 1, 0)
