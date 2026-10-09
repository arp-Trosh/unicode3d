# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
"""The Renderer: draws a scene's objects (scene.Object3D) into a framebuffer of pixels finer than terminal cells.

It packs the meshes and poses into arrays, projects every triangle, rasterizes several samples a pixel, shades
them (shading.py), blends see-through surfaces over the solid ones, and adds what mirrors show (mirrors.py),
shadows (shadows.py), fog and outlines, and the background.
"""
import math
import operator
import time
from dataclasses import dataclass

import numpy as np

from .background import SKYBOX_FACES, background_args, fill_background, fog_args
from .color import cached_linear_rgb
from .lights import LIGHT_COLUMNS, as_lights, light_rows
from .mirrors import Mirrors
from .raster import (ATTRS, CLEAR, CUT, FACE_CHUNK, ROW_BAND, RUN_CELLS, RUN_LEAST, RUN_MOST, SAMPLE_PATTERNS, SOLID,
                     VERTEX_CHUNK, FrameBuffer, bin_bands, count_bands, face_runs, project, rasterize, rasterize_layers,
                     rasterize_pixels, transform, upscale, NO_CLIP)
from .shading import blend, post_effects, resolve, resolve_cells
from .shadows import SHADOW_FITS, ShadowMaps
from .texture import alpha_kind, pack as pack_textures
from .threads import kernel_lock
from .transforms import normalize, perspective, scene_poses, view_axes
from .detail import FINEST, allowed_errors, detail_levels, forget_levels, known_levels, pick_levels

MAX_PIXELS = 1920 * 1080  # the default Renderer.max_pixels
SHADINGS = ("cell", "pixel", "coarse")  # Renderer.shading
EDGE_CONTRAST = 0.03  # linear-light spread among a pixel's first samples that marks it for more
DETAIL_BUDGET = 0.003  # seconds a frame spent making levels of detail (see Renderer._detail)
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


def _same(a, b):
    """Whether two lists hold the same objects (compared by identity), in the same order."""
    return a is b or len(a) == len(b) and all(map(operator.is_, a, b))


_object_attrs = operator.attrgetter("mesh", "opacity", "color", "simplify", "double_sided", "emissive",
                                    "cast_shadows", "reflectivity", "specular", "shininess")  # (see _instances)


def _numbers(values):
    """values as a float array, NaN (and None) as 0: a NaN colour or glow would reach the picture."""
    a = np.array(values, np.float64)
    return np.where(np.isnan(a), 0.0, a) if np.isnan(a).any() else a


def _determinants(m):
    """The determinants of matrices m (N, 3, 3), written out: several times faster than np.linalg.det for many
    small matrices."""
    return (m[:, 0, 0] * (m[:, 1, 1] * m[:, 2, 2] - m[:, 1, 2] * m[:, 2, 1])
            - m[:, 0, 1] * (m[:, 1, 0] * m[:, 2, 2] - m[:, 1, 2] * m[:, 2, 0])
            + m[:, 0, 2] * (m[:, 1, 0] * m[:, 2, 1] - m[:, 1, 1] * m[:, 2, 0]))


def _immutable_color(color):
    """Whether a colour can't be changed in place: a number (a named Color too), or a tuple of numbers."""
    if isinstance(color, (int, np.integer)):
        return True
    return type(color) is tuple and all(isinstance(c, (int, float, np.integer, np.floating)) for c in color)


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


@dataclass
class _Piece:
    """One mesh's share of a pack (see _Packer): its arrays copied (so edits made in place, unchecked, can't reach
    the kernels) and checked, with what is worked out from them."""
    vertices: np.ndarray  # (V, 3) float64
    normals: np.ndarray   # (V, 3) unit normals
    faces: np.ndarray     # (F, 3) int64, indexing this mesh's vertices
    uvs: np.ndarray       # (F, 3, 2)
    colors: np.ndarray    # (F, 3, 4) linear rgb and alpha at each corner
    materials: np.ndarray  # (F,) int64 into chains (zeros if untextured)
    chains: list          # each material's mipmap chain (Mesh.mipmaps), or [None] if untextured
    texels: np.ndarray    # (materials,) each texture's texels (height x width)
    used: np.ndarray      # the materials its faces use
    tinted: bool          # whether any of its corner colours is see-through
    sphere: tuple         # (centre x, y, z, radius)
    plane: tuple          # (normal x, y, z, d), NaN if not flat (see _flat_plane)
    runs: np.ndarray      # (R,) where each run of faces starts, culled as one (see _runs)
    run_spheres: np.ndarray  # (R, 4) a sphere around each run's corners (NaN: never culled)
    run_vertex: np.ndarray   # (R, 2) the vertices each run's faces use: first, end


_edits = 0  # how many times Renderer.invalidate() was called: pieces made before the last call are made afresh


def _mesh_piece(mesh, key):
    """mesh's _Piece (key: its _mesh_key), kept on the mesh while its arrays are the same ones (as Mesh keeps its
    normals), so that a mesh drawn again after a while, or by another Renderer, isn't worked out afresh."""
    cached = mesh.__dict__.get("_piece")
    if cached is None or cached[0] != _edits or not _same(cached[1], key[1:]):
        cached = mesh._piece = (_edits, key[1:], _piece(mesh))  # (not the mesh itself: no reference cycle)
    return cached[2]


def _piece(mesh):
    """A mesh's _Piece. Raises ValueError if its arrays don't fit together (Mesh.check)."""
    mesh.check()  # (the kernels index these arrays by each other's sizes without checking)
    vertices = np.array(mesh.vertices, dtype=np.float64)
    faces = np.array(mesh.faces, dtype=np.int64).reshape(-1, 3)
    n_faces = len(faces)
    if mesh.materials is not None and mesh.textures:
        materials = np.array(mesh.materials, dtype=np.int64)
        chains = [mesh.mipmaps(m) for m in range(len(mesh.textures))]
        texels = np.array([np.shape(t)[0] * np.shape(t)[1] for t in mesh.textures], dtype=float)
        uvs = np.array(mesh.uvs, dtype=np.float64)
    else:
        materials, chains, texels = np.zeros(n_faces, np.int64), [None], np.zeros(1)
        uvs = np.zeros((n_faces, 3, 2))
    corner = mesh.corner_colors()
    centre, radius = mesh.bounds()
    return _Piece(vertices, mesh.vertex_normals(), faces, uvs,
                  np.ones((n_faces, 3, 4)) if corner is None else corner, materials, chains, texels,
                  np.unique(materials), corner is not None and bool((corner[:, :, 3] < 1.0).any()),
                  (*centre, radius), _flat_plane(vertices, faces, radius), *_runs(vertices, faces, centre, radius))


def _runs(vertices, faces, centre, radius):
    """A mesh's faces cut into runs, each culled as one when it is out of view (raster.transform): (where each
    starts (R,), a sphere around each (R, 4), the vertices each uses (R, 2)). A mesh of up to FACE_CHUNK faces is
    one run, in the mesh's own sphere; a bigger one (a level, say) is cut where its runs would grow beyond
    1/RUN_CELLS of its size across (raster.face_runs), into runs of at least RUN_LEAST faces and at most about
    RUN_MOST runs, so that most of it can be left out where most of it is out of view. Faces keep their order, so
    what is drawn doesn't change."""
    n = len(faces)
    limit, least = np.inf, max(RUN_LEAST, -(-n // RUN_MOST))
    if n > FACE_CHUNK:
        used = vertices[np.unique(faces)]
        extent = float((used.max(axis=0) - used.min(axis=0)).max())
        if math.isfinite(extent):
            limit = extent / RUN_CELLS
    starts = np.empty(n, np.int64)
    starts = starts[:face_runs(vertices, faces, limit, least, FACE_CHUNK, starts)].copy()
    if not n:
        return starts, np.zeros((0, 4)), np.zeros((0, 2), np.int64)
    run_vertex = np.stack([np.minimum.reduceat(faces.min(axis=1), starts),
                           np.maximum.reduceat(faces.max(axis=1), starts) + 1], axis=1)
    if len(starts) == 1:
        return starts, np.array([(*centre, radius)], np.float64), run_vertex
    corners = vertices[faces]  # (F, 3, 3)
    lo = np.minimum.reduceat(corners.min(axis=1), starts)
    hi = np.maximum.reduceat(corners.max(axis=1), starts)
    middle = (lo + hi) / 2
    run_of = np.repeat(np.arange(len(starts)), np.diff(np.append(starts, n)))
    reach = np.sqrt(((corners - middle[run_of][:, None, :]) ** 2).sum(axis=2)).max(axis=1)
    spheres = np.concatenate([middle, np.maximum.reduceat(reach, starts)[:, None]], axis=1)
    spheres[~np.isfinite(spheres).all(axis=1)] = np.nan
    return starts, spheres, run_vertex


class _Packer:
    """Packs meshes' arrays together for the kernels (see pack()), from each mesh's _Piece (kept on the mesh) and
    textures kept from one pack to the next while they hold every chain the meshes show. Meshes are added to the end
    of the last pack and kept there after they go out of use, until it holds as many faces again as are in use: so a
    render list that gains an object (an arrow, a puff of dust) costs the new mesh's share and one copy of the
    arrays, not a hitch (in Castle Panic, working out every piece and packing every texture again took half a
    second, and packing every mesh again 5 ms)."""

    def __init__(self):
        self._textures = None  # (chains, {id(chain): index}, alpha kinds (K,), pack() of them, the array its
                               # texels are the start of) as last packed
        self._packed = None  # (meshes, their keys, {id(mesh): index}, pieces, pack, the room its textures are in)
        self._room = {}  # {name: array} the pack's arrays are the start of, with room for more (see _joined)

    def pack(self, meshes, keys):
        """(the pack: every mesh's arrays together for raster.project, and their textures for sampling; where each
        of the meshes is in it (M,); the keys (_mesh_key) of the pack's meshes, in its order). keys: each mesh's
        _mesh_key. The pack may hold other meshes too, drawn before."""
        packed = self._packed
        if packed is not None:
            held, held_keys, index, pieces, arrays, room = packed
            where, new, new_keys, ids = [], [], [], {}
            for mesh, key in zip(meshes, keys):
                i = index.get(id(mesh))
                if i is None:
                    i = ids.get(id(mesh))
                    if i is None:
                        i = ids[id(mesh)] = len(held) + len(new)
                        new.append(mesh)
                        new_keys.append(key)
                elif not _same(held_keys[i][1:], key[1:]):  # (its arrays were replaced: all afresh)
                    packed = None
                    break
                where.append(i)
        if packed is not None and new:
            more = [_mesh_piece(mesh, key) for mesh, key in zip(new, new_keys)]
            in_use = sum(len(p.faces) for p in (pieces[i] if i < len(pieces) else more[i - len(pieces)]
                                                for i in set(where)))
            if sum(len(p.faces) for p in pieces) + sum(len(p.faces) for p in more) > 2 * in_use + 65536:
                packed = None  # (mostly meshes out of use: only those in use, afresh)
            else:
                chains, index_of, kinds = self._pack_textures(pieces + more)
                if self._textures[4] is not room:  # (the textures were packed afresh: their places moved)
                    packed = None
                else:
                    added = _arrays(more, index_of, kinds, len(arrays["vertices"]), len(arrays["faces"]),
                                    len(arrays["run_first"]))
                    arrays = _joined(arrays, added, self._textures[3], self._room)
                    index = {**index, **{id(mesh): len(held) + k for k, mesh in enumerate(new)}}
                    self._packed = (held + new, held_keys + new_keys, index, pieces + more, arrays, room)
                    return arrays, np.array(where, np.int64), self._packed[1]
        if packed is not None:
            return arrays, np.array(where, np.int64), held_keys
        # Afresh, with just these meshes.
        unique, unique_keys, index = [], [], {}
        for mesh, key in zip(meshes, keys):
            if id(mesh) not in index:
                index[id(mesh)] = len(unique)
                unique.append(mesh)
                unique_keys.append(key)
        pieces = [_mesh_piece(mesh, key) for mesh, key in zip(unique, unique_keys)]
        chains, index_of, kinds = self._pack_textures(pieces)
        arrays = _arrays(pieces, index_of, kinds, 0, 0, 0)
        arrays["textures"] = self._textures[3]
        self._packed = (unique, unique_keys, index, pieces, arrays, self._textures[4])
        return arrays, np.array([index[id(mesh)] for mesh in meshes], np.int64), unique_keys

    def _pack_textures(self, pieces):
        """The texture pack's (chains, {id(chain): index}, alpha kinds) for the pieces. Chains it lacks are added in
        the room left at the end of its texels (earlier packs, which may still be in use, keep theirs); it is packed
        afresh with just the pieces' chains when that room runs out, or when it is more than twice the size of what
        the pieces show (after many textures are let go)."""
        wanted, seen = [], set()
        for p in pieces:
            for chain in p.chains:
                if chain is not None and id(chain) not in seen:
                    seen.add(id(chain))
                    wanted.append(chain)
        def texels(chains):
            return sum(level.shape[0] * level.shape[1] for chain in chains for level in chain)
        size, packed = texels(wanted), self._textures
        if packed is not None:
            chains, index, kinds, (done, levels, first), room = packed
            new = [chain for chain in wanted if id(chain) not in index]
            end = len(done) + texels(new)
            if not new and end <= 2 * size + 4096:
                return chains, index, kinds
            if new and end <= len(room) and end <= 2 * size + 4096:
                added, more_levels, more_first = pack_textures(new, room, len(done))
                chains = chains + new
                index = {**index, **{id(chain): len(index) + k for k, chain in enumerate(new)}}
                kinds = np.concatenate([kinds, np.array([alpha_kind(chain) for chain in new], np.int8)])
                self._textures = (chains, index, kinds, (added, np.concatenate([levels, more_levels]),
                                                         np.concatenate([first, more_first[1:] + len(levels)])), room)
                return chains, index, kinds
        # Room for as many textures again (the most it holds before it is packed afresh anyway): np.empty takes
        # memory from the system only as it is written.
        room = np.empty((2 * size + 4096, 4), np.float32)
        kinds = np.array([alpha_kind(chain) for chain in wanted], np.int8)
        self._textures = (wanted, {id(chain): k for k, chain in enumerate(wanted)}, kinds,
                          pack_textures(wanted, room), room)
        return self._textures[0], self._textures[1], self._textures[2]


def _arrays(pieces, index, kinds, first_vertex, first_face, first_run):
    """The pieces' arrays packed together (as _Packer.pack gives them, without the textures), their vertices,
    faces and runs numbered from first_vertex, first_face and first_run: to go after a pack of that many. index,
    kinds: the texture pack's {id(chain): index} and alpha kinds."""
    n_vertices = np.array([len(p.vertices) for p in pieces], np.int64)
    n_faces = np.array([len(p.faces) for p in pieces], np.int64)
    mesh_vertex = np.concatenate([[0], np.cumsum(n_vertices)])
    mesh_face = np.concatenate([[0], np.cumsum(n_faces)])
    n_runs = np.array([len(p.runs) for p in pieces], np.int64)
    run_first = np.concatenate([p.runs for p in pieces]).astype(np.int64) + np.repeat(mesh_face[:-1], n_runs)
    run_end = np.concatenate([np.append(p.runs[1:], len(p.faces)) if len(p.runs) else p.runs
                              for p in pieces]).astype(np.int64) + np.repeat(mesh_face[:-1], n_runs)
    # Every mesh's materials in one table: the chain each shows (-1: none) and its texture's size.
    table = np.array([-1 if c is None else index[id(c)] for p in pieces for c in p.chains], np.int64)
    first = np.concatenate([[0], np.cumsum([len(p.chains) for p in pieces])])
    face_chain = table[np.concatenate([p.materials for p in pieces]) + np.repeat(first[:-1], n_faces)]
    face_texels = np.concatenate([p.texels for p in pieces])[
        np.concatenate([p.materials for p in pieces]) + np.repeat(first[:-1], n_faces)]
    kind_of = np.append(kinds, np.int8(0))  # (chain -1: none)
    face_kind = kind_of[face_chain]
    # Whether light goes through any of it (see-through colours or textures), and whether its far side
    # shows through anything (those, or holes in its textures).
    tints, clear = [], []
    for p, at in zip(pieces, first):
        used = kind_of[table[at + p.used]]
        tints.append(p.tinted or bool((used == 2).any()))
        clear.append(tints[-1] or bool(used.any()))
    faces = (np.concatenate([p.faces for p in pieces]).reshape(-1, 3)
             + (first_vertex + np.repeat(mesh_vertex[:-1], n_faces))[:, None])
    c = np.ascontiguousarray
    return {"vertices": c(np.concatenate([p.vertices for p in pieces]).reshape(-1, 3), np.float64),
            "normals": c(np.concatenate([p.normals for p in pieces]).reshape(-1, 3), np.float64),
            "faces": c(faces, np.int64), "uvs": c(np.concatenate([p.uvs for p in pieces]).reshape(-1, 3, 2), np.float64),
            "colors": c(np.concatenate([p.colors for p in pieces]).reshape(-1, 3, 4), np.float64),
            "face_chain": c(face_chain, np.int64), "face_kind": c(face_kind, np.int8),
            "face_texels": c(face_texels, np.float64),
            "mesh_vertex": (first_vertex + mesh_vertex).astype(np.int64),
            "mesh_face": (first_face + mesh_face).astype(np.int64),
            "spheres": np.array([p.sphere for p in pieces], np.float64).reshape(-1, 4),
            "mesh_run": (first_run + np.concatenate([[0], np.cumsum(n_runs)])).astype(np.int64),
            "run_first": c(first_face + run_first, np.int64), "run_end": c(first_face + run_end, np.int64),
            "run_spheres": c(np.concatenate([p.run_spheres for p in pieces]).reshape(-1, 4), np.float64),
            "run_vertex": c(np.concatenate([p.run_vertex for p in pieces]).reshape(-1, 2), np.int64),
            "clear": np.array(clear, np.bool_), "tints": np.array(tints, np.bool_),
            "planes": np.array([p.plane for p in pieces], np.float64).reshape(-1, 4)}


def _joined(arrays, added, textures, room):
    """A pack (arrays) with more meshes (added: _arrays() of them, numbered to follow) after its own, and the
    texture pack now in use. Each array is the start of a longer one kept in room ({name: array}), and the meshes
    are written after it there while it has space: earlier packs, which may still be in use, are left as they
    were. Otherwise into new arrays with as much space again."""
    joined = {}
    for name, more in added.items():
        offsets = name in ("mesh_vertex", "mesh_face", "mesh_run")  # (the end of one is the start of the next)
        old = arrays[name][:-1] if offsets else arrays[name]
        end = len(old) + len(more)
        whole = room.get(name)
        if whole is None or whole.base is not None or len(whole) < end or not _starts(whole, old):
            whole = np.empty((2 * end + 64,) + more.shape[1:], more.dtype)
            whole[:len(old)] = old
            room[name] = whole
        whole[len(old):end] = more
        joined[name] = whole[:end]
    joined["textures"] = textures
    return joined


def _starts(whole, part):
    """Whether part is a view of the start of whole."""
    return part.base is whole and part.__array_interface__["data"][0] == whole.__array_interface__["data"][0]


def _work(pack, inst_mesh):
    """How raster.transform() and transform_depth() split the instances showing meshes inst_mesh of `pack` into
    pieces that run in parallel: (inst_vertex (where each instance's rows in `world` start, and their total), its
    chunks of vertices (vchunk_inst, vchunk_first, vchunk_end, and where each instance's start: inst_vchunk), its
    chunks of faces (chunk_inst, chunk_run, chunk_first, chunk_end): its mesh's runs (see _runs), each left out when
    out of view, with the vertex chunks only they use)."""
    n_inst = len(inst_mesh)
    vertex_counts = np.diff(pack["mesh_vertex"])[inst_mesh]
    inst_vertex = np.zeros(n_inst + 1, np.int64)
    np.cumsum(vertex_counts, out=inst_vertex[1:])
    vchunk_inst, vchunk_first, vchunk_end = _chunks(vertex_counts, np.zeros(n_inst, np.int64), VERTEX_CHUNK)
    inst_vchunk = np.zeros(n_inst + 1, np.int64)
    np.cumsum((vertex_counts + VERTEX_CHUNK - 1) // VERTEX_CHUNK, out=inst_vchunk[1:])
    mesh_run = pack["mesh_run"]
    first_run = mesh_run[inst_mesh]
    runs = mesh_run[inst_mesh + 1] - first_run
    chunk_inst = np.repeat(np.arange(n_inst, dtype=np.int64), runs)
    chunk_run = np.arange(len(chunk_inst), dtype=np.int64) + np.repeat(first_run - (np.cumsum(runs) - runs), runs)
    chunk_first, chunk_end = pack["run_first"][chunk_run], pack["run_end"][chunk_run]
    return (inst_vertex, (vchunk_inst, vchunk_first, vchunk_end, inst_vchunk),
            (chunk_inst, chunk_run, chunk_first, chunk_end))


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
    shadow_fit: what each Light's shadow map covers. "view" (the default): only what the camera can see of the
    scene (and a margin), so that shadows are sharper the less of the scene is seen (an
    isometric game's view of a big level: about 4x), and casters whose shadows fall out of sight aren't drawn. Its
    box moves in whole texels and changes size in steps, so shadows keep still as the camera moves; the casters
    that keep still are drawn into a map up to 1024 texels wider, which stays put while the camera moves about in
    it (redrawn when the camera leaves it). More texels of the map are drawn into than with "scene", as it is
    finer. Scenes with mirrors (which show what the camera doesn't) get "scene"'s maps, as do views of most of the
    scene. "scene": everything in the scene, so that moving only the camera costs nothing, but shadows are as
    coarse as the scene is big; cheaper to draw (quality.AutoQuality's "mid" and "low" step down to it).
    PointLights' shadows are the same either way.
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
    simplify: how many pixels on screen a simpler copy of a mesh (a level of detail, see detail.py) may differ by
    where it is drawn in the mesh's place, for objects small on screen; 0 (the default) draws every mesh as it
    is. About 1 makes scenes of many detailed models faster (Castle Panic: half the triangles drawn) for a
    change to the picture of under a pixel. Levels are made for a mesh the first time it is small enough to use
    one, a few meshes a frame (detail.detail_levels makes them ahead of time). Objects with simplify=False
    (lettering, say) are always drawn as they are.
    shading: how often surfaces are lit. "cell" (the default): once per terminal cell for each surface in it, as a
    cell's pixels end up as two colours anyway, except in cells a shadow's edge crosses, which are lit pixel by
    pixel; colours and textures are still worked out for every pixel (a quarter to a third less shading time than
    "pixel", the picture all but the same: highlights and light falloff vary from cell to cell rather than pixel to
    pixel). "pixel": every pixel lit on its own (exact, slowest). "coarse": everything once per cell and surface,
    textures too, and shadow edges not looked for (about two thirds less shading time; texture detail inside a cell,
    such as a brick's mortar, is lost, and shadow edges step from cell to cell): for slow machines (AutoQuality's
    lowest steps). Pixels at the edges of shapes get more samples (edge_samples) shaded per pixel whatever the
    setting.
    """

    def __init__(self, width, height, cell_pixels=(1, 2), cell_aspect=0.5, samples=4, edge_samples=8,
                 fog=0.3, outline=0.55, lod_bias=-0.5, background=None, shadow_size=1024, point_shadow_size=256,
                 shadow_softness=1.5, shadow_fit="view", shadows=True, transparency_layers=4, reflections=True, mirror_bounces=1,
                 max_pixels=MAX_PIXELS, simplify=0.0, shading="cell"):
        for n in (samples, edge_samples):
            if n not in SAMPLE_PATTERNS and n != 0:
                raise ValueError(f"sample counts must be one of {sorted(SAMPLE_PATTERNS)}, not {n}")
        self.cell_aspect = cell_aspect
        self.samples = samples
        self.edge_samples = edge_samples
        self.fog = fog
        self.outline = outline
        self.lod_bias = lod_bias
        self.simplify = simplify
        self.shading = shading
        self.background = background
        self.shadow_size = shadow_size
        self.point_shadow_size = point_shadow_size
        self.shadow_softness = shadow_softness
        self.shadow_fit = shadow_fit
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
        self._pack = None  # (detail version, flat mesh keys, the pack, each mesh's place in it, its keys and flat keys)
        self._detail_memo = None  # the levels of detail of the meshes last drawn (see _detail)
        self._chain_rows = {}  # {id(levels): (levels, mesh, keys, row)} (see _detail_rows)
        self._poses_memo = {}  # the scene graph's order as last drawn (see scene_poses), and each object's row in it
        self._flat = []  # the meshes' keys as last drawn, end to end (see _instances)
        self._colors = None  # (the objects' colours, their linear rgb, ...) as last drawn, if none can change in place
        self._packer = _Packer()
        self._objects = []  # the render list as last drawn, for pick()
        self._eye = self._axes = None  # the camera's position and its pixels' directions (view_axes) then
        self._ortho = 0.0  # for an orthographic camera, how far behind it the eye it is drawn from is (Camera.drawn_as)
        self._buffers = _Buffers()
        self._shadows = None  # (what the shadow maps depend on, the maps as _shadow_maps() returns them)
        self._settled = {}  # each shadowed light's map of its settled casters (shadows._Settled), by light
        self._caster_ages = None  # (shadows._Rows of the solid casters, shadow draws each has been unchanged for)
        self._caster_meshes = None  # (the pack's mesh keys, the id of each one's Mesh, the keys by those ids)
        self._shadow_draws = 0
        self._settled_draws = 0  # maps of settled casters drawn (shadows._Settled), for tests and benchmarks
        self._view_fits = {}  # each view-fitted Light's map's (light, half-width, its region's centre), by light
        self._mesh_boxes = None  # (the pack, each mesh's box: middles and half-widths) for shadow_fit "view"
        self.resize(width, height)

    @property
    def shading(self):
        """How often surfaces are lit: "cell", "pixel" or "coarse" (see the class)."""
        return self._shading

    @shading.setter
    def shading(self, value):
        if value not in SHADINGS:
            raise ValueError(f"shading must be one of {SHADINGS}, not {value!r}")
        self._shading = value

    @property
    def shadow_fit(self):
        """What each Light's shadow map covers: "scene" or "view" (see the class)."""
        return self._shadow_fit

    @shadow_fit.setter
    def shadow_fit(self, value):
        if value not in SHADOW_FITS:
            raise ValueError(f"shadow_fit must be one of {SHADOW_FITS}, not {value!r}")
        self._shadow_fit = value

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
        global _edits
        _edits += 1  # (every mesh's _Piece: which meshes were edited isn't known)
        self._last = None
        self._pack = None
        self._packer = _Packer()
        self._detail_memo = None
        self._chain_rows = {}
        forget_levels()
        self._shadows = None
        self._settled = {}
        self._caster_ages = None
        self._caster_meshes = None

    def _instances(self, objects, camera=None):
        """The objects to draw, as arrays: their meshes' place in the pack, world poses (linear (3, 3) and
        position), colours, materials, flags and ids (render-list index + 1). Packs the meshes afresh if the set
        of meshes has changed. With a camera and simplify on, each object's mesh is the level of detail (see
        _detail) it is drawn with."""
        # Everything read from each object, in one pass (_object_attrs). Wholly clear objects, or with a NaN
        # opacity, have nothing to draw, nor any shadow.
        attrs = list(map(_object_attrs, objects))
        candidates = [i for i, a in enumerate(attrs) if a[0] is not None and len(a[0].faces) and a[1] > 0.0]
        if not candidates:
            return None
        memo = self._poses_memo
        known = memo.get("order")
        if len(candidates) == len(objects):  # (as usual: every one has something to draw)
            nodes = objects
        else:
            nodes = [objects[i] for i in candidates]
            attrs = [attrs[i] for i in candidates]
        rows, linear, where, visible = scene_poses(nodes, memo=memo)
        if memo.get("order") is not known or "at" not in memo:  # (the order changed: where each one is, afresh)
            memo["at"] = np.array([rows[id(obj)] for obj in nodes], np.int64)
        at = memo["at"]
        # Hidden (or under a hidden parent), or NaN or infinite in a pose (a physics blow-up, say): not drawn.
        drawn = visible[at] & np.isfinite(where[at]).all(axis=1) & np.isfinite(linear[at]).all(axis=(1, 2))
        keep = np.flatnonzero(drawn)
        if not len(keep):
            return None
        at = at[keep]
        linear, where = linear[at], where[at]
        if len(keep) == len(candidates):  # (as usual: every one drawn)
            ident = np.array(candidates, np.int32) + 1
        else:
            attrs = [attrs[k] for k in keep]
            ident = np.array(candidates, np.int32)[keep] + 1
        on, alpha, colors, simplify, double, emissive, cast, shine, spec, shiny = zip(*attrs)
        last = memo.get("meshes")
        if last is not None and _same(last[0], on):  # (the same meshes on the same objects: the same indices)
            meshes, mesh_idx = last[1], last[2]
        else:
            meshes, mesh_of, mesh_idx = [], {}, []
            for mesh in on:
                m = mesh_of.get(id(mesh))
                if m is None:
                    m = mesh_of[id(mesh)] = len(meshes)
                    meshes.append(mesh)
                mesh_idx.append(m)
            mesh_idx = np.array(mesh_idx, np.int64)
            memo["meshes"] = (on, meshes, mesh_idx)
        keys = list(map(_mesh_key, meshes))  # (afresh: a mesh's arrays may have been replaced)
        rgb = self._linear_colors(colors)
        emissive, shine, spec, shiny = _numbers((emissive, shine, spec, shiny))
        shine = np.clip(shine, 0.0, 1.0)
        flat = [x for key in keys for x in (*key, None)]  # (None ends each mesh's: compared by identity)
        if _same(self._flat, flat):  # (as usual: then the same list, which _detail and the pack find at once)
            flat = self._flat
        self._flat = flat
        version = None
        if self.simplify > 0 and camera is not None:  # (every level of each mesh packed, and which one each draws)
            simplified = np.fromiter(simplify, np.bool_, len(simplify))
            meshes, keys, mesh_idx, version = self._detail(meshes, keys, flat, mesh_idx, linear, where, shine > 0.0,
                                                           simplified, camera)
        placed = self._pack
        if placed is None or placed[0] is not version or not _same(placed[1], flat):
            arrays, at_pack, pack_keys = self._packer.pack(meshes, keys)
            placed = self._pack = (version, flat, arrays, at_pack, pack_keys,
                                   [x for key in pack_keys for x in (*key, None)])
        _, _, arrays, at_pack, mesh_keys, keys = placed
        mesh_idx = at_pack[mesh_idx]
        alpha = np.clip(np.array(alpha, np.float64), 0.0, 1.0)
        # See-through objects, and those with holes, always show their far side, through their near one.
        double = np.array(double, np.bool_) | (alpha < 1.0) | arrays["clear"][mesh_idx]

        return {"mesh": mesh_idx, "pos": where, "lin": np.ascontiguousarray(linear),
                "flip": _determinants(linear) < 0.0,
                "rgb": rgb, "double": double, "alpha": alpha,
                "shine": shine,
                "specular": np.maximum(spec, 0.0),
                "shininess": np.maximum(shiny, 0.0),  # (None, each light's own, as 0)
                "ident": ident,
                "emissive": emissive, "cast": np.array(cast, np.bool_), "pack": arrays,
                "keys": keys, "mesh_keys": mesh_keys}

    def _detail(self, meshes, keys, flat, mesh_idx, linear, where, shiny, simplified, camera):
        """Levels of detail (detail.py) for the instances seen from camera: (meshes, keys, mesh_idx, version), with
        every level of each mesh among the meshes (so that an object changing level doesn't repack anything), each
        instance's mesh the coarsest level that moves no vertex more than `simplify` pixels on screen. Shiny
        objects (shiny: (N,) bool) never take a level that is flat where their mesh isn't: a flat shiny mesh is
        a mirror (see Mirrors), a pass over the whole scene; objects not `simplified` ((N,) bool: Object3D.simplify)
        are drawn as they are. Levels are made for a mesh once an object showing it
        is small enough to use one, for at most DETAIL_BUDGET seconds a frame; until then it is drawn as it is. version
        is the same object while the meshes are the same."""
        memo = self._detail_memo
        if memo is None or not _same(memo["flat"], flat):
            chains = [known_levels(mesh) for mesh in meshes]
            spheres = np.array([(*m.bounds()[0], m.bounds()[1]) for m in meshes], np.float64).reshape(-1, 4)
            memo = self._detail_memo = {"flat": flat, "chains": chains, "spheres": spheres, "levels": None}
        chains, spheres = memo["chains"], memo["spheres"]
        # How far each vertex may move, in its mesh's own units: `simplify` pixels where the object is nearest.
        allowed = np.empty(len(mesh_idx))
        allowed_errors(np.ascontiguousarray(linear), np.ascontiguousarray(where), mesh_idx, spheres,
                       np.ascontiguousarray(camera.position, np.float64), float(self.simplify) * self._pixel_size(camera),
                       float(camera.near), simplified, allowed)
        # Levels for meshes now small enough to use one, within the budget.
        wanted = np.zeros(len(meshes), np.bool_)
        wanted[mesh_idx[allowed >= FINEST * spheres[mesh_idx, 3]]] = True  # (no level moves less than that, nearly)
        missing = [j for j in np.flatnonzero(wanted) if chains[j] is None]
        if missing:
            start = time.perf_counter()
            for j in missing:
                chains[j] = detail_levels(meshes[j])
                memo["levels"] = None
                if time.perf_counter() - start > DETAIL_BUDGET:
                    break
        if memo["levels"] is None:  # (the pack's meshes, and each level's error, afresh)
            rows = [self._detail_rows(mesh, key, chain) for mesh, key, chain in zip(meshes, keys, chains)]
            first = np.cumsum([0] + [len(row[0]) for row in rows[:-1]])
            most = max(len(row[0]) for row in rows)
            error = np.full((len(meshes), most), np.inf)
            mirrorless = error.copy()
            for j, (_, _, e, m) in enumerate(rows):
                error[j, :len(e)] = e
                mirrorless[j, :len(m)] = m
            memo["levels"] = ([level for row in rows for level in row[0]], [key for row in rows for key in row[1]],
                              np.array(first, np.int64), error, mirrorless)
        levels = memo["levels"]
        every, every_keys, first, error, mirrorless = levels
        level = np.empty(len(mesh_idx), np.int64)
        pick_levels(mesh_idx, allowed, shiny, error, mirrorless, first, level)
        return every, every_keys, level, levels

    def _detail_rows(self, mesh, key, chain):
        """(levels, their keys, their errors, the errors of those a shiny object may take: up to the first one that
        is flat where the mesh isn't) of a mesh with levels `chain` (detail_levels, or None for just the mesh),
        kept from frame to frame for each chain."""
        if chain is None:
            return [mesh], [key], [0.0], [0.0]
        kept = self._chain_rows.get(id(chain))
        if kept is not None and kept[0] is chain and kept[1] is mesh and _same(kept[2][0][1:], key[1:]):
            return kept[3]
        levels = [level for level, _ in chain]
        keys = [key] + [_mesh_key(level) for level in levels[1:]]
        errors = [error for _, error in chain]
        flat = [math.isfinite(_mesh_piece(level, k).plane[0]) for level, k in zip(levels, keys)]
        shiny = errors[:next((i for i in range(1, len(flat)) if flat[i] and not flat[0]), len(flat))]
        if len(self._chain_rows) > 4 * len(self._pack[4] if self._pack else ()) + 1024:
            self._chain_rows.clear()
        row = (levels, keys, errors, shiny)
        self._chain_rows[id(chain)] = (chain, mesh, keys, row)
        return row

    def _linear_colors(self, colors):
        """The objects' colours (Object3D.color) in linear light, (N, 3) (NaN as 0). When every object has the very
        colour it had last frame and none of them can change in place (numbers, names, tuples of numbers), last
        frame's array is used again. When some are new (objects added or removed, or given another colour), the
        rows of the colours it had are reused (those colour objects are kept, so their ids are theirs) and only the
        new ones are looked up as cached_linear_rgb does."""
        last = self._colors
        if last is not None and _same(last[0], colors):
            return last[1]
        if last is None:
            rgb = _numbers([cached_linear_rgb(c) for c in colors]).reshape(-1, 3)
            immutable = all(map(_immutable_color, colors))
        else:
            if last[2] is None:
                last[2] = {id(c): k for k, c in enumerate(last[0])}
            row = last[2].get
            rows = np.fromiter((row(id(c), -1) for c in colors), np.int64, len(colors))
            rgb = last[1][rows]
            new = np.flatnonzero(rows < 0).tolist()
            if new:
                rgb[new] = _numbers([cached_linear_rgb(colors[k]) for k in new]).reshape(-1, 3)
            immutable = all(_immutable_color(colors[k]) for k in new)
        self._colors = [colors, rgb, None] if immutable else None  # (None: {id(colour): row}, made when wanted)
        return rgb

    def _scene_state(self, inst, camera):
        """Everything a render depends on: (values compared by equality, objects compared by identity).

        Meshes, their arrays and textures count as changed when replaced, as in Mesh's own caches;
        edits made inside them are not seen (call invalidate() after those).
        """
        fog = fog_args(self.fog)
        values = [self.width, self.height, self.cell_pixels, self.max_pixels, self.cell_aspect, self.samples,
                  self.edge_samples, self.transparency_layers, self.reflections, self.mirror_bounces,
                  fog[:3] + (fog[3].tobytes(), fog[4]), self.outline, self.lod_bias, self.shadow_size,
                  self.point_shadow_size, self.shadow_softness, self._shadow_fit, self._shading,
                  np.asarray(camera.position, float).tobytes(), np.asarray(camera.target, float).tobytes(),
                  np.asarray(camera.up, float).tobytes(), camera.fov, camera.near, camera.far,
                  self._light_rows.tobytes()]
        if inst is None:
            return values, []
        values += [inst[k].tobytes()
                   for k in ("mesh", "pos", "lin", "rgb", "double", "ident", "emissive", "cast", "alpha",
                             "shine", "specular", "shininess")]
        return values, inst["keys"]

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
        inst = self._instances(objects, camera)
        self._light_rows = light_rows(lights, self.shadows)
        aspect = self.width * self.cell_aspect / max(self.height, 1)
        bg_args, bg_state = background_args(self.background, camera, aspect, self._fb.height)
        state = self._scene_state(inst, camera)
        state = (state[0] + list(bg_state[0]), state[1] + list(bg_state[1]))
        last = self._last
        if last is not None and last[0] == state[0] and _same(last[1], state[1]):
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
        keep = self._reflects(scene["inst"], 0)  # (each sample's colour, for mirrors to replace their share of)
        # The base samples give every pixel its colour, coverage, depth and id; edge samples then add to some.
        base = self._accumulate(scene, SAMPLE_PATTERNS[n], "base", depth=fb.depth.reshape(-1), ids=fb.ids.reshape(-1),
                                frame_samples=0, keep=keep)
        layers = self._layers(scene, SAMPLE_PATTERNS[n], base["sample_depth"]) if (scene["see"] == 1).any() else None
        if self.edge_samples and n > 1:
            pixels = np.flatnonzero(base["more"])
            if len(pixels):
                # A different pattern from the base one: the same one turned a quarter.
                pattern = tuple((1.0 - y, x) for x, y in SAMPLE_PATTERNS[self.edge_samples])
                extra = self._accumulate(scene, pattern, "edge", pixels, frame_samples=n, keep=keep)
                rows = self._buffers.get("edge_rows", (fb.width * fb.height,), np.int64)
                rows.fill(-1)
                rows[pixels] = np.arange(len(pixels))
                base["extra"] = (rows, extra["tris"], extra["sample_rgb"])  # for mirrors to replace them too
        if keep:  # what mirrors show (the base samples drew every pixel)
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
                         layers["cover"], layers["count"], float(self._ortho))
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
        inst_vertex, (vchunk_inst, vchunk_first, vchunk_end, inst_vchunk), (chunk_inst, chunk_run, chunk_first,
                                                                            chunk_end) = _work(pack, inst["mesh"])
        n_chunks = len(chunk_inst)
        whole, cut = buf.get(prefix + "whole", (n_chunks,), np.int64), buf.get(prefix + "cut", (n_chunks,), np.int64)
        off_whole = buf.get(prefix + "off_whole", (n_chunks,), np.int64)
        off_cut = buf.get(prefix + "off_cut", (n_chunks,), np.int64)
        world = buf.get(prefix + "world", (int(inst_vertex[-1]), 10))
        k = transform(pack["vertices"], pack["normals"], pack["faces"], pack["mesh_vertex"], pack["spheres"],
                      inst["mesh"], inst["lin"], inst["pos"], double, inst["flip"], inst_vertex,
                      view_proj, eye, near, clip, vchunk_inst, vchunk_first, vchunk_end, inst_vchunk,
                      chunk_inst, chunk_run, chunk_first, chunk_end, pack["mesh_run"], pack["run_spheres"],
                      pack["run_vertex"],
                      world, buf.get(prefix + "visible", (n_inst,), np.bool_),
                      buf.get(prefix + "seen", (n_chunks,), np.bool_),
                      buf.get(prefix + "needed", (len(vchunk_inst),), np.bool_),
                      whole, cut, off_whole, off_cut)
        return k, world, inst_vertex, (chunk_inst, chunk_first, chunk_end, whole, cut, off_whole, off_cut)

    # ----- sampling and shading ---------------------------------------------------------

    def _accumulate(self, scene, pattern, name, pixels=None, depth=None, ids=None, frame_samples=None, keep=False):
        """Render every sample position in `pattern`, in all pixels or just the given flat pixel indices (in
        increasing order: see raster.rasterize_pixels).

        Returns _resolve()'s per-pixel outputs (rgb, cover, more) and, at each sample, its depth,
        triangle and colour (sample_depth, tris, sample_rgb; the colours only with `keep`, for mirrors, else
        sample_rgb is empty), in buffers named after `name`; the depth
        and object id of each pixel's nearest sample go into `depth` and `ids` if given. With frame_samples, each
        pixel's colour and coverage go into the framebuffer instead of rgb and cover (left empty): averaged with
        the frame_samples samples already there (0: none), as resolve() does.
        """
        fb, buf = self._fb, self._buffers
        n = len(pattern)
        whole = pixels is None
        if whole:
            pixels = buf.arange(fb.width * fb.height)
        m = len(pixels)
        sample_depth = buf.get(name + "_sample_depth", (m, n))
        tris = buf.get(name + "_tris", (m, n), np.int32)
        offsets = np.array(pattern, dtype=float)
        if whole:  # (its samples are emptied by rasterize, band by band)
            rasterize(sample_depth, tris, fb.width, fb.height, scene["xs"], scene["ys"], scene["inv_w"], offsets,
                      pixels, *scene["bands"], scene["see"], scene["attrs"], scene["chain"], scene["lod"],
                      *scene["textures"], True)
        else:  # (a few pixels: the edge pass, or a mirror's)
            sample_depth.fill(0.0)
            tris.fill(-1)
            row_start = np.searchsorted(pixels, buf.arange(fb.height + 1) * fb.width)
            rasterize_pixels(sample_depth, tris, fb.width, fb.height, scene["xs"], scene["ys"], scene["inv_w"],
                             offsets, pixels, row_start, buf.get("next_pixel", (fb.width * fb.height,), np.int64),
                             *scene["bands"], scene["see"], scene["attrs"], scene["chain"], scene["lod"],
                             *scene["textures"])
        if frame_samples is None:
            sums = buf.get(name + "_rgb", (m, 3)), buf.get(name + "_cover", (m,), np.int64)
            frame = np.zeros((0, 3)), np.zeros(0)
        else:
            sums = np.zeros((0, 3)), np.zeros(0, np.int64)
            frame = fb.rgb.reshape(-1, 3), fb.alpha.reshape(-1)
        out = {"rgb": sums[0], "cover": sums[1],
               "more": buf.get(name + "_more", (m,), np.bool_), "sample_depth": sample_depth, "tris": tris,
               "sample_rgb": buf.get(name + "_sample_rgb", (m, n, 3)) if keep else np.zeros((0, n, 3))}
        depth = buf.get("near_depth", (m,)) if depth is None else depth
        ids = buf.get("near_id", (m,), np.int32) if ids is None else ids
        args = (tris, sample_depth, pixels, fb.width, scene["xs"], scene["ys"], scene["inv_w"], scene["attrs"],
                scene["tri_inst"], scene["ident"], scene["emissive"], scene["specular"], scene["shininess"],
                scene["shine"], scene["chain"], scene["lod"], *scene["textures"], scene["lights"], *scene["shadows"],
                scene["pixel_size"], scene["eye"], *scene["sky"], EDGE_CONTRAST, out["sample_rgb"], out["rgb"],
                out["cover"], depth, ids, out["more"], *frame,
                frame_samples or 0)
        cell_w, cell_h = (max(int(v), 1) for v in fb.cell_pixels)
        if whole and self._shading != "pixel" and cell_w * cell_h > 1:  # (blocks of the frame's own cells)
            resolve_cells(*args, cell_w, cell_h, self._shading == "coarse")
        else:
            resolve(*args)
        return out
