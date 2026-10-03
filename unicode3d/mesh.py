# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
"""Triangle meshes, and textured boxes. (Loading models from files is in models.py.)"""
from dataclasses import dataclass, field

import numpy as np

from .color import srgb_to_linear
from .texture import build_mipmaps


def _srgb01(colors):
    """Colours as 0..1 sRGB floats: 0..255 ints are scaled down, floats are taken as they are."""
    c = np.asarray(colors)
    return c / 255.0 if np.issubdtype(c.dtype, np.integer) else c.astype(float)


@dataclass(eq=False)  # (equal only to itself: see scene._Placed)
class Mesh:
    """A triangle mesh. Its colours (per vertex, blended across each face, or per face) multiply the
    colour of the object showing it, as textures do: give the object color=(255, 255, 255) to show
    them as they are. Colours are 0..1 sRGB floats or 0..255 ints, with an optional fourth column of
    opacity (alpha, 0..1 or 0..255: 0 is clear, full is solid), which multiplies the object's."""
    vertices: np.ndarray                 # (V, 3) float
    faces: np.ndarray                    # (F, 3) int, counter-clockwise when seen from outside
    uvs: np.ndarray | None = None        # (F, 3, 2) per-corner texture coordinates (u, v): u runs left to right
                                         # across the image, v up it (v = 0 is its last row, v = 1 its row 0),
                                         # as in OBJ files (load_gltf flips glTF's, which run down)
    materials: np.ndarray | None = None  # (F,) index into `textures`
    textures: list = field(default_factory=list)  # (H, W) brightness multipliers, (H, W, 3) colours, 0..1 sRGB,
                                                  # or (H, W, 4) with alpha: holes (cut-outs) or see-through parts
    vertex_colors: np.ndarray | None = None  # (V, 3 or 4) a colour at each vertex, blended smoothly across faces
    face_colors: np.ndarray | None = None    # (F, 3 or 4) one colour for each face (used if vertex_colors is None)
    normals: np.ndarray | None = None        # (V, 3) the surface's normal at each vertex, as a model was made with
                                             # (need not be unit length); None works them out (vertex_normals)

    def vertex_normals(self):
        """The unit normal at each vertex (V, 3): `normals` where given, otherwise the area-weighted average of the
        normals of the faces around the vertex. Rows of `normals` that are zero or not finite are worked out too.

        Worked out, faces that share vertices shade smoothly across their seam, and faces with their own vertices
        (like the sides of make_box) stay flat. Cached, and worked out afresh when the arrays are replaced.
        """
        cached = self.__dict__.get("_normals")
        if (cached is not None and cached[0] is self.vertices and cached[1] is self.normals
                and cached[2] is self.faces):
            return cached[3]
        vertices = np.asarray(self.vertices, dtype=float).reshape(-1, 3)
        faces = np.asarray(self.faces, dtype=np.int64).reshape(-1, 3)
        given = None
        if self.normals is not None:
            given = np.asarray(self.normals, dtype=float).reshape(-1, 3)
            if len(given) != len(vertices):
                raise ValueError(f"Mesh.normals has {len(given)} rows for {len(vertices)} vertices")
            length = np.linalg.norm(given, axis=1)
            usable = np.isfinite(length) & (length > 1e-12)
        if given is not None and usable.all():
            normals = given / length[:, None]
        else:
            tri = vertices[faces]
            face_n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])  # length is twice the area
            normals = np.zeros_like(vertices)
            for k in range(3):
                np.add.at(normals, faces[:, k], face_n)
            normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
            if given is not None:
                normals[usable] = given[usable] / length[usable, None]
        self._normals = (self.vertices, self.normals, self.faces, normals)
        return normals

    def corner_colors(self):
        """Linear rgb and alpha (F, 3, 4) at each corner of each face, from vertex_colors or face_colors (alpha
        1 where they have no fourth column); None if neither.

        Cached, and worked out afresh when the colour or face arrays are replaced.
        """
        colors = self.vertex_colors if self.vertex_colors is not None else self.face_colors
        if colors is None:
            return None
        cached = self.__dict__.get("_corner_colors")
        if cached is not None and cached[0] is colors and cached[1] is self.faces:
            return cached[2]
        c = _srgb01(colors)
        c = c.reshape(-1, c.shape[-1] if c.ndim > 1 and c.shape[-1] == 4 else 3)
        rgba = np.ones((len(c), 4))
        rgba[:, :3] = srgb_to_linear(c[:, :3])
        if c.shape[1] == 4:
            rgba[:, 3] = np.clip(c[:, 3], 0.0, 1.0)  # opacity is a fraction, not a colour: no sRGB curve
        rgba = np.nan_to_num(rgba, nan=0.0, posinf=1.0, neginf=0.0)  # (NaN as black, or clear: never in the picture)
        if self.vertex_colors is not None:
            corners = rgba[self.faces]
        else:
            corners = np.repeat(rgba[:, None, :], 3, axis=1)
        self._corner_colors = (colors, self.faces, corners)
        return corners

    def bounds(self):
        """A sphere around the mesh: (centre (3,), radius), cached like vertex_normals."""
        cached = self.__dict__.get("_bounds")
        if cached is not None and cached[0] is self.vertices:
            return cached[1]
        v = np.asarray(self.vertices, dtype=float).reshape(-1, 3)
        if len(v):
            centre = (v.min(axis=0) + v.max(axis=0)) / 2
            radius = float(np.sqrt(((v - centre) ** 2).sum(axis=1).max()))
        else:
            centre, radius = np.zeros(3), 0.0
        self._bounds = (self.vertices, (centre, radius))
        return centre, radius

    def check(self):
        """Raise ValueError if the arrays don't fit together: faces naming vertices that aren't there, or colours,
        texture coordinates or materials with a row too few or too many. Renderer checks each mesh it draws, as
        the kernels trust these sizes (a mismatch would read memory outside the arrays, or crash)."""
        vertices = np.asarray(self.vertices)
        faces = np.asarray(self.faces)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError(f"Mesh.vertices must be (V, 3), not {vertices.shape}")
        if faces.ndim != 2 or faces.shape[1] != 3 or not (np.issubdtype(faces.dtype, np.integer) or not faces.size):
            raise ValueError(f"Mesh.faces must be (F, 3) whole numbers, not {faces.shape} {faces.dtype}")
        n_vertices, n_faces = len(vertices), len(faces)
        if n_faces and not (0 <= faces.min() and faces.max() < n_vertices):
            raise ValueError(f"Mesh.faces must index its {n_vertices} vertices (0 to {n_vertices - 1}), "
                             f"not {faces.min()} to {faces.max()}")
        by_vertex = self.vertex_colors is not None  # (corner_colors uses these, else face_colors)
        colors = np.asarray(self.vertex_colors if by_vertex else self.face_colors)
        if colors.ndim:
            width = 4 if colors.ndim > 1 and colors.shape[-1] == 4 else 3  # read as corner_colors() reads them
            rows, wanted = colors.size // width, n_vertices if by_vertex else n_faces
            if colors.size % width or (rows < wanted if by_vertex else rows != wanted):
                raise ValueError(f"Mesh.{'vertex_colors' if by_vertex else 'face_colors'} has {colors.size / width:g} "
                                 f"rows for {wanted} {'vertices' if by_vertex else 'faces'}")
        if self.materials is not None and self.textures:
            materials = np.asarray(self.materials)
            if materials.shape != (n_faces,) or not (np.issubdtype(materials.dtype, np.integer) or not n_faces):
                raise ValueError(f"Mesh.materials must be ({n_faces},) whole numbers, one for each face, not "
                                 f"{materials.shape} {materials.dtype}")
            if n_faces and not (0 <= materials.min() and materials.max() < len(self.textures)):
                raise ValueError(f"Mesh.materials must index its {len(self.textures)} textures, not "
                                 f"{materials.min()} to {materials.max()}")
            if np.shape(self.uvs) != (n_faces, 3, 2):
                raise ValueError(f"Mesh.uvs must be ({n_faces}, 3, 2) for a textured mesh, not {np.shape(self.uvs)}")

    def mipmaps(self, material):
        """Mipmap chain of textures[material], built on first use and rebuilt if the texture is replaced."""
        cache = self.__dict__.setdefault("_mipmaps", {})
        tex = self.textures[material]
        cached = cache.get(material)
        if cached is None or cached[0] is not tex:
            cached = cache[material] = (tex, build_mipmaps(tex))
        return cached[1]

    def normalized(self, size=2.0):
        """Copy centred on the origin with its largest extent equal to `size`."""
        lo, hi = self.vertices.min(axis=0), self.vertices.max(axis=0)
        extent = float((hi - lo).max()) or 1.0
        verts = (self.vertices - (lo + hi) / 2.0) * (size / extent)
        return Mesh(verts, self.faces, self.uvs, self.materials, self.textures, self.vertex_colors, self.face_colors,
                    self.normals)


def load_obj(path, **options):
    """A Wavefront OBJ file as one Mesh: models.load_obj, kept here where it used to be."""
    from .models import load_obj as load  # (models.py builds on scene.py, which needs this module first)
    return load(path, **options)


# Outward normal and in-face (u, v) axes of each box face, with u x v = normal.
BOX_FACES = (
    ((1, 0, 0), (0, 0, -1), (0, 1, 0)),    # +X
    ((-1, 0, 0), (0, 0, 1), (0, 1, 0)),    # -X
    ((0, 1, 0), (1, 0, 0), (0, 0, -1)),    # +Y
    ((0, -1, 0), (1, 0, 0), (0, 0, 1)),    # -Y
    ((0, 0, 1), (1, 0, 0), (0, 1, 0)),     # +Z
    ((0, 0, -1), (-1, 0, 0), (0, 1, 0)),   # -Z
)
BOX_FACE_NORMALS = np.array([f[0] for f in BOX_FACES], dtype=float)


def make_box(size=1.0, textures=None):
    """Axis-aligned cube centred on the origin.

    With `textures` (six arrays, in BOX_FACES order) each face is textured and
    face i uses material i; texture row 0 is the top of the face.
    """
    h = size / 2.0
    corners_uv = np.array([(0, 0), (1, 0), (1, 1), (0, 1)], dtype=float)
    verts, faces, uvs = [], [], []
    for n, u, v in BOX_FACES:
        n, u, v = (np.array(a, dtype=float) for a in (n, u, v))
        base = len(verts)
        for cu, cv in corners_uv:
            verts.append((n + (2 * cu - 1) * u + (2 * cv - 1) * v) * h)
        faces += [(base, base + 1, base + 2), (base, base + 2, base + 3)]
        uvs += [corners_uv[[0, 1, 2]], corners_uv[[0, 2, 3]]]
    mesh = Mesh(np.array(verts), np.array(faces, dtype=int))
    if textures is not None:
        if len(textures) != 6:
            raise ValueError("make_box needs exactly six textures")
        mesh.uvs = np.array(uvs)
        mesh.materials = np.repeat(np.arange(6), 2)
        mesh.textures = [np.asarray(t, dtype=float) for t in textures]
    return mesh
