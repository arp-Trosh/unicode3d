# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
"""Triangle meshes: OBJ loading and textured boxes."""
from dataclasses import dataclass, field

import numpy as np

from .color import srgb_to_linear
from .texture import build_mipmaps


def _srgb01(colors):
    """Colours as 0..1 sRGB floats: 0..255 ints are scaled down, floats are taken as they are."""
    c = np.asarray(colors)
    return c / 255.0 if np.issubdtype(c.dtype, np.integer) else c.astype(float)


@dataclass
class Mesh:
    """A triangle mesh. Its colours (per vertex, blended across each face, or per face) multiply the
    colour of the object showing it, as textures do: give the object color=(255, 255, 255) to show
    them as they are. Colours are 0..1 sRGB floats or 0..255 ints, with an optional fourth column of
    opacity (alpha, 0..1 or 0..255: 0 is clear, full is solid), which multiplies the object's."""
    vertices: np.ndarray                 # (V, 3) float
    faces: np.ndarray                    # (F, 3) int, counter-clockwise when seen from outside
    uvs: np.ndarray | None = None        # (F, 3, 2) per-corner texture coordinates
    materials: np.ndarray | None = None  # (F,) index into `textures`
    textures: list = field(default_factory=list)  # (H, W) brightness multipliers, (H, W, 3) colours, 0..1 sRGB,
                                                  # or (H, W, 4) with alpha: holes (cut-outs) or see-through parts
    vertex_colors: np.ndarray | None = None  # (V, 3 or 4) a colour at each vertex, blended smoothly across faces
    face_colors: np.ndarray | None = None    # (F, 3 or 4) one colour for each face (used if vertex_colors is None)

    def vertex_normals(self):
        """Area-weighted average of the normals of the faces around each vertex.

        Faces that share vertices shade smoothly across their seam; faces with
        their own vertices (like the sides of make_box) stay flat.
        """
        cached = self.__dict__.get("_normals")
        if cached is not None and cached[0] is self.vertices:
            return cached[1]
        tri = self.vertices[self.faces]
        face_n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])  # length is twice the area
        normals = np.zeros_like(self.vertices, dtype=float)
        for k in range(3):
            np.add.at(normals, self.faces[:, k], face_n)
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        self._normals = (self.vertices, normals)
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
        return Mesh(verts, self.faces, self.uvs, self.materials, self.textures, self.vertex_colors, self.face_colors)


def load_obj(path):
    """Load vertex positions and faces from a Wavefront OBJ, fan-triangulating polygons."""
    verts, faces = [], []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "v":
                verts.append([float(p) for p in parts[1:4]])
            elif parts[0] == "f":
                idx = []
                for p in parts[1:]:
                    i = int(p.split("/")[0])
                    idx.append(i - 1 if i > 0 else len(verts) + i)
                for k in range(1, len(idx) - 1):
                    faces.append((idx[0], idx[k], idx[k + 1]))
    return Mesh(np.array(verts, dtype=float).reshape(-1, 3), np.array(faces, dtype=int).reshape(-1, 3))


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
