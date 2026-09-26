"""Triangle meshes: OBJ loading and textured boxes."""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Mesh:
    vertices: np.ndarray                 # (V, 3) float
    faces: np.ndarray                    # (F, 3) int, counter-clockwise when seen from outside
    uvs: np.ndarray | None = None        # (F, 3, 2) per-corner texture coordinates
    materials: np.ndarray | None = None  # (F,) index into `textures`
    textures: list = field(default_factory=list)  # (H, W) float arrays, 0..1 brightness multipliers

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

    def normalized(self, size=2.0):
        """Copy centred on the origin with its largest extent equal to `size`."""
        lo, hi = self.vertices.min(axis=0), self.vertices.max(axis=0)
        extent = float((hi - lo).max()) or 1.0
        verts = (self.vertices - (lo + hi) / 2.0) * (size / extent)
        return Mesh(verts, self.faces, self.uvs, self.materials, self.textures)


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
