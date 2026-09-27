# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
"""Mesh builders: extruded bitmaps and text, ellipsoids, boxes, puffed-up 2D shapes."""
import numpy as np

from .mesh import Mesh, make_box


def merge_meshes(meshes):
    """One mesh holding all the given meshes' triangles (untextured)."""
    verts, faces, base = [], [], 0
    for m in meshes:
        verts.append(m.vertices)
        faces.append(m.faces + base)
        base += len(m.vertices)
    return Mesh(np.concatenate(verts), np.concatenate(faces))


def block_mesh(center, size, rotation=None):
    """Box with per-axis `size`, optionally rotated (3x3 matrix) about its centre."""
    verts = make_box(1.0).vertices * size
    if rotation is not None:
        verts = verts @ np.asarray(rotation).T
    return Mesh(verts + center, make_box(1.0).faces)


def blob_mesh(radii, center=(0.0, 0.0, 0.0), rings=8, segments=12, bump=None):
    """UV ellipsoid; bump(unit_directions) -> per-vertex radius multipliers."""
    lat = np.linspace(0, np.pi, rings + 1)[1:-1]
    lon = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    dirs = np.c_[(np.sin(la) * np.cos(lo)).ravel(), np.cos(la).ravel(), (np.sin(la) * np.sin(lo)).ravel()]
    dirs = np.r_[[(0.0, 1.0, 0.0)], dirs, [(0.0, -1.0, 0.0)]]
    r = bump(dirs)[:, None] if bump else 1.0
    verts = dirs * r * radii + center
    ring = lambda i: 1 + i * segments  # first vertex of ring i
    faces = []
    for j in range(segments):
        k = (j + 1) % segments
        faces.append((0, ring(0) + k, ring(0) + j))
        last = ring(rings - 2)
        faces.append((len(verts) - 1, last + j, last + k))
        for i in range(rings - 2):
            a, b, c, d = ring(i) + j, ring(i) + k, ring(i + 1) + j, ring(i + 1) + k
            faces += [(a, b, d), (a, d, c)]
    return Mesh(verts, np.array(faces, dtype=int))


def pillow_mesh(shape, res=48, span=1.1, thickness=0.22, rim=4):
    """A 2D shape (x, y -> inside mask, over -span..span) puffed up into a closed cushion.

    The front is flat in the middle and rounds off over the last `rim` grid steps to the
    outline, like a sticker or an embossed badge; the back mirrors it. Texture
    coordinates are the (x, y) position, so a texture of the same shape lines up.
    """
    c = np.linspace(-span, span, res + 1)
    x, y = np.meshgrid(c, -c)  # row 0 is the top
    inside = shape(x, y)
    depth = inside.astype(float)  # grid steps to the outline, by repeated erosion
    core = inside.copy()
    for _ in range(rim):
        core = core & np.roll(core, 1, 0) & np.roll(core, -1, 0) & np.roll(core, 1, 1) & np.roll(core, -1, 1)
        depth += core
    z = thickness * np.sqrt(np.clip((depth - 1) / rim, 0, 1))
    n = (res + 1) ** 2
    front = np.c_[x.ravel(), y.ravel(), z.ravel()]
    back = front * [1, 1, -1]
    idx = np.arange(n).reshape(res + 1, res + 1)
    a, b, c_, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]  # top-left, top-right, bottom-left, bottom-right
    solid = (inside[:-1, :-1] & inside[:-1, 1:] & inside[1:, :-1] & inside[1:, 1:]).ravel()
    quads = np.stack([a.ravel(), b.ravel(), c_.ravel(), d.ravel()], axis=1)[solid]
    tl, tr, bl, br = quads.T
    faces_front = np.r_[np.c_[tl, bl, br], np.c_[tl, br, tr]]  # counter-clockwise seen from +Z
    faces_back = np.c_[faces_front[:, 0], faces_front[:, 2], faces_front[:, 1]] + n
    faces = np.r_[faces_front, faces_back]
    used, faces = np.unique(faces, return_inverse=True)  # drop the grid points outside the shape
    mesh = Mesh(np.r_[front, back][used], faces.reshape(-1, 3))
    uv = (mesh.vertices[:, :2] / span + 1) / 2
    mesh.uvs = uv[mesh.faces]
    mesh.materials = np.zeros(len(mesh.faces), dtype=int)
    return mesh


def bitmap_mesh(cells, depth=1.0):
    """A 2D boolean grid (row 0 at the top) extruded along Z, one unit per cell, centred on the origin.

    Faces are merged into runs to keep the triangle count (and render time) down.
    """
    cells = np.asarray(cells, bool)
    h, w = cells.shape
    padded = np.pad(cells, 1)
    verts, faces = [], []
    zf, zb = depth / 2, -depth / 2

    def quad(corners, normal):
        corners = [np.array(c, dtype=float) for c in corners]
        if np.dot(np.cross(corners[1] - corners[0], corners[2] - corners[0]), normal) < 0:
            corners.reverse()
        base = len(verts)
        verts.extend(corners)
        faces.extend([(base, base + 1, base + 2), (base, base + 2, base + 3)])

    def runs(mask):
        """(start, end) index pairs of consecutive True values."""
        edges = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
        return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))

    for r in range(h):
        y0, y1 = -(r + 1), -r
        for c0, c1 in runs(cells[r]):
            quad([(c0, y0, zf), (c1, y0, zf), (c1, y1, zf), (c0, y1, zf)], (0, 0, 1))
            quad([(c0, y0, zb), (c1, y0, zb), (c1, y1, zb), (c0, y1, zb)], (0, 0, -1))
        for c0, c1 in runs(cells[r] & ~padded[r, 1:-1]):      # nothing above
            quad([(c0, y1, zb), (c1, y1, zb), (c1, y1, zf), (c0, y1, zf)], (0, 1, 0))
        for c0, c1 in runs(cells[r] & ~padded[r + 2, 1:-1]):  # nothing below
            quad([(c0, y0, zb), (c1, y0, zb), (c1, y0, zf), (c0, y0, zf)], (0, -1, 0))
    for c in range(w):
        for r0, r1 in runs(cells[:, c] & ~padded[1:-1, c]):        # nothing to the left
            quad([(c, -r0, zb), (c, -r1, zb), (c, -r1, zf), (c, -r0, zf)], (-1, 0, 0))
        for r0, r1 in runs(cells[:, c] & ~padded[1:-1, c + 2]):    # nothing to the right
            quad([(c + 1, -r0, zb), (c + 1, -r1, zb), (c + 1, -r1, zf), (c + 1, -r0, zf)], (1, 0, 0))

    verts = np.array(verts) - [w / 2, -h / 2, 0]
    return Mesh(verts, np.array(faces, dtype=int))


def text_bitmap(text, font, spacing=1):
    """Boolean grid of `text` in a bitmap font: {character: rows of "#" (ink) and "." (blank)}.

    Every glyph must have the same number of rows; `spacing` blank columns go between letters.
    """
    height = len(next(iter(font.values())))
    rows = ["" for _ in range(height)]
    for i, ch in enumerate(text):
        glyph = font[ch]
        for r in range(height):
            rows[r] += ("." * spacing if i else "") + glyph[r]
    return np.array([[c == "#" for c in row] for row in rows])


def text_mesh(text, font, depth=1.6, spacing=1):
    """Extruded voxel text, one unit per font pixel, centred on the origin: (mesh, width in units)."""
    cells = text_bitmap(text, font, spacing)
    return bitmap_mesh(cells, depth), cells.shape[1]
