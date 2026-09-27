# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
"""Z-buffered triangle rasterizer (compiled with Numba), and the framebuffer it renders into.

Pixels are finer than terminal cells: each cell covers a small grid of them
(FrameBuffer.cell_pixels), which the glyph set later turns into characters.
"""
import numpy as np
from numba import njit, prange


class FrameBuffer:
    """A rendered image, cell_pixels = (columns, rows) of pixels to each terminal cell.

    rgb: linear-light colour premultiplied by coverage (0 where nothing was drawn).
    alpha: how much of the pixel is covered, 0..1.
    depth: 1/w of the nearest surface (larger is nearer, 0 is empty).
    ids: which object owns the pixel, as its index in the render list plus one (0 is empty).
    """

    def __init__(self, width, height, cell_pixels=(1, 2)):
        self.cell_pixels = tuple(cell_pixels)
        self.resize(width, height)

    def resize(self, width, height):
        self.width, self.height = width, height
        self.rgb = np.zeros((height, width, 3))
        self.alpha = np.zeros((height, width))
        self.depth = np.zeros((height, width))
        self.ids = np.zeros((height, width), dtype=np.int32)

    def clear(self):
        self.rgb.fill(0.0)
        self.alpha.fill(0.0)
        self.depth.fill(0.0)
        self.ids.fill(0)

    def copy(self):
        fb = FrameBuffer.__new__(FrameBuffer)
        fb.cell_pixels, fb.width, fb.height = self.cell_pixels, self.width, self.height
        fb.rgb, fb.alpha, fb.depth, fb.ids = self.rgb.copy(), self.alpha.copy(), self.depth.copy(), self.ids.copy()
        return fb

    @property
    def drawn(self):
        return self.alpha > 0

    def colour(self):
        """Each pixel's own linear colour, undoing the premultiplication by coverage."""
        return self.rgb / np.maximum(self.alpha, 1e-9)[..., None]


ROW_BAND = 8  # rows of pixels each thread rasterizes at a time


@njit(cache=True, parallel=True)
def rasterize(depth, tris, width, height, xs, ys, inv_w, offsets, slots):
    """Depth-test every triangle at every sample position and keep the nearest surface.

    depth, tris: (M, S) nearest 1/w so far (larger is nearer, 0 is empty) and the
    index of the triangle it belongs to (-1 for none), for each of the S sample
    positions `offsets` ((x, y) within a pixel) in each of M pixels; updated in place.
    slots: (width * height,) each pixel's row in depth, or -1 to skip the pixel.
    xs, ys, inv_w: (T, 3) pixel coordinates and 1/w of each triangle corner.

    Where triangles tie, the first one wins. Rows of pixels are split into bands
    that are rasterized in parallel, each band taking the triangles in order.
    """
    n_samples = offsets.shape[0]
    oxmin, oxmax = offsets[:, 0].min(), offsets[:, 0].max()
    oymin, oymax = offsets[:, 1].min(), offsets[:, 1].max()
    n_bands = (height + ROW_BAND - 1) // ROW_BAND
    for band in prange(n_bands):
        band_y0 = band * ROW_BAND
        band_y1 = min(band_y0 + ROW_BAND, height) - 1
        for t in range(xs.shape[0]):
            x0, x1, x2 = xs[t, 0], xs[t, 1], xs[t, 2]
            y0, y1, y2 = ys[t, 0], ys[t, 1], ys[t, 2]
            # Pixels that any sample position could put inside the triangle.
            by0 = max(int(np.floor(min(y0, y1, y2) - oymax)), band_y0)
            by1 = min(int(np.ceil(max(y0, y1, y2) - oymin)), band_y1)
            if by0 > by1:
                continue
            area = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
            if abs(area) <= 1e-9:
                continue
            bx0 = max(int(np.floor(min(x0, x1, x2) - oxmax)), 0)
            bx1 = min(int(np.ceil(max(x0, x1, x2) - oxmin)), width - 1)
            w0, w1, w2 = inv_w[t, 0], inv_w[t, 1], inv_w[t, 2]
            for py in range(by0, by1 + 1):
                for px in range(bx0, bx1 + 1):
                    col = slots[py * width + px]
                    if col < 0:
                        continue
                    for s in range(n_samples):
                        cx, cy = px + offsets[s, 0], py + offsets[s, 1]
                        # Barycentric weights; the slack closes hairline gaps between triangles.
                        b0 = ((x2 - x1) * (cy - y1) - (y2 - y1) * (cx - x1)) / area
                        if b0 < -1e-4:
                            continue
                        b1 = ((x0 - x2) * (cy - y2) - (y0 - y2) * (cx - x2)) / area
                        if b1 < -1e-4:
                            continue
                        b2 = ((x1 - x0) * (cy - y0) - (y1 - y0) * (cx - x0)) / area
                        if b2 < -1e-4:
                            continue
                        z = b0 * w0 + b1 * w1 + b2 * w2
                        if z > depth[col, s]:
                            depth[col, s] = z
                            tris[col, s] = t


@njit(cache=True)
def barycentric(xs, ys, t, cx, cy):
    """Barycentric weights of point (cx, cy) in triangle t, clamped to the triangle.

    A point outside the triangle (a pixel centre just past the edge of a triangle
    that covers some of the pixel's samples) snaps onto it, so attributes are
    never extrapolated beyond the corners' values.
    """
    x0, x1, x2 = xs[t, 0], xs[t, 1], xs[t, 2]
    y0, y1, y2 = ys[t, 0], ys[t, 1], ys[t, 2]
    e12 = (x2 - x1) * (cy - y1) - (y2 - y1) * (cx - x1)
    e20 = (x0 - x2) * (cy - y2) - (y0 - y2) * (cx - x2)
    e01 = (x1 - x0) * (cy - y0) - (y1 - y0) * (cx - x0)
    area = e01 + e12 + e20
    if abs(area) <= 1e-12:
        area = 1e-12
    b0, b1, b2 = max(e12 / area, 0.0), max(e20 / area, 0.0), max(e01 / area, 0.0)
    total = max(b0 + b1 + b2, 1e-12)
    return b0 / total, b1 / total, b2 / total


@njit(cache=True)
def _emit(corners, k, width, height, xs, ys, inv_w, attrs):
    """Store triangle `corners` (3, 12: clip xyzw | world xyz | normal xyz | uv) as screen triangle k."""
    for j in range(3):
        iw = 1.0 / corners[j, 3]
        inv_w[k, j] = iw
        xs[k, j] = (corners[j, 0] * iw + 1.0) * 0.5 * width
        ys[k, j] = (1.0 - corners[j, 1] * iw) * 0.5 * height
        attrs[k, j, :] = corners[j, 4:]


@njit(cache=True)
def project(vertices, vertex_normals, faces, uvs, textured, rotation, scale, position, view_proj, eye, double_sided,
            near, width, height, scratch, xs, ys, inv_w, attrs, src, k):
    """A mesh's faces as screen-space triangles: placed in the world, culled, clipped against the near
    plane w = near, and projected.

    vertices, vertex_normals: (V, 3) in the mesh's own space; the mesh is scaled, rotated by
    `rotation` (3, 3) and moved to `position`. faces: (F, 3). uvs: (F, 3, 2) per-corner
    texture coordinates, read only if `textured`. eye: camera position, for back-face
    culling (skipped if double_sided: back faces are drawn with flipped normals).
    scratch: (V, 10) working space.

    The triangles are written to xs, ys, inv_w (pixel coordinates and 1/w of each
    corner), attrs (world xyz | normal xyz | uv of each corner) and src (each
    triangle's face) from row k on; these need room for 2F rows. Returns the row
    after the last one written. Faces wholly in front of the near plane come
    first, then the pieces of those it cuts; attributes are interpolated
    linearly, which is correct in clip space.
    """
    # Per vertex: world xyz | normal xyz | clip xyzw
    for v in range(vertices.shape[0]):
        for i in range(3):
            scratch[v, i] = (rotation[i, 0] * vertices[v, 0] + rotation[i, 1] * vertices[v, 1]
                             + rotation[i, 2] * vertices[v, 2]) * scale + position[i]
            scratch[v, 3 + i] = (rotation[i, 0] * vertex_normals[v, 0] + rotation[i, 1] * vertex_normals[v, 1]
                                 + rotation[i, 2] * vertex_normals[v, 2])
        for i in range(4):
            scratch[v, 6 + i] = (view_proj[i, 0] * scratch[v, 0] + view_proj[i, 1] * scratch[v, 1]
                                 + view_proj[i, 2] * scratch[v, 2] + view_proj[i, 3])
    corners = np.zeros((3, 12))
    poly = np.empty((4, 12))
    tri = np.empty((3, 12))
    for cut_pass in range(2):
        for f in range(faces.shape[0]):
            ahead = 0
            for j in range(3):
                ahead += scratch[faces[f, j], 9] > near
            if ahead == 0 or (ahead < 3) != (cut_pass == 1):
                continue
            a, b, c = scratch[faces[f, 0]], scratch[faces[f, 1]], scratch[faces[f, 2]]
            e1x, e1y, e1z = b[0] - a[0], b[1] - a[1], b[2] - a[2]
            e2x, e2y, e2z = c[0] - a[0], c[1] - a[1], c[2] - a[2]
            nx, ny, nz = e1y * e2z - e1z * e2y, e1z * e2x - e1x * e2z, e1x * e2y - e1y * e2x
            sign = 1.0
            if nx * (eye[0] - a[0]) + ny * (eye[1] - a[1]) + nz * (eye[2] - a[2]) <= 0:
                if not double_sided:
                    continue
                sign = -1.0
            # Per corner: clip xyzw | world xyz | normal xyz | uv
            for j in range(3):
                v = faces[f, j]
                corners[j, 0:4] = scratch[v, 6:10]
                corners[j, 4:7] = scratch[v, 0:3]
                corners[j, 7:10] = sign * scratch[v, 3:6]
                if textured:
                    corners[j, 10:12] = uvs[f, j]
            if ahead == 3:
                _emit(corners, k, width, height, xs, ys, inv_w, attrs)
                src[k] = f
                k += 1
                continue
            m = 0
            for j in range(3):
                p, q = corners[j], corners[(j + 1) % 3]
                if p[3] > near:
                    poly[m] = p
                    m += 1
                if (p[3] > near) != (q[3] > near):
                    poly[m] = p + (near - p[3]) / (q[3] - p[3]) * (q - p)
                    m += 1
            for j in range(1, m - 1):  # fan-triangulate the clipped polygon
                tri[0], tri[1], tri[2] = poly[0], poly[j], poly[j + 1]
                _emit(tri, k, width, height, xs, ys, inv_w, attrs)
                src[k] = f
                k += 1
    return k
