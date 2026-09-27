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


ROW_BAND = 4  # rows of pixels each thread rasterizes at a time


@njit(cache=True)
def _band_range(ys, t, height):
    """The bands of rows triangle t can touch (first, last), with a pixel's slack for sample positions;
    first > last if none."""
    top = min(ys[t, 0], ys[t, 1], ys[t, 2]) - 1.0
    bottom = max(ys[t, 0], ys[t, 1], ys[t, 2]) + 1.0
    if not (bottom >= 0.0 and top <= height - 1):  # also rejects NaN
        return 1, 0
    return int(np.floor(max(top, 0.0))) // ROW_BAND, int(np.ceil(min(bottom, height - 1.0))) // ROW_BAND


@njit(cache=True, parallel=True)
def count_bands(ys, height, band_count):
    """First pass of sorting triangles into bands of rows (see bin_bands): how many triangles each band gets.

    Each band counts for itself, so the counts need no shared totals."""
    n_bands = band_count.shape[0]
    chunk = max((ys.shape[0] + 63) // 64, 1)
    n_chunks = (ys.shape[0] + chunk - 1) // chunk
    # Per chunk of triangles and band: written by that chunk's iteration only, then summed per band.
    per_chunk = np.zeros((n_chunks, n_bands), np.int64)
    for c in prange(n_chunks):
        for t in range(c * chunk, min((c + 1) * chunk, ys.shape[0])):
            b0, b1 = _band_range(ys, t, height)
            for b in range(b0, b1 + 1):
                per_chunk[c, b] += 1
    for b in prange(n_bands):
        total = 0
        for c in range(n_chunks):
            total += per_chunk[c, b]
        band_count[b] = total
    return per_chunk


@njit(cache=True, parallel=True)
def bin_bands(ys, height, per_chunk, band_start, band_tris):
    """Second pass: the triangles touching band b, in order, into band_tris[band_start[b]:band_start[b + 1]].

    per_chunk is count_bands()'s per-chunk counts; each chunk of triangles writes its own part of each
    band's list, starting after the parts of the chunks before it."""
    n_chunks, n_bands = per_chunk.shape
    chunk = max((ys.shape[0] + 63) // 64, 1)
    for c in prange(n_chunks):
        cursor = np.empty(n_bands, np.int64)
        for b in range(n_bands):
            cursor[b] = band_start[b]
            for c2 in range(c):
                cursor[b] += per_chunk[c2, b]
        for t in range(c * chunk, min((c + 1) * chunk, ys.shape[0])):
            b0, b1 = _band_range(ys, t, height)
            for b in range(b0, b1 + 1):
                band_tris[cursor[b]] = t
                cursor[b] += 1


@njit(cache=True, parallel=True)
def rasterize(depth, tris, width, height, xs, ys, inv_w, offsets, slots, band_start, band_tris):
    """Depth-test every triangle at every sample position and keep the nearest surface.

    depth, tris: (M, S) nearest 1/w so far (larger is nearer, 0 is empty) and the
    index of the triangle it belongs to (-1 for none), for each of the S sample
    positions `offsets` ((x, y) within a pixel) in each of M pixels; updated in place.
    slots: (width * height,) each pixel's row in depth, or -1 to skip the pixel.
    xs, ys, inv_w: (T, 3) pixel coordinates and 1/w of each triangle corner.
    band_tris[band_start[b]:band_start[b + 1]]: the triangles that may touch band b of
    ROW_BAND rows, in order (from count_bands and bin_bands).

    Where triangles tie, the first one wins. Rows of pixels are split into bands
    that are rasterized in parallel, each band taking its triangles in order.
    """
    n_samples = offsets.shape[0]
    oxmin, oxmax = offsets[:, 0].min(), offsets[:, 0].max()
    oymin, oymax = offsets[:, 1].min(), offsets[:, 1].max()
    n_bands = (height + ROW_BAND - 1) // ROW_BAND
    for band in prange(n_bands):
        band_y0 = band * ROW_BAND
        band_y1 = min(band_y0 + ROW_BAND, height) - 1
        for i in range(band_start[band], band_start[band + 1]):
            t = band_tris[i]
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


ATTRS = 11  # per-corner attributes of a screen triangle: world xyz | normal xyz | uv | linear rgb


@njit(cache=True)
def _emit(corners, k, width, height, xs, ys, inv_w, attrs):
    """Store triangle `corners` (3, 4 + ATTRS: clip xyzw | attributes) as screen triangle k."""
    for j in range(3):
        iw = 1.0 / corners[j, 3]
        inv_w[k, j] = iw
        xs[k, j] = (corners[j, 0] * iw + 1.0) * 0.5 * width
        ys[k, j] = (1.0 - corners[j, 1] * iw) * 0.5 * height
        attrs[k, j, :] = corners[j, 4:]


@njit(cache=True)
def _outside_view(view_proj, cx, cy, cz, radius, near):
    """Whether a sphere lies wholly outside the view: beyond the left, right, top or bottom edge, or
    behind the near plane w = near."""
    for p in range(5):
        if p < 4:  # w + x, w - x, w + y, w - y: rows 3 +- 0 and 3 +- 1 of view_proj
            row, sign = p // 2, 1.0 - 2.0 * (p % 2)
            a = view_proj[3, 0] + sign * view_proj[row, 0]
            b = view_proj[3, 1] + sign * view_proj[row, 1]
            c = view_proj[3, 2] + sign * view_proj[row, 2]
            d = view_proj[3, 3] + sign * view_proj[row, 3]
        else:
            a, b, c, d = view_proj[3, 0], view_proj[3, 1], view_proj[3, 2], view_proj[3, 3] - near
        length = np.sqrt(a * a + b * b + c * c)
        if length > 0 and (a * cx + b * cy + c * cz + d) / length < -radius * (1.0 + 1e-9) - 1e-9:
            return True
    return False


@njit(cache=True)
def _rotation(q, rot):
    """Rotation matrix of unit quaternion q (w, x, y, z), into rot (3, 3)."""
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    rot[0, 0], rot[0, 1], rot[0, 2] = 1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)
    rot[1, 0], rot[1, 1], rot[1, 2] = 2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)
    rot[2, 0], rot[2, 1], rot[2, 2] = 2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)


@njit(cache=True)
def _face_ahead(world, faces, f, v0, near):
    """How many corners of face f are in front of the near plane."""
    ahead = 0
    for j in range(3):
        ahead += world[v0 + faces[f, j], 9] > near
    return ahead


@njit(cache=True)
def _facing(world, faces, f, v0, eye):
    """Whether face f (world positions in `world`) faces the eye."""
    a, b, c = world[v0 + faces[f, 0]], world[v0 + faces[f, 1]], world[v0 + faces[f, 2]]
    e1x, e1y, e1z = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    e2x, e2y, e2z = c[0] - a[0], c[1] - a[1], c[2] - a[2]
    nx, ny, nz = e1y * e2z - e1z * e2y, e1z * e2x - e1x * e2z, e1x * e2y - e1y * e2x
    return nx * (eye[0] - a[0]) + ny * (eye[1] - a[1]) + nz * (eye[2] - a[2]) > 0


FACE_CHUNK = 512     # faces projected as one piece of parallel work
VERTEX_CHUNK = 1024  # vertices transformed as one piece


@njit(cache=True, parallel=True)
def transform(vertices, vertex_normals, faces, mesh_vertex, spheres, inst_mesh, inst_quat, inst_scale, inst_pos,
              inst_double, inst_vertex, view_proj, eye, near, vchunk_inst, vchunk_first, vchunk_end, chunk_inst,
              chunk_first, chunk_end, world, visible, whole, cut, off_whole, off_cut):
    """First pass of projecting instances (see project()): vertices in the world, and where each piece
    of work will write its triangles. Returns the number of triangles.

    visible[i]: whether instance i is at least partly in view (its bounding sphere, spheres[mesh]).
    Vertex chunk j transforms vertices vchunk_first[j]:vchunk_end[j] (counted within the mesh) of
    instance vchunk_inst[j] into world, from row inst_vertex[i] for instance i: world xyz | normal
    xyz | clip xyzw. Face chunk c covers faces chunk_first[c]:chunk_end[c] of instance chunk_inst[c]
    (chunks in instance order): whole[c] of them will be drawn whole and `cut` pieces will be left
    by clipping, written from rows off_whole[c] and off_cut[c], so that each instance's whole faces
    come before the pieces of its clipped ones.
    """
    for inst in prange(inst_mesh.shape[0]):
        m = inst_mesh[inst]
        scale = inst_scale[inst]
        rot = np.empty((3, 3))
        _rotation(inst_quat[inst], rot)
        position = inst_pos[inst]
        sx, sy, sz = spheres[m, 0] * scale, spheres[m, 1] * scale, spheres[m, 2] * scale
        cx = rot[0, 0] * sx + rot[0, 1] * sy + rot[0, 2] * sz + position[0]
        cy = rot[1, 0] * sx + rot[1, 1] * sy + rot[1, 2] * sz + position[1]
        cz = rot[2, 0] * sx + rot[2, 1] * sy + rot[2, 2] * sz + position[2]
        visible[inst] = not _outside_view(view_proj, cx, cy, cz, spheres[m, 3] * abs(scale), near)
    for j in prange(vchunk_inst.shape[0]):
        inst = vchunk_inst[j]
        if not visible[inst]:
            continue
        m = inst_mesh[inst]
        scale = inst_scale[inst]
        rot = np.empty((3, 3))
        _rotation(inst_quat[inst], rot)
        position = inst_pos[inst]
        v0, w0 = mesh_vertex[m], inst_vertex[inst]
        for v in range(vchunk_first[j], vchunk_end[j]):
            p, n, out = vertices[v0 + v], vertex_normals[v0 + v], world[w0 + v]
            for i in range(3):
                out[i] = (rot[i, 0] * p[0] + rot[i, 1] * p[1] + rot[i, 2] * p[2]) * scale + position[i]
                out[3 + i] = rot[i, 0] * n[0] + rot[i, 1] * n[1] + rot[i, 2] * n[2]
            for i in range(4):
                out[6 + i] = view_proj[i, 0] * out[0] + view_proj[i, 1] * out[1] + view_proj[i, 2] * out[2] + view_proj[i, 3]
    for c in prange(chunk_inst.shape[0]):
        inst = chunk_inst[c]
        n_whole = n_cut = 0
        if visible[inst]:
            base = inst_vertex[inst] - mesh_vertex[inst_mesh[inst]]  # faces index the packed vertices
            for f in range(chunk_first[c], chunk_end[c]):
                ahead = _face_ahead(world, faces, f, base, near)
                if ahead == 0 or not (inst_double[inst] or _facing(world, faces, f, base, eye)):
                    continue
                if ahead == 3:
                    n_whole += 1
                else:
                    n_cut += 1 if ahead == 1 else 2  # two corners in front leave a quad
        whole[c], cut[c] = n_whole, n_cut
    k = c = 0
    while c < chunk_inst.shape[0]:
        e = c
        while e < chunk_inst.shape[0] and chunk_inst[e] == chunk_inst[c]:
            e += 1
        for x in range(c, e):
            off_whole[x] = k
            k += whole[x]
        for x in range(c, e):
            off_cut[x] = k
            k += cut[x]
        c = e
    return k


@njit(cache=True, parallel=True)
def project(faces, uvs, colors, face_chain, face_texels, mesh_vertex, inst_mesh, inst_rgb, inst_double, inst_vertex,
            eye, near, width, height, lod_bias, world, chunk_inst, chunk_first, chunk_end, whole, cut, off_whole,
            off_cut, xs, ys, inv_w, attrs, tri_inst, chain, lod):
    """Every instance's faces as screen-space triangles: culled, clipped against the near plane
    w = near, and projected, from the vertices and plan transform() made.

    The meshes are packed together: mesh m has vertices mesh_vertex[m]:mesh_vertex[m + 1], and
    faces index the packed vertices. Per face: `faces` (F, 3), `uvs` (F, 3, 2), `colors` (F, 3, 3:
    linear rgb of each corner), `face_chain` (F,: mipmap chain, or -1 if untextured) and
    `face_texels` (F,: texels in the face's texture). Instance i shows mesh inst_mesh[i] in colour
    inst_rgb[i] (linear); back faces (seen from `eye`) are culled unless inst_double[i]: then they
    are drawn with flipped normals.

    The triangles go to xs, ys, inv_w (pixel coordinates and 1/w of each corner), attrs (ATTRS
    attributes of each corner), tri_inst (the instance), chain (the face's mipmap chain) and lod (its
    mip level, lod_bias included), each face chunk's at the rows transform() gave it. Attributes of
    clipped faces are interpolated linearly, which is correct in clip space.
    """
    for c in prange(chunk_inst.shape[0]):
        if whole[c] + cut[c] == 0:
            continue
        inst = chunk_inst[c]
        base = inst_vertex[inst] - mesh_vertex[inst_mesh[inst]]
        corners = np.zeros((3, 4 + ATTRS))
        poly = np.empty((4, 4 + ATTRS))
        tri = np.empty((3, 4 + ATTRS))
        k_whole, k_cut = off_whole[c], off_cut[c]
        for f in range(chunk_first[c], chunk_end[c]):
            ahead = _face_ahead(world, faces, f, base, near)
            if ahead == 0:
                continue
            sign = 1.0
            if not _facing(world, faces, f, base, eye):
                if not inst_double[inst]:
                    continue
                sign = -1.0
            # Per corner: clip xyzw | world xyz | normal xyz | uv | rgb
            for j in range(3):
                v = base + faces[f, j]
                corners[j, 0:4] = world[v, 6:10]
                corners[j, 4:7] = world[v, 0:3]
                corners[j, 7:10] = sign * world[v, 3:6]
                corners[j, 10:12] = uvs[f, j]
                for i in range(3):
                    corners[j, 12 + i] = colors[f, j, i] * inst_rgb[inst, i]
            if ahead == 3:
                _emit(corners, k_whole, width, height, xs, ys, inv_w, attrs)
                chain[k_whole] = f
                k_whole += 1
                continue
            n_poly = 0
            for j in range(3):
                p, q = corners[j], corners[(j + 1) % 3]
                if p[3] > near:
                    poly[n_poly] = p
                    n_poly += 1
                if (p[3] > near) != (q[3] > near):
                    poly[n_poly] = p + (near - p[3]) / (q[3] - p[3]) * (q - p)
                    n_poly += 1
            for j in range(1, n_poly - 1):  # fan-triangulate the clipped polygon
                tri[0], tri[1], tri[2] = poly[0], poly[j], poly[j + 1]
                _emit(tri, k_cut, width, height, xs, ys, inv_w, attrs)
                chain[k_cut] = f
                k_cut += 1
        # Each triangle's instance, mipmap chain and mip level: log2 of how many texels span a pixel across it.
        for part in range(2):
            t0, t1 = (off_whole[c], k_whole) if part == 0 else (off_cut[c], k_cut)
            for t in range(t0, t1):
                f = chain[t]
                tri_inst[t] = inst
                chain[t] = face_chain[f]
                if face_chain[f] < 0:
                    lod[t] = 0.0
                    continue
                px_area = abs((xs[t, 1] - xs[t, 0]) * (ys[t, 2] - ys[t, 0]) - (ys[t, 1] - ys[t, 0]) * (xs[t, 2] - xs[t, 0]))
                d1u, d1v = attrs[t, 1, 6] - attrs[t, 0, 6], attrs[t, 1, 7] - attrs[t, 0, 7]
                d2u, d2v = attrs[t, 2, 6] - attrs[t, 0, 6], attrs[t, 2, 7] - attrs[t, 0, 7]
                ratio = abs(d1u * d2v - d1v * d2u) * face_texels[f] / max(px_area, 1e-9)
                lod[t] = 0.5 * np.log2(max(ratio, 1e-9)) + lod_bias
