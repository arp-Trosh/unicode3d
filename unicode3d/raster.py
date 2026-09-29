# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
"""Z-buffered triangle rasterizer (compiled with Numba), and the framebuffer it renders into.

Pixels are finer than terminal cells: each cell covers a small grid of them
(FrameBuffer.cell_pixels), which the glyph set later turns into characters.
"""
import numpy as np
from numba import njit, prange

from .texture import sample_alpha, sample_level


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
def _scattered(i, n):
    """Item i of 0..n-1 visited in a scattered order (each once), so that neighbouring items, which often
    have similar amounts of work (bands of rows over one small object), go to different threads."""
    step = 97
    while True:  # a step sharing no factor with n reaches every item
        a, b = step, n
        while b:
            a, b = b, a % b
        if a == 1:
            break
        step += 2
    return np.int64(i) * step % n


@njit(cache=True)
def _band_range(ys, t, height):
    """The bands of rows triangle t can touch (first, last), with a pixel's slack for sample positions;
    first > last if none."""
    top = min(ys[t, 0], ys[t, 1], ys[t, 2]) - 1.0
    bottom = max(ys[t, 0], ys[t, 1], ys[t, 2]) + 1.0
    if not (bottom >= 0.0 and top <= height - 1):  # also rejects NaN
        return 1, 0
    return int(np.floor(max(top, 0.0))) // ROW_BAND, int(np.ceil(min(bottom, height - 1.0))) // ROW_BAND


@njit(cache=True)
def _in_span(xs, t, span, b):
    """Whether triangle t reaches columns span[b, 0]..span[b, 1] (the pixels band b is drawn in), with a
    pixel's slack."""
    return (min(xs[t, 0], xs[t, 1], xs[t, 2]) - 1.0 <= span[b, 1]
            and max(xs[t, 0], xs[t, 1], xs[t, 2]) + 1.0 >= span[b, 0])


@njit(cache=True, parallel=True)
def count_bands(xs, ys, select, want, height, span, band_count):
    """First pass of sorting triangles into bands of rows (see bin_bands): how many triangles each band gets,
    of those whose kind select[t] is among `want` (bit k set for kind k: solid ones, say, or see-through
    ones) and reaching columns span[b, 0]..span[b, 1] of band b (the part drawn: all of it for a whole
    frame, less for a mirror's pixels). Returns, per chunk of triangles and band, how many of the band's
    triangles come from earlier chunks.

    Each band counts for itself, so the counts need no shared totals."""
    n_bands = band_count.shape[0]
    chunk = max((ys.shape[0] + 63) // 64, 1)
    n_chunks = (ys.shape[0] + chunk - 1) // chunk
    # Per chunk of triangles and band: written by that chunk's iteration only, then summed per band.
    per_chunk = np.zeros((n_chunks, n_bands), np.int64)
    for c in prange(n_chunks):
        for t in range(c * chunk, min((c + 1) * chunk, ys.shape[0])):
            if not (want >> select[t]) & 1:
                continue
            b0, b1 = _band_range(ys, t, height)
            for b in range(b0, b1 + 1):
                if _in_span(xs, t, span, b):
                    per_chunk[c, b] += 1
    for b in prange(n_bands):
        total = 0
        for c in range(n_chunks):
            count = per_chunk[c, b]
            per_chunk[c, b] = total
            total += count
        band_count[b] = total
    return per_chunk


@njit(cache=True, parallel=True)
def bin_bands(xs, ys, select, want, height, span, per_chunk, band_start, band_tris):
    """Second pass: the triangles whose kind select[t] is among `want` touching band b, in order, into
    band_tris[band_start[b]:band_start[b + 1]].

    per_chunk is what count_bands() returns; each chunk of triangles writes its own part of each band's
    list, starting after the parts of the chunks before it."""
    n_chunks, n_bands = per_chunk.shape
    chunk = max((ys.shape[0] + 63) // 64, 1)
    for c in prange(n_chunks):
        cursor = np.empty(n_bands, np.int64)
        for b in range(n_bands):
            cursor[b] = band_start[b] + per_chunk[c, b]
        for t in range(c * chunk, min((c + 1) * chunk, ys.shape[0])):
            if not (want >> select[t]) & 1:
                continue
            b0, b1 = _band_range(ys, t, height)
            for b in range(b0, b1 + 1):
                if _in_span(xs, t, span, b):
                    band_tris[cursor[b]] = t
                    cursor[b] += 1


@njit(cache=True, parallel=True)
def rasterize(depth, tris, width, height, xs, ys, inv_w, offsets, slots, band_start, band_tris, see, attrs, chain, lod,
              texels, levels, first):
    """Depth-test every triangle at every sample position and keep the nearest surface.

    depth, tris: (M, S) nearest 1/w so far (larger is nearer, 0 is empty) and the
    index of the triangle it belongs to (-1 for none), for each of the S sample
    positions `offsets` ((x, y) within a pixel) in each of M pixels; updated in place.
    slots: (width * height,) each pixel's row in depth, or -1 to skip the pixel.
    xs, ys, inv_w: (T, 3) pixel coordinates and 1/w of each triangle corner.
    band_tris[band_start[b]:band_start[b + 1]]: the triangles that may touch band b of
    ROW_BAND rows, in order (from count_bands and bin_bands).

    Where triangles tie, the first one wins. Rows of pixels are split into bands
    that are rasterized in parallel (in a scattered order, see _scattered), each band
    taking its triangles in order.

    Cut-outs (see[t] == 2: textures with holes, see project()) cover only the sample positions where
    their texture's alpha (from mip level lod[t] of mipmap chain chain[t], at the uv in attrs) is above
    (s + 0.5) / S for sample s of S: partly clear texels cover some of a pixel's samples (alpha to
    coverage), so the edges of holes are smoothed like the edges of shapes.
    """
    n_samples = offsets.shape[0]
    oxmin, oxmax = offsets[:, 0].min(), offsets[:, 0].max()
    oymin, oymax = offsets[:, 1].min(), offsets[:, 1].max()
    n_bands = (height + ROW_BAND - 1) // ROW_BAND
    for nth in prange(n_bands):
        band = _scattered(nth, n_bands)
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
            holes = see[t] == 2
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
                            if holes:  # perspective-correct uv, then the texture's alpha there
                                u = (b0 * w0 * attrs[t, 0, 6] + b1 * w1 * attrs[t, 1, 6] + b2 * w2 * attrs[t, 2, 6]) / z
                                v = (b0 * w0 * attrs[t, 0, 7] + b1 * w1 * attrs[t, 1, 7] + b2 * w2 * attrs[t, 2, 7]) / z
                                if sample_alpha(texels, levels, first, chain[t], u, v, lod[t]) <= (s + 0.5) / n_samples:
                                    continue
                            depth[col, s] = z
                            tris[col, s] = t


LAYER_MERGE = 0.02  # fragments of one object this close in depth (relative) are one surface: one layer


@njit(cache=True)
def bit_count(mask):
    """How many bits of mask are set."""
    n = 0
    while mask:
        n += mask & 1
        mask >>= 1
    return n


@njit(cache=True, parallel=True)
def rasterize_layers(solid, width, height, xs, ys, inv_w, tri_inst, offsets, band_start, band_tris, layer_depth,
                     layer_tri, layer_cover, layer_count):
    """The see-through surfaces in front of the solid ones: for each pixel, the K nearest (K =
    layer_depth.shape[1]) covering any of its sample positions `offsets` (at most 16) where they are
    nearer than the solid surface (solid (width * height, S): rasterize()'s depth at the same positions),
    nearest first. Per pixel and layer: its depth (1/w: the nearest of the samples it covers), a
    triangle to shade it with, and which samples it covers (bit s for offsets[s]); layer_count (width *
    height,) counts the layers, and where more surfaces cover a pixel the farthest are left out.

    Triangles of one object (tri_inst) whose depths in a pixel are within LAYER_MERGE of each other make
    one layer, covering the samples either covers: neighbours on a mesh, which would otherwise blend
    twice where they meet.

    band_start, band_tris: the see-through triangles touching each band of rows, as for rasterize().
    Bands are rasterized in parallel, each clearing and filling only its own pixels' lists.
    """
    n_samples, k_max = offsets.shape[0], layer_depth.shape[1]
    oxmin, oxmax = offsets[:, 0].min(), offsets[:, 0].max()
    oymin, oymax = offsets[:, 1].min(), offsets[:, 1].max()
    n_bands = (height + ROW_BAND - 1) // ROW_BAND
    for nth in prange(n_bands):
        band = _scattered(nth, n_bands)
        band_y0 = band * ROW_BAND
        band_y1 = min(band_y0 + ROW_BAND, height) - 1
        layer_count[band_y0 * width:(band_y1 + 1) * width] = 0
        for i in range(band_start[band], band_start[band + 1]):
            t = band_tris[i]
            x0, x1, x2 = xs[t, 0], xs[t, 1], xs[t, 2]
            y0, y1, y2 = ys[t, 0], ys[t, 1], ys[t, 2]
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
                    c = py * width + px
                    cover, nearest = 0, 0.0
                    for s in range(n_samples):
                        cx, cy = px + offsets[s, 0], py + offsets[s, 1]
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
                        if z > solid[c, s]:
                            cover |= 1 << s
                            nearest = max(nearest, z)
                    if cover == 0:
                        continue
                    n = layer_count[c]
                    # The same surface as a layer already there: take that one out, to put it back merged.
                    for j in range(n):
                        other = layer_tri[c, j]
                        if (tri_inst[other] == tri_inst[t]
                                and abs(layer_depth[c, j] - nearest) <= LAYER_MERGE * max(layer_depth[c, j], nearest)):
                            if bit_count(layer_cover[c, j]) >= bit_count(cover):
                                t_keep = other
                            else:
                                t_keep = t
                            cover |= layer_cover[c, j]
                            nearest = max(nearest, layer_depth[c, j])
                            for i in range(j, n - 1):
                                layer_depth[c, i], layer_tri[c, i], layer_cover[c, i] = (layer_depth[c, i + 1],
                                                                                         layer_tri[c, i + 1],
                                                                                         layer_cover[c, i + 1])
                            n -= 1
                            break
                    else:
                        t_keep = t
                    if n == k_max and nearest <= layer_depth[c, k_max - 1]:
                        continue  # behind all of a full list
                    # Insert it in order, nearest first, pushing the farthest out of a full list.
                    j = n if n < k_max else k_max - 1
                    while j > 0 and layer_depth[c, j - 1] < nearest:
                        layer_depth[c, j], layer_tri[c, j], layer_cover[c, j] = (layer_depth[c, j - 1],
                                                                                 layer_tri[c, j - 1],
                                                                                 layer_cover[c, j - 1])
                        j -= 1
                    layer_depth[c, j], layer_tri[c, j], layer_cover[c, j] = nearest, t_keep, cover
                    layer_count[c] = min(n + 1, k_max)


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


ATTRS = 12  # per-corner attributes of a screen triangle: world xyz | normal xyz | uv | linear rgb | alpha


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


NO_CLIP = np.array([0.0, 0.0, 0.0, 1.0])  # a clipping plane (a, b, c, d: keeps a x + b y + c z + d > 0) keeping all


@njit(cache=True)
def _plane_distance(world, v, clip):
    """How far in front of clipping plane `clip` vertex v of `world` is (positive: kept)."""
    return clip[0] * world[v, 0] + clip[1] * world[v, 1] + clip[2] * world[v, 2] + clip[3]


@njit(cache=True)
def _near_clip_distances(w0, w1, w2, d0, d1, d2, near, out):
    """Clip a triangle whose corners have clip-space w (w0, w1, w2) and plane distances (d0, d1, d2)
    against the near plane w = near, into out (4,): the plane distances of the polygon left, which it
    returns the length of. project() clips attributes the same way, so they agree on the result."""
    n = 0
    ws, ds = (w0, w1, w2), (d0, d1, d2)
    for j in range(3):
        wp, wq, dp, dq = ws[j], ws[(j + 1) % 3], ds[j], ds[(j + 1) % 3]
        if wp > near:
            out[n] = dp
            n += 1
        if (wp > near) != (wq > near):
            t = (near - wp) / (wq - wp)
            out[n] = dp + t * (dq - dp)
            n += 1
    return n


@njit(cache=True)
def _plane_clipped_length(ds, n):
    """How many corners a polygon of n corners with plane distances ds keeps, clipped against the plane."""
    m = 0
    for j in range(n):
        a, b = ds[j], ds[(j + 1) % n]
        m += (a > 0.0) + ((a > 0.0) != (b > 0.0))
    return m


FACE_CHUNK = 512     # faces projected as one piece of parallel work
VERTEX_CHUNK = 1024  # vertices transformed as one piece


@njit(cache=True, parallel=True)
def transform(vertices, vertex_normals, faces, mesh_vertex, spheres, inst_mesh, inst_quat, inst_scale, inst_pos,
              inst_double, inst_vertex, view_proj, eye, near, clip, vchunk_inst, vchunk_first, vchunk_end, chunk_inst,
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
    
    Faces are also clipped against the plane `clip` (see NO_CLIP): a mirror's, when drawing what it shows.
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
        radius = spheres[m, 3] * abs(scale)
        visible[inst] = (not _outside_view(view_proj, cx, cy, cz, radius, near)
                         and clip[0] * cx + clip[1] * cy + clip[2] * cz + clip[3] > -radius)
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
        ds = np.empty(4)
        if visible[inst]:
            base = inst_vertex[inst] - mesh_vertex[inst_mesh[inst]]  # faces index the packed vertices
            for f in range(chunk_first[c], chunk_end[c]):
                ahead = _face_ahead(world, faces, f, base, near)
                if ahead == 0 or not (inst_double[inst] or _facing(world, faces, f, base, eye)):
                    continue
                v0, v1, v2 = base + faces[f, 0], base + faces[f, 1], base + faces[f, 2]
                d0, d1, d2 = _plane_distance(world, v0, clip), _plane_distance(world, v1, clip), _plane_distance(
                    world, v2, clip)
                if ahead == 3 and d0 > 0.0 and d1 > 0.0 and d2 > 0.0:
                    n_whole += 1
                else:  # clipped by the near plane, then the clipping plane: what is left, as a fan
                    n = _near_clip_distances(world[v0, 9], world[v1, 9], world[v2, 9], d0, d1, d2, near, ds)
                    n_cut += max(_plane_clipped_length(ds, n) - 2, 0)
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
def project(faces, uvs, colors, face_chain, face_texels, face_kind, mesh_vertex, inst_mesh, inst_rgb, inst_alpha,
            inst_double, inst_vertex, eye, near, clip, width, height, lod_bias, world, chunk_inst, chunk_first,
            chunk_end, whole, cut, off_whole, off_cut, xs, ys, inv_w, attrs, tri_inst, chain, lod, see):
    """Every instance's faces as screen-space triangles: culled, clipped against the near plane
    w = near, and projected, from the vertices and plan transform() made.

    The meshes are packed together: mesh m has vertices mesh_vertex[m]:mesh_vertex[m + 1], and
    faces index the packed vertices. Per face: `faces` (F, 3), `uvs` (F, 3, 2), `colors` (F, 3, 4:
    linear rgb and alpha of each corner), `face_chain` (F,: mipmap chain, or -1 if untextured) and
    `face_texels` (F,: texels in the face's texture) and `face_kind` (F,: what its texture's alpha is for,
    see texture.alpha_kind). Instance i shows mesh inst_mesh[i] in colour
    inst_rgb[i] (linear) with opacity inst_alpha[i]; back faces (seen from `eye`) are culled unless
    inst_double[i]: then they are drawn with flipped normals.

    The triangles go to xs, ys, inv_w (pixel coordinates and 1/w of each corner), attrs (ATTRS
    attributes of each corner), tri_inst (the instance), chain (the face's mipmap chain), lod (its
    mip level, lod_bias included) and see (its kind: 1 if it can be seen through, as a corner's alpha
    is below 1 or its texture is partly see-through; else 2 if its texture has holes (cut-outs);
    else 0, solid), each face chunk's at the rows transform() gave it. Attributes of clipped faces are
    interpolated linearly, which is correct in clip space.
    """
    for c in prange(chunk_inst.shape[0]):
        if whole[c] + cut[c] == 0:
            continue
        inst = chunk_inst[c]
        base = inst_vertex[inst] - mesh_vertex[inst_mesh[inst]]
        corners = np.zeros((3, 4 + ATTRS))
        poly = np.empty((4, 4 + ATTRS))
        kept = np.empty((5, 4 + ATTRS))
        tri = np.empty((3, 4 + ATTRS))
        ds = np.empty(4)
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
            # Per corner: clip xyzw | world xyz | normal xyz | uv | rgb | alpha
            for j in range(3):
                v = base + faces[f, j]
                corners[j, 0:4] = world[v, 6:10]
                corners[j, 4:7] = world[v, 0:3]
                corners[j, 7:10] = sign * world[v, 3:6]
                corners[j, 10:12] = uvs[f, j]
                for i in range(3):
                    corners[j, 12 + i] = colors[f, j, i] * inst_rgb[inst, i]
                corners[j, 15] = colors[f, j, 3] * inst_alpha[inst]
            v0, v1, v2 = base + faces[f, 0], base + faces[f, 1], base + faces[f, 2]
            d0, d1, d2 = _plane_distance(world, v0, clip), _plane_distance(world, v1, clip), _plane_distance(
                world, v2, clip)
            if ahead == 3 and d0 > 0.0 and d1 > 0.0 and d2 > 0.0:
                _emit(corners, k_whole, width, height, xs, ys, inv_w, attrs)
                chain[k_whole] = f
                k_whole += 1
                continue
            # Clipped by the near plane, then by the clipping plane (with the distances worked out just as
            # transform() did, so that the triangles left are the ones it counted).
            n_poly = 0
            for j in range(3):
                p, q = corners[j], corners[(j + 1) % 3]
                if p[3] > near:
                    poly[n_poly] = p
                    n_poly += 1
                if (p[3] > near) != (q[3] > near):
                    poly[n_poly] = p + (near - p[3]) / (q[3] - p[3]) * (q - p)
                    n_poly += 1
            _near_clip_distances(corners[0, 3], corners[1, 3], corners[2, 3], d0, d1, d2, near, ds)
            n_kept = 0
            for j in range(n_poly):
                a, b = ds[j], ds[(j + 1) % n_poly]
                if a > 0.0:
                    kept[n_kept] = poly[j]
                    n_kept += 1
                if (a > 0.0) != (b > 0.0):
                    kept[n_kept] = poly[j] + a / (a - b) * (poly[(j + 1) % n_poly] - poly[j])
                    n_kept += 1
            for j in range(1, n_kept - 1):  # fan-triangulate the clipped polygon
                tri[0], tri[1], tri[2] = kept[0], kept[j], kept[j + 1]
                _emit(tri, k_cut, width, height, xs, ys, inv_w, attrs)
                chain[k_cut] = f
                k_cut += 1
        # Each triangle's instance, mipmap chain and mip level: log2 of how many texels span a pixel across it.
        for part in range(2):
            t0, t1 = (off_whole[c], k_whole) if part == 0 else (off_cut[c], k_cut)
            for t in range(t0, t1):
                f = chain[t]
                tri_inst[t] = inst
                if attrs[t, 0, 11] < 1.0 or attrs[t, 1, 11] < 1.0 or attrs[t, 2, 11] < 1.0 or face_kind[f] == 2:
                    see[t] = 1
                else:
                    see[t] = 2 if face_kind[f] == 1 else 0
                chain[t] = face_chain[f]
                if face_chain[f] < 0:
                    lod[t] = 0.0
                    continue
                px_area = abs((xs[t, 1] - xs[t, 0]) * (ys[t, 2] - ys[t, 0]) - (ys[t, 1] - ys[t, 0]) * (xs[t, 2] - xs[t, 0]))
                d1u, d1v = attrs[t, 1, 6] - attrs[t, 0, 6], attrs[t, 1, 7] - attrs[t, 0, 7]
                d2u, d2v = attrs[t, 2, 6] - attrs[t, 0, 6], attrs[t, 2, 7] - attrs[t, 0, 7]
                ratio = abs(d1u * d2v - d1v * d2u) * face_texels[f] / max(px_area, 1e-9)
                lod[t] = 0.5 * np.log2(max(ratio, 1e-9)) + lod_bias


SURFACE = 7  # per corner of a shadow map's triangle: linear rgb | alpha | u / w, v / w, 1 / w (for its texture)


@njit(cache=True)
def _emit_depth(corners, k, width, height, perspective, xs, ys, depth, surf):
    """Store triangle `corners` (3, 10: clip xyzw | rgb | alpha | uv) as screen triangle k of a shadow map
    (see project_depth)."""
    for j in range(3):
        iw = 1.0 / corners[j, 3]
        xs[k, j] = (corners[j, 0] * iw + 1.0) * 0.5 * width
        ys[k, j] = (1.0 - corners[j, 1] * iw) * 0.5 * height
        depth[k, j] = iw if perspective else (1.0 - corners[j, 2] * iw) * 0.5
        surf[k, j, 0:4] = corners[j, 4:8]
        surf[k, j, 4], surf[k, j, 5], surf[k, j, 6] = corners[j, 8] * iw, corners[j, 9] * iw, iw


@njit(cache=True)
def _corner_surface(colors, uvs, inst_rgb, inst_alpha, inst, f, j, out):
    """Corner j of face f of instance inst as a shadow map needs it: linear rgb into out[4:7], alpha into
    out[7] and uv into out[8:10]."""
    for i in range(3):
        out[4 + i] = min(max(colors[f, j, i] * inst_rgb[inst, i], 0.0), 1.0)
    out[7] = colors[f, j, 3] * inst_alpha[inst]
    out[8], out[9] = uvs[f, j, 0], uvs[f, j, 1]


@njit(cache=True)
def _face_kind(colors, inst_alpha, face_kind, inst, f):
    """Kind of face f of instance inst, as project() gives triangles: 1 see-through, 2 cut-out, 0 solid."""
    if face_kind[f] == 2 or inst_alpha[inst] < 1.0:
        return 1
    for j in range(3):
        if colors[f, j, 3] < 1.0:
            return 1
    return 2 if face_kind[f] == 1 else 0


@njit(cache=True)
def _finish_shadow_triangle(xs, ys, surf, k, f, kind, face_chain, face_texels, see, chain, lod):
    """Triangle k of a shadow map, from face f: its kind, mipmap chain, and mip level for one texel of the
    map (log2 of how many texture texels span it)."""
    see[k], chain[k] = kind, face_chain[f]
    if face_chain[f] < 0:
        lod[k] = 0.0
        return
    px_area = abs((xs[k, 1] - xs[k, 0]) * (ys[k, 2] - ys[k, 0]) - (ys[k, 1] - ys[k, 0]) * (xs[k, 2] - xs[k, 0]))
    u0, v0 = surf[k, 0, 4] / surf[k, 0, 6], surf[k, 0, 5] / surf[k, 0, 6]
    d1u, d1v = surf[k, 1, 4] / surf[k, 1, 6] - u0, surf[k, 1, 5] / surf[k, 1, 6] - v0
    d2u, d2v = surf[k, 2, 4] / surf[k, 2, 6] - u0, surf[k, 2, 5] / surf[k, 2, 6] - v0
    lod[k] = 0.5 * np.log2(max(abs(d1u * d2v - d1v * d2u) * face_texels[f] / max(px_area, 1e-9), 1e-9))


@njit(cache=True)
def _clip_near(corners, near, poly):
    """The part of triangle `corners` (3, N: clip xyzw | attributes) in front of the near plane w = near,
    into poly (4, N); returns its number of corners (0, 3 or 4)."""
    n_poly = 0
    for j in range(3):
        p, q = corners[j], corners[(j + 1) % 3]
        if p[3] > near:
            poly[n_poly] = p
            n_poly += 1
        if (p[3] > near) != (q[3] > near):
            poly[n_poly] = p + (near - p[3]) / (q[3] - p[3]) * (q - p)
            n_poly += 1
    return n_poly


@njit(cache=True, parallel=True)
def project_depth(faces, uvs, colors, face_chain, face_texels, face_kind, mesh_vertex, inst_mesh, inst_rgb, inst_alpha,
                  inst_vertex, world, width, height, near, perspective, chunk_inst, chunk_first, chunk_end, whole, cut,
                  off_whole, off_cut, xs, ys, depth, surf, see, chain, lod):
    """Every face as a screen triangle of a shadow map: the light's view of the scene, from the vertices
    and plan transform() made with every instance double-sided (no culling). Faces crossing the near
    plane w = near are clipped, as in project().

    depth (T, 3) is each corner's depth, larger nearer the light, as rasterize_depth() expects: 1/w for a
    perspective view, like rasterize(); for an orthographic one (w = 1), (1 - z) / 2 in normalized
    device coordinates. Both are linear across the screen, so they are interpolated exactly. surf (T, 3,
    SURFACE) is what each corner's surface does to light (colour and alpha, from `colors` (F, 3, 4) and
    the instance's, and its uv, for its texture); see (T,) is each triangle's kind as in project() (0
    solid, 1 see-through, 2 cut-out), chain its mipmap chain (-1 if untextured) and lod its mip level.
    """
    for c in prange(chunk_inst.shape[0]):
        if whole[c] + cut[c] == 0:
            continue
        inst = chunk_inst[c]
        base = inst_vertex[inst] - mesh_vertex[inst_mesh[inst]]
        corners = np.empty((3, 10))
        poly = np.empty((4, 10))
        tri = np.empty((3, 10))
        k_whole, k_cut = off_whole[c], off_cut[c]
        for f in range(chunk_first[c], chunk_end[c]):
            ahead = 0
            for j in range(3):
                corners[j, 0:4] = world[base + faces[f, j], 6:10]
                ahead += corners[j, 3] > near
            if ahead == 0:
                continue
            for j in range(3):
                _corner_surface(colors, uvs, inst_rgb, inst_alpha, inst, f, j, corners[j])
            kind = _face_kind(colors, inst_alpha, face_kind, inst, f)
            if ahead == 3:
                _emit_depth(corners, k_whole, width, height, perspective, xs, ys, depth, surf)
                _finish_shadow_triangle(xs, ys, surf, k_whole, f, kind, face_chain, face_texels, see, chain, lod)
                k_whole += 1
                continue
            n_poly = _clip_near(corners, near, poly)
            for j in range(1, n_poly - 1):
                tri[0], tri[1], tri[2] = poly[0], poly[j], poly[j + 1]
                _emit_depth(tri, k_cut, width, height, perspective, xs, ys, depth, surf)
                _finish_shadow_triangle(xs, ys, surf, k_cut, f, kind, face_chain, face_texels, see, chain, lod)
                k_cut += 1


@njit(cache=True)
def _cube_face_corners(world, faces, f, base, mat, near, corners):
    """Face f's corners in the clip space of one face of a cube map (`mat`), into corners[:, 0:4], and how
    many of them are in front of its near plane: 0 if none, or if all lie beyond one of its sides."""
    ahead = right = left = top = bottom = 0
    for j in range(3):
        v = world[base + faces[f, j]]
        for i in range(4):
            corners[j, i] = mat[i, 0] * v[0] + mat[i, 1] * v[1] + mat[i, 2] * v[2] + mat[i, 3]
        x, y, w = corners[j, 0], corners[j, 1], corners[j, 3]
        ahead += w > near
        right += x > w
        left += x < -w
        top += y > w
        bottom += y < -w
    return 0 if right == 3 or left == 3 or top == 3 or bottom == 3 else ahead


@njit(cache=True, parallel=True)
def count_cube(faces, mesh_vertex, inst_mesh, inst_vertex, world, mats, near, chunk_inst, chunk_first, chunk_end,
               counts):
    """First pass of projecting faces into the six faces of a cube shadow map (see project_cube): how many
    triangles each chunk of faces leaves in each cube face, into counts (chunks, 6)."""
    for i in prange(chunk_inst.shape[0] * 6):
        c, side = i // 6, i % 6
        inst = chunk_inst[c]
        base = inst_vertex[inst] - mesh_vertex[inst_mesh[inst]]
        corners = np.empty((3, 10))
        n = 0
        for f in range(chunk_first[c], chunk_end[c]):
            ahead = _cube_face_corners(world, faces, f, base, mats[side], near, corners)
            n += 1 if ahead == 3 or ahead == 1 else 2 if ahead == 2 else 0  # two in front leave a quad
        counts[c, side] = n


@njit(cache=True, parallel=True)
def project_cube(faces, uvs, colors, face_chain, face_texels, face_kind, mesh_vertex, inst_mesh, inst_rgb, inst_alpha,
                 inst_vertex, world, mats, near, size, chunk_inst, chunk_first, chunk_end, offsets, xs, ys, depth, surf,
                 see, chain, lod, tri_side):
    """The faces as screen triangles in a cube shadow map: its six faces (+x, -x, +y, -y, +z, -z, each
    size x size texels) stacked one above the other in one tall map, a triangle for each cube face it
    may show in, clipped against that face's near plane and carrying 1/w as its depth (see
    project_depth, as are surf, see, chain and lod). World positions come from transform(); chunk c
    writes its triangles in cube face `side` from row offsets[c, side] (count_cube()'s counts, summed
    up), and tri_side[k] is triangle k's cube face, which rasterize_depth() keeps it inside.
    """
    for i in prange(chunk_inst.shape[0] * 6):
        c, side = i // 6, i % 6
        inst = chunk_inst[c]
        base = inst_vertex[inst] - mesh_vertex[inst_mesh[inst]]
        corners = np.empty((3, 10))
        poly = np.empty((4, 10))
        tri = np.empty((3, 10))
        k = offsets[c, side]
        for f in range(chunk_first[c], chunk_end[c]):
            ahead = _cube_face_corners(world, faces, f, base, mats[side], near, corners)
            if ahead == 0:
                continue
            for j in range(3):
                _corner_surface(colors, uvs, inst_rgb, inst_alpha, inst, f, j, corners[j])
            kind = _face_kind(colors, inst_alpha, face_kind, inst, f)
            n_poly = 3
            if ahead == 3:
                poly[:3] = corners
            else:
                n_poly = _clip_near(corners, near, poly)
            for j in range(1, n_poly - 1):
                tri[0], tri[1], tri[2] = poly[0], poly[j], poly[j + 1]
                _emit_depth(tri, k, size, size, True, xs, ys, depth, surf)
                _finish_shadow_triangle(xs, ys, surf, k, f, kind, face_chain, face_texels, see, chain, lod)
                for m in range(3):
                    ys[k, m] += side * size
                tri_side[k] = side
                k += 1


@njit(cache=True)
def _surface_uv(surf, t, b0, b1, b2):
    """Perspective-correct uv at barycentric weights b of shadow triangle t."""
    w = max(b0 * surf[t, 0, 6] + b1 * surf[t, 1, 6] + b2 * surf[t, 2, 6], 1e-30)
    return ((b0 * surf[t, 0, 4] + b1 * surf[t, 1, 4] + b2 * surf[t, 2, 4]) / w,
            (b0 * surf[t, 0, 5] + b1 * surf[t, 1, 5] + b2 * surf[t, 2, 5]) / w)


@njit(cache=True, parallel=True)
def rasterize_depth(depth, xs, ys, dep, tri_side, side_rows, band_start, band_tris, see, surf, chain, lod, texels,
                    levels, first):
    """A shadow map: the largest `dep` (nearest the light) of any triangle covering each texel's centre,
    into depth (height, width), which it clears first; 0 where there is none. Triangle t is kept to rows
    tri_side[t] * side_rows to (tri_side[t] + 1) * side_rows - 1 (one face of a cube map, see
    project_cube; for any other map, tri_side is 0 and side_rows the height). Cut-outs (see[t] == 2)
    cover only texels where their texture's alpha is at least a half, so light shines through their holes.

    Like rasterize() with one sample per texel, and the same slack at triangle edges so that no light
    leaks between neighbours, but it only keeps depth, so each row of a triangle is filled as one span
    rather than testing every texel around it. Bands of rows are drawn in parallel (band_start and
    band_tris as in rasterize()), each clearing and drawing only its own rows.
    """
    height, width = depth.shape
    n_bands = (height + ROW_BAND - 1) // ROW_BAND
    for nth in prange(n_bands):
        band = _scattered(nth, n_bands)
        band_y0 = band * ROW_BAND
        band_y1 = min(band_y0 + ROW_BAND, height) - 1
        depth[band_y0:band_y1 + 1, :] = 0.0
        for i in range(band_start[band], band_start[band + 1]):
            t = band_tris[i]
            x0, x1, x2 = xs[t, 0], xs[t, 1], xs[t, 2]
            y0, y1, y2 = ys[t, 0], ys[t, 1], ys[t, 2]
            top = tri_side[t] * side_rows
            by0 = max(int(np.floor(min(y0, y1, y2) - 0.5)), band_y0, top)
            by1 = min(int(np.ceil(max(y0, y1, y2) - 0.5)), band_y1, top + side_rows - 1)
            area = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
            if by0 > by1 or abs(area) <= 1e-9:
                continue
            left = max(np.floor(min(x0, x1, x2) - 0.5), 0.0)
            right = min(np.ceil(max(x0, x1, x2) - 0.5), width - 1.0)
            # Barycentric weight k at (x, y) is ax[k] * x + (bx[k] * y + c[k]); depth is linear in them.
            ax0, bx0, c0 = -(y2 - y1) / area, (x2 - x1) / area, ((y2 - y1) * x1 - (x2 - x1) * y1) / area
            ax1, bx1, c1 = -(y0 - y2) / area, (x0 - x2) / area, ((y0 - y2) * x2 - (x0 - x2) * y2) / area
            ax2, bx2, c2 = -(y1 - y0) / area, (x1 - x0) / area, ((y1 - y0) * x0 - (x1 - x0) * y0) / area
            d0, d1, d2 = dep[t, 0], dep[t, 1], dep[t, 2]
            dx = ax0 * d0 + ax1 * d1 + ax2 * d2
            holes = see[t] == 2
            for py in range(by0, by1 + 1):
                cy = py + 0.5
                e0, e1, e2 = bx0 * cy + c0, bx1 * cy + c1, bx2 * cy + c2
                # The centres x inside all three edges (weights >= -1e-4), within the triangle's bounds.
                lo, hi = left + 0.5, right + 0.5
                for a, e in ((ax0, e0), (ax1, e1), (ax2, e2)):
                    if a > 0.0:
                        lo = max(lo, (-1e-4 - e) / a)
                    elif a < 0.0:
                        hi = min(hi, (-1e-4 - e) / a)
                    elif e < -1e-4:
                        hi = -1.0
                if hi < lo:
                    continue
                dc = e0 * d0 + e1 * d1 + e2 * d2
                row = depth[py]
                for px in range(int(np.ceil(lo - 0.5)), int(np.floor(hi - 0.5)) + 1):
                    cx = px + 0.5
                    z = dx * cx + dc
                    if z > row[px]:
                        if holes:
                            u, v = _surface_uv(surf, t, ax0 * cx + e0, ax1 * cx + e1, ax2 * cx + e2)
                            if sample_alpha(texels, levels, first, chain[t], u, v, lod[t]) < 0.5:
                                continue
                        row[px] = z


@njit(cache=True, parallel=True)
def rasterize_tint(trans, offset, width, height, xs, ys, dep, tri_side, side_rows, band_start, band_tris, surf, chain,
                   lod, texels, levels, first):
    """What see-through triangles do to the light at each texel of a shadow map (height x width texels,
    from trans[:, offset]): trans[0] is the depth of the nearest of them (0 if none; cleared first), and
    trans[1:4] the light they let through, rgb, multiplied together over all of them (only meaningful
    where trans[0] > 0, so the rest is not cleared). Triangles, bands, cube faces and spans as in
    rasterize_depth().

    A surface of colour c and alpha a (its corners' surf, times its texture's where it has one, from
    the mip level nearest lod[t]) lets through (1 - a) * (1 + a * c) of the light: clear glass almost
    all of it, coloured glass tinted by its colour, nearly solid glass little.
    """
    n_bands = (height + ROW_BAND - 1) // ROW_BAND
    for nth in prange(n_bands):
        band = _scattered(nth, n_bands)
        band_y0 = band * ROW_BAND
        band_y1 = min(band_y0 + ROW_BAND, height) - 1
        trans[0, offset + band_y0 * width:offset + (band_y1 + 1) * width] = 0.0
        for i in range(band_start[band], band_start[band + 1]):
            t = band_tris[i]
            x0, x1, x2 = xs[t, 0], xs[t, 1], xs[t, 2]
            y0, y1, y2 = ys[t, 0], ys[t, 1], ys[t, 2]
            top = tri_side[t] * side_rows
            by0 = max(int(np.floor(min(y0, y1, y2) - 0.5)), band_y0, top)
            by1 = min(int(np.ceil(max(y0, y1, y2) - 0.5)), band_y1, top + side_rows - 1)
            area = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
            if by0 > by1 or abs(area) <= 1e-9:
                continue
            left = max(np.floor(min(x0, x1, x2) - 0.5), 0.0)
            right = min(np.ceil(max(x0, x1, x2) - 0.5), width - 1.0)
            ax0, bx0, c0 = -(y2 - y1) / area, (x2 - x1) / area, ((y2 - y1) * x1 - (x2 - x1) * y1) / area
            ax1, bx1, c1 = -(y0 - y2) / area, (x0 - x2) / area, ((y0 - y2) * x2 - (x0 - x2) * y2) / area
            ax2, bx2, c2 = -(y1 - y0) / area, (x1 - x0) / area, ((y1 - y0) * x0 - (x1 - x0) * y0) / area
            for py in range(by0, by1 + 1):
                cy = py + 0.5
                e0, e1, e2 = bx0 * cy + c0, bx1 * cy + c1, bx2 * cy + c2
                lo, hi = left + 0.5, right + 0.5
                for a, e in ((ax0, e0), (ax1, e1), (ax2, e2)):
                    if a > 0.0:
                        lo = max(lo, (-1e-4 - e) / a)
                    elif a < 0.0:
                        hi = min(hi, (-1e-4 - e) / a)
                    elif e < -1e-4:
                        hi = -1.0
                if hi < lo:
                    continue
                for px in range(int(np.ceil(lo - 0.5)), int(np.floor(hi - 0.5)) + 1):
                    cx = px + 0.5
                    b0, b1, b2 = ax0 * cx + e0, ax1 * cx + e1, ax2 * cx + e2
                    r = b0 * surf[t, 0, 0] + b1 * surf[t, 1, 0] + b2 * surf[t, 2, 0]
                    g = b0 * surf[t, 0, 1] + b1 * surf[t, 1, 1] + b2 * surf[t, 2, 1]
                    b = b0 * surf[t, 0, 2] + b1 * surf[t, 1, 2] + b2 * surf[t, 2, 2]
                    a = b0 * surf[t, 0, 3] + b1 * surf[t, 1, 3] + b2 * surf[t, 2, 3]
                    if chain[t] >= 0:
                        u, v = _surface_uv(surf, t, b0, b1, b2)
                        tr, tg, tb, ta = sample_level(texels, levels, first, chain[t], u, v, lod[t])
                        r, g, b, a = r * tr, g * tg, b * tb, a * ta
                    a = min(max(a, 0.0), 1.0)
                    k = offset + py * width + px
                    fresh = trans[0, k] == 0.0
                    trans[0, k] = max(trans[0, k], b0 * dep[t, 0] + b1 * dep[t, 1] + b2 * dep[t, 2])
                    for ch, colour in ((1, r), (2, g), (3, b)):
                        through = (1.0 - a) * (1.0 + a * min(max(colour, 0.0), 1.0))
                        trans[ch, k] = through if fresh else trans[ch, k] * through
