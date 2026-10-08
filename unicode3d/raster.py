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

# Sample positions within a pixel (x, y from its top-left corner): the standard
# multisample patterns, where no two samples share a row or column, so near-vertical
# and near-horizontal edges get as many coverage steps as there are samples.
SAMPLE_PATTERNS = {
    1: ((0.5, 0.5),),
    4: ((0.375, 0.125), (0.875, 0.375), (0.125, 0.625), (0.625, 0.875)),
    8: ((0.5625, 0.3125), (0.4375, 0.6875), (0.8125, 0.5625), (0.3125, 0.1875),
        (0.1875, 0.8125), (0.0625, 0.4375), (0.6875, 0.9375), (0.9375, 0.0625)),
    16: ((0.5625, 0.5625), (0.4375, 0.3125), (0.3125, 0.625), (0.75, 0.4375),
         (0.1875, 0.375), (0.625, 0.8125), (0.8125, 0.6875), (0.6875, 0.1875),
         (0.375, 0.875), (0.5, 0.0625), (0.25, 0.125), (0.125, 0.75),
         (0.0, 0.5), (0.9375, 0.25), (0.875, 0.9375), (0.0625, 0.0)),
}
SOLID, CLEAR, CUT = 1, 2, 4  # kinds of triangle (project's `see`: 0, 1, 2) as bits, for Renderer._bins


@njit(cache=True, error_model="numpy", parallel=True)
def upscale(rgb, alpha, depth, ids, out_rgb, out_alpha, out_depth, out_ids):
    """Stretch a framebuffer's arrays (rgb, alpha, depth, ids) over bigger ones (out_*): colour and coverage
    bilinearly, depth and ids from the nearest pixel (so that each pixel keeps a surface that is there). Each
    row of the output is written by its own iteration."""
    h, w = alpha.shape
    big_h, big_w = out_alpha.shape
    for y in prange(big_h):
        sy = (y + 0.5) * h / big_h - 0.5
        y0 = min(max(int(np.floor(sy)), 0), h - 1)
        y1, fy = min(y0 + 1, h - 1), min(max(sy - y0, 0.0), 1.0)
        near_y = min(int((y + 0.5) * h / big_h), h - 1)
        for x in range(big_w):
            sx = (x + 0.5) * w / big_w - 0.5
            x0 = min(max(int(np.floor(sx)), 0), w - 1)
            x1, fx = min(x0 + 1, w - 1), min(max(sx - x0, 0.0), 1.0)
            w00, w01, w10, w11 = (1.0 - fy) * (1.0 - fx), (1.0 - fy) * fx, fy * (1.0 - fx), fy * fx
            for k in range(3):
                out_rgb[y, x, k] = (w00 * rgb[y0, x0, k] + w01 * rgb[y0, x1, k] + w10 * rgb[y1, x0, k]
                                    + w11 * rgb[y1, x1, k])
            out_alpha[y, x] = w00 * alpha[y0, x0] + w01 * alpha[y0, x1] + w10 * alpha[y1, x0] + w11 * alpha[y1, x1]
            near_x = min(int((x + 0.5) * w / big_w), w - 1)
            out_depth[y, x], out_ids[y, x] = depth[near_y, near_x], ids[near_y, near_x]


@njit(cache=True, error_model="numpy")
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


@njit(cache=True, error_model="numpy")
def _band_range(ys, t, height):
    """The bands of rows triangle t can touch (first, last), with a pixel's slack for sample positions;
    first > last if none."""
    top = min(ys[t, 0], ys[t, 1], ys[t, 2]) - 1.0
    bottom = max(ys[t, 0], ys[t, 1], ys[t, 2]) + 1.0
    if not (bottom >= 0.0 and top <= height - 1):  # also rejects NaN
        return 1, 0
    return int(np.floor(max(top, 0.0))) // ROW_BAND, int(np.ceil(min(bottom, height - 1.0))) // ROW_BAND


@njit(cache=True, error_model="numpy")
def pixel_range(lo, hi, first, last):
    """The whole pixels (a, b) from floor(lo) to ceil(hi), kept within first..last; empty (a > b) where there are
    none, or lo or hi is NaN. Coordinates are only converted to ints once they are in range: converting NaN, an
    infinity or anything beyond int64 gives an undefined int, and loops over it write outside the arrays."""
    if not lo <= hi:
        return 1, 0
    a = first if lo <= first else (last + 1 if lo >= last + 1 else int(np.floor(lo)))
    b = last if hi >= last else (first - 1 if hi <= first - 1 else int(np.ceil(hi)))
    return a, b


@njit(cache=True, error_model="numpy")
def drawable(area):
    """Whether a triangle whose signed area (in pixels, or texels) is `area` is worth drawing: not a sliver, and
    with finite corners (a NaN or infinite corner makes the area NaN or infinite)."""
    return 1e-9 < abs(area) < np.inf


@njit(cache=True, error_model="numpy")
def _in_span(xs, t, span, b):
    """Whether triangle t reaches columns span[b, 0]..span[b, 1] (the pixels band b is drawn in), with a
    pixel's slack."""
    return (min(xs[t, 0], xs[t, 1], xs[t, 2]) - 1.0 <= span[b, 1]
            and max(xs[t, 0], xs[t, 1], xs[t, 2]) + 1.0 >= span[b, 0])


@njit(cache=True, error_model="numpy", parallel=True)
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


@njit(cache=True, error_model="numpy", parallel=True)
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


SPAN_SLACK = 1e-6  # pixels a row's span is widened by at least
SPAN_ROUNDING = 1e-12  # and this much per pixel of the coordinates' size: about 1,000 times the rounding error


@njit(cache=True, error_model="numpy")
def _row_span(lo, hi, xa, ya, ex, ey, per_area, cy_lo, cy_hi, width):
    """(lo, hi) narrowed to the sample x positions where the barycentric weight of edge (xa, ya) + t (ex, ey) can
    pass rasterize()'s test (weight >= -1e-4) at some sample height cy_lo..cy_hi: the weight is
    (ex (cy - ya) - ey (cx - xa)) / area, a straight line in cx.

    The span is a pre-filter: every sample inside it still gets the exact tests, so it only has to be sure not to
    leave out a sample they would pass. Worked out in floating point a different way from them, it can be off by
    rounding, so it is widened by SPAN_SLACK plus SPAN_ROUNDING times the sizes involved (the coordinates, and how
    far the edge's height makes cx move: unbounded for an edge within a hair of horizontal, which then narrows
    nothing). Comparisons that are false for NaN leave (lo, hi) as they are."""
    d = -ey * per_area  # how the weight changes with cx
    if d == 0.0:  # a horizontal edge: it bounds rows, which the bounding box does already
        return lo, hi
    g = ex * per_area  # with cy
    a, b = g * (cy_lo - ya), g * (cy_hi - ya)
    c = (a if a > b else b) + d * -xa  # the weight at cx = 0, at the row's best sample height
    edge = (-1e-4 - c) / d  # where the weight crosses the threshold
    slack = SPAN_SLACK + SPAN_ROUNDING * ((abs(a) + abs(b)) / abs(d) + abs(xa) + abs(edge) + width + 2.0)
    if d > 0.0:  # the weight grows with cx: samples left of the edge fail
        if edge - slack > lo:
            lo = edge - slack
    elif edge + slack < hi:
        hi = edge + slack
    return lo, hi


OCCLUSION_TILE = 8  # columns of pixels in each tile (by ROW_BAND rows) whose farthest depth rasterize() keeps
OCCLUSION_STALE = 2  # triangles drawn over a tile before rasterize() looks for its farthest depth again
OCCLUSION_WAIT = 4  # and more after looking found it farther than a triangle


@njit(cache=True, error_model="numpy")
def nearest_bound(x0, x1, x2, y0, y1, y2, w0, w1, w2, area):
    """A depth (1/w) that no sample rasterize() writes for this triangle can be nearer than, given its corners'
    pixel coordinates and 1/w and its signed area as rasterize() works them out; NaN where none is sure.

    A sample's depth is b0 w0 + b1 w1 + b2 w2, from weights that rasterize() lets go down to -1e-4 each (two at
    most, as they add up to 1) and that it works out with rounding: their sum is 1 to within a few dozen rounding
    steps of the size of the products in them (the box's width times its height, with a pixel or two each way for
    the samples around it) relative to the area. So the depth is at most the largest of the corners' times
    1 + 2e-4 + that error, and a little more for rounding the sum. A triangle with a negative or NaN corner, or so
    thin that the error could be large (1e-3), gets NaN."""
    if not (w0 >= 0.0 and w1 >= 0.0 and w2 >= 0.0):
        return np.nan
    bw = max(x0, x1, x2) - min(x0, x1, x2)
    bh = max(y0, y1, y2) - min(y0, y1, y2)
    rounding = 2e-14 * (bw + 2.0) * (bh + 2.0) / abs(area) + 1e-12
    if not rounding < 1e-3:
        return np.nan
    return max(w0, w1, w2) * (1.0 + 2.5e-4 + rounding)


@njit(cache=True, error_model="numpy", inline="always")
def _behind(z, k, far, drawn, depth, slots, width, band_y0, band_y1):
    """Whether no sample of tile k of a band (rows band_y0..band_y1, OCCLUSION_TILE columns from
    k * OCCLUSION_TILE) is farther than z, so that a triangle no nearer than z anywhere changes none of them.

    far[k] is at most the farthest of them (NaN samples left out: nothing replaces them): samples only come
    nearer, so it stays so as triangles are drawn. drawn[k] counts the triangles drawn over the tile since far[k]
    was worked out; from OCCLUSION_STALE of them it is worked out again, where it would answer no."""
    if z <= far[k]:
        return True
    if drawn[k] < OCCLUSION_STALE or z != z:  # (NaN: no bound)
        return False
    m = np.inf
    for py in range(band_y0, band_y1 + 1):
        for px in range(k * OCCLUSION_TILE, min((k + 1) * OCCLUSION_TILE, width)):
            col = slots[py * width + px]
            if col >= 0:
                for s in range(depth.shape[1]):
                    d = depth[col, s]
                    if d < z:  # (no: far[k] stays as it was, and is not looked for again for a while)
                        drawn[k] = -OCCLUSION_WAIT
                        return False
                    if d < m:
                        m = d
    far[k], drawn[k] = m, 0
    return True


@njit(cache=True, error_model="numpy", parallel=True)
def rasterize(depth, tris, width, height, xs, ys, inv_w, offsets, slots, band_start, band_tris, see, attrs, chain, lod,
              texels, levels, first, clear):
    """Depth-test every triangle at every sample position and keep the nearest surface.

    depth, tris: (M, S) nearest 1/w so far (larger is nearer, 0 is empty) and the
    index of the triangle it belongs to (-1 for none), for each of the S sample
    positions `offsets` ((x, y) within a pixel) in each of M pixels; updated in place.
    slots: (width * height,) each pixel's row in depth, or -1 to skip the pixel.
    xs, ys, inv_w: (T, 3) pixel coordinates and 1/w of each triangle corner.
    band_tris[band_start[b]:band_start[b + 1]]: the triangles that may touch band b of
    ROW_BAND rows, in order (from count_bands and bin_bands).

    Each row of a triangle is walked only over the pixels its edges allow (_row_span), not its whole bounding box;
    the samples there get the same tests, so the result is the same as testing every sample in the box.

    Where triangles tie, the first one wins. Rows of pixels are split into bands
    that are rasterized in parallel (in a scattered order, see _scattered), each band
    taking its triangles in order.

    Each band also keeps, for tiles of OCCLUSION_TILE of its columns, a depth that none of the tile's samples is
    farther than (see _behind), and leaves out of a triangle's box the tiles at either end where that is no farther
    than the triangle can be near anywhere (nearest_bound): a triangle behind walls already drawn is skipped, or
    walked over less. Its samples there would all have failed the depth test, ties included, so the result is the
    same.

    Cut-outs (see[t] == 2: textures with holes, see project()) cover only the sample positions where
    their texture's alpha (from mipmap chain chain[t], at the uv in attrs, at the mip level texture_lod gives
    there plus lod[t]) is above (s + 0.5) / S for sample s of S: partly clear texels cover some of a pixel's
    samples (alpha to coverage), so the edges of holes are smoothed like the edges of shapes.

    With `clear`, each band first empties its own pixels' rows of depth and tris (0 and -1): in parallel, and just
    before they are drawn into, rather than in one pass over all of them beforehand (most of a big frame's memory
    traffic).
    """
    n_samples = offsets.shape[0]
    oxmin, oxmax = offsets[:, 0].min(), offsets[:, 0].max()
    oymin, oymax = offsets[:, 1].min(), offsets[:, 1].max()
    n_bands = (height + ROW_BAND - 1) // ROW_BAND
    for nth in prange(n_bands):
        band = _scattered(nth, n_bands)
        band_y0 = band * ROW_BAND
        band_y1 = min(band_y0 + ROW_BAND, height) - 1
        # Per tile of the band's pixels (OCCLUSION_TILE columns): a depth no sample there is farther than, and the
        # triangles drawn over it since (see _behind).
        n_tiles = (width + OCCLUSION_TILE - 1) // OCCLUSION_TILE
        far, drawn = np.full(n_tiles, -np.inf), np.full(n_tiles, OCCLUSION_STALE, np.int64)
        if clear:
            for c in range(band_y0 * width, (band_y1 + 1) * width):
                col = slots[c]
                if col >= 0:
                    for s in range(n_samples):
                        depth[col, s] = 0.0
                        tris[col, s] = -1
        for i in range(band_start[band], band_start[band + 1]):
            t = band_tris[i]
            x0, x1, x2 = xs[t, 0], xs[t, 1], xs[t, 2]
            y0, y1, y2 = ys[t, 0], ys[t, 1], ys[t, 2]
            area = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
            if not drawable(area):
                continue
            # Pixels that any sample position could put inside the triangle.
            by0, by1 = pixel_range(min(y0, y1, y2) - oymax, max(y0, y1, y2) - oymin, band_y0, band_y1)
            if by0 > by1:
                continue
            bx0, bx1 = pixel_range(min(x0, x1, x2) - oxmax, max(x0, x1, x2) - oxmin, 0, width - 1)
            w0, w1, w2 = inv_w[t, 0], inv_w[t, 1], inv_w[t, 2]
            # Less the tiles at either end of the box that are nowhere farther than the triangle can be near: its
            # samples there would all fail the depth test (ties keep the first).
            near = nearest_bound(x0, x1, x2, y0, y1, y2, w0, w1, w2, area)
            k0, k1 = bx0 // OCCLUSION_TILE, bx1 // OCCLUSION_TILE
            while k0 <= k1 and _behind(near, k0, far, drawn, depth, slots, width, band_y0, band_y1):
                k0 += 1
            while k1 >= k0 and _behind(near, k1, far, drawn, depth, slots, width, band_y0, band_y1):
                k1 -= 1
            if k0 > k1:
                continue
            bx0, bx1 = max(bx0, k0 * OCCLUSION_TILE), min(bx1, (k1 + 1) * OCCLUSION_TILE - 1)
            holes = see[t] == 2
            per_area = 1.0 / area  # (multiplying by it is several times quicker than dividing by area)
            for py in range(by0, by1 + 1):
                # Only the pixels of this row whose samples might pass all three edge tests (see _row_span).
                lo, hi = bx0 + oxmin, bx1 + oxmax
                lo, hi = _row_span(lo, hi, x1, y1, x2 - x1, y2 - y1, per_area, py + oymin, py + oymax, width)
                lo, hi = _row_span(lo, hi, x2, y2, x0 - x2, y0 - y2, per_area, py + oymin, py + oymax, width)
                lo, hi = _row_span(lo, hi, x0, y0, x1 - x0, y1 - y0, per_area, py + oymin, py + oymax, width)
                ax, bx = pixel_range(lo - oxmax, hi - oxmin, bx0, bx1)
                for px in range(ax, bx + 1):
                    col = slots[py * width + px]
                    if col < 0:
                        continue
                    for s in range(n_samples):
                        cx, cy = px + offsets[s, 0], py + offsets[s, 1]
                        # Barycentric weights; the slack closes hairline gaps between triangles.
                        b0 = ((x2 - x1) * (cy - y1) - (y2 - y1) * (cx - x1)) * per_area
                        if b0 < -1e-4:
                            continue
                        b1 = ((x0 - x2) * (cy - y2) - (y0 - y2) * (cx - x2)) * per_area
                        if b1 < -1e-4:
                            continue
                        b2 = ((x1 - x0) * (cy - y0) - (y1 - y0) * (cx - x0)) * per_area
                        if b2 < -1e-4:
                            continue
                        z = b0 * w0 + b1 * w1 + b2 * w2
                        if z > depth[col, s]:
                            if holes:  # perspective-correct uv, then the texture's alpha there
                                u = (b0 * w0 * attrs[t, 0, 6] + b1 * w1 * attrs[t, 1, 6] + b2 * w2 * attrs[t, 2, 6]) / z
                                v = (b0 * w0 * attrs[t, 0, 7] + b1 * w1 * attrs[t, 1, 7] + b2 * w2 * attrs[t, 2, 7]) / z
                                level = texture_lod(xs, ys, inv_w, attrs, t, b0, b1, b2, levels, first, chain[t])
                                if (sample_alpha(texels, levels, first, chain[t], u, v, level + lod[t])
                                        <= (s + 0.5) / n_samples):
                                    continue
                            depth[col, s] = z
                            tris[col, s] = t
            for k in range(k0, k1 + 1):
                drawn[k] += 1


@njit(cache=True, error_model="numpy", parallel=True)
def rasterize_pixels(depth, tris, width, height, xs, ys, inv_w, offsets, pixels, row_start, next_pixel, band_start,
                     band_tris, see, attrs, chain, lod, texels, levels, first):
    """rasterize() for scattered pixels (the edge pass and mirrors): the same tests in the same order, so the same
    result as rasterize() with slots marking just these pixels, but each band skips the triangles and rows that
    reach none of its pixels, and walks only its pixels within each row's span rather than every pixel of it.

    pixels: the flat indices of the pixels drawn, increasing; depth and tris (as for rasterize()) have a row for
    each, in that order, and are not cleared here. row_start (height + 1,): pixels[row_start[y]:row_start[y + 1]]
    are those in row y. next_pixel (width * height,): scratch space, where each band notes, for each of its pixels q,
    where in `pixels` the first one at or after q is, so that a row's pixels within a span are found at once."""
    n_samples = offsets.shape[0]
    oxmin, oxmax = offsets[:, 0].min(), offsets[:, 0].max()
    oymin, oymax = offsets[:, 1].min(), offsets[:, 1].max()
    n_bands = (height + ROW_BAND - 1) // ROW_BAND
    for nth in prange(n_bands):
        band = _scattered(nth, n_bands)
        band_y0 = band * ROW_BAND
        band_y1 = min(band_y0 + ROW_BAND, height) - 1
        # The columns the band's pixels span, so that triangles wholly beside them are skipped.
        col_lo, col_hi = width, -1
        for py in range(band_y0, band_y1 + 1):
            r0, r1 = row_start[py], row_start[py + 1]
            if r0 < r1:
                col_lo = min(col_lo, pixels[r0] - py * width)
                col_hi = max(col_hi, pixels[r1 - 1] - py * width)
        if col_lo > col_hi:
            continue
        j = row_start[band_y1 + 1]
        for q in range((band_y1 + 1) * width - 1, band_y0 * width - 1, -1):
            if j > 0 and pixels[j - 1] >= q:
                j -= 1
            next_pixel[q] = j
        for i in range(band_start[band], band_start[band + 1]):
            t = band_tris[i]
            x0, x1, x2 = xs[t, 0], xs[t, 1], xs[t, 2]
            y0, y1, y2 = ys[t, 0], ys[t, 1], ys[t, 2]
            area = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
            if not drawable(area):
                continue
            by0, by1 = pixel_range(min(y0, y1, y2) - oymax, max(y0, y1, y2) - oymin, band_y0, band_y1)
            if by0 > by1:
                continue
            bx0, bx1 = pixel_range(min(x0, x1, x2) - oxmax, max(x0, x1, x2) - oxmin, 0, width - 1)
            if bx0 > bx1 or bx1 < col_lo or bx0 > col_hi:  # (none of the band's pixels in the box)
                continue
            w0, w1, w2 = inv_w[t, 0], inv_w[t, 1], inv_w[t, 2]
            holes = see[t] == 2
            per_area = 1.0 / area
            for py in range(by0, by1 + 1):
                row = py * width
                r1 = row_start[py + 1]
                col = next_pixel[row + bx0]
                if col >= r1 or pixels[col] > row + bx1:
                    continue  # none of the pixels are in this row of the box
                lo, hi = bx0 + oxmin, bx1 + oxmax
                lo, hi = _row_span(lo, hi, x1, y1, x2 - x1, y2 - y1, per_area, py + oymin, py + oymax, width)
                lo, hi = _row_span(lo, hi, x2, y2, x0 - x2, y0 - y2, per_area, py + oymin, py + oymax, width)
                lo, hi = _row_span(lo, hi, x0, y0, x1 - x0, y1 - y0, per_area, py + oymin, py + oymax, width)
                ax, bx = pixel_range(lo - oxmax, hi - oxmin, bx0, bx1)
                if ax > bx:
                    continue
                col = next_pixel[row + ax]
                while col < r1 and pixels[col] <= row + bx:
                    px = pixels[col] - row
                    for s in range(n_samples):  # (the tests of rasterize())
                        cx, cy = px + offsets[s, 0], py + offsets[s, 1]
                        b0 = ((x2 - x1) * (cy - y1) - (y2 - y1) * (cx - x1)) * per_area
                        if b0 < -1e-4:
                            continue
                        b1 = ((x0 - x2) * (cy - y2) - (y0 - y2) * (cx - x2)) * per_area
                        if b1 < -1e-4:
                            continue
                        b2 = ((x1 - x0) * (cy - y0) - (y1 - y0) * (cx - x0)) * per_area
                        if b2 < -1e-4:
                            continue
                        z = b0 * w0 + b1 * w1 + b2 * w2
                        if z > depth[col, s]:
                            if holes:
                                u = (b0 * w0 * attrs[t, 0, 6] + b1 * w1 * attrs[t, 1, 6] + b2 * w2 * attrs[t, 2, 6]) / z
                                v = (b0 * w0 * attrs[t, 0, 7] + b1 * w1 * attrs[t, 1, 7] + b2 * w2 * attrs[t, 2, 7]) / z
                                level = texture_lod(xs, ys, inv_w, attrs, t, b0, b1, b2, levels, first, chain[t])
                                if (sample_alpha(texels, levels, first, chain[t], u, v, level + lod[t])
                                        <= (s + 0.5) / n_samples):
                                    continue
                            depth[col, s] = z
                            tris[col, s] = t
                    col += 1


LAYER_MERGE = 0.02  # fragments of one object this close in depth (relative) are one surface: one layer
LAYER_FRONT = 1e-5  # a see-through surface must be nearer than the solid one by this part of the solid one's distance
                    # from the camera's plane to show: one touching it (glass standing on a table) would otherwise
                    # flicker in and out with rounding


@njit(cache=True, error_model="numpy")
def bit_count(mask):
    """How many bits of mask are set."""
    n = 0
    while mask:
        n += mask & 1
        mask >>= 1
    return n


@njit(cache=True, error_model="numpy", parallel=True)
def rasterize_layers(solid, width, height, xs, ys, inv_w, tri_inst, offsets, band_start, band_tris, layer_depth,
                     layer_tri, layer_cover, layer_count, offset):
    """The see-through surfaces in front of the solid ones: for each pixel, the K nearest (K =
    layer_depth.shape[1]) covering any of its sample positions `offsets` (at most 16) where they are
    nearer than the solid surface (solid (width * height, S): rasterize()'s depth at the same positions) by more
    than LAYER_FRONT of its distance from the camera's plane (offset in front of the eye: see shading._plane_depth),
    nearest first. Per pixel and layer: its depth (1/w: the nearest of the samples it covers), a
    triangle to shade it with, and which samples it covers (bit s for offsets[s]); layer_count (width *
    height,) counts the layers, and where more surfaces cover a pixel the farthest are left out.

    Triangles of one object (tri_inst) whose depths in a pixel are within LAYER_MERGE of each other make
    one layer, covering the samples either covers: neighbours on a mesh, which would otherwise blend
    twice where they meet.

    band_start, band_tris: the see-through triangles touching each band of rows, as for rasterize().
    Bands are rasterized in parallel, each clearing and filling only its own pixels' lists. As in rasterize(), each
    row of a triangle is walked only over its span (_row_span), with the same tests deciding inside.
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
            area = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
            if not drawable(area):
                continue
            by0, by1 = pixel_range(min(y0, y1, y2) - oymax, max(y0, y1, y2) - oymin, band_y0, band_y1)
            if by0 > by1:
                continue
            bx0, bx1 = pixel_range(min(x0, x1, x2) - oxmax, max(x0, x1, x2) - oxmin, 0, width - 1)
            w0, w1, w2 = inv_w[t, 0], inv_w[t, 1], inv_w[t, 2]
            per_area = 1.0 / area  # (multiplying by it is several times quicker than dividing by area)
            for py in range(by0, by1 + 1):
                # Only the pixels of this row whose samples might pass all three edge tests (as in rasterize()).
                lo, hi = bx0 + oxmin, bx1 + oxmax
                lo, hi = _row_span(lo, hi, x1, y1, x2 - x1, y2 - y1, per_area, py + oymin, py + oymax, width)
                lo, hi = _row_span(lo, hi, x2, y2, x0 - x2, y0 - y2, per_area, py + oymin, py + oymax, width)
                lo, hi = _row_span(lo, hi, x0, y0, x1 - x0, y1 - y0, per_area, py + oymin, py + oymax, width)
                ax, bx = pixel_range(lo - oxmax, hi - oxmin, bx0, bx1)
                for px in range(ax, bx + 1):
                    c = py * width + px
                    cover, nearest = 0, 0.0
                    for s in range(n_samples):
                        cx, cy = px + offsets[s, 0], py + offsets[s, 1]
                        b0 = ((x2 - x1) * (cy - y1) - (y2 - y1) * (cx - x1)) * per_area
                        if b0 < -1e-4:
                            continue
                        b1 = ((x0 - x2) * (cy - y2) - (y0 - y2) * (cx - x2)) * per_area
                        if b1 < -1e-4:
                            continue
                        b2 = ((x1 - x0) * (cy - y0) - (y1 - y0) * (cx - x0)) * per_area
                        if b2 < -1e-4:
                            continue
                        z = b0 * w0 + b1 * w1 + b2 * w2
                        limit = solid[c, s]
                        if limit > 0.0:  # (0: nothing solid there; infinite: a pixel to leave alone)
                            limit = 1.0 / ((1.0 - LAYER_FRONT) / limit + LAYER_FRONT * offset)
                        if z > limit:
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


@njit(cache=True, error_model="numpy")
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


@njit(cache=True, error_model="numpy")
def texture_lod(xs, ys, inv_w, attrs, t, b0, b1, b2, levels, first, chain):
    """The mip level to sample triangle t's texture (mipmap chain `chain`) at, at barycentric weights b: log2 of
    how many texels of its finest level a pixel spans there, along whichever of the screen's x and y it spans
    more of (as graphics hardware does).

    u/w, v/w and 1/w are linear across the screen, so uv's rates of change there are exact at any point: a
    floor's texture is sampled finely where it is near and coarsely towards the horizon, where one level for
    the whole triangle would be too fine in the distance (shimmering) or too coarse close by (blurred). NaN for
    a degenerate triangle (sample() takes that as the finest level).
    """
    x0, x1, x2 = xs[t, 0], xs[t, 1], xs[t, 2]
    y0, y1, y2 = ys[t, 0], ys[t, 1], ys[t, 2]
    area = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
    # Rates of change of the barycentric weights along x and y, times each corner's 1/w.
    w0, w1, w2 = inv_w[t, 0], inv_w[t, 1], inv_w[t, 2]
    ax0, ax1, ax2 = -(y2 - y1) / area * w0, -(y0 - y2) / area * w1, -(y1 - y0) / area * w2
    ay0, ay1, ay2 = (x2 - x1) / area * w0, (x0 - x2) / area * w1, (x1 - x0) / area * w2
    u0, u1, u2 = attrs[t, 0, 6], attrs[t, 1, 6], attrs[t, 2, 6]
    v0, v1, v2 = attrs[t, 0, 7], attrs[t, 1, 7], attrs[t, 2, 7]
    d = b0 * w0 + b1 * w1 + b2 * w2  # 1/w here; u = (u/w) / (1/w), so u' = ((u/w)' - u (1/w)') / (1/w)
    u = (b0 * w0 * u0 + b1 * w1 * u1 + b2 * w2 * u2) / d
    v = (b0 * w0 * v0 + b1 * w1 * v1 + b2 * w2 * v2) / d
    dx, dy = ax0 + ax1 + ax2, ay0 + ay1 + ay2
    dudx = (ax0 * u0 + ax1 * u1 + ax2 * u2 - u * dx) / d
    dvdx = (ax0 * v0 + ax1 * v1 + ax2 * v2 - v * dx) / d
    dudy = (ay0 * u0 + ay1 * u1 + ay2 * u2 - u * dy) / d
    dvdy = (ay0 * v0 + ay1 * v1 + ay2 * v2 - v * dy) / d
    level = first[chain]
    h, w = levels[level, 1], levels[level, 2]
    along_x = (dudx * w) ** 2 + (dvdx * h) ** 2
    along_y = (dudy * w) ** 2 + (dvdy * h) ** 2
    return 0.5 * np.log2(max(along_x, along_y, 1e-18))


# The helpers transform() and project() call per face (and per instance and run) are inlined into them
# (inline="always"), as shading's are: a call that passes arrays costs more than the work in it.
@njit(cache=True, error_model="numpy", inline="always")
def _emit(corners, k, width, height, xs, ys, inv_w, attrs):
    """Store triangle `corners` (3, 4 + ATTRS: clip xyzw | attributes) as screen triangle k."""
    for j in range(3):
        iw = 1.0 / corners[j, 3]
        inv_w[k, j] = iw
        xs[k, j] = (corners[j, 0] * iw + 1.0) * 0.5 * width
        ys[k, j] = (1.0 - corners[j, 1] * iw) * 0.5 * height
        attrs[k, j, :] = corners[j, 4:]


@njit(cache=True, error_model="numpy", inline="always")
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


@njit(cache=True, error_model="numpy", inline="always")
def stretch(lin):
    """At most how much the linear map lin (3, 3) lengthens any vector (at least its largest singular value,
    and exactly that for a rotation times a scale along each axis): how far a bounding sphere's radius grows."""
    most = 0.0
    for i in range(3):
        row = 0.0
        for j in range(3):  # row i of lin^T lin, whose largest eigenvalue is the square of the answer
            row += abs(lin[0, i] * lin[0, j] + lin[1, i] * lin[1, j] + lin[2, i] * lin[2, j])
        most = max(most, row)
    return np.sqrt(most)


@njit(cache=True, error_model="numpy", inline="always")
def _normal_matrix(lin, out):
    """What turns normals when lin (3, 3) turns and stretches points, into out (3, 3): the inverse transpose, up
    to a positive factor (the cofactors, times the sign of the determinant), which stays finite when lin is
    singular. Normals it gives need normalizing."""
    for i in range(3):
        for j in range(3):
            a, b = (i + 1) % 3, (i + 2) % 3
            c, d = (j + 1) % 3, (j + 2) % 3
            out[i, j] = lin[a, c] * lin[b, d] - lin[a, d] * lin[b, c]
    det = lin[0, 0] * out[0, 0] + lin[0, 1] * out[0, 1] + lin[0, 2] * out[0, 2]
    if det < 0.0:
        for i in range(3):
            for j in range(3):
                out[i, j] = -out[i, j]


@njit(cache=True, error_model="numpy", inline="always")
def _face_ahead(world, faces, f, v0, near):
    """How many corners of face f are in front of the near plane."""
    ahead = 0
    for j in range(3):
        ahead += world[v0 + faces[f, j], 9] > near
    return ahead


@njit(cache=True, error_model="numpy", inline="always")
def _facing(world, faces, f, v0, eye):
    """Whether face f (world positions in `world`) faces the eye."""
    a, b, c = world[v0 + faces[f, 0]], world[v0 + faces[f, 1]], world[v0 + faces[f, 2]]
    e1x, e1y, e1z = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    e2x, e2y, e2z = c[0] - a[0], c[1] - a[1], c[2] - a[2]
    nx, ny, nz = e1y * e2z - e1z * e2y, e1z * e2x - e1x * e2z, e1x * e2y - e1y * e2x
    return nx * (eye[0] - a[0]) + ny * (eye[1] - a[1]) + nz * (eye[2] - a[2]) > 0


NO_CLIP = np.array([0.0, 0.0, 0.0, 1.0])  # a clipping plane (a, b, c, d: keeps a x + b y + c z + d > 0) keeping all


@njit(cache=True, error_model="numpy", inline="always")
def _plane_distance(world, v, clip):
    """How far in front of clipping plane `clip` vertex v of `world` is (positive: kept)."""
    return clip[0] * world[v, 0] + clip[1] * world[v, 1] + clip[2] * world[v, 2] + clip[3]


@njit(cache=True, error_model="numpy", inline="always")
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


@njit(cache=True, error_model="numpy", inline="always")
def _plane_clipped_length(ds, n):
    """How many corners a polygon of n corners with plane distances ds keeps, clipped against the plane."""
    m = 0
    for j in range(n):
        a, b = ds[j], ds[(j + 1) % n]
        m += (a > 0.0) + ((a > 0.0) != (b > 0.0))
    return m


FACE_CHUNK = 512     # faces projected as one piece of parallel work
VERTEX_CHUNK = 256   # vertices transformed as one piece
RUN_CELLS = 8        # a mesh of more than FACE_CHUNK faces is cut into runs at most 1/RUN_CELLS of its size across,
RUN_LEAST = 16       # of at least this many faces
RUN_MOST = 256       # and into at most about this many runs (so that each is worth its test)


@njit(cache=True, error_model="numpy")
def face_runs(vertices, faces, limit, least, most, starts):
    """Cut a mesh's faces, in their order, into runs for culling (see transform()): each run of at most `most`
    faces, and growing while the box around its corners is at most `limit` across (or it has fewer than `least`
    faces). Writes where each run starts into starts (F,), and returns how many there are. Faces stay in order, so
    the triangles drawn are the same and in the same order, whichever runs are culled; a mesh whose faces come
    in order of place (as a level put together room by room or object by object does) gets runs each in one
    place. NaN corners make no box too big (a comparison with NaN is false)."""
    n = 0
    f = 0
    while f < faces.shape[0]:
        starts[n] = f
        n += 1
        lo0 = lo1 = lo2 = np.inf
        hi0 = hi1 = hi2 = -np.inf
        count = 0
        while f < faces.shape[0] and count < most:
            a0, a1, a2, b0, b1, b2 = lo0, lo1, lo2, hi0, hi1, hi2
            for j in range(3):
                v = vertices[faces[f, j]]
                a0, a1, a2 = min(a0, v[0]), min(a1, v[1]), min(a2, v[2])
                b0, b1, b2 = max(b0, v[0]), max(b1, v[1]), max(b2, v[2])
            if count >= least and (b0 - a0 > limit or b1 - a1 > limit or b2 - a2 > limit):
                break
            lo0, lo1, lo2, hi0, hi1, hi2 = a0, a1, a2, b0, b1, b2
            f += 1
            count += 1
    return n


@njit(cache=True, error_model="numpy", parallel=True)
def transform(vertices, vertex_normals, faces, mesh_vertex, spheres, inst_mesh, inst_lin, inst_pos, inst_double,
              inst_flip, inst_vertex, view_proj, eye, near, clip, vchunk_inst, vchunk_first, vchunk_end, inst_vchunk,
              chunk_inst, chunk_run, chunk_first, chunk_end, mesh_run, run_spheres, run_vertex, world, visible, seen,
              needed, whole, cut, off_whole, off_cut):
    """First pass of projecting instances (see project()): vertices in the world, and where each piece
    of work will write its triangles. Returns the number of triangles.

    Instance i shows mesh inst_mesh[i], its points p placed at inst_lin[i] @ p + inst_pos[i] (inst_lin: rotation,
    scale along each axis, and the same of its parents, as one (3, 3) matrix); inst_flip[i] says the matrix
    mirrors (a negative determinant), which turns faces inside out, so that which way they face flips too.

    visible[i]: whether instance i is at least partly in view (its bounding sphere, spheres[mesh]).
    Face chunk c covers faces chunk_first[c]:chunk_end[c] of instance chunk_inst[c] (chunks in instance order):
    run chunk_run[c] of its mesh (mesh m has runs mesh_run[m]:mesh_run[m + 1], see face_runs), whose corners lie
    in sphere run_spheres[r] and are vertices run_vertex[r, 0]:run_vertex[r, 1] (counted within the mesh).
    seen[c]: whether chunk c is at least partly in view: its instance visible and, where its mesh has more
    than one run, its own sphere not wholly out of view (a NaN sphere is seen). Vertex chunk j transforms
    vertices vchunk_first[j]:vchunk_end[j] (counted within the mesh) of instance vchunk_inst[j] (whose vertex
    chunks start at inst_vchunk[i]) into world, from row inst_vertex[i] for instance i: world xyz | normal xyz |
    clip xyzw; only the chunks of vertices that seen face chunks use (needed[j]), the other rows are left as
    they were. Seen chunk c's faces: whole[c] of them will be drawn whole and `cut` pieces will be left
    by clipping, written from rows off_whole[c] and off_cut[c], so that each instance's whole faces
    come before the pieces of its clipped ones.

    Faces are also clipped against the plane `clip` (see NO_CLIP): a mirror's, when drawing what it shows.
    """
    for inst in prange(inst_mesh.shape[0]):
        m = inst_mesh[inst]
        lin = inst_lin[inst]
        position = inst_pos[inst]
        sx, sy, sz = spheres[m, 0], spheres[m, 1], spheres[m, 2]
        cx = lin[0, 0] * sx + lin[0, 1] * sy + lin[0, 2] * sz + position[0]
        cy = lin[1, 0] * sx + lin[1, 1] * sy + lin[1, 2] * sz + position[1]
        cz = lin[2, 0] * sx + lin[2, 1] * sy + lin[2, 2] * sz + position[2]
        radius = spheres[m, 3] * stretch(lin)
        visible[inst] = (not _outside_view(view_proj, cx, cy, cz, radius, near)
                         and clip[0] * cx + clip[1] * cy + clip[2] * cz + clip[3] > -radius)
    for c in prange(chunk_inst.shape[0]):
        inst = chunk_inst[c]
        m = inst_mesh[inst]
        seen[c] = visible[inst]
        if visible[inst] and mesh_run[m + 1] - mesh_run[m] > 1:  # (one run: its sphere is the mesh's)
            r = chunk_run[c]
            lin = inst_lin[inst]
            position = inst_pos[inst]
            sx, sy, sz = run_spheres[r, 0], run_spheres[r, 1], run_spheres[r, 2]
            cx = lin[0, 0] * sx + lin[0, 1] * sy + lin[0, 2] * sz + position[0]
            cy = lin[1, 0] * sx + lin[1, 1] * sy + lin[1, 2] * sz + position[1]
            cz = lin[2, 0] * sx + lin[2, 1] * sy + lin[2, 2] * sz + position[2]
            radius = run_spheres[r, 3] * stretch(lin)
            # (NaN culls nothing here: a run is drawn whenever its instance is, unless surely out of view)
            seen[c] = not (_outside_view(view_proj, cx, cy, cz, radius, near)
                           or clip[0] * cx + clip[1] * cy + clip[2] * cz + clip[3] <= -radius)
    # The vertex chunks the seen face chunks use (serially: chunks of one instance share vertex chunks).
    for j in range(needed.shape[0]):
        needed[j] = False
    for c in range(chunk_inst.shape[0]):
        if seen[c]:
            inst, r = chunk_inst[c], chunk_run[c]
            if run_vertex[r, 1] > run_vertex[r, 0]:
                for j in range(inst_vchunk[inst] + run_vertex[r, 0] // VERTEX_CHUNK,
                               inst_vchunk[inst] + (run_vertex[r, 1] - 1) // VERTEX_CHUNK + 1):
                    needed[j] = True
    for j in prange(vchunk_inst.shape[0]):
        if not needed[j]:
            continue
        inst = vchunk_inst[j]
        m = inst_mesh[inst]
        lin = inst_lin[inst]
        normal = np.empty((3, 3))
        _normal_matrix(lin, normal)
        position = inst_pos[inst]
        v0, w0 = mesh_vertex[m], inst_vertex[inst]
        for v in range(vchunk_first[j], vchunk_end[j]):
            p, n, out = vertices[v0 + v], vertex_normals[v0 + v], world[w0 + v]
            for i in range(3):
                out[i] = lin[i, 0] * p[0] + lin[i, 1] * p[1] + lin[i, 2] * p[2] + position[i]
                out[3 + i] = normal[i, 0] * n[0] + normal[i, 1] * n[1] + normal[i, 2] * n[2]
            length = np.sqrt(out[3] * out[3] + out[4] * out[4] + out[5] * out[5])
            if length > 0.0:
                out[3], out[4], out[5] = out[3] / length, out[4] / length, out[5] / length
            for i in range(4):
                out[6 + i] = view_proj[i, 0] * out[0] + view_proj[i, 1] * out[1] + view_proj[i, 2] * out[2] + view_proj[i, 3]
    for c in prange(chunk_inst.shape[0]):
        inst = chunk_inst[c]
        n_whole = n_cut = 0
        ds = np.empty(4)
        if seen[c]:
            base = inst_vertex[inst] - mesh_vertex[inst_mesh[inst]]  # faces index the packed vertices
            for f in range(chunk_first[c], chunk_end[c]):
                ahead = _face_ahead(world, faces, f, base, near)
                if ahead == 0 or not (inst_double[inst] or _facing(world, faces, f, base, eye) != inst_flip[inst]):
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


@njit(cache=True, error_model="numpy", parallel=True)
def project(faces, uvs, colors, face_chain, face_kind, mesh_vertex, inst_mesh, inst_rgb, inst_alpha, inst_double, inst_flip, inst_vertex, eye, near, clip, width, height, lod_bias, world, chunk_inst,
            chunk_first, chunk_end, whole, cut, off_whole, off_cut, xs, ys, inv_w, attrs, tri_inst, chain, lod, see):
    """Every instance's faces as screen-space triangles: culled, clipped against the near plane
    w = near, and projected, from the vertices and plan transform() made.

    The meshes are packed together: mesh m has vertices mesh_vertex[m]:mesh_vertex[m + 1], and
    faces index the packed vertices. Per face: `faces` (F, 3), `uvs` (F, 3, 2), `colors` (F, 3, 4:
    linear rgb and alpha of each corner), `face_chain` (F,: mipmap chain, or -1 if untextured) and
    `face_kind` (F,: what its texture's alpha is for, see texture.alpha_kind). Instance i shows mesh inst_mesh[i] in colour
    inst_rgb[i] (linear) with opacity inst_alpha[i]; back faces (seen from `eye`) are culled unless
    inst_double[i]: then they are drawn with flipped normals. inst_flip: see transform().

    The triangles go to xs, ys, inv_w (pixel coordinates and 1/w of each corner), attrs (ATTRS
    attributes of each corner), tri_inst (the instance), chain (the face's mipmap chain), lod (lod_bias:
    added to the mip level texture_lod() gives each pixel) and see (its kind: 1 if it can be seen through, as a corner's alpha
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
            if _facing(world, faces, f, base, eye) == inst_flip[inst]:
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
        # Each triangle's instance, mipmap chain and mip level bias.
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
                lod[t] = lod_bias


SURFACE = 7  # per corner of a shadow map's triangle: linear rgb | alpha | u / w, v / w, 1 / w (for its texture)


@njit(cache=True, error_model="numpy")
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


@njit(cache=True, error_model="numpy")
def _corner_surface(colors, uvs, inst_rgb, inst_alpha, inst, f, j, out):
    """Corner j of face f of instance inst as a shadow map needs it: linear rgb into out[4:7], alpha into
    out[7] and uv into out[8:10]."""
    for i in range(3):
        out[4 + i] = min(max(colors[f, j, i] * inst_rgb[inst, i], 0.0), 1.0)
    out[7] = colors[f, j, 3] * inst_alpha[inst]
    out[8], out[9] = uvs[f, j, 0], uvs[f, j, 1]


@njit(cache=True, error_model="numpy")
def _face_kind(colors, inst_alpha, face_kind, inst, f):
    """Kind of face f of instance inst, as project() gives triangles: 1 see-through, 2 cut-out, 0 solid."""
    if face_kind[f] == 2 or inst_alpha[inst] < 1.0:
        return 1
    for j in range(3):
        if colors[f, j, 3] < 1.0:
            return 1
    return 2 if face_kind[f] == 1 else 0


@njit(cache=True, error_model="numpy")
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


@njit(cache=True, error_model="numpy")
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


@njit(cache=True, error_model="numpy", parallel=True)
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


@njit(cache=True, error_model="numpy")
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


@njit(cache=True, error_model="numpy", parallel=True)
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


@njit(cache=True, error_model="numpy", parallel=True)
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


@njit(cache=True, error_model="numpy")
def _surface_uv(surf, t, b0, b1, b2):
    """Perspective-correct uv at barycentric weights b of shadow triangle t."""
    w = max(b0 * surf[t, 0, 6] + b1 * surf[t, 1, 6] + b2 * surf[t, 2, 6], 1e-30)
    return ((b0 * surf[t, 0, 4] + b1 * surf[t, 1, 4] + b2 * surf[t, 2, 4]) / w,
            (b0 * surf[t, 0, 5] + b1 * surf[t, 1, 5] + b2 * surf[t, 2, 5]) / w)


@njit(cache=True, error_model="numpy", parallel=True)
def rasterize_depth(depth, base, xs, ys, dep, tri_side, side_rows, band_start, band_tris, see, surf, chain, lod,
                    texels, levels, first):
    """A shadow map: the largest `dep` (nearest the light) of any triangle covering each texel's centre,
    into depth (height, width), which it clears first (to `base`, a map of other triangles drawn before, if that
    has rows; else to 0, where there is none). Triangle t is kept to rows
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
        if base.shape[0]:
            depth[band_y0:band_y1 + 1, :] = base[band_y0:band_y1 + 1, :]
        else:
            depth[band_y0:band_y1 + 1, :] = 0.0
        for i in range(band_start[band], band_start[band + 1]):
            t = band_tris[i]
            x0, x1, x2 = xs[t, 0], xs[t, 1], xs[t, 2]
            y0, y1, y2 = ys[t, 0], ys[t, 1], ys[t, 2]
            top = tri_side[t] * side_rows
            area = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
            if not drawable(area):
                continue
            by0, by1 = pixel_range(min(y0, y1, y2) - 0.5, max(y0, y1, y2) - 0.5, max(band_y0, top),
                                   min(band_y1, top + side_rows - 1))
            if by0 > by1:
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


@njit(cache=True, error_model="numpy", parallel=True)
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
            area = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
            if not drawable(area):
                continue
            by0, by1 = pixel_range(min(y0, y1, y2) - 0.5, max(y0, y1, y2) - 0.5, max(band_y0, top),
                                   min(band_y1, top + side_rows - 1))
            if by0 > by1:
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
