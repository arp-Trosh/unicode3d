# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
"""The rasterizers against plain references: every sample of every pixel in the triangle's bounding box tested
(rasterize as it was before it walked spans, roadmap item 26, and rasterize_layers as it was before it did too), on
random and awkward triangles."""
import unittest

import numpy as np
from numba import njit

from unicode3d.raster import (LAYER_FRONT, LAYER_MERGE, ROW_BAND, SAMPLE_PATTERNS, _row_span, bin_bands, bit_count,
                              count_bands, drawable, pixel_range, rasterize, rasterize_layers, rasterize_pixels,
                              texture_lod)
from unicode3d.texture import sample_alpha


@njit(cache=False, error_model="numpy")
def reference_rasterize(depth, tris, width, height, xs, ys, inv_w, offsets, slots, band_start, band_tris, see, attrs,
                        chain, lod, texels, levels, first, clear):
    """rasterize() as it was before spans (5f43094), serial: each sample of each pixel in the box tested."""
    n_samples = offsets.shape[0]
    oxmin, oxmax = offsets[:, 0].min(), offsets[:, 0].max()
    oymin, oymax = offsets[:, 1].min(), offsets[:, 1].max()
    n_bands = (height + ROW_BAND - 1) // ROW_BAND
    for band in range(n_bands):
        band_y0 = band * ROW_BAND
        band_y1 = min(band_y0 + ROW_BAND, height) - 1
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
            by0, by1 = pixel_range(min(y0, y1, y2) - oymax, max(y0, y1, y2) - oymin, band_y0, band_y1)
            if by0 > by1:
                continue
            bx0, bx1 = pixel_range(min(x0, x1, x2) - oxmax, max(x0, x1, x2) - oxmin, 0, width - 1)
            w0, w1, w2 = inv_w[t, 0], inv_w[t, 1], inv_w[t, 2]
            holes = see[t] == 2
            per_area = 1.0 / area
            for py in range(by0, by1 + 1):
                for px in range(bx0, bx1 + 1):
                    col = slots[py * width + px]
                    if col < 0:
                        continue
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


@njit(cache=False, error_model="numpy")
def reference_rasterize_layers(solid, width, height, xs, ys, inv_w, tri_inst, offsets, band_start, band_tris,
                               layer_depth, layer_tri, layer_cover, layer_count, offset):
    """rasterize_layers() as it was before spans (9a52786), serial: each sample of each pixel in the box tested
    (multiplying by 1 / area as the kernel now does, rather than dividing)."""
    n_samples, k_max = offsets.shape[0], layer_depth.shape[1]
    oxmin, oxmax = offsets[:, 0].min(), offsets[:, 0].max()
    oymin, oymax = offsets[:, 1].min(), offsets[:, 1].max()
    n_bands = (height + ROW_BAND - 1) // ROW_BAND
    for band in range(n_bands):
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
            per_area = 1.0 / area  # (as the kernel: multiplied, not divided, so the last bit agrees)
            for py in range(by0, by1 + 1):
                for px in range(bx0, bx1 + 1):
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
                        if limit > 0.0:
                            limit = 1.0 / ((1.0 - LAYER_FRONT) / limit + LAYER_FRONT * offset)
                        if z > limit:
                            cover |= 1 << s
                            nearest = max(nearest, z)
                    if cover == 0:
                        continue
                    n = layer_count[c]
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
                            for k in range(j, n - 1):
                                layer_depth[c, k], layer_tri[c, k], layer_cover[c, k] = (layer_depth[c, k + 1],
                                                                                         layer_tri[c, k + 1],
                                                                                         layer_cover[c, k + 1])
                            n -= 1
                            break
                    else:
                        t_keep = t
                    if n == k_max and nearest <= layer_depth[c, k_max - 1]:
                        continue
                    j = n if n < k_max else k_max - 1
                    while j > 0 and layer_depth[c, j - 1] < nearest:
                        layer_depth[c, j], layer_tri[c, j], layer_cover[c, j] = (layer_depth[c, j - 1],
                                                                                 layer_tri[c, j - 1],
                                                                                 layer_cover[c, j - 1])
                        j -= 1
                    layer_depth[c, j], layer_tri[c, j], layer_cover[c, j] = nearest, t_keep, cover
                    layer_count[c] = min(n + 1, k_max)


def triangles(rng, kind, n, width, height):
    """n triangles' corners (xs, ys: (n, 3) pixel coordinates) of one kind of awkwardness."""
    def around(centre_x, centre_y, size):
        return (centre_x[:, None] + rng.uniform(-1, 1, (n, 3)) * size[:, None],
                centre_y[:, None] + rng.uniform(-1, 1, (n, 3)) * size[:, None])
    cx, cy = rng.uniform(-2, width + 2, n), rng.uniform(-2, height + 2, n)
    if kind == "random":
        return rng.uniform(-0.2 * width, 1.2 * width, (n, 3)), rng.uniform(-0.2 * height, 1.2 * height, (n, 3))
    if kind == "tiny":  # smaller than a pixel, as levels of detail leave far things
        return around(cx, cy, 10.0 ** rng.uniform(-3, 0.5, n))
    if kind == "sliver":  # a corner almost on the line through the other two
        xs, ys = around(cx, cy, rng.uniform(1, 40, n))
        f = rng.uniform(-0.5, 1.5, n)
        nudge = 10.0 ** rng.uniform(-8, -1, n) * rng.choice([-1, 1], n)
        xs[:, 2] = xs[:, 0] + f * (xs[:, 1] - xs[:, 0]) + nudge
        ys[:, 2] = ys[:, 0] + f * (ys[:, 1] - ys[:, 0]) - nudge
        return xs, ys
    if kind == "flat":  # an edge within a hair of horizontal or vertical
        xs, ys = around(cx, cy, rng.uniform(0.5, 30, n))
        hair = 10.0 ** rng.uniform(-12, -2, n) * rng.choice([-1, 1], n)
        horizontal = rng.random(n) < 0.5
        ys[horizontal, 1] = ys[horizontal, 0] + hair[horizontal]
        xs[~horizontal, 1] = xs[~horizontal, 0] + hair[~horizontal]
        return xs, ys
    if kind == "huge":  # corners far off screen (one near the screen, or none)
        xs = rng.uniform(-1, 1, (n, 3)) * 10.0 ** rng.uniform(3, 9, (n, 1))
        ys = rng.uniform(-1, 1, (n, 3)) * 10.0 ** rng.uniform(3, 9, (n, 1))
        near = rng.random(n) < 0.5
        xs[near, 0], ys[near, 0] = cx[near], cy[near]
        return xs, ys
    if kind == "grid":  # corners on pixel and sample positions: samples exactly on edges
        step = rng.choice([0.0625, 0.125, 0.25, 0.5, 1.0], (n, 1))
        return (np.round(rng.uniform(-2, width + 2, (n, 3)) / step) * step,
                np.round(rng.uniform(-2, height + 2, (n, 3)) / step) * step)
    if kind == "mesh":  # a jittered grid of quads, two triangles each, sharing edges (cracks would show here)
        k = max(int(np.sqrt(n / 2)), 1)
        gx, gy = np.meshgrid(np.linspace(-1, width + 1, k + 1), np.linspace(-1, height + 1, k + 1))
        gx = gx + rng.uniform(-0.4, 0.4, gx.shape) * width / k
        gy = gy + rng.uniform(-0.4, 0.4, gy.shape) * height / k
        a, b = (gx[:-1, :-1], gy[:-1, :-1]), (gx[:-1, 1:], gy[:-1, 1:])
        c, d = (gx[1:, 1:], gy[1:, 1:]), (gx[1:, :-1], gy[1:, :-1])

        def corners(k):
            return np.concatenate([np.stack([a[k], b[k], c[k]], -1).reshape(-1, 3),
                                   np.stack([a[k], c[k], d[k]], -1).reshape(-1, 3)])
        xs, ys = corners(0), corners(1)
        return xs, ys
    if kind == "edge":  # a sample just at an edge's threshold (weight -1e-4, give or take rounding)
        size = np.where(rng.random(n) < 0.5, rng.uniform(0.5, 60, n), 10.0 ** rng.uniform(3, 8, n))
        xs, ys = rng.uniform(-1, 1, (n, 3)) * size[:, None], rng.uniform(-1, 1, (n, 3)) * size[:, None]
        k = rng.integers(0, 3, n)
        rows = np.arange(n)
        a, b, o = k, (k + 1) % 3, (k + 2) % 3  # the edge from corner a to b; the weight of the corner o opposite
        u = rng.uniform(0.2, 0.8, n)
        on_x = xs[rows, a] + u * (xs[rows, b] - xs[rows, a])
        on_y = ys[rows, a] + u * (ys[rows, b] - ys[rows, a])
        offsets = np.array(SAMPLE_PATTERNS[1] + SAMPLE_PATTERNS[4] + SAMPLE_PATTERNS[8])
        pick = offsets[rng.integers(0, len(offsets), n)]
        sample_x = rng.integers(0, max(width, 1), n) + pick[:, 0]
        sample_y = rng.integers(0, max(height, 1), n) + pick[:, 1]
        xs += (sample_x - on_x)[:, None]  # the point on the edge moved onto the sample
        ys += (sample_y - on_y)[:, None]
        # Then along x until the weight there is -1e-4, as rasterize works it out, plus a few rounding steps.
        x0, x1, x2 = xs[rows, o], xs[rows, a], xs[rows, b]
        y0, y1, y2 = ys[rows, o], ys[rows, a], ys[rows, b]
        with np.errstate(all="ignore"):
            per_area = 1.0 / ((x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0))
            weight = ((x2 - x1) * (sample_y - y1) - (y2 - y1) * (sample_x - x1)) * per_area
            shift = -(weight + 1e-4) / ((y2 - y1) * per_area)
        shift = np.where(np.isfinite(shift), shift, 0.0)
        steps = rng.integers(-8, 9, n) * np.spacing(np.maximum(abs(xs).max(axis=1), 1.0))
        xs += (shift + steps)[:, None]
        return xs, ys
    if kind == "bad":  # NaN and infinite corners among ordinary ones
        xs, ys = around(cx, cy, rng.uniform(1, 30, n))
        bad = rng.random((n, 3)) < 0.2
        xs[bad] = rng.choice([np.nan, np.inf, -np.inf, 1e300], bad.sum())
        bad = rng.random((n, 3)) < 0.1
        ys[bad] = rng.choice([np.nan, np.inf, -np.inf, -1e300], bad.sum())
        return xs, ys
    raise ValueError(kind)


# ("edge" twice: the closest calls)
KINDS = ("random", "tiny", "sliver", "flat", "huge", "grid", "mesh", "edge", "bad", "edge")


def draw(kernel, xs, ys, inv_w, see, attrs, chain, lod, texture, width, height, n_samples, slots):
    """depth and tris from one kernel, binned as Renderer._bins does."""
    n_bands = (height + ROW_BAND - 1) // ROW_BAND
    span = np.tile([-np.inf, np.inf], (n_bands, 1))
    select = see.astype(np.int64)
    band_count = np.zeros(n_bands, np.int64)
    per_chunk = count_bands(xs, ys, select, 7, height, span, band_count)
    band_start = np.zeros(n_bands + 1, np.int64)
    np.cumsum(band_count, out=band_start[1:])
    band_tris = np.zeros(int(band_start[-1]), np.int64)
    bin_bands(xs, ys, select, 7, height, span, per_chunk, band_start, band_tris)
    m = int(slots.max()) + 1
    depth = np.full((m, n_samples), 7.0)  # (anything: clear empties a whole frame's, others are filled here)
    tris = np.full((m, n_samples), 99, np.int32)
    whole = m == width * height
    if not whole or kernel is rasterize_pixels:  # (which leaves emptying them to its caller)
        depth.fill(0.0)
        tris.fill(-1)
    offsets = np.array(SAMPLE_PATTERNS[n_samples], float)
    if kernel is rasterize_pixels:  # (the pixels in order, with their rows: as Renderer._accumulate gives them)
        pixels = np.flatnonzero(slots >= 0)
        row_start = np.searchsorted(pixels, np.arange(height + 1) * width)
        kernel(depth, tris, width, height, xs, ys, inv_w, offsets, pixels, row_start,
               np.full(width * height, -99, np.int64), band_start, band_tris, see, attrs, chain, lod, *texture)
    else:
        kernel(depth, tris, width, height, xs, ys, inv_w, offsets, slots, band_start, band_tris, see, attrs, chain,
               lod, *texture, whole)
    return depth, tris


def draw_layers(kernel, xs, ys, inv_w, tri_inst, solid, width, height, n_samples, k_max, offset):
    """The layer lists from one kernel, binned as Renderer._layers does (every triangle see-through)."""
    n_bands = (height + ROW_BAND - 1) // ROW_BAND
    span = np.tile([-np.inf, np.inf], (n_bands, 1))
    select = np.ones(len(xs), np.int64)
    band_count = np.zeros(n_bands, np.int64)
    per_chunk = count_bands(xs, ys, select, 2, height, span, band_count)
    band_start = np.zeros(n_bands + 1, np.int64)
    np.cumsum(band_count, out=band_start[1:])
    band_tris = np.zeros(int(band_start[-1]), np.int64)
    bin_bands(xs, ys, select, 2, height, span, per_chunk, band_start, band_tris)
    m = width * height
    # (filled with anything: only the first layer_count of each pixel's are written, and compared)
    out = np.full((m, k_max), 7.0), np.full((m, k_max), 99, np.int32), np.full((m, k_max), 5, np.int32)
    count = np.full(m, 3, np.int32)
    kernel(solid, width, height, xs, ys, inv_w, tri_inst, np.array(SAMPLE_PATTERNS[n_samples], float), band_start,
           band_tris, *out, count, offset)
    return out + (count,)


@njit(cache=False, error_model="numpy")
def samples_left_out(edges, cxs, width):
    """How many of the sample positions (cx, cy) that pass rasterize()'s test for an edge fall outside the span
    _row_span gives that edge for the row (cy_lo..cy_hi). edges rows: xa, ya, ex, ey, per_area, cy_lo, cy_hi."""
    missed = 0
    for i in range(edges.shape[0]):
        xa, ya, ex, ey, per_area, cy_lo, cy_hi = edges[i]
        lo, hi = _row_span(-np.inf, np.inf, xa, ya, ex, ey, per_area, cy_lo, cy_hi, width)
        for cy in (cy_lo, cy_hi):
            for j in range(cxs.shape[1]):
                cx = cxs[i, j]
                if (ex * (cy - ya) - ey * (cx - xa)) * per_area >= -1e-4 and not lo <= cx <= hi:
                    missed += 1
    return missed


class RasterizeTests(unittest.TestCase):
    def test_a_row_span_keeps_every_sample_that_passes(self):
        """_row_span leaves out no sample position the exact test passes, at the scale of rounding errors (which
        rasterize's rounding to whole pixels mostly hides): sample positions a few rounding steps either side of
        where the weight crosses the threshold, for edges of every slope (within a hair of horizontal too), size
        and distance from the origin."""
        rng = np.random.default_rng(2610)
        n = 200000
        sign = rng.choice([-1.0, 1.0], (n, 4))
        ex, ey = sign[:, 0] * 10.0 ** rng.uniform(-6, 4, n), sign[:, 1] * 10.0 ** rng.uniform(-15, 4, n)
        xa, ya = sign[:, 2] * 10.0 ** rng.uniform(-2, 8, n), rng.uniform(-50, 250, n)
        per_area = sign[:, 3] * 10.0 ** rng.uniform(-8, 8, n)
        cy_lo = rng.integers(0, 200, n) + rng.choice([0.0625, 0.125, 0.5], n)
        cy_hi = cy_lo + rng.choice([0.0, 0.375, 0.75, 0.875], n)
        edges = np.stack([xa, ya, ex, ey, per_area, cy_lo, cy_hi], 1)
        with np.errstate(all="ignore"):  # where the weight crosses -1e-4 along the row's top and bottom sample heights
            cross = np.stack([xa + (ex * (cy - ya) + 1e-4 / per_area) / ey for cy in (cy_lo, cy_hi)], 1)
        cross = np.where(np.isfinite(cross), cross, rng.uniform(-10, 300, (n, 2)))
        steps = rng.integers(-40, 41, (n, 2, 16)) * np.spacing(np.maximum(abs(cross), 1.0))[:, :, None]
        cxs = (cross[:, :, None] + steps).reshape(n, -1)
        self.assertEqual(samples_left_out(edges, cxs, 400), 0)

    def test_same_samples_as_testing_the_whole_box(self):
        """Walking spans covers exactly the samples testing every sample of the bounding box does: random,
        tiny, sliver, nearly flat, huge, on-the-grid, mesh and broken triangles, cut-outs among them, at every
        sample count, for whole frames and for scattered pixels (as the edge pass and mirrors draw, through
        rasterize_pixels, which the whole frames check too: every pixel listed)."""
        rng = np.random.default_rng(26)
        size = 8
        alpha = rng.random((size, size))
        texels = np.concatenate([rng.random((size * size, 3)), alpha.reshape(-1, 1)], 1).astype(np.float32)
        texture = (texels, np.array([[0, size, size]], np.int64), np.array([0, 1], np.int64))
        checked = 0
        for trial in range(48):
            kind = KINDS[trial % len(KINDS)]
            width, height = (int(rng.integers(1, 90)), int(rng.integers(1, 50))) if trial % 3 else (61, 37)
            xs, ys = triangles(rng, kind, int(rng.integers(20, 300)), width, height)
            xs, ys = np.ascontiguousarray(xs, float), np.ascontiguousarray(ys, float)
            n = xs.shape[0]
            inv_w = rng.uniform(0.01, 2.0, (n, 3))
            see = np.where(rng.random(n) < 0.2, 2, 0).astype(np.int8)
            attrs = rng.uniform(-0.5, 1.5, (n, 3, 12))
            chain = np.where(see == 2, 0, -1).astype(np.int64)
            lod = rng.uniform(-1, 1, n)
            for n_samples in (1, 4, 8, 16):
                for scattered in (False, True):
                    if scattered:
                        pixels = np.flatnonzero(rng.random(width * height) < rng.choice([0.02, 0.3, 0.9]))
                        if not len(pixels):
                            continue
                        slots = np.full(width * height, -1, np.int64)
                        slots[pixels] = np.arange(len(pixels))
                    else:
                        slots = np.arange(width * height, dtype=np.int64)
                    args = (xs, ys, inv_w, see, attrs, chain, lod, texture, width, height, n_samples, slots)
                    want_depth, want_tris = draw(reference_rasterize, *args)
                    for kernel in (rasterize, rasterize_pixels):
                        got_depth, got_tris = draw(kernel, *args)
                        where = f"{kind}, {width}x{height}, {n_samples} samples, scattered {scattered}, {kernel}"
                        np.testing.assert_array_equal(got_tris, want_tris, err_msg=where)
                        np.testing.assert_array_equal(got_depth, want_depth, err_msg=where)
                    checked += int((want_tris >= 0).sum())
        self.assertGreater(checked, 100000)  # (the cases do cover samples)


    def test_same_layers_as_testing_the_whole_box(self):
        """rasterize_layers() walking spans makes the same layer lists as testing every sample of the bounding box:
        the same kinds of triangles, of a few objects (so that layers merge), over solid surfaces at random depths
        (some nearer than the triangles, some infinite, some empty), at every sample count and up to 4 layers."""
        rng = np.random.default_rng(2626)
        checked = 0
        for trial in range(40):
            kind = KINDS[trial % len(KINDS)]
            width, height = (int(rng.integers(1, 90)), int(rng.integers(1, 50))) if trial % 3 else (61, 37)
            xs, ys = triangles(rng, kind, int(rng.integers(20, 300)), width, height)
            xs, ys = np.ascontiguousarray(xs, float), np.ascontiguousarray(ys, float)
            n = xs.shape[0]
            inv_w = rng.uniform(0.01, 2.0, (n, 3))
            if trial % 4 == 1:  # one object at one depth: every neighbour merges
                inv_w[:] = rng.uniform(0.5, 1.0)
            tri_inst = rng.integers(0, int(rng.integers(1, 6)), n).astype(np.int64)
            offset = float(rng.choice([0.0, 0.1, 1.0]))
            for n_samples in (1, 4, 8, 16):
                solid = rng.uniform(0.0, 1.0, (width * height, n_samples))
                solid[rng.random(solid.shape) < 0.3] = 0.0
                solid[rng.random(solid.shape) < 0.05] = np.inf
                k_max = int(rng.integers(1, 5))
                args = (xs, ys, inv_w, tri_inst, solid, width, height, n_samples, k_max, offset)
                want = draw_layers(reference_rasterize_layers, *args)
                got = draw_layers(rasterize_layers, *args)
                where = f"{kind}, {width}x{height}, {n_samples} samples, {k_max} layers"
                np.testing.assert_array_equal(got[3], want[3], err_msg=where)
                held = np.arange(k_max)[None, :] < want[3][:, None]
                for a, b in zip(got[:3], want[:3]):
                    np.testing.assert_array_equal(a[held], b[held], err_msg=where)
                checked += int(want[3].sum())
        self.assertGreater(checked, 50000)  # (the cases do cover layers)

if __name__ == "__main__":
    unittest.main()
