# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
"""Shading kernels: lighting a surface (_shade), shading and averaging each pixel's samples (resolve), blending
see-through layers over the solid picture (blend), and fog and outlines (post_effects)."""
import numpy as np
from numba import njit, prange

from .background import SKYBOX, sky_colour
from .raster import barycentric, bit_count, texture_lod
from .shadows import shadow_lookup
from .texture import sample as sample_texture

# Relaxed floating-point rules for shading (Numba's fastmath, less the flags that assume no NaN or infinity, which
# the clamping relies on: min(1, max(0, x)) turning NaN into 0, comparisons false for NaN): sums reordered and fused,
# divisions and powers a little rounder. About a tenth less shading time; colours move by rounding errors, and each
# pixel's arithmetic is the same whichever thread does it. Not for rasterizing, whose coverage is tested exactly.
MATH = {"nsz", "arcp", "contract", "afn", "reassoc"}
# The helpers resolve() and resolve_cells() call per pixel are inlined into them (inline="always"): a call passes
# each of its many arrays as a structure of several words, which cost more than the shading itself (inlined, a
# pixel's shading took a third less time). They compile into each kernel calling them, so the kernels take longer
# to compile (seconds, once: Numba's cache keeps them). So are the texture sampler's (texture.sample and the helpers
# it calls) and raster's barycentric and texture_lod: called, a textured pixel's sampling cost about a tenth of
# resolve_cells's time more.


def _decode_exact(level):
    """A perceived light level (0..1, like an sRGB value) as linear light."""
    return np.where(level <= 0.04045, level / 12.92, ((level + 0.055) / 1.055) ** 2.4)


DECODE_STEPS = 1024
_DECODE = _decode_exact(np.linspace(0.0, 1.0, DECODE_STEPS + 1))
# A highlight x ** power (x the cosine at the half vector) is at most exp(-power (1 - x)), as ln x <= x - 1: where
# power (1 - x) is past this, it is below 1/65,536 and not worked out.
SPEC_CUTOFF = float(np.log(65536.0))
# A shadow lookup this near fully dark or fully lit counts as that, for sharing a cell's lighting (resolve_cells).
SHADOW_EDGE = 1e-9


@njit(cache=True, error_model="numpy", fastmath=MATH)
def _decode(level):
    """A perceived light level as linear light (_decode_exact), from a table with straight lines between its entries:
    under 1e-6 off, a fraction of an 8-bit level, and much cheaper than the power. Clamped to 0..1 first
    (NaN as 0), before it becomes an index."""
    x = min(1.0, max(0.0, level)) * DECODE_STEPS
    i = min(int(x), DECODE_STEPS - 1)
    return _DECODE[i] + (x - i) * (_DECODE[i + 1] - _DECODE[i])


@njit(cache=True, error_model="numpy", fastmath=MATH, inline="always")
def _light(t, b0, b1, b2, inv_w, attrs, lights, shadow_texels, shadow_trans, shadow_mats, shadow_params, pixel_size,
           eye, emissive, specular, shininess, span):
    """The lighting of triangle t at barycentric weights b: Blinn-Phong from every light (rows of
    lights.light_rows()), plus `emissive`. Highlights take the light's colour, their strength times `specular` and
    their exponent `shininess` (the surface's material; 0: the light's own). Lights with a shadow map light only what
    they reach (see shadows.shadow_lookup), tinted by see-through things in the way; their ambient light is
    everywhere. pixel_size is the width of a pixel one unit from the eye, in the world; each shadow lookup's square
    is widened by span - 1 pixels (0 for a pixel of its own; more to answer for a whole cell).

    Returns (the light levels decoded to linear light, r g b; the highlight, r g b, not clamped; how squarely the
    surface faces the eye (the cosine of the angle); the direction a view from the eye is reflected in, x y z;
    whether every shadow lookup came out fully lit or fully dark, untinted: 1.0, else 0.0)."""
    # Perspective-correct interpolation of world position and normal.
    w0, w1, w2 = b0 * inv_w[t, 0], b1 * inv_w[t, 1], b2 * inv_w[t, 2]
    ws = w0 + w1 + w2
    at = attrs[t]

    def lerp(k):
        return (w0 * at[0, k] + w1 * at[1, k] + w2 * at[2, k]) / ws

    nx, ny, nz = lerp(3), lerp(4), lerp(5)
    # A normal (or a view direction) that isn't finite, as at a face too large for its normal to be worked out,
    # counts as none: the facing and reflection returned go unclamped into see-through surfaces' opacity.
    nl = np.sqrt(nx * nx + ny * ny + nz * nz)
    if not nl < np.inf:  # (NaN too)
        nx = ny = nz = nl = 0.0
    nl = max(nl, 1e-12)
    nx, ny, nz = nx / nl, ny / nl, nz / nl
    px, py, pz = lerp(0), lerp(1), lerp(2)
    ex, ey, ez = eye[0] - px, eye[1] - py, eye[2] - pz
    el = np.sqrt(ex * ex + ey * ey + ez * ez)
    if not el < np.inf:
        ex = ey = ez = el = 0.0
    el = max(el, 1e-12)
    ex, ey, ez = ex / el, ey / el, ez / el
    level_r = level_g = level_b = emissive
    spec_r = spec_g = spec_b = 0.0
    uniform = 1.0
    for i in range(lights.shape[0]):
        light = lights[i]
        lx, ly, lz, fade = light[1], light[2], light[3], 1.0
        if light[0] != 0:  # a point light: towards it, fading out smoothly by its range
            lx, ly, lz = lx - px, ly - py, lz - pz
            d = max(np.sqrt(lx * lx + ly * ly + lz * lz), 1e-12)
            lx, ly, lz = lx / d, ly / d, lz / d
            f = max(1.0 - (d / max(light[14], 1e-12)) ** 2, 0.0)
            fade = f * f
            if fade <= 0.0:
                continue
        ndl = nx * lx + ny * ly + nz * lz
        reach = tr = tg = tb = 1.0
        if light[15] >= 0:  # a surface facing away is in its own shadow
            if ndl > 0.0:
                reach, tr, tg, tb = shadow_lookup(shadow_texels, shadow_trans, shadow_mats, shadow_params, int(light[15]),
                                            px, py, pz, nx, ny, nz, ndl, pixel_size * el,
                                            pixel_size * el * (span - 1.0))
                if not ((reach <= SHADOW_EDGE or reach >= 1.0 - SHADOW_EDGE) and tr == 1.0 and tg == 1.0
                        and tb == 1.0):
                    uniform = 0.0  # (NaN too)
            else:
                reach = 0.0
        if tr == 1.0 and tg == 1.0 and tb == 1.0:
            level = (light[10] + light[11] * max(ndl, 0.0) * reach) * fade
            level_r, level_g, level_b = (level_r + light[4] * level, level_g + light[5] * level,
                                         level_b + light[6] * level)
        else:  # light through coloured glass: tinted, channel by channel
            lit = light[11] * max(ndl, 0.0) * reach
            level_r += light[4] * (light[10] + lit * tr) * fade
            level_g += light[5] * (light[10] + lit * tg) * fade
            level_b += light[6] * (light[10] + lit * tb) * fade
        if light[12] > 0.0 and reach > 0.0 and specular > 0.0:
            hx, hy, hz = lx + ex, ly + ey, lz + ez
            hl = max(np.sqrt(hx * hx + hy * hy + hz * hz), 1e-12)
            power = shininess if shininess > 0.0 else light[13]
            nh = (nx * hx + ny * hy + nz * hz) / hl
            if nh > 0.0 and power * (1.0 - nh) < SPEC_CUTOFF:  # (else too faint to show, or NaN)
                spec = light[12] * specular * nh ** power * fade * reach
                spec_r, spec_g, spec_b = (spec_r + light[7] * spec * tr, spec_g + light[8] * spec * tg,
                                          spec_b + light[9] * spec * tb)
    # Light levels are perceived brightness (0.5 looks half as bright), as artists tune them, so they
    # are decoded like any sRGB value; everything after this point works in linear light.
    kr = _decode(level_r)
    kg = kr if level_g == level_r else _decode(level_g)
    kb = kr if level_b == level_r else _decode(level_b)
    ne = nx * ex + ny * ey + nz * ez  # the view reflected about the surface: 2 (n . e) n - e
    return (kr, kg, kb, spec_r, spec_g, spec_b, abs(ne), 2 * ne * nx - ex, 2 * ne * ny - ey, 2 * ne * nz - ez,
            uniform)


@njit(cache=True, error_model="numpy", fastmath=MATH, inline="always")
def _surface(t, b0, b1, b2, xs, ys, inv_w, attrs, chain, lod, texels, levels, first):
    """The colour (r, g, b) and alpha of triangle t's surface at barycentric weights b, before lighting: interpolated
    from its corners, times its texture if it has a mipmap chain (at the mip level raster.texture_lod gives for this
    pixel, from the triangle's corners on screen, xs and ys, plus its bias lod[t])."""
    level = texture_lod(xs, ys, inv_w, attrs, t, b0, b1, b2, levels, first, chain[t]) + lod[t] if chain[t] >= 0 else 0.0
    return _surface_at(t, b0, b1, b2, inv_w, attrs, chain, texels, levels, first, level)


@njit(cache=True, error_model="numpy", fastmath=MATH, inline="always")
def _surface_at(t, b0, b1, b2, inv_w, attrs, chain, texels, levels, first, level):
    """_surface with the texture's mip level given (resolve_cells works it out once per block and triangle)."""
    w0, w1, w2 = b0 * inv_w[t, 0], b1 * inv_w[t, 1], b2 * inv_w[t, 2]
    ws = w0 + w1 + w2
    at = attrs[t]

    def lerp(k):
        return (w0 * at[0, k] + w1 * at[1, k] + w2 * at[2, k]) / ws

    r, g, b, a = lerp(8), lerp(9), lerp(10), lerp(11)
    if chain[t] >= 0:
        tr, tg, tb, ta = sample_texture(texels, levels, first, chain[t], lerp(6), lerp(7), level)
        r, g, b, a = r * tr, g * tg, b * tb, a * ta
    return r, g, b, a


@njit(cache=True, error_model="numpy", fastmath=MATH)
def _shade(t, b0, b1, b2, xs, ys, inv_w, attrs, chain, lod, texels, levels, first, lights, shadow_texels,
           shadow_trans, shadow_mats, shadow_params, pixel_size, eye, emissive, specular, shininess, out, split,
           spec_out):
    """Linear RGB, into out (3,), of triangle t at barycentric weights b: its surface (_surface) lit (_light).
    With split, out gets the surface's own lit colour and spec_out (3,) the highlight, apart (for see-through and
    shiny surfaces, see _reflect_sky).

    Returns the surface's alpha there (its texture's included), how squarely it faces the eye (the cosine
    of the angle), and the direction a view from the eye is reflected in (rx, ry, rz)."""
    kr, kg, kb, spec_r, spec_g, spec_b, facing, rx, ry, rz, _ = _light(
        t, b0, b1, b2, inv_w, attrs, lights, shadow_texels, shadow_trans, shadow_mats, shadow_params, pixel_size, eye,
        emissive, specular, shininess, 1.0)
    r, g, b, a = _surface(t, b0, b1, b2, xs, ys, inv_w, attrs, chain, lod, texels, levels, first)
    _lit(r, g, b, kr, kg, kb, spec_r, spec_g, spec_b, out, split, spec_out)
    return a, facing, rx, ry, rz


@njit(cache=True, error_model="numpy", fastmath=MATH, inline="always")
def _lit(r, g, b, kr, kg, kb, spec_r, spec_g, spec_b, out, split, spec_out):
    """A surface colour r g b under light levels k and highlight spec, into out (and spec_out, with split), as
    _shade gives it."""
    # (Clamped as min(1, max(0, x)), which turns NaN into 0: Numba's max(a, NaN) is a.)
    if split:
        out[0], out[1], out[2] = min(1.0, max(0.0, r * kr)), min(1.0, max(0.0, g * kg)), min(1.0, max(0.0, b * kb))
        spec_out[0], spec_out[1], spec_out[2] = max(0.0, spec_r), max(0.0, spec_g), max(0.0, spec_b)
    else:
        out[0] = min(1.0, max(0.0, r * kr + spec_r))
        out[1] = min(1.0, max(0.0, g * kg + spec_g))
        out[2] = min(1.0, max(0.0, b * kb + spec_b))


@njit(cache=True, error_model="numpy", fastmath=MATH, inline="always")
def _polish(c, s, gloss, rx, ry, rz, sky, sky_colors, sky_faces, sky_texels, sky_levels, sky_first, sky_lod,
            sample_rgb, spec_rgb):
    """A polished surface's sample (sample_rgb[c, s], with its highlight apart in spec_rgb[c]): part (gloss) the
    background seen reflected in direction r, and its highlight on top."""
    sr, sg, sb = sky_colour(sky, sky_colors, sky_faces, sky_texels, sky_levels, sky_first, sky_lod, rx, ry, rz)
    for k, reflected in ((0, sr), (1, sg), (2, sb)):
        sample_rgb[c, s, k] = min(sample_rgb[c, s, k] * (1.0 - gloss) + gloss * reflected + spec_rgb[c, k], 1.0)


@njit(cache=True, error_model="numpy", fastmath=MATH, inline="always")
def _sum_samples(c, tris, depth, pixels, n, tri_inst, ident, contrast, sample_rgb, rgb, cover, near_depth, near_id,
                 more, frame_rgb, frame_alpha, frame_samples):
    """Pixel c's samples, shaded into sample_rgb[c], summed up as resolve() describes."""
    r = g = b = 0.0
    lr = lg = lb = np.inf
    hr = hg = hb = -np.inf
    covered, mixed, best, best_id = 0, False, -1.0, 0
    first_id = ident[tri_inst[tris[c, 0]]] if tris[c, 0] >= 0 else 0
    for s in range(n):
        t = tris[c, s]
        sid = ident[tri_inst[t]] if t >= 0 else 0
        covered += t >= 0
        mixed |= sid != first_id
        if depth[c, s] > best:
            best, best_id = depth[c, s], sid
        vr, vg, vb = sample_rgb[c, s, 0], sample_rgb[c, s, 1], sample_rgb[c, s, 2]
        r, g, b = r + vr, g + vg, b + vb
        lr, lg, lb = min(lr, vr), min(lg, vg), min(lb, vb)
        hr, hg, hb = max(hr, vr), max(hg, vg), max(hb, vb)
    if frame_rgb.shape[0] == 0:
        rgb[c, 0], rgb[c, 1], rgb[c, 2] = r, g, b
        cover[c] = covered
    else:
        p, total = pixels[c], n + frame_samples
        had = frame_samples  # (0 for the first samples: then there is nothing to take back)
        frame_rgb[p, 0] = (frame_rgb[p, 0] * had + r) / total if had else r / n
        frame_rgb[p, 1] = (frame_rgb[p, 1] * had + g) / total if had else g / n
        frame_rgb[p, 2] = (frame_rgb[p, 2] * had + b) / total if had else b / n
        frame_alpha[p] = (frame_alpha[p] * had + covered) / total if had else covered / n
    near_depth[c], near_id[c] = best, best_id
    more[c] = 0 < covered < n or mixed or max(hr - lr, hg - lg, hb - lb) > contrast


@njit(cache=True, error_model="numpy", parallel=True, fastmath=MATH)
def resolve(tris, depth, pixels, width, xs, ys, inv_w, attrs, tri_inst, ident, emissive, specular, shininess, shine,
            chain, lod, texels, levels, first, lights, shadow_texels, shadow_trans, shadow_mats, shadow_params,
            pixel_size, eye, sky, sky_colors, sky_faces, sky_texels, sky_levels, sky_first, sky_lod, contrast,
            sample_rgb, spec_rgb, rgb, cover, near_depth, near_id, more, frame_rgb, frame_alpha, frame_samples):
    """Shade the rasterized samples and sum them up per pixel.

    Each pixel is shaded once per triangle covering it, at the pixel centre, like
    hardware multisampling: the samples only decide coverage. (Texture detail is
    smoothed by mipmapping instead.)

    Writes, per pixel: rgb (summed colour of the covered samples), cover (how many
    samples were covered), the depth and object id of the nearest sample, and
    more: whether the samples disagree (some covered and some not, different
    objects, or colours further apart than `contrast`), so the pixel is worth more
    samples. sample_rgb (M, S, 3) and spec_rgb (M, 3) are scratch space.

    With frame_rgb and frame_alpha (whole-frame, flat; empty for none), the pixel's colour (premultiplied) and
    coverage go there instead of rgb and cover: the average of its samples, or, where frame_samples of them are
    there already, the average of those and these (more samples at an edge). That takes back what is there exactly,
    as sample counts are powers of two (SAMPLE_PATTERNS), so it is what averaging all the sums at once would give.

    Shiny surfaces (shine, per instance) mix that much of the background seen reflected in
    them (sky and its colours etc.: background.sky_colour's arguments) into their colour.
    """
    m, n = tris.shape
    for c in prange(m):
        cx, cy = pixels[c] % width + 0.5, pixels[c] // width + 0.5
        for s in range(n):
            t = tris[c, s]
            if t < 0:
                sample_rgb[c, s, :] = 0.0
                continue
            prev = s
            for s2 in range(s):
                if tris[c, s2] == t:
                    prev = s2
                    break
            if prev < s:
                sample_rgb[c, s, :] = sample_rgb[c, prev, :]
            else:
                b0, b1, b2 = barycentric(xs, ys, t, cx, cy)
                inst = tri_inst[t]
                gloss = shine[inst]
                kr, kg, kb, spec_r, spec_g, spec_b, _, rx, ry, rz, _ = _light(
                    t, b0, b1, b2, inv_w, attrs, lights, shadow_texels, shadow_trans, shadow_mats, shadow_params,
                    pixel_size, eye, emissive[inst], specular[inst], shininess[inst], 1.0)
                r, g, b, _ = _surface(t, b0, b1, b2, xs, ys, inv_w, attrs, chain, lod, texels, levels, first)
                _lit(r, g, b, kr, kg, kb, spec_r, spec_g, spec_b, sample_rgb[c, s], gloss > 0.0, spec_rgb[c])
                if gloss > 0.0:  # polished: part the background reflected in it, and its highlight on top
                    _polish(c, s, gloss, rx, ry, rz, sky, sky_colors, sky_faces, sky_texels, sky_levels, sky_first,
                            sky_lod, sample_rgb, spec_rgb)
        _sum_samples(c, tris, depth, pixels, n, tri_inst, ident, contrast, sample_rgb, rgb, cover, near_depth, near_id,
                     more, frame_rgb, frame_alpha, frame_samples)


@njit(cache=True, error_model="numpy", parallel=True, fastmath=MATH)
def resolve_cells(tris, depth, pixels, width, xs, ys, inv_w, attrs, tri_inst, ident, emissive, specular, shininess,
                  shine, chain, lod, texels, levels, first, lights, shadow_texels, shadow_trans, shadow_mats,
                  shadow_params, pixel_size, eye, sky, sky_colors, sky_faces, sky_texels, sky_levels, sky_first,
                  sky_lod, contrast, sample_rgb, spec_rgb, rgb, cover, near_depth, near_id, more, frame_rgb,
                  frame_alpha, frame_samples, cell_w, cell_h, coarse):
    """resolve() for a whole frame (pixels: every pixel, in order; tris has a row for each), shading less often than
    every pixel: in blocks of cell_w x cell_h pixels (a terminal cell's, whose pixels end up as two colours anyway),
    each triangle in a block is lit once (_light), at the mean of the centres of the block's pixels it covers.

    Where a shadow's edge may cross the block (a shadow lookup smoothed over the whole block comes out neither fully
    lit nor fully dark, or tinted), the block's pixels are lit one by one, as resolve() lights them: shadows keep
    their edges, and fully lit or fully dark blocks, the most, are lit once. The surface's colour and texture are
    still worked out per pixel, at a texture mip level worked out once per block and triangle (at the same point,
    with how it changes along x and y, for each pixel's); with `coarse`, they too are once per block and triangle
    (cheaper, and coarser: texture detail inside a cell is lost), and shadow edges are not looked for.

    Writes what resolve() writes. Blocks at the frame's right and bottom edges may be smaller. Loops in parallel
    over rows of blocks, each writing only its own pixels.
    """
    m, n = tris.shape
    height = m // max(width, 1)
    cell_w, cell_h = max(cell_w, 1), max(cell_h, 1)
    cols, rows = (width + cell_w - 1) // cell_w, (height + cell_h - 1) // cell_h
    span = 1.0 if coarse else np.sqrt(cell_w * cell_w + cell_h * cell_h) + 1.0  # (the block's diagonal, a pixel more)
    most = cell_w * cell_h * n  # the most different triangles a block can hold
    for row in prange(rows):
        # Each block's triangles: which, where they are (summed pixel centres, how many; then their mean), their
        # lighting (light levels, highlight, reflected view, whether shared: _light's), coarse, their surface colour,
        # and their texture's mip level (and its change along x and y).
        block_tris = np.empty(most, np.int64)
        block = np.empty((most, 18))
        y0, y1 = row * cell_h, min(row * cell_h + cell_h, height)
        for col in range(cols):
            x0, x1 = col * cell_w, min(col * cell_w + cell_w, width)
            k = 0
            for y in range(y0, y1):
                for x in range(x0, x1):
                    c = y * width + x
                    for s in range(n):
                        t = tris[c, s]
                        if t < 0:
                            continue
                        repeated = False
                        for s2 in range(s):
                            if tris[c, s2] == t:
                                repeated = True
                                break
                        if repeated:
                            continue
                        j = 0
                        while j < k and block_tris[j] != t:
                            j += 1
                        if j == k:
                            block_tris[k] = t
                            block[k, 0] = block[k, 1] = block[k, 2] = 0.0
                            k += 1
                        block[j, 0] += x + 0.5
                        block[j, 1] += y + 0.5
                        block[j, 2] += 1.0
            for j in range(k):
                t, inst = block_tris[j], tri_inst[block_tris[j]]
                mx, my = block[j, 0] / block[j, 2], block[j, 1] / block[j, 2]
                b0, b1, b2 = barycentric(xs, ys, t, mx, my)
                (block[j, 3], block[j, 4], block[j, 5], block[j, 6], block[j, 7], block[j, 8], _, block[j, 9],
                 block[j, 10], block[j, 11], block[j, 2]) = _light(
                    t, b0, b1, b2, inv_w, attrs, lights, shadow_texels, shadow_trans, shadow_mats, shadow_params,
                    pixel_size, eye, emissive[inst], specular[inst], shininess[inst], span)
                # The texture's mip level here, and how it changes a pixel to the right and a pixel down, for each
                # pixel's own (three texture_lod calls instead of one per pixel; within a level of 8-bit colour).
                block[j, 0], block[j, 1] = mx, my
                block[j, 15] = block[j, 16] = block[j, 17] = 0.0
                if chain[t] >= 0 and not coarse:
                    level = texture_lod(xs, ys, inv_w, attrs, t, b0, b1, b2, levels, first, chain[t])
                    block[j, 15] = level + lod[t]
                    c0, c1, c2 = barycentric(xs, ys, t, mx + 1.0, my)
                    block[j, 16] = texture_lod(xs, ys, inv_w, attrs, t, c0, c1, c2, levels, first, chain[t]) - level
                    c0, c1, c2 = barycentric(xs, ys, t, mx, my + 1.0)
                    block[j, 17] = texture_lod(xs, ys, inv_w, attrs, t, c0, c1, c2, levels, first, chain[t]) - level
                if coarse:
                    block[j, 2] = 1.0
                    block[j, 12], block[j, 13], block[j, 14], _ = _surface(t, b0, b1, b2, xs, ys, inv_w, attrs, chain,
                                                                           lod, texels, levels, first)
            for y in range(y0, y1):
                for x in range(x0, x1):
                    c = y * width + x
                    for s in range(n):
                        t = tris[c, s]
                        if t < 0:
                            sample_rgb[c, s, :] = 0.0
                            continue
                        prev = s
                        for s2 in range(s):
                            if tris[c, s2] == t:
                                prev = s2
                                break
                        if prev < s:
                            sample_rgb[c, s, :] = sample_rgb[c, prev, :]
                            continue
                        j = 0
                        while block_tris[j] != t:
                            j += 1
                        inst = tri_inst[t]
                        gloss = shine[inst]
                        kr, kg, kb, spec_r, spec_g, spec_b = (block[j, 3], block[j, 4], block[j, 5], block[j, 6],
                                                              block[j, 7], block[j, 8])
                        rx, ry, rz = block[j, 9], block[j, 10], block[j, 11]
                        if not coarse:
                            b0, b1, b2 = barycentric(xs, ys, t, x + 0.5, y + 0.5)
                            if block[j, 2] == 0.0:  # a shadow's edge may be here: lit at this pixel, as resolve() does
                                kr, kg, kb, spec_r, spec_g, spec_b, _, rx, ry, rz, _ = _light(
                                    t, b0, b1, b2, inv_w, attrs, lights, shadow_texels, shadow_trans, shadow_mats,
                                    shadow_params, pixel_size, eye, emissive[inst], specular[inst], shininess[inst], 1.0)
                            level = block[j, 15] + block[j, 16] * (x + 0.5 - block[j, 0]) + block[j, 17] * (
                                y + 0.5 - block[j, 1])
                            r, g, b, _ = _surface_at(t, b0, b1, b2, inv_w, attrs, chain, texels, levels, first, level)
                        else:
                            r, g, b = block[j, 12], block[j, 13], block[j, 14]
                        _lit(r, g, b, kr, kg, kb, spec_r, spec_g, spec_b, sample_rgb[c, s], gloss > 0.0, spec_rgb[c])
                        if gloss > 0.0:
                            _polish(c, s, gloss, rx, ry, rz, sky, sky_colors, sky_faces, sky_texels, sky_levels,
                                    sky_first, sky_lod, sample_rgb, spec_rgb)
                    _sum_samples(c, tris, depth, pixels, n, tri_inst, ident, contrast, sample_rgb, rgb, cover,
                                 near_depth, near_id, more, frame_rgb, frame_alpha, frame_samples)


@njit(cache=True, error_model="numpy", parallel=True, fastmath=MATH)
def blend(pixels, rgb, alpha, depth, ids, layer_count, layer_depth, layer_tri, layer_cover, n_samples, width, xs, ys,
          inv_w, attrs, tri_inst, ident, emissive, specular, shininess, shine, chain, lod, texels, levels, first, lights,
          shadow_texels, shadow_trans, shadow_mats, shadow_params, pixel_size, eye, sky, sky_colors, sky_faces,
          sky_texels, sky_levels, sky_first, sky_lod, scratch):
    """Blend the see-through layers (raster.rasterize_layers) of each of `pixels` (flat indices of those that
    have any: shared out evenly among threads, where the pixels themselves may be bunched together) over
    its colour and coverage (flat framebuffer arrays, premultiplied by coverage), farthest first, and give
    the pixel the depth and object id of the nearest.

    Each layer is shaded at the pixel centre and covers the fraction of the pixel's n_samples it
    covered (layer_cover: a bit per sample). Its opacity is its interpolated alpha, raised towards 1
    where the surface is seen edge-on (Fresnel's law, as glass reflects more at a slant), the extra
    showing the background reflected in it; its highlights show at full strength however clear it
    is, as on glass; both fade out below an alpha of 0.25, so that something faded to nothing
    disappears. sky < 0: reflections are off, and the extra shows the surface's own colour. The
    nearest layer that shows at all gives the pixel its depth and id. A layer that is shiny as well
    (shine, per instance) takes that much of the reflected background into its own colour. scratch
    (len(pixels), 9) is room for the shading, a row for each.
    """
    for p in prange(pixels.shape[0]):
        c = pixels[p]
        n = layer_count[c]
        cx, cy = c % width + 0.5, c // width + 0.5
        r, g, b, a = rgb[c, 0], rgb[c, 1], rgb[c, 2], alpha[c]
        shows = -1  # the nearest layer that shows so far
        for layer in range(n - 1, -1, -1):
            t = layer_tri[c, layer]
            b0, b1, b2 = barycentric(xs, ys, t, cx, cy)
            clear, facing, rx, ry, rz = _shade(t, b0, b1, b2, xs, ys, inv_w, attrs, chain, lod, texels, levels,
                                               first, lights, shadow_texels, shadow_trans, shadow_mats, shadow_params,
                                               pixel_size, eye, emissive[tri_inst[t]], specular[tri_inst[t]],
                                               shininess[tri_inst[t]], scratch[p, 0:3], True, scratch[p, 3:6])
            scratch[p, 6], scratch[p, 7], scratch[p, 8] = sky_colour(sky, sky_colors, sky_faces, sky_texels,
                                                                     sky_levels, sky_first, sky_lod, rx, ry, rz)
            polish = shine[tri_inst[t]]  # a shiny see-through surface: part of its colour the background reflected
            if polish > 0.0:
                for k in range(3):
                    scratch[p, k] = scratch[p, k] * (1.0 - polish) + polish * scratch[p, 6 + k]
            clear = min(1.0, max(0.0, clear))  # (NaN as 0)
            gloss = min(4.0 * clear, 1.0)
            fresnel = 0.04 + 0.96 * (1.0 - min(facing, 1.0)) ** 5
            # What it gains in opacity at a slant is the background reflected in it (unless reflections are off).
            mirrored = (1.0 - clear) * fresnel * gloss
            opacity = clear + mirrored
            if sky < 0:
                mirrored = 0.0
            part = bit_count(layer_cover[c, layer]) / n_samples
            over = opacity * part
            r = r * (1.0 - over) + part * ((opacity - mirrored) * scratch[p, 0] + mirrored * scratch[p, 6]
                                            + gloss * scratch[p, 3])
            g = g * (1.0 - over) + part * ((opacity - mirrored) * scratch[p, 1] + mirrored * scratch[p, 7]
                                            + gloss * scratch[p, 4])
            b = b * (1.0 - over) + part * ((opacity - mirrored) * scratch[p, 2] + mirrored * scratch[p, 8]
                                            + gloss * scratch[p, 5])
            a = a * (1.0 - over) + over
            if over > 0.0:
                shows = layer
        rgb[c, 0], rgb[c, 1], rgb[c, 2] = min(r, a), min(g, a), min(b, a)
        alpha[c] = a
        if shows >= 0:  # what the pixel shows first (a layer faded to nothing in front of it shows nothing)
            depth[c], ids[c] = layer_depth[c, shows], ident[tri_inst[layer_tri[c, shows]]]


@njit(cache=True, error_model="numpy")
def _plane_depth(d, offset):
    """A depth (1/w) measured from the camera's plane rather than from the eye offset behind it: an orthographic
    view is drawn as a perspective one from far back (see Camera), and its depths count from the plane the camera
    is on. d itself for a perspective view (offset 0); 0 (empty) stays 0."""
    if offset == 0.0:
        return d
    return 1.0 / (1.0 / d - offset)


@njit(cache=True, error_model="numpy", parallel=True)
def post_effects(rgb, alpha, depth, fog, fog_start, fog_end, fog_rgb, fog_clear, tan_x, tan_y, offset, outline, sky,
                 sky_colors, sky_basis, sky_faces, sky_texels, sky_levels, sky_first, haze_lod):
    """Outlines, then fog, applied in place to a framebuffer's arrays (see Renderer for both). Depths are measured
    from `offset` in front of the eye (see _plane_depth).

    fog_end > 0: fog in the world (background.Fog): each surface fades from fog_start to fog_end, by its
    distance from the eye (tan_x, tan_y: the tangents of half the view's width and height, which give each
    pixel's direction), into fog_rgb (linear), or, with fog_clear, into whatever is behind it (it loses its
    coverage, so the background, or the terminal's own, shows through). Otherwise `fog` dims surfaces by how far
    back they sit within the picture's depth range (depth cueing).

    Behind a sky box (sky == SKYBOX: the background's kind, colours, basis and textures as fill_background takes
    them), fog_clear fades surfaces instead into the sky box blurred (sampled at mip level haze_lod) in the
    direction they are seen in, and they keep their coverage: haze hides the sky's fine detail (stars, say),
    which would otherwise show through a fogged wall as if it were glass.
    """
    h, w = depth.shape
    if outline:
        # Darken pixels just behind a depth edge. 1/w is linear across any plane on screen, so its
        # second difference is ~0 on flat and gently curved surfaces and large where one surface
        # passes in front of another. It is positive on the far side of the edge, which is the side
        # darkened, so shapes in front keep their full size.
        for y in prange(h):
            for x in range(w):
                d = _plane_depth(depth[y, x], offset)
                if alpha[y, x] <= 0:
                    continue
                edge = (0 < y < h - 1 and alpha[y - 1, x] > 0 and alpha[y + 1, x] > 0
                        and _plane_depth(depth[y - 1, x], offset) + _plane_depth(depth[y + 1, x], offset) - 2 * d
                        > 0.08 * d)
                edge = edge or (0 < x < w - 1 and alpha[y, x - 1] > 0 and alpha[y, x + 1] > 0
                                and _plane_depth(depth[y, x - 1], offset) + _plane_depth(depth[y, x + 1], offset)
                                - 2 * d > 0.08 * d)
                if edge:
                    rgb[y, x, :] *= 1.0 - outline
    if fog_end > 0.0:
        span = max(fog_end - fog_start, 1e-9)
        for y in prange(h):
            ny = (1.0 - (y + 0.5) / h * 2.0) * tan_y
            for x in range(w):
                a, d = alpha[y, x], _plane_depth(depth[y, x], offset)
                if not (a > 0.0 and d > 0.0):
                    continue
                nx = ((x + 0.5) / w * 2.0 - 1.0) * tan_x
                distance = np.sqrt(1.0 + nx * nx + ny * ny) / d  # 1/d is the depth along the view
                f = min(max((distance - fog_start) / span, 0.0), 1.0)
                if not f > 0.0:
                    continue
                if fog_clear and sky == SKYBOX:
                    sx, sy = (x + 0.5) / w * 2.0 - 1.0, 1.0 - (y + 0.5) / h * 2.0
                    dx = sky_basis[0, 0] + sx * sky_basis[1, 0] + sy * sky_basis[2, 0]
                    dy = sky_basis[0, 1] + sx * sky_basis[1, 1] + sy * sky_basis[2, 1]
                    dz = sky_basis[0, 2] + sx * sky_basis[1, 2] + sy * sky_basis[2, 2]
                    dl = max(np.sqrt(dx * dx + dy * dy + dz * dz), 1e-12)
                    hr, hg, hb = sky_colour(sky, sky_colors, sky_faces, sky_texels, sky_levels, sky_first, haze_lod,
                                            dx / dl, dy / dl, dz / dl)
                    for k, haze in ((0, hr), (1, hg), (2, hb)):
                        rgb[y, x, k] = rgb[y, x, k] * (1.0 - f) + f * a * haze
                elif fog_clear:
                    alpha[y, x] = a * (1.0 - f)
                    for k in range(3):
                        rgb[y, x, k] *= 1.0 - f
                else:
                    for k in range(3):
                        rgb[y, x, k] = rgb[y, x, k] * (1.0 - f) + f * a * fog_rgb[k]
    elif fog:
        # Dim pixels in proportion to how far back they sit within the scene's depth range (of the pixels
        # drawn that have a depth): each row's range into its own slot, then all of them.
        row_near, row_far = np.empty(h), np.empty(h)
        for y in prange(h):
            lo, hi = np.inf, -np.inf
            for x in range(w):
                if alpha[y, x] > 0 and depth[y, x] > 0:
                    d = _plane_depth(depth[y, x], offset)
                    lo, hi = min(lo, 1.0 / d), max(hi, 1.0 / d)
            row_near[y], row_far[y] = lo, hi
        near, far = np.inf, -np.inf
        for y in range(h):
            near, far = min(near, row_near[y]), max(far, row_far[y])
        span = max(far - near, 0.25 * near)
        for y in prange(h):
            for x in range(w):
                if alpha[y, x] > 0 and depth[y, x] > 0:
                    rgb[y, x, :] *= 1.0 - fog * (1.0 / _plane_depth(depth[y, x], offset) - near) / span
