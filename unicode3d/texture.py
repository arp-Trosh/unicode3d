# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
"""Texture mipmaps and filtered sampling.

A die face drawn from a 48x48 texture often covers only a few pixels on
screen. Point-sampling it picks a few arbitrary texels, so the picture shimmers
as the die turns; sampling a copy pre-shrunk to about the on-screen size
(a mipmap level), blended between the two nearest levels, keeps it steady.

Textures with an alpha channel (opacity) are kept with their colour premultiplied
by it, so that shrinking them averages the colour of what is there, not of the
clear parts: a leaf's edge stays green rather than fading to black.
"""
import io

import numpy as np
from numba import njit
from PIL import Image

from .color import srgb_to_linear


CUTOUT, BLEND = 1, 2  # kinds of alpha in a texture (see alpha_kind); 0 is none (solid)
MAX_TEXTURE = 1024     # load_image's default limit on a texture's width and height, in texels


def load_image(path, max_size=MAX_TEXTURE):
    """An image file (PNG, JPEG, or anything else Pillow reads) as a texture: (H, W, 3) colours, 0..1 sRGB, or
    (H, W, 4) with alpha where the image has transparency. path is a file name, an open binary file, or the
    file's contents (bytes). Images wider or taller than max_size (None: no limit) are shrunk to fit: a terminal
    shows few pixels, and a 4096x4096 texture would take about a gigabyte once mipmapped in floats. Raises
    OSError for a file Pillow can't read, and ValueError for one too large to decode (beyond Pillow's limit on
    pixels, which guards against decompression bombs)."""
    if isinstance(path, (bytes, bytearray, memoryview)):
        path = io.BytesIO(path)
    try:
        with Image.open(path) as image:
            image.load()
            alpha = image.mode in ("RGBA", "LA", "PA", "RGBa", "La") or "transparency" in image.info
            if image.mode in ("I", "I;16", "I;16B", "I;16L", "F"):  # 16-bit or float greyscale: scale to 8 bits
                grey = np.nan_to_num(np.asarray(image, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
                image = Image.fromarray(np.clip(grey * (255.0 / max(grey.max(), 1.0)), 0, 255).astype(np.uint8))
            image = image.convert("RGBA" if alpha else "RGB")
            if max_size and max(image.size) > max_size:
                scale = max_size / max(image.size)
                image = image.resize((max(round(image.width * scale), 1), max(round(image.height * scale), 1)),
                                     Image.Resampling.BOX)
            return np.asarray(image, dtype=float) / 255.0
    except Image.DecompressionBombError as e:  # (not an OSError: the model loaders would not take it as a warning)
        raise ValueError(f"image too large to load ({e})") from None


def build_mipmaps(texture):
    """Mipmap chain of a texture, finest first, each level (H, W, C) in linear light.

    texture: (H, W) brightness multipliers, (H, W, 3) colours, both 0..1 sRGB, or (H, W, 4)
    colours and alpha (opacity, 0..1, not gamma-encoded). With alpha, levels are (H, W, 4),
    colour premultiplied by alpha. NaN counts as 0 (black, or clear), so that it never reaches the picture.
    """
    texture = np.asarray(texture, dtype=float)
    if texture.ndim not in (2, 3) or not texture.size:  # (sampling reads at least one texel of every level)
        raise ValueError(f"a texture must be (H, W) or (H, W, channels) with at least one texel, not {texture.shape}")
    texture = np.nan_to_num(texture, nan=0.0, posinf=1.0, neginf=0.0)
    if texture.ndim == 3 and texture.shape[2] == 4:
        alpha = np.clip(texture[..., 3:], 0.0, 1.0)
        level = np.concatenate([srgb_to_linear(texture[..., :3]) * alpha, alpha], axis=2)
    else:
        level = srgb_to_linear(texture)
    if level.ndim == 2:
        level = level[..., None]
    levels = [level]
    while max(level.shape[:2]) > 1:
        h, w = level.shape[:2]
        if h % 2 or w % 2:  # repeat the last row/column so it halves evenly
            level = np.pad(level, ((0, h % 2), (0, w % 2), (0, 0)), mode="edge")
        level = 0.25 * (level[0::2, 0::2] + level[1::2, 0::2] + level[0::2, 1::2] + level[1::2, 1::2])
        levels.append(level)
    return levels


def alpha_kind(chain):
    """What a texture's alpha (its mipmap chain's) is for: 0 if it has none (solid all over), CUTOUT if it
    is mostly solid or clear, with at most soft edges between (leaves, a fence, lettering), BLEND if more
    than a tenth of it is partly see-through (stained glass): those are drawn with the see-through surfaces."""
    level = chain[0]
    if level.shape[2] < 4 or level[..., 3].min() >= 1.0:
        return 0
    alpha = level[..., 3]
    return BLEND if ((alpha > 0.1) & (alpha < 0.9)).mean() > 0.1 else CUTOUT


def pack(chains):
    """Mipmap chains (as build_mipmaps gives them) in flat arrays, for sample().

    Returns texels (N, 4): every level's texels, row by row, as colours premultiplied by alpha and
    alpha (brightness textures repeat their one channel; textures without alpha have alpha 1);
    levels (L, 3): each level's [first texel, height, width]; first (K + 1,): chain k's levels are
    levels[first[k]:first[k + 1]].
    """
    texels, levels, first, offset = [np.zeros((0, 4))], [], [0], 0
    for chain in chains:
        for level in chain:
            h, w, c = level.shape
            rgba = np.ones((h, w, 4))
            rgba[..., :3] = level[..., :3] if c >= 3 else level[..., :1]
            if c == 4:
                rgba[..., 3] = level[..., 3]
            texels.append(rgba.reshape(-1, 4))
            levels.append((offset, h, w))
            offset += h * w
        first.append(len(levels))
    return (np.ascontiguousarray(np.concatenate(texels)), np.array(levels, dtype=np.int64).reshape(-1, 3),
            np.array(first, dtype=np.int64))


@njit(cache=True, error_model="numpy")
def clamp_index(v, n):
    """A whole number v (a float) as an index into n items: v kept within 0..n-1, and 0 for NaN.

    Converting NaN, an infinity or anything beyond the range of int64 to an int gives
    an undefined result, and an index from one reads or writes outside the array.
    """
    if v >= n - 1:
        return n - 1
    if v > 0.0:
        return int(v)
    return 0


@njit(cache=True, error_model="numpy")
def repeat(u):
    """A texture coordinate beyond 0..1 brought back into it, so that textures repeat (tile) over faces whose uv
    go further (a floor tiled ten times: u from 0 to 10). Coordinates within 0..1 are kept as they are, so
    textures meant to fit a face exactly are clamped at its edges, not blended with the opposite edge. NaN and
    infinities become 0, so that a texel is still picked (and the colour stays a number)."""
    if u >= 0.0 and u <= 1.0:
        return u
    f = u - np.floor(u)
    return f if f >= 0.0 else 0.0  # (an infinity gives NaN, which fails the comparison too)


@njit(cache=True, error_model="numpy")
def _bilinear(texels, levels, level, u, v):
    offset, h, w = levels[level, 0], levels[level, 1], levels[level, 2]
    u, v = repeat(u), repeat(v)
    x = u * w - 0.5
    y = (1.0 - v) * h - 0.5  # texture row 0 is the top (v = 1)
    fx, fy = x - np.floor(x), y - np.floor(y)
    x0 = clamp_index(np.floor(x), w)
    y0 = clamp_index(np.floor(y), h)
    x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
    a, b = texels[offset + y0 * w + x0], texels[offset + y0 * w + x1]
    c, d = texels[offset + y1 * w + x0], texels[offset + y1 * w + x1]
    r = (a[0] * (1 - fx) + b[0] * fx) * (1 - fy) + (c[0] * (1 - fx) + d[0] * fx) * fy
    g = (a[1] * (1 - fx) + b[1] * fx) * (1 - fy) + (c[1] * (1 - fx) + d[1] * fx) * fy
    bl = (a[2] * (1 - fx) + b[2] * fx) * (1 - fy) + (c[2] * (1 - fx) + d[2] * fx) * fy
    al = (a[3] * (1 - fx) + b[3] * fx) * (1 - fy) + (c[3] * (1 - fx) + d[3] * fx) * fy
    return r, g, bl, al


@njit(cache=True, error_model="numpy")
def sample(texels, levels, first, chain, u, v, lod):
    """Trilinear sample of chain `chain` of pack()'s arrays: bilinear on the two mip levels around `lod`,
    blended. Returns (r, g, b, alpha), the colour not premultiplied (as it was painted)."""
    n = first[chain + 1] - first[chain]
    if not lod > 0.0:  # (NaN too)
        lod = 0.0
    elif lod > n - 1.0:
        lod = n - 1.0
    lo = int(np.floor(lod))
    frac = lod - lo
    r, g, b, a = _bilinear(texels, levels, first[chain] + lo, u, v)
    if lo + 1 < n:
        r2, g2, b2, a2 = _bilinear(texels, levels, first[chain] + lo + 1, u, v)
        r, g, b = r * (1 - frac) + r2 * frac, g * (1 - frac) + g2 * frac, b * (1 - frac) + b2 * frac
        a = a * (1 - frac) + a2 * frac
    if a < 1.0 - 1e-9:  # (a solid texture's alpha, blended, can come out a rounding error below 1)
        k = 1.0 / max(a, 1e-9)
        r, g, b = min(r * k, 1.0), min(g * k, 1.0), min(b * k, 1.0)
    return r, g, b, a


@njit(cache=True, error_model="numpy")
def sample_level(texels, levels, first, chain, u, v, lod):
    """Like sample(), but bilinear on the one mip level nearest `lod`: cheaper, for shadow maps."""
    n = first[chain + 1] - first[chain]
    r, g, b, a = _bilinear(texels, levels, first[chain] + clamp_index(np.floor(lod + 0.5), n), u, v)
    if a < 1.0 - 1e-9:
        k = 1.0 / max(a, 1e-9)
        r, g, b = min(r * k, 1.0), min(g * k, 1.0), min(b * k, 1.0)
    return r, g, b, a


@njit(cache=True, error_model="numpy")
def sample_alpha(texels, levels, first, chain, u, v, lod):
    """Just the alpha of chain `chain` at (u, v), bilinear on the mip level nearest `lod` (for cut-outs)."""
    n = first[chain + 1] - first[chain]
    level = first[chain] + clamp_index(np.floor(lod + 0.5), n)
    offset, h, w = levels[level, 0], levels[level, 1], levels[level, 2]
    u, v = repeat(u), repeat(v)
    x = u * w - 0.5
    y = (1.0 - v) * h - 0.5
    fx, fy = x - np.floor(x), y - np.floor(y)
    x0 = clamp_index(np.floor(x), w)
    y0 = clamp_index(np.floor(y), h)
    x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
    return ((texels[offset + y0 * w + x0, 3] * (1 - fx) + texels[offset + y0 * w + x1, 3] * fx) * (1 - fy)
            + (texels[offset + y1 * w + x0, 3] * (1 - fx) + texels[offset + y1 * w + x1, 3] * fx) * fy)
