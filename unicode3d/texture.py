# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
"""Texture mipmaps and filtered sampling.

A die face drawn from a 48x48 texture often covers only a few pixels on
screen. Point-sampling it picks a few arbitrary texels, so the picture shimmers
as the die turns; sampling a copy pre-shrunk to about the on-screen size
(a mipmap level), blended between the two nearest levels, keeps it steady.
"""
import numpy as np
from numba import njit

from .color import srgb_to_linear


def build_mipmaps(texture):
    """Mipmap chain of a texture, finest first, each level (H, W, C) in linear light.

    texture: (H, W) brightness multipliers or (H, W, 3) colours, both 0..1 sRGB.
    """
    level = srgb_to_linear(np.asarray(texture, dtype=float))
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


def pack(chains):
    """Mipmap chains (as build_mipmaps gives them) in flat arrays, for sample().

    Returns texels (N, 3): every level's texels, row by row, as colours (brightness
    textures repeat their one channel); levels (L, 3): each level's [first texel,
    height, width]; first (K + 1,): chain k's levels are levels[first[k]:first[k + 1]].
    """
    texels, levels, first, offset = [np.zeros((0, 3))], [], [0], 0
    for chain in chains:
        for level in chain:
            h, w, c = level.shape
            texels.append(np.broadcast_to(level, (h, w, 3)).reshape(-1, 3))
            levels.append((offset, h, w))
            offset += h * w
        first.append(len(levels))
    return (np.ascontiguousarray(np.concatenate(texels)), np.array(levels, dtype=np.int64).reshape(-1, 3),
            np.array(first, dtype=np.int64))


@njit(cache=True)
def _bilinear(texels, levels, level, u, v):
    offset, h, w = levels[level, 0], levels[level, 1], levels[level, 2]
    x = u * w - 0.5
    y = (1.0 - v) * h - 0.5  # texture row 0 is the top (v = 1)
    fx, fy = x - np.floor(x), y - np.floor(y)
    x0 = min(max(int(np.floor(x)), 0), w - 1)
    y0 = min(max(int(np.floor(y)), 0), h - 1)
    x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
    a, b = texels[offset + y0 * w + x0], texels[offset + y0 * w + x1]
    c, d = texels[offset + y1 * w + x0], texels[offset + y1 * w + x1]
    r = (a[0] * (1 - fx) + b[0] * fx) * (1 - fy) + (c[0] * (1 - fx) + d[0] * fx) * fy
    g = (a[1] * (1 - fx) + b[1] * fx) * (1 - fy) + (c[1] * (1 - fx) + d[1] * fx) * fy
    bl = (a[2] * (1 - fx) + b[2] * fx) * (1 - fy) + (c[2] * (1 - fx) + d[2] * fx) * fy
    return r, g, bl


@njit(cache=True)
def sample(texels, levels, first, chain, u, v, lod):
    """Trilinear sample of chain `chain` of pack()'s arrays: bilinear on the two mip levels around `lod`, blended."""
    n = first[chain + 1] - first[chain]
    lod = min(max(lod, 0.0), n - 1.0)
    lo = int(np.floor(lod))
    frac = lod - lo
    r, g, b = _bilinear(texels, levels, first[chain] + lo, u, v)
    if lo + 1 < n:
        r2, g2, b2 = _bilinear(texels, levels, first[chain] + lo + 1, u, v)
        r, g, b = r * (1 - frac) + r2 * frac, g * (1 - frac) + g2 * frac, b * (1 - frac) + b2 * frac
    return r, g, b
