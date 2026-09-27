# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
"""Texture mipmaps and filtered sampling.

A die face drawn from a 48x48 texture often covers only a few pixels on
screen. Point-sampling it picks a few arbitrary texels, so the picture shimmers
as the die turns; sampling a copy pre-shrunk to about the on-screen size
(a mipmap level), blended between the two nearest levels, keeps it steady.
"""
import numpy as np

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


def _bilinear(tex, u, v):
    h, w = tex.shape[:2]
    x = u * w - 0.5
    y = (1.0 - v) * h - 0.5  # texture row 0 is the top (v = 1)
    x0, y0 = np.floor(x), np.floor(y)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    x0 = np.clip(x0.astype(int), 0, w - 1)
    y0 = np.clip(y0.astype(int), 0, h - 1)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    top = tex[y0, x0] * (1 - fx) + tex[y0, x1] * fx
    bottom = tex[y1, x0] * (1 - fx) + tex[y1, x1] * fx
    return top * (1 - fy) + bottom * fy


def sample(levels, u, v, lod):
    """Trilinear sample: bilinear on the two mip levels around `lod`, blended. Returns (N, C)."""
    lod = np.clip(lod, 0.0, len(levels) - 1)
    lo = np.floor(lod).astype(int)
    frac = (lod - lo)[:, None]
    out = np.empty((len(u), levels[0].shape[2]))
    for lvl in np.unique(lo):
        sel = lo == lvl
        a = _bilinear(levels[lvl], u[sel], v[sel])
        if lvl + 1 < len(levels):
            a = a * (1 - frac[sel]) + _bilinear(levels[lvl + 1], u[sel], v[sel]) * frac[sel]
        out[sel] = a
    return out
