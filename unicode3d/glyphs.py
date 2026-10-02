# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
"""Turning a pixel framebuffer into terminal cells: a character plus two colours each.

A cell can show exactly two colours, but the character decides how the cell is
split between them. Each glyph set covers a grid of sub-pixels per cell with
one character for every way of splitting that grid in two:

  half     1x2  ▀ ▄           every font has them
  quad     2x2  ▘ ▝ ▖ ▗ ▚ ...  Block Elements, in every font with ▀
  sextant  2x3  🬀 🬁 🬂 ...     Symbols for Legacy Computing (Unicode 13): in Cascadia (Windows
                              Terminal's font) and Iosevka, drawn by kitty and WezTerm; missing
                              from older fonts
  ascii    1x2  " .:-=+*#%@"  a brightness ramp, for terminals without Unicode

For each cell the split with the least colour error wins (the approach chafa
uses), so edges land on the right sub-pixel while flat areas stay solid.
"""
from dataclasses import dataclass

import numpy as np
from numba import njit, prange

from .color import linear_to_srgb, luminance
from .threads import kernel_lock

RAMP = " .:-=+*#%@"  # dark -> bright; the first character is the empty background
ALPHA_WEIGHT = 0.5   # how much coverage counts, against colour, when choosing a split
MIN_ALPHA = 0.4      # a side of a cell covered less than this shows the terminal background
MIN_SPLIT = 0.02     # a split must cut the colour error by about this much (RMS, linear) to beat a solid cell


def _sextant_chars():
    chars = []
    for v in range(64):
        if v == 0:
            chars.append(" ")
        elif v == 21:
            chars.append("▌")  # left column
        elif v == 42:
            chars.append("▐")  # right column
        elif v == 63:
            chars.append("█")
        else:
            chars.append(chr(0x1FB00 + v - 1 - (v > 21) - (v > 42)))
    return chars


@dataclass(frozen=True)
class GlyphSet:
    """Characters indexed by a bit mask of the cell's sub-pixels (bit i: pixel i, row by row) in the foreground."""
    name: str
    cell_pixels: tuple  # (columns, rows) of sub-pixels per cell
    chars: tuple

    @property
    def pixel_count(self):
        return self.cell_pixels[0] * self.cell_pixels[1]


GLYPH_SETS = {
    "half": GlyphSet("half", (1, 2), (" ", "▀", "▄", "█")),
    "quad": GlyphSet("quad", (2, 2), tuple(" ▘▝▀▖▌▞▛▗▚▐▜▄▙▟█")),
    "sextant": GlyphSet("sextant", (2, 3), tuple(_sextant_chars())),
    "ascii": GlyphSet("ascii", (1, 2), ()),
}
GLYPH_MODES = tuple(GLYPH_SETS)


@dataclass
class Cells:
    """A framebuffer as terminal cells.

    chars: (H, W) str. fg, bg: (H, W, 3) linear RGB. fg_on, bg_on: (H, W) bool,
    False where that colour is the terminal's own (nothing was drawn there).
    """
    chars: np.ndarray
    fg: np.ndarray
    bg: np.ndarray
    fg_on: np.ndarray
    bg_on: np.ndarray


def _cell_pixels(fb, glyphs, background):
    """(H, W, P, 3) colours composited over `background` and (H, W, P) coverage, grouped by cell."""
    pw, ph = glyphs.cell_pixels
    h, w = fb.height // ph, fb.width // pw
    rgb = fb.rgb[:h * ph, :w * pw]
    alpha = fb.alpha[:h * ph, :w * pw]
    if background is not None:
        rgb = rgb + (1.0 - alpha)[..., None] * background
    rgb = rgb.reshape(h, ph, w, pw, 3).transpose(0, 2, 1, 3, 4).reshape(h, w, ph * pw, 3)
    alpha = alpha.reshape(h, ph, w, pw).transpose(0, 2, 1, 3).reshape(h, w, ph * pw)
    return rgb, alpha


def _masks(p):
    """Every split of p sub-pixels in two, counting a split and its mirror once; the unsplit cell first."""
    full = (1 << p) - 1
    masks = np.array([full] + list(range(1 << (p - 1), full)))
    bits = (masks[:, None] >> np.arange(p)) & 1
    return masks, bits.astype(float)


@njit(cache=True, error_model="numpy", parallel=True)
def _match(rgb, alpha, pw, ph, background, masks, bits, min_alpha, min_gain, mask_out, fg, bg, fg_on, bg_on):
    """match_cells() for a framebuffer's rgb (premultiplied) and alpha, with pw x ph pixels to a cell.

    background: linear RGB composited under partly covered pixels (black for the
    terminal's own). masks, bits: _masks(). Writes each cell's glyph mask, colours
    and whether each colour is drawn.

    Each cell's pixels are features (r, g, b, ALPHA_WEIGHT * alpha). Splitting
    them into sides with means m1, m0 leaves an error of sum|c|^2 - n1|m1|^2 -
    n0|m0|^2, so the best split maximizes |S1|^2/n1 + |S0|^2/n0. Nearly flat cells
    stay solid (the unsplit mask is first): splitting them into two almost equal
    colours costs output and, in small palettes, shows up as noise. No split can
    gain more than the error of the unsplit cell, so cells where that is already
    too small are settled without scoring any split: in most frames that is
    nearly every cell (empty or flat surface).
    """
    h, w = mask_out.shape
    p = pw * ph
    full = (1 << p) - 1
    for y in prange(h):
        feat, total, s1 = np.empty((p, 4)), np.empty(4), np.empty(4)  # scratch space for this row's cells
        for x in range(w):
            total[:] = 0.0
            energy = 0.0
            for i in range(p):
                py, px = y * ph + i // pw, x * pw + i % pw
                a = alpha[py, px]
                for k in range(3):
                    feat[i, k] = rgb[py, px, k] + (1.0 - a) * background[k]
                feat[i, 3] = ALPHA_WEIGHT * a
                for k in range(4):
                    total[k] += feat[i, k]
                    energy += feat[i, k] * feat[i, k]
            unsplit = (total[0] ** 2 + total[1] ** 2 + total[2] ** 2 + total[3] ** 2) / p
            best = 0
            s1[:] = total
            if energy - unsplit >= min_gain:
                best_score = unsplit
                for m in range(1, masks.shape[0]):
                    a0 = a1 = a2 = a3 = 0.0
                    n1 = 0.0
                    for i in range(p):
                        if bits[m, i]:
                            a0, a1, a2, a3 = a0 + feat[i, 0], a1 + feat[i, 1], a2 + feat[i, 2], a3 + feat[i, 3]
                            n1 += 1.0
                    r0, r1, r2, r3 = total[0] - a0, total[1] - a1, total[2] - a2, total[3] - a3
                    score = (a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3) / n1 + (r0 * r0 + r1 * r1 + r2 * r2 + r3 * r3) / (p - n1)
                    if score > best_score:
                        best, best_score = m, score
                if best_score - unsplit < min_gain:
                    best = 0
                if best:
                    s1[:] = 0.0
                    for i in range(p):
                        if bits[best, i]:
                            s1 += feat[i]
            mask = masks[best]
            n1 = 0
            for i in range(p):
                n1 += (mask >> i) & 1
            n0 = max(p - n1, 1)
            on1 = s1[3] / n1 >= min_alpha
            on0 = (total[3] - s1[3]) / n0 >= min_alpha and mask != full
            swap = not on1 and on0  # only the background side is drawn: show it as the foreground of the mirrored glyph
            for k in range(3):
                side1, side0 = s1[k] / n1, (total[k] - s1[k]) / n0
                fg[y, x, k], bg[y, x, k] = (side0, side1) if swap else (side1, side0)
            fg_on[y, x], bg_on[y, x] = on1 or on0, on1 and on0
            mask_out[y, x] = (full ^ mask if swap else mask) if on1 or on0 else 0


def match_cells(fb, glyphs, background=None):
    """Pick each cell's character and colours from the framebuffer's sub-pixels.

    background: linear RGB the screen is filled with, or None for the terminal's
    own background (taken as black when blending edges).
    """
    if glyphs.name == "ascii":
        return _match_ascii(fb, glyphs, background)
    pw, ph = glyphs.cell_pixels
    h, w = fb.height // ph, fb.width // pw
    masks, bits = _masks(glyphs.pixel_count)
    mask = np.empty((h, w), np.int64)
    fg, bg = np.empty((h, w, 3)), np.empty((h, w, 3))
    fg_on, bg_on = np.empty((h, w), bool), np.empty((h, w), bool)
    min_alpha = MIN_ALPHA * ALPHA_WEIGHT if background is None else -1.0  # a filled background is always opaque
    with kernel_lock():
        _match(fb.rgb, fb.alpha, pw, ph, np.zeros(3) if background is None else np.asarray(background, dtype=float),
               masks, bits, min_alpha, glyphs.pixel_count * MIN_SPLIT ** 2, mask, fg, bg, fg_on, bg_on)
    return Cells(np.array(glyphs.chars)[mask], fg, bg, fg_on, bg_on)


def _match_ascii(fb, glyphs, background, ramp=RAMP):
    rgb, alpha = _cell_pixels(fb, glyphs, None)
    coverage = alpha.mean(axis=2)
    colour = rgb.sum(axis=2) / np.maximum(alpha.sum(axis=2), 1e-9)[..., None]  # the drawn pixels' own colour
    drawn = coverage >= 0.25
    brightness = linear_to_srgb(luminance(colour))
    idx = 1 + np.clip(np.rint(brightness * (len(ramp) - 2)), 0, len(ramp) - 2).astype(int)
    chars = np.array(list(ramp))[np.where(drawn, idx, 0)]
    bg = np.zeros_like(colour) if background is None else np.broadcast_to(background, colour.shape)
    return Cells(chars, colour, bg, drawn, np.full(drawn.shape, background is not None))


def frame_to_text(fb, glyphs="ascii"):
    """Plain-text rendering of a framebuffer, for tests and debugging (ASCII unless told otherwise)."""
    cells = match_cells(fb, GLYPH_SETS[glyphs])
    return "\n".join("".join(row) for row in cells.chars)
