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

from .color import linear_to_srgb, luminance

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


def match_cells(fb, glyphs, background=None):
    """Pick each cell's character and colours from the framebuffer's sub-pixels.

    background: linear RGB the screen is filled with, or None for the terminal's
    own background (taken as black when blending edges).
    """
    if glyphs.name == "ascii":
        return _match_ascii(fb, glyphs, background)
    rgb, alpha = _cell_pixels(fb, glyphs, background)
    p = glyphs.pixel_count
    masks, bits = _masks(p)
    feat = np.concatenate([rgb, ALPHA_WEIGHT * alpha[..., None]], axis=-1)  # (H, W, P, 4)

    # Splitting a cell into sides with means m1, m0 leaves an error of
    # sum|c|^2 - n1|m1|^2 - n0|m0|^2, so the best split maximizes |S1|^2/n1 + |S0|^2/n0.
    s1 = np.einsum("kp,hwpc->hwkc", bits, feat)
    s0 = feat.sum(axis=2)[:, :, None, :] - s1
    n1 = bits.sum(axis=1)
    n0 = p - n1
    score = (s1 ** 2).sum(-1) / n1 + (s0 ** 2).sum(-1) / np.maximum(n0, 1)
    best = score.argmax(axis=-1)
    # Nearly flat cells stay solid (the unsplit mask is first): splitting them into two almost equal
    # colours costs output and, in small palettes, shows up as noise.
    gain = np.take_along_axis(score, best[..., None], axis=-1)[..., 0] - score[..., 0]
    best = np.where(gain < p * MIN_SPLIT ** 2, 0, best)
    pick = best[..., None, None]
    side1 = np.take_along_axis(s1, pick, axis=2)[:, :, 0] / n1[best][..., None]
    side0 = np.take_along_axis(s0, pick, axis=2)[:, :, 0] / np.maximum(n0[best], 1)[..., None]
    mask = masks[best]

    full = (1 << p) - 1
    min_alpha = MIN_ALPHA * ALPHA_WEIGHT if background is None else -1.0  # a filled background is always opaque
    on1 = side1[..., 3] >= min_alpha
    on0 = (side0[..., 3] >= min_alpha) & (mask != full)
    swap = ~on1 & on0  # only the background side is drawn: show it as the foreground of the mirrored glyph
    mask = np.where(swap, full ^ mask, mask)
    fg = np.where(swap[..., None], side0[..., :3], side1[..., :3])
    bg = np.where(swap[..., None], side1[..., :3], side0[..., :3])
    fg_on = on1 | on0
    bg_on = on1 & on0
    mask = np.where(fg_on, mask, 0)
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
