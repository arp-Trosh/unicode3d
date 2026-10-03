# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
"""Pictures of what a terminal shows: a Screen's cells drawn as an image, block glyphs as their shapes and text in
a small font, for previews, screenshots, docs and tests (Screen.picture). Not used while drawing frames."""
import numpy as np

from .color import _RGB, xterm_rgb
from .glyphs import GLYPH_SETS

CELL = (8, 16)  # pixels of a cell in the pictures, wide x high
TERMINAL_FG, TERMINAL_BG = (204, 204, 204), (12, 12, 16)  # the terminal's own colours, in the pictures
BOLD, DIM, REVERSE = 1, 2, 4  # (as terminal.py's)

# A 5x7 bitmap of each character of the ASCII ramp, drawn in the middle of the cell (at CELL's size; scaled to
# others).
ASCII_BITMAPS = {
    ".": ["", "", "", "", "", "..#..", "..#.."],
    ":": ["", "..#..", "..#..", "", "..#..", "..#..", ""],
    "-": ["", "", "", "#####", "", "", ""],
    "=": ["", "", "#####", "", "#####", "", ""],
    "+": ["", "..#..", "..#..", "#####", "..#..", "..#..", ""],
    "*": ["", "#.#.#", ".###.", "#####", ".###.", "#.#.#", ""],
    "#": [".#.#.", "#####", ".#.#.", ".#.#.", "#####", ".#.#.", ""],
    "%": ["##..#", "##.#.", "..#..", ".#...", "#..##", "...##", ""],
    "@": [".###.", "#...#", "#.###", "#.#.#", "#.###", "#....", ".###."],
}

_masks = {}  # (glyphs, cell): {character: mask}
_fonts = {}  # cell height: Pillow font


def unpack_colors(packed, default):
    """sRGB 0..255 (..., 3) uint8 of packed cell colours (see color.quantize): truecolor, palette or `default`
    (the terminal's own colour)."""
    packed = np.asarray(packed, np.int64)
    out = np.empty(packed.shape + (3,), np.uint8)
    out[...] = default
    rgb = packed >= 0
    truecolor = rgb & (packed & _RGB != 0)
    for k, shift in enumerate((16, 8, 0)):
        out[..., k] = np.where(truecolor, (packed >> shift) & 255, out[..., k])
    indexed = rgb & ~truecolor
    out[indexed] = xterm_rgb()[packed[indexed] & 255].astype(np.uint8)
    return out


def _glyph_masks(glyphs, cell):
    """{character: (cell[1], cell[0]) bool, True where it shows the foreground} for a glyph set's characters."""
    key = (glyphs, cell)
    if key not in _masks:
        w, h = cell
        if glyphs == "ascii":
            masks = {" ": np.zeros((h, w), bool)}
            for c, rows in ASCII_BITMAPS.items():
                small = np.zeros(CELL[::-1], bool)
                for y, row in enumerate(rows):
                    for x, bit in enumerate(row):
                        small[2 * y + 1:2 * y + 3, x + 1] = bit == "#"
                ys, xs = np.arange(h) * CELL[1] // h, np.arange(w) * CELL[0] // w
                masks[c] = small[ys[:, None], xs[None, :]]
        else:
            glyph_set = GLYPH_SETS[glyphs]
            pw, ph = glyph_set.cell_pixels
            xs, ys = np.arange(w) * pw // w, np.arange(h) * ph // h
            sub = ys[:, None] * pw + xs[None, :]  # which of the cell's pixels each picture pixel is in
            masks = {c: (bits >> sub & 1).astype(bool) for bits, c in enumerate(glyph_set.chars)}
        _masks[key] = masks
    return _masks[key]


def _text_mask(c, cell):
    """A character of text, in Pillow's default font at about the cell's height: (cell[1], cell[0]) bool."""
    from PIL import Image, ImageDraw, ImageFont
    w, h = cell
    font = _fonts.get(h)
    if font is None:
        try:
            font = ImageFont.load_default(size=max(int(h * 0.8), 6))
        except (TypeError, OSError):  # (without FreeType: the small bitmap font)
            font = ImageFont.load_default()
        font = _fonts[h] = font
    image = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = draw.textbbox((0, 0), c, font=font)
    draw.text(((w - (right - left)) / 2 - left, (h - (bottom - top)) / 2 - top), c, fill=255, font=font)
    return np.asarray(image) >= 128


def cells_picture(chars, fg, bg, attrs=None, glyphs="sextant", cell=CELL, default_fg=TERMINAL_FG,
                  default_bg=TERMINAL_BG):
    """What a terminal shows for these cells (as a Screen holds them: characters, packed colours, attributes), as
    (rows * cell[1], cols * cell[0], 3) uint8 sRGB. Block glyphs of the glyph set are drawn as their shapes, other
    characters in a small font; reverse swaps a cell's colours, dim draws text half way to the background (bold
    is left as it is). The terminal's own colours are default_fg and default_bg."""
    chars = np.asarray(chars)
    if chars.dtype == np.uint32:
        chars = chars.view("<U1")
    rows, cols = chars.shape
    w, h = cell = (int(cell[0]), int(cell[1]))
    fore, back = unpack_colors(fg, default_fg), unpack_colors(bg, default_bg)
    if attrs is not None:
        attrs = np.asarray(attrs)
        dim = attrs & DIM != 0
        fore[dim] = ((fore[dim].astype(np.uint16) + back[dim]) // 2).astype(np.uint8)
        swap = attrs & REVERSE != 0
        fore[swap], back[swap] = back[swap], fore[swap].copy()
    masks = _glyph_masks(str(glyphs), cell)
    found, which = np.unique(chars, return_inverse=True)
    stack = np.stack([masks[c] if c in masks else np.zeros((h, w), bool) if c == " " else _text_mask(c, cell)
                      for c in found.tolist()]) if len(found) else np.zeros((0, h, w), bool)
    shown = stack[which.reshape(rows, cols)]  # (rows, cols, h, w)
    out = np.where(shown[..., None], fore[:, :, None, None], back[:, :, None, None])
    return np.ascontiguousarray(out.transpose(0, 2, 1, 3, 4).reshape(rows * h, cols * w, 3))
