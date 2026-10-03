# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
"""Pictures of what a terminal shows: a Screen's cells drawn as an image, block glyphs as their shapes and text in
a small font, for previews, screenshots, docs and tests (Screen.picture). Not used while drawing frames."""
import unicodedata

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
_fonts = {}  # cell: [(Pillow font, its mask for a character it lacks), ...] (_cell_fonts)


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
    """A character of text, in Pillow's default font at about the cell's height: (cell[1], cell[0]) bool. Box
    drawing and block elements (which that font lacks) are drawn as shapes instead, as terminals draw them, and
    other characters it lacks (symbols such as a bullet or arrows) in the first of SYMBOL_FONTS installed that has
    them."""
    drawn = _drawn_mask(c, cell)
    if drawn is not None:
        return drawn
    fonts = _cell_fonts(cell)
    for font, missing in fonts:
        mask = _font_mask(c, cell, font)
        if missing is None or not np.array_equal(mask, missing):
            return mask
    return _font_mask(c, cell, fonts[0][0])


# Fonts tried, in order, for characters Pillow's default font lacks; found by Pillow in the system's font folders.
SYMBOL_FONTS = ("DejaVuSans.ttf", "NotoSansSymbols2-Regular.ttf", "NotoSansSymbols-Regular.ttf", "seguisym.ttf",
                "Symbola.ttf", "Apple Symbols.ttf")


def _cell_fonts(cell):
    """[(Pillow font, its mask for a character it lacks, or None), ...] for text in cells cell[1] pixels high:
    the default font, then those of SYMBOL_FONTS that are installed."""
    if cell not in _fonts:
        from PIL import ImageFont
        h = cell[1]
        size = max(int(h * 0.8), 6)
        try:
            fonts = [ImageFont.load_default(size=size)]
        except (TypeError, OSError):  # (without FreeType: the small bitmap font, and no others)
            fonts = [ImageFont.load_default()]
        else:
            for name in SYMBOL_FONTS:
                try:
                    fonts.append(ImageFont.truetype(name, size))
                except OSError:
                    pass
        _fonts[cell] = [(f, _font_mask("\U0010fffd", cell, f) if len(fonts) > 1 else None) for f in fonts]
    return _fonts[cell]


def _font_mask(c, cell, font):
    """A character in a font, centred in the cell, squeezed across to fit if it is wider (an m or a w)."""
    from PIL import Image, ImageDraw
    w, h = cell
    left, top, right, bottom = ImageDraw.Draw(Image.new("L", (1, 1))).textbbox((0, 0), c, font=font)
    wide = max(w, int(right - left) + 1)
    image = Image.new("L", (wide, h), 0)
    ImageDraw.Draw(image).text(((wide - (right - left)) / 2 - left, (h - (bottom - top)) / 2 - top), c, fill=255,
                               font=font)
    if wide > w:
        image = image.resize((w, h), Image.Resampling.BOX)
    return np.asarray(image) >= 100  # (the default font's thin strokes, such as an r's arm, land near half)


_WEIGHTS = {"LIGHT": "light", "SINGLE": "light", "HEAVY": "heavy", "DOUBLE": "double"}
_ARMS = {"UP": ("up",), "DOWN": ("down",), "LEFT": ("left",), "RIGHT": ("right",), "VERTICAL": ("up", "down"),
         "HORIZONTAL": ("left", "right")}
_EIGHTHS = {"ONE": 1, "TWO": 2, "THREE": 3, "FOUR": 4, "FIVE": 5, "SIX": 6, "SEVEN": 7}


def _drawn_mask(c, cell):
    """A box-drawing character (U+2500-257F) or block element (U+2580-259F) drawn as its shape, from its Unicode
    name: (cell[1], cell[0]) bool, or None for any other character."""
    if not 0x2500 <= ord(c) <= 0x259F:
        return None
    name = unicodedata.name(c, "")
    w, h = cell
    mask = np.zeros((h, w), bool)
    if name.startswith("BOX DRAWINGS "):
        _draw_box(mask, name[len("BOX DRAWINGS "):])
        return mask
    words = name.split()
    if name.endswith(" SHADE"):  # light, medium, dark: a quarter, half, three quarters of the pixels
        y, x = np.mgrid[0:h, 0:w]
        mask[:] = {"LIGHT": (x % 2 == 0) & (y % 2 == 0), "MEDIUM": (x + y) % 2 == 0,
                   "DARK": ~((x % 2 == 1) & (y % 2 == 1))}[words[0]]
    elif words[0] == "QUADRANT":
        for quarter in name[len("QUADRANT "):].split(" AND "):
            upper, left = quarter.startswith("UPPER"), quarter.endswith("LEFT")
            mask[slice(0, h // 2) if upper else slice(h // 2, h), slice(0, w // 2) if left else slice(w // 2, w)] = True
    elif words[-1] == "BLOCK":
        if words[0] == "FULL":
            mask[:] = True
            return mask
        eighths = 4 if words[1] == "HALF" else _EIGHTHS.get(words[1], 8)
        side = words[0]
        if side in ("UPPER", "LOWER"):
            k = round(h * eighths / 8)
            if side == "UPPER":
                mask[:k] = True
            else:
                mask[h - k:] = True
        else:
            k = round(w * eighths / 8)
            if side == "LEFT":
                mask[:, :k] = True
            else:
                mask[:, w - k:] = True
    return mask


def _draw_box(mask, name):
    """Lines from the cell's middle to its edges, as a box-drawing character's name (after "BOX DRAWINGS ") says:
    which arms (up, down, left, right), each light, heavy or double; also dashed lines and diagonals."""
    h, w = mask.shape
    t = max(1, round(min(w, h / 2) / 8))  # a light line's thickness
    dashes = 1
    for n, word in ((2, "DOUBLE DASH"), (3, "TRIPLE DASH"), (4, "QUADRUPLE DASH")):
        if word in name:
            dashes, name = n, name.replace(word, "")
    if "DIAGONAL" in name:
        for k in range(max(w, h) * 4):
            f = k / (max(w, h) * 4 - 1)
            y = min(h - 1, int(f * h))
            if "UPPER LEFT TO LOWER RIGHT" in name or "CROSS" in name:
                mask[y, min(w - 1, int(f * w))] = True
            if "UPPER RIGHT TO LOWER LEFT" in name or "CROSS" in name:
                mask[y, min(w - 1, int((1 - f) * w))] = True
        return
    arms, last = {}, None
    groups = [g.split() for g in name.split(" AND ")]
    weights = [next((_WEIGHTS[x] for x in g if x in _WEIGHTS), None) for g in groups]
    for i, g in enumerate(groups):  # a group without a weight takes the one before it (or after, for the first)
        weight = weights[i] or last or next((x for x in weights if x), "light")
        last = weight
        for x in g:
            for arm in _ARMS.get(x, ()):
                arms[arm] = weight

    def band(weight, middle):
        """(start, end) of the arm across its length, and (start, end) of the gap a double line has inside."""
        thick = {"light": t, "heavy": 2 * t, "double": 3 * t}[weight]
        lo = middle - thick // 2
        return (lo, lo + thick), (lo + t, lo + 2 * t) if weight == "double" else None

    cx, cy = w // 2, h // 2
    across = {a: band(wt, cx if a in ("up", "down") else cy) for a, wt in arms.items()}
    vertical = [across[a][0] for a in ("up", "down") if a in arms] or [band("light", cx)[0]]
    horizontal = [across[a][0] for a in ("left", "right") if a in arms] or [band("light", cy)[0]]
    reach = {"up": (0, max(e for _, e in horizontal)), "down": (min(s for s, _ in horizontal), h),
             "left": (0, max(e for _, e in vertical)), "right": (min(s for s, _ in vertical), w)}
    gaps, singles = np.zeros_like(mask), np.zeros_like(mask)
    middle = {"up": slice(0, cy), "down": slice(cy, h), "left": slice(0, cx), "right": slice(cx, w)}
    for arm, ((s, e), gap) in across.items():
        along = slice(*reach[arm])
        lines = singles if gap is None else mask
        if arm in ("up", "down"):
            mask[along, s:e] = lines[along, s:e] = True
            if gap:
                gaps[middle[arm], gap[0]:gap[1]] = True
        else:
            mask[s:e, along] = lines[s:e, along] = True
            if gap:
                gaps[gap[0]:gap[1], middle[arm]] = True
    mask &= ~gaps  # a double line is two, with the gap between them running into the middle
    mask |= singles  # where a single line crosses a double one, it runs through
    if dashes > 1:  # dashed: each of `dashes` pieces of the line leaves its last third out
        step = (w if "HORIZONTAL" in name else h) / dashes
        for k in range(dashes):
            a, b = int(k * step + step * 2 / 3), int((k + 1) * step)
            if "HORIZONTAL" in name:
                mask[:, a:b] = False
            else:
                mask[a:b] = False


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
