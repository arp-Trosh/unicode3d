# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
"""Colour: linear light <-> sRGB, the named hues, and quantizing to what a terminal can show.

The renderer works in linear light (so averaging samples and blending edges is
physically right) and converts to sRGB only for output. Terminals then get one
of four colour depths:

  truecolor  24-bit RGB (Windows Terminal, most modern Linux/macOS terminals)
  256        xterm's 256 colours, chosen by nearest match in OKLab
  16         the 16 ANSI colours
  mono       no colour at all

Colours on their way to the terminal are packed into one int per cell (see
`encode_rgb` / `encode_index`); -1 means the terminal's default colour.
"""
from enum import IntEnum

import numpy as np
from numba import njit, prange

from .threads import kernel_lock

COLOR_MODES = ("truecolor", "256", "16", "mono")

DEFAULT = -1        # the terminal's own foreground/background colour
_INDEXED = 1 << 24  # packed colour: palette index in the low byte
_RGB = 1 << 25      # packed colour: 0xRRGGBB in the low 24 bits


class Color(IntEnum):
    """Named hues: for text they are the terminal's ANSI colours, for 3D objects the RGB in HUE_RGB."""
    DEFAULT = 0
    WHITE = 1
    RED = 2
    GREEN = 3
    YELLOW = 4
    BLUE = 5
    MAGENTA = 6
    CYAN = 7


# ANSI colour number of each named hue, for text.
ANSI = {Color.WHITE: 7, Color.RED: 1, Color.GREEN: 2, Color.YELLOW: 3, Color.BLUE: 4, Color.MAGENTA: 5, Color.CYAN: 6}

# Fully lit surface colour (sRGB) of each named hue when used on a 3D object.
HUE_RGB = {
    Color.DEFAULT: (235, 235, 235),
    Color.WHITE: (235, 235, 235),
    Color.RED: (225, 45, 35),
    Color.GREEN: (50, 205, 50),
    Color.YELLOW: (240, 200, 40),
    Color.BLUE: (60, 110, 240),
    Color.MAGENTA: (210, 70, 210),
    Color.CYAN: (40, 200, 215),
}


def srgb_to_linear(c):
    """sRGB values 0..1 to linear light (in a parallel kernel: a 1024 x 1024 texture's took 40 ms in numpy)."""
    c = np.asarray(c, dtype=float, order="C")
    out = np.empty_like(c)
    with kernel_lock():
        _decode_srgb(c.reshape(-1), out.reshape(-1))
    return out


@njit(cache=True, error_model="numpy", parallel=True)
def _decode_srgb(srgb, out):
    for i in prange(srgb.shape[0]):
        c = srgb[i]
        out[i] = c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4  # (NaN stays NaN, as in numpy)


@njit(cache=True, error_model="numpy", parallel=True)
def _encode_srgb(linear, out):
    for i in prange(linear.shape[0]):
        c = linear[i]
        c = min(c, 1.0) if c > 0.0 else 0.0  # (NaN as 0)
        out[i] = c * 12.92 if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055


def linear_to_srgb(c):
    """Linear light to sRGB values 0..1 (clipped)."""
    c = np.ascontiguousarray(c, dtype=float)
    out = np.empty_like(c)
    with kernel_lock():
        _encode_srgb(c.reshape(-1), out.reshape(-1))
    return out


def to_srgb(color):
    """sRGB values 0..1 (3,) of a named Color, an (r, g, b) tuple of 0..255 ints, or of 0..1 floats."""
    if isinstance(color, (int, np.integer)):
        return np.array(HUE_RGB[Color(int(color))]) / 255.0
    c = np.asarray(color, dtype=float)
    if c.shape != (3,):
        raise ValueError(f"colour must be a Color or an (r, g, b) triple, not {color!r}")
    return c / 255.0 if np.issubdtype(np.asarray(color).dtype, np.integer) or c.max() > 1.0 else c


def to_linear_rgb(color):
    """Linear RGB (3,) of a named Color, an (r, g, b) tuple of 0..255 ints, or of 0..1 floats."""
    return srgb_to_linear(to_srgb(color))


def luminance(linear_rgb):
    """Relative luminance of linear RGB (..., 3)."""
    return np.asarray(linear_rgb) @ np.array([0.2126, 0.7152, 0.0722])


_LINEAR_CACHE = {}


def cached_linear_rgb(color):
    """to_linear_rgb(color), remembered for colours seen before (objects mostly keep theirs)."""
    if isinstance(color, (int, np.integer)):
        key = int(color)
    else:
        a = np.asarray(color)
        key = (a.dtype.kind, tuple(a.ravel().tolist()))
    c = _LINEAR_CACHE.get(key)
    if c is None:
        if len(_LINEAR_CACHE) > 4096:
            _LINEAR_CACHE.clear()
        c = _LINEAR_CACHE[key] = to_linear_rgb(color)
    return c


# ----- palettes ------------------------------------------------------------------------

def xterm_rgb():
    """sRGB (0..255) of all 256 xterm colours: 16 ANSI, the 6x6x6 cube, the grey ramp."""
    steps = np.array([0, 95, 135, 175, 215, 255])
    cube = np.array([(steps[r], steps[g], steps[b]) for r in range(6) for g in range(6) for b in range(6)])
    grey = np.array([(8 + 10 * i,) * 3 for i in range(24)])
    return np.concatenate([ANSI_RGB, cube, grey]).astype(float)


# xterm's defaults for the 16 ANSI colours. The real ones depend on the terminal's theme.
ANSI_RGB = np.array([
    (0, 0, 0), (205, 0, 0), (0, 205, 0), (205, 205, 0), (0, 0, 238), (205, 0, 205), (0, 205, 205), (229, 229, 229),
    (127, 127, 127), (255, 0, 0), (0, 255, 0), (255, 255, 0), (92, 92, 255), (255, 0, 255), (0, 255, 255), (255, 255, 255),
], dtype=float)


def oklab(srgb255):
    """OKLab coordinates of sRGB (..., 3) 0..255: distances there track how different colours look."""
    lin = srgb_to_linear(np.asarray(srgb255, dtype=float) / 255.0)
    lms = lin @ np.array([[0.4122214708, 0.2119034982, 0.0883024619],
                          [0.5363325363, 0.6806995451, 0.2817188376],
                          [0.0514459929, 0.1073969566, 0.6299787005]])
    lms = np.cbrt(lms)
    return lms @ np.array([[0.2104542553, 1.9779984951, 0.0259040371],
                           [0.7936177850, -2.4285922050, 0.7827717662],
                           [-0.0040720468, 0.4505937099, -0.8086757660]])


LUT_BITS = 5  # nearest-colour lookup tables are indexed by the top 5 bits of each channel


def _nearest_lut(candidates, first_index, chroma_weight):
    """(32, 32, 32) table: sRGB bucket -> palette index of the nearest candidate in OKLab.

    chroma_weight > 1 favours keeping the hue over matching the brightness, so a
    shaded green stays green rather than turning grey.
    """
    n = 1 << LUT_BITS
    centres = (np.arange(n) + 0.5) * (256 / n)
    grid = np.stack(np.meshgrid(centres, centres, centres, indexing="ij"), axis=-1).reshape(-1, 3)
    scale = np.array([1.0, np.sqrt(chroma_weight), np.sqrt(chroma_weight)])
    lab, cand = oklab(grid) * scale, oklab(candidates) * scale
    best = np.empty(len(grid), dtype=np.int16)
    for s in range(0, len(grid), 4096):
        d = ((lab[s:s + 4096, None, :] - cand[None]) ** 2).sum(axis=2)
        best[s:s + 4096] = d.argmin(axis=1) + first_index
    return best.reshape(n, n, n)


_LUTS = {}


def _lut(mode):
    if mode not in _LUTS:
        # 256 colours: only the cube and grey ramp, whose RGB are fixed; the first 16 follow the theme.
        _LUTS[mode] = _nearest_lut(xterm_rgb()[16:], 16, 1.5) if mode == "256" else _nearest_lut(ANSI_RGB, 0, 4.0)
    return _LUTS[mode]


# Typical distance between neighbouring palette colours (0..1 sRGB), which sets the dither amplitude.
_DITHER_STEP = {"256": 24 / 255, "16": 48 / 255}

BAYER4 = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]) / 16.0 + 1 / 32 - 0.5


def encode_rgb(srgb255):
    """Pack sRGB (..., 3) 0..255 ints into truecolor cell colours."""
    c = np.asarray(srgb255).astype(np.int64)
    return _RGB | (c[..., 0] << 16) | (c[..., 1] << 8) | c[..., 2]


def encode_index(index):
    """Pack palette indices into cell colours."""
    return _INDEXED | np.asarray(index).astype(np.int64)


def quantize(linear_rgb, mode, ys=None, xs=None, dither=True):
    """Cell colours (packed ints) for linear RGB (..., 3) in the given colour mode.

    ys, xs: positions (broadcastable to the colours' shape) that pick each colour's
    ordered-dither threshold. Dithering trades the banding of a small palette for
    a fine, fixed pattern; it is skipped in truecolor, where there is no banding.
    """
    srgb = linear_to_srgb(linear_rgb)
    if mode == "truecolor":
        return encode_rgb(np.rint(srgb * 255))
    if mode == "mono":
        return np.full(srgb.shape[:-1], DEFAULT, dtype=np.int64)
    if dither and ys is not None:
        srgb = srgb + (BAYER4[np.asarray(ys) % 4, np.asarray(xs) % 4] * _DITHER_STEP[mode])[..., None]
    idx = np.clip((srgb * 255).astype(int) >> (8 - LUT_BITS), 0, (1 << LUT_BITS) - 1)
    return encode_index(_lut(mode)[idx[..., 0], idx[..., 1], idx[..., 2]])


def sgr_color(packed, background=False):
    """SGR parameters that select a packed colour as the foreground (or background)."""
    packed = int(packed)
    if packed < 0:
        return "49" if background else "39"
    if packed & _RGB:
        return f"{48 if background else 38};2;{(packed >> 16) & 255};{(packed >> 8) & 255};{packed & 255}"
    index = packed & 255
    if index < 8:
        return str((40 if background else 30) + index)
    if index < 16:
        return str((100 if background else 90) + index - 8)
    return f"{48 if background else 38};5;{index}"


@njit(cache=True, error_model="numpy")
def put_int(buf, k, value):
    """Write a non-negative int in decimal into byte buffer buf at k; returns the index after it."""
    digits = 1
    while value >= 10 ** digits:
        digits += 1
    for i in range(digits - 1, -1, -1):
        buf[k + i] = 48 + value % 10
        value //= 10
    return k + digits


@njit(cache=True, error_model="numpy")
def put_sgr_color(buf, k, packed, background):
    """sgr_color(packed, background), written as ASCII into byte buffer buf at k; returns the index after it."""
    if packed < 0:
        return put_int(buf, k, 49 if background else 39)
    if packed & _RGB:
        k = put_int(buf, k, 48 if background else 38)
        buf[k], buf[k + 1], buf[k + 2] = 59, 50, 59  # ";2;"
        k = put_int(buf, k + 3, (packed >> 16) & 255)
        buf[k] = 59
        k = put_int(buf, k + 1, (packed >> 8) & 255)
        buf[k] = 59
        return put_int(buf, k + 1, packed & 255)
    index = packed & 255
    if index < 8:
        return put_int(buf, k, (40 if background else 30) + index)
    if index < 16:
        return put_int(buf, k, (100 if background else 90) + index - 8)
    k = put_int(buf, k, 48 if background else 38)
    buf[k], buf[k + 1], buf[k + 2] = 59, 53, 59  # ";5;"
    return put_int(buf, k + 3, index)


def ansi_color(color):
    """Packed cell colour of a named Color as ANSI text colour (DEFAULT stays the terminal's own)."""
    color = Color(int(color))
    return DEFAULT if color == Color.DEFAULT else int(encode_index(ANSI[color]))
