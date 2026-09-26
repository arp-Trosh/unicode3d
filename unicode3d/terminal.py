"""Turning framebuffers into terminal cells and drawing them with curses.

Two drawing modes:

  half   Unicode half blocks: each cell shows two pixels, the top one in the
         foreground colour of "▀" and the bottom one in its background colour.
         Brightness comes from a 256-colour palette of shades for each hue.
  ascii  One character per cell from a brightness ramp, for terminals without
         Unicode or 256 colours. Where 256 colours exist the characters are also
         tinted by brightness; otherwise dim/bold attributes add two more tones.
"""
import curses
import locale
import time
from enum import IntEnum

import numpy as np

from .raster import PIXELS_PER_CELL

RAMP = " .:-=+*#%@"  # dark -> bright; the first character is the empty background
MODES = ("half", "ascii")
SHADE_LEVELS = 24


class Color(IntEnum):
    DEFAULT = 0
    WHITE = 1
    RED = 2
    GREEN = 3
    YELLOW = 4
    BLUE = 5
    MAGENTA = 6
    CYAN = 7


_CURSES_COLORS = {
    Color.WHITE: curses.COLOR_WHITE,
    Color.RED: curses.COLOR_RED,
    Color.GREEN: curses.COLOR_GREEN,
    Color.YELLOW: curses.COLOR_YELLOW,
    Color.BLUE: curses.COLOR_BLUE,
    Color.MAGENTA: curses.COLOR_MAGENTA,
    Color.CYAN: curses.COLOR_CYAN,
}

# Fully lit surface colour of each hue; darker shades scale toward black.
HUE_RGB = {
    Color.DEFAULT: (235, 235, 235),
    Color.WHITE: (235, 235, 235),
    Color.RED: (240, 70, 55),
    Color.GREEN: (90, 230, 90),
    Color.YELLOW: (245, 215, 70),
    Color.BLUE: (90, 130, 245),
    Color.MAGENTA: (220, 90, 220),
    Color.CYAN: (80, 215, 225),
}


def xterm_rgb():
    """RGB of xterm-256 colours 16..255 (the 6x6x6 cube and the grey ramp)."""
    steps = np.array([0, 95, 135, 175, 215, 255])
    cube = np.array([(steps[r], steps[g], steps[b]) for r in range(6) for g in range(6) for b in range(6)])
    grey = np.array([(8 + 10 * i,) * 3 for i in range(24)])
    return np.concatenate([cube, grey]).astype(float)


LUMA = np.array([0.30, 0.59, 0.11])


def hue_sat(rgb):
    """Hue in degrees and HSV saturation of RGB rows."""
    rgb = np.atleast_2d(rgb).astype(float)
    hi, lo = rgb.max(axis=1), rgb.min(axis=1)
    c = np.maximum(hi - lo, 1e-9)
    r, g, b = rgb.T
    h = np.where(hi == r, ((g - b) / c) % 6, np.where(hi == g, (b - r) / c + 2, (r - g) / c + 4)) * 60
    return h, np.where(hi > 0, (hi - lo) / np.maximum(hi, 1e-9), 0)


def build_palette(levels=SHADE_LEVELS):
    """palette[hue, level] -> xterm colour index, never getting darker from one level to the next.

    Shadows keep more of their hue than a plain scale toward black would give them:
    xterm's 256 colours have few dark saturated entries, and without this a green
    die's shaded faces would turn grey.
    """
    rgb = xterm_rgb()
    luma = rgb @ LUMA
    cand_hue, cand_sat = hue_sat(rgb)
    table = np.zeros((len(Color), levels), dtype=np.int16)
    for hue, base in HUE_RGB.items():
        base = np.array(base, dtype=float)
        grey = base @ LUMA
        (h,), (sat,) = hue_sat(base)
        off_hue = np.abs((cand_hue - h + 180) % 360 - 180) > 20
        banned = off_hue & (cand_sat > 0.25) if sat > 0.25 else cand_sat > 0.25  # greys are always allowed
        prev = -1.0
        for lvl in range(levels):
            f = 0.08 + 0.92 * lvl / (levels - 1)
            want = grey * f + (base - grey) * np.sqrt(f)
            dist = ((rgb - np.clip(want, 0, 255)) ** 2).sum(axis=1)
            dist[(luma < prev) | banned] = np.inf
            best = int(np.argmin(dist))
            table[hue, lvl], prev = 16 + best, luma[best]
    return table


PALETTE = build_palette()


def shade_levels(shade, levels=SHADE_LEVELS):
    return np.clip(np.rint(shade * (levels - 1)), 0, levels - 1).astype(int)


def cell_shade(fb):
    """Collapse a pixel framebuffer to one (shade, colour) per cell, for character modes.

    Shade is the mean of a cell's drawn pixels (-1 if none); colour is the nearer pixel's.
    """
    h = fb.height // PIXELS_PER_CELL
    shade = fb.shade[:h * PIXELS_PER_CELL].reshape(h, PIXELS_PER_CELL, fb.width)
    depth = fb.depth[:h * PIXELS_PER_CELL].reshape(h, PIXELS_PER_CELL, fb.width)
    color = fb.color[:h * PIXELS_PER_CELL].reshape(h, PIXELS_PER_CELL, fb.width)
    drawn = shade >= 0
    n = drawn.sum(axis=1)
    mean = np.where(n > 0, np.where(drawn, shade, 0).sum(axis=1) / np.maximum(n, 1), -1.0)
    nearest = depth.argmax(axis=1)[:, None]
    return mean, np.take_along_axis(color, nearest, axis=1)[:, 0]


def shade_to_chars(shade, ramp=RAMP):
    """Map a per-cell shade array to ASCII codes (H, W) uint8.

    Drawn cells never use the background character, so even unlit surfaces
    keep their silhouette.
    """
    codes = np.frombuffer(ramp.encode("ascii"), dtype=np.uint8)
    n = len(codes)
    idx = 1 + np.clip(np.rint(shade * (n - 2)), 0, n - 2).astype(int)
    idx[shade < 0] = 0
    return codes[idx]


def frame_to_text(fb, ramp=RAMP):
    """Plain-text rendering of a framebuffer, one character per cell, for tests and debugging."""
    shade, _ = cell_shade(fb)
    return "\n".join(row.tobytes().decode("ascii") for row in shade_to_chars(shade, ramp))


def init_locale():
    """Let curses use the terminal's encoding (UTF-8 for half blocks). Call before curses starts."""
    try:
        locale.setlocale(locale.LC_ALL, "")
    except locale.Error:
        pass


def pick_mode(stdscr):
    """The richest mode this terminal supports."""
    encoding = (getattr(stdscr, "encoding", None) or locale.getpreferredencoding(False) or "").lower()
    unicode_ok = encoding.replace("-", "").replace("_", "") in ("utf8",)
    try:
        colors_ok = curses.has_colors() and curses.COLORS >= 256 and curses.COLOR_PAIRS >= 1024
    except AttributeError:  # COLORS is only defined once colours have been started
        colors_ok = False
    return "half" if unicode_ok and colors_ok else "ascii"


HALF_GLYPHS = np.array([" ", "▀", "▄", "█"])


class Screen:
    """Thin wrapper over a curses window."""

    def __init__(self, stdscr, ramp=RAMP, mode=None):
        self.stdscr = stdscr
        self.ramp = ramp
        stdscr.nodelay(True)
        stdscr.keypad(True)
        curses.set_escdelay(25)  # make a bare Esc key register promptly
        try:
            curses.curs_set(0)
        except curses.error:
            pass
        self.has_color = curses.has_colors()
        self.background = -1
        if self.has_color:
            curses.start_color()
            try:
                curses.use_default_colors()
            except curses.error:
                self.background = curses.COLOR_BLACK
            for color, curses_color in _CURSES_COLORS.items():
                curses.init_pair(int(color), curses_color, self.background)
        self.shaded = self.has_color and curses.COLORS >= 256
        self.mode = mode if mode in MODES else pick_mode(stdscr)
        if self.mode == "half" and not self.shaded:
            self.mode = "ascii"
        self._pairs = {}
        self._next_pair = len(Color)
        self._max_pairs = curses.COLOR_PAIRS if self.has_color else 0

    def size(self):
        """(rows, cols) of the terminal."""
        return self.stdscr.getmaxyx()

    def keys(self):
        """All keys pressed since the last call."""
        keys = []
        while (k := self.stdscr.getch()) != -1:
            keys.append(k)
        return keys

    def erase(self):
        self.stdscr.erase()

    def pair(self, fg, bg=-1):
        """Colour pair number for xterm colours fg on bg (-1: the terminal's background)."""
        key = (int(fg), int(bg))
        pair = self._pairs.get(key)
        if pair is None:
            if self._next_pair >= self._max_pairs:
                return self._pairs.get((key[0], -1), 0)  # out of pairs: drop the background
            pair = self._next_pair
            self._next_pair += 1
            curses.init_pair(pair, key[0], self.background if key[1] < 0 else key[1])
            self._pairs[key] = pair
        return pair

    def draw_frame(self, fb, top=0, left=0):
        if self.mode == "half":
            self._draw_half(fb, top, left)
        else:
            self._draw_ascii(fb, top, left)

    def _draw_half(self, fb, top, left):
        rows = fb.height // PIXELS_PER_CELL
        lit = np.where(fb.shade >= 0, PALETTE[fb.color, shade_levels(fb.shade)], -1)
        for y in range(rows):
            upper, lower = lit[2 * y], lit[2 * y + 1]
            has_up, has_low = upper >= 0, lower >= 0
            glyph = has_up * 1 + has_low * 2          # 0 empty, 1 ▀, 2 ▄, 3 both
            same = has_up & has_low & (upper == lower)
            glyph[same] = 3                           # █ needs no background colour
            fg = np.where(has_up, upper, lower)
            bg = np.where(has_up & has_low & ~same, lower, -1)
            fg[glyph == 0] = -1
            text = "".join(HALF_GLYPHS[glyph])
            key = fg * 512 + bg
            breaks = (np.flatnonzero(np.diff(key)) + 1).tolist()
            for s, e in zip([0] + breaks, breaks + [fb.width]):
                pair = 0 if fg[s] < 0 else self.pair(fg[s], bg[s])
                self._put(top + y, left + s, text[s:e], curses.color_pair(pair))

    def _draw_ascii(self, fb, top, left):
        shade, color = cell_shade(fb)
        chars = shade_to_chars(shade, self.ramp)
        if self.shaded:
            tone = np.where(shade >= 0, PALETTE[color, shade_levels(np.maximum(shade, 0))], -1)
        else:  # 8 colours: the hue from the object, two extra tones from attributes
            tone = np.where(shade < 0, 0, np.where(shade < 0.3, 1, np.where(shade > 0.7, 2, 0)))
        for y in range(chars.shape[0]):
            row = chars[y].tobytes().decode("ascii")
            if not self.has_color:
                self._put(top + y, left, row, 0)
                continue
            key = color[y].astype(np.int64) * 1024 + tone[y]
            breaks = (np.flatnonzero(np.diff(key)) + 1).tolist()
            for s, e in zip([0] + breaks, breaks + [fb.width]):
                if self.shaded:
                    attr = curses.color_pair(0 if tone[y, s] < 0 else self.pair(tone[y, s]))
                else:
                    attr = curses.color_pair(int(color[y, s])) | (0, curses.A_DIM, curses.A_BOLD)[tone[y, s]]
                self._put(top + y, left + s, row[s:e], attr)

    def text(self, y, x, s, color=Color.DEFAULT, bold=False, reverse=False, dim=False):
        attr = curses.color_pair(int(color)) if self.has_color else 0
        if bold:
            attr |= curses.A_BOLD
        if reverse:
            attr |= curses.A_REVERSE
        if dim:
            attr |= curses.A_DIM
        self._put(y, x, s, attr)

    def refresh(self):
        self.stdscr.noutrefresh()
        curses.doupdate()

    def _put(self, y, x, s, attr):
        rows, cols = self.size()
        if y < 0 or y >= rows or x >= cols:
            return
        if x < 0:
            s, x = s[-x:], 0
        try:
            self.stdscr.addstr(y, x, s[:cols - x], attr)
        except curses.error:
            pass  # writing the bottom-right cell raises after the text is drawn


def run_loop(stdscr, frame_fn, fps=30, mode=None):
    """Call frame_fn(screen, dt, keys) at up to `fps` until it returns False."""
    screen = Screen(stdscr, mode=mode)
    period = 1.0 / fps
    last = time.perf_counter()
    while True:
        start = time.perf_counter()
        dt, last = start - last, start
        if frame_fn(screen, dt, screen.keys()) is False:
            return
        remaining = period - (time.perf_counter() - start)
        if remaining > 0:
            time.sleep(remaining)
