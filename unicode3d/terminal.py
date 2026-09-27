# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
"""The screen: a grid of terminal cells that text and rendered frames are drawn into.

Screen keeps what should be on screen and what is on screen, and refresh()
sends only the cells that changed, as VT escape sequences. It needs no curses,
so the same code runs in Windows Terminal, the Windows console and Unix terminals.

How frames look is set by two independent choices (see glyphs.py and color.py):

  glyphs  half, quad, sextant or ascii: how many sub-pixels a cell shows
  color   truecolor, 256, 16 or mono: how many colours the terminal has

Both are detected, and can be forced with arguments, the UNICODE3D_GLYPHS and
UNICODE3D_COLOR environment variables, or the command-line flags that
add_display_args() adds.
"""
import threading
import time
import unicodedata

import numpy as np
from numba import njit

from .color import COLOR_MODES, DEFAULT, Color, ansi_color, put_int, put_sgr_color, quantize, to_linear_rgb
from .console import detect_color_mode, detect_glyphs, open_console
from .glyphs import GLYPH_MODES, GLYPH_SETS, frame_to_text, match_cells
from .keys import InputDecoder, Key, MouseEvent
from .mesh import make_box
from .scene import Camera, Light, Object3D, Renderer

__all__ = ["Color", "Key", "MouseEvent", "Screen", "run", "compile_kernels", "add_display_args", "display_options",
           "frame_to_text"]

BOLD, DIM, REVERSE = 1, 2, 4
MERGE_GAP = 4  # rewrite up to this many unchanged cells rather than move the cursor past them
CELL_BYTES = 64  # the most one cell can take to send: a cursor move, a style with two RGB colours, a character
COMPILE_MESSAGE = "First run compile, please wait..."
COMPILE_NOTICE_DELAY = 0.5  # seconds: loading compiled kernels from Numba's cache takes less, compiling them more
SYNC_BEGIN, SYNC_END = b"\x1b[?2026h", b"\x1b[0m\x1b[?2026l"  # synchronized output
_CLEAR = np.frombuffer(b"\x1b[0m\x1b[2J", np.uint8)


def _printable(ch):
    """ch if it takes exactly one cell, otherwise '?' (the grid has no room for wide or zero-width characters)."""
    if ch.isprintable() and unicodedata.east_asian_width(ch) not in "WF" and not unicodedata.combining(ch):
        return ch
    return "?"


@njit(cache=True)
def _encode_updates(chars, fg, bg, attrs, shown_chars, shown_fg, shown_bg, shown_attrs, full, mono, ascii_only, buf):
    """Write the escape sequences that bring the terminal from the shown_* grid to the current one into
    byte buffer buf (CELL_BYTES a cell, plus a little); returns how many bytes were written.

    chars and shown_chars are code points. With `full`, everything is redrawn
    after clearing the screen, whatever shown_* hold. ascii_only replaces
    characters outside ASCII with '?'.
    """
    rows, cols = chars.shape
    k = 0
    if full:
        buf[:len(_CLEAR)] = _CLEAR
        k = len(_CLEAR)
    styled, style_fg, style_bg, style_attrs = False, 0, 0, 0
    for y in range(rows):
        x = 0
        while x < cols:
            if not (full or chars[y, x] != shown_chars[y, x] or fg[y, x] != shown_fg[y, x]
                    or bg[y, x] != shown_bg[y, x] or attrs[y, x] != shown_attrs[y, x]):
                x += 1
                continue
            # A run of changed cells, rewriting gaps of up to MERGE_GAP unchanged ones rather than moving past them.
            last = x
            for j in range(x + 1, cols):
                if full or chars[y, j] != shown_chars[y, j] or fg[y, j] != shown_fg[y, j] \
                        or bg[y, j] != shown_bg[y, j] or attrs[y, j] != shown_attrs[y, j]:
                    if j - last > MERGE_GAP:
                        break
                    last = j
            buf[k], buf[k + 1] = 27, 91  # ESC [
            k = put_int(buf, k + 2, y + 1)
            buf[k] = 59  # ;
            k = put_int(buf, k + 1, x + 1)
            buf[k] = 72  # H
            k += 1
            for c in range(x, last + 1):
                f, b, a = fg[y, c], bg[y, c], attrs[y, c]
                if not styled or f != style_fg or b != style_bg or a != style_attrs:
                    styled, style_fg, style_bg, style_attrs = True, f, b, a
                    buf[k], buf[k + 1], buf[k + 2] = 27, 91, 48  # ESC [ 0
                    k += 3
                    for bit, code in ((BOLD, 49), (DIM, 50), (REVERSE, 55)):  # 1, 2, 7
                        if a & bit:
                            buf[k], buf[k + 1] = 59, code
                            k += 2
                    if not mono:
                        buf[k] = 59
                        k = put_sgr_color(buf, k + 1, f, False)
                        buf[k] = 59
                        k = put_sgr_color(buf, k + 1, b, True)
                    buf[k] = 109  # m
                    k += 1
                cp = chars[y, c]
                if cp < 0x80 or ascii_only:
                    buf[k] = cp if cp < 0x80 else 63  # ?
                    k += 1
                elif cp < 0x800:
                    buf[k], buf[k + 1] = 0xC0 | cp >> 6, 0x80 | cp & 0x3F
                    k += 2
                elif cp < 0x10000:
                    buf[k], buf[k + 1], buf[k + 2] = 0xE0 | cp >> 12, 0x80 | cp >> 6 & 0x3F, 0x80 | cp & 0x3F
                    k += 3
                else:
                    buf[k], buf[k + 1] = 0xF0 | cp >> 18, 0x80 | cp >> 12 & 0x3F
                    buf[k + 2], buf[k + 3] = 0x80 | cp >> 6 & 0x3F, 0x80 | cp & 0x3F
                    k += 4
            x = last + 1
    return k


class Screen:
    """What is drawn on the terminal: text, and frames from a Renderer.

    console: the Console to draw on (None for an off-screen grid, e.g. in tests).
    glyphs, color: force a glyph set / colour mode instead of detecting one.
    background: (r, g, b) to fill the screen with, or None for the terminal's own
    background. Knowing the background lets anti-aliased edges blend into it exactly.
    """

    def __init__(self, console=None, glyphs=None, color=None, background=None, size=(24, 80)):
        self.console = console
        self.unicode = console.unicode if console is not None else True
        color = color or detect_color_mode()
        if glyphs is None:
            # Without colour, blocks would only show silhouettes; the ASCII ramp still shows shading.
            glyphs = "ascii" if color == "mono" else detect_glyphs(unicode_ok=self.unicode)
        if glyphs in GLYPH_SETS and not self.unicode:
            glyphs = "ascii"
        self.background = None if background is None else to_linear_rgb(background)
        self.fps = 30               # frames a second run() aims for; change it any time
        self.measured_fps = None    # frames run() actually drew in the last second
        self._decoder = InputDecoder()
        self._rows = self._cols = 0
        self.set_glyphs(glyphs)
        self.set_color(color)
        self._resize(*(size if console is None else console.size()[::-1]))

    @property
    def glyph_modes(self):
        """The glyph sets this terminal can take: all of them, or only ascii without Unicode."""
        return GLYPH_MODES if self.unicode else ("ascii",)

    def set_glyphs(self, glyphs):
        """Switch glyph set; renderers pick it up through cell_pixels on their next resize()."""
        if glyphs not in GLYPH_SETS:
            raise ValueError(f"glyphs must be one of {GLYPH_MODES}, not {glyphs!r}")
        if glyphs not in self.glyph_modes:
            raise ValueError(f"{glyphs} glyphs need Unicode, which this terminal lacks")
        self.glyphs = GLYPH_SETS[glyphs]

    def set_color(self, color):
        """Switch colour mode. Cells already in the grid keep their colours until drawn again."""
        if color not in COLOR_MODES:
            raise ValueError(f"color must be one of {COLOR_MODES}, not {color!r}")
        self.color_mode = color
        self._bg_cell = DEFAULT if self.background is None else int(quantize(self.background, color))
        self._shown = None  # resend everything, in the new colours

    @property
    def cell_pixels(self):
        """(columns, rows) of pixels in a cell; pass it to Renderer.resize."""
        return self.glyphs.cell_pixels

    @property
    def mode(self):
        """Name of the glyph set in use."""
        return self.glyphs.name

    def size(self):
        """(rows, cols) of the terminal, as of the start of this frame."""
        return self._rows, self._cols

    def _resize(self, rows, cols):
        self._rows, self._cols = rows, cols
        self.chars = np.full((rows, cols), " ", dtype="<U1")
        self.fg = np.full((rows, cols), DEFAULT, dtype=np.int64)
        self.bg = np.full((rows, cols), self._bg_cell, dtype=np.int64)
        self.attrs = np.zeros((rows, cols), dtype=np.uint8)
        self._shown = None  # unknown: the next refresh redraws everything
        self._out = np.empty(rows * cols * CELL_BYTES + 64, np.uint8)

    def poll_size(self):
        """Pick up a change in terminal size (run() calls this before every frame)."""
        if self.console is not None:
            cols, rows = self.console.size()
            if (rows, cols) != (self._rows, self._cols):
                self._resize(rows, cols)

    # ----- input ---------------------------------------------------------------------------

    def keys(self):
        """Keys pressed and mouse clicks since the last call: ints (see keys.Key) and MouseEvents."""
        if self.console is None:
            return []
        now = time.monotonic()
        text = self.console.read()
        events = self._decoder.feed(text, now) if text else []
        return events + self._decoder.flush(now)

    # ----- drawing -------------------------------------------------------------------------

    def erase(self):
        self.chars.fill(" ")
        self.fg.fill(DEFAULT)
        self.bg.fill(self._bg_cell)
        self.attrs.fill(0)

    def text(self, y, x, s, color=Color.DEFAULT, bold=False, reverse=False, dim=False):
        """Write a string at row y, column x in a named colour (the terminal's ANSI palette), clipped to the screen."""
        if not 0 <= y < self._rows or x >= self._cols:
            return
        if x < 0:
            s, x = s[-x:], 0
        s = s[:self._cols - x]
        if not s:
            return
        n = len(s)
        self.chars[y, x:x + n] = [_printable(c) for c in s]
        self.fg[y, x:x + n] = DEFAULT if self.color_mode == "mono" else ansi_color(color)
        self.bg[y, x:x + n] = self._bg_cell
        self.attrs[y, x:x + n] = BOLD * bold | DIM * dim | REVERSE * reverse

    def draw_frame(self, fb, top=0, left=0):
        """Draw a Renderer's framebuffer with its top-left cell at (top, left)."""
        if fb.cell_pixels != self.cell_pixels:
            raise ValueError(f"framebuffer has {fb.cell_pixels} pixels per cell but the screen's glyphs need "
                             f"{self.cell_pixels}: pass screen.cell_pixels to Renderer.resize")
        cells = match_cells(fb, self.glyphs, self.background)
        h, w = cells.chars.shape
        y0, x0 = max(top, 0), max(left, 0)
        y1, x1 = min(top + h, self._rows), min(left + w, self._cols)
        if y0 >= y1 or x0 >= x1:
            return
        sub = (slice(y0 - top, y1 - top), slice(x0 - left, x1 - left))
        ys, xs = np.mgrid[y0:y1, x0:x1]
        fg = quantize(cells.fg[sub], self.color_mode, ys, 2 * xs)
        bg = quantize(cells.bg[sub], self.color_mode, ys, 2 * xs + 1)  # a different dither threshold from fg
        region = (slice(y0, y1), slice(x0, x1))
        self.chars[region] = cells.chars[sub]
        self.fg[region] = np.where(cells.fg_on[sub], fg, DEFAULT)
        self.bg[region] = np.where(cells.bg_on[sub], bg, self._bg_cell)
        self.attrs[region] = 0

    # ----- output --------------------------------------------------------------------------

    def _updates(self):
        """render_updates() as UTF-8 (ASCII if the terminal lacks Unicode)."""
        full = self._shown is None
        if full:
            self._shown = (self.chars.copy(), self.fg.copy(), self.bg.copy(), self.attrs.copy())
        chars, fg, bg, attrs = self._shown
        n = _encode_updates(self.chars.view(np.uint32), self.fg, self.bg, self.attrs, chars.view(np.uint32), fg, bg,
                            attrs, full, self.color_mode == "mono", not self.unicode, self._out)
        if not n:
            return b""
        for shown, now in zip(self._shown, (self.chars, self.fg, self.bg, self.attrs)):
            np.copyto(shown, now)
        # Synchronized output: terminals that support it show the whole update at once, others ignore it.
        return SYNC_BEGIN + self._out[:n].tobytes() + SYNC_END

    def render_updates(self):
        """The escape sequences that bring the terminal up to date with the grid, and mark it as shown."""
        return self._updates().decode("utf-8")

    def refresh(self):
        data = self._updates()
        if data and self.console is not None:
            self.console.write(data)


# ----- running an app ------------------------------------------------------------------------

def compile_kernels():
    """Compile everything drawing a frame runs (or load it from Numba's cache), by drawing a small scene off-screen.

    Compiling takes about 10 seconds on the first run after installing or
    upgrading; loading the cache, a fraction of a second. run() calls this
    before its first frame, showing COMPILE_MESSAGE while it compiles; programs
    that drive a Screen themselves can call it too, or the first frame waits
    for the compiling instead.
    """
    screen = Screen(None, glyphs="sextant", color="truecolor", size=(8, 16))
    renderer = Renderer(16, 8, screen.cell_pixels)
    objects = [Object3D(make_box(textures=[np.ones((4, 4))] * 6)),                     # textured
               Object3D(make_box(), position=np.array([1.0, 0.0, 0.0]), color=Color.RED)]  # plain
    screen.draw_frame(renderer.render(objects, Camera(), Light()))
    screen.render_updates()


def _compile_with_notice(console):
    """compile_kernels(), showing COMPILE_MESSAGE on the console if it takes long enough to be compiling."""
    def notice():
        cols, rows = console.size()
        y, x = max(rows // 2, 1), max((cols - len(COMPILE_MESSAGE)) // 2 + 1, 1)
        console.write(f"\x1b[0m\x1b[2J\x1b[{y};{x}H{COMPILE_MESSAGE}")

    timer = threading.Timer(COMPILE_NOTICE_DELAY, notice)
    timer.start()
    try:
        compile_kernels()
    finally:
        timer.cancel()
        timer.join()  # the notice is written in full or not at all; the first frame then clears the screen


def run(frame_fn, fps=30, glyphs=None, color=None, mouse=False, background=None, title=None):
    """Take over the terminal and call frame_fn(screen, dt, keys) up to `fps` times a second until it returns False.

    frame_fn may change screen.fps (the target) as it runs; screen.measured_fps is the rate achieved.

    title sets the terminal window's title while the app runs. The terminal is
    restored however the loop ends. Ctrl-C raises KeyboardInterrupt as usual.
    """
    with open_console(mouse=mouse, title=title) as console:
        screen = Screen(console, glyphs=glyphs, color=color, background=background)
        screen.fps = fps
        _compile_with_notice(console)
        last = time.perf_counter()
        second, frames = last, 0
        while True:
            start = time.perf_counter()
            dt, last = start - last, start
            if start - second >= 1.0:
                screen.measured_fps, second, frames = frames / (start - second), start, 0
            frames += 1
            screen.poll_size()
            if frame_fn(screen, dt, screen.keys()) is False:
                return
            remaining = 1.0 / screen.fps - (time.perf_counter() - start)
            if remaining > 0:
                time.sleep(remaining)


def add_display_args(parser):
    """Add --glyphs, --color and --ascii to an argparse parser; pass the result to display_options()."""
    group = parser.add_argument_group("display")
    group.add_argument("--glyphs", choices=GLYPH_MODES, help="characters to draw with (default: detected; "
                       "sextant gives the most detail but needs a font with Unicode 13 block symbols)")
    group.add_argument("--color", choices=COLOR_MODES, help="colour depth (default: detected)")
    group.add_argument("--ascii", action="store_true", help="same as --glyphs ascii")


def display_options(args):
    """Keyword arguments for run() from parsed add_display_args() flags."""
    return {"glyphs": "ascii" if args.ascii else args.glyphs, "color": args.color}
