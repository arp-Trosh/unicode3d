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
import contextlib
import faulthandler
import math
import os
import platform
import signal
import sys
import threading
import time
import traceback
import unicodedata

import numpy as np
from numba import njit

from .color import _RGB, COLOR_MODES, DEFAULT, Color, ansi_color, put_int, put_sgr_color, quantize, to_linear_rgb
from .console import detect_color_mode, detect_glyphs, open_console
from .glyphs import GLYPH_MODES, GLYPH_SETS, frame_to_text, match_cells
from .keys import HeldKeys, InputDecoder, Key, KeyRelease, MouseEvent
from .background import Sky
from .mesh import Mesh, make_box
from .queries import Colliders
from .scene import Camera, Light, Object3D, PointLight, Renderer

__all__ = ["Color", "Key", "KeyRelease", "MouseEvent", "Screen", "run", "compile_kernels", "add_display_args", "display_options",
           "frame_to_text", "crash_log_path"]

BOLD, DIM, REVERSE = 1, 2, 4
MERGE_GAP = 4  # rewrite up to this many unchanged cells rather than move the cursor past them
CELL_BYTES = 64  # the most one cell can take to send: a cursor move, a style with two RGB colours, a character
SETTLE_FRAMES = 30  # refreshes a cell may stay within Screen.color_tolerance of its colour before it is sent anyway
BACKGROUND_OUTPUT = True  # run() sends each frame to the terminal while drawing the next (see _Writer)
COMPILE_MESSAGE = "First run compile, please wait..."
COMPILE_NOTICE_DELAY = 0.5  # seconds: loading compiled kernels from Numba's cache takes less, compiling them more
CRASH_LOG_LIMIT = 1 << 20  # bytes: a longer crash log is cut down to its last quarter when run() opens it
SYNC_BEGIN, SYNC_END = b"\x1b[?2026h", b"\x1b[0m\x1b[?2026l"  # synchronized output
_CLEAR = np.frombuffer(b"\x1b[0m\x1b[2J", np.uint8)


def _cell(value):
    """A row, column or count of cells as an int (fractions rounded down), or None for NaN or an infinity."""
    if isinstance(value, (int, np.integer)):
        return int(value)
    value = float(value)
    return math.floor(value) if math.isfinite(value) else None


def _printable(ch):
    """ch if it takes exactly one cell, otherwise '?' (the grid has no room for wide or zero-width characters)."""
    if ch.isprintable() and unicodedata.east_asian_width(ch) not in "WF" and not unicodedata.combining(ch):
        return ch
    return "?"


@njit(cache=True, error_model="numpy")
def _near(a, b, tolerance):
    """Whether packed colours a and b are both RGB, with each channel within `tolerance` levels of the other's."""
    if a < 0 or b < 0 or not (a & _RGB) or not (b & _RGB):
        return False
    return (abs((a >> 16 & 255) - (b >> 16 & 255)) <= tolerance and abs((a >> 8 & 255) - (b >> 8 & 255)) <= tolerance
            and abs((a & 255) - (b & 255)) <= tolerance)


@njit(cache=True, error_model="numpy")
def _mark_updates(chars, fg, bg, attrs, shown_chars, shown_fg, shown_bg, shown_attrs, age, tolerance, send):
    """Mark in `send` the cells to send: those whose character or style changed, or whose colours moved by more
    than `tolerance` levels, and also those within it that have been off for SETTLE_FRAMES refreshes (counted in
    `age`), or all that are off at all when nothing moved by more (the picture has settled)."""
    rows, cols = chars.shape
    moved = 0
    for y in range(rows):
        for x in range(cols):
            f, b = fg[y, x], bg[y, x]
            sf, sb = shown_fg[y, x], shown_bg[y, x]
            send[y, x] = 0
            if chars[y, x] != shown_chars[y, x] or attrs[y, x] != shown_attrs[y, x]:
                send[y, x] = 1
            elif f == sf and b == sb:
                age[y, x] = 0
            elif (f == sf or _near(f, sf, tolerance)) and (b == sb or _near(b, sb, tolerance)):
                if age[y, x] < 255:
                    age[y, x] += 1
                if age[y, x] >= SETTLE_FRAMES:
                    send[y, x] = 2
                continue
            else:
                send[y, x] = 1
            moved += send[y, x]
    if moved == 0:  # still: send what is off, so that a picture that stops changing ends up exact
        for y in range(rows):
            for x in range(cols):
                if fg[y, x] != shown_fg[y, x] or bg[y, x] != shown_bg[y, x]:
                    send[y, x] = 2


@njit(cache=True, error_model="numpy")
def _encode_updates(chars, fg, bg, attrs, shown_chars, shown_fg, shown_bg, shown_attrs, age, send, full, mono,
                    ascii_only, tolerance, buf):
    """Write the escape sequences that bring the terminal from the shown_* grid towards the current one into
    byte buffer buf (CELL_BYTES a cell, plus a little), and update shown_* to what they leave on screen; returns
    how many bytes were written.

    chars and shown_chars are code points. With `full`, everything is redrawn
    after clearing the screen, whatever shown_* hold. A cell whose colours moved
    by at most `tolerance` levels (of 255, both RGB) is left as it is, up to
    SETTLE_FRAMES refreshes or until the picture stops changing (see
    _mark_updates; age counts the refreshes, and send is scratch space). A style
    sends only what changed: a colour on its own, the reset only when bold, dim
    or reverse change. ascii_only replaces characters outside ASCII with '?'.
    """
    rows, cols = chars.shape
    k = 0
    if full:
        buf[:len(_CLEAR)] = _CLEAR
        k = len(_CLEAR)
        send[:] = 1
    else:
        _mark_updates(chars, fg, bg, attrs, shown_chars, shown_fg, shown_bg, shown_attrs, age, tolerance, send)
    styled, style_fg, style_bg, style_attrs = False, 0, 0, 0
    for y in range(rows):
        x = 0
        while x < cols:
            if not send[y, x]:
                x += 1
                continue
            # A run of cells to send, rewriting gaps of up to MERGE_GAP others rather than moving past them.
            last = x
            for j in range(x + 1, cols):
                if send[y, j]:
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
                if not styled or a != style_attrs:
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
                elif not mono and (f != style_fg or b != style_bg):
                    buf[k], buf[k + 1] = 27, 91  # ESC [
                    k += 2
                    if f != style_fg:
                        k = put_sgr_color(buf, k, f, False)
                        if b != style_bg:
                            buf[k] = 59
                            k += 1
                    if b != style_bg:
                        k = put_sgr_color(buf, k, b, True)
                    buf[k] = 109  # m
                    k += 1
                    style_fg, style_bg = f, b
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
                shown_chars[y, c], shown_fg[y, c], shown_bg[y, c], shown_attrs[y, c] = chars[y, c], f, b, a
                age[y, c] = 0
            x = last + 1
    return k


class Screen:
    """What is drawn on the terminal: text, and frames from a Renderer.

    console: the Console to draw on (None for an off-screen grid, e.g. in tests).
    glyphs, color: force a glyph set / colour mode instead of detecting one.
    background: (r, g, b) to fill the screen with, or None for the terminal's own
    background. Knowing the background lets anti-aliased edges blend into it exactly.

    color_tolerance (in truecolor): a cell whose character and style are the same and whose colours moved by at
    most this many levels (of 255) is not sent again, which saves output (half the cells for many small moving
    objects at 1). What the terminal shows is then up to that many levels off, under half a just-noticeable
    difference at 1, and only for a moment: cells still off are sent once the picture stops changing, or after
    SETTLE_FRAMES refreshes. 0 sends every change.
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
        self.color_tolerance = 1    # levels a cell's colours may be off before it's sent again (truecolor only)
        self.key_release = console is not None and console.key_release  # keys() returns KeyRelease events
        self.held = HeldKeys()      # which keys are down, updated by keys()
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
        self._age = np.zeros((rows, cols), np.uint8)   # refreshes each cell has been left within the tolerance
        self._send = np.zeros((rows, cols), np.uint8)  # scratch space for _encode_updates

    def poll_size(self):
        """Pick up a change in terminal size (run() calls this before every frame)."""
        if self.console is not None:
            cols, rows = self.console.size()
            if (rows, cols) != (self._rows, self._cols):
                self._resize(rows, cols)

    # ----- input ---------------------------------------------------------------------------

    def keys(self):
        """Keys pressed and mouse events since the last call: ints (see keys.Key) and MouseEvents,
        plus KeyReleases if the console was opened with key_release (run(key_release=True)).

        Also brings `held` up to date. Where releases are reported, Ctrl-C arrives
        as a key rather than a signal, and raises KeyboardInterrupt here as usual.
        """
        if self.console is None:
            return []
        now = time.monotonic()
        text = self.console.read()
        events = (self._decoder.feed(text, now) if text else []) + self._decoder.flush(now)
        if self._decoder.kitty and 3 in events:
            raise KeyboardInterrupt
        if self.console.reports_releases:
            # Ask the keyboard about every key held before this frame; one released and pressed again
            # within the frame shows up as its new press, after the release.
            for key in self.held.keys():
                if not self.console.key_is_down(key):  # None (can't tell) counts as released
                    events.insert(0, KeyRelease(key))
        self.held.exact = self.console.reports_releases or self._decoder.kitty
        self.held.update(events, now)
        if not self.key_release:
            events = [e for e in events if not isinstance(e, KeyRelease)]
        return events

    # ----- drawing -------------------------------------------------------------------------

    def erase(self):
        self.chars.fill(" ")
        self.fg.fill(DEFAULT)
        self.bg.fill(self._bg_cell)
        self.attrs.fill(0)

    def text(self, y, x, s, color=Color.DEFAULT, bold=False, reverse=False, dim=False):
        """Write a string at row y, column x in a named colour (the terminal's ANSI palette), clipped to the screen."""
        y, x = _cell(y), _cell(x)
        if y is None or x is None or not 0 <= y < self._rows or x >= self._cols:
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

    def label(self, renderer, point, text, color=Color.WHITE, top=0, left=0, owner=None, clamp=False, hide=True,
              dy=0, bold=False, reverse=False, dim=False):
        """Write text at a point in the world, as the renderer's last frame (drawn at top, left) shows it: centred on
        the cell the point lands on, dy rows lower (negative: higher). A name over a character, a number where a
        hit landed, a marker on a goal.

        Nothing is written for a point behind the camera or outside the frame, unless clamp is set, which puts
        the text at the frame's edge in the direction the point lies, kept whole inside the frame; nor, with hide,
        for one behind something drawn (owner: the object, objects or Model the point belongs to, whose own
        surface doesn't count). Returns the Anchor (see Renderer.anchor), or None if nothing was written."""
        anchor = renderer.anchor(point, owner, clamp)
        top, left, dy = _cell(top), _cell(left), _cell(dy)
        if anchor is None or (hide and anchor.hidden) or None in (top, left, dy):
            return None
        x, y = left + anchor.x - len(text) // 2, top + anchor.y + dy
        if clamp:  # whole, inside the frame
            x = max(min(x, left + renderer.width - len(text)), left)
            y = max(min(y, top + renderer.height - 1), top)
        self.text(y, x, text, color, bold=bold, reverse=reverse, dim=dim)
        return anchor

    def bar(self, y, x, width, fraction, color=Color.GREEN, empty=Color.DEFAULT):
        """A meter: a bar `width` cells long at row y, column x, filled `fraction` (0..1) of the way in `color`, to
        an eighth of a cell (with Unicode; # and - without), the rest in `empty` (dim). For progress and levels of
        any kind (loading, memory or disk in use, a volume, frame time), on its own or at a label's Anchor (a
        character's health, say)."""
        y, x, width = _cell(y), _cell(x), _cell(width)
        if y is None or x is None or width is None:
            return
        fraction = min(1.0, max(0.0, float(fraction)))  # (NaN as 0)
        full, part = divmod(round(fraction * max(width, 0) * 8), 8)
        if self.unicode:
            tip = " \u258f\u258e\u258d\u258c\u258b\u258a\u2589"[part] if part else ""
        else:
            full, tip = full + (part >= 4), ""
        # Only the cells on the screen are made into text (a width far beyond it costs nothing).
        start, stop = max(0, -x), min(width, self._cols - x)
        filled = min(max(full - start, 0), stop - start)
        tip = tip if start <= full < stop else ""
        self.text(y, x + start, ("\u2588" if self.unicode else "#") * filled + tip, color)
        rest = max(stop - max(start, full + len(tip)), 0)
        self.text(y, x + start + filled + len(tip), ("\u2591" if self.unicode else "-") * rest, empty, dim=True)

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
                            attrs, self._age, self._send, full, self.color_mode == "mono", not self.unicode,
                            self._tolerance(), self._out)
        if not n:
            return b""
        # Synchronized output: terminals that support it show the whole update at once, others ignore it.
        return SYNC_BEGIN + self._out[:n].tobytes() + SYNC_END

    def _tolerance(self):
        """color_tolerance as an int in 0..255; 0 outside truecolor (where colours are palette entries already) or
        for anything that isn't a positive number."""
        if self.color_mode != "truecolor":
            return 0
        try:
            tolerance = float(self.color_tolerance)
        except (TypeError, ValueError):
            return 0
        return int(min(tolerance, 255.0)) if tolerance > 0 else 0  # (NaN fails the comparison)

    def render_updates(self):
        """The escape sequences that bring the terminal up to date with the grid, and mark it as shown."""
        return self._updates().decode("utf-8")

    def refresh(self):
        """Send the terminal what changed in the grid since the last refresh. Under run(), the sending goes on in
        the background while the next frame is drawn (see _Writer); the grid is free to change once this returns."""
        data = self._updates()
        if data and self.console is not None:
            (self._writer or self.console).write(data)

    _writer = None  # set by run(): the _Writer that sends refresh()'s output


# ----- running an app ------------------------------------------------------------------------

def compile_kernels():
    """Compile everything drawing a frame runs (or load it from Numba's cache), by drawing a small scene off-screen.

    Compiling takes about 10 seconds on the first run after installing or
    upgrading, on several cores at once (precompile.py; about 40 seconds on
    one); loading the cache, a fraction of a second. run() calls this
    before its first frame, showing COMPILE_MESSAGE while it compiles; programs
    that drive a Screen themselves can call it too, or the first frame waits
    for the compiling instead.
    """
    from .precompile import precompile
    precompile()  # the first time, on several cores at once; then drawing loads them from the cache
    screen = Screen(None, glyphs="sextant", color="truecolor", size=(8, 16))
    renderer = Renderer(16, 8, screen.cell_pixels, background=Sky())
    objects = [Object3D(make_box(textures=[np.ones((4, 4))] * 6)),                     # textured
               Object3D(make_box(), position=np.array([1.0, 0.0, 0.0]), color=Color.RED)]  # plain
    glass = make_box()
    glass.vertex_colors = np.full((len(glass.vertices), 4), 255)
    glass.vertex_colors[::2, 3] = 100  # alpha in the mesh's colours
    objects += [Object3D(glass, position=np.array([-1.0, 0.0, 0.0]), opacity=0.5)]  # see-through: more kernels
    holes = np.ones((4, 4, 4))
    holes[::2, :, 3] = 0.0
    stained = np.full((4, 4, 4), 0.5)
    objects += [Object3D(make_box(textures=[t] * 6), position=np.array([x, 1.0, 0.0]))  # alpha in textures
                for t, x in ((holes, -0.5), (stained, 0.5))]
    mirror = Mesh(np.array([(-2, -1, -1), (2, -1, -1), (2, 1, -1), (-2, 1, -1)], float),
                  np.array([(0, 1, 2), (0, 2, 3)]))
    objects += [Object3D(mirror, reflectivity=0.8), Object3D(make_box(), position=np.array([0.0, -1.0, 0.0]),
                                                             reflectivity=0.5)]  # a mirror, and something shiny
    lights = [Light(shadows=True), PointLight(np.array([0.0, 2.0, 2.0]), shadows=True)]  # shadows' kernels too
    screen.draw_frame(renderer.render(objects, Camera(), lights))
    renderer.max_pixels = 16 * 8  # drawn smaller than the screen needs, and stretched (as in huge terminals)
    screen.draw_frame(renderer.render(objects, Camera(), lights))
    screen.render_updates()
    # Ray and overlap queries.
    solid = Colliders(objects)
    solid.raycast((0.0, 0.0, 5.0), (0.0, 0.0, -1.0))
    solid.raycast((0.0, 0.0, 5.0), (0.0, 0.0, -1.0), all=True)
    solid.raycast_many(np.zeros((4, 3)), np.eye(4, 3) - 0.5)
    solid.push_out((0.0, 0.0, 0.0), 0.5, end=(0.0, 1.0, 0.0))
    solid.overlap_box((0.0, 0.0, 0.0), 1.0)


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


def crash_log_path(env=None, windows=None):
    """The file run() records crashes in: unicode3d/crash.log in the user's cache directory (XDG_CACHE_HOME
    or ~/.cache; LOCALAPPDATA on Windows)."""
    env = os.environ if env is None else env
    windows = os.name == "nt" if windows is None else windows
    base = env.get("LOCALAPPDATA") if windows else env.get("XDG_CACHE_HOME")
    return os.path.join(base or os.path.join(os.path.expanduser("~"), ".cache"), "unicode3d", "crash.log")


def _open_crash_log():
    """crash_log_path() opened for appending (cut down first if it has grown long), or None if it can't be."""
    path = crash_log_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path) and os.path.getsize(path) > CRASH_LOG_LIMIT:
            with open(path, "rb") as f:
                f.seek(-CRASH_LOG_LIMIT // 4, os.SEEK_END)
                tail = f.read()
            with open(path, "wb") as f:
                f.write(tail[tail.find(b"\n") + 1:])
        return open(path, "a", encoding="utf-8", errors="replace")
    except OSError:
        return None


def _describe(screen):
    """One line about the program and the screen, for the crash log."""
    import numba
    from . import __version__
    rows, cols = screen.size()
    return (f"{' '.join(sys.argv) or 'python'}: {cols}x{rows} cells, {screen.mode} glyphs, {screen.color_mode}, "
            f"{screen.measured_fps or 0:.0f}/{screen.fps} fps; unicode3d {__version__}, "
            f"Python {platform.python_version()}, numba {numba.__version__}, numpy {np.__version__}, "
            f"{platform.platform()}")


def _frame_period(fps):
    """The seconds a frame takes at `fps` frames a second: 0 (no waiting between frames) for 0, None, or anything else
    that isn't a positive number."""
    try:
        return 1.0 / fps if fps > 0 else 0.0  # (NaN fails the comparison too)
    except TypeError:
        return 0.0


class _Writer:
    """Writes to a console from a thread of its own, one write at a time, so that run() draws the next frame
    while the terminal takes in the last one (writing a big frame can take as long as drawing it).

    write() returns once the data is queued, after waiting for the write before it to finish: at most one frame is
    on its way at a time, so a slow terminal slows the frame rate rather than letting frames pile up. refresh()
    hands over bytes it has finished with, so the grid is free to change while they are sent. An error the console
    raised in the background is raised by the next write()."""

    WAIT = 0.1  # seconds between checks while waiting (an untimed wait can't be interrupted by Ctrl-C on Windows)

    def __init__(self, console):
        self._console = console
        self._pending = None  # the data being written, until it has been
        self._error = None
        self._closing = False
        self._done = threading.Condition()
        self._thread = threading.Thread(target=self._send, name="unicode3d output", daemon=True)
        self._thread.start()

    def write(self, data):
        with self._done:
            self._wait()
            self._pending = data
            self._done.notify_all()

    def close(self):
        """Wait for the write on its way, then stop the thread. An error it raised is dropped: the console's
        own write on closing (restoring the terminal) meets the same trouble and raises it."""
        with self._done:
            while self._pending is not None:
                self._done.wait(self.WAIT)
            self._closing = True
            self._done.notify_all()
        self._thread.join()

    def _wait(self):
        """Wait (holding the lock) until no write is on its way, and raise what the last one raised."""
        while self._pending is not None:
            self._done.wait(self.WAIT)
        error, self._error = self._error, None
        if error is not None:
            raise error

    def _send(self):
        while True:
            with self._done:
                while self._pending is None and not self._closing:
                    self._done.wait()
                if self._pending is None:
                    return
                data = self._pending
            error = None
            try:
                self._console.write(data)
            except BaseException as e:  # (for the program's own thread to raise)
                error = e
            with self._done:
                self._pending, self._error = None, error
                self._done.notify_all()


@contextlib.contextmanager
def _exit_on_sigterm():
    """While the loop runs, SIGTERM (kill, a closing session) ends the program as SystemExit does, so that the
    terminal is restored on the way out: the signal's default ends Python at once, leaving the terminal in raw mode
    on the alternate screen. A handler the program set itself is left alone."""
    if (not hasattr(signal, "SIGTERM") or threading.current_thread() is not threading.main_thread()
            or signal.getsignal(signal.SIGTERM) is not signal.SIG_DFL):
        yield
        return

    def stop(signum, frame):
        raise SystemExit(128 + signum)  # (the exit status a shell shows for a program the signal ended)

    signal.signal(signal.SIGTERM, stop)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)


def run(frame_fn, fps=30, glyphs=None, color=None, mouse=False, background=None, title=None, key_release=False):
    """Take over the terminal and call frame_fn(screen, dt, keys) up to `fps` times a second until it returns False.

    frame_fn may change screen.fps (the target) as it runs; screen.measured_fps is the rate achieved. An fps of 0 draws
    frames as fast as it can.

    mouse: False, True (clicks and wheel), "drag" (also moves while a button is
    held, for sliders) or "move" (every move, for hover effects).
    key_release: ask the terminal to report key releases, and pass them to
    frame_fn as KeyRelease events. screen.held tells which keys are down either
    way, exactly where releases are reported and by estimate elsewhere.
    title sets the terminal window's title while the app runs. The terminal is
    restored however the loop ends. Ctrl-C raises KeyboardInterrupt as usual.

    Each run is noted in crash_log_path(), with the terminal's size and settings; if the
    loop raises, the traceback goes there too (and is raised as usual), and a crash of
    Python itself leaves a stack trace there (faulthandler), where the screen would lose it. SIGTERM ends the
    program as sys.exit() would, with the terminal restored (unless the program handles that signal itself).
    """
    log, screen, frame, handler = _open_crash_log(), None, 0, faulthandler.is_enabled()
    if log is not None:
        faulthandler.enable(log)
    try:
        with _exit_on_sigterm(), open_console(mouse=mouse, title=title, key_release=key_release) as console:
            screen = Screen(console, glyphs=glyphs, color=color, background=background)
            screen.fps = fps
            if log is not None:
                log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} started {_describe(screen)}\n")
                log.flush()
            _compile_with_notice(console)
            if BACKGROUND_OUTPUT:
                screen._writer = _Writer(console)
            try:
                last = time.perf_counter()
                second, frames = last, 0
                while True:
                    start = time.perf_counter()
                    dt, last = start - last, start
                    if start - second >= 1.0:
                        screen.measured_fps, second, frames = frames / (start - second), start, 0
                    frames += 1
                    frame += 1
                    screen.poll_size()
                    if frame_fn(screen, dt, screen.keys()) is False:
                        return
                    remaining = _frame_period(screen.fps) - (time.perf_counter() - start)
                    if remaining > 0:
                        time.sleep(remaining)
            finally:
                if screen._writer is not None:  # the last frame sent before the terminal is restored
                    screen._writer.close()
                    screen._writer = None
    except Exception:  # (the terminal is restored by now)
        if log is not None:
            where = _describe(screen) if screen is not None else " ".join(sys.argv)
            log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} crashed in frame {frame}: {where}\n"
                      f"{traceback.format_exc()}\n")
            log.flush()
            print(f"unicode3d: stopped by an error; the details are also in {log.name}", file=sys.stderr)
        raise
    finally:
        if log is not None:
            faulthandler.disable()
            if handler:
                faulthandler.enable()
            log.close()


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
