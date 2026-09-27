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
import time
import unicodedata

import numpy as np

from .color import COLOR_MODES, DEFAULT, Color, ansi_color, quantize, sgr_color, to_linear_rgb
from .console import detect_color_mode, detect_glyphs, open_console
from .glyphs import GLYPH_MODES, GLYPH_SETS, frame_to_text, match_cells
from .keys import InputDecoder, Key, MouseEvent

__all__ = ["Color", "Key", "MouseEvent", "Screen", "run", "add_display_args", "display_options", "frame_to_text"]

BOLD, DIM, REVERSE = 1, 2, 4
_ATTR_SGR = {BOLD: "1", DIM: "2", REVERSE: "7"}
MERGE_GAP = 4  # rewrite up to this many unchanged cells rather than move the cursor past them
MAX_STYLES = 1 << 16  # cached SGR sequences; shaded truecolor frames bring thousands of new ones a second


def _printable(ch):
    """ch if it takes exactly one cell, otherwise '?' (the grid has no room for wide or zero-width characters)."""
    if ch.isprintable() and unicodedata.east_asian_width(ch) not in "WF" and not unicodedata.combining(ch):
        return ch
    return "?"


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
        self._sgr = {}
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
        self._sgr.clear()
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

    def _style(self, fg, bg, attrs):
        key = (fg, bg, attrs)
        seq = self._sgr.get(key)
        if seq is None:
            if len(self._sgr) >= MAX_STYLES:
                self._sgr.clear()
            params = ["0"] + [code for bit, code in _ATTR_SGR.items() if attrs & bit]
            if self.color_mode != "mono":
                params += [sgr_color(fg), sgr_color(bg, background=True)]
            seq = self._sgr[key] = "\x1b[" + ";".join(params) + "m"
        return seq

    def render_updates(self):
        """The escape sequences that bring the terminal up to date with the grid, and mark it as shown."""
        full = self._shown is None
        if full:
            changed = np.ones(self.chars.shape, bool)
        else:
            chars, fg, bg, attrs = self._shown
            changed = (self.chars != chars) | (self.fg != fg) | (self.bg != bg) | (self.attrs != attrs)
        out = ["\x1b[0m\x1b[2J"] if full else []
        style = None
        for y in np.flatnonzero(changed.any(axis=1)):
            xs = np.flatnonzero(changed[y])
            splits = np.flatnonzero(np.diff(xs) > MERGE_GAP) + 1
            for run in np.split(xs, splits):
                s, e = int(run[0]), int(run[-1]) + 1
                out.append(f"\x1b[{y + 1};{s + 1}H")
                fg, bg, at = self.fg[y, s:e], self.bg[y, s:e], self.attrs[y, s:e]
                breaks = np.flatnonzero((fg[1:] != fg[:-1]) | (bg[1:] != bg[:-1]) | (at[1:] != at[:-1])) + 1
                row = self.chars[y, s:e]
                for a, b in zip([0, *breaks.tolist()], [*breaks.tolist(), e - s]):
                    key = (int(fg[a]), int(bg[a]), int(at[a]))
                    if key != style:
                        out.append(self._style(*key))
                        style = key
                    out.append("".join(row[a:b]))
        self._shown = (self.chars.copy(), self.fg.copy(), self.bg.copy(), self.attrs.copy())
        if not out:
            return ""
        # Synchronized output: terminals that support it show the whole update at once, others ignore it.
        return "\x1b[?2026h" + "".join(out) + "\x1b[0m\x1b[?2026l"

    def refresh(self):
        text = self.render_updates()
        if text and self.console is not None:
            self.console.write(text)


# ----- running an app ------------------------------------------------------------------------

def run(frame_fn, fps=30, glyphs=None, color=None, mouse=False, background=None, title=None):
    """Take over the terminal and call frame_fn(screen, dt, keys) up to `fps` times a second until it returns False.

    frame_fn may change screen.fps (the target) as it runs; screen.measured_fps is the rate achieved.

    title sets the terminal window's title while the app runs. The terminal is
    restored however the loop ends. Ctrl-C raises KeyboardInterrupt as usual.
    """
    with open_console(mouse=mouse, title=title) as console:
        screen = Screen(console, glyphs=glyphs, color=color, background=background)
        screen.fps = fps
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
