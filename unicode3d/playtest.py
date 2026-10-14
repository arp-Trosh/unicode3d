# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
"""Playing a program off-screen, for checking it without a terminal: scripted keys, clicks and held keys, then
screenshots, contact sheets and text dumps of what the terminal would show, and a report of frame times, output,
warnings and blank frames. Made for the loop of building a game with an agent, which can't see a terminal: change,
playtest, look, fix.

    from unicode3d.playtest import Playtest

    test = Playtest(Game(), size=(40, 120), out="playtest")  # size as (rows, cols), as Screen takes it
    test.wait(10)                  # ten frames with no input
    test.shot("title")             # playtest/title.png and playtest/title.txt
    test.press("enter")            # a frame with Enter pressed (let go with the next)
    test.hold("w", frames=20)      # 20 frames with W held down, then one with it let go
    test.click_text("Start")       # a click on a word on the screen
    assert "Score" in test.text()  # the screen as text
    test.press("q")
    assert test.quit               # frame() returned False
    print(test.report())

From the command line, the same as a script of commands (see SCRIPT_HELP):

    python -m unicode3d.playtest play mygame.py:Game "wait 10; shot title; press enter; hold w 20; shot walk"
    python -m unicode3d.playtest preview models/goblin.glb

preview() draws contact sheets of a model's animations, as the terminal shows them.
"""
import argparse
import ast
import collections
import dataclasses
import importlib
import importlib.util
import inspect
import os
import re
import shlex
import sys
import time
import traceback
import warnings

import numpy as np

from . import mesh as mesh_module
from .glyphs import GLYPH_MODES, GLYPH_SETS
from .keys import Key, KeyRelease, MouseEvent, key_codes
from .color import COLOR_MODES
from .terminal import Screen, compile_kernels

__all__ = ["Playtest", "PlaytestError", "Report", "Found", "parse_key", "contact_sheet", "preview", "parse_script", "run_script", "load_target",
           "SCRIPT_HELP"]

REPEAT_DELAY = 0.5  # seconds a held key waits before a terminal starts repeating it
REPEAT_RATE = 30    # and then the times a second it repeats
SHEET_GAP = 4       # pixels between the tiles of a contact sheet
GAP_COLOR = 90      # their grey

_KEY_NAMES = {**{k.name.lower(): int(k) for k in Key},
              "space": ord(" "), "escape": Key.ESC, "return": Key.ENTER, "esc": Key.ESC, "pgup": Key.PAGE_UP,
              "pgdn": Key.PAGE_DOWN, "del": Key.DELETE, "shift+tab": Key.BACK_TAB, "backtab": Key.BACK_TAB}


def parse_key(key):
    """A key as frame() receives it, from an int (a character's code or a Key), a single character ("a", "A",
    "+"), or a name: a Key's ("enter", "esc", "up", "f2", "page_down", ...), "space", or "ctrl+x" for a control
    character. Raises ValueError for anything else, listing the names."""
    if isinstance(key, (int, np.integer)) and not isinstance(key, bool):
        return int(key)
    if isinstance(key, str):
        if len(key) == 1:
            return ord(key)
        name = key.lower()
        if name in _KEY_NAMES:
            return int(_KEY_NAMES[name])
        if name.startswith("ctrl+") and len(name) == 6 and "a" <= name[5] <= "z":
            return ord(name[5]) & 0x1F
    raise ValueError(f"unknown key {key!r}: use a character, an int, ctrl+<letter> or one of "
                     f"{', '.join(sorted(_KEY_NAMES))}")


Found = collections.namedtuple("Found", "row col")
Found.__doc__ = "Where text is on the screen: row and column (col), from 0. click() takes them as x=col, y=row."

BUTTONS = ("left", "middle", "right", "wheel_up", "wheel_down")


def _button(name):
    if str(name).lower() not in BUTTONS:
        raise ValueError(f"button must be one of {', '.join(BUTTONS)}, not {name!r}")
    return getattr(MouseEvent, str(name).upper())


class PlaytestError(Exception):
    """frame() raised (the original exception is the cause), an expect failed, or the program was played after it
    quit."""


class _Console:
    """What a Screen draws on in a playtest: a size the test sets, no input (Playtest passes the events), and
    output counted rather than written."""
    reports_releases = False

    def __init__(self, rows, cols, unicode, key_release):
        self.rows, self.cols, self.unicode, self.key_release = rows, cols, unicode, key_release
        self.written = 0

    def size(self):
        return self.cols, self.rows

    def read(self):
        return ""

    def key_is_down(self, key):
        return None

    def write(self, data):
        self.written += len(data)


@dataclasses.dataclass
class Report:
    """What a playtest saw. Times are wall-clock milliseconds for frame() itself; the first frame is left out of
    them (it loads textures, packs meshes, compiles anything compile_kernels() missed)."""
    frames: int                 # frames drawn
    seconds: float              # the time they covered in the program (frames * dt)
    size: tuple                 # (rows, cols) at the end
    glyphs: str
    color: str
    quit_at: int | None         # the frame whose frame() returned False, if any
    first_ms: float | None      # the first frame's time
    mean_ms: float | None       # mean, 95th percentile and slowest of the rest
    p95_ms: float | None
    max_ms: float | None
    slowest_frame: int | None
    mean_bytes: float | None    # output to the terminal per frame (after the first), what a slow link must carry
    max_bytes: int | None
    blank_frames: list          # frames that left the screen empty
    warnings: list              # (frame, "Category: message") for each different warning raised while playing
    shots: list                 # paths of the pictures and text dumps saved

    def __str__(self):
        lines = [f"{self.frames} frames ({self.seconds:.2f} s of play) at {self.size[1]}x{self.size[0]} cells, "
                 f"{self.glyphs} glyphs, {self.color}"]
        if self.mean_ms is not None:
            lines.append(f"frame time: mean {self.mean_ms:.1f} ms, 95% {self.p95_ms:.1f} ms, slowest "
                         f"{self.max_ms:.1f} ms (frame {self.slowest_frame}); first frame {self.first_ms:.0f} ms")
            lines.append(f"output: {self.mean_bytes / 1024:.1f} KB a frame on average, {self.max_bytes / 1024:.1f} KB "
                         f"at most")
        elif self.first_ms is not None:
            lines.append(f"frame time: {self.first_ms:.0f} ms (one frame)")
        if self.quit_at is not None:
            lines.append(f"quit: frame() returned False at frame {self.quit_at}")
        if self.blank_frames:
            shown = ", ".join(map(str, self.blank_frames[:10])) + (" ..." if len(self.blank_frames) > 10 else "")
            lines.append(f"BLANK SCREEN in {len(self.blank_frames)} frames: {shown}")
        for frame, message in self.warnings:
            lines.append(f"WARNING (first at frame {frame}): {message}")
        for path in self.shots:
            lines.append(f"saved {path}")
        return "\n".join(lines)


class Playtest:
    """Drives frame(screen, dt, events), the function run() would call, off-screen.

    frame: the program's frame function, or an object with a frame() method (a game).
    size: (rows, cols) of the pretend terminal, as Screen(size=...) takes it (scripts and the command line give
    sizes as COLSxROWS); glyphs, color: its glyph set and colour mode (fixed, so pictures are the same on every
    machine); dt: the seconds each frame is told have passed (fixed: a playtest plays the same every time).
    unicode=False plays it as a terminal without Unicode (ASCII glyphs only). out: the folder shots and sheets are
    saved in. compile: compile the kernels first (as run() does), so the first frame's time is the program's own.

    Keys: a pressed key is let go in the next frame, and a held one (hold(), down()) repeats as a terminal repeats
    it, REPEAT_RATE times a second after REPEAT_DELAY. releases: the terminal reports releases (as with the kitty
    keyboard protocol), so screen.held knows exactly which keys are down; with releases=False, as in most
    terminals, only presses and repeats arrive, and screen.held guesses (see HeldKeys). key_release: as run()'s,
    pass the releases to frame() as KeyRelease events (where the terminal reports them). erase: as run()'s, erase
    the screen before each frame.

    Each frame erases and refreshes the screen as run() does (a playtest counts the bytes instead of writing them).
    An exception from frame() is raised as a PlaytestError naming the frame, after saving a shot of the screen as it
    was. Once frame() has returned False, what plays frames raises PlaytestError, and
    what looks at the screen (text, find, expect, shot) sees the last one. Warnings frame() raises are collected for
    the report (under the program's own warning filters: one made an error is raised as an error); those raised on
    the program's other threads are not.

    check_edits: check, each frame, that no mesh drawn (or in a Colliders set) had its arrays edited in place without
    invalidate(), which renderers and Colliders don't see; one that was is a warning in the report. It takes
    checksums of the meshes' arrays each frame (of a big one, such as a texture, a slice of its rows at a time, so
    an edit to a few rows shows some frames later: see mesh.CHECK_BYTES), which frame times leave out.
    """

    def __init__(self, frame, size=(40, 120), glyphs="sextant", color="truecolor", dt=1 / 30, unicode=True,
                 releases=True, key_release=False, out="playtest", compile=True, check_edits=True, erase=True):
        frame = getattr(frame, "frame", frame)
        if not callable(frame):
            raise TypeError(f"frame must be the program's frame function (screen, dt, events) or an object with a "
                            f"frame() method, not {frame!r}")
        if glyphs not in GLYPH_MODES:
            raise ValueError(f"glyphs must be one of {GLYPH_MODES}, not {glyphs!r}")
        self.frame_fn = frame
        self.dt = float(dt)
        self.out = out
        self.releases = bool(releases)
        self.check_edits = bool(check_edits)
        self.erase = bool(erase)
        self.console = _Console(*size, unicode=unicode, key_release=key_release)
        if compile:
            compile_kernels()
        self.screen = Screen(self.console, glyphs=glyphs if unicode else "ascii", color=color)
        self.screen.held.exact = self.releases
        self.frames = 0          # frames drawn so far
        self.quit = False        # frame() returned False
        self.quit_at = None
        self.times = []          # seconds each frame() took
        self.bytes = []          # bytes each frame sent the terminal
        self.blank = []          # frames that left the screen empty
        self.warnings = {}       # "Category: message": the first frame it was raised in
        self.shots = []          # paths saved
        self._held = {}          # key code held with down(): the time (now) of the frame its press went with
        self._queued = []        # events for the next frame (presses from down(), releases)

    # ----- playing -------------------------------------------------------------------------

    @property
    def now(self):
        """Seconds of play so far."""
        return self.frames * self.dt

    def _repeats(self, since, t):
        """How many times a key pressed at `since` has repeated by time t."""
        t -= since + REPEAT_DELAY
        return 0 if t < -1e-9 else int(t * REPEAT_RATE + 1e-9) + 1

    def step(self, events=()):
        """One frame, given these events (ints, KeyReleases, MouseEvents) as screen.keys() would return them, as
        they are (keys pressed here are not let go: see press()). Keys held with down() repeat as a terminal
        repeats them. Returns what frame() returned."""
        self._playing()
        events, self._queued = self._queued + list(events), []
        if self.frames:
            for key, since in self._held.items():  # a terminal's auto-repeat, at its own rate whatever the frame's
                n = self._repeats(since, self.now) - self._repeats(since, self.now - self.dt)
                events += [key] * min(n, 100)
        if not self.releases:
            events = [e for e in events if not isinstance(e, KeyRelease)]
        screen = self.screen
        screen.held.update(events, self.now)
        if not screen.key_release:
            events = [e for e in events if not isinstance(e, KeyRelease)]
        screen.poll_size()
        before = None
        if self.erase:  # (kept, for a frame() that quits: the terminal goes on showing the last frame drawn)
            before = (screen.chars.copy(), screen.fg.copy(), screen.bg.copy(), screen.attrs.copy())
            screen.erase()
        written = self.console.written
        self.frames += 1
        checking, checked = mesh_module.CHECK_EDITS, mesh_module.check_seconds
        mesh_module.CHECK_EDITS = self.check_edits
        start = time.perf_counter()
        try:
            with warnings.catch_warnings(record=True) as caught:  # (the program's own filters still apply)
                result = self.frame_fn(screen, self.dt, key_codes(events))
                if result is not False:
                    screen.refresh()
        except Exception as e:
            path = self._save_picture(f"crash-frame-{self.frames}")
            raise PlaytestError(f"frame() raised {type(e).__name__} in frame {self.frames} ({self.now:.2f} s of "
                                f"play): {e}\n(the screen as it was is in {path})") from e
        finally:
            mesh_module.CHECK_EDITS = checking
        elapsed = time.perf_counter() - start - (mesh_module.check_seconds - checked)  # (the checks left out)
        for w in caught:
            self.warnings.setdefault(f"{w.category.__name__}: {w.message} ({os.path.basename(w.filename)}:"
                                     f"{w.lineno})", self.frames)
        self.times.append(elapsed)
        self.bytes.append(self.console.written - written)
        screen.frame_time, screen.measured_fps = elapsed, 1.0 / self.dt if self.dt > 0 else None
        screen.frame_count += 1
        if result is False and before is not None:
            screen.chars[...], screen.fg[...], screen.bg[...], screen.attrs[...] = before
        if self._is_blank():
            self.blank.append(self.frames)
        if result is False:
            self.quit, self.quit_at = True, self.frames
        return result

    def _playing(self):
        if self.quit:
            raise PlaytestError(f"the program already quit: frame() returned False at frame {self.quit_at}")

    def _tap(self, codes):
        """One frame with these keys pressed, each let go with the next frame (unless held with down())."""
        result = self.step(codes)
        self._queued += [KeyRelease(c) for c in dict.fromkeys(codes) if c not in self._held]
        return result

    def wait(self, frames=1):
        """Frames with no new input (held keys stay held). Stops early if the program quits."""
        self._playing()
        for _ in range(int(frames)):
            if self.step() is False:
                break
        return self

    def press(self, *keys):
        """One frame with these keys pressed (see parse_key for how to name them), let go with the next frame:
        `press("enter")`, `press("a", "b")` for two in the same frame."""
        self._tap([parse_key(k) for k in keys])
        return self

    def type(self, text):
        """Each character of text pressed in a frame of its own."""
        self._playing()
        for ch in text:
            if self._tap([ord(ch)]) is False:
                break
        return self

    def down(self, *keys):
        """Press keys and keep them held until up(): screen.held has them, and they repeat as a terminal repeats
        a held key. The press goes with the next frame."""
        self._playing()
        for code in map(parse_key, keys):
            if code not in self._held:
                self._held[code] = self.now
                self._queued.append(code)
        return self

    def up(self, *keys):
        """Let go of keys held with down() (all of them with none named). The release goes with the next frame."""
        for code in [parse_key(k) for k in keys] if keys else list(self._held):
            if self._held.pop(code, None) is not None:
                self._queued.append(KeyRelease(code))
        return self

    def hold(self, *keys, frames=1):
        """Hold keys down for this many frames, then let go: frames + 1 frames in all, the last with the release
        (so the program has seen it by the time hold() returns)."""
        self.down(*keys)
        self.wait(frames)
        self.up(*keys)
        if not self.quit:
            self.step()
        return self

    def click(self, x, y, button="left"):
        """One frame with a mouse click (press and release) on cell (x, y): column x, row y, from 0.
        button: "left", "middle", "right", "wheel_up" or "wheel_down"."""
        b = _button(button)
        events = [MouseEvent(int(x), int(y), b, True)] + ([MouseEvent(int(x), int(y), b, False)] if b < 64 else [])
        self.step(events)
        return self

    def click_text(self, text, button="left"):
        """Click the middle of the first place text is on the screen (a button's label, say). Raises PlaytestError
        if it isn't there."""
        self._playing()
        found = self.find(text)
        if found is None:
            raise PlaytestError(self._missing(text))
        return self.click(found.col + len(text) // 2, found.row, button)

    def move(self, x, y):
        """One frame with the mouse moved to cell (x, y), no button held (for programs run with mouse="move")."""
        self.step([MouseEvent(int(x), int(y), MouseEvent.NONE, False, True)])
        return self

    def resize(self, rows, cols):
        """Change the pretend terminal's size, rows then columns as Screen(size=...) takes them; the program sees
        it from the next frame, as under run()."""
        self.console.rows, self.console.cols = int(rows), int(cols)
        return self

    # ----- looking -------------------------------------------------------------------------

    def text(self, picture=True):
        """The screen's characters as text, a line per row (trailing spaces left out). picture=False leaves out the
        block characters 3D frames are drawn with (with Unicode glyph sets; ASCII pictures stay), keeping the
        words and numbers. For the picture itself as text, play with glyphs="ascii": shading as characters."""
        chars = self.screen.chars
        if not picture and self.screen.mode != "ascii":
            chars = np.where(np.isin(chars, GLYPH_SETS[self.screen.mode].chars), " ", chars)
        return "\n".join("".join(row).rstrip() for row in chars)

    def find(self, text):
        """Where text first is on the screen, as Found(row, col), or None."""
        for y, row in enumerate(self.screen.chars):
            x = "".join(row).find(text)
            if x >= 0:
                return Found(y, x)
        return None

    def expect(self, text):
        """Raise PlaytestError (saving a shot) unless text is on the screen."""
        if self.find(text) is None:
            raise PlaytestError(self._missing(text))
        return self

    def expect_not(self, text):
        """Raise PlaytestError (saving a shot) if text is on the screen."""
        found = self.find(text)
        if found is not None:
            path = self._save_picture(f"unexpected-frame-{self.frames}")
            raise PlaytestError(f"{text!r} is on the screen at row {found.row}, column {found.col} after frame "
                                f"{self.frames} (see {path})")
        return self

    def _missing(self, text):
        path = self._save_picture(f"missing-frame-{self.frames}")
        return (f"{text!r} is not on the screen after frame {self.frames} (see {path}); the words on it:\n"
                f"{self.text(picture=False)}")

    def picture(self):
        """The screen as the terminal shows it: (rows * 16, cols * 8, 3) uint8 (see Screen.picture)."""
        return self.screen.picture()

    def shot(self, name):
        """Save the screen as <out>/<name>.png, and its text as <out>/<name>.txt. Returns the picture's path."""
        path = self._save_picture(name)
        with open(os.path.join(self.out, f"{name}.txt"), "w", encoding="utf-8") as f:
            f.write(self.text() + "\n")
        self.shots.append(os.path.join(self.out, f"{name}.txt"))
        return path

    def sheet(self, name, frames, shots=6, columns=3):
        """Play `frames` frames (no new input, held keys stay held), taking `shots` pictures evenly through them,
        and save them as one contact sheet, <out>/<name>.png, each labelled with its frame and time. For
        movement and animation. Returns the path."""
        frames, shots = max(int(frames), 1), max(int(shots), 1)
        when = {max(1, round(frames * (k + 1) / shots)) for k in range(shots)}  # (the last at the end)
        tiles, labels = [], []
        for i in range(1, frames + 1):
            stopped = self.step() is False
            if i in when or stopped:
                tiles.append(self.picture())
                labels.append(f"frame {self.frames}  {self.now:.2f}s")
            if stopped:
                break
        return self._save(name, contact_sheet(tiles, labels, columns))

    def _is_blank(self):
        s = self.screen
        return not (s.chars != " ").any() and not (s.bg != s._bg_cell).any()

    def _save_picture(self, name):
        return self._save(name, self.picture())

    def _save(self, name, image):
        from PIL import Image
        os.makedirs(self.out, exist_ok=True)
        path = os.path.join(self.out, f"{name}.png")
        Image.fromarray(image).save(path)
        self.shots.append(path)
        return path

    # ----- reporting -----------------------------------------------------------------------

    def report(self):
        """A Report of the frames so far; print it for a summary."""
        rest = np.array(self.times[1:]) * 1000.0
        sent = np.array(self.bytes[1:])
        notes = dict(mean_ms=None, p95_ms=None, max_ms=None, slowest_frame=None, mean_bytes=None, max_bytes=None)
        if len(rest):
            notes = dict(mean_ms=float(rest.mean()), p95_ms=float(np.percentile(rest, 95)), max_ms=float(rest.max()),
                         slowest_frame=int(rest.argmax()) + 2, mean_bytes=float(sent.mean()),
                         max_bytes=int(sent.max()))
        warned = sorted((frame, message) for message, frame in self.warnings.items())
        return Report(frames=self.frames, seconds=self.now, size=self.screen.size(), glyphs=self.screen.mode,
                      color=self.screen.color_mode, quit_at=self.quit_at,
                      first_ms=self.times[0] * 1000.0 if self.times else None, blank_frames=list(self.blank),
                      warnings=warned, shots=list(self.shots), **notes)


def contact_sheet(tiles, labels=None, columns=3, background=(12, 12, 16)):
    """Pictures ((h, w, 3) uint8, sizes may differ) laid out in rows of `columns`, each with its label (if any)
    above it in the terminal font, on grey gaps. Returns (H, W, 3) uint8."""
    if not len(tiles):
        raise ValueError("no pictures for the contact sheet")
    labels = list(labels) if labels is not None else [None] * len(tiles)
    cells = []
    for tile, label in zip(tiles, labels):
        tile = np.asarray(tile, np.uint8)
        if label:
            strip = Screen(size=(1, max(tile.shape[1] // 8, 1)), glyphs="ascii", color="truecolor")
            strip.text(0, 0, str(label))
            pic = strip.picture(bg=background)[:, :tile.shape[1]]
            pad = np.zeros((pic.shape[0], tile.shape[1] - pic.shape[1], 3), np.uint8) + np.uint8(background)
            tile = np.concatenate([np.concatenate([pic, pad], axis=1), tile], axis=0)
        cells.append(tile)
    columns = max(1, min(int(columns), len(cells)))
    rows = [cells[i:i + columns] for i in range(0, len(cells), columns)]
    cell_h = [max(c.shape[0] for c in row) for row in rows]
    cell_w = [max(rows[r][c].shape[1] for r in range(len(rows)) if c < len(rows[r])) for c in range(columns)]
    height = sum(cell_h) + SHEET_GAP * (len(rows) + 1)
    width = sum(cell_w) + SHEET_GAP * (columns + 1)
    out = np.full((height, width, 3), GAP_COLOR, np.uint8)
    y = SHEET_GAP
    for r, row in enumerate(rows):
        x = SHEET_GAP
        for c, cell in enumerate(row):
            out[y:y + cell.shape[0], x:x + cell.shape[1]] = cell
            x += cell_w[c] + SHEET_GAP
        y += cell_h[r] + SHEET_GAP
    return out


def preview(model, path=None, clips=None, shots=5, yaw=25.0, cells=36, glyphs="sextant", pixels=False,
            background=(28, 26, 30)):
    """A contact sheet of a model's animations as a terminal shows them: a row per clip (or one picture of a model
    without any), `shots` moments across each, the camera `yaw` degrees around the model and a little above it,
    each tile `cells` terminal cells wide (36: a character seen close up; 14: one among many on a board).

    model: a Model or a file load_model() takes. clips: the names of the clips to show (default: all).
    pixels: draw raw square pixels instead of terminal cells (more detail than any terminal shows; for checking
    geometry, not looks). Saves it to path if given. Returns the sheet, (H, W, 3) uint8."""
    from .color import linear_to_srgb
    from .lights import Light
    from .models import load_model
    from .scene import Camera, Renderer
    if isinstance(model, (str, os.PathLike)):
        model = load_model(model)
    objects = [*model]
    if not objects:
        raise ValueError("the model has nothing to draw")
    lo, hi = model.world_bounds()
    height = max(float(hi[1] - lo[1]), float(np.max(hi - lo)) * 0.5, 0.2)
    centre = (lo + hi) / 2
    animations = getattr(model, "animations", {}) or {}
    unknown = [n for n in clips or () if n not in animations]
    if unknown:
        raise ValueError(f"no clips named {unknown}; the model has {list(animations) or 'none'}")
    names = [n for n in animations if not clips or n in clips] or [None]
    a = np.radians(yaw)
    eye = centre + np.array([height * 1.9 * np.sin(a), height * 0.35, height * 1.9 * np.cos(a)])
    camera = Camera(position=eye, target=centre, fov=40)
    lights = [Light(direction=np.array([-0.5, -1.0, -0.4]), ambient=0.25, diffuse=0.8, color=(255, 236, 205),
                    shadows=True),
              Light(direction=np.array([0.7, -0.3, 0.6]), ambient=0.0, diffuse=0.35, color=(150, 170, 255))]
    if pixels:
        renderer = Renderer(240, 240, (1, 1), cell_aspect=1.0, background=background)
    else:
        cell_pixels = GLYPH_SETS[glyphs].cell_pixels
        rows = max(int(cells * 0.62), 1)
        renderer = Renderer(cells, rows, cell_pixels, background=background)
        screen = Screen(size=(rows, cells), glyphs=glyphs, color="truecolor", background=background)
    tiles, labels = [], []
    for name in names:
        clip = animations.get(name) if name else None
        for k in range(shots if clip else 1):
            t = 0.0
            if clip:
                t = clip.duration * k / max(1, shots - 1) if clip.loop == "once" else clip.duration * k / shots
                clip.apply(t)
            fb = renderer.render(objects, camera, lights)
            if pixels:
                tiles.append(np.rint(linear_to_srgb(np.clip(fb.colour(), 0, 1)) * 255).astype(np.uint8))
            else:
                screen.erase()
                screen.draw_frame(fb)
                tiles.append(screen.picture())
            labels.append(f"{name} {t:.2f}s" if clip else "(no animations)")
    sheet = contact_sheet(tiles, labels, columns=shots if len(names) > 1 or names[0] else 1)
    if path is not None:
        from PIL import Image
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        Image.fromarray(sheet).save(path)
    return sheet


# ----- scripts and the command line ------------------------------------------------------------

SCRIPT_HELP = """\
A script is commands separated by ";" or new lines (# starts a comment); arguments with spaces go in quotes.
Positions and sizes go columns first, then rows, as on the command line (--size 120x40).
  wait [N]            N frames (default 1) with no new input
  press KEY...        one frame with these keys pressed, let go with the next frame (a character, or a name:
                      enter, esc, space, tab, backspace, up, down, left, right, f1..f12, ctrl+c, ...)
  type TEXT           each character pressed in a frame of its own
  hold KEY... N       the keys held down for N frames (repeating as a terminal repeats them), then a frame
                      with them let go
  down KEY... / up [KEY...]
                      press and keep holding / let go (all with none named)
  click X Y [BUTTON]  a click on column X, row Y, from 0 (left, middle, right, wheel_up, wheel_down)
  click TEXT          a click on the middle of TEXT where it is on the screen
  move X Y            the mouse moved to column X, row Y
  resize COLSxROWS    the terminal resized (e.g. resize 80x24)
  glyphs MODE / color MODE
                      switch the glyph set / colour mode, as the F2 and F3 keys do
  shot NAME           save the screen as NAME.png and NAME.txt
  sheet NAME N [SHOTS]
                      play N frames and save SHOTS pictures (default 6) through them as one contact sheet
  expect TEXT / expect_not TEXT
                      stop with an error unless TEXT is (is not) on the screen
  dump [words]        print the screen as text (words: without the 3D picture's block characters)
After the program quits, the commands that play frames stop the script with an error; expect, expect_not, dump,
shot and the others still look at the last screen."""

# Commands that play frames, and so can't follow the program quitting.
PLAYING = {"wait", "press", "type", "hold", "down", "click", "move", "sheet"}


def _commands(script):
    """The script's commands, each a list of words."""
    commands = []
    for line in script.splitlines():
        lexer = shlex.shlex(line, posix=True, punctuation_chars=";")
        lexer.whitespace_split, lexer.commenters = True, "#"
        words = []
        try:
            for token in lexer:
                if token and set(token) == {";"}:
                    if words:
                        commands.append(words)
                    words = []
                else:
                    words.append(token)
        except ValueError as e:  # (an unclosed quote)
            raise ValueError(f"can't read the script line {line!r}: {e}") from None
        if words:
            commands.append(words)
    return commands


def _count(word, what="a number of frames"):
    try:
        n = int(word)
    except ValueError:
        raise ValueError(f"{word!r} is not {what}") from None
    if n < 0:
        raise ValueError(f"{word!r} is not {what}")
    return n


def _position(words):
    try:
        x, y = (int(w) for w in words)
    except ValueError:
        raise ValueError(f"{' '.join(words)!r} is not a column and a row") from None
    return x, y


def _parse_size(text):
    """(rows, cols) from COLSxROWS."""
    try:
        cols, rows = (int(n) for n in text.lower().split("x"))
    except ValueError:
        raise ValueError(f"size as COLSxROWS (e.g. 120x40), not {text!r}") from None
    if rows < 1 or cols < 1:
        raise ValueError(f"size as COLSxROWS, each at least 1, not {text!r}")
    return rows, cols


def _parse_command(words, echo):
    """A command's words as a function of a Playtest that carries it out, with every argument checked."""
    cmd, args = words[0], words[1:]

    def needs(low, high=None):
        if len(args) < low or (high is not None and len(args) > high):
            n = f"{low}" if high == low else f"{low} to {high}" if high is not None else f"at least {low}"
            raise ValueError(f"{cmd} takes {n} argument{'s' if (high or low) != 1 else ''}, not {len(args)}")

    if cmd == "wait":
        needs(0, 1)
        n = _count(args[0]) if args else 1
        return lambda t: t.wait(n)
    if cmd in ("press", "down"):
        needs(1)
        keys = [parse_key(k) for k in args]
        return (lambda t: t.press(*keys)) if cmd == "press" else (lambda t: t.down(*keys))
    if cmd == "up":
        keys = [parse_key(k) for k in args]
        return lambda t: t.up(*keys)
    if cmd == "type":
        needs(1)
        text = " ".join(args)
        return lambda t: t.type(text)
    if cmd == "hold":
        needs(2)
        keys, n = [parse_key(k) for k in args[:-1]], _count(args[-1])
        return lambda t: t.hold(*keys, frames=n)
    if cmd == "click":
        needs(1)
        try:
            x, y = _position(args[:2])
        except ValueError:
            if args[0].lstrip("-").isdigit():
                raise ValueError("click takes a column and a row (click 10 5 [BUTTON]), or text to click on")
            text = " ".join(args)
            return lambda t: t.click_text(text)
        needs(2, 3)
        button = args[2] if len(args) > 2 else "left"
        _button(button)
        return lambda t: t.click(x, y, button)
    if cmd == "move":
        needs(2, 2)
        x, y = _position(args)
        return lambda t: t.move(x, y)
    if cmd == "resize":
        if len(args) != 1:
            raise ValueError("resize takes the size as COLSxROWS, e.g. resize 80x24")
        rows, cols = _parse_size(args[0])
        return lambda t: t.resize(rows, cols)
    if cmd == "glyphs":
        needs(1, 1)
        if args[0] not in GLYPH_MODES:
            raise ValueError(f"glyphs must be one of {', '.join(GLYPH_MODES)}, not {args[0]!r}")
        return lambda t: t.screen.set_glyphs(args[0])
    if cmd == "color":
        needs(1, 1)
        if args[0] not in COLOR_MODES:
            raise ValueError(f"color must be one of {', '.join(COLOR_MODES)}, not {args[0]!r}")
        return lambda t: t.screen.set_color(args[0])
    if cmd == "shot":
        needs(1, 1)
        return lambda t: t.shot(args[0])
    if cmd == "sheet":
        needs(2, 3)
        n = _count(args[1])
        shots = _count(args[2], "a number of pictures") if len(args) > 2 else 6
        if not n or not shots:
            raise ValueError("a sheet needs at least one frame and one picture")
        return lambda t: t.sheet(args[0], n, shots)
    if cmd in ("expect", "expect_not"):
        needs(1)
        text = " ".join(args)
        return (lambda t: t.expect(text)) if cmd == "expect" else (lambda t: t.expect_not(text))
    if cmd == "dump":
        needs(0, 1)
        if args and args[0] != "words":
            raise ValueError(f"dump takes nothing or 'words', not {args[0]!r}")
        words_only = bool(args)
        return lambda t: echo(f"--- {'words on the ' if words_only else ''}screen after frame {t.frames} ---\n"
                              f"{t.text(picture=not words_only)}\n---")
    raise ValueError(f"unknown command {cmd!r}")


def parse_script(script, echo=print):
    """The script's commands (see SCRIPT_HELP) as (words, plays frames, function of a Playtest) for each, every
    argument checked: a ValueError for the first that's wrong, before anything plays. echo: what dump prints with."""
    parsed = []
    for words in _commands(script):
        try:
            parsed.append((words, words[0] in PLAYING, _parse_command(words, echo)))
        except ValueError as e:
            raise ValueError(f"bad command {' '.join(words)!r}: {e}\n\n{SCRIPT_HELP}") from None
    return parsed


def run_script(test, script, echo=print):
    """Play a script (see SCRIPT_HELP) on a Playtest. Raises ValueError for a bad command (before playing
    anything) and PlaytestError as the Playtest does, or for a command that plays frames after the program quit."""
    for words, plays, command in parse_script(script, echo):
        if plays and test.quit:
            raise PlaytestError(f"the program quit at frame {test.quit_at}, before {' '.join(words)!r}")
        try:
            command(test)
        except ValueError as e:  # (one only the screen can tell: glyphs needing Unicode on a terminal without it)
            raise ValueError(f"bad command {' '.join(words)!r}: {e}") from e
    return test


def load_target(target):
    """The frame function named by "file.py:Name" or "package.module:Name". Name may be a frame function
    (screen, dt, events), an object with a frame() method, or a class or function that makes one (a game's
    class, say): called with no arguments, or with those given as Python literals, "game.py:Game(seed=1)"."""
    call = None
    if target.endswith(")") and "(" in target:
        target, _, rest = target.partition("(")
        try:
            call = ast.parse(f"f({rest}", mode="eval").body
            args = [ast.literal_eval(a) for a in call.args]
            kwargs = {k.arg: ast.literal_eval(k.value) for k in call.keywords}
        except (SyntaxError, ValueError) as e:
            raise ValueError(f"the arguments in {target}({rest} must be Python literals (numbers, strings, "
                             f"lists, ...): {e}") from None
    where, sep, name = target.rpartition(":")
    dotted = r"[A-Za-z_]\w*(\.[A-Za-z_]\w*)*"
    if not sep or not where or not re.fullmatch(dotted, name):   # (C:\game.py splits at the drive's colon)
        raise ValueError(f"name the program as file.py:Name or package.module:Name, not {target!r}")
    if where.endswith(".py") or os.sep in where or "/" in where:
        path = os.path.abspath(where)
        if not os.path.isfile(path):
            raise ValueError(f"no file {where}")
        stem = os.path.splitext(os.path.basename(path))[0]
        loaded = sys.modules.get(stem)
        if stem in sys.stdlib_module_names or (loaded is not None and getattr(loaded, "__file__", None) != path):
            raise ValueError(f"{os.path.basename(path)} has the name of a module Python already has ({stem}): "
                             f"rename the file, or the program and that module would hide each other")
        sys.path.insert(0, os.path.dirname(path))  # (for the program's own imports)
        spec = importlib.util.spec_from_file_location(stem, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    else:
        sys.path.insert(0, os.getcwd())
        module = importlib.import_module(where)
    obj = module
    for part in name.split("."):
        if not hasattr(obj, part):
            raise ValueError(f"{where} has no {name}")
        obj = getattr(obj, part)
    if call is not None:
        obj = obj(*args, **kwargs)
    elif inspect.isclass(obj):
        try:
            inspect.signature(obj).bind()
        except TypeError as e:
            raise ValueError(f"{name}() needs arguments ({e}): give them as {target}(...), Python literals") from None
        obj = obj()
    elif callable(obj) and not hasattr(obj, "frame"):
        required = [p for p in inspect.signature(obj).parameters.values()
                    if p.default is p.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        if not required:
            obj = obj()
    frame = getattr(obj, "frame", obj)
    if not callable(frame):
        raise ValueError(f"{target} is not a frame function or something with a frame() method")
    return frame


def _size(text):
    try:
        return _parse_size(text)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from None


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m unicode3d.playtest", description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    play = sub.add_parser("play", help="play a program with a script", description="Play a program off-screen with a "
                          "script, save what it showed, and print a report.\n\n" + SCRIPT_HELP,
                          formatter_class=argparse.RawDescriptionHelpFormatter)
    play.add_argument("target", help="the program: file.py:Name or package.module:Name (a frame function, an "
                      "object with frame(), or a class or function making one with no arguments)")
    play.add_argument("script", nargs="?", default="wait 30", help="the commands (default: wait 30)")
    play.add_argument("-f", "--script-file", help="read the commands from this file instead")
    play.add_argument("--size", type=_size, default=(40, 120), help="COLSxROWS of the terminal (default 120x40)")
    play.add_argument("--glyphs", choices=GLYPH_MODES, default="sextant")
    play.add_argument("--color", choices=COLOR_MODES, default="truecolor")
    play.add_argument("--dt", type=float, default=1 / 30, help="seconds per frame (default 1/30)")
    play.add_argument("--no-unicode", action="store_true", help="a terminal without Unicode")
    play.add_argument("--no-releases", action="store_true", help="a terminal that reports no key releases (most "
                      "do not): held keys are guessed from presses and repeats")
    play.add_argument("--key-release", action="store_true", help="pass KeyRelease events, as run(key_release=True)")
    play.add_argument("--no-erase", action="store_true", help="keep the screen from frame to frame, as run(erase=False)")
    play.add_argument("--no-check-edits", action="store_true", help="don't check meshes for arrays edited in place "
                      "without invalidate() (checksums of the meshes' arrays each frame)")
    play.add_argument("--out", default="playtest", help="folder for the pictures (default: playtest)")
    view = sub.add_parser("preview", help="contact sheets of models' animations",
                          description=preview.__doc__.split("\n\n")[0])
    view.add_argument("models", nargs="+")
    view.add_argument("--clips", nargs="*", help="only these clips")
    view.add_argument("--shots", type=int, default=5, help="moments across each clip (default 5)")
    view.add_argument("--yaw", type=float, default=25.0, help="camera angle around the model, degrees")
    view.add_argument("--cells", type=int, default=36, help="tile width in terminal cells (default 36)")
    view.add_argument("--glyphs", choices=GLYPH_MODES, default="sextant")
    view.add_argument("--pixels", action="store_true", help="raw square pixels instead of terminal cells")
    view.add_argument("--out", default="playtest", help="folder for the sheets (default: playtest)")
    args = parser.parse_args(argv)

    if args.command == "preview":
        from .models import load_model
        for m in args.models:
            model = load_model(m)
            path = os.path.join(args.out, os.path.splitext(os.path.basename(m))[0] + ("-px" if args.pixels else "")
                                + ".png")
            preview(model, path, args.clips, args.shots, args.yaw, args.cells, args.glyphs, args.pixels)
            clips = ", ".join(f"{n} {c.duration:.2f}s" for n, c in model.animations.items()) or "no animations"
            lo, hi = model.world_bounds()
            print(f"{path}: {clips}; {len(model)} parts, {sum(len(o.mesh.faces) for o in model)} triangles, "
                  f"size {' x '.join(f'{v:.2f}' for v in hi - lo)}")
            for warning in model.warnings:
                print(f"  {m}: {warning}")
        return 0

    try:
        script = args.script
        if args.script_file:
            with open(args.script_file, encoding="utf-8") as f:
                script = f.read()
        parse_script(script)  # (a script with a bad command fails before the program loads)
        frame = load_target(args.target)
    except (OSError, ValueError) as e:
        print(f"playtest: {e}", file=sys.stderr)
        return 2
    test = Playtest(frame, size=args.size, glyphs=args.glyphs, color=args.color, dt=args.dt,
                    unicode=not args.no_unicode, releases=not args.no_releases, key_release=args.key_release,
                    erase=not args.no_erase, out=args.out, check_edits=not args.no_check_edits)
    status = 0
    try:
        run_script(test, script)
    except ValueError as e:
        status = 2
        print(f"playtest: {e}", file=sys.stderr)
    except PlaytestError as e:
        status = 1
        cause = e.__cause__
        if cause is not None:
            traceback.print_exception(type(cause), cause, cause.__traceback__)
        print(f"playtest: {e}", file=sys.stderr)
    finally:
        if test.frames:
            test.shot("last")
        print(test.report())
    return status


if __name__ == "__main__":
    sys.exit(main())
