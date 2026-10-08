# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
"""Text widgets for the screen: buttons, toggles, sliders and choices, laid out in a Panel.

A Panel draws its widgets with Screen.text and remembers where each one went,
so the next frame's mouse events can find them; handle() takes a frame's
events, acts on the ones meant for the panel and returns the rest:

    panel = Panel([Slider("Balls", 20, 1, 200, keys="[]"), Toggle("Spin", True, key="s"), Button("Reset", reset)])

    def frame(screen, dt, events):
        events = panel.handle(events)          # clicks, drags, the widgets' keys, Tab to move the focus
        ...
        panel.draw(screen, rows - 1, 0)

Each widget reacts to a click on it, to the mouse wheel over it, and to its own
key (`key`, or a (decrease, increase) pair for sliders). With keyboard=True, Tab
and Shift-Tab move the focus from widget to widget, and then Left/Right,
Enter and Space work the focused one. Dragging a slider's knob needs mouse
moves: run(..., mouse="drag").

DisplayControls is the panel of display settings every program can offer
(glyphs on F2, colours on F3, frame rate on F4), as Zombie Dice has it.
"""
from .color import COLOR_MODES, Color
from .glyphs import GLYPH_MODES
from .keys import Key, MouseEvent
from .quality import ALIASES as QUALITY_ALIASES, MODES as QUALITY_MODES, AutoQuality

ACTIVATE = (Key.ENTER, 10, ord(" "))
FPS_STEPS = (30, 60, 120, 144)


def _code(key):
    """A key as the int events carry: a one-character string becomes its code."""
    return ord(key) if isinstance(key, str) else key


def cycle(options, current, step=1):
    """The option `step` places after `current`, wrapping round; for a number not listed, the next one up or down."""
    options = list(options)
    if current in options:
        return options[(options.index(current) + step) % len(options)]
    if step > 0:
        return next((o for o in options if o > current), options[0])
    return next((o for o in reversed(options) if o < current), options[-1])


class Widget:
    """Base class. `value` is kept by the widget, or read and written through get/set when given.

    on_change(value) is called whenever the user changes the value.
    """

    def __init__(self, label="", value=None, on_change=None, key=None, get=None, set=None):
        self.label = label
        self._value = value
        self.on_change = on_change
        self.key = key
        self._get, self._set = get, set

    @property
    def value(self):
        return self._get() if self._get is not None else self._value

    @value.setter
    def value(self, v):
        if self._set is not None:
            self._set(v)
        else:
            self._value = v

    def change(self, v):
        """Set the value as the user would, calling on_change if it changed."""
        if v != self.value:
            self.value = v
            if self.on_change is not None:
                self.on_change(self.value)

    @property
    def width(self):
        return len(self.text(False))

    def text(self, unicode):
        return self.label

    def draw(self, screen, y, x, focused, hovered):
        screen.text(y, x, self.text(screen.unicode), bold=hovered, reverse=focused)

    # What the widget does with input; each returns whether it used it.
    def shortcut(self, k):
        return False

    def press(self, k):
        """A key while the widget has the focus."""
        return False

    def click(self, dx, button):
        """A click (button 0, 1 or 2) or wheel step (64 up, 65 down) at column dx within the widget."""
        return False

    def drag(self, dx):
        return False


class Button(Widget):
    """Drawn as "[ label ]"; calls action() when clicked, when its key is pressed, or on Enter/Space with the focus."""

    def __init__(self, label, action, key=None):
        super().__init__(label, key=key)
        self.action = action

    def text(self, unicode):
        return f"[ {self.label} ]"

    def shortcut(self, k):
        if self.key is not None and k == _code(self.key):
            self.action()
            return True
        return False

    def press(self, k):
        if k in ACTIVATE:
            self.action()
            return True
        return False

    def click(self, dx, button):
        if button == MouseEvent.LEFT:
            self.action()
            return True
        return False


class Toggle(Widget):
    """An on/off setting drawn as "[x] label" / "[ ] label"; flipped by a click, its key, or Enter/Space/arrows."""

    def __init__(self, label, value=False, on_change=None, key=None, get=None, set=None):
        super().__init__(label, bool(value), on_change, key, get, set)

    def text(self, unicode):
        return f"[{'x' if self.value else ' '}] {self.label}"

    def draw(self, screen, y, x, focused, hovered):
        screen.text(y, x, f"[{'x' if self.value else ' '}]", Color.GREEN if self.value else Color.DEFAULT,
                    bold=True, reverse=focused)
        screen.text(y, x + 3, " ")
        screen.text(y, x + 4, self.label, bold=hovered, dim=not self.value)

    def flip(self):
        self.change(not self.value)

    def shortcut(self, k):
        if self.key is not None and k == _code(self.key):
            self.flip()
            return True
        return False

    def press(self, k):
        if k in ACTIVATE or k in (Key.LEFT, Key.RIGHT):
            self.flip()
            return True
        return False

    def click(self, dx, button):
        if button in (MouseEvent.LEFT, MouseEvent.RIGHT):
            self.flip()
            return True
        return False


class Slider(Widget):
    """A number from lo to hi in steps of `step`, drawn as "label ━━━━●──── value".

    Set by clicking or dragging along the track, the mouse wheel over it, Left/Right
    with the focus, or its keys: a (decrease, increase) pair such as "[]" or "-+".
    length: the track's length in cells. fmt formats the value shown after it.
    """

    def __init__(self, label, value, lo, hi, step=1, on_change=None, keys=None, length=16, fmt=None,
                 get=None, set=None):
        super().__init__(label, None, on_change, keys, get, set)
        self.lo, self.hi, self.step, self.length = lo, hi, step, max(length, 2)
        self.fmt = fmt or (lambda v: f"{v:g}")
        if get is None:
            self._value = self.snap(value)

    def snap(self, v):
        """v clamped to lo..hi and rounded to a whole number of steps from lo (NaN as lo)."""
        v = min(max(v, self.lo), self.hi) if v == v else self.lo
        if self.step:
            v = self.lo + round((v - self.lo) / self.step) * self.step
            v = min(max(v, self.lo), self.hi)
            if isinstance(self.step, int) and isinstance(self.lo, int):
                v = int(v)
            else:
                v = round(v, 12)  # 0.2 + 17 * 0.1 is 1.9, not 1.9000000000000001
        return v

    def nudge(self, steps):
        self.change(self.snap(self.value + steps * (self.step or (self.hi - self.lo) / (self.length - 1))))

    @property
    def _prefix(self):
        return f"{self.label} " if self.label else ""

    @property
    def _value_width(self):
        return max(len(self.fmt(self.lo)), len(self.fmt(self.hi)))

    @property
    def width(self):
        return len(self._prefix) + self.length + 1 + self._value_width

    def knob(self):
        """Cell of the track the knob is drawn in."""
        span = self.hi - self.lo
        f = (self.value - self.lo) / span if span else 0.0
        f = min(max(f, 0.0), 1.0) if f == f else 0.0  # (a value from get= may be out of range, infinite or NaN)
        return round(f * (self.length - 1))

    def draw(self, screen, y, x, focused, hovered):
        filled, knob, empty = ("━", "●", "─") if screen.unicode else ("=", "O", "-")
        k = self.knob()
        screen.text(y, x, self._prefix, bold=hovered, reverse=focused)
        x += len(self._prefix)
        screen.text(y, x, filled * k, Color.CYAN)
        screen.text(y, x + k, knob, Color.WHITE, bold=True)
        screen.text(y, x + k + 1, empty * (self.length - k - 1), dim=True)
        screen.text(y, x + self.length, " ")
        screen.text(y, x + self.length + 1, f"{self.fmt(self.value):>{self._value_width}}", bold=True)

    def shortcut(self, k):
        if self.key is None:
            return False
        dec, inc = (_code(c) for c in self.key)
        if k == dec:
            self.nudge(-1)
        elif k == inc:
            self.nudge(1)
        else:
            return False
        return True

    def press(self, k):
        if k == Key.LEFT:
            self.nudge(-1)
        elif k == Key.RIGHT:
            self.nudge(1)
        else:
            return False
        return True

    def click(self, dx, button):
        if button == MouseEvent.WHEEL_UP:
            self.nudge(1)
        elif button == MouseEvent.WHEEL_DOWN:
            self.nudge(-1)
        elif button == MouseEvent.LEFT:
            self.drag(dx)
        else:
            return False
        return True

    def drag(self, dx):
        cell = min(max(dx - len(self._prefix), 0), self.length - 1)
        self.change(self.snap(self.lo + (self.hi - self.lo) * cell / (self.length - 1)))
        return True


class Choice(Widget):
    """One of a list of options, drawn as "label value" (padded to the longest option, so nothing moves).

    A click or its key picks the next option (a right click or the wheel goes back too),
    as do Left/Right/Enter with the focus. show(value) gives the text shown for an option.
    """

    def __init__(self, label, options, value=None, on_change=None, key=None, show=str, get=None, set=None,
                 width=None):
        super().__init__(label, options[0] if value is None else value, on_change, key, get, set)
        self.options, self.show = tuple(options), show
        self._width = width or max(len(show(o)) for o in self.options)

    def text(self, unicode):
        prefix = f"{self.label} " if self.label else ""
        return prefix + f"{self.show(self.value):<{self._width}}"[:self._width]

    def draw(self, screen, y, x, focused, hovered):
        prefix = f"{self.label} " if self.label else ""
        screen.text(y, x, prefix, dim=True, reverse=focused)
        screen.text(y, x + len(prefix), f"{self.show(self.value):<{self._width}}"[:self._width], bold=hovered,
                    reverse=focused)

    def step(self, n):
        self.change(cycle(self.options, self.value, n))

    def shortcut(self, k):
        if self.key is not None and k == _code(self.key):
            self.step(1)
            return True
        return False

    def press(self, k):
        if k in ACTIVATE or k == Key.RIGHT:
            self.step(1)
        elif k == Key.LEFT:
            self.step(-1)
        else:
            return False
        return True

    def click(self, dx, button):
        if button in (MouseEvent.LEFT, MouseEvent.WHEEL_DOWN):
            self.step(1)
        elif button in (MouseEvent.RIGHT, MouseEvent.WHEEL_UP):
            self.step(-1)
        else:
            return False
        return True


class Panel:
    """Widgets in a row (or a column, with vertical=True), `gap` cells apart.

    keyboard: Tab and Shift-Tab move the focus through the widgets, and the
    focused one takes Left/Right/Enter/Space. Esc is never taken, so it stays
    free for quitting. Programs whose arrow keys do something else can leave
    keyboard off and rely on the widgets' own keys and the mouse.
    clear: blank the panel's area (with a cell of margin at the sides) before
    drawing, so it reads cleanly over a rendered frame.
    """

    def __init__(self, widgets, vertical=False, gap=2, keyboard=True, clear=True):
        self.widgets = list(widgets)
        self.vertical, self.gap, self.keyboard, self.clear = vertical, gap, keyboard, clear
        self.focus = None     # index of the focused widget, or None
        self._areas = []      # (widget, y, x, width) as last drawn
        self._dragging = None  # (widget, x) while the mouse drags a widget
        self._hover = None

    @property
    def width(self):
        ws = [w.width for w in self.widgets]
        if not ws:
            return 0
        return max(ws) if self.vertical else sum(ws) + self.gap * (len(ws) - 1)

    @property
    def height(self):
        return len(self.widgets) if self.vertical else 1

    def draw(self, screen, y, x):
        """Draw the widgets with the first at row y, column x, and remember where they are for handle()."""
        self._areas = []
        if self.clear:
            for row in range(y, y + self.height):
                screen.text(row, x - 1, " " * (self.width + 2))
        for i, w in enumerate(self.widgets):
            w.draw(screen, y, x, self.focus == i, self._hover is w)
            self._areas.append((w, y, x, w.width))
            if self.vertical:
                y += 1
            else:
                x += w.width + self.gap

    def _at(self, ev):
        for w, y, x, width in self._areas:
            if ev.y == y and x <= ev.x < x + width:
                return w, x
        return None, 0

    def handle(self, events):
        """Act on the events meant for the panel (as last drawn); returns the others, in order."""
        rest = []
        for ev in events:
            if not self._take(ev):
                rest.append(ev)
        return rest

    def _take(self, ev):
        if isinstance(ev, MouseEvent):
            if ev.moved:
                if self._dragging is not None and ev.pressed:
                    w, x = self._dragging
                    return w.drag(ev.x - x)
                self._hover = self._at(ev)[0]
                return self._hover is not None
            if not ev.pressed and ev.button == MouseEvent.LEFT and self._dragging is not None:
                self._dragging = None
                return True
            if not ev.pressed:
                return False
            w, x = self._at(ev)
            if w is None:
                return False
            if ev.button == MouseEvent.LEFT:
                self._dragging = (w, x)
            return w.click(ev.x - x, ev.button)
        if not isinstance(ev, int):
            return False
        for w in self.widgets:
            if w.shortcut(ev):
                return True
        if not self.keyboard:
            return False
        if ev in (Key.TAB, Key.BACK_TAB) and self.widgets:
            step = 1 if ev == Key.TAB else -1
            self.focus = (step if step < 0 else 0) % len(self.widgets) if self.focus is None \
                else (self.focus + step) % len(self.widgets)
            return True
        if self.focus is not None and self.focus < len(self.widgets):
            return self.widgets[self.focus].press(ev)
        return False


class DisplayControls(Panel):
    """The display settings, each clickable and on a function key: glyphs (F2), colours (F3), frame rate (F4),
    and, if given a Renderer, shadows (F5), reflections (F6), detail (F7) and quality (F8), which it switches with
    renderer.shadows, renderer.reflections and renderer.simplify: "standard" detail draws objects small on screen
    from simpler copies of their meshes that differ by at most a pixel (levels of detail, Renderer.simplify 1),
    "high" every mesh as it is (simplify 0). Quality (quality.AutoQuality, as `self.auto_quality`): "high" draws
    as the other settings say (the default); "mid" takes the first steps down (edge samples off, detail 2 px),
    keeping the shading; "low" holds the lowest step (coarse shading, the picture drawn smaller too); "auto" steps the
    picture down below the settings while frames take longer than the frame rate allows and run under
    AutoQuality.min_fps (20), and back up when there is time (shown as "auto -n", n steps down). Settings saved
    with the old name "fast" are taken as "low". Shadows and reflections are left as they are set. Frames are timed by run()
    (screen.frame_time), once a frame when the controls are drawn or handled. quality: the mode to start in (None:
    the renderer's as it is; DisplayControls on one renderer share its AutoQuality.of()).

    The frame rate shows as achieved/target. Draw it every frame (it needs the
    screen it changes), e.g. at the bottom right:

        controls.draw(screen, rows - 1, cols - controls.width - 1)

    show: the names of the widgets to draw, of SETTINGS (all by default); the others still answer their keys and
    are in settings(). show=("fps",) draws only the frame rate, for a program with a settings page of its own.
    If it is never drawn, pass the screen to handle() each frame instead.

    settings() gives the values as {name: value}, ready for JSON, and apply() takes them back (on another run,
    say: keeping them in a file is the program's part).
    """

    SETTINGS = ("glyphs", "color", "fps", "shadows", "reflections", "detail", "quality")
    DETAIL = {"standard": 1.0, "high": 0.0}  # each detail setting's Renderer.simplify

    def __init__(self, fps_steps=FPS_STEPS, keyboard=False, renderer=None, show=None, quality=None):
        self.screen = None
        self.renderer = renderer
        self.auto_quality = AutoQuality.of(renderer) if renderer is not None else None  # (shared by the renderer's)
        if quality is not None and self.auto_quality is not None:
            self.auto_quality.mode = quality
        self._timed = None  # the screen's frame_count when the quality was last given a frame time
        self._pending = {}  # values given to apply() before the screen was known
        steps = tuple(fps_steps)
        fps_width = len(f"999/{max(steps)}fps")

        def fps_text(target):
            measured = self.screen.measured_fps if self.screen is not None else None
            return f"{'--' if measured is None else f'{min(measured, 999):.0f}'}/{target}fps"

        self.glyphs = Choice("F2", GLYPH_MODES, key=Key.F2,
                             get=lambda: self.screen.mode if self.screen else "sextant", set=self._set_glyphs)
        self.color = Choice("F3", COLOR_MODES, key=Key.F3,
                            get=lambda: self.screen.color_mode if self.screen else COLOR_MODES[0],
                            set=lambda v: self.screen and self.screen.set_color(v))
        self.fps = Choice("F4", steps, key=Key.F4, show=fps_text, width=fps_width,
                          get=lambda: self.screen.fps if self.screen else steps[0], set=self._set_fps)
        widgets = [self.glyphs, self.color, self.fps]
        if renderer is not None:
            self.shadows = Choice("F5", (True, False), key=Key.F5, show=lambda v: "shadows" if v else "no shadows",
                                  get=lambda: self.renderer.shadows, set=self._set_shadows)
            self.reflections = Choice("F6", (True, False), key=Key.F6,
                                      show=lambda v: "reflections" if v else "no reflections",
                                      get=lambda: self.renderer.reflections, set=self._set_reflections)
            self.detail = Choice("F7", tuple(self.DETAIL), key=Key.F7, show=lambda v: {"standard": "std detail"}.get(v, f"{v} detail"),
                                 get=lambda: "high" if not self.auto_quality.user("simplify") > 0 else "standard",
                                 set=self._set_detail)
            self.quality = Choice("F8", QUALITY_MODES, key=Key.F8, width=len("auto -5"), show=self._quality_text,
                                  get=lambda: self.auto_quality.mode, set=self._set_quality)
            widgets += [self.shadows, self.reflections, self.detail, self.quality]
        self._named = {name: getattr(self, name) for name in self.SETTINGS if hasattr(self, name)}
        unknown = set(show or ()) - set(self.SETTINGS)
        if unknown:
            raise ValueError(f"no such display setting: {', '.join(sorted(unknown))} (they are {self.SETTINGS})")
        shown = [w for w in widgets if show is None or any(self._named.get(n) is w for n in show)]
        self._hidden = [w for w in widgets if w not in shown]
        super().__init__(shown, keyboard=keyboard)

    def settings(self):
        """{name: value} of each setting (glyphs, color, fps, and with a renderer shadows, reflections and detail),
        as strings, numbers and booleans."""
        values = {name: w.value for name, w in self._named.items()}
        values.update({k: v for k, v in self._pending.items() if k in values})
        return values

    def apply(self, values):
        """Put settings from settings() into effect, skipping names it doesn't know and values this screen can't
        take (sextant glyphs on a terminal without Unicode, say). Before the screen is known (the first draw()
        or handle()), they wait for it."""
        for name, value in dict(values).items():
            if name == "quality" and isinstance(value, str):
                value = QUALITY_ALIASES.get(value, value)
            w = self._named.get(name)
            options = GLYPH_MODES if name == "glyphs" and self.screen is None else w.options if w else ()
            if not any(value == o and type(value) is type(o) for o in options):
                continue
            if self.screen is None and name in ("glyphs", "color", "fps"):  # (they are the screen's)
                self._pending[name] = value
                continue
            try:
                w.value = value
            except ValueError:
                pass

    def _take(self, ev):
        if isinstance(ev, int) and not isinstance(ev, bool):
            for w in self._hidden:
                if w.shortcut(ev):
                    return True
        return super()._take(ev)

    def _found_screen(self, screen):
        self.screen = screen
        self.glyphs.options = tuple(screen.glyph_modes)  # ascii only, without Unicode
        if self._pending:
            pending, self._pending = self._pending, {}
            self.apply(pending)

    def _set_shadows(self, v):
        self.renderer.shadows = v

    def _set_reflections(self, v):
        self.renderer.reflections = v

    def _set_detail(self, v):
        self.auto_quality.set_user("simplify", self.DETAIL[v])

    def _set_quality(self, v):
        self.auto_quality.mode = v

    def _quality_text(self, mode):
        auto = self.auto_quality
        level = auto.level if auto is not None and mode == auto.mode == "auto" else 0
        return f"auto -{level}" if level else mode

    def _time_frame(self):
        """Give the quality the last frame's time, once a frame (run() times them)."""
        screen = self.screen
        if self.auto_quality is None or screen is None or screen.frame_count == self._timed:
            return
        self._timed = screen.frame_count
        if screen.frame_time is not None:
            self.auto_quality.update(screen.frame_time, 1.0 / (screen.fps or 30))

    def _set_glyphs(self, v):
        if self.screen is not None:
            self.screen.set_glyphs(v)

    def _set_fps(self, v):
        if self.screen is not None:
            self.screen.fps = v

    def handle(self, events, screen=None):
        if screen is not None or self.screen is not None:
            self._found_screen(screen if screen is not None else self.screen)
            self._time_frame()
        return super().handle(events)

    def draw(self, screen, y, x):
        self._found_screen(screen)
        self._time_frame()
        super().draw(screen, y, x)
