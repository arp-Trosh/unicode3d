# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
"""Quality presets for a Renderer (high, mid, low), and "auto", choosing among them by the frame rate: for slow
machines (no kernels)."""
import statistics

MODES = ("high", "mid", "low", "auto")
STEPS = ("edge samples off", "detail 2 px", "shadows fitted to the scene", "coarse shading", "render scale 85%",
         "render scale 70%")
MID = STEPS[:3]  # the steps "mid" takes: the picture's shading kept, its edges, small things and shadows made cheaper
SIMPLIFY = 2.0  # Renderer.simplify at the "detail" step (pixels)
SCALES = (0.85, 0.70)  # parts of the framebuffer's pixels drawn at the "render scale" steps

PRESETS = ("high", "mid", "low")  # what "auto" chooses among, best first
HIGH_FPS = 27.0  # "auto" draws "high" while it runs at least this many frames a second (by default) ...
MIN_FPS = 20.0  # ... "mid" while it runs at least this many, and "low" below that
WINDOW = 1.0  # seconds of frames judged at a time (and at least MIN_FRAMES of them)
MIN_FRAMES = 5
MARGIN = 1.10  # a better preset is taken when its frames are expected to clear its frame rate by this much ...
FAST_FOR = 3.0  # ... for this many seconds
BACK_OFF = 2.0  # after a move up that had to be undone, the wait before the next is this many times longer
MAX_WAIT = 60.0
UNDONE = 5.0  # seconds after a move up in which a move down undoes it
# How many times longer a preset's frames take than the next one down's, until seen (high against mid, mid against
# low): on the slow side, so that a first move up is not taken too hopefully.
GUESSED_COST = {"high": 1.3, "mid": 1.8}


class AutoQuality:
    """Quality presets for a Renderer, and "auto", which chooses among them by the frame rate.

    The presets lower the renderer's settings by steps, in order (STEPS): edge samples off (edges are still
    smoothed by each pixel's samples, more coarsely); levels of detail at SIMPLIFY pixels; sun shadows fitted to
    the whole scene rather than the view (Renderer.shadow_fit "scene": coarser, cheaper to draw); coarse shading
    (Renderer.shading "coarse": everything once per cell, texture detail inside cells lost); the picture drawn at
    85%, then 70% of its pixels and stretched (softer). Shadows and reflections are never turned off. A step that
    would change nothing (edge samples already off, say) is skipped.

    mode: "high" leaves the renderer as it is; "mid" takes the first steps (MID: edge samples off, detail 2 px,
    shadows fitted to the scene), keeping the shading; "low" takes them all; "auto" draws "high" while frames come at high_fps (27) a second or
    more, "mid" while they come at min_fps (20) or more, and "low" below that. `preset` is the one in effect.

    The renderer's settings as they were, or as anything other than this sets them later (a settings key, the
    program), are the ceiling: steps only ever go below them. user(name) gives those values (edge_samples,
    simplify, shading, shadow_fit, max_pixels), for showing and saving as the user's choice.

    Call update(frame_time, target) once a frame with how long the last frame took to make (seconds, without any
    wait for the frame rate: so the rate it could run at, whatever the program's cap). It judges the median of about
    a second of frames, so one slow frame doesn't move it; frames the renderer reused unchanged (Renderer.draws not
    counting up) and frames of no measured time are left out. It moves down a preset when frames run below the
    current one's frame rate. As frames drawn in one preset say nothing directly of another's, it keeps how much
    longer each preset's frames take than the next one down's (measured over the windows either side of each move,
    GUESSED_COST until then), and moves up when the better preset's frames are expected to clear its frame rate by
    MARGIN for FAST_FOR seconds, waiting BACK_OFF times longer each time a move up had to be undone (up to
    MAX_WAIT), so that it settles rather than flickering between two. target (the time a frame may take) is not
    needed for that, and is taken for programs written for the previous version.
    """

    def __init__(self, renderer, mode="high", min_fps=MIN_FPS, high_fps=HIGH_FPS):
        self.renderer = renderer
        self.min_fps, self.high_fps = min_fps, high_fps
        self._user = self._settings()
        self._applied = dict(self._user)  # what this last set
        self.preset = "high"  # (in "auto", the one chosen; otherwise the mode)
        self._cost = dict(GUESSED_COST)
        self._moved = None  # (the preset left, its frames' median time) until the first window after a move
        self._times, self._span = [], 0.0
        self._fast_for = 0.0
        self._wait = FAST_FOR
        self._since_up = None  # seconds since the last move up (None: none to undo)
        self._draws = renderer.draws
        self.mode = mode

    @classmethod
    def of(cls, renderer):
        """The renderer's AutoQuality, made on first asking (mode "high"): one per renderer, so that everything
        showing or changing its quality (several DisplayControls, say) agrees."""
        quality = getattr(renderer, "_auto_quality", None)
        if quality is None:
            quality = renderer._auto_quality = cls(renderer)
        return quality

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        if value not in MODES:
            raise ValueError(f"quality mode must be one of {MODES}, not {value!r}")
        self._mode = value
        self._take_user_changes()
        self.preset = value if value in PRESETS else "high"  # ("auto" starts from the best)
        self._times, self._span, self._fast_for, self._since_up, self._moved = [], 0.0, 0.0, None, None
        self._reapply()

    @property
    def level(self):
        """How many steps are taken now."""
        return len(self._taken())

    def user(self, name):
        """The user's own value of a setting the steps change (edge_samples, simplify, shading, shadow_fit or
        max_pixels)."""
        self._take_user_changes()
        return self._user[name]

    def set_user(self, name, value):
        """Set the user's value of a setting (a settings key, say), keeping the steps below it."""
        self._take_user_changes()
        self._user[name] = value
        self._reapply()

    @property
    def steps(self):
        """The names of the steps in effect now (of STEPS), in order: a later step of the same setting replaces an
        earlier one (render scale 70% replaces 85%)."""
        taken = self._taken()
        return [name for i, (name, setting, _) in enumerate(taken) if all(s != setting for _, s, _ in taken[i + 1:])]

    def update(self, frame_time, target=0.0):
        """Take the time the last frame took to make (seconds); in "auto", move to another preset if it is time to."""
        self._take_user_changes()
        drew = self.renderer.draws != self._draws  # (not a frame reused unchanged)
        self._draws = self.renderer.draws
        self._reapply()  # (the framebuffer's size may have changed: render scale is a part of it)
        if self._mode != "auto" or not (frame_time > 0.0 and drew):
            return
        lasted = max(frame_time, target) if target > 0.0 else frame_time  # (run() waits out the rest of a frame)
        self._times.append(frame_time)
        self._span += lasted
        if self._since_up is not None:
            self._since_up += lasted
        if self._span < WINDOW or len(self._times) < MIN_FRAMES:
            return
        typical = statistics.median(self._times)
        self._times, self._span = [], 0.0
        if self._moved is not None:  # the first window since a move: how much the move changed the frames
            left, before = self._moved
            self._moved = None
            upper, ratio = (left, before / typical) if PRESETS.index(left) < PRESETS.index(self.preset) else \
                (self.preset, typical / before)
            self._cost[upper] = min(max(ratio, 1.0), 10.0)  # (a preset is never cheaper than the next one down)
        i = PRESETS.index(self.preset)
        rate = 1.0 / typical
        if i < len(PRESETS) - 1 and rate < self._needs(self.preset):
            if self._since_up is not None and self._since_up < UNDONE:
                self._wait = min(self._wait * BACK_OFF, MAX_WAIT)  # the move up didn't hold
            self._since_up, self._fast_for = None, 0.0
            self._move(PRESETS[i + 1], typical)
        elif i > 0 and 1.0 / (typical * self._cost[PRESETS[i - 1]]) >= MARGIN * self._needs(PRESETS[i - 1]):
            self._fast_for += WINDOW
            if self._fast_for >= self._wait:
                self._fast_for, self._since_up = 0.0, 0.0
                self._move(PRESETS[i - 1], typical)
        else:
            self._fast_for = 0.0

    # ----- internals

    SETTINGS = ("edge_samples", "simplify", "shading", "shadow_fit", "max_pixels")

    def _settings(self):
        return {name: getattr(self.renderer, name) for name in self.SETTINGS}

    def _take_user_changes(self):
        """A setting that isn't what this last set was changed by someone else: it is the user's choice now."""
        for name in self.SETTINGS:
            value = getattr(self.renderer, name)
            if value != self._applied[name]:
                self._user[name] = self._applied[name] = value

    def _pixels(self):
        fb = self.renderer.framebuffer
        return fb.width * fb.height

    def _steps(self):
        """The steps that would change something, in order, as (name, setting, value)."""
        user, steps = self._user, []
        if user["edge_samples"]:
            steps.append((STEPS[0], "edge_samples", 0))
        if not user["simplify"] >= SIMPLIFY:  # (0 or less: none)
            steps.append((STEPS[1], "simplify", SIMPLIFY))
        if user["shadow_fit"] != "scene":
            steps.append((STEPS[2], "shadow_fit", "scene"))
        if user["shading"] != "coarse":
            steps.append((STEPS[3], "shading", "coarse"))
        pixels = self._pixels()
        budget = user["max_pixels"] if user["max_pixels"] else pixels
        for name, part in zip(STEPS[4:], SCALES):
            if pixels and part * pixels < min(budget, pixels):
                steps.append((name, "max_pixels", max(int(part * pixels), 1)))
        return steps

    def _needs(self, preset):
        """The frame rate "auto" draws a preset at (at least)."""
        return self.high_fps if preset == "high" else self.min_fps if preset == "mid" else 0.0

    def _move(self, preset, typical):
        self._moved = (self.preset, typical)
        self.preset = preset
        self._reapply()

    def _taken(self):
        """The steps the preset takes, as _steps() gives them."""
        steps = self._steps()
        if self.preset == "low":
            return steps
        if self.preset == "mid":
            return [step for step in steps if step[0] in MID]
        return []

    def _reapply(self):
        """The settings for the preset, worked out again (the steps depend on the user's settings and the
        framebuffer's size)."""
        values = dict(self._user)
        for _, name, value in self._taken():
            values[name] = value
        for name, value in values.items():
            if getattr(self.renderer, name) != value:
                setattr(self.renderer, name, value)
        self._applied = values
