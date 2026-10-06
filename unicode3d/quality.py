# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
"""Automatic quality for slow machines: a Renderer's settings stepped down while frames take longer than the
target, and back up when there is time to spare (no kernels)."""
import statistics

MODES = ("high", "auto", "fast")
STEPS = ("edge samples off", "detail 2 px", "render scale 85%", "render scale 70%")
SIMPLIFY = 2.0  # Renderer.simplify at the "detail" step (pixels)
SCALES = (0.85, 0.70)  # parts of the framebuffer's pixels drawn at the "render scale" steps

WINDOW = 1.0  # seconds of frames judged at a time (and at least MIN_FRAMES of them)
MIN_FRAMES = 5
SLOW = 1.10  # a step down when the median frame takes more than this times the target
FAST = 0.75  # a step up when it takes less than this times the target ...
FAST_FOR = 3.0  # ... for this many seconds
BACK_OFF = 2.0  # after a step up that had to be undone, the wait before the next is this many times longer
MAX_WAIT = 60.0
UNDONE = 5.0  # seconds after a step up in which a step down undoes it


class AutoQuality:
    """Steps a Renderer's quality down while frames run long, and back up when there is headroom.

    The steps, in order (STEPS): edge samples off (edges are still smoothed by each pixel's samples, more
    coarsely); levels of detail at SIMPLIFY pixels; the picture drawn at 85%, then 70% of its pixels and stretched
    (softer). Shadows and reflections are never touched. A step that would change nothing (edge samples already off,
    say) is skipped.

    mode: "high" leaves the renderer as it is; "fast" holds the lowest step; "auto" moves between them.

    The renderer's settings as they were, or as anything other than this sets them later (a settings key, the
    program), are the ceiling: steps only ever go below them. user(name) gives those values (edge_samples,
    simplify, max_pixels), for showing and saving as the user's choice.

    Call update(frame_time, target) once a frame with how long the last frame took to make (seconds, without any
    wait for the frame rate) and the time a frame may take. It judges the median of about a second of frames, so
    one slow frame doesn't step it down; frames the renderer reused unchanged (Renderer.draws not counting up) and
    frames of no measured time are left out. Steps down when frames take more than SLOW times the target; steps up
    when they take less than FAST times it for FAST_FOR seconds, waiting BACK_OFF times longer each time a step up
    had to be undone (up to MAX_WAIT), so that it settles rather than flickering between two steps.
    """

    def __init__(self, renderer, mode="high"):
        self.renderer = renderer
        self._user = self._settings()
        self._applied = dict(self._user)  # what this last set
        self.level = 0  # steps down from the user's settings
        self._times, self._span = [], 0.0
        self._fast_for = 0.0
        self._wait = FAST_FOR
        self._since_up = None  # seconds since the last step up (None: none to undo)
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
        self.level = 0
        self._reapply()

    def user(self, name):
        """The user's own value of a setting the steps change (edge_samples, simplify or max_pixels)."""
        self._take_user_changes()
        return self._user[name]

    def set_user(self, name, value):
        """Set the user's value of a setting (a settings key, say), keeping the steps below it."""
        self._take_user_changes()
        self._user[name] = value
        self._reapply()

    @property
    def steps(self):
        """The names of the steps taken now (of STEPS)."""
        return [name for name, _, _ in self._steps()[:self.level]]

    def update(self, frame_time, target):
        """Take the time the last frame took (seconds) and the time one may take; step if it is time to."""
        self._take_user_changes()
        drew = self.renderer.draws != self._draws  # (not a frame reused unchanged)
        self._draws = self.renderer.draws
        self._reapply()  # (the framebuffer's size may have changed: render scale is a part of it)
        if self._mode != "auto" or not (frame_time > 0.0 and target > 0.0 and drew):
            return
        lasted = max(frame_time, target)  # (a quick frame still lasts the frame period: run() waits out the rest)
        self._times.append(frame_time)
        self._span += lasted
        if self._since_up is not None:
            self._since_up += lasted
        if self._span < WINDOW or len(self._times) < MIN_FRAMES:
            return
        typical = statistics.median(self._times)
        self._times, self._span = [], 0.0
        if typical > SLOW * target and self.level < len(self._steps()):
            if self._since_up is not None and self._since_up < UNDONE:
                self._wait = min(self._wait * BACK_OFF, MAX_WAIT)  # the step up didn't hold
            self._since_up, self._fast_for = None, 0.0
            self._set_level(self.level + 1)
        elif typical < FAST * target and self.level > 0:
            self._fast_for += WINDOW
            if self._fast_for >= self._wait:
                self._fast_for, self._since_up = 0.0, 0.0
                self._set_level(self.level - 1)
        else:
            self._fast_for = 0.0

    # ----- internals

    SETTINGS = ("edge_samples", "simplify", "max_pixels")

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
        pixels = self._pixels()
        budget = user["max_pixels"] if user["max_pixels"] else pixels
        for name, part in zip(STEPS[2:], SCALES):
            if pixels and part * pixels < min(budget, pixels):
                steps.append((name, "max_pixels", max(int(part * pixels), 1)))
        return steps

    def _reapply(self):
        """The settings for the mode and level, worked out again (the steps depend on the user's settings and
        the framebuffer's size)."""
        self._set_level(len(self._steps()) if self._mode == "fast" else self.level)

    def _set_level(self, level):
        steps = self._steps()
        self.level = max(0, min(level, len(steps)))
        values = dict(self._user)
        for _, name, value in steps[:self.level]:
            values[name] = value
        for name, value in values.items():
            if getattr(self.renderer, name) != value:
                setattr(self.renderer, name, value)
        self._applied = values
