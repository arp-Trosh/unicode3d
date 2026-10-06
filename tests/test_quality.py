# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
"""Automatic quality (quality.AutoQuality and DisplayControls' quality setting), with made-up frame times."""
import unittest

from unicode3d.quality import AutoQuality, STEPS
from unicode3d.scene import Renderer
from unicode3d.ui import DisplayControls

TARGET = 1 / 30


def renderer():
    r = Renderer(1, 1, simplify=1.0)
    r.resize(100, 40, (2, 3))  # 200 x 120 pixels
    return r


def play(quality, seconds, frame_time, target=TARGET):
    """Frames for `seconds`, each drawn anew (Renderer.draws counting up), taking frame_time(level) seconds.
    Returns the levels passed through, in order, with repeats left out."""
    levels, t = [quality.level], 0.0
    while t < seconds:
        took = frame_time(quality.level)
        quality.renderer.draws += 1
        quality.update(took, target)
        t += max(took, target)
        if quality.level != levels[-1]:
            levels.append(quality.level)
    return levels


class AutoQualityTests(unittest.TestCase):
    def test_steps_down_in_order_while_slow_and_never_touches_shadows(self):
        r = renderer()
        q = AutoQuality(r, "auto", min_fps=0)
        self.assertEqual(play(q, 30, lambda level: 0.1), [0, 1, 2, 3, 4])
        self.assertEqual(q.steps, [STEPS[0], STEPS[1], STEPS[3]])  # (70% replaces 85%)
        self.assertEqual((r.edge_samples, r.simplify, r.max_pixels), (0, 2.0, int(0.7 * 200 * 120)))
        self.assertTrue(r.shadows and r.reflections)
        # Plenty of time again: back up, step by step, to the settings it started from.
        self.assertEqual(play(q, 60, lambda level: 0.005), [4, 3, 2, 1, 0])
        self.assertEqual((r.edge_samples, r.simplify, r.max_pixels), (8, 1.0, Renderer(1, 1).max_pixels))

    def test_stops_where_frames_keep_up(self):
        q = AutoQuality(renderer(), "auto", min_fps=0)
        # Edge samples off is enough here: frames then take 0.9 of the target, neither slow nor with time to spare.
        self.assertEqual(play(q, 60, lambda level: TARGET * (1.3 if level == 0 else 0.9)), [0, 1])

    def test_one_slow_frame_is_not_enough(self):
        q = AutoQuality(renderer(), "auto", min_fps=0)
        frames = iter([0.5] + [0.02] * 100000)
        self.assertEqual(play(q, 20, lambda level: next(frames)), [0])

    def test_frames_reused_unchanged_do_not_count(self):
        r = renderer()
        q = AutoQuality(r, "auto", min_fps=0)
        for _ in range(200):  # cheap frames the renderer didn't draw: they say nothing about drawing
            q.update(0.001, TARGET)
        play(q, 3, lambda level: 0.1)
        self.assertGreater(q.level, 0)
        level = q.level
        for _ in range(400):
            q.update(0.001, TARGET)
        self.assertEqual(q.level, level)

    def test_settles_rather_than_flickering_between_two_steps(self):
        # Fast enough to step up from level 1, too slow once there: each try is undone, and the tries get rarer.
        q = AutoQuality(renderer(), "auto", min_fps=0)
        levels = play(q, 300, lambda level: TARGET * (1.2 if level == 0 else 0.5))
        tries = levels.count(0) - 1  # steps back up to 0
        self.assertGreaterEqual(tries, 2)
        self.assertLessEqual(tries, 8)  # waits of 3, 6, 12, 24, 48 s, then once a minute (every 3 s: about 75)
        self.assertEqual(levels[-1], 1)

    def test_the_users_settings_are_the_ceiling(self):
        r = renderer()
        q = AutoQuality(r, "auto", min_fps=0)
        play(q, 10, lambda level: 0.1)
        self.assertEqual(q.level, 4)
        r.edge_samples = 4  # the program changes a setting itself: that is the user's choice now
        self.assertEqual(q.user("edge_samples"), 4)
        play(q, 60, lambda level: 0.005)
        self.assertEqual((q.level, r.edge_samples, r.simplify), (0, 4, 1.0))
        # A step that would change nothing is skipped: edge samples already off, high detail kept at 2 px.
        r.edge_samples, r.simplify = 0, 3.0
        self.assertEqual(play(q, 30, lambda level: 0.1)[-1], 2)
        self.assertEqual((q.level, q.steps), (2, [STEPS[3]]))
        self.assertEqual(r.simplify, 3.0)

    def test_steps_down_only_below_min_fps(self):
        # Aiming at 30 fps, frames at 22-29 fps are left as they are (the game plays well there); below 20 it steps.
        q = AutoQuality(renderer(), "auto")
        self.assertEqual(q.min_fps, 20)
        self.assertEqual(play(q, 30, lambda level: 1 / 22), [0])
        self.assertEqual(play(q, 30, lambda level: 1 / 18 if level == 0 else 1 / 24), [0, 1])
        # Back up only with room to spare below a 20 fps frame (FAST of it: 37.5 ms, under 27 fps).
        self.assertEqual(play(q, 60, lambda level: 1 / 25), [1])
        self.assertEqual(play(q, 60, lambda level: 1 / 30), [1, 0])
        # A target slower than min_fps still counts: aiming at 15 fps, 18 fps is fine.
        q = AutoQuality(renderer(), "auto")
        self.assertEqual(play(q, 30, lambda level: 1 / 18, target=1 / 15), [0])

    def test_render_scale_follows_the_framebuffer(self):
        r = renderer()
        q = AutoQuality(r, "fast")
        self.assertEqual(r.max_pixels, int(0.7 * 200 * 120))
        r.resize(50, 20, (2, 3))
        q.update(0.0, TARGET)
        self.assertEqual(r.max_pixels, int(0.7 * 100 * 60))
        # A smaller budget of the user's own already below a step leaves that step out.
        q.mode = "high"
        r.max_pixels = 1000
        q.mode = "fast"
        self.assertEqual(q.steps, list(STEPS[:2]))
        self.assertEqual(r.max_pixels, 1000)

    def test_modes(self):
        r = renderer()
        q = AutoQuality(r)
        self.assertEqual(q.mode, "high")
        play(q, 20, lambda level: 0.5)  # "high" never steps
        self.assertEqual((q.level, r.edge_samples), (0, 8))
        q.mode = "fast"
        self.assertEqual(q.level, 4)
        q.mode = "high"
        self.assertEqual((r.edge_samples, r.simplify, r.max_pixels), (8, 1.0, Renderer(1, 1).max_pixels))
        with self.assertRaises(ValueError):
            q.mode = "medium"


class FakeScreen:
    """What DisplayControls reads of a Screen when it times frames."""
    def __init__(self):
        self.fps, self.measured_fps, self.frame_time, self.frame_count = 30, None, None, 0
        self.mode, self.color_mode, self.glyph_modes = "sextant", "truecolor", ("sextant",)


class DisplayControlsQualityTests(unittest.TestCase):
    def test_quality_setting_and_the_users_detail(self):
        r = renderer()
        controls = DisplayControls(renderer=r)
        self.assertEqual(controls.settings()["quality"], "high")
        controls.apply({"quality": "auto", "detail": "standard"})
        screen = FakeScreen()
        for frame in range(400):  # slow frames, timed by run()
            screen.frame_time, screen.frame_count = 0.1, frame + 1
            r.draws += 1
            controls.handle([], screen)
            controls.handle([], screen)  # (twice in one frame counts once)
        self.assertEqual(controls.auto_quality.level, 4)
        self.assertEqual(r.simplify, 2.0)
        # What is shown and saved is the user's choice, not the step it is on.
        self.assertEqual(controls.settings()["detail"], "standard")
        self.assertEqual(controls.quality.text(True), "F8 auto -4")
        controls.apply({"detail": "high"})
        self.assertEqual(controls.auto_quality.user("simplify"), 0.0)
        self.assertEqual(r.simplify, 2.0)  # (still stepped down)
        controls.apply({"quality": "high"})
        self.assertEqual((r.simplify, r.edge_samples), (0.0, 8))


if __name__ == "__main__":
    unittest.main()
