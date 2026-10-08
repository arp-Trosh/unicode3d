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
    """Frames for `seconds`, each drawn anew (Renderer.draws counting up), taking frame_time(preset) seconds.
    Returns the presets passed through, in order, with repeats left out."""
    presets, t = [quality.preset], 0.0
    while t < seconds:
        took = frame_time(quality.preset)
        quality.renderer.draws += 1
        quality.update(took, target)
        t += max(took, target)
        if quality.preset != presets[-1]:
            presets.append(quality.preset)
    return presets


def rates(high, mid, low):
    """frame_time for play(): each preset's frames at so many a second."""
    return lambda preset: 1.0 / {"high": high, "mid": mid, "low": low}[preset]


class AutoQualityTests(unittest.TestCase):
    def test_presets(self):
        r = renderer()
        q = AutoQuality(r)
        self.assertEqual((q.mode, q.preset, q.steps), ("high", "high", []))
        q.mode = "mid"  # the picture's shading kept; edges and detail cheaper
        self.assertEqual(q.steps, list(STEPS[:2]))
        self.assertEqual((r.edge_samples, r.simplify, r.shading, r.max_pixels),
                         (0, 2.0, "cell", Renderer(1, 1).max_pixels))
        q.mode = "low"
        self.assertEqual(q.steps, [STEPS[0], STEPS[1], STEPS[2], STEPS[4]])  # (70% replaces 85%)
        self.assertEqual((r.edge_samples, r.simplify, r.shading, r.max_pixels),
                         (0, 2.0, "coarse", int(0.7 * 200 * 120)))
        self.assertTrue(r.shadows and r.reflections)
        play(q, 20, lambda preset: 0.5)  # only "auto" moves
        self.assertEqual(q.preset, "low")
        with self.assertRaises(ValueError):
            q.mode = "fast"  # (the lowest preset's name before 0.17: "low")
        q.mode = "high"
        self.assertEqual((r.edge_samples, r.simplify, r.shading, r.max_pixels),
                         (8, 1.0, "cell", Renderer(1, 1).max_pixels))
        with self.assertRaises(ValueError):
            q.mode = "medium"

    def test_auto_chooses_by_frame_rate(self):
        # High at 35 fps: kept. High at 24: mid (20-27). Mid at 18: low (below 20).
        q = AutoQuality(renderer(), "auto")
        self.assertEqual(play(q, 20, rates(35, 50, 70)), ["high"])
        self.assertEqual(play(q, 20, rates(24, 32, 45)), ["high", "mid"])
        self.assertEqual(play(q, 20, rates(15, 18, 30)), ["mid", "low"])
        self.assertEqual(q.steps, [STEPS[0], STEPS[1], STEPS[2], STEPS[4]])

    def test_moves_up_when_the_better_preset_is_expected_to_keep_up(self):
        q = AutoQuality(renderer(), "auto")
        play(q, 20, rates(24, 32, 45))  # down to mid: mid's frames take 24/32 of high's
        self.assertEqual(q.preset, "mid")
        # Mid at 33 fps: high would be about 25, under 27: kept at mid.
        self.assertEqual(play(q, 60, rates(25, 33, 45)), ["mid"])
        # The scene gets lighter: mid at 45, high expected at about 34 (over 27 by MARGIN): back up, and kept.
        self.assertEqual(play(q, 60, rates(34, 45, 60)), ["mid", "high"])

    def test_learns_how_much_dearer_each_preset_is(self):
        # Low runs at 40 fps, mid at only 19 (a scene whose cost is where low saves most). Moving from mid to low
        # shows mid's frames take twice low's, so mid isn't tried again on low's 40 fps, as it would be on the
        # guess (GUESSED_COST 1.8: mid expected at 22 fps, clearing 20 by MARGIN).
        q = AutoQuality(renderer(), "auto")
        presets = play(q, 300, rates(15, 19, 40))
        self.assertEqual(presets[:3], ["high", "mid", "low"])
        self.assertLessEqual(len(presets), 4)
        self.assertEqual(presets[-1], "low")

    def test_settles_rather_than_flickering(self):
        # A scene where mid looks affordable from low but isn't: each try is undone, and the tries get rarer.
        q = AutoQuality(renderer(), "auto")
        frames = {"high": 1 / 10, "mid": 1 / 19, "low": 1 / 60}
        presets = play(q, 300, lambda preset: frames[preset])
        tries = presets.count("mid") - 1
        self.assertLessEqual(tries, 6)
        self.assertEqual(presets[-1], "low")

    def test_one_slow_frame_is_not_enough(self):
        q = AutoQuality(renderer(), "auto")
        frames = iter([0.5] + [0.02] * 100000)
        self.assertEqual(play(q, 20, lambda preset: next(frames)), ["high"])

    def test_frames_reused_unchanged_do_not_count(self):
        r = renderer()
        q = AutoQuality(r, "auto")
        for _ in range(200):  # cheap frames the renderer didn't draw: they say nothing about drawing
            q.update(0.001, TARGET)
        play(q, 3, lambda preset: 0.1)
        self.assertNotEqual(q.preset, "high")
        preset = q.preset
        for _ in range(400):
            q.update(0.001, TARGET)
        self.assertEqual(q.preset, preset)

    def test_the_frame_rates_can_be_set(self):
        q = AutoQuality(renderer(), "auto", min_fps=40, high_fps=55)  # (a fast game)
        self.assertEqual(play(q, 20, rates(50, 60, 90)), ["high", "mid"])
        self.assertEqual(play(q, 20, rates(30, 38, 60)), ["mid", "low"])

    def test_the_users_settings_are_the_ceiling(self):
        r = renderer()
        q = AutoQuality(r, "auto")
        play(q, 10, lambda preset: 0.1)
        self.assertEqual(q.preset, "low")
        r.edge_samples = 4  # the program changes a setting itself: that is the user's choice now
        self.assertEqual(q.user("edge_samples"), 4)
        play(q, 60, lambda preset: 0.005)
        self.assertEqual((q.preset, r.edge_samples, r.simplify), ("high", 4, 1.0))
        # A step that would change nothing is skipped: edge samples already off, high detail kept at 2 px, shading
        # already coarse.
        r.edge_samples, r.simplify, r.shading = 0, 3.0, "coarse"
        q.mode = "low"
        self.assertEqual(q.steps, [STEPS[4]])
        self.assertEqual((r.simplify, r.shading), (3.0, "coarse"))

    def test_render_scale_follows_the_framebuffer(self):
        r = renderer()
        q = AutoQuality(r, "low")
        self.assertEqual(r.max_pixels, int(0.7 * 200 * 120))
        r.resize(50, 20, (2, 3))
        q.update(0.0, TARGET)
        self.assertEqual(r.max_pixels, int(0.7 * 100 * 60))
        # A smaller budget of the user's own already below a step leaves that step out.
        q.mode = "high"
        r.max_pixels = 1000
        q.mode = "low"
        self.assertEqual(q.steps, list(STEPS[:3]))
        self.assertEqual(r.max_pixels, 1000)


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
        self.assertEqual(controls.auto_quality.preset, "low")
        self.assertEqual(r.simplify, 2.0)
        # What is shown and saved is the user's choice, not the step it is on.
        self.assertEqual(controls.settings()["detail"], "standard")
        self.assertEqual(controls.quality.text(True).rstrip(), "F8 auto low")
        controls.apply({"detail": "high"})
        self.assertEqual(controls.auto_quality.user("simplify"), 0.0)
        self.assertEqual(r.simplify, 2.0)  # (still stepped down)
        controls.apply({"quality": "high"})
        self.assertEqual((r.simplify, r.edge_samples), (0.0, 8))
        controls.apply({"quality": "fast"})  # not a preset (settings saved before 0.17 are the program's to convert)
        self.assertEqual(controls.settings()["quality"], "high")


if __name__ == "__main__":
    unittest.main()
