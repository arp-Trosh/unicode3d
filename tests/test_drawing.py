# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
"""A frame without boilerplate: Screen.draw sizing the renderer and drawing in one call, run() and the playtest
erasing before each frame and sending it after, and pick(), ray() and label() knowing where the frame was drawn."""
import tempfile
import unittest

import numpy as np

from unicode3d import Color, MouseEvent, ScreenSize
from unicode3d.mesh import make_box
from unicode3d.playtest import Playtest
from unicode3d.scene import Camera, Light, Object3D, Renderer
from unicode3d.terminal import Screen

CAMERA = Camera(position=np.array([0.0, 0.0, 5.0]))


def click(x, y):
    return MouseEvent(x, y, MouseEvent.LEFT, True, False)


class DrawTests(unittest.TestCase):
    def test_size_has_names(self):
        size = Screen(size=(24, 80)).size()
        self.assertEqual(size, (24, 80))
        self.assertEqual((size.rows, size.cols), (24, 80))
        self.assertIsInstance(size, ScreenSize)
        rows, cols = size
        self.assertEqual((rows, cols), (24, 80))

    def test_draw_sizes_the_renderer_to_its_part_of_the_screen(self):
        screen = Screen(size=(24, 80), glyphs="quad", color="truecolor")
        renderer = Renderer()
        box = Object3D(make_box())
        for where, size, corner in [({}, (80, 24), (0, 0)), ({"rows": -1}, (80, 23), (0, 0)),
                                    ({"top": 2, "left": 10}, (70, 22), (2, 10)),
                                    ({"top": 1, "rows": 10, "cols": -5}, (75, 10), (1, 0)),
                                    ({"rows": -100}, (80, 1), (0, 0)), ({"top": 30}, (80, 1), (30, 0)),
                                    ({"rows": float("nan"), "left": float("inf")}, (80, 24), (0, 0))]:
            fb = screen.draw(renderer, [box], CAMERA, Light(), **where)
            self.assertEqual((renderer.width, renderer.height), size, where)
            self.assertEqual(renderer.cell_pixels, screen.cell_pixels)
            self.assertIs(fb, renderer.framebuffer)
            self.assertEqual((fb.top, fb.left), corner)
        screen.set_glyphs("sextant")  # (F2): the next draw follows the glyph set
        screen.draw(renderer, [box], CAMERA, Light())
        self.assertEqual(renderer.cell_pixels, screen.cell_pixels)

    def test_draw_puts_the_frame_where_asked(self):
        screen = Screen(size=(20, 60), glyphs="quad", color="truecolor")
        renderer = Renderer(background=(200, 0, 0))
        screen.draw(renderer, [], CAMERA, Light(), top=3, left=5, rows=-2, cols=20)
        drawn = screen.chars != " "
        rows, cols = np.nonzero(drawn)
        self.assertEqual((rows.min(), rows.max(), cols.min(), cols.max()), (3, 17, 5, 24))

    def test_pick_ray_and_label_take_screen_cells_from_mouse_events(self):
        screen = Screen(size=(30, 80), glyphs="quad", color="truecolor")
        renderer = Renderer()
        box = Object3D(make_box(), position=np.array([1.0, 0.0, 0.0]))
        top, left = 2, 10
        screen.draw(renderer, [box], CAMERA, Light(), top=top, left=left, rows=-1)
        x, y = renderer.project(box.position)
        ev = click(int(x) + left, int(y) + top)
        self.assertIs(renderer.pick(ev).object, box)
        np.testing.assert_allclose(renderer.pick(ev).position, renderer.pick(int(x), int(y)).position)
        np.testing.assert_allclose(renderer.ray(ev)[1], renderer.ray(int(x) + 0.5, int(y) + 0.5)[1])
        self.assertIsNone(renderer.pick(click(int(x), int(y))))  # (the same cell of the screen, uncorrected, misses)
        self.assertIsNone(renderer.pick(click(left - 1, top)))  # left of the frame
        self.assertIsNone(renderer.pick(click(left, top - 1)))  # above it
        anchor = screen.label(renderer, box.position, "box", owner=box)
        self.assertEqual("".join(screen.chars[top + anchor.y, left + anchor.x - 1:left + anchor.x + 2]), "box")
        with self.assertRaises(TypeError):
            renderer.pick((3, 4))
        self.assertIsNone(renderer.pick(float("nan"), 3))


class Frames:
    """Writes a line on row 0 in the first frame, then draws a box below that row and a status line; never erases
    or refreshes."""

    def __init__(self):
        self.renderer = Renderer()
        self.count = 0

    def frame(self, screen, dt, events):
        self.count += 1
        if ord("q") in events:
            return False
        if self.count == 1:
            screen.text(0, 0, "only in the first frame")
            return
        rows, cols = screen.size()
        screen.draw(self.renderer, [Object3D(make_box())], CAMERA, Light(), top=1, rows=-1)
        screen.text(rows - 1, 0, f"frame {self.count}", Color.YELLOW)


class RunTests(unittest.TestCase):
    def play(self, **kw):
        out = tempfile.TemporaryDirectory()
        self.addCleanup(out.cleanup)
        return Playtest(Frames(), size=(20, 60), out=out.name, compile=False, **kw)

    def test_frames_are_erased_and_sent_for_the_program(self):
        test = self.play()
        test.wait(1)
        test.expect("only in the first frame")
        test.wait(1)
        test.expect("frame 2").expect_not("only in the first frame")
        report = test.report()
        self.assertEqual((report.warnings, report.blank_frames), ([], []))
        self.assertGreater(report.max_bytes, 100)  # sent without a refresh() of its own
        test.press("q")
        test.expect("frame 2")  # quitting leaves the last frame on the screen, as in a terminal
        self.assertEqual(test.report().blank_frames, [])

    def test_erase_false_keeps_the_screen(self):
        test = self.play(erase=False)
        test.wait(2)
        test.expect("frame 2").expect("only in the first frame")


if __name__ == "__main__":
    unittest.main()


class ColourNameTests(unittest.TestCase):
    def test_more_named_colours(self):
        from unicode3d.color import ANSI, HUE_RGB, encode_index
        self.assertIs(Color.GRAY, Color.GREY)
        for mode in ("truecolor", "256", "16"):
            screen = Screen(size=(2, 10), color=mode)
            for color in Color:
                screen.text(0, 0, "x", color)
                if color == Color.DEFAULT:
                    continue
                if color in ANSI:  # the terminal's own palette, so text follows its theme
                    self.assertEqual(screen.fg[0, 0], encode_index(ANSI[color]), (mode, color))
                else:              # hues it lacks: their RGB, as an (r, g, b) text colour is shown
                    screen.text(1, 0, "x", HUE_RGB[color])
                    self.assertEqual(screen.fg[0, 0], screen.fg[1, 0], (mode, color))
        screen = Screen(size=(2, 10), color="truecolor")
        screen.text(0, 0, "x", Color.ORANGE)
        self.assertEqual(screen.fg[0, 0] & 0xFFFFFF, 0xF5821E)
        for color in (Color.ORANGE, Color.BROWN):  # on objects, as any colour: red over green over blue
            renderer = Renderer()
            renderer.resize(20, 10)
            fb = renderer.render([Object3D(make_box(2.0), color=color)], CAMERA, Light(direction=(0.0, 0.0, -1.0)))
            r, g, b = fb.rgb[10, 10]
            self.assertTrue(r > g > b, (color, r, g, b))


class SceneLightTests(unittest.TestCase):
    def scene(self, color=Color.WHITE, target=(0.0, 0.0, 0.0)):
        box = Object3D(make_box(), color=color)
        camera = Camera(position=np.array([0.0, 1.0, 4.0]), target=np.array(target))
        renderer = Renderer()

        def frame(screen, dt, events):
            screen.draw(renderer, [box], camera, Light(), rows=-1)
            screen.text(screen.size().rows - 1, 0, "status line", Color.WHITE)
        out = tempfile.TemporaryDirectory()
        self.addCleanup(out.cleanup)
        test = Playtest(frame, size=(20, 60), out=out.name, compile=False)
        test.wait(2)
        test.shot("scene")
        return test.report()

    def test_a_lit_scene_reports_its_brightness(self):
        report = self.scene()
        self.assertEqual((report.dark_frames, report.empty_frames), ([], []))
        (path, light), = report.brightness.items()
        self.assertTrue(path.endswith("scene.png"))
        self.assertGreater(light.covered, 0.02)
        self.assertLess(light.covered, 0.5)     # a box on nothing: most of the frame is empty, not dark
        self.assertGreater(light.brightness, 0.3)
        self.assertLess(light.near_black, 0.1)
        self.assertIn("(3D scene: ", str(report))

    def test_a_black_scene_is_dark(self):
        report = self.scene(color=(0, 0, 0))
        self.assertEqual(report.dark_frames, [1, 2])
        self.assertIn("DARK SCENE in 2 frames", str(report))

    def test_a_camera_looking_away_sees_an_empty_scene(self):
        report = self.scene(target=(0.0, 20.0, 0.0))
        self.assertEqual(report.empty_frames, [1, 2])
        self.assertIn("EMPTY SCENE in 2 frames", str(report))
        self.assertIn("(3D scene: nothing drawn)", str(report))

    def test_no_scene_no_measure(self):
        out = tempfile.TemporaryDirectory()
        self.addCleanup(out.cleanup)
        test = Playtest(lambda screen, dt, events: screen.text(0, 0, "title"), size=(10, 30), out=out.name,
                        compile=False)
        test.wait(1)
        test.shot("title")
        report = test.report()
        self.assertEqual((report.dark_frames, report.empty_frames, report.brightness), ([], [], {}))
