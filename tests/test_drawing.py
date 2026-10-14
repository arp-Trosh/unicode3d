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
