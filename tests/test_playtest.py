# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
"""The playtest tool: driving a program off-screen with scripted input, and what it saves and reports."""
import contextlib
import io
import os
import tempfile
import textwrap
import unittest
import warnings

import numpy as np

from unicode3d.animation import Animation, Clip, Track
from unicode3d.keys import Key, KeyRelease, MouseEvent
from unicode3d.mesh import make_box
from unicode3d.playtest import (Playtest, PlaytestError, contact_sheet, load_target, main, parse_key, parse_script,
                                preview, run_script)
from unicode3d.scene import Camera, Light, Model, Node, Object3D, Renderer


class Game:
    """A small program: a box in a frame, a score line, keys and clicks noted. q quits, x raises, w warns."""

    def __init__(self, draw=True):
        self.renderer = Renderer(1, 1)
        self.camera = Camera(position=np.array([0.0, 1.0, 4.0]))
        self.box = Object3D(make_box())
        self.events, self.held, self.dts = [], [], []
        self.draw = draw

    def frame(self, screen, dt, events):
        self.events.append(events)
        self.held.append(set(screen.held.keys()))
        self.dts.append(dt)
        for e in events:
            if e == ord("q"):
                return False
            if e == ord("x"):
                raise RuntimeError("boom")
            if e == ord("w"):
                warnings.warn("a hazard", UserWarning)
        if self.draw:
            rows, cols = screen.size()
            self.renderer.resize(cols, rows - 1, screen.cell_pixels)
            screen.erase()
            screen.draw_frame(self.renderer.render([self.box], self.camera, Light()))
            screen.text(rows - 1, 2, f"Score {len(self.events)}  [Start]")
            screen.refresh()  # (as games did before run() refreshed: still harmless)
        return True


class PlaytestTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.out = self.dir.name

    def play(self, game=None, **kw):
        self.game = game or Game()
        return Playtest(self.game.frame, size=(20, 60), out=self.out, compile=False, **kw)

    def test_keys_by_name(self):
        self.assertEqual(parse_key("enter"), Key.ENTER)
        self.assertEqual(parse_key("Esc"), Key.ESC)
        self.assertEqual(parse_key("space"), ord(" "))
        self.assertEqual(parse_key("A"), ord("A"))
        self.assertEqual(parse_key("f2"), Key.F2)
        self.assertEqual(parse_key("ctrl+c"), 3)
        self.assertEqual(parse_key(Key.UP), Key.UP)
        with self.assertRaisesRegex(ValueError, "enter"):  # the names are listed
            parse_key("jump")

    def test_presses_reach_the_program_with_a_fixed_dt(self):
        test = self.play(dt=0.05)
        test.wait(2).press("enter").press("a", "b").type("hi")
        self.assertEqual(self.game.events, [[], [], [Key.ENTER], [ord("a"), ord("b")], [ord("h")], [ord("i")]])
        self.assertEqual(self.game.dts, [0.05] * 6)
        self.assertEqual(test.frames, 6)
        self.assertAlmostEqual(test.now, 0.3)
        self.assertIn("Score 6", test.text())

    def test_held_keys_are_exact_and_repeat_like_a_terminal(self):
        test = self.play(dt=0.1)
        test.hold("w", "right", frames=8)
        test.wait(1)
        w, right = ord("w"), Key.RIGHT
        self.assertEqual(self.game.events[0], [w, right])          # pressed with the first frame
        self.assertEqual(self.game.events[1:5], [[], [], [], []])  # then quiet until the repeat delay
        self.assertEqual(self.game.events[5], [w, right])          # the first repeat at 0.5 s,
        self.assertEqual(self.game.events[6:8], [[w] * 3 + [right] * 3] * 2)  # then 30 a second: 3 in 0.1 s
        self.assertTrue(all(h == {w, right} for h in self.game.held[:8]))
        self.assertEqual(test.frames, 10)                          # hold() played the release frame itself
        self.assertEqual(self.game.events[8], [])                  # (the release isn't passed without key_release)
        self.assertEqual(self.game.held[8], set())

    def test_repeats_come_at_the_same_rate_at_any_frame_rate(self):
        counts = []
        for dt in (1 / 60, 1 / 30, 0.25):
            test = self.play(dt=dt)
            test.hold("d", frames=round(1.5 / dt) + 1)  # the last frame held at 1.5 s
            counts.append(sum(e.count(ord("d")) for e in self.game.events))
        self.assertEqual(counts, [32, 32, 32])  # the press, then from 0.5 s a repeat every 1/30 s to 1.5 s

    def test_a_pressed_key_is_let_go(self):
        test = self.play()
        test.press("a").wait(60)
        self.assertEqual(self.game.held[0], {ord("a")})
        self.assertEqual(self.game.held[1:], [set()] * 60)
        test.type("bc").wait(1)
        self.assertEqual(self.game.held[-3:], [{ord("b")}, {ord("c")}, set()])
        test.down("z").press("z").wait(2)  # a key held with down() stays held through a press of it
        self.assertEqual(self.game.held[-3:], [{ord("z")}] * 3)

    def test_releases_are_passed_with_key_release(self):
        test = self.play(key_release=True)
        test.down("a").wait(2).up().wait(1)
        test.press("b").wait(1)
        self.assertEqual(self.game.events, [[ord("a")], [], [KeyRelease(ord("a"))], [ord("b")],
                                            [KeyRelease(ord("b"))]])
        self.assertEqual(self.game.held[1], {ord("a")})
        self.assertEqual(self.game.held[2], set())

    def test_a_terminal_without_releases(self):
        test = self.play(releases=False, key_release=True)
        test.hold("w", frames=45).wait(30)
        self.assertFalse(any(isinstance(e, KeyRelease) for ev in self.game.events for e in ev))
        self.assertFalse(test.screen.held.exact)
        self.assertTrue(all(h == {ord("w")} for h in self.game.held[:45]))  # guessed held through the pause
        self.assertEqual(self.game.held[-1], set())                        # and let go once repeats stop
        test.press("a").wait(30)
        self.assertEqual(self.game.held[-30], {ord("a")})  # a tap counts as held for a moment (HeldKeys.first_hold)
        self.assertEqual(self.game.held[-1], set())

    def test_clicks_and_text(self):
        test = self.play()
        test.wait(1)
        self.assertEqual(test.find("[Start]"), (19, 2 + len("Score 1  ")))
        self.assertEqual(test.find("[Start]").col, 2 + len("Score 1  "))
        test.click_text("[Start]")
        y, x = test.find("[Start]")
        self.assertEqual(self.game.events[-1], [MouseEvent(x + 3, y, 0, True), MouseEvent(x + 3, y, 0, False)])
        test.click(5, 6, "right").move(1, 2).click(0, 0, "wheel_up")
        self.assertEqual(self.game.events[-3][0].button, MouseEvent.RIGHT)
        self.assertTrue(self.game.events[-2][0].moved)
        self.assertEqual(self.game.events[-1], [MouseEvent(0, 0, MouseEvent.WHEEL_UP, True)])
        with self.assertRaisesRegex(PlaytestError, "(?s)'Missing' is not on the screen.*Score 5"):
            test.click_text("Missing")
        test.expect("Score").expect_not("Game over")
        with self.assertRaises(PlaytestError):
            test.expect_not("Score")

    def test_text_without_the_picture(self):
        test = self.play()
        test.wait(1)
        self.assertEqual(test.text(picture=False).strip(), "Score 1  [Start]")
        self.assertGreater(len(test.text().strip()), 100)  # the box, in block characters

    def test_shots_and_sheets_are_saved(self):
        from PIL import Image
        test = self.play()
        test.wait(1)
        path = test.shot("title")
        with Image.open(path) as image:
            self.assertEqual(image.size, (60 * 8, 20 * 16))
        with open(os.path.join(self.out, "title.txt"), encoding="utf-8") as f:
            self.assertIn("Score 1", f.read())
        sheet = test.sheet("walk", 9, shots=3, columns=3)
        self.assertEqual(test.frames, 10)
        with Image.open(sheet) as image:
            w, h = image.size
        self.assertGreater(w, 3 * 60 * 8)  # three tiles across, each with a label above
        self.assertGreater(h, 20 * 16)
        self.assertEqual(contact_sheet([np.zeros((10, 20, 3), np.uint8)] * 5, columns=2).shape, (3 * 10 + 16, 2 * 20 + 12, 3))

    def test_an_exception_names_its_frame_and_keeps_the_screen(self):
        test = self.play()
        test.wait(3)
        with self.assertRaisesRegex(PlaytestError, "RuntimeError in frame 4") as caught:
            test.press("x")
        self.assertIsInstance(caught.exception.__cause__, RuntimeError)
        self.assertTrue(os.path.exists(os.path.join(self.out, "crash-frame-4.png")))

    def test_quitting(self):
        test = self.play()
        test.wait(2).press("q")
        self.assertTrue(test.quit)
        self.assertEqual(test.quit_at, 3)
        for play in (lambda: test.wait(1), lambda: test.press("a"), lambda: test.hold("a", frames=2),
                     lambda: test.click(1, 1), lambda: test.type("x"), lambda: test.sheet("s", 3)):
            with self.assertRaisesRegex(PlaytestError, "already quit"):
                play()
        self.assertEqual(test.frames, 3)
        test.expect("Score 2").expect_not("Score 3")  # the last screen can still be looked at
        test.shot("end")
        self.assertIn("quit: frame() returned False at frame 3", str(test.report()))

    def test_a_game_object(self):
        game = Game()
        test = Playtest(game, size=(20, 60), out=self.out, compile=False)
        test.wait(2)
        self.assertEqual(len(game.events), 2)

    def test_warnings_follow_the_programs_filters(self):
        test = self.play()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            test.press("w")
        self.assertEqual(test.report().warnings, [])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with self.assertRaisesRegex(PlaytestError, "UserWarning in frame 2") as caught:
                test.press("w")
        self.assertIsInstance(caught.exception.__cause__, UserWarning)

    def test_report(self):
        test = self.play()
        test.wait(3).press("w").press("w").wait(2)
        report = test.report()
        self.assertEqual(report.frames, 7)
        self.assertEqual(report.size, (20, 60))
        self.assertGreater(report.mean_bytes, 0)
        self.assertEqual(report.blank_frames, [])
        self.assertEqual(len(report.warnings), 1)  # the same warning twice is listed once, with its first frame
        self.assertEqual(report.warnings[0][0], 4)
        self.assertIn("UserWarning: a hazard", str(report))

    def test_a_blank_screen_is_reported(self):
        test = self.play(Game(draw=False))
        test.wait(3)
        self.assertEqual(test.report().blank_frames, [1, 2, 3])
        self.assertIn("BLANK SCREEN in 3 frames", str(test.report()))

    def test_resizing(self):
        test = self.play()
        test.wait(1).resize(10, 30).wait(1)
        self.assertEqual(test.screen.size(), (10, 30))
        self.assertIn("Score 2", test.text())
        run_script(test, "resize 50x12; wait")  # columns first in scripts, as on the command line
        self.assertEqual(test.screen.size(), (12, 50))

    def test_scripts(self):
        test = self.play()
        printed = []
        run_script(test, "wait 2; press enter  # a comment\nhold d 3; click '[Start]'; shot one; dump words",
                   echo=printed.append)
        self.assertEqual(test.frames, 8)  # (hold d 3 is 3 frames held and one let go)
        self.assertEqual(self.game.events[2], [Key.ENTER])
        self.assertTrue(os.path.exists(os.path.join(self.out, "one.png")))
        self.assertIn("Score 8  [Start]", printed[0])
        with self.assertRaisesRegex(ValueError, "unknown command 'jump'"):
            run_script(test, "wait 1; jump")
        self.assertEqual(test.frames, 8)  # nothing played from a script with a bad command
        run_script(test, "press q; expect 'Score 8'; dump; shot end", echo=printed.append)  # looking after quitting
        self.assertEqual(test.quit_at, 9)
        with self.assertRaisesRegex(PlaytestError, "quit at frame 9, before 'wait 1'"):
            run_script(test, "expect Score; wait 1")

    def test_bad_commands_fail_before_anything_plays(self):
        for bad in ["wait ten", "wait -1", "wait 1 2", "hold w", "hold w x", "hold 5", "click 5", "click 5 6 7",
                    "click 5 6 sideways", "move 1", "move a b", "press", "press jump", "type", "resize 80",
                    "resize 24 80", "resize 0x5", "glyphs huge", "color pink", "shot", "shot a b", "sheet s",
                    "sheet s x", "sheet s 0", "expect", "dump all", "up jump", "'unclosed"]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_script(f"wait 1; {bad}")
        test = self.play()
        with self.assertRaisesRegex(ValueError, "bad command 'hold w': hold takes at least 2 arguments"):
            run_script(test, "wait 3; hold w")
        self.assertEqual(test.frames, 0)
        self.assertEqual(len(parse_script("wait; press a b; type 'hello there'; hold w a 3; down x; up; click 1 2 "
                                          "right; click Start Game; move 3 4; resize 80x24; glyphs quad; color 256; "
                                          "shot s; sheet s 10 2; expect a b; expect_not c; dump; dump words")), 18)


class TargetTests(unittest.TestCase):
    def test_loading_a_program(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "mygame.py")
            with open(path, "w") as f:
                f.write(textwrap.dedent("""
                    class Game:
                        def __init__(self, level=1):
                            self.level = level
                        def frame(self, screen, dt, events):
                            screen.text(0, 0, f"level {self.level}")
                            screen.refresh()

                    class Needy:
                        def __init__(self, count):
                            pass

                    def frame(screen, dt, events):
                        screen.text(0, 0, "a function")

                    def make():
                        return Game(7)
                """))
            self.assertEqual(load_target(f"{path}:Game").__self__.level, 1)
            self.assertEqual(load_target(f"{path}:Game(level=3)").__self__.level, 3)
            self.assertEqual(load_target(f"{path}:make").__self__.level, 7)
            self.assertEqual(load_target(f"{path}:frame").__name__, "frame")
            with self.assertRaisesRegex(ValueError, "Needy\\(\\) needs arguments"):
                load_target(f"{path}:Needy")
            with self.assertRaisesRegex(ValueError, "literals"):
                load_target(f"{path}:Game(level=open('x'))")
            with self.assertRaisesRegex(ValueError, "file.py:Name"):
                load_target(path)
            with self.assertRaisesRegex(ValueError, "file.py:Name"):   # not a module "C" (Windows' drive colon)
                load_target(r"C:\games\mygame.py")
            shadow = os.path.join(d, "random.py")
            with open(shadow, "w") as f:
                f.write("def frame(screen, dt, events):\n    pass\n")
            with self.assertRaisesRegex(ValueError, "rename the file"):
                load_target(f"{shadow}:frame")

            out = os.path.join(d, "shots")
            with contextlib.redirect_stdout(io.StringIO()) as printed:
                status = main(["play", f"{path}:Game(level=2)", "wait 3; expect 'level 2'", "--size", "40x10",
                               "--out", out, "--glyphs", "quad"])
            self.assertEqual(status, 0)
            self.assertIn("3 frames", printed.getvalue())
            self.assertIn("40x10 cells, quad glyphs", printed.getvalue())
            with open(os.path.join(out, "last.txt")) as f:
                self.assertEqual(f.read(), "level 2\n" + "\n" * 9)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as err:
                self.assertEqual(main(["play", f"{path}:Game", "expect 'level 9'", "--out", out]), 1)
            self.assertIn("'level 9' is not on the screen", err.getvalue())
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["play", f"{path}:Needy", "--out", out]), 2)


class PreviewTests(unittest.TestCase):
    def test_contact_sheet_of_clips(self):
        root = Node()
        box = Object3D(make_box(), parent=root)
        model = Model(root, [box])
        lift = Animation(box, position=Track([(0.0, (0.0, 0.0, 0.0)), (1.0, (0.0, 1.0, 0.0))]))
        spin = Animation(box, position=Track([(0.0, (0.0, 0.0, 0.0)), (2.0, (1.0, 0.0, 0.0))]))
        model.animations = {"lift": Clip([lift], name="lift"), "slide": Clip([spin], name="slide")}
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "sub", "box.png")
            sheet = preview(model, path, shots=3, cells=20)
            self.assertTrue(os.path.exists(path))
        tile_w, tile_h = 20 * 8, int(20 * 0.62) * 16
        self.assertEqual(sheet.shape[1], 3 * tile_w + 4 * 4)          # three moments across
        self.assertEqual(sheet.shape[0], 2 * (tile_h + 16) + 3 * 4)    # a row per clip, each tile labelled
        self.assertEqual(preview(model, clips=["lift"], shots=2, cells=20).shape[0], tile_h + 16 + 2 * 4)
        with self.assertRaisesRegex(ValueError, "no clips named \\['jump'\\]"):
            preview(model, clips=["jump"])
        still = preview(Model(Node(), [Object3D(make_box())]), cells=10)
        self.assertEqual(still.shape[1], 10 * 8 + 2 * 4)


if __name__ == "__main__":
    unittest.main()
