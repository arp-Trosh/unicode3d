# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
"""Stability: huge terminals (a full screen at a tiny font), resizing, bad numbers, the real terminal, crash logs.

Frames may take longer in a huge terminal, but nothing may crash, and memory has to stay bounded.
"""
import importlib
import os
import pkgutil
import select
import sys
import tempfile
import time
import unittest
import warnings

import numpy as np
import unicode3d
from numba.core.registry import CPUDispatcher

from unicode3d.background import Fog, Sky, SkyBox
from unicode3d.examples.room import Walk
from unicode3d.keys import Key, MouseEvent
from unicode3d.mesh import Mesh, make_box
from unicode3d.scene import Camera, Light, Object3D, PointLight, Renderer
from unicode3d.terminal import Screen, crash_log_path
from unicode3d.transforms import quat_axis_angle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Full screen at a tiny font: kitty at font size 2 on a 1080p screen, and about a 4K screen's worth.
TINY_FONT, HUGE = (215, 960), (540, 1920)
SYNC = b"\x1b[?2026h"  # starts each frame's output


class FakeConsole:
    """A console whose size the test sets, and that takes no input: poll_size() picks up the size."""
    unicode = True
    key_release = False
    reports_releases = False

    def __init__(self, rows, cols):
        self.rows, self.cols = rows, cols
        self.written = 0

    def size(self):
        return self.cols, self.rows

    def read(self):
        return ""

    def key_is_down(self, key):
        return None

    def write(self, data):
        self.written += len(data)


def walk_frames(demo, screen, count, keys=(), held=()):
    """Draw `count` frames of the room demo, walking and turning, with `keys` pressed one a frame."""
    screen.held.exact = True
    screen.held.update([ord(k) if isinstance(k, str) else k for k in held], 0.0)
    for i in range(count):
        events = [keys[i]] if i < len(keys) else []
        if screen.console is not None:
            screen.poll_size()
        assert demo.frame(screen, 0.05, events) is not False
        if screen.console is None:
            screen.render_updates()  # (the demo refreshes a console itself)


class HugeTerminalTests(unittest.TestCase):
    def test_room_in_a_full_screen_terminal_with_a_tiny_font(self):
        demo = Walk(seed=1)
        screen = Screen(glyphs="sextant", color="truecolor", size=TINY_FONT)
        rows, cols = TINY_FONT
        clicks = [MouseEvent(cols // 2, rows // 3, MouseEvent.LEFT, True), MouseEvent(cols + 3, rows + 3, 0, True)]
        walk_frames(demo, screen, 8, keys=[ord("b"), ord("l"), ord("u"), ord("]"), *clicks], held="wd")
        self.assertTrue((screen.chars != " ").mean() > 0.5)  # the whole screen is drawn
        self.assertEqual(demo.renderer.framebuffer.width, cols * 2)
        self.assertEqual(demo.renderer.drawn_size, (cols * 2, (rows - 2) * 3))  # within the budget: full detail

    def test_every_glyph_set_and_colour_mode_in_a_huge_terminal(self):
        demo = Walk(seed=2)
        screen = Screen(glyphs="sextant", color="truecolor", size=TINY_FONT)
        # F2 through all four glyph sets, F3 through all four colour modes, F5 and F6 both ways.
        keys = [Key.F2] * 4 + [Key.F3] * 4 + [Key.F5, Key.F6, Key.F5, Key.F6]
        walk_frames(demo, screen, len(keys) + 1, keys=keys, held="a")
        self.assertEqual((screen.mode, screen.color_mode), ("sextant", "truecolor"))

    def test_rendering_stays_within_the_pixel_budget(self):
        demo = Walk(seed=3)
        screen = Screen(glyphs="sextant", color="256", size=HUGE)
        walk_frames(demo, screen, 2, held=[Key.LEFT])
        r = demo.renderer
        w, h = r.drawn_size
        self.assertLessEqual(w * h, r.max_pixels)
        self.assertGreater(w * h, 0.95 * r.max_pixels)  # but no smaller than it has to be
        self.assertEqual((r.framebuffer.width, r.framebuffer.height), (HUGE[1] * 2, (HUGE[0] - 2) * 3))
        self.assertAlmostEqual(w / h, r.framebuffer.width / r.framebuffer.height, delta=0.01)
        self.assertTrue((screen.chars != " ").mean() > 0.5)
        hit = r.pick(HUGE[1] // 2, HUGE[0] // 2)  # picking works on the stretched frame
        self.assertIsNotNone(hit)

    def test_resizing_while_running(self):
        # A terminal whose font is made smaller and bigger while the program runs, down to a single cell.
        console = FakeConsole(24, 80)
        demo = Walk(seed=4)
        screen = Screen(console, glyphs="sextant", color="truecolor")
        for size in [(24, 80), TINY_FONT, (1, 1), (2, 3), (3, 200), (300, 4), HUGE, (217, 961), (24, 80)]:
            console.rows, console.cols = size
            walk_frames(demo, screen, 2, keys=[MouseEvent(size[1] - 1, size[0] - 1, 0, True)], held="w")
            self.assertEqual(screen.size(), size)
        self.assertGreater(console.written, 0)


class PixelBudgetTests(unittest.TestCase):
    def scene(self):
        objects = [Object3D(make_box(textures=[np.ones((4, 4))] * 6), rotation=quat_axis_angle((1, 1, 0), 0.6)),
                   Object3D(make_box(), position=np.array([1.2, 0.0, -1.0]), color=(200, 50, 50), opacity=0.5)]
        return objects, Camera(position=np.array([0.0, 0.5, 4.0])), [Light(shadows=True)]

    def test_a_smaller_budget_draws_the_same_picture_coarser(self):
        objects, camera, light = self.scene()
        full = Renderer(80, 30, (2, 3), max_pixels=None).render(objects, camera, light).copy()
        small = Renderer(80, 30, (2, 3), max_pixels=full.width * full.height // 4)
        fb = small.render(objects, camera, light)
        self.assertEqual(small.drawn_size, (full.width // 2, full.height // 2))
        self.assertEqual((fb.width, fb.height), (full.width, full.height))
        self.assertGreater(((fb.alpha > 0.5) == (full.alpha > 0.5)).mean(), 0.97)
        self.assertGreater((fb.ids == full.ids).mean(), 0.97)
        self.assertLess(np.abs(fb.colour() - full.colour())[(fb.alpha > 0.9) & (full.alpha > 0.9)].mean(), 0.05)
        self.assertIs(small.pick(40, 15).object, full_pick(objects, camera, light).object)

    def test_budget_changes_take_effect(self):
        objects, camera, light = self.scene()
        r = Renderer(80, 30, (2, 3))
        self.assertEqual(r.drawn_size, (160, 90))  # far below the default budget
        r.render(objects, camera, light)
        r.max_pixels = 1000
        draws = r.draws
        fb = r.render(objects, camera, light)
        self.assertEqual(r.draws, draws + 1)  # a new budget redraws
        self.assertLessEqual(np.prod(r.drawn_size), 1000)
        self.assertEqual(fb.width, 160)
        r.max_pixels = 0  # no limit
        r.render(objects, camera, light)
        self.assertEqual(r.drawn_size, (160, 90))


def full_pick(objects, camera, light):
    r = Renderer(80, 30, (2, 3), max_pixels=None)
    r.render(objects, camera, light)
    return r.pick(40, 15)


class BadNumberTests(unittest.TestCase):
    """NaN and infinities (from a physics blow-up, say) are drawn around: no exception, no warning."""

    def setUp(self):
        caught = warnings.catch_warnings()
        caught.__enter__()
        self.addCleanup(caught.__exit__, None, None, None)
        warnings.simplefilter("error")  # a warning would be printed over the picture

    def test_every_kernel_uses_the_numpy_error_model(self):
        # Numba's default raises ZeroDivisionError on a float division by zero (or, in a helper called from
        # a parallel loop, SystemError), which ends the program; with numpy's, it gives inf or NaN.
        kernels = []
        for name in (m.name for m in pkgutil.iter_modules(unicode3d.__path__) if not m.ispkg and m.name != "__main__"):
            module = importlib.import_module(f"unicode3d.{name}")
            kernels += [(name, k) for k, v in vars(module).items()
                        if isinstance(v, CPUDispatcher) and v.__module__ == module.__name__]
        self.assertGreater(len(kernels), 40)
        for name, k in kernels:
            module = importlib.import_module(f"unicode3d.{name}")
            self.assertEqual(getattr(module, k).targetoptions.get("error_model"), "numpy", f"{name}.{k}")

    def test_see_through_surface_fading_to_nothing_with_fog(self):
        # It used to leave pixels covered but without a depth, and fog divided by that depth.
        def quad(z, s, alphas):
            m = Mesh(np.array([(-s, -s, z), (s, -s, z), (s, s, z), (-s, s, z)], float),
                     np.array([(0, 1, 2), (0, 2, 3)]))
            m.vertex_colors = np.array([(255, 255, 255, a) for a in alphas])
            return m

        far = Object3D(quad(-1.0, 3.0, (128,) * 4))
        near = Object3D(quad(0.0, 1.0, (0, 255, 255, 0)), rotation=quat_axis_angle((0, 0, 1), 0.3))
        fb = Renderer(40, 20, fog=0.3).render([near, far], Camera(), Light())
        self.assertFalse(((fb.alpha > 0) & (fb.depth == 0)).any())

    def test_nan_and_infinite_positions_and_vertices(self):
        box = make_box(textures=[np.ones((4, 4))] * 6)
        box.vertices = box.vertices.copy()
        box.vertices[0] = np.nan
        spike = make_box()
        spike.vertices = spike.vertices.copy()
        spike.vertices[3] = (1e300, -np.inf, 5.0)
        holes = np.ones((8, 8, 4))
        holes[::2, :, 3] = 0.0
        cutout = make_box(textures=[holes] * 6)
        cutout.vertices = cutout.vertices.copy()
        cutout.vertices[5] = (np.inf, 0.0, 0.0)
        good = Object3D(make_box(), position=np.array([-1.5, 0.0, 0.0]), reflectivity=0.6)
        objects = [Object3D(box), Object3D(spike, position=np.array([1.5, 0, 0]), opacity=0.5), good,
                   Object3D(cutout, position=np.array([0.0, 1.5, 0.0])),
                   Object3D(make_box(), position=np.array([np.nan, 0.0, 0.0])),
                   Object3D(make_box(), rotation=np.array([np.inf, 0.0, 0.0, 0.0])),
                   Object3D(make_box(), scale=np.nan)]
        mirror = Mesh(np.array([(-3, -1, -2), (3, -1, -2), (3, 2, -2), (-3, 2, -2)], float),
                      np.array([(0, 1, 2), (0, 2, 3)]))
        objects.append(Object3D(mirror, reflectivity=0.8))
        sky = SkyBox([np.full((4, 4, 3), v) for v in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)])
        for background in (sky, Sky(), None):
            screen = Screen(glyphs="sextant", color="256", size=(20, 40))
            r = Renderer(40, 20, screen.cell_pixels, background=background)
            fb = r.render(objects, Camera(), [Light(shadows=True), PointLight(np.array([0.0, 2.0, 2.0]), shadows=True)])
            screen.draw_frame(fb)
            screen.render_updates()
            self.assertTrue(np.isfinite(fb.rgb).all())
            self.assertTrue((fb.ids == 3).any())  # the good box is drawn
        everything_bad = [o for i, o in enumerate(objects) if i in (4, 5, 6)]
        self.assertFalse(Renderer(40, 20).render(everything_bad, Camera(), Light()).drawn.any())

    def test_bad_fog_materials_and_scales(self):
        mirror = Mesh(np.array([(-3, -1, -2), (3, -1, -2), (3, 2, -2), (-3, 2, -2)], float),
                      np.array([(0, 1, 2), (0, 2, 3)]))
        good = Object3D(make_box(), position=np.array([-1.5, 0.0, 0.0]))
        objects = [good,
                   Object3D(make_box(), scale=(0.0, 1.0, 1.0)),  # squashed flat: no inverse for its normals
                   Object3D(make_box(), scale=(0.0, 0.0, 0.0)),
                   Object3D(make_box(), scale=(np.inf, 1.0, 1.0)),
                   Object3D(make_box(), scale=(1e300, 1e-300, 1.0)),
                   Object3D(make_box(), position=np.array([1.5, 0.0, 0.0]), specular=np.nan, shininess=np.nan),
                   Object3D(make_box(), position=np.array([0.0, 1.5, 0.0]), specular=np.inf, shininess=-5.0),
                   Object3D(make_box(), position=np.array([0.0, -1.5, 0.0]), specular=1.0, shininess=1e308),
                   Object3D(mirror, reflectivity=0.8, scale=(1.0, 1.0, 0.0)),
                   Object3D(mirror, reflectivity=0.8, scale=(0.0, 1.0, 1.0), position=np.array([0.0, 0.0, -1.0]))]
        fogs = [Fog(np.nan, np.inf), Fog(10.0, 5.0), Fog(0.0, 0.0), Fog(-np.inf, np.nan), Fog(1.0, np.inf),
                Fog(-np.inf, 3.0, (255, 0, 0)), np.nan, np.inf]
        for fog in fogs:
            screen = Screen(glyphs="sextant", color="256", size=(20, 40))
            r = Renderer(40, 20, screen.cell_pixels, background=Sky(), fog=fog)
            fb = r.render(objects, Camera(), [Light(shadows=True), PointLight(np.array([0.0, 2.0, 2.0]), shadows=True)])
            screen.draw_frame(fb)
            screen.render_updates()
            self.assertTrue((fb.ids == 1).any(), fog)  # the good box is drawn
            self.assertTrue(np.isfinite(fb.rgb).all() and np.isfinite(fb.alpha).all(), fog)

    def test_bad_texture_coordinates_and_normals(self):
        # Mip levels come from how fast uv change across the screen, and textures repeat beyond 0..1: neither may
        # turn NaN, infinite or huge coordinates into an index outside the texture.
        holes = np.ones((8, 8, 4))
        holes[::2, :, 3] = 0.0
        objects = []
        for i, bad in enumerate((np.nan, np.inf, -np.inf, 1e300, -1e300, 1e-300)):
            box = make_box(textures=[holes if i % 2 else np.ones((8, 8, 3))] * 6)
            box.uvs = box.uvs.copy()
            box.uvs[::3, 1] = bad
            box.normals = np.full((len(box.vertices), 3), bad)
            box.normals[::2] = (0.0, 1.0, 0.0)
            objects.append(Object3D(box, position=np.array([1.2 * i - 3.0, 0.0, 0.0]), opacity=1.0 - 0.3 * (i % 3 == 2)))
        good = Object3D(make_box(), position=np.array([0.0, 1.5, 0.0]))
        screen = Screen(glyphs="sextant", color="256", size=(20, 40))
        fb = Renderer(40, 20, screen.cell_pixels, background=Sky()).render(
            [good, *objects], Camera(position=np.array([0.0, 0.0, 6.0])), [Light(shadows=True), PointLight(
                np.array([0.0, 2.0, 2.0]), shadows=True)])
        screen.draw_frame(fb)
        screen.render_updates()
        self.assertTrue((fb.ids == 1).any())
        self.assertTrue(np.isfinite(fb.rgb).all())

    def test_camera_at_nan(self):
        screen = Screen(glyphs="quad", color="truecolor", size=(20, 40))
        fb = Renderer(40, 20, screen.cell_pixels, background=Sky()).render(
            [Object3D(make_box())], Camera(position=np.array([np.nan] * 3)), Light(shadows=True))
        screen.draw_frame(fb)
        screen.render_updates()


class CrashLogTests(unittest.TestCase):
    def test_crash_log_path(self):
        self.assertEqual(crash_log_path({"XDG_CACHE_HOME": "/c"}, windows=False),
                         os.path.join("/c", "unicode3d", "crash.log"))
        self.assertEqual(crash_log_path({"LOCALAPPDATA": "D:\\l"}, windows=True),
                         os.path.join("D:\\l", "unicode3d", "crash.log"))
        self.assertTrue(crash_log_path({}, windows=False).endswith(os.path.join(".cache", "unicode3d", "crash.log")))


def run_in_pty(args, rows, cols, script, env=None, timeout=240):
    """Run `python args` in a pseudo-terminal of rows x cols, reading everything it writes. `script` is a list
    of (frames, action): once that many frames (synchronized updates) have arrived, action(fd, set_size) runs,
    e.g. to resize the terminal or type a key; or sooner, once a frame has come since the last action and
    a few seconds have passed (a frame where nothing changed sends nothing). Returns (exit code, output)."""
    import fcntl
    import pty
    import struct
    import termios

    def set_size(fd, r, c):
        fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", r, c, 0, 0))

    pid, fd = pty.fork()
    if pid == 0:  # the child: the program in its terminal
        try:
            os.chdir(ROOT)
            os.execve(sys.executable, [sys.executable, *args], dict(os.environ, TERM="xterm-256color", **(env or {})))
        finally:
            os._exit(127)
    set_size(fd, rows, cols)
    out, frames, deadline, steps = b"", 0, time.monotonic() + timeout, list(script)
    since, then = None, 0  # frames when the last action ran (None: none yet), and when that was
    try:
        while time.monotonic() < deadline:
            ready = select.select([fd], [], [], 0.05)[0]
            if ready:
                try:
                    data = os.read(fd, 1 << 20)
                except OSError:  # the program has exited and closed the terminal
                    break
                if not data:
                    break
                frames += (out[-(len(SYNC) - 1):] + data).count(SYNC)  # (a marker split between reads too)
                out = (out + data)[-65536:]
            while steps and (frames >= steps[0][0] or since is not None and frames > since
                             and time.monotonic() - then > 3.0):
                steps.pop(0)[1](fd, set_size)
                since, then = frames, time.monotonic()
        else:
            os.kill(pid, 9)
            raise AssertionError(f"still running after {timeout} s:\n{out[-2000:]!r}")
        _, status = os.waitpid(pid, 0)
    finally:
        os.close(fd)
    return os.waitstatus_to_exitcode(status), out


@unittest.skipIf(os.name == "nt", "needs a Unix pseudo-terminal")
class RealTerminalTests(unittest.TestCase):
    """The programs in a real (pseudo-)terminal: its size, resizes while running, output, keys, and crashes."""

    def test_room_in_a_resized_terminal(self):
        def resize(r, c):
            return lambda fd, set_size: set_size(fd, r, c)

        with tempfile.TemporaryDirectory() as cache:
            code, out = run_in_pty(["-m", "unicode3d.examples.room"], *TINY_FONT, [
                (2, resize(24, 80)), (4, resize(300, 1200)), (6, resize(1, 1)), (8, resize(50, 180)),
                (10, lambda fd, _: os.write(fd, b"\x1b"))], env={"XDG_CACHE_HOME": cache})
            with open(crash_log_path({"XDG_CACHE_HOME": cache}, windows=False)) as f:
                log = f.read()
        self.assertEqual(code, 0, out[-3000:])
        self.assertIn(b"\x1b[?1049l", out[-200:])  # the terminal was restored
        self.assertIn("started", log)
        self.assertIn("960x215 cells", log)
        self.assertNotIn("crashed", log)

    def test_a_crash_is_logged(self):
        program = ("from unicode3d.terminal import run\n"
                   "def frame(screen, dt, keys):\n"
                   "    screen.text(0, 0, 'x'); screen.refresh()\n"
                   "    if screen.measured_fps is None and dt < 10: raise RuntimeError('broken frame')\n"
                   "run(frame)\n")
        with tempfile.TemporaryDirectory() as cache:
            code, out = run_in_pty(["-c", program], 24, 80, [], env={"XDG_CACHE_HOME": cache})
            path = crash_log_path({"XDG_CACHE_HOME": cache}, windows=False)
            with open(path) as f:
                log = f.read()
        self.assertEqual(code, 1)
        text = out.decode("utf-8", "replace")
        self.assertIn(f"the details are also in {path}", text)
        self.assertLess(text.rfind("\x1b[?1049l"), text.find("unicode3d: stopped"))  # after restoring the terminal
        self.assertIn("RuntimeError: broken frame", log)
        self.assertIn("crashed in frame 1: -c: 80x24 cells", log)


if __name__ == "__main__":
    unittest.main()
