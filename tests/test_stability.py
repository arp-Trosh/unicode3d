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

from unicode3d.background import Fog, Gradient, Sky, SkyBox
from unicode3d.examples.room import Walk
from unicode3d.keys import HeldKeys, InputDecoder, Key, KeyRelease, MouseEvent
from unicode3d.mesh import Mesh, make_box
from unicode3d.scene import Camera, Light, Object3D, PointLight, Renderer
from unicode3d.terminal import Screen, crash_log_path
from unicode3d.transforms import quat_axis_angle
from unicode3d.ui import Choice, DisplayControls, Panel, Slider, Toggle

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
        self.assertTrue(np.isfinite(fb.rgb).all())

    def test_nan_colours_lights_and_textures_never_reach_the_picture(self):
        # Colours, glow, opacity and light settings gone NaN or infinite: the objects show black, clear or not at
        # all, but the picture stays numbers (and nothing warns).
        nan, inf = np.nan, np.inf
        by_vertex, by_face = make_box(), make_box()
        by_vertex.vertex_colors = np.full((len(by_vertex.vertices), 4), nan)
        by_face.face_colors = np.full((len(by_face.faces), 3), inf)
        alpha = np.ones((4, 4, 4))
        alpha[..., 3] = nan
        meshes = [make_box(), make_box(), by_vertex, by_face, make_box(textures=[np.full((4, 4, 3), nan)] * 6),
                  make_box(textures=[alpha] * 6), make_box(), make_box()]
        options = [{}, {"color": (nan, 0.5, 0.5), "emissive": nan, "reflectivity": nan}, {}, {"specular": inf},
                   {"opacity": 0.5}, {}, {"opacity": nan}, {"color": (inf, 0.0, 0.0), "emissive": -inf}]
        objects = [Object3D(m, position=np.array([1.5 * (i % 4) - 2.0, 1.5 * (i // 4) - 0.5, 0.0]), **o)
                   for i, (m, o) in enumerate(zip(meshes, options))]
        lights = [Light(ambient=nan, color=(nan, 1.0, 1.0), shadows=True), Light(direction=np.full(3, nan), diffuse=inf,
                                                                               shadows=True),
                  PointLight(np.array([0.0, 2.0, 2.0]), range=nan, specular=nan, shadows=True),
                  PointLight(np.full(3, nan), range=inf, shadows=True)]
        settings = [(Sky((nan, 0, 0), (0, nan, 0), (0, 0, nan)), Fog(1.0, 20.0, (nan, 0, 0))), ((nan, nan, nan), 0.3),
                    (Gradient((nan, 0, 0), (inf, 0, 0)), Fog(1.0, 20.0))]
        for background, fog in settings:
            screen = Screen(glyphs="sextant", color="256", size=(20, 40))
            r = Renderer(40, 20, screen.cell_pixels, background=background, fog=fog)
            fb = r.render(objects, Camera(position=np.array([0.0, 0.0, 7.0])), lights)
            screen.draw_frame(fb)
            screen.render_updates()
            self.assertTrue(np.isfinite(fb.rgb).all() and np.isfinite(fb.alpha).all(), background)
            self.assertTrue((fb.ids == 1).any())  # the plain box is drawn

    def test_odd_renderer_settings(self):
        # Settings a slider or a typo can give: shadow maps of no texels, NaN softness, no layers, too many bounces,
        # and a size from rows - 2 in a terminal of one row.
        mirror = Mesh(np.array([(-3, -1, -2), (3, -1, -2), (3, 2, -2), (-3, 2, -2)], float),
                      np.array([(0, 1, 2), (0, 2, 3)]))
        objects = [Object3D(make_box()), Object3D(make_box(), position=np.array([1.5, 0, 0]), opacity=0.5),
                   Object3D(mirror, reflectivity=0.8)]
        lights = [Light(shadows=True), PointLight(np.array([0.0, 2.0, 2.0]), shadows=True)]
        for settings in ({"shadow_size": 0, "point_shadow_size": 0}, {"shadow_size": -4, "point_shadow_size": 16},
                         {"shadow_softness": np.nan, "lod_bias": np.nan, "outline": np.nan},
                         {"transparency_layers": 0, "mirror_bounces": 99}, {"cell_aspect": 0.0}):
            r = Renderer(40, 20, **settings)
            self.assertTrue(np.isfinite(r.render(objects, Camera(), lights).rgb).all(), settings)
        r = Renderer(40, -1)
        self.assertEqual(r.framebuffer.height, 0)
        r.render(objects, Camera(), lights)
        r.resize(40, 20)
        self.assertTrue(r.render(objects, Camera(), lights).drawn.any())

    def test_sliders_bound_to_bad_numbers(self):
        # A slider showing a value it reads (get=) that has gone out of range, infinite or NaN keeps its knob on
        # its track, and moving it gives a value within range again.
        value = [0.0]
        slider = Slider("Speed", 0.0, 0.0, 10.0, step=0.5, get=lambda: value[0], set=lambda v: value.__setitem__(0, v))
        screen = Screen(glyphs="half", color="16", size=(3, 60))
        for bad in (np.nan, np.inf, -np.inf, 1e300, -5.0):
            value[0] = bad
            Panel([slider]).draw(screen, 0, 1)
            self.assertIn(slider.knob(), range(slider.length))
            slider.nudge(1)
            self.assertTrue(0.0 <= value[0] <= 10.0, bad)


class MalformedMeshTests(unittest.TestCase):
    """Mesh arrays that don't fit together are refused with a ValueError saying what is wrong: the kernels index them
    by each other's sizes without checking, so a mismatch used to read outside them, or crash Python outright."""

    def test_mismatched_arrays_are_refused(self):
        def changed(edit, textured=False):
            mesh = make_box(textures=[np.ones((4, 4))] * 6) if textured else make_box()
            edit(mesh)
            return mesh

        def set_face(mesh, value):
            mesh.faces = mesh.faces.copy()
            mesh.faces[3, 1] = value

        cases = [
            (changed(lambda m: set_face(m, 24)), "faces must index its 24 vertices"),
            (changed(lambda m: set_face(m, -1)), "faces must index"),  # (numpy would quietly take the last vertex)
            (changed(lambda m: setattr(m, "faces", m.faces.astype(float))), "whole numbers"),
            (changed(lambda m: setattr(m, "vertices", m.vertices[:, :2])), r"vertices must be \(V, 3\)"),
            (changed(lambda m: setattr(m, "face_colors", np.full((11, 3), 200))), "face_colors has 11 rows for 12"),
            (changed(lambda m: setattr(m, "face_colors", np.full((13, 4), 200))), "face_colors has 13 rows"),
            (changed(lambda m: setattr(m, "vertex_colors", np.full((23, 3), 200))), "vertex_colors has 23 rows"),
            (changed(lambda m: setattr(m, "materials", m.materials[:2]), True), r"materials must be \(12,\)"),
            (changed(lambda m: setattr(m, "materials", m.materials + 1), True), "materials must index its 6"),
            (changed(lambda m: setattr(m, "uvs", m.uvs[:-1]), True), r"uvs must be \(12, 3, 2\)"),
            (make_box(textures=[np.zeros((0, 4, 3))] * 6), "at least one texel"),
            (make_box(textures=[np.ones(5)] * 6), "at least one texel"),
        ]
        r = Renderer(20, 10)
        for mesh, message in cases:
            with self.assertRaisesRegex(ValueError, message):
                r.render([Object3D(make_box()), Object3D(mesh)], Camera(), Light(shadows=True))
        # The renderer goes on drawing good meshes, and a sky box needs textures with texels too.
        self.assertTrue(r.render([Object3D(make_box())], Camera(), Light()).drawn.any())
        with self.assertRaisesRegex(ValueError, "at least one texel"):
            Renderer(20, 10, background=SkyBox([np.zeros((0, 0, 3))] * 6)).render([], Camera(), Light())

    def test_the_engines_own_meshes_fit_together(self):
        from unicode3d.examples.dice import make_die
        from unicode3d.shapes import blob_mesh, block_mesh, merge_meshes, pillow_mesh, text_mesh
        from unicode3d.examples.room import FONT
        meshes = [make_box(), make_box(textures=[np.ones((2, 2))] * 6), make_die(), blob_mesh((1.0, 1.0, 1.0)),
                  block_mesh((0.0, 0.0, 0.0), (1.0, 2.0, 3.0)), pillow_mesh(lambda x, y: x * x + y * y < 1.0),
                  text_mesh("CODE", FONT)[0], merge_meshes([make_box(), blob_mesh((1.0, 1.0, 1.0))], [(255, 0, 0), (0, 0, 255)])]
        for mesh in meshes:
            mesh.check()


class DegenerateCameraTests(unittest.TestCase):
    """Cameras no proper view comes from (near == far, at their own target, a zero up, a fov of 0): drawn as best
    they can be, never an exception or a warning, also with mirrors, which used to invert the view, and in pick()
    and ray(), which did too."""

    def setUp(self):
        caught = warnings.catch_warnings()
        caught.__enter__()
        self.addCleanup(caught.__exit__, None, None, None)
        warnings.simplefilter("error")

    def test_degenerate_cameras(self):
        p = lambda *v: np.array(v, float)  # noqa: E731
        mirror = Mesh(np.array([(-3, -1, -2), (3, -1, -2), (3, 2, -2), (-3, 2, -2)], float),
                      np.array([(0, 1, 2), (0, 2, 3)]))
        objects = [Object3D(make_box()), Object3D(mirror, reflectivity=0.8),
                   Object3D(make_box(), position=p(1.5, 0, 0), opacity=0.5),
                   Object3D(make_box(), position=p(-1.5, 0, 0), reflectivity=0.5)]
        lights = [Light(shadows=True), PointLight(p(0, 2, 2), shadows=True)]
        cameras = [Camera(near=5.0, far=5.0), Camera(near=0.0), Camera(near=-1.0), Camera(near=50.0, far=1.0),
                   Camera(far=np.inf), Camera(near=np.nan), Camera(fov=0.0), Camera(fov=180.0), Camera(fov=-50.0),
                   Camera(fov=np.nan), Camera(position=p(0, 0, 0), target=p(0, 0, 0)), Camera(up=p(0, 0, 0)),
                   Camera(up=p(np.nan, 0, 0)), Camera(position=p(0, 5, 0), target=p(0, 0, 0), up=p(0, 1, 0)),
                   Camera(position=p(0, 0, 1e300)), Camera(position=p(0, 0, np.inf)), Camera(target=p(np.inf, 0, 0))]
        sky_box = SkyBox([np.full((4, 4, 3), v) for v in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)])
        screen = Screen(glyphs="sextant", color="256", size=(20, 40))
        for camera in cameras:
            for background in (Sky(), sky_box, None):
                r = Renderer(40, 20, screen.cell_pixels, background=background, fog=Fog(1.0, 5.0))
                fb = r.render(objects, camera, lights)
                screen.draw_frame(fb)
                screen.render_updates()
                self.assertTrue(np.isfinite(fb.rgb).all() and np.isfinite(fb.alpha).all(), camera)
                for x, y in ((20, 10), (0, 0), (39.5, 19.5)):
                    r.pick(x, y)
                    r.ray(x, y)
                r.project((0.0, 0.0, 0.0))
        # near == far still draws the scene: only depth is lost, which a perspective's 1/w doesn't need.
        r = Renderer(40, 20)
        self.assertTrue(r.render(objects, Camera(near=1.0, far=1.0), lights).drawn.any())
        self.assertIs(r.pick(20, 10).object, objects[0])


class GarbledInputTests(unittest.TestCase):
    """Whatever the terminal sends (line noise over ssh, a paste of escape sequences, a terminal of its own mind),
    the decoder and the widgets never raise."""

    def test_garbled_input_never_raises(self):
        rng = np.random.default_rng(7)
        pieces = ["\x1b", "\x1b[", "\x1b[<", "\x1bO", "[", "<", ";", ":", "u", "~", "M", "m", "?", ">", "-", "A", "Z", "1",
                  "9", "0", "65", "99999999999999999999999", "1114112", "57441", "a", "é", "\x00", "\x7f", "\U0001fb00"]
        decoder, held = InputDecoder(), HeldKeys()
        controls = DisplayControls(renderer=Renderer(10, 5))
        panel = Panel([Slider("x", 5, 0, 10, keys="[]"), Toggle("t", key="t"), Choice("c", ("a", "b"), key="c")])
        screen = Screen(glyphs="sextant", color="256", size=(6, 80))
        now = 0.0
        for i in range(3000):
            text = "".join(rng.choice(pieces, size=int(rng.integers(1, 12))))
            now += float(rng.uniform(0.0, 0.05))
            events = decoder.feed(text, now) + decoder.flush(now)
            for event in events:
                self.assertIsInstance(event, (int, KeyRelease, MouseEvent))
            held.update(events, now)
            panel.draw(screen, 0, 0)
            controls.draw(screen, 1, 0)
            controls.handle(panel.handle(events), screen)
        for sequence in ("\x1b[97;5;99999999999999999999u", "\x1b[1114112u", "\x1b[97;1;1114112u", "\x1b[97;1;-3u",
                         "\x1b[<0;-5;-7M", "\x1b[" + "9" * 5000 + "~", "\x1b[<99999999999999999999;1;1M", "\x1b[;;;u"):
            InputDecoder().feed(sequence, 0.0)


class ThreadTests(unittest.TestCase):
    """Drawing from several threads at once (a server drawing for each client, say), each with its own Renderer and
    Screen: the same frames as drawn one at a time, and no crash on any of Numba's threading layers."""

    def test_threads_draw_the_same_frames(self):
        import threading

        def scene(seed):
            rng = np.random.default_rng(seed)
            return [Object3D(make_box(), position=rng.normal(size=3), opacity=0.6 if i % 3 == 0 else 1.0,
                             color=tuple(int(c) for c in rng.integers(0, 255, 3))) for i in range(12)]

        lights = [Light(shadows=True)]
        alone = {s: Renderer(40, 20, (2, 3)).render(scene(s), Camera(), lights).copy() for s in range(3)}
        wrong = []

        def draw(seed):
            r, screen = Renderer(40, 20, (2, 3)), Screen(glyphs="sextant", size=(20, 40))
            for _ in range(8):
                r.invalidate()
                fb = r.render(scene(seed), Camera(), lights)
                screen.draw_frame(fb)
                if not (np.array_equal(fb.rgb, alone[seed].rgb) and np.array_equal(fb.ids, alone[seed].ids)):
                    wrong.append(seed)

        threads = [threading.Thread(target=draw, args=(s,)) for s in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(wrong, [])

    def test_threads_on_the_workqueue_layer(self):
        # Numba's fallback threading layer aborts the process when two threads launch parallel kernels at once;
        # kernel_lock() makes them take turns there.
        import subprocess
        program = ("import threading, numpy as np\n"
                   "from unicode3d import Camera, Light, Object3D, Renderer, make_box\n"
                   "from unicode3d.terminal import Screen\n"
                   "def draw():\n"
                   "    r, screen = Renderer(40, 20, (2, 3)), Screen(glyphs='sextant', size=(20, 40))\n"
                   "    for _ in range(15):\n"
                   "        r.invalidate()\n"
                   "        screen.draw_frame(r.render([Object3D(make_box())], Camera(), Light(shadows=True)))\n"
                   "threads = [threading.Thread(target=draw) for _ in range(3)]\n"
                   "[t.start() for t in threads]; [t.join() for t in threads]\n"
                   "import numba; print(numba.threading_layer())\n")
        result = subprocess.run([sys.executable, "-c", program], cwd=ROOT, capture_output=True, text=True, timeout=600,
                                env=dict(os.environ, NUMBA_THREADING_LAYER="workqueue"))
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        self.assertEqual(result.stdout.strip(), "workqueue")


class FrameRateTests(unittest.TestCase):
    def test_any_frame_rate_setting(self):
        # screen.fps is the program's to change while it runs: 0 or None (as fast as it can) and nonsense mustn't
        # stop run() with a ZeroDivisionError or a TypeError.
        from unicode3d.terminal import _frame_period
        self.assertAlmostEqual(_frame_period(30), 1 / 30)
        for fps in (0, 0.0, None, -5, np.nan, np.inf, "fast"):
            self.assertEqual(_frame_period(fps), 0.0, fps)


class CrashLogTests(unittest.TestCase):
    def test_crash_log_path(self):
        self.assertEqual(crash_log_path({"XDG_CACHE_HOME": "/c"}, windows=False),
                         os.path.join("/c", "unicode3d", "crash.log"))
        self.assertEqual(crash_log_path({"LOCALAPPDATA": "D:\\l"}, windows=True),
                         os.path.join("D:\\l", "unicode3d", "crash.log"))
        self.assertTrue(crash_log_path({}, windows=False).endswith(os.path.join(".cache", "unicode3d", "crash.log")))


def run_in_pty(args, rows, cols, script, env=None, timeout=240):
    """Run `python args` in a pseudo-terminal of rows x cols, reading everything it writes. `script` is a list
    of (frames, action): once that many frames (synchronized updates) have arrived, action(fd, set_size, pid)
    runs, e.g. to resize the terminal, type a key or send the program a signal; or sooner, once a frame has
    come since the last action and a few seconds have passed (a frame where nothing changed sends nothing).
    Returns (exit code, output)."""
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
                steps.pop(0)[1](fd, set_size, pid)
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
            return lambda fd, set_size, pid: set_size(fd, r, c)

        with tempfile.TemporaryDirectory() as cache:
            code, out = run_in_pty(["-m", "unicode3d.examples.room"], *TINY_FONT, [
                (2, resize(24, 80)), (4, resize(300, 1200)), (6, resize(1, 1)), (8, resize(50, 180)),
                (10, lambda fd, *_: os.write(fd, b"\x1b"))], env={"XDG_CACHE_HOME": cache})
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

    def test_sigterm_restores_the_terminal(self):
        # kill (SIGTERM) ends the program as sys.exit() would, so the terminal leaves raw mode and the alternate
        # screen; by default the signal ends Python at once, leaving both. (And fps=0 runs as fast as it can.)
        import signal
        program = ("from unicode3d.terminal import run\n"
                   "frames = 0\n"
                   "def frame(screen, dt, keys):\n"
                   "    global frames\n"
                   "    frames += 1\n"
                   "    screen.text(0, 0, str(frames)); screen.refresh()\n"
                   "run(frame, fps=0)\n")
        with tempfile.TemporaryDirectory() as cache:
            code, out = run_in_pty(["-c", program], 24, 80, [(20, lambda fd, set_size, pid: os.kill(pid, signal.SIGTERM))],
                                   env={"XDG_CACHE_HOME": cache})
            with open(crash_log_path({"XDG_CACHE_HOME": cache}, windows=False)) as f:
                log = f.read()
        self.assertEqual(code, 128 + signal.SIGTERM, out[-2000:])
        self.assertIn(b"\x1b[?2026l", out[-200:])  # a frame cut short is ended
        self.assertIn(b"\x1b[?1049l", out[-200:])  # the alternate screen left
        self.assertNotIn("crashed", log)


if __name__ == "__main__":
    unittest.main()
