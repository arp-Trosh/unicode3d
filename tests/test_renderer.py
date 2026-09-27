# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
import os
import tempfile
import unittest

import numpy as np

from unicode3d.examples.dice import DIE_VALUES, RollAnimation, make_die, orientation_showing, top_face
from unicode3d.mesh import Mesh, load_obj, make_box
from unicode3d.color import Color, linear_to_srgb, luminance, quantize, sgr_color, srgb_to_linear, to_linear_rgb, xterm_rgb
from unicode3d.console import WindowsInput, detect_color_mode, detect_glyphs
from unicode3d.glyphs import GLYPH_SETS, frame_to_text, match_cells
from unicode3d.keys import InputDecoder, Key, MouseEvent
from unicode3d.raster import FrameBuffer
from unicode3d.scene import Camera, Light, Object3D, Renderer
from unicode3d.shapes import block_mesh, merge_meshes, text_mesh
from unicode3d.terminal import Screen
from unicode3d.texture import build_mipmaps
from unicode3d.transforms import quat_axis_angle, quat_between, quat_to_matrix


class TransformTests(unittest.TestCase):
    def test_quat_between_maps_u_onto_v(self):
        rng = np.random.default_rng(1)
        pairs = [(rng.normal(size=3), rng.normal(size=3)) for _ in range(20)]
        pairs += [((0, 1, 0), (0, -1, 0)), ((1, 0, 0), (-1, 0, 0)), ((0, 0, 1), (0, 0, 1))]
        for u, v in pairs:
            u, v = np.asarray(u, float), np.asarray(v, float)
            got = quat_to_matrix(quat_between(u, v)) @ (u / np.linalg.norm(u))
            np.testing.assert_allclose(got, v / np.linalg.norm(v), atol=1e-9)


class DiceTests(unittest.TestCase):
    def test_orientation_showing_puts_face_on_top(self):
        for face in range(6):
            for yaw in (0.0, 1.0, 4.0):
                self.assertEqual(top_face(orientation_showing(face, yaw)), face)

    def test_opposite_faces_sum_to_seven(self):
        for face in range(0, 6, 2):
            self.assertEqual(DIE_VALUES[face] + DIE_VALUES[face + 1], 7)

    def test_roll_ends_at_rest_showing_chosen_face(self):
        rng = np.random.default_rng(7)
        rest = np.array([1.0, 0.5, 0.0])
        for face in range(6):
            anim = RollAnimation(face, rest, rng)
            self.assertLess(anim.duration, 3.0)
            pos, rot = anim.pose(anim.duration + 1.0)
            np.testing.assert_allclose(pos, rest, atol=1e-9)
            self.assertEqual(top_face(rot), face)
            start_pos, _ = anim.pose(0.0)
            self.assertGreater(start_pos[1], rest[1] + 4.0)


def volume(mesh):
    v = mesh.vertices[mesh.faces]
    return np.einsum("ij,ij->i", v[:, 0], np.cross(v[:, 1], v[:, 2])).sum() / 6


class ShapeTests(unittest.TestCase):
    def test_text_mesh_is_closed(self):
        # Every edge of a closed, consistently wound mesh is shared by exactly two triangles in opposite directions
        # -- except where merged runs create T-junctions, so just check the volume is right instead.
        font = {"I": ["###", ".#.", ".#.", ".#.", ".#.", ".#.", "###"]}
        mesh, width = text_mesh("II", font, depth=1.0)
        self.assertAlmostEqual(volume(mesh), 22.0)  # "I" has 11 pixels
        self.assertEqual(width, 7)

    def test_merged_meshes_keep_their_volume(self):
        mesh = merge_meshes([block_mesh((0.0, 0.0, 0.0), (1.0, 2.0, 3.0)), block_mesh((5.0, 0.0, 0.0), (1.0, 1.0, 1.0))])
        self.assertAlmostEqual(volume(mesh), 7.0)
        self.assertEqual(len(mesh.faces), 24)


def lum(fb):
    """Luminance of each pixel's own colour."""
    return luminance(fb.colour())


class RenderTests(unittest.TestCase):
    def render(self, objects, w=80, h=30, camera=None, **kwargs):
        camera = camera or Camera(position=np.array([0.0, 0.0, 5.0]))
        return Renderer(w, h, **kwargs).render(objects, camera, Light())

    def test_die_is_drawn_centred(self):
        fb = self.render([Object3D(make_die())])
        self.assertEqual((fb.width, fb.height), (80, 60))  # two pixels per cell, stacked
        drawn = fb.alpha > 0.5
        self.assertTrue(drawn[30, 40])
        self.assertFalse(fb.drawn[0, 0] or fb.drawn[-1, -1])
        rows, cols = np.nonzero(drawn)
        self.assertAlmostEqual(rows.mean(), 30, delta=1.5)
        self.assertAlmostEqual(cols.mean(), 40, delta=1.5)
        # The centre pip is darker than the lit face around it.
        self.assertLess(lum(fb)[30, 40], 0.5 * lum(fb)[30, 44])
        self.assertTrue(set(frame_to_text(fb)) & set(".:-"))
        self.assertEqual(len(frame_to_text(fb).splitlines()), 30)

    def test_die_is_round_not_stretched(self):
        # Pixels are about square, so a cube seen face-on covers a square of pixels.
        fb = self.render([Object3D(make_box())], samples=1, fog=0, outline=0)
        rows, cols = np.nonzero(fb.drawn)
        self.assertAlmostEqual(np.ptp(rows) / np.ptp(cols), 1.0, delta=0.1)

    def test_finer_cells_keep_proportions(self):
        # 2x3 pixels per cell: pixels are then narrower, and a cube must still come out square on screen.
        fb = self.render([Object3D(make_box())], cell_pixels=(2, 3), samples=1, fog=0, outline=0)
        self.assertEqual((fb.width, fb.height), (160, 90))
        rows, cols = np.nonzero(fb.drawn)
        cell_h, cell_w = np.ptp(rows) / 3, np.ptp(cols) / 2
        self.assertAlmostEqual(cell_h / (cell_w * 0.5), 1.0, delta=0.1)  # a cell is half as wide as tall

    def test_nearer_object_wins_depth_test(self):
        near = Object3D(make_box(), position=np.array([0.0, 0.0, 1.0]))
        far = Object3D(make_box(2.0), position=np.array([0.0, 0.0, -2.0]))
        for order, near_id in (([near, far], 1), ([far, near], 2)):
            fb = self.render(order)
            self.assertEqual(fb.ids[30, 40], near_id)

    def test_unchanged_scene_is_not_redrawn(self):
        renderer, camera, light = Renderer(40, 15), Camera(position=np.array([0.0, 0.0, 5.0])), Light()
        box = Object3D(make_box())
        first = renderer.render([box], camera, light).copy()
        draws = renderer.draws
        renderer.render([box], Camera(position=np.array([0.0, 0.0, 5.0])), Light())  # equal, not the same objects
        self.assertEqual(renderer.draws, draws)
        np.testing.assert_array_equal(renderer.framebuffer.rgb, first.rgb)
        for change in (lambda: setattr(box, "position", np.array([0.1, 0.0, 0.0])),
                       lambda: setattr(camera, "fov", 40.0),
                       lambda: setattr(light, "ambient", 0.5),
                       lambda: setattr(box.mesh, "vertices", box.mesh.vertices * 1.1),  # replaced, as Mesh's caches expect
                       lambda: renderer.resize(41, 15),
                       renderer.invalidate):
            change()
            renderer.render([box], camera, light)
            renderer.render([box], camera, light)
            self.assertEqual(renderer.draws, draws + 1)
            draws = renderer.draws

    def test_back_faces_are_culled(self):
        fb = self.render([Object3D(make_box(), position=np.array([0.0, 0.0, 6.0]))])  # camera inside the box
        self.assertFalse(fb.drawn.any())

    def test_supersampling_matches_plain_render(self):
        plain = self.render([Object3D(make_box())], samples=1, fog=0, outline=0)
        smooth = self.render([Object3D(make_box())], samples=16, fog=0, outline=0)
        self.assertAlmostEqual(smooth.alpha.sum(), plain.alpha.sum(), delta=0.05 * plain.alpha.sum())

    def test_edges_get_extra_samples(self):
        # A slanted edge: with 4 samples, edge pixels have coverage in quarters; the extra
        # samples there give finer steps, while fully covered pixels stay solid.
        box = Object3D(make_box(), rotation=quat_axis_angle((0, 0, 1), 0.3))
        base = self.render([box], samples=4, edge_samples=0, fog=0, outline=0)
        fine = self.render([box], samples=4, edge_samples=8, fog=0, outline=0)
        levels = lambda fb: set(np.round(fb.alpha[(fb.alpha > 0) & (fb.alpha < 1)] * 12).astype(int))
        self.assertTrue(levels(base) <= {3, 6, 9})
        self.assertGreater(len(levels(fine) - {3, 6, 9}), 3)
        self.assertEqual(fine.alpha[30, 40], 1.0)

    def test_edge_pixels_blend_by_coverage(self):
        # A pixel half covered by a flat face carries half its light (linear, premultiplied by coverage).
        box = Object3D(make_box(), rotation=quat_axis_angle((0, 0, 1), 0.3))
        fb = self.render([box], samples=16, edge_samples=0, fog=0, outline=0)
        partial = (fb.alpha > 0.3) & (fb.alpha < 0.7)
        self.assertTrue(partial.any())
        face = np.median(fb.rgb[fb.alpha == 1], axis=0)
        np.testing.assert_allclose(fb.rgb[partial], fb.alpha[partial, None] * face, rtol=0.15)

    def test_rgb_object_colour(self):
        fb = self.render([Object3D(make_box(), color=(255, 0, 0))], fog=0, outline=0)
        r, g, b = fb.colour()[30, 40]
        self.assertGreater(r, 0.2)
        self.assertLess(max(g, b), 0.2 * r + 0.05)  # the white highlight may add a little of each

    def test_smooth_shading_on_shared_vertices(self):
        # An octahedron shades smoothly: neighbouring pixels differ gently, not in flat steps.
        v = np.array([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)], float)
        f = np.array([(4, 0, 2), (4, 2, 1), (4, 1, 3), (4, 3, 0), (5, 2, 0), (5, 1, 2), (5, 3, 1), (5, 0, 3)])
        mesh = Mesh(v, f)
        np.testing.assert_allclose(np.linalg.norm(mesh.vertex_normals(), axis=1), 1.0)
        fb = self.render([Object3D(mesh)], samples=1, fog=0, outline=0)
        row = lum(fb)[30][fb.drawn[30]]
        self.assertGreater(len(np.unique(np.round(row, 3))), 10)

    def test_camera_inside_scene_clips_instead_of_dropping(self):
        # A floor stretching from behind the camera to far in front: its near part must still draw.
        floor = Mesh(np.array([(-5, 0, 5), (5, 0, 5), (5, 0, -50), (-5, 0, -50)], float), np.array([(0, 1, 2), (0, 2, 3)]))
        camera = Camera(position=np.array([0.0, 1.0, 0.0]), target=np.array([0.0, 0.0, -3.0]))
        fb = self.render([Object3D(floor)], camera=camera)
        self.assertTrue(fb.drawn[-1].all())  # the bottom row is floor right up to the camera

    def test_outline_darkens_far_side_of_overlap(self):
        near = Object3D(make_box(), position=np.array([0.0, 0.0, 1.5]))
        far = Object3D(make_box(4.0), position=np.array([0.0, 0.0, -3.0]))
        plain = self.render([near, far], fog=0, outline=0)
        lined = self.render([near, far], fog=0, outline=0.6)
        darker = luminance(lined.rgb) < luminance(plain.rgb) - 1e-6
        self.assertTrue(darker.any())
        self.assertEqual(set(lined.ids[darker].tolist()), {2})
        self.assertFalse(darker[30, 35:46].any())  # the near box is untouched

    def test_fog_dims_distant_surfaces(self):
        near = Object3D(make_box(), position=np.array([-1.0, 0.0, 0.0]))
        far = Object3D(make_box(), position=np.array([1.5, 0.0, -6.0]))
        clear = self.render([near, far], fog=0, outline=0)
        foggy = self.render([near, far], fog=0.5, outline=0)
        far_px = (clear.ids == 2) & (clear.alpha == 1)
        self.assertTrue(far_px.any())
        self.assertLess(luminance(foggy.rgb)[far_px].mean(), 0.75 * luminance(clear.rgb)[far_px].mean())

    def test_textures_are_mipmapped(self):
        # A fine checkerboard seen small averages to grey instead of aliasing into random black and white.
        checks = (np.indices((64, 64)).sum(axis=0) % 2).astype(float)
        levels = build_mipmaps(checks)
        self.assertEqual([lv.shape[0] for lv in levels], [64, 32, 16, 8, 4, 2, 1])
        np.testing.assert_allclose(levels[-1], srgb_to_linear(np.array(1.0)) / 2)
        box = make_box(1.0, [checks] * 6)
        fb = self.render([Object3D(box, color=Color.WHITE)], w=16, h=6, samples=1, fog=0, outline=0)
        face = lum(fb)[fb.alpha == 1]
        self.assertLess(face.std(), 0.1 * face.mean())

    def test_load_obj_triangulates_quads(self):
        src = "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nvt 0 0\nf 1/1 2/1 3/1 4/1\nf -4 -3 -2\n"
        with tempfile.NamedTemporaryFile("w", suffix=".obj", delete=False) as f:
            f.write(src)
        try:
            mesh = load_obj(f.name)
        finally:
            os.unlink(f.name)
        self.assertEqual(mesh.vertices.shape, (4, 3))
        np.testing.assert_array_equal(mesh.faces, [[0, 1, 2], [0, 2, 3], [0, 1, 2]])


class ColorTests(unittest.TestCase):
    def test_srgb_round_trip(self):
        c = np.linspace(0, 1, 11)
        np.testing.assert_allclose(linear_to_srgb(srgb_to_linear(c)), c, atol=1e-9)

    def test_truecolor_is_exact(self):
        packed = quantize(srgb_to_linear(np.array([[10, 200, 30]]) / 255), "truecolor")
        self.assertEqual(sgr_color(packed[0]), "38;2;10;200;30")
        self.assertEqual(sgr_color(packed[0], background=True), "48;2;10;200;30")

    def test_palettes_keep_the_hue(self):
        rgb = xterm_rgb()
        for mode in ("256", "16"):
            for hue in (Color.RED, Color.GREEN, Color.BLUE):
                packed = int(quantize(to_linear_rgb(hue) * 0.5, mode))
                r, g, b = rgb[packed & 255]
                channel = {Color.RED: r, Color.GREEN: g, Color.BLUE: b}[hue]
                self.assertEqual(channel, max(r, g, b), (mode, hue.name))
                self.assertGreater(channel, min(r, g, b))

    def test_dither_mixes_neighbouring_colours(self):
        # A flat colour between two palette entries comes out as a pattern of both.
        c = np.full((8, 8, 3), srgb_to_linear(115 / 255))
        ys, xs = np.mgrid[0:8, 0:8]
        self.assertGreater(len(np.unique(quantize(c, "256", ys, xs))), 1)
        self.assertEqual(len(np.unique(quantize(c, "256", ys, xs, dither=False))), 1)


def fb_from(alpha, rgb=(1.0, 1.0, 1.0), cell_pixels=(2, 2)):
    alpha = np.asarray(alpha, float)
    fb = FrameBuffer(alpha.shape[1], alpha.shape[0], cell_pixels)
    fb.alpha[:] = alpha
    fb.rgb[:] = alpha[..., None] * np.asarray(rgb, float)
    return fb


class GlyphTests(unittest.TestCase):
    def test_glyph_tables(self):
        for g in GLYPH_SETS.values():
            if g.chars:
                self.assertEqual(len(g.chars), 2 ** g.pixel_count, g.name)
                self.assertEqual(len(set(g.chars)), len(g.chars), g.name)
                self.assertEqual((g.chars[0], g.chars[-1]), (" ", "█"))
        sextant = GLYPH_SETS["sextant"].chars
        self.assertEqual(sextant[1], "\U0001FB00")  # top-left sixth
        self.assertEqual(sextant[62], "\U0001FB3B")  # all but the top-left
        self.assertEqual(sextant[21], "▌")

    def test_quad_picks_the_covered_quarters(self):
        cells = match_cells(fb_from([[1, 0], [1, 1]]), GLYPH_SETS["quad"])
        self.assertEqual(cells.chars[0, 0], "▙")
        self.assertTrue(cells.fg_on[0, 0])
        self.assertFalse(cells.bg_on[0, 0])  # the empty quarter shows the terminal background

    def test_background_side_becomes_foreground(self):
        # Only the bottom-right is drawn: that has to be the foreground of ▗, not the background of ▛.
        cells = match_cells(fb_from([[0, 0], [0, 1]]), GLYPH_SETS["quad"])
        self.assertEqual(cells.chars[0, 0], "▗")
        self.assertTrue(cells.fg_on[0, 0] and not cells.bg_on[0, 0])

    def test_two_colours_split_along_their_edge(self):
        fb = FrameBuffer(2, 3, (2, 3))
        fb.alpha[:] = 1.0
        fb.rgb[:] = (0.9, 0.1, 0.1)
        fb.rgb[:, 1] = (0.1, 0.1, 0.9)  # right column blue
        cells = match_cells(fb, GLYPH_SETS["sextant"])
        self.assertIn(cells.chars[0, 0], "▌▐")
        self.assertTrue(cells.fg_on[0, 0] and cells.bg_on[0, 0])
        colours = {tuple(np.round(cells.fg[0, 0], 2)), tuple(np.round(cells.bg[0, 0], 2))}
        self.assertEqual(colours, {(0.9, 0.1, 0.1), (0.1, 0.1, 0.9)})

    def test_empty_and_faint_cells_stay_blank(self):
        cells = match_cells(fb_from([[0.1, 0], [0, 0.1]]), GLYPH_SETS["quad"])
        self.assertEqual(cells.chars[0, 0], " ")
        self.assertFalse(cells.fg_on[0, 0])

    def test_nearly_flat_cells_stay_solid_beside_split_ones(self):
        # One frame, three cells: flat, nearly flat (below MIN_SPLIT), and a real edge.
        fb = FrameBuffer(6, 2, (2, 2))
        fb.alpha[:] = 1.0
        fb.rgb[:] = 0.5
        fb.rgb[0, 2] = 0.51
        fb.rgb[:, 5] = 0.0
        cells = match_cells(fb, GLYPH_SETS["quad"])
        self.assertEqual(cells.chars[0].tolist(), ["█", "█", "▐"])
        np.testing.assert_allclose(cells.fg[0, 1], 0.5025)  # the mean of the nearly flat cell
        np.testing.assert_allclose(cells.fg[0, 2], 0.0)
        np.testing.assert_allclose(cells.bg[0, 2], 0.5)

    def test_half_blocks(self):
        cells = match_cells(fb_from([[1, 0], [0, 1]], cell_pixels=(1, 2)), GLYPH_SETS["half"])
        self.assertEqual(cells.chars[0].tolist(), ["▀", "▄"])


class InputTests(unittest.TestCase):
    def test_keys_and_sequences(self):
        d = InputDecoder()
        self.assertEqual(d.feed("a\r\x1b[A\x1b[B\x1bOC\x1b[3~\x1b[1;5D\x7f", 0.0),
                         [ord("a"), 13, Key.UP, Key.DOWN, Key.RIGHT, Key.DELETE, Key.LEFT, 127])

    def test_split_sequence_waits_for_the_rest(self):
        d = InputDecoder()
        self.assertEqual(d.feed("\x1b[", 0.0), [])
        self.assertEqual(d.flush(0.01), [])
        self.assertEqual(d.feed("A", 0.02), [Key.UP])

    def test_lone_escape_after_timeout(self):
        d = InputDecoder()
        self.assertEqual(d.feed("\x1b", 0.0), [])
        self.assertEqual(d.flush(0.1), [Key.ESC])
        self.assertEqual(d.feed("\x1bq", 0.2), [Key.ESC, ord("q")])

    def test_sgr_mouse(self):
        d = InputDecoder()
        events = d.feed("\x1b[<0;10;5M\x1b[<0;10;5m\x1b[<65;1;1M\x1b[<32;3;3M", 0.0)
        self.assertEqual(events, [MouseEvent(9, 4, 0, True), MouseEvent(9, 4, 0, False), MouseEvent(0, 0, 65, True)])

    def test_windows_records_translate_to_vt(self):
        from types import SimpleNamespace as NS
        con = WindowsInput()
        key = lambda down, ch, vk=0: NS(bKeyDown=down, wRepeatCount=1, uChar=ch, wVirtualKeyCode=vk)
        self.assertEqual(con.key(key(True, ord("x"))), "x")
        self.assertEqual(con.key(key(True, 0, 0x26)), "\x1b[A")
        self.assertEqual(con.key(key(False, ord("x"))), "")
        mouse = lambda x, y, state, flags=0: NS(dwMousePosition=NS(X=x, Y=y), dwButtonState=state, dwEventFlags=flags)
        text = con.mouse(mouse(4, 2, 1)) + con.mouse(mouse(4, 2, 0))
        self.assertEqual(InputDecoder().feed(text, 0.0), [MouseEvent(4, 2, 0, True), MouseEvent(4, 2, 0, False)])


class ScreenTests(unittest.TestCase):
    def test_detection(self):
        self.assertEqual(detect_color_mode({"WT_SESSION": "x", "TERM": "xterm-256color"}, windows=False), "truecolor")
        self.assertEqual(detect_color_mode({}, windows=True), "truecolor")
        self.assertEqual(detect_color_mode({"TERM": "xterm-256color"}, windows=False), "256")
        self.assertEqual(detect_color_mode({"TERM": "xterm-256color", "COLORTERM": "truecolor"}, windows=False), "truecolor")
        self.assertEqual(detect_color_mode({"TERM": "xterm", "UNICODE3D_COLOR": "16"}, windows=False), "16")
        self.assertEqual(detect_glyphs({}, unicode_ok=True), "quad")
        self.assertEqual(detect_glyphs({}, unicode_ok=False), "ascii")
        self.assertEqual(detect_glyphs({"UNICODE3D_GLYPHS": "sextant"}), "sextant")
        self.assertEqual(detect_glyphs({"TERM": "xterm-256color"}), "quad")
        self.assertEqual(detect_glyphs({"TERM": "linux"}), "half")
        for env in ({"TERM": "xterm-kitty"}, {"TERM": "foot"}, {"TERM": "xterm-ghostty"},
                    {"TERM": "xterm-256color", "TERM_PROGRAM": "WezTerm"},
                    {"TERM": "tmux-256color", "KITTY_WINDOW_ID": "1"},  # tmux inside kitty
                    {"TERM": "xterm-256color", "WT_SESSION": "x"}):     # Windows Terminal, or WSL inside it
            self.assertEqual(detect_glyphs(env), "sextant", env)
        self.assertEqual(detect_glyphs({"TERM": "xterm-kitty", "UNICODE3D_GLYPHS": "quad"}), "quad")
        self.assertEqual(detect_glyphs({"TERM": "xterm-kitty"}, unicode_ok=False), "ascii")

    def test_refresh_sends_only_changes(self):
        screen = Screen(glyphs="quad", color="truecolor", size=(5, 20))
        screen.text(1, 2, "hello", Color.GREEN, bold=True)
        first = screen.render_updates()
        self.assertIn("\x1b[2J", first)  # the first refresh redraws everything
        self.assertIn("\x1b[0;1;32;49mhello", first)
        self.assertEqual(screen.render_updates(), "")  # nothing changed
        screen.text(1, 3, "a")
        second = screen.render_updates()
        self.assertIn("\x1b[2;4H", second)
        self.assertNotIn("\x1b[2J", second)
        self.assertNotIn("hello", second)

    def test_frame_needs_matching_cell_pixels(self):
        screen = Screen(glyphs="sextant", color="truecolor", size=(10, 20))
        renderer = Renderer(10, 5)
        fb = renderer.render([Object3D(make_box())], Camera(), Light())
        with self.assertRaises(ValueError):
            screen.draw_frame(fb)
        renderer.resize(10, 5, screen.cell_pixels)
        screen.draw_frame(renderer.render([Object3D(make_box())], Camera(), Light()), 2, 3)
        drawn = screen.chars[2:7, 3:13] != " "
        self.assertTrue(drawn.any())
        self.assertTrue((screen.fg[2:7, 3:13][drawn] > 0).all())  # truecolor, not the default colour
        out = screen.render_updates()
        self.assertIn("38;2;", out)

    def test_display_settings_change_at_run_time(self):
        screen = Screen(glyphs="quad", color="truecolor", size=(4, 10), background=(10, 20, 30))
        screen.text(0, 0, "hi", Color.GREEN)
        screen.render_updates()
        screen.set_glyphs("sextant")
        self.assertEqual(screen.cell_pixels, (2, 3))
        screen.set_color("256")
        screen.erase()
        screen.text(0, 0, "hi", Color.GREEN)
        update = screen.render_updates()
        self.assertIn("\x1b[2J", update)  # a new colour mode resends the whole screen
        self.assertIn(";48;5;", update)    # with the background in the palette
        self.assertNotIn(";48;2;", update)
        with self.assertRaises(ValueError):
            screen.set_color("8")
        screen.unicode = False
        self.assertEqual(screen.glyph_modes, ("ascii",))
        with self.assertRaises(ValueError):
            screen.set_glyphs("quad")

    def test_wide_characters_are_replaced(self):
        screen = Screen(color="mono", size=(2, 10))
        screen.text(0, 0, "a\u4e2db\u0301")
        self.assertEqual("".join(screen.chars[0, :4]), "a?b?")


if __name__ == "__main__":
    unittest.main()
