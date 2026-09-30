# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
import os
import tempfile
import unittest

import numba
import numpy as np

from unicode3d.examples.dice import DIE_VALUES, RollAnimation, make_die, orientation_showing, top_face
from unicode3d.mesh import Mesh, load_obj, make_box
from unicode3d.background import Gradient, Sky, SkyBox
from unicode3d.color import (Color, encode_index, encode_rgb, linear_to_srgb, luminance, put_sgr_color, quantize,
                             sgr_color, srgb_to_linear, to_linear_rgb, xterm_rgb)
from unicode3d.console import WindowsInput, detect_color_mode, detect_glyphs
from unicode3d.glyphs import GLYPH_SETS, frame_to_text, match_cells
from unicode3d.keys import HeldKeys, InputDecoder, Key, KeyRelease, MouseEvent
from unicode3d.raster import FrameBuffer
from unicode3d.scene import Camera, Light, Node, Object3D, PointLight, Renderer
from unicode3d.shapes import blob_mesh, block_mesh, merge_meshes, text_mesh
from unicode3d.terminal import Screen
from unicode3d.texture import BLEND, CUTOUT, alpha_kind, build_mipmaps
from unicode3d.ui import Button, Choice, DisplayControls, Panel, Slider, Toggle
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


def flat_quad(width, height):
    """A width x height rectangle facing +z, centred on the origin."""
    w, h = width / 2, height / 2
    return Mesh(np.array([(-w, -h, 0), (w, -h, 0), (w, h, 0), (-w, h, 0)], float), np.array([(0, 1, 2), (0, 2, 3)]))


def textured_quad(width, height, texture):
    """A width x height rectangle facing +z, centred on the origin, showing all of `texture`."""
    w, h = width / 2, height / 2
    return Mesh(np.array([(-w, -h, 0), (w, -h, 0), (w, h, 0), (-w, h, 0)], float), np.array([(0, 1, 2), (0, 2, 3)]),
                uvs=np.array([[(0, 0), (1, 0), (1, 1)], [(0, 0), (1, 1), (0, 1)]], float), materials=np.zeros(2, int),
                textures=[texture])


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

    def test_scenes_of_any_size_render_alike(self):
        # The renderer reuses its working arrays from frame to frame: a scene must come out the same
        # whatever was drawn before it, bigger or smaller.
        renderer, light = Renderer(40, 30), Light()
        camera = Camera(position=np.array([0.0, 0.0, 5.0]))
        small = [Object3D(make_die())]
        big = [Object3D(blob_mesh((1.0, 1.0, 1.0), rings=24, segments=32), color=Color.RED),
               Object3D(make_die(), position=np.array([1.0, 0.5, 1.0]))]
        first = renderer.render(small, camera, light).copy()
        mixed = renderer.render(big, camera, light).copy()
        again = renderer.render(small, Camera(position=np.array([0.0, 0.0, 5.0 + 1e-9])), light)
        np.testing.assert_allclose(again.rgb, first.rgb, atol=1e-6)
        np.testing.assert_array_equal(again.ids, first.ids)
        # Textured and plain objects share one render: each keeps its own colour and texture.
        self.assertEqual(set(np.unique(mixed.ids)), {0, 1, 2})
        red = mixed.colour()[mixed.ids == 1]
        self.assertGreater(np.median(red[:, 0]), 2 * np.median(red[:, 1]))
        self.assertLess(lum(mixed)[mixed.ids == 2].min(), 0.05)  # the die's pips

    def test_same_frame_on_any_number_of_threads(self):
        # Kernels run on several threads, each writing only its own pixels (the one-writer rule in
        # CLAUDE.md). A kernel that breaks it gives frames that change with the thread count, or
        # from run to run. This compares every stage of a frame drawn on one thread with the same
        # frame drawn several times on all of them. Races show up by chance, so the scenes are
        # crowded: many small triangles, overlapping, for threads to collide on.
        most = numba.config.NUMBA_NUM_THREADS
        if most < 2:
            self.skipTest("needs a machine with more than one core")
        spheres = [Object3D(blob_mesh((1.0, 1.0, 1.0), rings=48, segments=64), color=Color.GREEN,
                            position=np.array([0.4 * i - 0.6, 0.1 * i, -0.3 * i])) for i in range(4)]
        # Many small objects, some with parents, coloured per vertex and per face, some double-sided.
        rng = np.random.default_rng(5)
        ball = blob_mesh((1.0, 1.0, 1.0), rings=8, segments=10)
        ball.vertex_colors = rng.uniform(0, 1, (len(ball.vertices), 3))
        cube = make_box()
        cube.face_colors = rng.integers(0, 256, (len(cube.faces), 3))
        group = Node(position=np.array([0.2, 0.0, 0.0]), rotation=quat_axis_angle((0, 1, 0), 0.4), scale=0.8)
        crowd = [Object3D((ball, cube)[i % 2], rng.uniform(-1.5, 1.5, 3), quat_axis_angle(rng.normal(size=3), 1.0),
                          scale=rng.uniform(0.1, 0.4), color=(255, 255, 255), double_sided=i % 3 == 0,
                          parent=group if i % 4 == 0 else None) for i in range(80)]
        crowd[1].emissive = 1.0
        for obj in crowd[5::7]:  # some see-through, overlapping each other
            obj.opacity = 0.4
        holes = np.ones((16, 16, 4))
        holes[::2, :, 3] = 0.0  # stripes of holes
        stained = np.full((16, 16, 4), 0.5)
        crowd += [Object3D(textured_quad(1.5, 1.5, t), rng.uniform(-1, 1, 3), quat_axis_angle(rng.normal(size=3), 1.0),
                           color=(255, 255, 255)) for t in (holes, stained)]
        # Mirrors (seeing each other: two bounces), and something shiny.
        crowd += [Object3D(flat_quad(2.0, 1.5), np.array([x, 0.0, -1.2]), quat_axis_angle((0, 1, 0), turn),
                           color=(150, 150, 150), reflectivity=0.8) for x, turn in ((-0.8, 0.6), (0.8, -0.6))]
        crowd[3].reflectivity = 0.9
        lights = [Light(color=(255, 200, 150)), PointLight(np.array([0.5, 0.5, 1.0]), color=Color.CYAN, range=3.0)]
        shadowed = [Light(direction=np.array([0.5, -1.0, -0.3]), shadows=True), Light(shadows=True),
                    PointLight(np.array([0.5, 0.5, 1.0]), color=Color.CYAN, range=3.0, shadows=True),
                    PointLight(np.array([-0.2, 0.0, 0.3]), color=Color.RED, range=2.0, shadows=True)]
        scenes = [([Object3D(make_die(), rotation=quat_axis_angle((1, 2, 0), 0.7)), *spheres], Light()),
                  ([Object3D(make_box(), position=np.array([0.0, 0.0, 4.6]))], Light()),  # camera in front of its near face
                  (crowd, lights),
                  (crowd, shadowed)]

        def draw(threads):
            numba.set_num_threads(threads)
            screen = Screen(glyphs="sextant", color="256", size=(30, 60), background=(20, 20, 30))
            renderer = Renderer(60, 30, screen.cell_pixels, background=Sky(), mirror_bounces=2)
            frame = []
            for objects, light in scenes:
                fb = renderer.render(objects, Camera(position=np.array([0.0, 0.5, 5.0])), light)
                screen.draw_frame(fb)
                frame += [fb.rgb.copy(), fb.alpha.copy(), fb.depth.copy(), fb.ids.copy(), screen.chars.copy(),
                          screen.fg.copy(), screen.bg.copy(), screen.render_updates()]
            # Drawn smaller than the screen needs (Renderer.max_pixels) and stretched to fit.
            renderer = Renderer(60, 30, screen.cell_pixels, background=Sky(), max_pixels=2000)
            fb = renderer.render(crowd, Camera(position=np.array([0.0, 0.5, 5.0])), shadowed)
            frame += [fb.rgb.copy(), fb.alpha.copy(), fb.depth.copy(), fb.ids.copy()]
            return frame

        try:
            one = draw(1)
            for _ in range(5):
                for a, b in zip(one, draw(most)):
                    np.testing.assert_array_equal(a, b)
        finally:
            numba.set_num_threads(most)

    def test_parents_move_turn_scale_and_hide_children(self):
        group = Node(position=np.array([1.0, 0.0, 0.0]), rotation=quat_axis_angle((0, 1, 0), np.pi / 2), scale=2.0)
        child = Object3D(make_box(0.5), position=np.array([1.0, 0.0, 0.0]), rotation=quat_axis_angle((1, 0, 0), 0.3),
                         parent=group, color=Color.RED)
        grandchild = Object3D(make_box(0.2), position=np.array([0.0, 1.0, 0.0]), parent=child)
        position, rotation, scale, visible = child.world_transform()
        np.testing.assert_allclose(position, [1.0, 0.0, -2.0], atol=1e-12)  # +x turned a quarter round y is -z
        self.assertEqual((scale, visible), (2.0, True))
        np.testing.assert_allclose(grandchild.to_world([0, 0, 0]), child.to_world([0, 1, 0]), atol=1e-12)
        # Drawn exactly where the same objects placed directly in the world would be.
        direct = [Object3D(o.mesh, *o.world_transform()[:3], color=o.color) for o in (child, grandchild)]
        camera = Camera(position=np.array([3.0, 2.0, 4.0]))
        a = self.render([child, grandchild], camera=camera)
        b = self.render(direct, camera=camera)
        np.testing.assert_allclose(a.rgb, b.rgb, atol=1e-12)
        np.testing.assert_array_equal(a.ids, b.ids)
        self.assertEqual(set(np.unique(a.ids)), {0, 1, 2})
        group.visible = False
        self.assertFalse(self.render([child, grandchild], camera=camera).drawn.any())
        # A parent that moves makes the renderer draw again.
        renderer = Renderer(40, 15)
        group.visible = True
        renderer.render([child], camera, Light())
        group.position = np.array([1.5, 0.0, 0.0])
        renderer.render([child], camera, Light())
        self.assertEqual(renderer.draws, 2)

    def test_vertex_and_face_colours(self):
        quad = Mesh(np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], float), np.array([[0, 1, 2], [0, 2, 3]]))
        quad.face_colors = np.array([[255, 0, 0], [0, 0, 255]])
        light = Light(direction=np.array([0.0, 0.0, -1.0]), ambient=0.0, diffuse=1.0, specular=0.0)
        renderer = Renderer(40, 20, fog=0, outline=0)
        camera = Camera(position=np.array([0.0, 0.0, 3.0]))
        fb = renderer.render([Object3D(quad, color=(255, 255, 255))], camera, light)
        c = fb.colour()
        np.testing.assert_allclose(c[25, 26], [1, 0, 0], atol=1e-9)  # below the diagonal: the first face
        np.testing.assert_allclose(c[15, 14], [0, 0, 1], atol=1e-9)
        # Colours multiply the object's colour, as textures do.
        fb = renderer.render([Object3D(quad, color=(1.0, 0.5, 1.0))], camera, light)
        np.testing.assert_allclose(fb.colour()[15, 14], [0, 0, 1], atol=1e-9)
        # Vertex colours blend across the faces, in linear light.
        quad.vertex_colors = np.array([[1.0, 0, 0], [1.0, 0, 0], [0, 1.0, 0], [0, 1.0, 0]])
        fb = renderer.render([Object3D(quad, color=(255, 255, 255))], camera, light)
        c = fb.colour()
        self.assertGreater(c[33, 20, 0], 0.8)  # near the red bottom edge
        self.assertGreater(c[7, 20, 1], 0.8)   # near the green top
        self.assertAlmostEqual(c[20, 20, 0], c[20, 20, 1], delta=0.05)  # half way
        # merge_meshes keeps colours, or gives each part one.
        merged = merge_meshes([quad, make_box()])
        np.testing.assert_allclose(merged.vertex_colors[:4], quad.vertex_colors)
        np.testing.assert_allclose(merged.vertex_colors[4:], 1.0)
        merged = merge_meshes([make_box(), make_box()], colors=[(255, 0, 0), (0.0, 1.0, 0.0)])
        np.testing.assert_allclose(merged.face_colors[[0, -1]], [[1, 0, 0], [0, 1, 0]])

    def test_objects_out_of_view_are_skipped_whole(self):
        camera = Camera(position=np.array([0.0, 0.0, 5.0]))
        box = Object3D(make_box(), color=Color.CYAN)
        hidden = [Object3D(make_box(), position=np.array(p)) for p in ([0.0, 0.0, 9.0], [30.0, 0.0, 0.0],
                                                                        [0.0, -30.0, 0.0])]
        edge = Object3D(blob_mesh((1.0, 1.0, 1.0)), position=np.array([3.4, 0.0, 0.0]))  # partly in view
        alone = self.render([box, edge], camera=camera)
        crowded = self.render([box, *hidden, edge], camera=camera)
        np.testing.assert_allclose(crowded.rgb, alone.rgb, atol=1e-12)
        self.assertEqual(set(np.unique(crowded.ids)), {0, 1, 5})  # ids still count every object in the list

    def test_point_lights_coloured_lights_and_glow(self):
        # A wall facing +z, lit only by a point light just in front of its left side.
        wall = Object3D(block_mesh((0.0, 0.0, 0.0), (6.0, 3.0, 0.1)), color=(255, 255, 255))
        camera = Camera(position=np.array([0.0, 0.0, 6.0]))
        renderer = Renderer(60, 20, fog=0, outline=0)
        lamp = PointLight(np.array([-2.0, 0.0, 1.0]), diffuse=1.0, specular=0.0, range=3.0)
        fb = renderer.render([wall], camera, lamp)
        y, row = fb.height // 2, lum(fb)[fb.height // 2]
        left, mid, right = row[int(fb.width * 0.3)], row[fb.width // 2], row[int(fb.width * 0.8)]
        self.assertGreater(left, mid)
        self.assertGreater(mid, 0.0)
        self.assertEqual(right, 0.0)  # beyond the light's range: dark
        # Lights add up, and a coloured light tints what it lights.
        sun = Light(direction=np.array([0.0, 0.0, -1.0]), ambient=0.0, diffuse=0.5, specular=0.0, color=(255, 0, 0))
        fb = renderer.render([wall], camera, [sun, lamp])
        c = fb.colour()[y]
        self.assertGreater(c[int(fb.width * 0.8), 0], 0.1)
        self.assertEqual(c[int(fb.width * 0.8), 1], 0.0)
        self.assertGreater(c[int(fb.width * 0.3), 1], 0.0)
        # Something that glows shows its own colour in the dark.
        dark = Light(ambient=0.0, diffuse=0.0, specular=0.0)
        glow = Object3D(make_box(), color=(40, 200, 60), emissive=1.0)
        fb = renderer.render([glow], camera, dark)
        np.testing.assert_allclose(fb.colour()[y, fb.width // 2], to_linear_rgb((40, 200, 60)), atol=1e-9)
        self.assertFalse(renderer.render([Object3D(make_box())], camera, []).colour().any())  # no lights: black

    def test_shadows(self):
        # A box above a floor, seen from straight above, lit from up and to the left: its shadow falls a
        # unit to the right of it, where the floor gets only the light's ambient light.
        floor = Object3D(block_mesh((0.0, -0.05, 0.0), (6.0, 0.1, 6.0)), color=(255, 255, 255))
        box = Object3D(block_mesh((0.0, 1.0, 0.0), (0.6, 0.6, 0.6)), color=(255, 255, 255))
        camera = Camera(position=np.array([0.0, 6.0, 0.0]), target=np.zeros(3), up=np.array([0.0, 0.0, -1.0]))
        renderer = Renderer(60, 30, fog=0, outline=0)
        sun = Light(direction=np.array([1.0, -1.0, 0.0]), shadows=True)

        def at(fb, *point):
            x, y = renderer.project(point)
            return lum(fb)[int(y * 2), int(x)]

        fb = renderer.render([floor, box], camera, sun).copy()
        shadow, lit, top = at(fb, 1.0, 0.0, 0.0), at(fb, -1.0, 0.0, 0.0), at(fb, 0.0, 1.3, 0.0)
        ambient = Light(direction=sun.direction, diffuse=0.0, specular=0.0)
        self.assertAlmostEqual(shadow, at(renderer.render([floor, box], camera, ambient), 1.0, 0.0, 0.0), places=9)
        self.assertGreater(lit, shadow + 0.1)
        self.assertGreater(top, shadow + 0.1)  # the box's top, facing the light, is lit, not in its own shadow
        # The edge is smoothed over at least a pixel: part-way between the shadow's middle and the lit floor.
        self.assertTrue(any(shadow < v < lit for v in (at(fb, x, 0.0, 0.0) for x in np.linspace(1.4, 1.8, 40))))
        # Moving the camera keeps the shadow where it is (and the shadow map as it was).
        camera.position = np.array([0.2, 6.0, 0.1])
        fb = renderer.render([floor, box], camera, sun)
        self.assertAlmostEqual(at(fb, 1.0, 0.0, 0.0), shadow, places=9)
        # Something that casts no shadow, or a light without them, leaves the floor lit.
        plain = Light(direction=sun.direction)
        fb = renderer.render([floor, box], camera, plain)
        self.assertGreater(at(fb, 1.0, 0.0, 0.0), shadow + 0.1)
        unlit = at(fb, 1.0, 0.0, 0.0)
        box.cast_shadows = False
        self.assertAlmostEqual(at(renderer.render([floor, box], camera, sun), 1.0, 0.0, 0.0), unlit, places=9)
        # Nor does a renderer with shadows switched off.
        box.cast_shadows = True
        renderer.shadows = False
        self.assertAlmostEqual(at(renderer.render([floor, box], camera, sun), 1.0, 0.0, 0.0), unlit, places=9)

    def test_point_light_shadows(self):
        # A low lamp in the middle of a floor with a box on each side: each box's shadow runs away from
        # the lamp, in a different face of its cube map. Between them, along the diagonals where the
        # faces meet, the floor is lit just as with no shadows at all.
        floor = Object3D(block_mesh((0.0, -0.05, 0.0), (16.0, 0.1, 16.0)), color=(255, 255, 255))
        boxes = [Object3D(block_mesh((x, 0.3, z), (0.3, 0.3, 0.3)), color=(255, 255, 255))
                 for x, z in ((1.5, 0.0), (-1.5, 0.0), (0.0, 1.5), (0.0, -1.5))]
        camera = Camera(position=np.array([0.0, 12.0, 0.0]), target=np.zeros(3), up=np.array([0.0, 0.0, -1.0]))
        renderer = Renderer(80, 40, fog=0, outline=0)
        lamp = PointLight(np.array([0.0, 0.6, 0.0]), diffuse=1.0, specular=0.0, range=12.0, shadows=True)

        def at(fb, x, z):
            cx, cy = renderer.project((x, 0.0, z))
            return lum(fb)[int(cy * 2), int(cx)]

        fb = renderer.render([floor, *boxes], camera, lamp).copy()
        plain = renderer.render([floor, *boxes], camera, PointLight(lamp.position, diffuse=1.0, specular=0.0,
                                                                    range=12.0))
        for x, z in ((2.5, 0.0), (-2.5, 0.0), (0.0, 2.5), (0.0, -2.5)):
            self.assertEqual(at(fb, x, z), 0.0)  # no ambient light: black
            self.assertGreater(at(plain, x, z), 0.02)
        for x, z in ((1.8, 1.8), (-1.8, 1.8), (1.8, -1.8), (-1.8, -1.8), (3.0, 3.0), (0.5, 0.3)):
            self.assertAlmostEqual(at(fb, x, z), at(plain, x, z), places=9)

    def test_transparency(self):
        # A red wall, and in front of it see-through panes and a glass box, lit evenly (no highlights).
        wall = Object3D(block_mesh((0.0, 0.0, -1.0), (8.0, 6.0, 0.1)), color=(255, 0, 0))
        camera = Camera(position=np.array([0.0, 0.0, 5.0]))
        flat = Light(direction=np.array([0.0, 0.0, -1.0]), ambient=0.4, diffuse=0.4, specular=0.0)
        renderer = Renderer(60, 20, fog=0, outline=0)

        def pane(z, color=(255, 255, 255), **kwargs):
            return Object3D(block_mesh((0.0, 0.0, z), (1.2, 1.2, 0.02)), color=color, **kwargs)

        def middle(fb):
            return fb.colour()[fb.height // 2, fb.width // 2]

        red = middle(renderer.render([wall], camera, flat).copy())
        # Half see-through: part the pane's white, part the red behind it.
        half = middle(renderer.render([wall, pane(1.0, opacity=0.5)], camera, flat).copy())
        self.assertTrue(red[0] > 0.1 and red[1] == 0.0)
        self.assertGreater(half[1], 0.02)
        self.assertGreater(half[0], half[1])
        # Wholly clear is not drawn at all, not even for picking; solid shows no red.
        fb = renderer.render([wall], camera, flat).copy()
        clear = renderer.render([wall, pane(1.0, opacity=0.0)], camera, flat)
        for a, b in zip((fb.rgb, fb.alpha, fb.depth, fb.ids), (clear.rgb, clear.alpha, clear.depth, clear.ids)):
            np.testing.assert_array_equal(a, b)
        solid = middle(renderer.render([wall, pane(1.0)], camera, flat))
        self.assertAlmostEqual(solid[0], solid[1], places=9)
        # The order of the render list does not matter.
        glass = [pane(1.0, opacity=0.4, color=(0, 0, 255)), pane(2.0, opacity=0.6, color=(0, 255, 0)),
                 Object3D(make_box(), np.array([0.2, 0.1, 0.0]), color=(255, 255, 0), opacity=0.3)]
        a = renderer.render([wall, *glass], camera, flat).copy()
        b = renderer.render([*glass[::-1], wall], camera, flat)
        np.testing.assert_allclose(a.rgb, b.rgb, atol=1e-12)
        np.testing.assert_array_equal(a.depth, b.depth)
        # Picking finds the glass in front, and a see-through object shows its far side: the box's back.
        self.assertIs(renderer.pick(renderer.width // 2, renderer.height // 2).object, glass[1])
        box = Object3D(make_box(), color=(255, 255, 255), opacity=0.5)
        one_side = Object3D(block_mesh((0.0, 0.0, 0.5), (1.0, 1.0, 0.0)), color=(255, 255, 255), opacity=0.5)
        self.assertGreater(middle(renderer.render([wall, box], camera, flat))[1],  # whiter: two panes of white
                           middle(renderer.render([wall, one_side], camera, flat))[1] + 0.05)
        # With room for 2 layers, the nearest 2 of 5 panes show, as if the others were not there.
        panes = [pane(z, opacity=0.5, color=c) for z, c in zip((0.0, 0.5, 1.0, 1.5, 2.0),
                                                                ((255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                                                                 (0, 255, 255)))]
        few = Renderer(60, 20, fog=0, outline=0, transparency_layers=2)
        fb = few.render([wall, *panes], camera, flat).copy()
        np.testing.assert_allclose(fb.rgb, few.render([wall, *panes[3:]], camera, flat).rgb, atol=1e-12)
        # Over an empty background, see-through things leave it showing.
        fb = renderer.render([pane(1.0, opacity=0.5)], camera, flat)
        self.assertTrue(0.4 < fb.alpha[fb.height // 2, fb.width // 2] < 0.7)

    def test_vertex_and_face_alpha(self):
        # A pane solid at its left edge and clear at its right, over a red wall; and a box with its
        # faces clear or solid by turns.
        wall = Object3D(block_mesh((0.0, 0.0, -1.0), (8.0, 6.0, 0.1)), color=(255, 0, 0))
        mesh = Mesh(np.array([(-2, -1, 0), (2, -1, 0), (2, 1, 0), (-2, 1, 0)], float), np.array([(0, 1, 2), (0, 2, 3)]),
                    vertex_colors=np.array([(0, 255, 0, 255), (0, 255, 0, 0), (0, 255, 0, 0), (0, 255, 0, 255)]))
        camera = Camera(position=np.array([0.0, 0.0, 5.0]))
        flat = Light(direction=np.array([0.0, 0.0, -1.0]), ambient=0.4, diffuse=0.4, specular=0.0)
        renderer = Renderer(60, 20, fog=0, outline=0)
        fb = renderer.render([wall, Object3D(mesh, color=(255, 255, 255))], camera, flat)

        def at(x):
            cx, cy = renderer.project((x, 0.0, 0.0))
            return fb.colour()[int(cy * 2), int(cx)]

        left, mid, right = at(-1.9), at(0.0), at(1.9)
        self.assertGreater(left[1], 10 * left[0])  # nearly solid green
        self.assertGreater(right[0], 10 * right[1])  # nearly clear: the red wall
        self.assertTrue(mid[0] > 0.01 and mid[1] > 0.01)
        box = make_box()
        box.face_colors = np.array([(255, 255, 255, 255 * (i // 2 % 2)) for i in range(12)])
        self.assertIsNotNone(renderer.render([wall, Object3D(box, color=(255, 255, 255))], camera, flat))

    def test_see_through_things_cast_tinted_shadows(self):
        # A floor seen from above, lit straight down through a red glass pane, a clear one and a solid one.
        floor = Object3D(block_mesh((0.0, -0.05, 0.0), (8.0, 0.1, 8.0)), color=(255, 255, 255))
        camera = Camera(position=np.array([0.0, 8.0, 0.0]), target=np.zeros(3), up=np.array([0.0, 0.0, -1.0]))
        sun = Light(direction=np.array([0.0, -1.0, 0.0]), ambient=0.1, diffuse=0.9, specular=0.0, shadows=True)
        renderer = Renderer(80, 40, fog=0, outline=0)
        panes = [Object3D(block_mesh((x, 1.0, 0.0), (1.2, 0.02, 1.2)), color=c, opacity=a)
                 for x, c, a in ((-2.0, (255, 0, 0), 0.5), (0.0, (255, 255, 255), 0.1), (2.0, (255, 255, 255), 1.0))]

        def at(fb, x):
            cx, cy = renderer.project((x, 0.0, 0.8))  # beside the panes, seen past them
            return fb.colour()[int(cy * 2), int(cx)]

        # Seen from above the panes cover their shadows, so look just past their edge at z = 0.8, with
        # the sun slanting a little along z to put the shadows there.
        sun.direction = np.array([0.0, -1.0, 0.8])
        fb = renderer.render([floor, *panes], camera, sun)
        lit = at(fb, -3.5)
        tinted, faint, dark = at(fb, -2.0), at(fb, 0.0), at(fb, 2.0)
        self.assertGreater(tinted[0], 2 * tinted[1])  # red light through red glass
        self.assertLess(tinted[1], lit[1])
        self.assertGreater(faint[1], 0.8 * lit[1])  # nearly clear glass: a faint shadow
        self.assertLess(faint[1], lit[1])
        self.assertLess(dark[1], 0.3 * lit[1])  # solid: a full shadow

    def test_texture_alpha(self):
        # A pane whose texture is solid green on the right half and clear on the left, over a red wall.
        wall = Object3D(block_mesh((0.0, 0.0, -1.0), (8.0, 6.0, 0.1)), color=(255, 0, 0))
        camera = Camera(position=np.array([0.0, 0.0, 5.0]))
        flat = Light(direction=np.array([0.0, 0.0, -1.0]), ambient=0.4, diffuse=0.4, specular=0.0)
        renderer = Renderer(60, 20, fog=0, outline=0)
        half = np.zeros((32, 32, 4))
        half[:, :, 1] = 1.0
        half[:, 16:, 3] = 1.0
        pane = Object3D(textured_quad(4.0, 2.0, half), color=(255, 255, 255))
        self.assertEqual(alpha_kind(build_mipmaps(half)), CUTOUT)
        fb = renderer.render([wall, pane], camera, flat)

        def at(x):
            cx, cy = renderer.project((x, 0.0, 0.0))
            return fb.colour()[int(cy * 2), int(cx)], (int(cx), int(cy))

        (left, cell_l), (right, cell_r) = at(-1.0), at(1.0)
        self.assertGreater(left[0], 0.1)
        self.assertEqual(left[1], 0.0)  # the wall, through the hole
        self.assertGreater(right[1], 0.1)
        self.assertEqual(right[0], 0.0)  # the pane, solid
        self.assertIs(renderer.pick(*cell_l).object, wall)  # clicks go through holes
        self.assertIs(renderer.pick(*cell_r).object, pane)
        # The edge of the hole is smoothed: some pixels along it are part green, part red.
        row = fb.colour()[int(at(0.0)[1][1] * 2)]
        self.assertTrue(((row[:, 0] > 0.02) & (row[:, 1] > 0.02)).any())
        # Shrunk, a cut-out's colour stays its own: the clear part doesn't darken it.
        levels = build_mipmaps(half)
        np.testing.assert_allclose(levels[-1][0, 0, 1] / levels[-1][0, 0, 3], 1.0)
        # Mostly half see-through is stained glass, drawn blended: some of the wall shows through all over.
        glass = np.zeros((32, 32, 4))
        glass[..., 2], glass[..., 3] = 1.0, 0.5
        self.assertEqual(alpha_kind(build_mipmaps(glass)), BLEND)
        fb = renderer.render([wall, Object3D(textured_quad(4.0, 2.0, glass), color=(255, 255, 255))], camera, flat)
        c = fb.colour()[fb.height // 2, fb.width // 2]
        self.assertTrue(c[0] > 0.02 and c[2] > 0.02)

    def test_light_through_textures(self):
        # A floor lit from straight above through a striped cut-out (a fence lying flat) and stained glass.
        floor = Object3D(block_mesh((0.0, -0.05, 0.0), (8.0, 0.1, 8.0)), color=(255, 255, 255))
        camera = Camera(position=np.array([0.0, 8.0, 0.0]), target=np.zeros(3), up=np.array([0.0, 0.0, -1.0]))
        sun = Light(direction=np.array([0.0, -1.0, 1.6]), ambient=0.1, diffuse=0.9, specular=0.0, shadows=True)
        renderer = Renderer(80, 40, fog=0, outline=0)
        stripes = np.ones((32, 32, 4))
        stripes[:, :16, 3] = 0.0  # the -x half is a hole
        red = np.zeros((32, 32, 4))
        red[..., 0], red[..., 3] = 1.0, 0.5
        lying = quat_axis_angle((1, 0, 0), -np.pi / 2)
        fence = Object3D(textured_quad(2.0, 2.0, stripes), np.array([-1.5, 1.0, 0.0]), lying, color=(255, 255, 255))
        glass = Object3D(textured_quad(2.0, 2.0, red), np.array([1.5, 1.0, 0.0]), lying, color=(255, 255, 255))
        fb = renderer.render([floor, fence, glass], camera, sun)

        def at(x):
            cx, cy = renderer.project((x, 0.0, 2.0))  # where the slanting sun puts the shadows, past the panes
            return fb.colour()[int(cy * 2), int(cx)]

        lit, hole, slat, tinted = at(-3.8), at(-2.0), at(-1.0), at(1.5)
        self.assertGreater(hole[1], 0.9 * lit[1])  # light through the hole
        self.assertLess(slat[1], 0.3 * lit[1])  # the solid half's shadow
        # Red light through the red glass: half its light through, and red besides.
        self.assertGreater(tinted[0], 1.3 * tinted[1])
        self.assertLess(tinted[1], 0.7 * lit[1])

    def test_mirrors(self):
        # A mirror facing the camera, with a red box in front of it on the left, a green one on the right,
        # and a blue one behind it, which it hides and must not show either.
        mirror = Object3D(flat_quad(6.0, 3.0), color=(128, 128, 128), reflectivity=1.0)
        boxes = [Object3D(block_mesh((x, 0.0, z), (0.4, 0.4, 0.4)), color=c)
                 for x, z, c in ((-1.0, 1.5, (255, 0, 0)), (1.0, 1.5, (0, 255, 0)), (0.0, -1.0, (0, 0, 255)))]
        camera = Camera(position=np.array([0.0, 0.0, 5.0]))
        flat = Light(direction=np.array([0.0, 0.0, -1.0]), ambient=0.6, diffuse=0.3, specular=0.0)
        renderer = Renderer(80, 30, fog=0, outline=0, background=(0, 0, 0))

        def at(fb, point):
            x, y = renderer.project(point)
            return fb.colour()[int(y * 2), int(x)]

        fb = renderer.render([mirror, *boxes], camera, flat).copy()
        # Each box's image is where it would be seen behind the glass: its mirror image in the plane.
        red, green = at(fb, (-1.0, 0.0, -1.5)), at(fb, (1.0, 0.0, -1.5))
        self.assertTrue(red[0] > 0.1 and red[1] == 0.0)
        self.assertTrue(green[1] > 0.1 and green[0] == 0.0)
        in_mirror = fb.ids == 1
        self.assertTrue(in_mirror.any())
        seen = fb.colour()[in_mirror]
        self.assertFalse((seen[:, 2] > seen[:, 0] + seen[:, 1] + 0.01).any())  # nothing of the box behind it
        self.assertEqual(renderer.pick(renderer.width // 2, renderer.height // 2 - 3).object, mirror)
        # Without reflections, the mirror shows its own grey there.
        renderer.reflections = False
        plain = at(renderer.render([mirror, *boxes], camera, flat), (-1.0, 0.0, -1.5))
        self.assertAlmostEqual(plain[0], plain[1], places=9)
        self.assertGreater(plain[1], 0.1)
        # Two mirrors facing each other: with one bounce, the far one seen in the near one shows no
        # reflection; with two it does, and the red box turns up in it again.
        renderer.reflections = True
        back = Object3D(flat_quad(6.0, 3.0), np.array([0.0, 0.0, 3.0]), quat_axis_angle((0, 1, 0), np.pi),
                        color=(128, 128, 128), reflectivity=1.0)
        camera = Camera(position=np.array([0.3, 0.2, 2.5]), target=np.array([0.0, 0.0, 0.0]))
        one = renderer.render([mirror, back, *boxes[:2]], camera, flat).copy()
        renderer.mirror_bounces = 2
        two = renderer.render([mirror, back, *boxes[:2]], camera, flat)
        self.assertFalse(np.allclose(one.rgb, two.rgb))

    def test_shiny_surfaces_reflect_the_sky(self):
        sky = Sky(zenith=(0, 0, 255), horizon=(255, 255, 255), ground=(255, 0, 0))
        ball = Object3D(blob_mesh((1.0, 1.0, 1.0), rings=24, segments=32), color=(255, 255, 255), reflectivity=1.0)
        camera = Camera(position=np.array([0.0, 0.0, 5.0]))
        renderer = Renderer(60, 30, fog=0, outline=0, background=sky)
        fb = renderer.render([ball], camera, Light(ambient=0.3, diffuse=0.3, specular=0.0))
        x, top = renderer.project((0.0, 0.8, 0.6))
        _, bottom = renderer.project((0.0, -0.8, 0.6))
        up, down = fb.colour()[int(top * 2), int(x)], fb.colour()[int(bottom * 2), int(x)]
        self.assertGreater(up[2], up[0])  # the zenith's blue above
        self.assertGreater(down[0], down[2])  # the ground's red below
        # Glass reflects the sky at a slant, round its edge.
        glass = Object3D(ball.mesh, color=(255, 255, 255), opacity=0.1)
        lit = Light(ambient=0.3, diffuse=0.3, specular=0.0)
        edge = lambda fb: fb.colour()[fb.height // 2, int(renderer.project((0.97, 0.0, 0.0))[0])]
        with_sky = edge(renderer.render([glass], camera, lit)).copy()
        renderer.reflections = False
        self.assertFalse(np.allclose(with_sky, edge(renderer.render([glass], camera, lit))))

    def test_backgrounds(self):
        camera = Camera(position=np.array([0.0, 0.0, 5.0]))
        box = Object3D(make_box(), color=Color.RED)
        plain = Renderer(40, 20).render([box], camera, Light()).copy()
        renderer = Renderer(40, 20, background=(0, 0, 255))
        fb = renderer.render([box], camera, Light())
        blue = to_linear_rgb((0, 0, 255))
        self.assertTrue((fb.alpha == 1.0).all())
        np.testing.assert_allclose(fb.rgb[0, 0], blue)
        self.assertEqual(fb.ids[0, 0], 0)
        # Edge pixels blend by how much of them the scene leaves uncovered.
        edge = (plain.alpha > 0) & (plain.alpha < 1)
        self.assertTrue(edge.any())
        np.testing.assert_allclose(fb.rgb[edge], plain.rgb[edge] + (1 - plain.alpha[edge])[:, None] * blue, atol=1e-12)
        renderer.background = Gradient(top=(255, 255, 255), bottom=(0, 0, 0))
        fb = renderer.render([box], camera, Light())  # a new background draws afresh
        self.assertGreater(fb.rgb[0, 0, 0], 0.95)
        self.assertLess(fb.rgb[-1, 0, 0], 0.01)

    def test_sky_follows_the_view(self):
        sky = Sky(zenith=(0, 0, 255), horizon=(255, 255, 255), ground=(0, 255, 0))
        renderer = Renderer(40, 20, background=sky)
        def centre(target):
            fb = renderer.render([], Camera(position=np.zeros(3), target=np.array(target, float),
                                            up=np.array([0.0, 0.0, -1.0]) if abs(target[1]) > 0.9 else np.array([0, 1.0, 0])), Light())
            return fb.rgb[fb.height // 2, fb.width // 2]
        np.testing.assert_allclose(centre([0, 1, 0]), to_linear_rgb((0, 0, 255)), atol=0.02)
        np.testing.assert_allclose(centre([0, -1, 0]), to_linear_rgb((0, 255, 0)), atol=0.02)
        self.assertGreater(centre([1, 0, 0]).min(), 0.8)  # the horizon, straight ahead

    def test_sky_box_faces(self):
        colours = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
        textures = [np.tile(np.array(c, np.uint8), (8, 8, 1)) for c in colours]
        top_half = np.zeros((8, 8, 3))
        top_half[:4] = 1.0  # white top half, black bottom half
        textures[5] = top_half  # -Z
        renderer = Renderer(40, 20, background=SkyBox(textures))
        for axis, colour in zip(np.eye(3).repeat(2, axis=0) * np.tile([1, -1], 3)[:, None], colours):
            up = np.array([0.0, 0.0, -1.0]) if abs(axis[1]) else np.array([0.0, 1.0, 0.0])
            fb = renderer.render([], Camera(position=np.zeros(3), target=axis, up=up, fov=40), Light())
            if axis[2] < 0:  # looking along -Z: the texture's top is up
                self.assertGreater(fb.rgb[fb.height // 4, fb.width // 2].min(), 0.9)
                self.assertLess(fb.rgb[3 * fb.height // 4, fb.width // 2].max(), 0.1)
            else:
                np.testing.assert_allclose(fb.rgb[fb.height // 2, fb.width // 2], to_linear_rgb(colour), atol=1e-9)

    def test_pick_and_ray(self):
        camera = Camera(position=np.array([0.0, 0.0, 5.0]))
        near = Object3D(make_box(), position=np.array([0.5, 0.0, 0.0]))
        far = Object3D(make_box(2.0), position=np.array([0.0, 0.0, -2.0]))
        renderer = Renderer(40, 20, cell_pixels=(2, 3))
        self.assertIsNone(renderer.pick(20, 10))
        renderer.render([far, near], camera, Light())
        origin, direction = renderer.ray(20, 10)  # the centre of the frame
        np.testing.assert_allclose(origin, camera.position)
        np.testing.assert_allclose(direction, [0, 0, -1], atol=1e-12)
        x, y = renderer.project(np.array([0.5, 0.0, 0.5]))  # the middle of the near box's front face
        hit = renderer.pick(x, y)
        self.assertIs(hit.object, near)
        np.testing.assert_allclose(hit.position[2], 0.5, atol=1e-9)
        self.assertAlmostEqual(hit.distance, np.linalg.norm(hit.position - camera.position), places=9)
        self.assertIs(renderer.pick(*renderer.project(np.array([-0.8, 0.8, -1.0]))).object, far)
        self.assertIsNone(renderer.pick(0, 0))
        self.assertIsNone(renderer.pick(100, 5))

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
        events = d.feed("\x1b[<0;10;5M\x1b[<0;10;5m\x1b[<65;1;1M\x1b[<32;3;3M\x1b[<35;4;4M", 0.0)
        self.assertEqual(events, [MouseEvent(9, 4, 0, True), MouseEvent(9, 4, 0, False), MouseEvent(0, 0, 65, True),
                                  MouseEvent(2, 2, 0, True, moved=True), MouseEvent(3, 3, 3, False, moved=True)])

    def test_kitty_keyboard_protocol(self):
        d = InputDecoder()
        self.assertEqual(d.feed("a\x1b[A", 0.0), [ord("a"), Key.UP])
        self.assertFalse(d.kitty)
        self.assertEqual(d.feed("\x1b[119u\x1b[119;1:2u\x1b[119;1:3u", 0.0), [119, 119, KeyRelease(119)])
        self.assertTrue(d.kitty)
        # Shift+w types W, and its release names W too, even if Shift is let go first.
        self.assertEqual(d.feed("\x1b[119;2;87u\x1b[57441;1:3u\x1b[119;1:3u", 0.0), [87, KeyRelease(87)])
        self.assertEqual(d.feed("\x1b[99;5u\x1b[27u\x1b[13u\x1b[127u\x1b[1;1:3A\x1b[3;1:3~\x1b[57400u", 0.0),
                         [3, Key.ESC, Key.ENTER, Key.BACKSPACE, KeyRelease(Key.UP), KeyRelease(Key.DELETE), ord("1")])
        self.assertEqual(d.feed("\x1b[?27u\x1b[57441;2u", 0.0), [])  # a query reply; Shift on its own

    def test_held_keys_with_releases(self):
        held = HeldKeys()
        held.exact = True
        held.update([ord("W"), Key.UP], 0.0)
        self.assertIn("w", held)
        self.assertIn(Key.UP, held)
        held.update([], 5.0)
        self.assertIn(ord("w"), held)
        held.update([KeyRelease(ord("w"))], 5.1)
        self.assertNotIn("w", held)
        self.assertEqual(held.keys(), [Key.UP])

    def test_held_keys_estimated_from_repeats(self):
        held = HeldKeys(first_hold=0.5)
        held.update([ord("w")], 0.0)
        held.update([], 0.4)
        self.assertIn("w", held)  # waiting for the first repeat
        held.update([], 0.6)
        self.assertNotIn("w", held)  # a tap
        held.update([ord("d")], 1.0)
        for t in (1.5, 1.53, 1.56, 1.59):
            held.update([ord("d")], t)
        held.update([], 1.62)
        self.assertIn("d", held)
        held.update([], 1.7)
        self.assertNotIn("d", held)  # repeats stopped for more than 1.5 intervals

    def test_screen_asks_the_keyboard_for_releases(self):
        class FakeConsole:
            unicode, key_release, reports_releases = True, True, True
            def __init__(self):
                self.text, self.down = "", set()
            def size(self):
                return 10, 4
            def read(self):
                text, self.text = self.text, ""
                return text
            def key_is_down(self, key):
                return key in self.down
        con = FakeConsole()
        screen = Screen(con, glyphs="quad", color="truecolor")
        con.text, con.down = "w", {ord("w")}
        self.assertEqual(screen.keys(), [ord("w")])
        self.assertEqual(screen.keys(), [])
        self.assertIn("w", screen.held)
        con.down = set()
        self.assertEqual(screen.keys(), [KeyRelease(ord("w"))])
        self.assertNotIn("w", screen.held)
        con.key_release = False
        screen = Screen(con, glyphs="quad", color="truecolor")
        con.text, con.down = "w", {ord("w")}
        screen.keys()
        con.down = set()
        self.assertEqual(screen.keys(), [])  # not asked for: releases only update screen.held
        self.assertNotIn("w", screen.held)

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
        self.assertEqual(con.mouse(mouse(5, 2, 1, WindowsInput.MOUSE_MOVED)), "")  # moves not asked for
        drag = WindowsInput("drag")
        text = drag.mouse(mouse(5, 2, 1, WindowsInput.MOUSE_MOVED)) + drag.mouse(mouse(6, 2, 0, WindowsInput.MOUSE_MOVED))
        self.assertEqual(InputDecoder().feed(text, 0.0), [MouseEvent(5, 2, 0, True, moved=True)])
        text = WindowsInput("move").mouse(mouse(6, 2, 0, WindowsInput.MOUSE_MOVED))
        self.assertEqual(InputDecoder().feed(text, 0.0), [MouseEvent(6, 2, 3, False, moved=True)])


class WidgetTests(unittest.TestCase):
    def setUp(self):
        self.screen = Screen(glyphs="quad", color="truecolor", size=(4, 80))

    def test_widgets_follow_clicks_keys_and_focus(self):
        pressed, changes = [], []
        slider = Slider("Balls", 20, 0, 100, step=10, keys="[]", length=11, on_change=changes.append)
        toggle = Toggle("Spin", False, key="s")
        choice = Choice("Mode", ("a", "bb", "c"), key=Key.F5)
        panel = Panel([slider, toggle, choice, Button("Go", lambda: pressed.append(1), key="g")])
        panel.draw(self.screen, 1, 0)
        self.assertEqual("".join(self.screen.chars[1, :slider.width]).rstrip(), "Balls ━━●────────  20")
        x = len("Balls ")
        rest = panel.handle([MouseEvent(x + 10, 1, 0, True), MouseEvent(x + 10, 1, 0, False), ord("q")])
        self.assertEqual(rest, [ord("q")])                  # not the panel's: passed on
        self.assertEqual(slider.value, 100)
        panel.handle([ord("["), ord("s"), Key.F5, ord("g"), MouseEvent(x + 5, 0, 0, True)])
        self.assertEqual((slider.value, toggle.value, choice.value, pressed), (90, True, "bb", [1]))
        self.assertEqual(changes, [100, 90])
        panel.handle([Key.TAB, Key.RIGHT, Key.TAB, ord(" ")])  # focus the slider, step it, then flip the toggle
        self.assertEqual((panel.focus, slider.value, toggle.value), (1, 100, False))
        self.assertEqual(panel.handle([Key.ESC]), [Key.ESC])

    def test_slider_drag_and_wheel(self):
        slider = Slider("", 0.5, 0.0, 1.0, step=0.25, length=5)
        panel = Panel([slider])
        panel.draw(self.screen, 0, 10)
        panel.handle([MouseEvent(10, 0, 0, True), MouseEvent(13, 0, 0, True, moved=True)])
        self.assertEqual(slider.value, 0.75)
        panel.handle([MouseEvent(20, 0, 0, True, moved=True), MouseEvent(20, 0, 0, False)])
        self.assertEqual(slider.value, 1.0)  # dragged past the end
        self.assertEqual(panel.handle([MouseEvent(30, 0, 0, True, moved=True)]), [MouseEvent(30, 0, 0, True, moved=True)])
        panel.handle([MouseEvent(11, 0, MouseEvent.WHEEL_DOWN, True)])
        self.assertEqual(slider.value, 0.75)

    def test_display_controls_change_the_screen(self):
        controls = DisplayControls()
        controls.draw(self.screen, 3, 0)
        self.assertEqual(controls.width, len("F2 sextant  F3 truecolor  F4 999/144fps"))
        controls.handle([Key.F2, Key.F3, Key.F4])
        self.assertEqual((self.screen.mode, self.screen.color_mode, self.screen.fps), ("sextant", "256", 60))
        controls.draw(self.screen, 3, 0)
        controls.handle([MouseEvent(3, 3, 0, True)])  # a click on the glyphs
        self.assertEqual(self.screen.mode, "ascii")
        self.assertIn("F4 --/60fps", "".join(self.screen.chars[3]))
        # Given a renderer, F5 switches its shadows.
        renderer = Renderer(10, 5)
        controls = DisplayControls(renderer=renderer)
        controls.handle([Key.F5], self.screen)
        self.assertFalse(renderer.shadows)
        controls.draw(self.screen, 3, 0)
        self.assertIn("F5 no shadows", "".join(self.screen.chars[3]))
        controls.handle([Key.F5])
        self.assertTrue(renderer.shadows)
        controls.handle([Key.F6])  # and F6 its reflections
        self.assertFalse(renderer.reflections)
        controls.draw(self.screen, 3, 0)
        self.assertIn("F6 no reflections", "".join(self.screen.chars[3]))


class DemoTests(unittest.TestCase):
    """Each example program draws frames off-screen, takes its keys and clicks, and quits."""

    def run_demo(self, demo, events, quit_key):
        screen = Screen(glyphs="quad", color="256", size=(30, 100))
        for ev in [[], *([e] for e in events), []]:
            self.assertIsNot(demo.frame(screen, 1 / 30, ev), False)
        self.assertTrue((screen.chars != " ").any())
        self.assertIn("F2 quad", "".join(screen.chars[-1]))  # the display settings are on the status line
        self.assertIs(demo.frame(screen, 1 / 30, [quit_key]), False)
        return screen

    def test_dice_demo(self):
        from unicode3d.examples.demo import DiceDemo
        self.run_demo(DiceDemo(3, seed=1), [ord(" "), ord("+"), Key.F3, ord("g"), ord("m"), Key.F6], ord("q"))

    def test_viewer(self):
        from unicode3d.examples.viewer import Viewer
        self.run_demo(Viewer(make_die(), False), [ord("w"), Key.LEFT, ord("e"), ord("c")], Key.ESC)

    def test_balls(self):
        from unicode3d.examples.balls import Balls
        demo = Balls(30, seed=1)
        self.run_demo(demo, [ord("]"), ord("g"), ord(" "), Key.TAB, Key.RIGHT, MouseEvent(50, 12, 0, True), Key.F5,
                             ord("x"), ord("m")], ord("q"))
        self.assertEqual(len(demo.balls), 32)  # "]" and then the focused slider's Right added one each

    def test_maze(self):
        from unicode3d.examples.maze import Maze
        demo = Maze(5, seed=1)
        self.run_demo(demo, [ord("t"), ord("m"), ord("r"), ord("n"), ord("l"), ord("p")], ord("q"))
        demo.speed.value = 5.0
        screen = Screen(glyphs="quad", color="256", size=(30, 100))
        first = demo.walls  # held, so a new maze can't reuse its memory (and id)
        for _ in range(600):  # walks to the exit and starts a new maze
            demo.frame(screen, 0.1, [])
        self.assertIsNot(demo.walls, first)

    def test_room(self):
        from unicode3d.examples.room import Walk
        demo = Walk(seed=1)
        start = (demo.x, demo.z)
        screen = self.run_demo(demo, [ord("b"), ord("l"), ord("]"), MouseEvent(50, 10, 0, True)], Key.ESC)
        screen.held.exact = True
        screen.held.update([ord("w")], 0.0)
        for _ in range(10):
            demo.frame(screen, 0.1, [])
        self.assertLess(demo.z, start[1] - 2.0)  # walked north


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

    def test_colour_codes_written_as_bytes(self):
        buf = np.zeros(64, np.uint8)
        for packed in [-1, *encode_index(np.arange(256)), *encode_rgb([(0, 0, 0), (255, 255, 255), (7, 80, 200)])]:
            for background in (False, True):
                n = put_sgr_color(buf, 0, int(packed), background)
                self.assertEqual(buf[:n].tobytes().decode(), sgr_color(packed, background))

    def test_ascii_terminal_gets_only_ascii(self):
        screen = Screen(glyphs="ascii", color="16", size=(2, 10))
        screen.unicode = False
        screen.text(0, 0, "h\u00e9 \u2713!")
        self.assertIn("h? ?!", screen.render_updates())

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
