# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import unittest.mock

import numba
import numpy as np

from unicode3d import fonts, pictures, terminal
from unicode3d.animation import Animation, Clip, Track
from unicode3d.examples.dice import DIE_VALUES, RollAnimation, make_die, orientation_showing, top_face
from unicode3d.mesh import Mesh, load_obj, make_box
from unicode3d.queries import Colliders
from unicode3d.detail import detail_levels
from unicode3d.background import Fog, Gradient, Sky, SkyBox
from unicode3d.color import (Color, encode_index, encode_rgb, linear_to_srgb, luminance, put_sgr_color, quantize,
                             sgr_color, srgb_to_linear, to_linear_rgb, xterm_rgb)
from unicode3d.console import WindowsInput, detect_color_mode, detect_glyphs
from unicode3d.glyphs import GLYPH_SETS, frame_to_text, match_cells
from unicode3d.keys import HeldKeys, InputDecoder, Key, KeyRelease, MouseEvent
from unicode3d.raster import FrameBuffer
from unicode3d.scene import Camera, Light, Model, Node, Object3D, PointLight, Renderer, union_bounds
from unicode3d.shapes import blob_mesh, block_mesh, merge_meshes, text_bitmap, text_mesh
from unicode3d.kernel_signatures import KERNELS
from unicode3d.precompile import missing_kernels, share_out
from unicode3d.terminal import SETTLE_FRAMES, Screen, compile_kernels
from unicode3d.texture import BLEND, CUTOUT, alpha_kind, build_mipmaps
from unicode3d.ui import Button, Choice, DisplayControls, Panel, Slider, Toggle
from unicode3d.animation import (EASINGS, LOOPS, Animation, RotationTrack, SplineTrack, Track, ease_in, ease_out,
                                 ease_out_back)
from unicode3d.transforms import normalize, quat_axis_angle, quat_between, quat_identity, quat_slerp, quat_to_matrix, scene_poses


class TransformTests(unittest.TestCase):
    def test_quat_between_maps_u_onto_v(self):
        rng = np.random.default_rng(1)
        pairs = [(rng.normal(size=3), rng.normal(size=3)) for _ in range(20)]
        pairs += [((0, 1, 0), (0, -1, 0)), ((1, 0, 0), (-1, 0, 0)), ((0, 0, 1), (0, 0, 1))]
        for u, v in pairs:
            u, v = np.asarray(u, float), np.asarray(v, float)
            got = quat_to_matrix(quat_between(u, v)) @ (u / np.linalg.norm(u))
            np.testing.assert_allclose(got, v / np.linalg.norm(v), atol=1e-9)

    def test_scene_poses_compose_every_parent_once(self):
        # A random tree (shared parents, numbers and triples as scales, a hidden branch): every node's pose is its
        # parent's composed with its own, and world_matrix() gives the same to the bit.
        rng = np.random.default_rng(3)
        nodes = []
        for i in range(60):
            parent = nodes[rng.integers(len(nodes))] if nodes and i % 7 else None
            scale = float(rng.uniform(0.5, 2.0)) if i % 2 else rng.uniform(0.5, 2.0, 3)
            node = (Node if i % 3 else Object3D)(*([make_box()] if i % 3 == 0 else []), position=rng.normal(size=3),
                                                 rotation=rng.normal(size=4), scale=scale, parent=parent)
            nodes.append(node)
        nodes[8].visible = False
        rows, linear, position, visible = scene_poses(nodes[::-1])
        self.assertEqual(len(rows), len(nodes))
        for node in nodes:
            r = rows[id(node)]
            own = quat_to_matrix(node.rotation) * np.broadcast_to(node.scale, 3)
            up_lin, up_pos, up_vis = (np.eye(3), np.zeros(3), True) if node.parent is None else (
                linear[rows[id(node.parent)]], position[rows[id(node.parent)]], visible[rows[id(node.parent)]])
            np.testing.assert_allclose(linear[r], up_lin @ own, atol=1e-12)
            np.testing.assert_allclose(position[r], up_pos + up_lin @ node.position, atol=1e-12)
            self.assertEqual(visible[r], up_vis and node.visible)
            lin, pos, vis = node.world_matrix()
            np.testing.assert_array_equal(lin, linear[r])
            np.testing.assert_array_equal(pos, position[r])
            self.assertEqual(vis, visible[r])
        self.assertFalse(visible.all())
        loop = Node()
        loop.parent = Node(parent=loop)
        with self.assertRaises(ValueError):
            scene_poses([loop])
        with self.assertRaises(ValueError):
            scene_poses([Node(scale=(1.0, 2.0))])


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

    def test_the_pixel_font_has_every_printable_character(self):
        printable = [chr(c) for c in range(32, 127)]
        self.assertEqual(sorted(fonts.PIXEL), sorted(printable))
        for ch, rows in fonts.PIXEL.items():
            self.assertEqual(len(rows), 9, ch)
            self.assertEqual(len({len(r) for r in rows}), 1, ch)
            self.assertTrue(ch == " " or any("#" in r for r in rows), ch)
        mesh, width = text_mesh("Fox, jumps!")  # the built-in font by default
        cells = text_bitmap("Fox, jumps!")
        self.assertEqual(width, cells.shape[1])
        self.assertAlmostEqual(volume(mesh), cells.sum() * 1.6)
        with self.assertRaises(KeyError):
            text_bitmap("\u00e9")
        with self.assertRaises(ValueError):
            fonts.font(3, {"x": "##/#"})

    def test_block_text_has_a_box_per_pixel(self):
        cells = text_bitmap("Hi")
        mesh, width = text_mesh("Hi", depth=1.0, blocks=True, gap=0.2)
        self.assertEqual(len(mesh.faces), 12 * cells.sum())
        self.assertAlmostEqual(volume(mesh), cells.sum() * 0.8 * 0.8)
        h, w = cells.shape
        rows, cols = np.nonzero(cells)
        for k in (0, len(rows) // 2, len(rows) - 1):  # faces 12k..12k+11 are the k-th pixel's, in reading order
            v = mesh.vertices[mesh.faces[12 * k:12 * k + 12]].reshape(-1, 3)
            centre = (v.min(axis=0) + v.max(axis=0)) / 2
            np.testing.assert_allclose(centre, (cols[k] + 0.5 - w / 2, h / 2 - rows[k] - 0.5, 0.0), atol=1e-12)

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

    def test_objects_parented_to_the_camera_stay_in_front_of_it(self):
        for camera in (Camera(position=np.array([3.0, 2.0, 5.0]), target=np.array([0.0, 0.5, 0.0])),
                       Camera(position=np.array([-4.0, 1.0, -2.0]), target=np.array([1.0, 0.0, 1.0]), fov=70),
                       Camera(position=np.array([0.0, 9.0, 0.0])),  # looking straight down
                       Camera(position=np.array([2.0, 6.0, 2.0]), projection="ortho", size=4.0)):
            right, up, forward = camera.basis()
            np.testing.assert_allclose([right @ up, up @ forward, forward @ right], 0.0, atol=1e-12)
            np.testing.assert_allclose(quat_to_matrix(camera.rotation), np.column_stack([right, up, -forward]),
                                       atol=1e-12)
            box = Object3D(make_box(0.5), position=np.array([0.0, 0.0, -3.0]), parent=camera)
            linear, position, visible = box.world_matrix()
            np.testing.assert_allclose(position, camera.position + 3.0 * forward, atol=1e-12)
            fb = Renderer(40, 20).render([box], camera, Light())
            rows, cols = np.nonzero(fb.alpha > 0.5)
            self.assertAlmostEqual(rows.mean(), 19.5, delta=1.0)
            self.assertAlmostEqual(cols.mean(), 19.5, delta=1.0)
            tall = camera.height_at(3.0)  # the box is 0.5 of that, in 40 pixel rows
            self.assertAlmostEqual(np.ptp(rows) + 1, 40 * 0.5 / tall, delta=40 * 0.08 / tall + 1)
        self.assertEqual(Camera(projection="ortho", size=7.0).height_at(100.0), 7.0)
        self.assertAlmostEqual(Camera(fov=90.0).height_at(2.0), 4.0)
        stuck = Camera(position=np.zeros(3), target=np.zeros(3))  # at its target: drawn as nothing, no exception
        Renderer(20, 10).render([Object3D(make_box(), position=np.array([0.0, 0.0, -3.0]), parent=stuck)], stuck,
                                Light())

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

    def test_render_lists_that_change_render_as_afresh(self):
        # A renderer keeps the meshes and textures it has packed: whatever it drew before, gaining and losing
        # objects and textures (added to its pack, or packed afresh), each frame is what a new renderer draws.
        rng = np.random.default_rng(5)
        checks = np.kron(np.indices((8, 8)).sum(axis=0) % 2, np.ones((8, 8)))  # (64, 64)
        noise = rng.uniform(0.0, 1.0, (128, 128, 3))
        holes = np.ones((32, 32, 4))
        holes[8:24, 8:24, 3] = 0.0
        small = rng.uniform(0.0, 1.0, (16, 16, 3))
        a, a2 = Object3D(make_box(1.0, [checks] * 6)), Object3D(make_box(0.8, [checks] * 6))
        self.assertIs(a.mesh.mipmaps(0), a2.mesh.mipmaps(0))  # (one chain for every mesh showing the texture)
        b, c = Object3D(make_box(1.0, [noise] * 6)), Object3D(make_box(1.0, [holes] * 6))
        d, plain = Object3D(make_box(0.7, [small] * 6)), Object3D(blob_mesh((0.5, 0.5, 0.5)), color=Color.RED)
        for k, obj in enumerate((a, a2, b, c, d, plain)):
            obj.position = np.array([k * 1.3 - 3.2, 0.0, 0.0])
            obj.rotation = quat_axis_angle([1.0, 1.0, 0.3], 0.5 + k)
        renderer, camera, light = Renderer(60, 20), Camera(position=np.array([0.0, 1.0, 6.0]), fov=60.0), Light()
        for objects in ([a, plain], [a, plain, a2], [a, plain, a2, b], [c], [c, d], [a, b, c, d, plain], [a],
                        "edit", [plain, a2, c]):
            if objects == "edit":
                a.mesh.vertices *= 1.2  # (edited in place: seen after invalidate())
                renderer.invalidate()
                objects = [a, plain, d]
            got = renderer.render(objects, camera, light)
            want = Renderer(60, 20).render(objects, camera, light)
            np.testing.assert_array_equal(got.rgb, want.rgb)
            np.testing.assert_array_equal(got.ids, want.ids)
            np.testing.assert_array_equal(got.depth, want.depth)

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
        # A floor whose textures repeat (uv beyond 0..1), with holes too, and authored normals.
        floor = textured_quad(4.0, 3.0, holes)
        floor.uvs = floor.uvs * 5.0 - 1.0
        floor.normals = rng.normal(size=(4, 3))
        crowd += [Object3D(floor, np.array([0.0, -1.0, -0.5]), quat_axis_angle((1, 0, 0), -1.3), color=(255, 255, 255),
                           double_sided=True)]
        # Mirrors (seeing each other: two bounces), and something shiny.
        crowd += [Object3D(flat_quad(2.0, 1.5), np.array([x, 0.0, -1.2]), quat_axis_angle((0, 1, 0), turn),
                           color=(150, 150, 150), reflectivity=0.8) for x, turn in ((-0.8, 0.6), (0.8, -0.6))]
        crowd[3].reflectivity = 0.9
        # Materials, and shapes stretched unevenly, mirrored, and sheared by a stretched parent.
        crowd[6].specular, crowd[7].specular, crowd[7].shininess = 0.0, 2.5, 90.0
        stretched = Node(position=np.array([-0.3, 0.2, 0.0]), scale=(1.0, 2.0, 0.5))
        crowd[9].scale, crowd[10].scale, crowd[11].parent = (0.4, 0.1, 0.3), (-0.3, 0.3, 0.3), stretched
        lights = [Light(color=(255, 200, 150)), PointLight(np.array([0.5, 0.5, 1.0]), color=Color.CYAN, range=3.0)]
        shadowed = [Light(direction=np.array([0.5, -1.0, -0.3]), shadows=True), Light(shadows=True),
                    PointLight(np.array([0.5, 0.5, 1.0]), color=Color.CYAN, range=3.0, shadows=True),
                    PointLight(np.array([-0.2, 0.0, 0.3]), color=Color.RED, range=2.0, shadows=True)]
        scenes = [([Object3D(make_die(), rotation=quat_axis_angle((1, 2, 0), 0.7)), *spheres], Light()),
                  ([Object3D(make_box(), position=np.array([0.0, 0.0, 4.6]))], Light()),  # camera in front of its near face
                  (crowd, lights),
                  (crowd, shadowed)]
        ortho = Camera(position=np.array([3.0, 3.0, 4.0]), target=np.zeros(3), projection="ortho", size=4.0)

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
            # Fog in the world, into the background and into a colour.
            for fog in (Fog(start=4.0, end=7.0), Fog(start=3.0, end=8.0, color=(200, 100, 50))):
                renderer = Renderer(60, 30, screen.cell_pixels, background=Sky(), fog=fog)
                fb = renderer.render(crowd, Camera(position=np.array([0.0, 0.5, 5.0])), lights)
                frame += [fb.rgb.copy(), fb.alpha.copy(), fb.depth.copy(), fb.ids.copy()]
            # Through an orthographic camera, with outlines and fog.
            renderer = Renderer(60, 30, screen.cell_pixels, background=Sky(), outline=0.4, fog=Fog(start=3.0, end=8.0))
            fb = renderer.render(crowd, ortho, shadowed)
            frame += [fb.rgb.copy(), fb.alpha.copy(), fb.depth.copy(), fb.ids.copy()]
            # Ray queries, many at once (a chunk of rays to each thread).
            rays = np.random.default_rng(9).normal(size=(2, 3000, 3))
            objects, *arrays = Colliders(crowd).raycast_many(rays[0] * 3.0, rays[1])
            index = {id(o): i for i, o in enumerate(crowd)}
            frame += [np.array([index[id(o)] if o is not None else -1 for o in objects]), *arrays]
            return frame

        try:
            one = draw(1)
            for _ in range(5):
                for a, b in zip(one, draw(most)):
                    np.testing.assert_array_equal(a, b)
        finally:
            numba.set_num_threads(most)

    def test_world_bounds(self):
        group = Node(position=np.array([1.0, 2.0, 0.0]), rotation=quat_axis_angle((0, 0, 1), np.pi / 2), scale=2.0)
        plank = Object3D(make_box(), np.array([0.0, 1.0, 0.0]), scale=(3.0, 0.1, 1.0), parent=group)
        lo, hi = plank.world_bounds()  # 3 long along x, turned upright by the group, then doubled
        np.testing.assert_allclose(lo, [1.0 - 2.0 - 0.1, 2.0 - 3.0, -1.0], atol=1e-12)
        np.testing.assert_allclose(hi, [1.0 - 2.0 + 0.1, 2.0 + 3.0, 1.0], atol=1e-12)
        ball = Object3D(blob_mesh((1.0, 1.0, 1.0)), np.array([5.0, 0.0, 0.0]))
        lo, hi = union_bounds([plank, Model(Node(), [ball])])
        np.testing.assert_allclose(lo, [-1.1, -1.0, -1.0], atol=1e-12)
        np.testing.assert_allclose(hi, [6.0, 5.0, 1.0], atol=1e-12)
        np.testing.assert_allclose(Model(Node(), [plank, ball]).world_bounds()[1], hi)
        holed = make_box()
        holed.vertices = holed.vertices.copy()
        holed.vertices[0] = np.nan  # left out
        np.testing.assert_allclose(Object3D(holed).world_bounds()[1], 0.5)
        self.assertIsNone(Object3D(make_box(), np.array([np.nan, 0.0, 0.0])).world_bounds())
        lost = Object3D(make_box(), np.array([0.0, 0.0, 0.0]), scale=np.inf)  # left out of a union
        np.testing.assert_allclose(union_bounds([lost, ball])[0], [4.0, -1.0, -1.0], atol=1e-12)
        flat = make_box()
        flat.vertices = flat.vertices * [1e300, 1.0, 1.0]
        self.assertIsNone(Object3D(flat, scale=1e10).world_bounds())  # (overflows to infinity)
        self.assertIsNone(Object3D(Mesh(np.zeros((0, 3)), np.zeros((0, 3), int))).world_bounds())
        self.assertIsNone(union_bounds([]))

    def test_objects_are_equal_only_to_themselves(self):
        # Lists of objects work with in, index() and remove() (comparing their arrays raised ValueError), and
        # objects can be set members and dict keys: a hit's object looked up among the enemies, say.
        a, b = Object3D(make_box()), Object3D(make_box())
        twin = Object3D(a.mesh, a.position, a.rotation)  # the very same arrays, but another object
        objects = [a, b]
        self.assertIn(b, objects)
        self.assertNotIn(twin, objects)
        self.assertEqual(objects.index(b), 1)
        objects.remove(b)
        self.assertEqual(objects, [a])
        self.assertEqual(len({a, b, twin}), 3)
        self.assertEqual({a: 1, b: 2}[b], 2)
        self.assertNotEqual(Node(), Node())
        mesh = make_box()
        self.assertEqual([m for m in (make_box(), mesh) if m == mesh], [mesh])
        model = Model(Node(), [a])
        self.assertEqual([Model(Node(), [b]), model].index(model), 1)

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

    def test_what_is_kept_between_frames_sees_every_change(self):
        # A renderer keeps the scene graph's order and colours that can't change in place from frame to frame;
        # each change below must be drawn exactly as a fresh renderer draws it.
        a, b = Node(position=np.array([-1.0, 0.0, 0.0])), Node(position=np.array([1.0, 0.5, 0.0]), scale=0.5)
        child = Object3D(make_box(0.6), parent=a, color=(200, 40, 40))
        other = Object3D(make_box(0.4), position=np.array([0.0, -1.0, 0.0]), color=[40, 200, 40])
        tinted = Object3D(make_box(0.3), position=np.array([0.0, 1.0, 0.0]), color=np.array([40.0, 40.0, 200.0]))
        objects = [child, other, tinted]
        camera, light = Camera(position=np.array([0.0, 1.0, 6.0])), Light()
        kept = Renderer(60, 24)

        def same():
            got, fresh = kept.render(objects, camera, light), Renderer(60, 24).render(objects, camera, light)
            np.testing.assert_array_equal(got.rgb, fresh.rgb)
            np.testing.assert_array_equal(got.ids, fresh.ids)

        changes = [lambda: None,
                   lambda: setattr(child, "parent", b),           # re-parented
                   lambda: setattr(b, "parent", a),               # its parent gets one
                   lambda: setattr(child, "color", (40, 40, 40)),  # a new tuple
                   lambda: other.color.__setitem__(0, 250),       # a list edited in place
                   lambda: tinted.color.__setitem__(2, 90.0),     # an array edited in place
                   lambda: setattr(b, "scale", (1.0, 2.0, 0.5)),  # a number to three
                   lambda: setattr(child, "scale", 2),            # an int
                   lambda: objects.reverse()]                     # the same objects in another order
        for change in changes:
            change()
            same()
        # scene_poses with a memo gives what it gives without one, through the same kind of changes.
        memo = {}
        for change in (lambda: None, lambda: setattr(child, "parent", None), lambda: setattr(b, "parent", None)):
            change()
            for x, y in zip(scene_poses(objects, memo=memo)[1:], scene_poses(objects)[1:]):
                np.testing.assert_array_equal(x, y)

    def test_levels_of_detail(self):
        # A knobbly ball with a colour per face and a texture: each level has fewer faces, keeps faces' colours,
        # texture coordinates and materials, and moves no vertex further than its error.
        ball = blob_mesh((1.0, 1.0, 1.0), rings=24, segments=32)
        ball.vertices = ball.vertices * (1.0 + 0.1 * np.sin(6.0 * ball.vertices[:, :1]))
        rng = np.random.default_rng(5)
        ball.face_colors = rng.integers(0, 256, (len(ball.faces), 3))
        ball.textures = [np.full((4, 4), 0.8)]
        ball.materials = np.zeros(len(ball.faces), np.int64)
        ball.uvs = rng.uniform(0, 1, (len(ball.faces), 3, 2))
        levels = detail_levels(ball)
        self.assertIs(levels[0][0], ball)
        self.assertGreater(len(levels), 3)
        self.assertIs(detail_levels(ball), levels)  # (kept on the mesh)
        rows = {tuple(c) + tuple(uv.ravel()) for c, uv in zip(ball.face_colors, ball.uvs)}
        for (finer, e0), (level, error) in zip(levels, levels[1:]):
            level.check()
            self.assertLessEqual(len(level.faces), 0.75 * len(finer.faces))
            self.assertGreaterEqual(error, e0)
            self.assertIs(level.textures, ball.textures)
            self.assertTrue({tuple(c) + tuple(uv.ravel()) for c, uv in zip(level.face_colors, level.uvs)} <= rows)
            nearest = np.sqrt(((level.vertices[:, None] - ball.vertices[None]) ** 2).sum(axis=2)).min(axis=1)
            self.assertLessEqual(nearest.max(), error + 1e-12)
        ball.vertices = ball.vertices * 1.0  # (replaced: made afresh)
        self.assertIsNot(detail_levels(ball), levels)
        broken = blob_mesh((1.0, 1.0, 1.0))
        broken.vertices[3] = np.nan
        self.assertEqual(len(detail_levels(broken)), 1)

    def test_simplify_draws_small_objects_from_levels_of_detail(self):
        ball = blob_mesh((1.0, 1.0, 1.0), rings=48, segments=64)
        light = Light()

        def drawn(objects, camera, simplify):
            renderer = Renderer(80, 30, simplify=simplify)
            for _ in range(20):  # (levels are made a few meshes a frame)
                fb = renderer.render(objects, camera, light)
            inst = renderer._instances(objects, camera)
            return fb, int(np.diff(inst["pack"]["mesh_face"])[inst["mesh"]].sum()), inst

        # Near: as it is, exactly.
        near = Camera(position=np.array([0.0, 0.0, 2.5]))
        full, faces, _ = drawn([Object3D(ball, color=(200, 100, 50))], near, 0.0)
        same, fewer, _ = drawn([Object3D(ball, color=(200, 100, 50))], near, 1.0)
        np.testing.assert_array_equal(full.rgb, same.rgb)
        self.assertEqual(faces, fewer)
        # Far: far fewer faces, and the picture all but the same.
        far = Camera(position=np.array([0.0, 0.0, 25.0]), fov=30.0)  # (the ball about 10 pixels across)
        full, faces, _ = drawn([Object3D(ball, color=(200, 100, 50))], far, 0.0)
        simple, fewer, _ = drawn([Object3D(ball, color=(200, 100, 50))], far, 1.0)
        self.assertLess(fewer, faces / 3)
        self.assertLess(np.abs(simple.colour() - full.colour()).mean(), 0.01)
        self.assertLessEqual(abs(int(full.drawn.sum()) - int(simple.drawn.sum())), 3)  # (of about 75)
        # An object with simplify=False (lettering, say) is drawn as it is, however small, beside one that isn't.
        exact = Object3D(ball, color=(200, 100, 50), simplify=False)
        _, both, _ = drawn([exact, Object3D(ball, position=np.array([1.5, 0.0, 0.0]))], far, 1.0)
        self.assertEqual(both, faces + fewer)
        # A flattened ball's coarsest levels are flat: a shiny one never takes those (a flat shiny mesh is a
        # mirror); a matt one does.
        slab = blob_mesh((1.0, 0.02, 1.0), rings=24, segments=32)
        for shine, mirror in ((0.0, True), (0.8, False)):
            _, _, inst = drawn([Object3D(slab, color=(200, 200, 200), reflectivity=shine)], far, 40.0)
            self.assertEqual(bool(np.isfinite(inst["pack"]["planes"][inst["mesh"][0], 0])), mirror)
        # Arrays edited in place, and invalidate(): levels made afresh.
        levels = detail_levels(ball)
        ball.vertices *= 1.5
        Renderer(10, 10).invalidate()
        self.assertIsNot(detail_levels(ball), levels)

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
        # Textures, normals and opacity are kept: untextured faces get a white texture, a part without normals
        # gets those it would have had, colours without opacity are solid.
        checks = (np.indices((4, 4)).sum(axis=0) % 2).astype(float)
        textured = make_box(textures=[checks] * 6)
        smooth = blob_mesh((1.0, 1.0, 1.0))
        smooth.normals = smooth.vertex_normals() * 3.0
        glassy = make_box()
        glassy.face_colors = np.full((12, 4), 0.5)
        merged = merge_meshes([textured, smooth, glassy])
        merged.check()
        self.assertEqual(len(merged.textures), 2)  # the one checkerboard (shared by six faces), and white
        np.testing.assert_array_equal(merged.materials[:12], 0)
        np.testing.assert_array_equal(merged.materials[12:], 1)
        np.testing.assert_allclose(merged.textures[1], 1.0)
        np.testing.assert_allclose(merged.uvs[:12], textured.uvs)
        np.testing.assert_allclose(merged.vertex_normals()[:len(textured.vertices)], textured.vertex_normals())
        n = len(textured.vertices) + len(smooth.vertices)
        np.testing.assert_allclose(merged.vertex_normals()[len(textured.vertices):n], smooth.vertex_normals())
        self.assertEqual(merged.face_colors.shape, (len(merged.faces), 4))
        np.testing.assert_allclose(merged.face_colors[:12], 1.0)
        np.testing.assert_allclose(merged.face_colors[-12:], 0.5)

    def test_bake_draws_a_model_in_few_parts(self):
        # Parts through turned, stretched and mirrored parents, textured, coloured and plain, merge into one mesh;
        # a see-through part keeps a mesh of its own, and a hidden one is left out. It draws as the model did.
        checks = (np.indices((8, 8)).sum(axis=0) % 2 * 0.6 + 0.4)[..., None] * np.array([1.0, 0.8, 0.5])
        table = Node()
        root = Node(np.array([0.2, -0.3, 0.0]), quat_axis_angle((0, 1, 0), 0.5), (1.0, 1.2, 0.9), parent=table)
        arm = Node(np.array([0.6, 0.0, 0.0]), quat_axis_angle((0, 0, 1), 0.4), scale=(1.0, -1.0, 1.0), parent=root)
        tinted = make_box()
        tinted.vertex_colors = np.random.default_rng(2).uniform(0.3, 1.0, (len(tinted.vertices), 3))
        parts = [Object3D(make_box(textures=[checks] * 6), np.array([-0.7, 0.0, 0.0]), scale=0.8, color=(255, 120, 90),
                          parent=root),
                 Object3D(tinted, np.array([0.0, 0.6, 0.0]), scale=(0.5, 0.3, 0.5), color=(0.4, 0.9, 1.0), parent=arm),
                 Object3D(blob_mesh((0.4, 0.4, 0.4)), np.array([0.5, 0.0, 0.3]), color=Color.YELLOW, parent=arm),
                 Object3D(make_box(0.4), np.array([0.0, -0.8, 0.5]), opacity=0.5, color=Color.CYAN, parent=root),
                 Object3D(make_box(), np.array([0.0, 3.0, 0.0]), visible=False, parent=root)]
        model = Model(root, parts)
        baked = model.bake()
        self.assertEqual(len(baked), 2)
        self.assertIs(baked.root.parent, table)
        self.assertTrue(all(o.parent is baked.root for o in baked))
        self.assertEqual(sorted(o.opacity for o in baked), [0.5, 1.0])
        self.assertEqual((baked.nodes, baked.animations), ({}, {}))
        camera = Camera(position=np.array([1.0, 1.5, 4.0]))
        lights = [Light(shadows=True), PointLight(np.array([1.0, 2.0, 2.0]))]
        a = Renderer(80, 40, fog=0).render([*model], camera, lights).copy()
        b = Renderer(80, 40, fog=0).render([*baked], camera, lights)
        self.assertGreater(a.drawn.sum(), 600)
        diff = np.abs(a.colour() - b.colour()).max(axis=2)
        self.assertLess((diff > 0.02).sum(), 20)  # (a sample or two at the parts' edges)
        np.testing.assert_allclose(np.median(diff[a.drawn]), 0.0, atol=1e-3)
        np.testing.assert_allclose(union_bounds(parts[:4]), baked.world_bounds(), atol=1e-9)  # (the hidden one left out)
        self.assertIs(parts[0].parent, root)  # (the model is left as it was)

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

    def test_settled_casters_shadow_maps_come_out_as_drawn_whole(self):
        # Each light keeps a map of its settled casters and draws the rest over it (shadows._Settled): every frame's
        # maps must be those drawn from every caster, to the bit, while casters settle, start moving, come and go,
        # and the light turns. See-through ones (multiplied in order) are never kept.
        from unicode3d import shadows

        box, other, stretched = make_box(), make_box(), make_box()  # (casters are told apart by mesh and pose)

        def scene(f):
            boxes = [Object3D(box, position=np.array([x, -1.0, z]), scale=0.4)
                     for x in (-1.5, 0.0, 1.5) for z in (-1.0, 0.5)]
            walker = Object3D(box, position=np.array([np.sin(f / 5), 0.0, 0.0]), scale=0.3)  # always moving
            stops = Object3D(box, position=np.array([0.0, 0.5, min(f, 12) / 10]), scale=0.3)  # then still
            starts = Object3D(box, position=np.array([max(f - 20, 0) / 10, 0.8, -0.5]), scale=0.3)
            glass = Object3D(box, position=np.array([-0.8, 0.2, 0.3]), scale=0.4, opacity=0.5, color=(200, 60, 60))
            floor = Object3D(other, position=np.array([0.0, -1.6, 0.0]), scale=(4.0, 0.1, 4.0))
            if f == 36:
                stretched.vertices = stretched.vertices * (1.0, 2.0, 1.0)  # a settled mesh's arrays replaced
            things = boxes + [walker, stops, starts, glass, floor,
                              Object3D(stretched, position=np.array([-1.2, -0.2, -1.2]), scale=0.3)]
            if 15 <= f < 25:
                things.remove(boxes[2])  # one goes, and comes back
            if f % 4 == 0:  # meshes coming and going (the pack made again)
                things.append(Object3D(make_box(), position=np.array([1.5, 1.0, 1.0]), scale=0.1))
            if f >= 28:
                things.append(Object3D(make_box(), position=np.array([1.0, 1.0, 1.0]), scale=0.2))
            sun = Light(direction=np.array([-0.4, -1.0, -0.3 + (0.2 if f >= 32 else 0.0)]), shadows=True)
            lamp = PointLight(np.array([0.3, 1.5, 0.8]), range=6.0, shadows=True)
            return things, [sun, lamp]

        camera = Camera(position=np.array([0.0, 2.0, 5.0]))
        kept, whole = Renderer(48, 24), Renderer(48, 24)
        settle = shadows.SETTLE_DRAWS
        try:
            for f in range(44):
                things, lights = scene(f)
                shadows.SETTLE_DRAWS = 3
                kept.render(things, camera, lights)
                a = [x.copy() for x in kept._shadows[1]]
                shadows.SETTLE_DRAWS = 10 ** 9  # nothing settles: every caster drawn every time
                whole.render(things, camera, lights)
                b = whole._shadows[1]
                for x, y in zip(a[:1] + a[2:], b[:1] + b[2:]):  # texels, mats, params
                    np.testing.assert_array_equal(x, y, f"frame {f}")
                tinted = a[1][0] > 0.0  # (trans holds tints only behind see-through things; the rest isn't cleared)
                np.testing.assert_array_equal(a[1][0], b[1][0], f"frame {f}")
                np.testing.assert_array_equal(a[1][1:, tinted], b[1][1:, tinted], f"frame {f}")
                self.assertTrue(tinted.any())
                np.testing.assert_array_equal(kept.framebuffer.rgb, whole.framebuffer.rgb, f"frame {f}")
                if f in (10, 35, 43):  # kept, through repacks and the mesh changed
                    self.assertTrue(all(entry is not None for entry in kept._settled.values()), f)
        finally:
            shadows.SETTLE_DRAWS = settle

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

    def test_glass_resting_on_a_surface_hides_the_face_touching_it(self):
        # A glass case's bottom on a table lies in the table's own plane: rounding used to decide, pixel by pixel and
        # differently as the camera moved, whether it was in front, so the table flickered under the case (the room
        # demo's dice). A see-through face exactly on a solid one never shows, from anywhere.
        table = Object3D(block_mesh((0.0, 0.5, 0.0), (3.0, 1.0, 3.0)), color=(90, 90, 110))
        glass = Object3D(block_mesh((0.1, 0.0, -0.2), (1.1, 0.0, 1.1)), np.array([0.0, 1.0, 0.0]),
                         color=(200, 225, 255), opacity=0.15, double_sided=True)  # just a bottom face, on the top
        light = Light(direction=np.array([0.5, -1.0, -0.35]), ambient=0.3, diffuse=0.6)
        renderer = Renderer(80, 30, fog=0, outline=0)
        for k in range(12):
            a = 0.6 + 0.004 * k
            camera = Camera(position=np.array([np.sin(a) * 2.6, 1.9, np.cos(a) * 2.6]), target=np.array([0.0, 1.0, 0.0]))
            bare = renderer.render([table], camera, light).copy()
            fb = renderer.render([table, glass], camera, light)
            np.testing.assert_array_equal(fb.rgb, bare.rgb, err_msg=f"camera {k}")
            np.testing.assert_array_equal(fb.ids, bare.ids, err_msg=f"camera {k}")

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

    def test_sky_straight_up_turns_as_the_scene_does(self):
        # Looking straight up along the default up, the view takes another up (look_at); the sky behind it used to
        # take none, and smeared each row of the sky box into one colour.
        textures = [np.random.default_rng(i).uniform(0.0, 1.0, (8, 8, 3)) for i in range(6)]
        for background in (SkyBox(textures), Sky()):
            renderer = Renderer(40, 20, background=background)
            default = renderer.render([], Camera(position=np.zeros(3), target=np.array([0.0, 5.0, 0.0])), Light()).copy()
            chosen = renderer.render([], Camera(position=np.zeros(3), target=np.array([0.0, 5.0, 0.0]),
                                                up=np.array([0.0, 0.0, -1.0])), Light())
            np.testing.assert_allclose(default.rgb, chosen.rgb, atol=1e-12)
            if isinstance(background, SkyBox):
                self.assertGreater(np.ptp(default.rgb[:, :, 0], axis=1).max(), 0.1)  # (rows aren't one colour)

    def test_objects_may_come_from_a_generator(self):
        boxes = [Object3D(make_box(), position=np.array([x, 0.0, 0.0])) for x in (-1.5, 1.5)]
        renderer = Renderer(40, 20)
        fb = renderer.render((box for box in boxes if box.visible), Camera(), Light())
        self.assertEqual(set(np.unique(fb.ids)), {0, 1, 2})
        self.assertIs(renderer.pick(*renderer.project(boxes[1].position)).object, boxes[1])

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

    def test_orthographic_camera(self):
        # Things the same size however far off, where an exact orthographic projection puts them.
        view = Camera(position=np.array([4.0, 3.0, 5.0]), target=np.zeros(3), projection="ortho", size=6.0)
        renderer = Renderer(80, 40, cell_pixels=(2, 3), outline=0.0, fog=0.0)
        near, far = (Object3D(make_box(), position=np.array(p)) for p in ((1.5, 0.0, 1.5), (-1.5, 0.0, -1.5)))
        fb = renderer.render([near, far], view, Light())
        self.assertAlmostEqual((fb.ids == 1).sum() / (fb.ids == 2).sum(), 1.0, delta=0.02)
        forward = -np.array([4.0, 3.0, 5.0]) / np.linalg.norm([4.0, 3.0, 5.0])
        right = np.cross(forward, (0.0, 1.0, 0.0))
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        aspect = 80 * renderer.cell_aspect / 40
        for point in ((0.0, 0.0, 0.0), (1.0, 2.0, -3.0), (-2.0, 0.5, 1.0)):
            offset = np.array(point) - view.position
            expected = ((offset @ right / (3.0 * aspect) + 1) * 40, (1 - offset @ up / 3.0) * 20)
            np.testing.assert_allclose(renderer.project(point), expected, atol=1e-4)  # (a thousandth of a pixel)
        self.assertIsNone(renderer.project((8.0, 6.0, 10.0)))  # behind the camera
        # Rays are parallel, from the camera's plane; pick() measures from that plane too.
        (o1, d1), (o2, d2) = renderer.ray(10, 10), renderer.ray(60, 30)
        np.testing.assert_allclose(d1, forward, atol=1e-12)
        np.testing.assert_allclose(d2, forward, atol=1e-12)
        self.assertAlmostEqual((o1 - view.position) @ forward, 0.0, places=9)
        np.testing.assert_allclose(o2 - o1, (50 / 40 * 3.0 * aspect) * right - (20 / 20 * 3.0) * up, atol=1e-9)
        x, y = renderer.project((1.5, 0.0, 2.0))  # the middle of the near box's front face
        hit = renderer.pick(x, y)
        self.assertIs(hit.object, near)
        np.testing.assert_allclose(hit.position, [1.5, 0.0, 2.0], atol=0.2)  # (somewhere in the cell)
        self.assertAlmostEqual(hit.distance, (hit.position - view.position) @ forward, places=6)
        self.assertAlmostEqual(np.abs(hit.position - near.position).max(), 0.5, delta=0.03)  # on its surface (to a pixel)
        # Clicking with a ray query finds what pick() does.
        self.assertIs(Colliders([near, far]).raycast(*renderer.ray(x, y)).object, near)
        # What is behind the camera, or nearer than `near`, isn't drawn.
        view.position = np.array([1.5, 0.0, 1.5]) + 0.2 * np.array([4.0, 3.0, 5.0]) / np.linalg.norm([4.0, 3.0, 5.0])
        view.target = view.position + forward
        fb = renderer.render([near, far], view, Light())
        self.assertFalse((fb.ids == 1).all())
        self.assertTrue((fb.ids == 2).any())

    def test_orthographic_fog_outlines_and_depth_cues(self):
        # Measured from the camera's plane: a box 3 from it is clear of fog that starts at 4, and one 9 off is
        # wholly in fog that ends at 7, whatever their distance from the eye the view is drawn from.
        boxes = [Object3D(make_box(), position=np.array([x, 0.0, -z]), color=Color.RED) for x, z in ((-1, 3), (1, 9))]
        view = Camera(position=np.zeros(3), target=np.array([0.0, 0.0, -1.0]), projection="ortho", size=4.0)
        fogged = Renderer(60, 30, cell_pixels=(2, 3), fog=Fog(start=3.6, end=7.0, color=(0, 0, 255)), outline=0.0)
        fb = fogged.render(boxes, view, Light())
        clear, deep = fb.rgb[fb.ids == 1], fb.rgb[fb.ids == 2]
        self.assertTrue((clear[:, 2] < 0.05).all())  # no blue in the red box near
        np.testing.assert_allclose(deep[deep[:, 0] > 0, 0], 0.0, atol=1e-9)  # only fog colour on the far one
        # Depth cueing dims the far box; outlines darken the edge where the near box passes in front of the far.
        cued = Renderer(60, 30, cell_pixels=(2, 3), fog=0.5, outline=0.0).render(boxes, view, Light())
        self.assertLess(cued.rgb[cued.ids == 2, 0].max(), 0.8 * cued.rgb[cued.ids == 1, 0].max())
        overlapping = [Object3D(make_box(), position=np.array([0.3, 0.0, -3.0])), boxes[1]]
        lined = Renderer(60, 30, cell_pixels=(2, 3), fog=0.0, outline=0.5).render(overlapping, view, Light()).copy()
        plain = Renderer(60, 30, cell_pixels=(2, 3), fog=0.0, outline=0.0).render(overlapping, view, Light())
        darker = (lined.rgb.sum(axis=2) < plain.rgb.sum(axis=2) - 1e-9)
        self.assertTrue(darker.any())
        self.assertTrue((lined.ids[darker] == 2).all())  # the far side of the edge

    def test_orthographic_camera_with_bad_numbers(self):
        renderer = Renderer(30, 15, cell_pixels=(2, 3))
        for size in (0.0, -3.0, np.nan, np.inf, 1e300, 1e-300):
            renderer.render([Object3D(make_box())], Camera(projection="ortho", size=size), Light(shadows=True))
            renderer.ray(5, 5), renderer.pick(15, 7), renderer.project((0.0, 0.0, 0.0))
        for camera in (Camera(projection="ortho", position=np.zeros(3), target=np.zeros(3)),
                       Camera(projection="ortho", near=np.nan), Camera(projection="ortho", far=-5.0)):
            renderer.render([Object3D(make_box())], camera, Light())
            renderer.ray(5, 5), renderer.pick(15, 7)
        # A view of negative size or angle (upside down): a point on a surface isn't hidden by that surface.
        for camera in (Camera(projection="ortho", size=-3.0), Camera(fov=-50.0)):
            renderer.render([Object3D(make_box())], camera, Light())
            anchor = renderer.anchor((0.0, 0.0, 0.5))
            self.assertEqual((anchor.x, anchor.hidden), (15, False), camera)
        with self.assertRaises(ValueError):
            Camera(projection="orthographic")

    def test_labels(self):
        for camera in (Camera(position=np.array([0.0, 1.0, 6.0])),
                       Camera(position=np.array([0.0, 1.0, 6.0]), target=np.array([0.0, 1.0, 0.0]), projection="ortho",
                              size=6.0)):
            screen = Screen(glyphs="quad", color="truecolor", size=(24, 60))
            renderer = Renderer(60, 22, screen.cell_pixels)
            self.assertIsNone(renderer.anchor((0.0, 0.0, 0.0)))  # before the first render
            wall = Object3D(make_box(), np.array([-1.5, 0.0, 1.0]), scale=(1.2, 3.0, 0.2))
            hidden, seen = Object3D(make_box(), np.array([-1.5, 0.0, -2.0])), Object3D(make_box(), np.array([1.5, 0, 0]))
            screen.draw_frame(renderer.render([wall, hidden, seen], camera, Light()), top=1)
            self.assertIsNone(screen.label(renderer, (-1.5, 0.0, -2.0), "x", top=1))  # behind the wall
            anchor = renderer.anchor((-1.5, 0.0, -2.0))
            self.assertTrue(anchor.hidden)
            self.assertFalse(anchor.edge)
            self.assertFalse(renderer.anchor((-1.5, 0.0, -2.0), owner=wall).hidden)  # the wall's own surface
            # Where project() puts it; text centred there, a row of the screen down for the frame's top.
            anchor = screen.label(renderer, (1.5, 0.9, 0.0), "ball", top=1, dy=-1)
            x, y = renderer.project((1.5, 0.9, 0.0))
            self.assertEqual((anchor.x, anchor.y), (int(x), int(y)))
            self.assertEqual("".join(screen.chars[anchor.y, anchor.x - 2:anchor.x + 2]), "ball")
            # On its own surface: hidden only where the owner isn't given, never by the surface it is on.
            self.assertFalse(renderer.anchor((1.5, 0.2, 0.5)).hidden)
            self.assertFalse(renderer.anchor((1.5, 0.2, -0.5), owner=seen).hidden)  # (its far side)
            self.assertTrue(renderer.anchor((1.5, 0.2, -0.5)).hidden)
            # Off the frame, or behind the camera: nothing, or at the edge with clamp, the whole text inside.
            self.assertIsNone(screen.label(renderer, (40.0, 1.0, 0.0), "far"))
            anchor = screen.label(renderer, (40.0, 1.0, 0.0), "far right", top=1, clamp=True)
            self.assertEqual((anchor.x, anchor.edge), (59, True))
            self.assertEqual("".join(screen.chars[anchor.y + 1, 51:60]), "far right")
            behind = renderer.anchor((-3.0, -5.0, 30.0), clamp=True)
            self.assertTrue(behind.edge)
            self.assertLess(behind.x, 30)  # to the left, and down
            self.assertEqual(behind.y, 21)
            for point in ((np.nan, 0.0, 0.0), (np.inf, 0.0, 0.0), (0.0, 1.0, 6.0)):  # (and at the camera)
                renderer.anchor(point, clamp=True)
                screen.label(renderer, point, "?", clamp=True)
        with self.assertRaises(ValueError):
            renderer.anchor((0.0, 0.0))

    def test_bars(self):
        screen = Screen(color="truecolor", size=(3, 10))
        for fraction, expected in ((0.0, "\u2591" * 4), (0.5, "\u2588\u2588\u2591\u2591"),
                                   (0.3, "\u2588\u258e\u2591\u2591"), (1.0, "\u2588" * 4), (7.0, "\u2588" * 4),
                                   (np.nan, "\u2591" * 4)):
            screen.erase()
            screen.bar(1, 2, 4, fraction)
            self.assertEqual("".join(screen.chars[1, 2:6]), expected, fraction)
            self.assertEqual("".join(screen.chars[1, 6:]), "    ")
        screen.unicode = False
        screen.bar(0, 0, 4, 0.6)
        self.assertEqual("".join(screen.chars[0, :4]), "##--")

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

    def test_world_fog(self):
        # Fog by distance in the world: clear up to start, gone at end, into the background or a colour.
        sky = (40, 80, 160)
        near = Object3D(make_box(), position=np.array([-1.0, 0.0, 2.0]), color=Color.RED)
        middle = Object3D(make_box(), position=np.array([0.5, 0.0, -2.0]), color=Color.RED)
        gone = Object3D(make_box(4.0), position=np.array([6.0, 2.0, -30.0]), color=Color.RED)
        renderer = Renderer(80, 30, fog=Fog(start=4.0, end=20.0), outline=0, background=sky)
        clear = Renderer(80, 30, fog=0, outline=0, background=sky)
        camera = Camera(position=np.array([0.0, 0.0, 5.0]), far=100.0)
        fb = renderer.render([near, middle, gone], camera, Light()).copy()
        plain = clear.render([near, middle, gone], camera, Light()).copy()
        bare = Renderer(80, 30, fog=0, outline=0).render([near, middle, gone], camera, Light())
        # Pixels wholly inside object i (with no background, so that coverage shows).
        inside = lambda i: (bare.ids == i) & (bare.alpha == 1) & (np.roll(bare.ids, 2, 1) == i)
        self.assertTrue(all(inside(i).sum() > 4 for i in (1, 2, 3)))
        np.testing.assert_allclose(fb.rgb[inside(1)], plain.rgb[inside(1)])  # nearer than start
        bg = to_linear_rgb(sky)
        np.testing.assert_allclose(fb.rgb[inside(3)], np.broadcast_to(bg, fb.rgb[inside(3)].shape),
                                   atol=1e-9)  # beyond end: only the sky
        # Between: part surface, part sky. Picking still finds what the fog hides.
        mid = fb.rgb[inside(2)]
        self.assertTrue(((mid[:, 2] > plain.rgb[inside(2)][:, 2]) & (mid[:, 2] < bg[2])).all())
        self.assertIs(renderer.pick(*renderer.project((6.0, 2.0, -30.0))).object, gone)
        # Unlike depth cueing, one object's fog doesn't change as others come into view or leave it.
        alone = renderer.render([near, middle], camera, Light())
        np.testing.assert_array_equal(alone.rgb[inside(2)], mid)
        # Into a colour of its own.
        renderer.fog = Fog(start=4.0, end=20.0, color=(255, 255, 0))
        fb = renderer.render([near, middle, gone], camera, Light())
        np.testing.assert_allclose(fb.rgb[inside(3)][0], to_linear_rgb((255, 255, 0)), atol=1e-9)
        # With no background, fogged surfaces give way to the terminal's own.
        renderer.background = None
        renderer.fog = Fog(start=4.0, end=20.0)
        fb = renderer.render([near, middle, gone], camera, Light())
        self.assertEqual(fb.alpha[inside(3)].max(), 0.0)
        self.assertTrue((fb.alpha[inside(2)] < 1.0).all() and (fb.alpha[inside(2)] > 0.0).all())

    def test_fog_hides_a_sky_box_detail(self):
        # Fogged into a starry sky box, a wall fades into the sky blurred, not into the stars right behind it: they
        # used to show through the fog as if the wall were glass.
        rng = np.random.default_rng(2)
        dark = [np.full((64, 64, 3), 0.05) for _ in range(6)]
        starry = [face.copy() for face in dark]
        for face in starry:
            face[rng.integers(0, 64, 60), rng.integers(0, 64, 60)] = 1.0  # stars: single bright texels
        wall = Object3D(flat_quad(40.0, 20.0), position=np.array([0.0, 0.0, -8.0]), color=(120, 120, 120))
        camera, light = Camera(position=np.array([0.0, 0.0, 5.0])), Light(direction=np.array([0.0, 0.0, -1.0]))

        def draw(faces, fog):
            return Renderer(80, 30, fog=fog, outline=0, background=SkyBox(faces)).render([wall], camera, light).copy()

        fog = Fog(start=2.0, end=20.0)
        fb = draw(starry, fog)
        self.assertTrue((fb.ids == 1).all() and (fb.alpha == 1).all())
        self.assertLess(np.abs(lum(fb) - lum(draw(dark, fog))).max(), 0.02)  # no stars on it
        self.assertLess(lum(fb).mean(), lum(draw(starry, 0)).mean())       # but fogged, towards the dark sky

    def test_materials(self):
        # Object3D.specular scales each light's highlight; shininess sets how tight it is.
        ball = blob_mesh((1.0, 1.0, 1.0), rings=32, segments=48)
        light = Light(direction=np.array([0.0, 0.0, -1.0]), ambient=0.2, diffuse=0.4, specular=0.5, shininess=20.0)
        renderer = Renderer(60, 30, fog=0, outline=0)
        camera = Camera(position=np.array([0.0, 0.0, 5.0]))
        shots = {}
        for name, kwargs in (("plain", {}), ("matte", {"specular": 0.0}), ("strong", {"specular": 2.0}),
                             ("tight", {"shininess": 200.0}), ("light's", {"shininess": 20.0})):
            shots[name] = lum(renderer.render([Object3D(ball, color=(80, 80, 200), **kwargs)], camera, light).copy())
        centre = (30, 30)
        self.assertLess(shots["matte"][centre], shots["plain"][centre])
        self.assertGreater(shots["strong"][centre], shots["plain"][centre])
        np.testing.assert_array_equal(shots["light's"], shots["plain"])  # the same exponent as the light's
        # A tighter highlight: about as bright at its peak, over far fewer pixels.
        self.assertAlmostEqual(shots["tight"].max(), shots["plain"].max(), delta=0.1)
        lit = lambda shot: int((shot > shots["matte"] + 0.05).sum())
        self.assertLess(3 * lit(shots["tight"]), lit(shots["plain"]))

    def test_uneven_scale(self):
        # scale=(x, y, z) stretches along the object's own axes; normals stay square to the stretched surface.
        light = Light(direction=np.array([0.0, -1.0, -1.0]), ambient=0.2, diffuse=0.8, specular=0.0)
        renderer = Renderer(80, 40, fog=0, outline=0, samples=1)
        camera = Camera(position=np.array([0.0, 0.0, 6.0]))
        wide = renderer.render([Object3D(make_box(), scale=(2.0, 1.0, 1.0))], camera, light).copy()
        cube = renderer.render([Object3D(make_box())], camera, light).copy()
        rows, cols = np.nonzero(wide.drawn)
        rows1, cols1 = np.nonzero(cube.drawn)
        self.assertAlmostEqual(np.ptp(cols) / np.ptp(cols1), 2.0, delta=0.1)
        self.assertAlmostEqual(np.ptp(rows) / np.ptp(rows1), 1.0, delta=0.1)
        # A slab tilted towards the light is lit as its face, not as its corners' stretched normals would be.
        turn = quat_axis_angle((1, 0, 0), 0.6)
        slab = renderer.render([Object3D(make_box(), rotation=turn, scale=(1.5, 0.2, 1.5))], camera, light).copy()
        plate = Mesh(np.array([(-0.75, 0.1, 0.75), (0.75, 0.1, 0.75), (0.75, 0.1, -0.75), (-0.75, 0.1, -0.75)]),
                     np.array([(0, 1, 2), (0, 2, 3)]))
        top = renderer.render([Object3D(plate, rotation=turn)], camera, light).copy()
        face = (top.alpha == 1) & (slab.alpha == 1) & (slab.ids == 1)
        self.assertGreater(face.sum(), 100)
        np.testing.assert_allclose(lum(slab)[face].mean(), lum(top)[face].mean(), rtol=0.02)
        # A negative scale mirrors the shape, which is still drawn from outside (not culled as inside out).
        mirrored = renderer.render([Object3D(make_box(), scale=(-1.0, 1.0, 1.0))], camera, light)
        np.testing.assert_allclose(mirrored.alpha, cube.alpha)
        np.testing.assert_allclose(lum(mirrored), lum(cube), atol=1e-9)
        # Through a parent: a node stretched along y with a turned child shears it, as world_matrix() says.
        group = Node(position=np.array([0.5, 0.0, 0.0]), scale=(1.0, 3.0, 1.0))
        child = Object3D(make_box(), position=np.array([0.0, 0.2, 0.0]), rotation=quat_axis_angle((0, 0, 1), 0.7),
                         scale=0.5, parent=group)
        linear, position, visible = child.world_matrix()
        corner = np.array([0.5, 0.5, 0.5])
        expected = group.position + np.diag([1.0, 3.0, 1.0]) @ (child.position + quat_to_matrix(child.rotation)
                                                               @ (0.5 * corner))
        np.testing.assert_allclose(child.to_world(corner), expected)
        np.testing.assert_allclose(linear @ corner + position, expected)
        self.assertTrue(visible)
        with self.assertRaises(ValueError):
            Renderer(10, 5).render([Object3D(make_box(), scale=(1.0, 2.0))], camera, light)

    def test_texture_v_runs_up_the_image(self):
        # As the README says: v = 1 shows the image's row 0 (its top), v = 0 its last row; u runs left to right.
        quad = Mesh(np.array([(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)], float), np.array([(0, 1, 2), (0, 2, 3)]))
        quad.uvs = np.array([(0, 0), (1, 0), (1, 1), (0, 1)], float)[quad.faces]
        quad.materials = np.zeros(2, int)
        image = np.zeros((8, 8, 3))
        image[:4, :, 0] = 1.0  # top half red
        image[:, 4:, 2] = 1.0  # right half blue
        quad.textures = [image]
        fb = Renderer(40, 20, fog=0, outline=0).render([Object3D(quad, color=(255, 255, 255))],
                                                       Camera(position=np.array([0.0, 0.0, 3.0])),
                                                       Light(ambient=1.0, diffuse=0.0, specular=0.0))
        rows, cols = np.nonzero(fb.alpha == 1)
        top, bottom, left, right = rows.min() + 2, rows.max() - 2, cols.min() + 2, cols.max() - 2
        c = fb.colour()
        np.testing.assert_allclose(c[top, left], [1, 0, 0], atol=0.02)
        np.testing.assert_allclose(c[top, right], [1, 0, 1], atol=0.02)
        np.testing.assert_allclose(c[bottom, left], [0, 0, 0], atol=0.02)
        np.testing.assert_allclose(c[bottom, right], [0, 0, 1], atol=0.02)

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


class AnimationTests(unittest.TestCase):
    def test_slerp_turns_steadily_the_short_way(self):
        a, b = quat_axis_angle((0, 1, 0), 0.2), quat_axis_angle((0, 1, 0), 1.4)
        np.testing.assert_allclose(quat_slerp(a, b, 0.0), a, atol=1e-12)
        np.testing.assert_allclose(quat_slerp(a, b, 1.0), b, atol=1e-12)
        np.testing.assert_allclose(quat_slerp(a, b, 0.25), quat_axis_angle((0, 1, 0), 0.5), atol=1e-12)
        # -b is the same rotation as b: still the short way (0.6 rad), not the long way round.
        np.testing.assert_allclose(quat_to_matrix(quat_slerp(a, -b, 0.5)), quat_to_matrix(quat_axis_angle((0, 1, 0), 0.8)),
                                   atol=1e-12)
        near = quat_axis_angle((1, 0, 0), 1e-6)  # nearly equal: no division by ~0
        self.assertTrue(np.isfinite(quat_slerp(quat_identity(), near, 0.5)).all())

    def test_easings_run_from_0_to_1(self):
        for name, ease in EASINGS.items():
            self.assertAlmostEqual(ease(0.0), 0.0, msg=name)
            self.assertAlmostEqual(ease(1.0), 1.0, msg=name)
        self.assertLess(ease_in(0.5), 0.5)
        self.assertGreater(ease_out(0.5), 0.5)
        self.assertGreater(max(ease_out_back(t) for t in np.linspace(0, 1, 50)), 1.0)  # overshoots

    def test_tracks(self):
        track = Track([(2.0, (0.0, 0.0, 0.0)), (0.0, (0.0, 4.0, 0.0)), (3.0, (1.0, 0.0, 0.0), "ease_in")])
        np.testing.assert_allclose(track.at(-1.0), (0, 4, 0))  # held before the first key
        np.testing.assert_allclose(track.at(1.0), (0, 2, 0))   # keys sorted by time; linear by default
        np.testing.assert_allclose(track.at(2.5), (ease_in(0.5), 0, 0))  # the key's own easing, leading up to it
        np.testing.assert_allclose(track.at(9.0), (1, 0, 0))  # held after the last
        looped = Track([(0.0, 0.0), (1.0, 10.0)], loop="loop")
        pingpong = Track([(0.0, 0.0), (1.0, 10.0)], loop="pingpong")
        self.assertAlmostEqual(looped.at(1.25), 2.5)
        self.assertAlmostEqual(pingpong.at(1.25), 7.5)
        self.assertAlmostEqual(pingpong.at(-0.25), 2.5)
        with self.assertRaises(ValueError):
            Track([])
        turn = RotationTrack([(0.0, quat_identity()), (1.0, quat_axis_angle((0, 0, 1), 1.0))])
        np.testing.assert_allclose(turn.at(0.5), quat_axis_angle((0, 0, 1), 0.5), atol=1e-12)

    def test_animation_moves_an_object(self):
        box = Object3D(make_box())
        anim = Animation(box, position=Track([(0.0, (0, 0, 0)), (2.0, (4, 0, 0))]),
                         rotation=RotationTrack([(0.0, quat_identity()), (1.0, quat_axis_angle((0, 1, 0), 1.0))]),
                         scale=Track([(0.0, 1.0), (2.0, (1.0, 3.0, 1.0))]))
        self.assertTrue(anim.update(0.5))
        np.testing.assert_allclose(box.position, (1, 0, 0))
        np.testing.assert_allclose(box.rotation, quat_axis_angle((0, 1, 0), 0.5), atol=1e-12)
        np.testing.assert_allclose(box.scale, (1.0, 1.5, 1.0))
        self.assertFalse(anim.update(2.0))  # ended at 2 s, and stays at its last pose
        np.testing.assert_allclose(box.position, (4, 0, 0))
        fb = Renderer(40, 20).render([box], Camera(position=np.array([2.0, 0.0, 8.0]), target=np.array([2.0, 0, 0])),
                                     Light())
        self.assertTrue(fb.drawn.any())  # a scale from a track (an array) renders

    @staticmethod
    def _twin_clips(specs, starts, loop="once", kernel=True):
        """Two clips playing the same tracks (specs: one {attribute: track} per animation) on targets posed alike
        (starts: (position, rotation, scale) each), the second through Track.at (no kernel)."""
        class Target:
            pass
        clips, targets = [], []
        for use_kernel in (kernel, False):
            ts = []
            for position, rotation, scale in starts:
                t = Target()
                t.position, t.rotation = np.array(position, float), np.array(rotation, float)
                t.scale = scale.copy() if isinstance(scale, np.ndarray) else scale
                ts.append(t)
            clip = Clip([Animation(t, **tracks) for t, tracks in zip(ts, specs)], loop=loop, speed=1.3)
            clip._kernel = use_kernel
            clips.append(clip)
            targets.append(ts)
        return clips, targets

    def assertSamePoses(self, a, b):
        for x, y in zip(a, b):
            for name in ("position", "rotation", "scale"):
                u, v = getattr(x, name), getattr(y, name)
                self.assertEqual(np.shape(u), np.shape(v), name)
                self.assertEqual(isinstance(u, np.ndarray), isinstance(v, np.ndarray), name)
                np.testing.assert_array_equal(u, v, name)

    def test_clips_sample_their_tracks_in_a_kernel_exactly_as_tracks_do(self):
        # animation._sample against Track.at, value for value: every kind of track, loop and easing, repeated key
        # times, a single key, fades; the same arithmetic in the same order, so the same numbers.
        rng = np.random.default_rng(7)
        names = list(EASINGS)

        def track(kind):
            times = np.sort(rng.uniform(-1, 3, int(rng.integers(1, 6))))
            if len(times) > 2:
                times[1] = times[0]
            loop = LOOPS[rng.integers(3)]
            if kind == 0:
                return "scale", Track([(t, float(rng.normal()), rng.choice(names)) for t in times], loop=loop)
            if kind == 1:
                return "position", Track([(t, rng.normal(size=3), rng.choice(names)) for t in times], loop=loop,
                                         easing=rng.choice(names))
            if kind == 2:
                return "rotation", RotationTrack([(t, rng.normal(size=4), rng.choice(names)) for t in times], loop=loop)
            if kind == 3:
                return "position", SplineTrack([(t, *rng.normal(size=(3, 3))) for t in times], loop=loop)
            if kind == 4:
                return "scale", SplineTrack([(t, *rng.normal(size=3).tolist()) for t in times], loop=loop)
            return "rotation", SplineTrack([(t, normalize(rng.normal(size=4)), *rng.normal(size=(2, 4))) for t in times],
                                           loop=loop, rotation=True)

        for trial in range(60):
            specs = [dict(track(k) for k in rng.choice(6, size=int(rng.integers(1, 4)), replace=False))
                     for _ in range(int(rng.integers(1, 5)))]
            starts = [(rng.normal(size=3), normalize(rng.normal(size=4)), 1.5) for _ in specs]
            clips, targets = self._twin_clips(specs, starts, loop=LOOPS[trial % 3])
            for clip in clips:
                clip.start(fade=0.4 if trial % 2 else 0.0)
            self.assertTrue(clips[0]._packed.ok)
            for _ in range(25):
                dt = float(rng.uniform(0, 0.2))
                self.assertEqual(clips[0].update(dt), clips[1].update(dt))
                self.assertSamePoses(*targets)
                self.assertEqual([a.time for a in clips[0].animations], [a.time for a in clips[1].animations])

    def test_clips_the_kernel_cant_play_play_as_before(self):
        # Easings and Track subclasses of a program's own, values of more than one shape, two tracks moving one
        # attribute: Track.at plays the clip. A fade from a pose of another shape (a scale of three numbers into a
        # track of one) blends in Python too.
        class Doubled(Track):
            def blend(self, a, b, t):
                return 2.0 * super().blend(a, b, t)

        pose = [((0.0, 0.0, 0.0), quat_identity(), np.array([1.0, 2.0, 3.0]))]
        cases = [{"scale": Track([(0.0, 1.0), (1.0, 2.0, lambda t: t * t)])},
                 {"position": Doubled([(0.0, (0.0, 0.0, 0.0)), (1.0, (1.0, 2.0, 3.0))])},
                 {"scale": Track([(0.0, 1.0), (2.0, (1.0, 3.0, 1.0))])}]
        for specs in [[c] for c in cases]:
            clips, targets = self._twin_clips(specs, pose * len(specs))
            for _ in range(5):
                [clip.update(0.3) for clip in clips]
                self.assertSamePoses(*targets)
            self.assertFalse(clips[0]._packed.ok)
        box = Object3D(make_box())
        twice = Clip([Animation(box, scale=Track([(0.0, 1.0), (1.0, 2.0)])),
                      Animation(box, scale=Track([(0.0, 5.0), (1.0, 7.0)]))])
        twice.apply(0.5)
        self.assertEqual(box.scale, 6.0)  # the last one, as before
        self.assertFalse(twice._packed.ok)
        clips, targets = self._twin_clips([{"scale": Track([(0.0, 1.0), (1.0, 2.0)])}], pose)
        [clip.start(fade=0.5) for clip in clips]
        for _ in range(4):
            [clip.update(0.2) for clip in clips]
            self.assertSamePoses(*targets)
        self.assertTrue(clips[0]._packed.ok)

    def test_a_clip_follows_changes_to_its_animations(self):
        box, ball = Object3D(make_box()), Object3D(make_box())
        clip = Clip([Animation(box, position=Track([(0.0, (0.0, 0.0, 0.0)), (1.0, (2.0, 0.0, 0.0))]))])
        clip.apply(0.5)
        np.testing.assert_allclose(box.position, (1, 0, 0))
        clip.animations[0].tracks["position"] = Track([(0.0, (0.0, 0.0, 0.0)), (1.0, (0.0, 4.0, 0.0))])
        clip.apply(0.5)
        np.testing.assert_allclose(box.position, (0, 2, 0))  # a track replaced
        clip.animations[0].target = ball
        clip.apply(0.25)
        np.testing.assert_allclose(ball.position, (0, 1, 0))  # another target
        clip.animations.append(Animation(box, scale=Track([(0.0, 1.0), (2.0, 3.0)])))
        self.assertEqual(clip.duration, 2.0)  # an animation added
        clip.apply(1.0)
        self.assertEqual(box.scale, 2.0)
        self.assertIsInstance(box.scale, float)
        before, kept = ball.position, ball.position.copy()
        two = Clip([Animation(box, position=Track([(0.0, (0.0, 0.0, 0.0)), (1.0, (1.0, 1.0, 1.0))])),
                    Animation(ball, position=Track([(0.0, (0.0, 0.0, 0.0)), (1.0, (1.0, 1.0, 1.0))]))])
        two.apply(0.5)
        box.position[0] = 9.0  # each target its own array, not one shared or reused
        np.testing.assert_allclose(ball.position, (0.5, 0.5, 0.5))
        np.testing.assert_array_equal(before, kept)  # last frame's array left as it was


class KernelCacheTests(unittest.TestCase):
    def test_kernel_signatures_are_current(self):
        # precompile.py compiles the kernels listed in kernel_signatures.py on several cores before the first
        # frame; a kernel missing from the list, or listed with other argument types, is compiled one at a time
        # again. After changing a kernel's arguments or adding one: python -m unicode3d.precompile --update
        compile_kernels()  # into the cache, so that the run below only loads them
        out = subprocess.run([sys.executable, "-m", "unicode3d.precompile", "--record"], capture_output=True,
                             text=True, check=True, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        used = {tuple(line.split("\t")[i] for i in (0, 1, 3)) for line in out.stdout.splitlines()}
        listed = {(module, name, text) for module, name, _, texts in KERNELS for text in texts}
        self.assertEqual(used, listed, "run: python -m unicode3d.precompile --update")

    def test_cache_can_be_checked(self):
        # precompile.missing_kernels() looks into Numba's cache; None means it no longer can (a Numba that keeps
        # its cache differently), and then every first run compiles one kernel at a time.
        compile_kernels()
        self.assertEqual(missing_kernels(), [])

    def test_work_is_shared_out_evenly(self):
        kernels = [("m", str(i), s, []) for i, s in enumerate((4.0, 3.0, 3.0, 2.0, 2.0, 1.0, 1.0))]
        groups = share_out(kernels, 3)
        self.assertEqual(sorted(k for g in groups for k in g), sorted(kernels))
        self.assertEqual(sorted(sum(k[2] for k in g) for g in groups), [5.0, 5.0, 6.0])
        self.assertEqual(len(share_out(kernels[:2], 5)), 2)


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
        wide = Screen(glyphs="quad", color="256", size=(4, 120))
        controls.draw(wide, 3, 0)
        self.assertIn("F7 high detail", "".join(wide.chars[3]))  # and F7 its levels of detail
        controls.handle([Key.F7])
        self.assertEqual(renderer.simplify, 1.0)
        controls.draw(wide, 3, 0)
        self.assertIn("F7 std detail", "".join(wide.chars[3]))
        controls.handle([Key.F7])
        self.assertEqual(renderer.simplify, 0.0)


    def test_display_controls_show_some_and_keep_their_settings(self):
        renderer = Renderer(10, 5)
        controls = DisplayControls(renderer=renderer, show=("fps",))
        self.assertEqual(controls.width, len("F4 999/144fps"))
        controls.draw(self.screen, 3, 0)
        self.assertEqual("".join(self.screen.chars[3]).strip(), "F4 --/30fps")
        controls.handle([Key.F2, Key.F5])  # the hidden ones still answer their keys
        self.assertEqual((self.screen.mode, renderer.shadows), ("sextant", False))
        self.assertEqual(controls.settings(), {"glyphs": "sextant", "color": "truecolor", "fps": 30,
                                               "shadows": False, "reflections": True, "detail": "high", "quality": "high"})
        # Settings handed back, as from a file: before the screen is known they wait for it; names and values it
        # doesn't know are skipped.
        later = DisplayControls(renderer=Renderer(10, 5))
        later.apply({"glyphs": "half", "fps": 60, "shadows": False, "color": "rainbow", "volume": 11,
                     "reflections": 1, "detail": "standard"})
        self.assertEqual(later.settings(), {"glyphs": "half", "color": "truecolor", "fps": 60, "shadows": False,
                                            "reflections": True, "detail": "standard", "quality": "high"})
        self.assertEqual(later.renderer.simplify, 1.0)
        screen = Screen(glyphs="quad", color="256", size=(4, 80))
        later.handle([], screen)
        self.assertEqual((screen.mode, screen.color_mode, screen.fps), ("half", "256", 60))
        plain = Screen(glyphs="ascii", color="16", size=(4, 80))
        plain.unicode = False  # (as on a terminal without it)
        again = DisplayControls()
        again.apply({"glyphs": "sextant", "color": "truecolor"})
        again.handle([], plain)
        self.assertEqual((plain.mode, plain.color_mode), ("ascii", "truecolor"))
        with self.assertRaises(ValueError):
            DisplayControls(show=("frame rate",))


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
        demo = DiceDemo(3, seed=1)
        self.run_demo(demo, [ord(" "), ord("+"), Key.F3, ord("g"), ord("m"), Key.F6, ord("f"), ord("f"), Key.F7],
                      ord("q"))
        self.assertEqual(demo.renderer.simplify, 1.0)  # F7: standard detail
        self.assertEqual(demo.finish, "chrome")  # f went plastic, rubber, chrome
        self.assertTrue(all(die.reflectivity == 0.7 and die.shininess == 40.0 for die in demo.dice))

    def test_viewer(self):
        from unicode3d.examples.viewer import Viewer
        root = Node()
        model = Model(root, [Object3D(make_die(), parent=root), Object3D(make_box(), parent=root, reflectivity=0.3)])
        lift = Animation(model.objects[1], position=Track([(0.0, (0.0, 0.0, 0.0)), (1.0, (0.0, 1.0, 0.0))]))
        model.animations = {"lift": Clip([lift], name="lift")}
        viewer = Viewer(model)
        screen = self.run_demo(viewer, [ord("w"), Key.LEFT, ord("e"), ord("c"), ord("o")], Key.ESC)
        self.assertEqual(model.objects[1].reflectivity, 0.85)
        self.assertGreater(model.objects[1].position[1], 0.0)  # playing, looped
        wide = Screen(glyphs="quad", color="256", size=(30, 240))
        viewer.frame(wide, 0.1, [])
        self.assertIn("animation: lift (1/1)", "".join(wide.chars[-1]))
        self.assertIn("[o] ortho", "".join(wide.chars[-1]))
        self.assertEqual(viewer.camera.projection, "ortho")
        viewer.frame(screen, 0.1, [ord("n")])
        self.assertIsNone(viewer.playing)

    def test_balls(self):
        from unicode3d.examples.balls import Balls
        demo = Balls(30, seed=1)
        self.run_demo(demo, [ord("]"), ord("g"), ord(" "), Key.TAB, Key.RIGHT, MouseEvent(50, 12, 0, True), Key.F5,
                             ord("x"), ord("m")], ord("q"))
        self.assertEqual(len(demo.balls), 32)  # "]" and then the focused slider's Right added one each

    def test_balls_squash_on_bumps(self):
        from unicode3d.examples.balls import Balls, FINISHES, MAX_SQUASH
        demo = Balls(2, seed=1)
        demo.collide.value, demo.mixed.value = False, True
        screen = Screen(glyphs="quad", color="256", size=(30, 100))
        demo.frame(screen, 1 / 30, [])
        half = demo.room_size.value / 2
        demo.pos[:] = [(half - 0.5, 0.0, 0.0), (0.0, 0.0, 0.0)]  # the first about to hit the +x wall, head on
        demo.vel[:] = [(1.0, 0.0, 0.0), (0.0, 0.0, 1e-3)]
        for _ in range(10):
            demo.frame(screen, 1 / 30, [])
            if demo.squash_amount[0]:
                break
        self.assertGreater(demo.squash_amount[0], 0.1)
        self.assertLessEqual(demo.squash_amount[0], MAX_SQUASH)
        np.testing.assert_allclose(np.abs(demo.squash_dir[0]), (1, 0, 0), atol=1e-9)  # along the wall's normal
        self.assertEqual(demo.squash_amount[1], 0.0)  # the other drifted, unbumped
        ball = demo.balls[0]
        self.assertEqual(np.shape(ball.scale), (3,))  # flattened along x (turned onto it) and bulging the other ways
        world = np.abs(ball.world_matrix()[0])
        self.assertLess(world[0, 0], world[1, 1])
        self.assertEqual(demo.balls[1].scale, demo.size.value * demo.radius_factor[1])  # round
        self.assertEqual({(b.specular, b.shininess, b.reflectivity) for b in demo.balls} <= set(FINISHES), True)
        for _ in range(40):  # wobbles back to round
            demo.frame(screen, 1 / 30, [])
        self.assertEqual(np.ndim(ball.scale), 0)

    def test_maze(self):
        from unicode3d.examples.maze import Maze
        demo = Maze(5, seed=1)
        self.run_demo(demo, [ord("t"), ord("m"), ord("r"), ord("n"), ord("l"), ord("p"), ord("f")], ord("q"))
        self.assertEqual(demo.renderer.fog.color, (60, 62, 66))  # haze, with the headlamp off (l)
        heights = set()
        for _ in range(10):  # the gems bob
            demo.frame(Screen(glyphs="quad", color="256", size=(30, 100)), 0.1, [])
            heights.add(round(demo.start_marker.position[1], 6))
        self.assertGreater(len(heights), 5)
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
        # Standing by the door opens it, and it shuts again once the walker has gone.
        door, screen = demo.court.door, Screen(glyphs="quad", color="256", size=(30, 100))  # no keys held
        demo.x, demo.z = 1.4, 3.0
        for _ in range(15):
            demo.frame(screen, 0.1, [])
        self.assertTrue(door.opening)
        self.assertGreater(abs(door.hinge.rotation[2]), 0.5)  # turned about y, well open
        demo.x, demo.z = -6.0, 8.0
        for _ in range(20):
            demo.frame(screen, 0.1, [])
        np.testing.assert_allclose(door.hinge.rotation, (1, 0, 0, 0), atol=1e-9)
        demo.fog_into.value = "mist"
        self.assertIsNot(demo.frame(screen, 0.1, []), False)
        self.assertEqual(demo.renderer.fog.color, (225, 228, 232))
        # Each name just above the top of its thing (all of the objects of that name near each other), however
        # long or flat the thing is; one for each of four pillars.
        tags = demo.court.name_tags(np.array([0.0, 1.6, 0.0]), 100.0)
        for name, point, objects in tags:
            top = max((o.mesh.vertices @ o.world_matrix()[0].T + o.world_matrix()[1])[:, 1].max() for o in objects)
            self.assertAlmostEqual(point[1] - top, 0.15, msg=name)
        self.assertEqual(sum(name == "a pillar" for name, _, _ in tags), 4)
        self.assertEqual(sum(name == "a lacquered table" for name, _, _ in tags), 1)
        # Names over the things nearby: the door ahead, from the start.
        demo.x, demo.z, demo.tags.value = 1.4, 5.5, True
        demo.frame(screen, 0.1, [])
        self.assertIn("a door (to nowhere)", "\n".join("".join(row) for row in screen.chars))
        # Walking into the pedestal stops at it (Colliders), sliding along nothing when square on.
        demo.x, demo.z, demo.yaw = -4.0, -1.7, 0.0  # (clear of the trellis, facing north)
        screen.held.exact = True
        screen.held.update([ord("w")], 0.0)
        for _ in range(30):
            demo.frame(screen, 0.1, [])
        self.assertAlmostEqual(demo.z, -3.0 + 0.6 + 0.35, places=6)
        self.assertAlmostEqual(demo.x, -4.0, places=6)

    def test_tactics(self):
        from unicode3d.examples.tactics import HOP, N, Tactics, tile_centre, tile_top
        demo = Tactics(seed=3)
        screen = self.run_demo(demo, [ord("e"), ord("+"), ord("-"), ord("o"), ord("o"), ord("w"), Key.TAB,
                                      MouseEvent(50, 12, MouseEvent.WHEEL_UP, True), MouseEvent(40, 10, 3, False, True)],
                               Key.ESC)
        self.assertEqual(demo.camera.projection, "ortho")
        self.assertIs(demo.selected, demo.pawns[1])  # Tab
        self.assertGreater(demo.yaw, np.pi / 2)  # turning a quarter, towards 3/4 pi
        self.assertIn("[o] to perspective", "".join(screen.chars[-1]))
        self.assertIn("orthographic view   [o] switch to perspective", "".join(screen.chars[0]))
        demo.frame(screen, 1 / 30, [ord("o")])
        self.assertEqual(demo.camera.projection, "perspective")
        self.assertIn("[o] to orthographic", "".join(screen.chars[-1]))
        self.assertIn("perspective view   [o] switch to orthographic", "".join(screen.chars[0]))
        # Click a pawn to pick it, then a tile it can reach, found by the ray through that cell: it hops there.
        demo.frame(screen, 1 / 30, [ord("o")])
        pawn = demo.pawns[2]
        x, y = demo.renderer.project(pawn.obj.position + (0.0, 0.3, 0.0))
        demo.frame(screen, 1 / 30, [MouseEvent(int(x), int(y), MouseEvent.LEFT, True)])
        self.assertIs(demo.selected, pawn)
        for tile in sorted(((i, j) for i in range(N) for j in range(N)), key=lambda t: abs(t[0] - pawn.tile[0])
                           + abs(t[1] - pawn.tile[1]))[3:]:
            path = demo.path_to(pawn, tile)
            if not path:
                continue
            cx, cz = tile_centre(*tile)
            x, y = demo.renderer.project((cx, tile_top(demo.levels[tile]), cz))
            if demo.tile_at(int(x), int(y)) == (None, tile):
                break
        demo.frame(screen, 1 / 30, [MouseEvent(int(x), int(y), MouseEvent.LEFT, True)])
        self.assertEqual(pawn.path, path)
        self.assertTrue(all(abs(demo.levels[a] - demo.levels[b]) <= 1 for a, b in zip([pawn.tile] + path, path)))
        for _ in range(int(len(path) * HOP / 0.05) + 3):
            demo.frame(screen, 0.05, [])
        self.assertEqual(pawn.tile, tile)
        self.assertEqual(pawn.path, [])
        # Zoomed in on a corner, pawns out of view are pointed at from the screen's edge.
        demo.zoom, demo.pan = 4.0, np.array([N / 2, 0.0, N / 2])
        demo.frame(screen, 1 / 30, [])
        text = "\n".join("".join(row) for row in screen.chars)
        self.assertTrue(any(ch in text for ch in "\u25c0\u25b6\u25b2\u25bc"))

    def test_workshop(self):
        from unicode3d.examples.workshop import Workshop
        demo = Workshop()
        self.run_demo(demo, [ord("o"), ord("c"), ord("f"), ord("b"), ord("e"), Key.TAB, Key.TAB, Key.TAB, Key.RIGHT,
                             ord("h"), ord("d"), ord("z")], ord("q"))
        self.assertEqual(demo.shape.value, "cube")
        self.assertEqual(demo.specular.value, 1.1)  # Tab to the third widget, the specular slider, then Right
        self.assertIsNotNone(demo.hopping)
        # The panel's settings reach the object and the renderer.
        demo.sx.value, demo.sy.value, demo.shininess.value, demo.fog.value = -1.5, 0.5, 80, "mist"
        screen = Screen(glyphs="quad", color="256", size=(30, 100))
        for _ in range(60):  # past the end of the hop
            demo.frame(screen, 1 / 30, [])
        self.assertIsNone(demo.hopping)
        np.testing.assert_allclose(demo.obj.scale, (-1.5, 0.5, 1.0))
        self.assertEqual(demo.obj.shininess, 80)
        self.assertEqual(demo.renderer.fog.color, (225, 228, 232))
        demo.frame(screen, 1 / 30, [ord("r")])
        self.assertEqual((demo.shape.value, demo.sx.value, demo.fog.value), ("sphere", 1.0, "off"))


class Terminal:
    """What a terminal shows after the escape sequences Screen sends: characters, packed colours (as the Screen's grid
    holds them) and attributes. Knows only the sequences Screen uses."""

    def __init__(self, rows, cols):
        self.chars = np.full((rows, cols), " ", dtype="<U1")
        self.fg = np.full((rows, cols), -1, np.int64)
        self.bg = np.full((rows, cols), -1, np.int64)
        self.attrs = np.zeros((rows, cols), np.uint8)
        self.y = self.x = 0
        self.style = [-1, -1, 0]  # fg, bg, attributes

    def feed(self, text):
        for m in re.finditer(r"\x1b\[(\??)([0-9;]*)([A-Za-z])|(.)", text, re.S):
            if m.group(4) is not None:
                y, x = self.y, self.x
                self.chars[y, x], (self.fg[y, x], self.bg[y, x], self.attrs[y, x]) = m.group(4), self.style
                self.x += 1
            elif m.group(1):
                continue  # (synchronized output)
            elif m.group(3) == "H":
                self.y, self.x = (int(v) - 1 for v in m.group(2).split(";"))
            elif m.group(3) == "J":
                self.chars[:], self.fg[:], self.bg[:], self.attrs[:] = " ", self.style[0], self.style[1], 0
            elif m.group(3) == "m":
                codes = [int(v) for v in m.group(2).split(";")] if m.group(2) else [0]
                while codes:
                    n = codes.pop(0)
                    if n == 0:
                        self.style = [-1, -1, 0]
                    elif n in (1, 2, 7):
                        self.style[2] |= {1: 1, 2: 2, 7: 4}[n]
                    elif n in (38, 48):
                        kind = codes.pop(0)
                        color = (int(encode_rgb([codes.pop(0), codes.pop(0), codes.pop(0)])) if kind == 2
                                 else int(encode_index(codes.pop(0))))
                        self.style[n == 48] = color
                    elif n in (39, 49):
                        self.style[n == 49] = -1
                    else:  # 30-37, 40-47, 90-97, 100-107
                        background, n = (n >= 40 and n < 90) or n >= 100, n % 10 + (8 if n >= 90 else 0)
                        self.style[background] = int(encode_index(n))

    def levels_off(self, screen):
        """How many levels the terminal's colours are off the screen's grid at most; fails if a character or
        attribute differs."""
        np.testing.assert_array_equal(self.chars, screen.chars)
        np.testing.assert_array_equal(self.attrs, screen.attrs)
        rgb = lambda p: np.stack([p >> 16 & 255, p >> 8 & 255, p & 255], -1)
        return int(max(np.abs(rgb(self.fg) - rgb(screen.fg)).max(), np.abs(rgb(self.bg) - rgb(screen.bg)).max()))


class ScreenTests(unittest.TestCase):
    def test_text_in_any_colour_on_any_background(self):
        screen = Screen(None, color="truecolor", size=(4, 20))
        screen.text(0, 0, "card", (200, 30, 30), bg=(250, 240, 200))
        screen.text(1, 0, "ansi", Color.YELLOW, bg=Color.BLUE)
        screen.text(2, 0, "dark", (1, 0, 0), bg=(1.0, 0.0, 0.0))  # ints 0..255, floats 0..1: not the same red
        screen.text(3, 0, "odd", (np.nan, np.inf, -1.0), bg=Color.DEFAULT)  # (no crash)
        out = screen.render_updates()
        self.assertIn("\x1b[0;38;2;200;30;30;48;2;250;240;200mcard", out)
        self.assertIn("\x1b[33;44mansi", out)
        self.assertIn("38;2;1;0;0;48;2;255;0;0mdark", out)
        self.assertEqual(screen.bg[3, 0], -1)
        np.testing.assert_array_equal(pictures.unpack_colors(screen.bg[0, :4], (0, 0, 0)), [[250, 240, 200]] * 4)
        for mode, fg, bg in (("256", "38;5;160", "48;5;230"), ("16", "31", "107")):
            screen.set_color(mode)
            screen.text(0, 0, "card", (200, 30, 30), bg=(250, 240, 200))
            self.assertIn(f"\x1b[0;{fg};{bg}mcard", screen.render_updates())
        with self.assertRaises(ValueError):
            screen.text(0, 0, "x", (1, 2))
        screen.set_color("mono")
        screen.text(0, 0, "card", (200, 30, 30), bg=(250, 240, 200))
        self.assertEqual((screen.fg[0, 0], screen.bg[0, 0]), (-1, -1))

    def test_picture_shows_the_cells(self):
        self.assertEqual((pictures.BOLD, pictures.DIM, pictures.REVERSE), (terminal.BOLD, terminal.DIM, terminal.REVERSE))
        for glyphs in ("sextant", "quad", "half", "ascii"):
            screen = Screen(None, glyphs=glyphs, color="truecolor", size=(10, 30))
            fb = Renderer(30, 10, screen.cell_pixels).render([Object3D(make_box(), color=(255, 0, 0))], Camera(),
                                                             Light(ambient=1.0, diffuse=0.0, specular=0.0))
            screen.draw_frame(fb)
            screen.text(0, 0, "Hi", Color.YELLOW)
            screen.text(1, 0, "W", Color.GREEN, reverse=True)
            screen.text(2, 0, "W", Color.WHITE, dim=True)
            pic = screen.picture(cell=(6, 12), bg=(0, 0, 0))
            self.assertEqual(pic.shape, (120, 180, 3))
            self.assertEqual(pic.dtype, np.uint8)
            middle = pic[60 - 6:60 + 6, 90 - 3:90 + 3].reshape(-1, 3)  # the box, red
            self.assertTrue((middle[:, 0] > 150).any() and (middle[:, 1:] < 60).all(), glyphs)
            letters = pic[:12, :12].reshape(-1, 3)  # some pixels in the text's colour, the rest the background
            yellow = tuple(int(v) for v in pictures.unpack_colors(screen.fg[0, 0], (0, 0, 0)))
            self.assertTrue(10 < (letters == yellow).all(axis=1).sum() < len(letters) / 2)
            self.assertTrue((letters == 0).all(axis=1).sum() > len(letters) / 2)
            reverse = pic[12:24, :6].reshape(-1, 3)  # mostly green behind a dark letter
            self.assertGreater((reverse[:, 1] > 100).sum(), len(reverse) / 2)
            dim = pic[24:36, :6].reshape(-1, 3).max(axis=0)
            self.assertLess(int(dim.max()), int(pictures.unpack_colors(screen.fg[2, 0], (0, 0, 0)).max()))

    def test_picture_draws_box_drawing_and_block_characters(self):
        screen = Screen(None, glyphs="sextant", color="truecolor", size=(3, 6))
        for row, text in enumerate(("┌─┐▀▁", "│█│░▚", "╚═╝╋┄")):
            screen.text(row, 0, text, Color.WHITE)
        pic = screen.picture(bg=(0, 0, 0)).max(axis=2) > 0
        cell = lambda r, c: pic[r * 16:(r + 1) * 16, c * 8:(c + 1) * 8]
        self.assertTrue(cell(0, 1)[7:9].any(axis=0).all() and not cell(0, 1)[:6].any())  # ─ across the middle
        self.assertTrue(cell(1, 0)[:, 3:5].any(axis=1).all() and not cell(1, 0)[:, :3].any())  # │ down it
        corner = cell(0, 0)  # ┌: right and down from the middle, nothing up or left
        self.assertTrue(corner[8, 7] and corner[15, 4] and not corner[:6].any() and not corner[:, :3].any())
        self.assertTrue(cell(1, 1).all())  # █
        self.assertTrue(cell(0, 3)[:8].all() and not cell(0, 3)[8:].any())  # ▀
        self.assertTrue(cell(0, 4)[14:].all() and not cell(0, 4)[:14].any())  # ▁, an eighth
        self.assertAlmostEqual(cell(1, 3).mean(), 0.25)  # ░
        self.assertEqual(cell(2, 1)[:, 2].sum(), 2)  # ═: two lines
        self.assertEqual(np.count_nonzero(np.diff(np.r_[0, cell(2, 4)[8].astype(int)]) == 1), 3)  # ┄: three dashes
        masks = {}
        for code in range(0x2500, 0x25A0):  # each draws something, and (bar a few that only look alike) its own
            mask = pictures._text_mask(chr(code), (8, 16))
            self.assertTrue(mask.any(), hex(code))
            masks.setdefault(mask.tobytes(), []).append(chr(code))
        self.assertGreater(len(masks), 130)

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

    def test_text_takes_one_cell_per_character(self):
        # Wide, zero-width and control characters become '?' (the grid has a cell per character), the first time a
        # string is written and every time after (text() keeps what it made of each string), clipped or not.
        screen = Screen(glyphs="quad", color="truecolor", size=(3, 12))
        for _ in range(2):
            screen.text(0, 0, "│♖ab漢é\t🙂")
            screen.text(1, -2, "xy│end")
            screen.text(2, 9, "tail")
            self.assertEqual("".join(screen.chars[0]), "│♖ab?e??? " + "  ")
            self.assertEqual("".join(screen.chars[1]), "│end" + " " * 8)
            self.assertEqual("".join(screen.chars[2]), " " * 9 + "tai")
            screen.erase()

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

    def test_style_sends_only_what_changed(self):
        screen = Screen(color="truecolor", size=(1, 10))
        screen.color_tolerance = 0
        screen.text(0, 0, "ab")
        screen.fg[0, 0], screen.fg[0, 1] = encode_rgb([(10, 20, 30), (40, 50, 60)])
        screen.bg[0, :2] = encode_rgb((1, 2, 3))
        screen.text(0, 2, "c", bold=True)
        out = screen.render_updates()
        self.assertIn("\x1b[0;38;2;10;20;30;48;2;1;2;3ma\x1b[38;2;40;50;60mb", out)  # only the foreground changes
        self.assertIn("b\x1b[0;1;39;49mc", out)  # bold, after none: reset
        terminal = Terminal(1, 10)
        terminal.feed(out)
        self.assertEqual(terminal.levels_off(screen), 0)

    def test_colour_tolerance(self):
        screen = Screen(color="truecolor", size=(2, 10))
        self.assertEqual(screen.color_tolerance, 1)  # the default
        terminal = Terminal(2, 10)
        screen.text(0, 0, "abc")
        screen.fg[0, :3] = encode_rgb((100, 100, 100))
        terminal.feed(screen.render_updates())
        screen.fg[0, 0] = encode_rgb((101, 99, 100))  # a level: left as it is while something else changes
        screen.fg[0, 2] = encode_rgb((102, 100, 100))  # two levels: sent
        screen.text(1, 0, "x")
        out = screen.render_updates()
        terminal.feed(out)
        self.assertIn("\x1b[1;3H", out)
        self.assertNotIn("\x1b[1;1H", out)
        self.assertEqual(terminal.levels_off(screen), 1)
        terminal.feed(screen.render_updates())  # nothing else changed: the cell left over is sent
        self.assertEqual(terminal.levels_off(screen), 0)
        self.assertEqual(screen.render_updates(), "")

        # A cell kept within the tolerance while other cells keep changing is sent after SETTLE_FRAMES refreshes.
        screen.fg[0, 0] = encode_rgb((100, 100, 100))
        for i in range(SETTLE_FRAMES + 1):
            screen.text(1, 0, "yx"[i % 2])  # (row 1 shows "x" already)
            terminal.feed(screen.render_updates())
            self.assertEqual(terminal.levels_off(screen), int(i < SETTLE_FRAMES - 1), i)

        # 0, or anything that isn't a positive number, sends every change; so do the palette modes.
        for tolerance, mode in ((0, "truecolor"), (float("nan"), "truecolor"), ("x", "truecolor"), (5, "256")):
            screen = Screen(color=mode, size=(1, 4))
            screen.color_tolerance = tolerance
            screen.text(0, 0, "ab")
            screen.fg[0, 0] = int(encode_rgb((100, 100, 100)) if mode == "truecolor" else encode_index(100))
            screen.render_updates()
            screen.fg[0, 0] += 1
            screen.text(0, 1, "c")
            self.assertIn("\x1b[1;1H", screen.render_updates(), (tolerance, mode))

    def test_terminal_shows_frames_within_the_tolerance(self):
        # What the terminal shows, replayed from the output, is at most color_tolerance levels off each frame of
        # a spinning scene, and exact once it stops.
        for tolerance in (0, 1, 2):
            screen = Screen(glyphs="sextant", color="truecolor", size=(20, 40))
            screen.color_tolerance = tolerance
            renderer = Renderer(40, 20, screen.cell_pixels)
            terminal = Terminal(20, 40)
            balls = [Object3D(blob_mesh((0.5, 0.5, 0.5)), np.array([x, 0.0, 0.0]), color=(200, 90, 60 + 40 * x))
                     for x in (-1.0, 0.0, 1.0)]
            for i in range(12):
                for ball in balls:
                    ball.rotation = quat_axis_angle((0.3, 1.0, 0.2), 0.05 * i)
                screen.draw_frame(renderer.render(balls, Camera(), Light(direction=(np.sin(0.1 * i), -1.0, -0.5))))
                terminal.feed(screen.render_updates())
                self.assertLessEqual(terminal.levels_off(screen), tolerance)
            screen.draw_frame(renderer.render(balls, Camera(), Light(direction=(np.sin(1.1), -1.0, -0.5))))
            terminal.feed(screen.render_updates())
            self.assertEqual(terminal.levels_off(screen), 0)

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


class SlowConsole:
    """A console for run() whose writes of frames wait until `release` is set, as a slow terminal's would."""

    unicode, key_release, reports_releases = True, False, False

    def __init__(self, fail=False):
        self.writes, self.fail = [], fail
        self.writing, self.release = threading.Event(), threading.Event()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.writes.append("restored")

    def size(self):
        return 20, 4

    def read(self):
        return ""

    def key_is_down(self, key):
        return None

    def write(self, data):
        if isinstance(data, bytes):  # a frame (the compile notice is text)
            self.writing.set()
            self.release.wait(2)
            if self.fail:
                raise OSError("terminal gone")
        self.writes.append(data)


class OutputTests(unittest.TestCase):
    """run() sends each frame to the terminal while the next one is drawn."""

    def run_frames(self, console, frame_fn):
        with tempfile.TemporaryDirectory() as cache, \
                unittest.mock.patch.dict(os.environ, {"XDG_CACHE_HOME": cache}), \
                unittest.mock.patch.object(terminal, "open_console", lambda **kw: console):
            terminal.run(frame_fn)

    def test_the_next_frame_is_drawn_while_the_last_is_written(self):
        console, seen = SlowConsole(), []
        frames_written = lambda: sum(isinstance(w, bytes) for w in console.writes)

        def frame(screen, dt, keys):
            screen.text(0, 0, f"frame {len(seen)}")
            if not seen:
                screen.refresh()  # returns with the frame still being written
                seen.append(frames_written() == 0 and console.writing.wait(10))
                return True
            seen.append(frames_written() == 0)  # drawn while the terminal still takes in the first frame
            console.release.set()
            screen.text(0, 0, "last")
            screen.refresh()
            return False

        self.run_frames(console, frame)
        self.assertEqual(seen, [True, True])
        frames = [w for w in console.writes if isinstance(w, bytes)]
        self.assertEqual(len(frames), 2)
        self.assertIn(b"frame 0", frames[0])
        self.assertIn(b"last", frames[1])
        self.assertEqual(console.writes[-1], "restored")  # after the last frame was written in full

    def test_an_error_writing_is_raised_by_the_next_refresh(self):
        console = SlowConsole(fail=True)
        console.release.set()
        frames = []

        def frame(screen, dt, keys):
            frames.append(dt)
            screen.text(0, 0, str(len(frames)))
            screen.refresh()

        with self.assertRaisesRegex(OSError, "terminal gone"), unittest.mock.patch("sys.stderr"):
            self.run_frames(console, frame)
        self.assertEqual(len(frames), 2)  # the first frame's error, raised by the second's refresh
        self.assertEqual(console.writes[-1], "restored")

    def test_writes_keep_their_order(self):
        written = []

        class Console:
            def write(self, data):
                time.sleep(0.001 * (len(written) % 3))
                written.append(data)

        writer = terminal._Writer(Console())
        for i in range(50):
            writer.write(i)
        writer.close()
        self.assertEqual(written, list(range(50)))


if __name__ == "__main__":
    unittest.main()
