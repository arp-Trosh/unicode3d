import os
import tempfile
import unittest

import numpy as np

from unicode3d.dice import DIE_VALUES, RollAnimation, make_die, orientation_showing, top_face
from unicode3d.mesh import Mesh, load_obj, make_box
from unicode3d.raster import FrameBuffer
from unicode3d.scene import Camera, Light, Object3D, Renderer
from unicode3d.terminal import PALETTE, Color, cell_shade, frame_to_text
from unicode3d.transforms import quat_between, quat_to_matrix


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


class RenderTests(unittest.TestCase):
    def render(self, objects, w=80, h=30, camera=None, **kwargs):
        camera = camera or Camera(position=np.array([0.0, 0.0, 5.0]))
        return Renderer(w, h, **kwargs).render(objects, camera, Light())

    def test_die_is_drawn_centred(self):
        fb = self.render([Object3D(make_die())])
        self.assertEqual((fb.width, fb.height), (80, 60))  # two pixels per cell, stacked
        drawn = fb.shade >= 0
        self.assertTrue(drawn[30, 40])
        self.assertFalse(drawn[0, 0] or drawn[-1, -1])
        rows, cols = np.nonzero(drawn)
        self.assertAlmostEqual(rows.mean(), 30, delta=1.5)
        self.assertAlmostEqual(cols.mean(), 40, delta=1.5)
        # The centre pip is darker than the lit face around it.
        self.assertLess(fb.shade[30, 40], fb.shade[30, 44] - 0.2)
        self.assertTrue(set(frame_to_text(fb)) & set(".:-"))
        self.assertEqual(len(frame_to_text(fb).splitlines()), 30)

    def test_die_is_round_not_stretched(self):
        # Pixels are about square, so a cube seen face-on covers a square of pixels.
        fb = self.render([Object3D(make_box())], supersample=1, fog=0, outline=0)
        rows, cols = np.nonzero(fb.shade >= 0)
        self.assertAlmostEqual(np.ptp(rows) / np.ptp(cols), 1.0, delta=0.1)

    def test_nearer_object_wins_depth_test(self):
        near = Object3D(make_box(), position=np.array([0.0, 0.0, 1.0]), color=2)
        far = Object3D(make_box(2.0), position=np.array([0.0, 0.0, -2.0]), color=3)
        for order in ([near, far], [far, near]):
            fb = self.render(order)
            self.assertEqual(fb.color[30, 40], 2)

    def test_back_faces_are_culled(self):
        fb = self.render([Object3D(make_box(), position=np.array([0.0, 0.0, 6.0]))])  # camera inside the box
        self.assertFalse((fb.shade >= 0).any())

    def test_supersampling_matches_plain_render(self):
        plain = self.render([Object3D(make_box())], supersample=1, fog=0, outline=0)
        smooth = self.render([Object3D(make_box())], supersample=3, fog=0, outline=0)
        self.assertLess(np.abs((plain.shade >= 0).sum() - (smooth.shade >= 0).sum()), 0.1 * (plain.shade >= 0).sum())

    def test_smooth_shading_on_shared_vertices(self):
        # An octahedron shades smoothly: neighbouring pixels differ gently, not in flat steps.
        v = np.array([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)], float)
        f = np.array([(4, 0, 2), (4, 2, 1), (4, 1, 3), (4, 3, 0), (5, 2, 0), (5, 1, 2), (5, 3, 1), (5, 0, 3)])
        mesh = Mesh(v, f)
        np.testing.assert_allclose(np.linalg.norm(mesh.vertex_normals(), axis=1), 1.0)
        fb = self.render([Object3D(mesh)], supersample=1, fog=0, outline=0)
        row = fb.shade[30][fb.shade[30] >= 0]
        self.assertGreater(len(np.unique(np.round(row, 3))), 10)

    def test_camera_inside_scene_clips_instead_of_dropping(self):
        # A floor stretching from behind the camera to far in front: its near part must still draw.
        floor = Mesh(np.array([(-5, 0, 5), (5, 0, 5), (5, 0, -50), (-5, 0, -50)], float), np.array([(0, 1, 2), (0, 2, 3)]))
        camera = Camera(position=np.array([0.0, 1.0, 0.0]), target=np.array([0.0, 0.0, -3.0]))
        fb = self.render([Object3D(floor)], camera=camera)
        self.assertTrue((fb.shade[-1] >= 0).all())  # the bottom row is floor right up to the camera

    def test_outline_darkens_far_side_of_overlap(self):
        near = Object3D(make_box(), position=np.array([0.0, 0.0, 1.5]))
        far = Object3D(make_box(4.0), position=np.array([0.0, 0.0, -3.0]))
        plain = self.render([near, far], fog=0, outline=0)
        lined = self.render([near, far], fog=0, outline=0.6)
        darker = lined.shade < plain.shade - 1e-6
        self.assertTrue(darker.any())
        self.assertEqual(lined.color[darker].tolist(), [0] * int(darker.sum()))
        self.assertFalse((lined.shade[30, 35:46] < plain.shade[30, 35:46] - 1e-6).any())  # the near box is untouched

    def test_fog_dims_distant_surfaces(self):
        near = Object3D(make_box(), position=np.array([-1.0, 0.0, 0.0]))
        far = Object3D(make_box(), position=np.array([1.5, 0.0, -6.0]))
        clear = self.render([near, far], fog=0, outline=0)
        foggy = self.render([near, far], fog=0.5, outline=0)
        far_px = (clear.color == 0) & (clear.shade >= 0) & (np.arange(80) > 44)
        self.assertTrue(far_px.any())
        self.assertLess(foggy.shade[far_px].mean(), 0.75 * clear.shade[far_px].mean())

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


class TerminalTests(unittest.TestCase):
    def test_palette_gets_brighter(self):
        from unicode3d.terminal import xterm_rgb
        rgb = xterm_rgb()
        for hue in Color:
            lum = rgb[PALETTE[hue] - 16] @ [0.30, 0.59, 0.11]
            self.assertTrue((np.diff(lum) >= 0).all(), hue)
            self.assertLess(lum[0], lum[-1])

    def test_cell_shade_merges_pixel_pairs(self):
        fb = FrameBuffer(2, 4)
        fb.shade[:] = [[0.2, -1], [0.6, -1], [-1, -1], [-1, 0.5]]
        fb.depth[:] = [[0.1, 0], [0.2, 0], [0, 0], [0, 0.3]]
        fb.color[:] = [[1, 0], [2, 0], [0, 0], [0, 3]]
        shade, color = cell_shade(fb)
        np.testing.assert_allclose(shade, [[0.4, -1], [-1, 0.5]])
        self.assertEqual(color.tolist(), [[2, 0], [0, 3]])  # the nearer pixel's colour


if __name__ == "__main__":
    unittest.main()
