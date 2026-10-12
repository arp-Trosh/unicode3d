import unittest

import numpy as np

from unicode3d import Camera, Colliders, Light, Lines, Mesh, Node, Object3D, Renderer, make_box, union_bounds
from unicode3d.color import to_linear_rgb

CAMERA = Camera(position=np.array([0.0, 0.0, 5.0]), target=np.zeros(3))
FLAT = Camera(position=np.array([0.0, 1.0, 5.0]), target=np.array([0.0, 0.0, -10.0]), far=300.0)  # (low over a floor)


def frame(objects, camera=CAMERA, renderer=None):
    fb = (renderer or Renderer(80, 40)).render(objects, camera, Light())
    return fb.rgb.copy(), fb.depth.copy(), fb.ids.copy()


def across(objects, camera=CAMERA, axis=0):
    """How wide the line drawn alone is, in pixels, in each column (axis=0) or row (axis=1) it crosses: how much
    of them it covers (partly covered pixels at its edges counting in part)."""
    alpha = Renderer(80, 40).render(objects, camera, Light()).alpha
    width = alpha.sum(axis=axis)
    return width[width > 0]


class LinesTest(unittest.TestCase):
    def test_pixel_widths_are_the_same_near_and_far(self):
        # Across the view, and running off towards the horizon: as many pixels wide all along, without gaps.
        for width in (1, 2, 3):
            np.testing.assert_allclose(across([Lines([(-3, 0, 0)], [(3, 0, 0)], width=width)]), width, atol=0.1)
        line = Lines([(0.0, 0.0, 4.0)], [(0.0, 0.0, -250.0)], width=2)  # (straight up the picture)
        ids = frame([line], FLAT)[2]
        rows = np.flatnonzero((ids > 0).any(axis=1))
        self.assertEqual(len(rows), rows[-1] - rows[0] + 1)  # (no row missed between its ends)
        widths = across([line], FLAT, axis=1)[1:-1]  # (but where it ends, at the horizon and the bottom)
        self.assertGreater(widths.min(), 1.9)
        self.assertLess(widths.max(), 2.3)  # (at most a tenth too wide)

    def test_world_widths_and_their_least_pixels(self):
        ortho = Camera(projection="ortho", size=5.0)  # (80 pixels tall for 5 units: 16 a unit)
        np.testing.assert_allclose(across([Lines([(-3, 0, 0)], [(3, 0, 0)], width=0.5, units="world")], ortho), 8.0,
                                   atol=0.1)
        # Far off and thinner than a pixel: still drawn (min_pixels), unbroken; or as thin as it is with 0.
        far = Lines([(-50, -5, -80)], [(50, -5, -80)], width=0.001, units="world")
        ids = frame([far], FLAT)[2]
        self.assertTrue((ids > 0).any(axis=0).all())
        far.min_pixels = 0.0
        self.assertLess((frame([far], FLAT)[2] > 0).sum(), 20)
        # In the Lines' own units: a parent's scale widens it.
        group = Node(scale=2.0)
        wide = Lines([(-1.5, 0, 0)], [(1.5, 0, 0)], width=0.25, units="world", parent=group)
        np.testing.assert_allclose(across([wide], ortho), 8.0, atol=0.1)

    def test_hidden_behind_things_and_picked(self):
        line = Lines([(-3, 0, -1)], [(3, 0, -1)], width=2)
        box = Object3D(make_box())
        _, _, ids = frame([box, line])
        self.assertEqual(set(np.unique(ids[:, 40])), {0, 1})  # (in the middle, the box in front)
        self.assertIn(2, ids[:, 5])
        renderer = Renderer(80, 40)
        renderer.render([box, line], CAMERA, Light())
        row = int(np.flatnonzero(ids[:, 5] == 2)[0])
        self.assertIs(renderer.pick(5, row // 2).object, line)

    def test_colours_unshaded_by_default(self):
        lines = Lines([(-3, 1, 0), (-3, -1, 0)], [(3, 1, 0), (3, -1, 0)], colors=[(255, 0, 0), (0, 0, 255)],
                      color=(255, 255, 128), width=3)
        rgb, _, ids = frame([lines])
        top, bottom = rgb[np.flatnonzero(ids[:, 40])[1], 40], rgb[np.flatnonzero(ids[:, 40])[-2], 40]
        np.testing.assert_allclose(top, to_linear_rgb((255, 0, 0)) * to_linear_rgb((255, 255, 128)), atol=0.01)
        np.testing.assert_allclose(bottom, to_linear_rgb((0, 0, 255)) * to_linear_rgb((255, 255, 128)), atol=0.01)
        lines.colors[1] = (0, 255, 0)  # (changed in place)
        self.assertGreater(frame([lines])[0][np.flatnonzero(ids[:, 40])[-2], 40, 1], 0.5)

    def test_changed_between_frames(self):
        lines = Lines([(-3, 0, 0)], [(3, 0, 0)])
        renderer = Renderer(80, 40)
        first = frame([lines], renderer=renderer)[2].copy()
        lines.ends[0] = (3, 2, 0)  # (in place)
        moved = frame([lines], renderer=renderer)[2]
        self.assertFalse(np.array_equal(first, moved))
        lines.set([(0, -2, 0), (-2, 0, 0)], [(0, 2, 0), (2, 0, 0)])
        self.assertEqual(len(lines), 2)
        cross = frame([lines], renderer=renderer)[2] > 0  # (a cross: up the middle and across it)
        self.assertGreater(cross.any(axis=1).sum(), 25)
        self.assertGreater(cross.any(axis=0).sum(), 25)
        with self.assertRaises(ValueError):
            lines.set([(0, 0, 0)], [(1, 0, 0), (2, 0, 0)])
        with self.assertRaises(ValueError):
            Lines(units="metres")
        lines.starts = np.zeros((2, 2))  # (assigned wrong: an error, not a read beyond the array)
        with self.assertRaises(ValueError):
            frame([lines])

    def test_drawn_the_same_with_levels_of_detail(self):
        grid = np.linspace(-5, 5, 11)
        lines = Lines(np.c_[grid, np.zeros(11), np.full(11, -5.0)], np.c_[grid, np.zeros(11), np.full(11, 5.0)])
        objects = [Object3D(make_box(), np.array([0.0, 0.5, 0.0])), lines]
        for a, b in zip(frame(objects, FLAT), frame(objects, FLAT, Renderer(80, 40, simplify=2.0))):
            np.testing.assert_array_equal(a, b)

    def test_facing_the_camera_in_mirrors_too(self):
        # Seen only in a mirror, which the camera looks into at an angle: a line as wide in the reflection as from
        # the reflected eye, and a billboard as big (turned to the frame's camera, both were seen from the side).
        mirror = Object3D(Mesh(np.array([(-20.0, -20, 0), (20, -20, 0), (20, 20, 0), (-20, 20, 0)]),
                               np.array([(0, 1, 2), (0, 2, 3)])), color=(0, 0, 0), reflectivity=1.0, specular=0.0)
        at, along = np.array([-3.6, 0.0, 4.8]), np.array([0.8, 0.0, 0.6])  # (6 beyond the mirror; across the view)
        line = Lines([at + (0, 1.2, 0) - 1.5 * along], [at + (0, 1.2, 0) + 1.5 * along], width=3, color=(255, 0, 0))
        card = Object3D(make_box(), at - (0, 1.0, 0), scale=(1.0, 1.0, 0.0), color=(0, 255, 0), emissive=1.0,
                        billboard=True)
        seen = Camera(position=np.array([3.0, 0.0, 4.0]), target=np.zeros(3))
        rgb = Renderer(80, 40).render([mirror, line, card], seen, Light()).rgb
        direct = Renderer(80, 40).render([line, card], Camera(position=np.array([3.0, 0.0, -4.0]), target=np.zeros(3)),
                                         Light()).rgb  # (from the reflected eye)
        # (how wide in each column: the mirror passes on less light at some angles than at others)
        red, alone = (c[..., 0].sum(axis=0) / np.maximum(c[..., 0].max(axis=0), 1e-9) for c in (rgb, direct))
        self.assertGreater((red > 0.5).sum(), 15)
        np.testing.assert_allclose(red[red > 0.5][1:-1], 3.0, atol=0.15)
        np.testing.assert_allclose(alone[alone > 0.5][1:-1], 3.0, atol=0.15)
        self.assertAlmostEqual((rgb[..., 1] > 0.5).sum() / (direct[..., 1] > 0.5).sum(), 1.0, delta=0.1)

    def test_left_out_of_queries_and_bounds(self):
        lines = Lines([(-3, 0, 0)], [(3, 0, 0)], width=0.5, units="world")
        box = Object3D(make_box(), np.array([3.0, 0.0, 0.0]))
        self.assertIs(Colliders([lines, box]).raycast((0.0, 0.0, 5.0), (0.0, 0.0, -1.0)), None)
        np.testing.assert_allclose(union_bounds([lines, box])[0], (2.5, -0.5, -0.5))

    def test_bad_numbers_and_odd_cases_do_not_crash(self):
        under = Camera(position=np.array([0.0, 0.5, 0.0]), target=np.array([0.0, 0.0, -1.0]))
        cases = [Lines(), Lines([(0, 0, 10)], [(0, 0, -10)]),  # (empty; through the camera's place)
                 Lines([(0, 0, 5)], [(1, 0, 6)]), Lines([(0, 0.5, -1)], [(0, 0.5, -5)]),  # (behind; end on)
                 Lines([(np.nan, 0, 0)], [(1, 0, 0)]), Lines([(np.inf, 0, -1)], [(1, 0, -1)]),
                 Lines([(0, 0, -1)], [(0, 0, -1)]), Lines([(0, 0, -1)], [(1, 0, -1)], width=np.nan),
                 Lines([(-1e300, 0, -1)], [(1e300, 0, -1)], width=1e300, units="world"),
                 Lines([(0, 0, -1)], [(1, 0, -1)], colors=[(np.nan, 0, 0)], min_pixels=np.inf),
                 Lines([(0, 0, -1)], [(1, 0, -1)], position=(np.nan, 0, 0))]
        self.assertGreater((frame(cases[1:2], under)[2] > 0).sum(), 50)
        for camera in (under, CAMERA, Camera(near=0.0), Camera(near=-1.0), Camera(near=np.nan),
                       Camera(position=np.zeros(3), target=np.zeros(3)), Camera(projection="ortho")):
            frame([Object3D(make_box()), *cases], camera)


if __name__ == "__main__":
    unittest.main()
