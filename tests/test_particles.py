import unittest

import numpy as np

from unicode3d import Camera, Colliders, Light, Mesh, Node, Object3D, Particles, Renderer, make_box, union_bounds
from unicode3d.shapes import blob_mesh
from unicode3d.transforms import quat_axis_angle

SQUARE = Mesh(np.array([(-0.5, -0.5, 0.0), (0.5, -0.5, 0.0), (0.5, 0.5, 0.0), (-0.5, 0.5, 0.0)]),
              np.array([(0, 1, 2), (0, 2, 3)]))
CAMERA = Camera(position=np.array([0.0, 0.5, 5.0]), target=np.zeros(3))


def frame(objects, camera=CAMERA, lights=None, renderer=None):
    fb = (renderer or Renderer(60, 30)).render(objects, camera, lights or Light())
    return fb.rgb.copy(), fb.depth.copy(), fb.ids.copy()


class ParticlesTest(unittest.TestCase):
    def test_drawn_as_the_objects_they_stand_for(self):
        # Each particle is an instance of the mesh, as an Object3D at its place and size would be.
        ball = blob_mesh((0.5, 0.5, 0.5), rings=6, segments=8)
        at = np.array([(-1.0, 0.0, 0.0), (0.3, 0.4, -0.5), (1.2, -0.3, 0.4)])
        sizes, colors = (0.5, 0.8, 0.6), ((255, 80, 40), (40, 200, 90), (90, 90, 255))
        swarm = Particles(ball, specular=1.0)
        for p, s, c in zip(at, sizes, colors):
            swarm.emit(p, size=s, color=c, angle=0.0, life=10.0)
        floor = Object3D(make_box(), np.array([0.0, -1.5, 0.0]), scale=(4.0, 0.2, 4.0), color=(200, 200, 200))
        objects = [Object3D(ball, p, scale=s, color=c, cast_shadows=False) for p, s, c in zip(at, sizes, colors)]
        sun = Light(direction=np.array([0.3, -1.0, -0.2]), shadows=True)
        got, want = frame([floor, swarm], lights=sun), frame([floor, *objects], lights=sun)
        np.testing.assert_allclose(got[0], want[0], atol=1e-6)
        np.testing.assert_allclose(got[1], want[1], atol=1e-9)
        self.assertEqual(set(np.unique(got[2])) - {0}, {1, 2})  # (every particle: the Particles' id)

    def test_no_outlines_unless_asked_for(self):
        # Outlines darken what is behind an object's edges; particles have none by default, nor objects with
        # outline=False: the frame is as drawn with no outlines at all.
        floor = Object3D(make_box(), np.array([0.0, -1.0, 0.0]), scale=(6.0, 0.2, 6.0), color=(200, 200, 200),
                         outline=False)  # (its own edges outlined: not what is looked at)
        swarm = Particles(make_box(), seed=1)
        swarm.emit((0.0, -0.4, 0.0), 6, radius=0.5, size=0.3, angle=0.0, life=10.0)  # (in front of the floor)
        box = Object3D(make_box(), np.array([1.0, -0.4, 0.5]), scale=0.4, outline=False)
        objects = [floor, swarm, box]
        plain = frame(objects, renderer=Renderer(60, 30, outline=0.0))[0]
        np.testing.assert_array_equal(frame(objects)[0], plain)
        swarm.outline = True
        self.assertGreater((frame(objects)[0] != plain).any(axis=2).sum(), 10)
        swarm.outline, box.outline = False, True
        self.assertGreater((frame(objects)[0] != plain).any(axis=2).sum(), 10)
        # A see-through puff seen from above against an outlined floor: the floor next to it is level with its
        # neighbours on the far side (to rounding), which must not count as the edge's near side.
        floor.outline = True
        puff = Particles(blob_mesh((1.0, 1.0, 1.0), rings=5, segments=8), seed=1)
        puff.emit((0.0, 0.6, 0.0), size=0.6, opacity=0.45, life=10.0)
        above = Camera(position=np.array([0.0, 5.0, 3.0]), target=np.zeros(3))
        flat = Object3D(make_box(), np.array([0.0, -0.5, 0.0]), scale=(6.0, 1.0, 6.0))
        for size in ((60, 30), (120, 60)):
            np.testing.assert_array_equal(frame([flat, puff], above, renderer=Renderer(*size))[0],
                                          frame([flat, puff], above, renderer=Renderer(*size, outline=0.0))[0])

    def test_pick_gives_the_particles(self):
        swarm = Particles(seed=1)
        swarm.emit((0.0, 0.0, 0.0), 50, radius=0.5, size=0.3, life=10.0)
        renderer = Renderer(60, 30)
        renderer.render([Object3D(make_box(), np.array([5.0, 0.0, 0.0])), swarm], CAMERA, Light())
        self.assertIs(renderer.pick(30, 15).object, swarm)

    def test_update_moves_them_and_drops_the_old(self):
        swarm = Particles(gravity=(0.0, -10.0, 0.0), drag=0.0)
        swarm.emit((0.0, 5.0, 0.0), velocity=(1.0, 2.0, 0.0), life=1.0)
        swarm.emit((0.0, 5.0, 0.0), life=0.25)
        for _ in range(10):
            swarm.update(0.05)
        self.assertEqual(len(swarm), 1)  # (the second's life is over)
        # Velocity then position, a step at a time.
        v, p = np.array([1.0, 2.0, 0.0]), np.array([0.0, 5.0, 0.0])
        for _ in range(10):
            v = v + np.array([0.0, -10.0, 0.0]) * 0.05
            p = p + v * 0.05
        np.testing.assert_allclose(swarm.positions[0], p)
        np.testing.assert_allclose(swarm.velocities[0], v)
        self.assertAlmostEqual(swarm.ages[0], 0.5)

    def test_bounce_on_the_ground(self):
        swarm = Particles(gravity=(0.0, -10.0, 0.0), ground=0.0, bounce=0.5, friction=0.5)
        swarm.emit((0.0, 0.15, 0.0), velocity=(2.0, -1.0, 0.0), size=0.2, life=10.0)
        swarm.update(0.1)
        self.assertEqual(swarm.positions[0, 1], 0.1)  # (resting on its underside)
        np.testing.assert_allclose(swarm.velocities[0], (1.0, 1.0, 0.0))  # (up at half, along at half)

    def test_limit_makes_room_by_dropping_the_earliest(self):
        swarm = Particles(limit=5)
        swarm.emit((0.0, 0.0, 0.0), 3, life=1.0)
        swarm.emit((1.0, 0.0, 0.0), 4, life=2.0)
        self.assertEqual(len(swarm), 5)
        np.testing.assert_array_equal(swarm.lives, (1.0, 2.0, 2.0, 2.0, 2.0))
        self.assertEqual(swarm.emit((0.0, 0.0, 0.0), 9), 5)

    def test_the_same_seed_gives_the_same_particles(self):
        def made(seed):
            swarm = Particles(seed=seed, gravity=(0.0, -9.8, 0.0))
            swarm.emit((0.0, 1.0, 0.0), 20, speed=(1.0, 2.0), radius=0.3, life=(0.5, 1.0), spin=(-3.0, 3.0))
            swarm.update(0.1)
            return frame([swarm])
        for a, b in zip(made(3), made(3)):
            np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(made(3)[1], made(4)[1]))

    def test_size_colour_and_opacity_over_their_life(self):
        swarm = Particles(face_camera=True)
        swarm.emit((0.0, 0.0, 0.0), size=1.0, size_end=3.0, color=(255, 0, 0), color_end=(0, 0, 255),
                   opacity=1.0, opacity_end=0.0, life=1.0)
        rgb, _, ids = frame([swarm])
        small, red = (ids > 0).sum(), rgb[ids > 0].mean(axis=0)
        swarm.update(0.5)
        rgb, _, ids = frame([swarm])
        self.assertGreater((ids > 0).sum(), 3.0 * small)  # (twice as wide)
        self.assertGreater(red[0], red[2])
        middle = rgb[15 * 2, 30]
        self.assertGreater(middle[0], 0.02)
        self.assertAlmostEqual(middle[0], middle[2], places=5)  # (half way to blue, and half see-through)
        swarm.update(0.499)
        self.assertLess(frame([swarm])[0].max(), 0.02)  # (all but clear)

    def test_face_camera_and_billboards_face_it_from_anywhere(self):
        side = Camera(position=np.array([5.0, 0.0, 0.0]), target=np.zeros(3))  # (edge on to the xy plane)
        flat = Particles(SQUARE)
        flat.emit((0.0, 0.0, 0.0), size=1.5, angle=0.0, life=10.0)
        self.assertEqual(frame([flat], side)[2].max(), 0)
        facing = Particles(SQUARE, face_camera=True)
        facing.emit((0.0, 0.0, 0.0), size=1.5, life=10.0)
        self.assertGreater((frame([facing], side)[2] > 0).sum(), 100)
        card = Object3D(SQUARE, scale=1.5)
        self.assertEqual(frame([card], side)[2].max(), 0)
        card.billboard = True
        _, depth, ids = frame([card], side)
        self.assertGreater((ids > 0).sum(), 100)
        _, depth_facing, _ = frame([facing], side)
        np.testing.assert_allclose(depth, depth_facing, atol=1e-9)  # (as a face-on particle, the same size)

    def test_placed_through_a_parent(self):
        group = Node(position=np.array([1.0, 0.0, 0.0]), scale=2.0)
        swarm = Particles(parent=group)
        swarm.emit((0.0, 0.0, 0.0), size=0.5, angle=0.0, life=10.0)
        one = Object3D(Particles().mesh, np.array([1.0, 0.0, 0.0]), scale=1.0)
        np.testing.assert_allclose(frame([swarm])[1], frame([one])[1], atol=1e-9)
        swarm.visible = False
        self.assertEqual(frame([swarm])[2].max(), 0)

    def test_a_still_frame_is_not_drawn_again(self):
        swarm = Particles(seed=2)
        swarm.emit((0.0, 0.0, 0.0), 30, speed=1.0, radius=1.0, life=10.0)
        renderer = Renderer(60, 30)
        for _ in range(3):
            renderer.render([swarm], CAMERA, Light())
        self.assertEqual(renderer.draws, 1)
        swarm.update(0.1)
        renderer.render([swarm], CAMERA, Light())
        self.assertEqual(renderer.draws, 2)

    def test_empty_and_emptied(self):
        swarm = Particles()
        frame([swarm])
        swarm.emit((0.0, 0.0, 0.0), 5, life=0.1)
        swarm.update(1.0)
        self.assertEqual(len(swarm), 0)
        self.assertEqual(frame([swarm])[2].max(), 0)
        swarm.emit((0.0, 0.0, 0.0), 5)
        swarm.clear()
        self.assertEqual(len(swarm), 0)
        self.assertEqual(frame([Object3D(make_box()), swarm])[2].max(), 1)

    def test_left_out_of_queries_and_bounds(self):
        swarm = Particles()
        swarm.emit((0.0, 0.0, 0.0), 5, life=10.0)
        box = Object3D(make_box(), np.array([3.0, 0.0, 0.0]))
        self.assertIs(Colliders([swarm, box]).raycast((0.0, 0.0, 5.0), (0.0, 0.0, -1.0)), None)
        np.testing.assert_allclose(union_bounds([swarm, box])[0], (2.5, -0.5, -0.5))

    def test_bad_numbers_do_not_crash(self):
        swarm = Particles(gravity=(np.nan, np.inf, 0.0), drag=np.nan, ground=np.nan, seed=1)
        swarm.emit((np.nan, 0.0, 0.0), 3)
        swarm.emit((0.0, 0.0, 0.0), 3, size=np.inf, life=np.nan, opacity=np.nan)
        swarm.emit((0.0, 0.0, 0.0), 3, speed=1e308, size=(1e300, np.inf))
        swarm.emit((0.0, 0.0, 0.0), float("nan"))
        swarm.emit((0.0, 0.0, 0.0), np.inf)
        good = Particles(face_camera=True, position=(np.nan, 0.0, 0.0))
        good.emit((0.0, 0.0, 0.0), 3)
        for dt in (0.1, np.nan, -1.0, np.inf, 1e308):
            swarm.update(dt)
            frame([swarm, good, Object3D(make_box())])
        card = Object3D(SQUARE, billboard=True, scale=(np.inf, 1.0, 1.0))
        frame([card], Camera(position=np.zeros(3), target=np.zeros(3)))  # (a camera at its target: no basis)


if __name__ == "__main__":
    unittest.main()
