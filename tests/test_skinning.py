# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
"""Skinned meshes (skinning.py): drawn as their joints pose them, culled, shadowed and kept from frame to frame."""
import types
import unittest
import unittest.mock

import numpy as np

from unicode3d import skinning
from unicode3d.mesh import Mesh, make_box
from unicode3d.scene import (Camera, Light, Lines, Model, Node, Object3D, Particles, PointLight, Renderer, Skin,
                             union_bounds)
from unicode3d.shadows import SETTLE_DRAWS
from unicode3d.transforms import quat_axis_angle

from .figures import baked, chain, figure, tube_mesh

CAMERA = Camera(position=np.array([0.0, 1.5, 6.0]), target=np.array([0.0, 1.5, 0.0]))
SUN = Light(direction=np.array([0.4, -1.0, -0.5]), shadows=True)


def floor():
    return Object3D(Mesh(np.array([(-4, 0, -4), (4, 0, -4), (4, 0, 4), (-4, 0, 4)], float) - (0, 0.01, 0),
                         np.array([(0, 2, 1), (0, 3, 2)])), color=(200, 200, 200))


class SkinningTests(unittest.TestCase):
    def assert_same_picture(self, a, b, most=0.002):
        """The two framebuffers agree but for at most a fraction `most` of their pixels (where two corners a
        rounding apart put a sample's edge on the other side)."""
        differ = (np.abs(a.rgb - b.rgb).max(axis=2) > 1e-5) | (a.ids != b.ids)
        self.assertLessEqual(differ.mean(), most, f"{int(differ.sum())} pixels differ")
        self.assertGreater((a.ids > 0).sum(), 100)  # (something drawn)

    def render(self, objects, lights=SUN, camera=CAMERA, renderer=None, **options):
        renderer = renderer or Renderer(60, 30, **options)
        fb = renderer.render(objects, camera, lights)
        return types.SimpleNamespace(rgb=fb.rgb.copy(), ids=fb.ids.copy())

    def test_the_kernel_poses_as_skin_pose_does(self):
        obj, joints = figure(bend=0.6)
        joints[1].scale = (1.0, 1.3, 0.7)  # stretched unevenly: normals turn by the inverse transpose
        joints[2].position = np.array([0.2, 1.0, 0.1])
        renderer = Renderer(40, 20)
        renderer.render([obj], CAMERA, SUN)
        inst = renderer._instances([obj], CAMERA)
        start = int(inst["src"][0])
        positions, normals = obj.skin.pose(obj.mesh)
        rows = inst["skin"][0][:, start:start + len(positions)]
        np.testing.assert_allclose(rows[0], positions, atol=1e-12)
        np.testing.assert_allclose(rows[1] / np.linalg.norm(rows[1], axis=1, keepdims=True), normals, atol=1e-12)
        np.testing.assert_array_equal(inst["lin"][0], np.eye(3))  # (placed by the identity: rows in the world)
        np.testing.assert_array_equal(inst["pos"][0], np.zeros(3))

    def test_drawn_as_the_same_pose_baked_into_a_mesh(self):
        obj, joints = figure(bend=0.5, color=(220, 160, 120))
        joints[0].position = np.array([-0.5, 0.0, 0.3])
        joints[0].rotation = quat_axis_angle((0, 1, 0), 0.8)
        for lights in (SUN, [PointLight(np.array([1.0, 2.5, 1.5]), range=6.0, shadows=True)]):
            self.assert_same_picture(self.render([obj, floor()], lights), self.render([baked(obj), floor()], lights))

    def test_its_own_place_does_not_count_its_visibility_does(self):
        obj, _ = figure(bend=0.4)
        before = self.render([obj])
        obj.position, obj.scale = np.array([5.0, 0.0, 0.0]), 3.0
        self.assert_same_picture(before, self.render([obj]), most=0.0)
        group = Node(visible=False)
        obj.parent = group
        self.assertEqual((self.render([obj]).ids > 0).sum(), 0)

    def test_moving_a_joint_draws_the_frame_afresh_and_its_shadow(self):
        obj, joints = figure(bend=0.2)
        ground = floor()
        mover = Object3D(make_box(), position=np.array([2.5, 0.5, 0.0]), scale=0.3)  # (moved every frame: drawn)
        renderer = Renderer(60, 30, shadow_fit="scene")
        for k in range(SETTLE_DRAWS + 3):  # (the figure's row in the shadow map settles: drawn from its own map)
            mover.position = np.array([2.5, 0.5 + 0.01 * k, 0.0])
            self.render([obj, ground, mover], renderer=renderer)
        draws = renderer.draws
        joints[1].rotation = quat_axis_angle((0, 0, 1), 1.0)  # bent: nothing else changed
        fb = self.render([obj, ground, mover], renderer=renderer)
        self.assertEqual(renderer.draws, draws + 1)
        self.assert_same_picture(fb, self.render([baked(obj), ground, mover], shadow_fit="scene"))

    def test_objects_that_keep_their_pose_are_not_skinned_again(self):
        obj, joints = figure()
        other, _ = figure(position=(1.0, 0.0, 0.0))
        renderer = Renderer(40, 20)
        with unittest.mock.patch.object(skinning, "skin", wraps=skinning.skin) as kernel:
            renderer.render([obj, other], CAMERA, SUN)
            first = renderer._instances([obj, other], CAMERA)["serial"].copy()
            self.assertEqual(kernel.call_count, 1)
            obj.position = np.array([1.0, 0.0, 0.0])  # (its own place doesn't count)
            renderer.render([obj, other], CAMERA, SUN)
            self.assertEqual(kernel.call_count, 1)
            joints[2].rotation = quat_axis_angle((1, 0, 0), 0.3)
            renderer.render([obj, other], CAMERA, SUN)
            self.assertEqual(kernel.call_count, 2)
            serial = renderer._instances([obj, other], CAMERA)["serial"]
            self.assertNotEqual(serial[0], first[0])
            self.assertEqual(serial[1], first[1])
            self.assertEqual(kernel.call_args.args[11].shape[0], -(-len(obj.mesh.vertices) // 256))  # (its chunks)

    def test_bounds_hold_every_vertex_in_any_pose(self):
        mesh = tube_mesh(bones=4, rings=10, segments=16)  # (1280 faces: runs of their own)
        obj, joints = figure(mesh, bones=4)
        rng = np.random.default_rng(3)
        renderer = Renderer(40, 20)
        for _ in range(20):
            for joint in joints:
                joint.rotation = quat_axis_angle(rng.normal(size=3), rng.uniform(-1.5, 1.5))
                joint.scale = rng.uniform(0.5, 1.5, 3)
            joints[0].position = rng.normal(size=3)
            inst = renderer._instances([obj], CAMERA)
            bounds, b = inst["skin"][1], int(inst["bound"][0])
            positions = obj.skin.pose(mesh)[0]
            reach = np.linalg.norm(positions - bounds[b, :3], axis=1)
            self.assertTrue((reach <= bounds[b, 3] * (1 + 1e-9)).all())
            pack = inst["pack"]
            m = int(inst["mesh"][0])
            runs = range(int(pack["mesh_run"][m]), int(pack["mesh_run"][m + 1]))
            self.assertGreater(len(runs), 1)
            for q, r in enumerate(runs):
                faces = pack["faces"][pack["run_first"][r]:pack["run_end"][r]] - pack["mesh_vertex"][m]
                used = positions[np.unique(faces)]
                sphere = bounds[b + 1 + q]
                self.assertTrue((np.linalg.norm(used - sphere[:3], axis=1) <= sphere[3] * (1 + 1e-9)).all())

    def test_culled_before_skinning_only_where_nothing_out_of_view_shows(self):
        obj, _ = figure(position=(0.0, 0.0, 20.0))  # (behind the camera)
        box = Object3D(make_box())
        renderer = Renderer(40, 20)

        def drawn(lights, objects=(box, obj)):
            inst = renderer._instances(list(objects), CAMERA, lights)
            return 2 in inst["ident"].tolist()

        self.assertFalse(drawn([Light()]))  # (no shadows: out of view, it shows nowhere)
        self.assertFalse(drawn([PointLight(np.array([0.0, 1.0, 0.0]), range=3.0, shadows=True)]))  # (out of reach)
        self.assertTrue(drawn([PointLight(np.array([0.0, 1.0, 17.0]), range=3.0, shadows=True)]))  # (its shadow)
        self.assertTrue(drawn([SUN]))  # (a sun's map may hold what is out of view)
        mirror = Object3D(make_box(), reflectivity=0.8)
        self.assertTrue(drawn([Light()], (mirror, obj)))
        renderer.shadows = False
        self.assertFalse(drawn([SUN]))
        obj.skin.joints[0].position = np.array([0.0, 0.0, 0.0])  # (in view)
        renderer.shadows = True
        self.assertTrue(drawn([Light()]))

    def test_a_joint_that_is_not_finite_leaves_out_its_object(self):
        obj, joints = figure()
        box = Object3D(make_box(), position=np.array([1.5, 1.0, 0.0]))
        joints[1].position = np.array([np.nan, 0.0, 0.0])
        fb = self.render([obj, box])
        self.assertEqual(set(np.unique(fb.ids).tolist()), {0, 2})
        joints[1].position = np.array([0.0, 1.0, 0.0])
        self.assertIn(1, np.unique(self.render([obj, box]).ids).tolist())

    def test_never_a_mirror(self):
        flat = Mesh(np.array([(-1, 0, 0), (1, 0, 0), (1, 2, 0), (-1, 2, 0)], float), np.array([(0, 1, 2), (0, 2, 3)]),
                    joints=np.zeros((4, 4), np.int64), weights=np.tile([1.0, 0, 0, 0], (4, 1)))
        cape = Object3D(flat, skin=Skin([Node()]), reflectivity=0.9)
        renderer = Renderer(40, 20)
        renderer.render([cape, Object3D(make_box(), position=np.array([0.0, 0.5, 2.0]))], CAMERA, SUN)
        inst = renderer._instances([cape], CAMERA)
        self.assertFalse(renderer._mirrors(inst).any())

    def test_drawn_in_mirrors_from_the_rows_of_the_frame(self):
        obj, _ = figure(bend=0.5)
        mirror = Object3D(Mesh(np.array([(-3, 0, -2), (3, 0, -2), (3, 3, -2), (-3, 3, -2)], float),
                               np.array([(0, 1, 2), (0, 2, 3)])), reflectivity=0.9)
        card = Object3D(make_box(), position=np.array([1.5, 0.5, 0.0]), scale=0.3, billboard=True)  # (passes afresh)
        with unittest.mock.patch.object(skinning, "skin", wraps=skinning.skin) as kernel:
            with_skin = self.render([obj, mirror, card, floor()])
            self.assertEqual(kernel.call_count, 1)  # (the mirror's pass reused them)
        self.assert_same_picture(with_skin, self.render([baked(obj), mirror, card, floor()]))

    def test_among_particles_lines_and_levels_of_detail(self):
        obj, _ = figure(bend=0.5)
        sparks = Particles(seed=1)
        sparks.emit((0.0, 1.0, 1.0), 20, speed=1.0, size=0.05)
        sparks.update(0.1)
        beam = Lines([(-2.0, 2.0, 0.0)], [(2.0, 0.5, 0.0)], width=0.05, units="world")
        box = Object3D(make_box(), position=np.array([-1.5, 0.5, 0.0]))
        for options in ({}, {"simplify": 2.0}):
            with_skin = self.render([sparks, obj, beam, box, floor()], **options)
            self.assert_same_picture(with_skin, self.render([sparks, baked(obj), beam, box, floor()], **options))

    def test_listed_twice_or_seen_through_an_orthographic_camera(self):
        obj, _ = figure(bend=0.5)
        ortho = Camera(position=np.array([0.0, 1.5, 6.0]), target=np.array([0.0, 1.5, 0.0]), projection="ortho",
                       size=4.0)
        for camera in (CAMERA, ortho):
            self.assert_same_picture(self.render([obj, obj], [Light()], camera),
                                     self.render([baked(obj)] * 2, [Light()], camera))

    def test_mirrored_skeletons_turn_their_faces(self):
        obj, joints = figure(bend=0.5)
        joints[0].scale = (-1.0, 1.0, 1.0)
        positions, normals = obj.skin.pose(obj.mesh)
        inside_out = Object3D(Mesh(positions, obj.mesh.faces[:, [0, 2, 1]], normals=normals))
        self.assert_same_picture(self.render([obj, floor()]), self.render([inside_out, floor()]))

    def test_weights_made_safe(self):
        mesh = tube_mesh()
        mesh.weights = mesh.weights * 3.0
        mesh.weights[0] = (np.nan, -1.0, 0.0, 0.0)
        joints, weights = skinning.skin_arrays(mesh)
        np.testing.assert_allclose(weights.sum(axis=1), 1.0)
        np.testing.assert_array_equal(weights[0], (1.0, 0.0, 0.0, 0.0))
        self.assertEqual(mesh.weights[0, 1], -1.0)  # (the mesh's own left as they were)
        obj = Object3D(mesh, skin=Skin(chain(2)))  # (a skin with too few joints for the mesh)
        with self.assertRaises(ValueError):
            Renderer(10, 5).render([obj], CAMERA, SUN)
        with self.assertRaises(ValueError):
            Mesh(mesh.vertices, mesh.faces, joints=mesh.joints[:3], weights=mesh.weights).check()
        with self.assertRaises(ValueError):
            Skin(chain(2), inverse_bind=np.zeros((3, 4, 4)))

    def test_world_bounds_in_the_pose_drawn(self):
        obj, joints = figure(bend=0.0)
        np.testing.assert_allclose(obj.world_bounds()[1][1], 3.0)
        joints[0].position = np.array([0.0, 2.0, 0.0])
        lo, hi = obj.world_bounds()
        np.testing.assert_allclose((lo[1], hi[1]), (2.0, 5.0))
        box = Object3D(make_box(), position=np.array([0.0, -3.0, 0.0]))
        lo, hi = union_bounds([obj, box])
        np.testing.assert_allclose((lo[1], hi[1]), (-3.5, 5.0))
        root = Node(np.array([5.0, 0.0, 0.0]), scale=2.0)  # Model.bounds() and fit(): in root's space, as posed
        joints = chain(3, parent=root)
        model = Model(root, [Object3D(tube_mesh(), skin=Skin(joints), parent=root)])
        joints[0].position = np.array([0.0, 1.0, 0.0])
        lo, hi = model.bounds()
        np.testing.assert_allclose((lo[1], hi[1]), (1.0, 2.5))  # (bound where it stood in the world: 0..3)
        model.fit(2.0)
        lo, hi = model.world_bounds()
        np.testing.assert_allclose((lo[1], hi[1]), (-1.0, 1.0), atol=1e-12)

    def test_copies_follow_their_own_joints(self):
        outside = Node(position=np.array([0.0, 1.0, 0.0]))  # (a joint outside the model: shared)
        root = Node()
        joints = chain(2, parent=root)
        obj = Object3D(tube_mesh(2), skin=Skin(joints + [outside]), parent=root)
        model = Model(root, [obj], nodes={"hip": joints[0]})
        twin = model.copy()
        copied = twin.objects[0]
        self.assertIsNot(copied.skin, obj.skin)
        self.assertIs(copied.skin.joints[0], twin.nodes["hip"])
        self.assertIsNot(copied.skin.joints[1], joints[1])
        self.assertIs(copied.skin.joints[1].parent, copied.skin.joints[0])
        self.assertIs(copied.skin.joints[2], outside)
        self.assertIs(copied.skin.inverse_bind, obj.skin.inverse_bind)
        twin.root.position = np.array([3.0, 0.0, 0.0])
        np.testing.assert_allclose(copied.skin.pose(copied.mesh)[0][:, 0] - obj.skin.pose(obj.mesh)[0][:, 0], 3.0,
                                   atol=1e-12)

    def test_baked_as_posed_now(self):
        root = Node(np.array([0.3, 0.2, 0.0]), quat_axis_angle((0, 1, 0), 0.7), 0.8)
        joints = chain(3, parent=root)
        obj = Object3D(tube_mesh(), skin=Skin(joints), parent=root, color=(220, 160, 120))
        box = Object3D(make_box(), position=np.array([1.2, 0.5, 0.0]), parent=root, color=(220, 160, 120))
        model = Model(root, [obj, box])
        for joint in joints[1:]:
            joint.rotation = quat_axis_angle((0.0, 0.0, 1.0), 0.6)
        for mirrored in (False, True):
            joints[0].scale = (-1.0, 1.0, 1.0) if mirrored else 1.0
            still = model.bake()
            self.assertEqual(len(still.objects), 1)  # (the figure and the box alike: one mesh)
            for part in still.objects:
                part.simplify = False  # (skinned parts aren't simplified)
            live, baked_ = self.render([*model, floor()]), self.render([*still, floor()])
            live.ids = np.array([0, 1, 1, 2])[live.ids]  # (the figure and the box drawn as one object)
            self.assert_same_picture(live, baked_)
            joints[0].position = np.array([0.0, 1.0, 0.0])  # (the still stays as it was)
            self.assertFalse(np.allclose(union_bounds(still.objects)[1], union_bounds(model.objects)[1]))
            joints[0].position = np.zeros(3)


if __name__ == "__main__":
    unittest.main()
