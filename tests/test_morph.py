# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
"""Morph targets (Mesh.morph_targets, Object3D.morph_weights): drawn morphed, with or without a skin, kept from
frame to frame, bounded, and played by Clips."""
import types
import unittest
import unittest.mock

import numpy as np

from unicode3d import skinning
from unicode3d.animation import Animation, Clip, SplineTrack, Track
from unicode3d.mesh import Mesh
from unicode3d.scene import Camera, Light, Model, Node, Object3D, PointLight, Renderer, Skin
from unicode3d.shadows import SETTLE_DRAWS
from unicode3d.skinning import posing
from unicode3d.transforms import quat_axis_angle

from .figures import chain, figure, tube_mesh

CAMERA = Camera(position=np.array([0.0, 1.5, 6.0]), target=np.array([0.0, 1.5, 0.0]))
SUN = Light(direction=np.array([0.4, -1.0, -0.5]), shadows=True)
LAMP = [PointLight(np.array([1.0, 2.5, 1.5]), range=6.0, shadows=True)]


def floor():
    return Object3D(Mesh(np.array([(-4, 0, -4), (4, 0, -4), (4, 0, 4), (-4, 0, 4)], float) - (0, 0.01, 0),
                         np.array([(0, 2, 1), (0, 3, 2)])), color=(200, 200, 200))


def morphing_tube(bones=3, rings=6, segments=10):
    """tube_mesh() with three targets: a bulge (wider in the middle), a lean (its top moved along x) and a twist
    of its rings, the bulge and twist with normal offsets too."""
    mesh = tube_mesh(bones, rings=rings, segments=segments)
    v = mesh.vertices
    height = v[:, 1] / v[:, 1].max()
    middle = np.sin(np.pi * height)[:, None]
    bulge = v * (1.0, 0.0, 1.0) * middle
    lean = np.zeros_like(v)
    lean[:, 0] = 0.6 * height ** 2
    twist = np.stack([-v[:, 2], np.zeros(len(v)), v[:, 0]], axis=1) * height[:, None]
    mesh.morph_targets = np.stack([bulge, lean, twist])
    normals = mesh.vertex_normals()
    turns = np.zeros((3, len(v), 3))
    turns[0, :, 1] = -0.3 * np.cos(np.pi * height)
    turns[2] = np.stack([-normals[:, 2], np.zeros(len(v)), normals[:, 0]], axis=1) * height[:, None]
    mesh.morph_normals = turns
    return mesh


def baked(obj):
    """An Object3D showing obj's mesh morphed (and posed) as now, as a plain mesh in the world."""
    positions, normals = posing(obj).pose(obj.mesh, morph_weights=obj.morph_weights)
    inside_out = np.linalg.det(posing(obj).palette()[:, :3, :3]).sum() < 0.0  # (drawn with its faces turned)
    mesh = Mesh(positions, obj.mesh.faces[:, [0, 2, 1]] if inside_out else obj.mesh.faces, normals=normals)
    return Object3D(mesh, color=obj.color, double_sided=obj.double_sided, simplify=False)


class MorphTests(unittest.TestCase):
    def assert_same_picture(self, a, b, most=0.002):
        differ = (np.abs(a.rgb - b.rgb).max(axis=2) > 1e-5) | (a.ids != b.ids)
        self.assertLessEqual(differ.mean(), most, f"{int(differ.sum())} pixels differ")
        self.assertGreater((a.ids > 0).sum(), 100)  # (something drawn)

    def render(self, objects, lights=SUN, camera=CAMERA, renderer=None, **options):
        renderer = renderer or Renderer(60, 30, **options)
        fb = renderer.render(objects, camera, lights)
        return types.SimpleNamespace(rgb=fb.rgb.copy(), ids=fb.ids.copy())

    def test_the_kernel_morphs_as_pose_does(self):
        obj, joints = figure(morphing_tube(), bend=0.6)  # skinned and morphed
        obj.morph_weights = np.array([0.7, -0.4, 1.3])
        joints[1].scale = (1.0, 1.3, 0.7)
        alone = Object3D(morphing_tube(), position=np.array([1.0, 0.2, -0.5]), rotation=quat_axis_angle((1, 1, 0), 0.4),
                         scale=(1.2, 0.8, 1.0), morph_weights=[0.0, 0.9, -0.6])  # morphed, posed by itself
        renderer = Renderer(40, 20)
        renderer.render([obj, alone], CAMERA, SUN)
        inst = renderer._instances([obj, alone], CAMERA)
        for k, o in enumerate((obj, alone)):
            start = int(inst["src"][k])
            positions, normals = posing(o).pose(o.mesh, morph_weights=o.morph_weights)
            rows = inst["skin"][0][:, start:start + len(positions)]
            np.testing.assert_allclose(rows[0], positions, atol=1e-12)
            np.testing.assert_allclose(rows[1] / np.linalg.norm(rows[1], axis=1, keepdims=True), normals, atol=1e-12)
        unmorphed = posing(alone).pose(alone.mesh)[0]
        self.assertGreater(np.abs(unmorphed - posing(alone).pose(alone.mesh, morph_weights=[0, 0.9, -0.6])[0]).max(),
                           0.3)

    def test_drawn_as_the_same_morph_baked_into_a_mesh(self):
        obj, joints = figure(morphing_tube(), bend=0.5, color=(220, 160, 120))
        obj.morph_weights = [1.0, 0.5, 0.4]
        joints[0].position = np.array([-0.8, 0.0, 0.3])
        alone = Object3D(morphing_tube(), position=np.array([1.0, 0.0, 0.0]), rotation=quat_axis_angle((0, 1, 0), 0.8),
                         scale=0.7, color=(160, 200, 120), morph_weights=[0.8, -1.0, 0.7])
        mirrored = Object3D(morphing_tube(), position=np.array([0.2, 0.0, -1.5]), scale=(-0.6, 0.6, 0.6),
                            morph_weights=[1.0, 1.0, 0.0])  # (turned inside out by its scale: faces turned)
        for lights in (SUN, LAMP):  # (the sun's map fitted to the whole scene: fitted to the view, a skinned or
            # morphed object's sphere, wider than a mesh's box, fits it otherwise, and shadows' edges move a little)
            self.assert_same_picture(self.render([obj, alone, mirrored, floor()], lights, shadow_fit="scene"),
                                     self.render([baked(obj), baked(alone), baked(mirrored), floor()], lights,
                                                 shadow_fit="scene"))

    def test_without_weights_drawn_as_its_mesh_is(self):
        mesh = morphing_tube()
        obj = Object3D(mesh, position=np.array([0.0, 0.0, 0.0]))
        renderer = Renderer(40, 20)
        inst = renderer._instances([obj], CAMERA)
        self.assertEqual(inst["src"][0], -1)  # (not posed by the skinning pass at all)
        plain = Object3D(Mesh(mesh.vertices, mesh.faces))
        self.assert_same_picture(self.render([obj, floor()]), self.render([plain, floor()]), most=0.0)
        obj.morph_weights = np.zeros(3)  # (weights of 0: through the pass, the same picture)
        self.assert_same_picture(self.render([obj, floor()]), self.render([plain, floor()]))

    def test_morphed_again_when_its_weights_change_or_it_moves(self):
        obj = Object3D(morphing_tube(), morph_weights=np.array([0.5, 0.0, 0.0]))
        other, _ = figure(morphing_tube(), position=(1.5, 0.0, 0.0))
        other.morph_weights = np.array([0.0, 0.3, 0.0])
        ground = floor()
        mover = Object3D(Mesh(np.eye(3), np.array([(0, 1, 2)])), position=np.array([2.5, 0.5, 0.0]))
        renderer = Renderer(60, 30, shadow_fit="scene")
        with unittest.mock.patch.object(skinning, "skin", wraps=skinning.skin) as kernel:
            for k in range(SETTLE_DRAWS + 3):  # (their rows in the shadow map settle)
                mover.position = np.array([2.5, 0.5 + 0.01 * k, 0.0])
                self.render([obj, other, ground, mover], renderer=renderer)
            self.assertEqual(kernel.call_count, 1)
            draws = renderer.draws
            obj.morph_weights[0] = 1.0  # (changed in place: nothing else did)
            fb = self.render([obj, other, ground, mover], renderer=renderer)
            self.assertEqual(renderer.draws, draws + 1)
            self.assertEqual(kernel.call_count, 2)
            self.assertEqual(kernel.call_args.args[11].shape[0], 1)  # (only its chunk)
            self.assertEqual(len(kernel.call_args.args[8]), 1)  # (by the one target whose weight isn't 0)
            self.assert_same_picture(fb, self.render([baked(obj), baked(other), ground, mover], shadow_fit="scene"))
            obj.position = np.array([-1.0, 0.0, 0.0])  # (posed by itself: moving it moves its rows)
            fb = self.render([obj, other, ground, mover], renderer=renderer)
            self.assertEqual(kernel.call_count, 3)
            self.assert_same_picture(fb, self.render([baked(obj), baked(other), ground, mover], shadow_fit="scene"))
            self.render([obj, other, ground, mover], renderer=renderer)
            self.assertEqual(kernel.call_count, 3)

    def test_bounds_hold_every_vertex_however_it_morphs(self):
        mesh = morphing_tube(bones=4, rings=10, segments=16)  # (1280 faces: runs of their own)
        obj, joints = figure(mesh, bones=4)
        alone = Object3D(mesh)
        rng = np.random.default_rng(5)
        renderer = Renderer(40, 20)
        for _ in range(20):
            for joint in joints:
                joint.rotation = quat_axis_angle(rng.normal(size=3), rng.uniform(-1.5, 1.5))
                joint.scale = rng.uniform(0.5, 1.5, 3)
            obj.morph_weights = rng.uniform(-2.0, 2.0, 3)
            alone.morph_weights = rng.uniform(-2.0, 2.0, 3)
            alone.position, alone.scale = rng.normal(size=3), rng.uniform(0.3, 2.0, 3)
            inst = renderer._instances([obj, alone], CAMERA)
            pack = inst["pack"]
            for k, o in enumerate((obj, alone)):
                bounds, b = inst["skin"][1], int(inst["bound"][k])
                positions = posing(o).pose(mesh, morph_weights=o.morph_weights)[0]
                reach = np.linalg.norm(positions - bounds[b, :3], axis=1)
                self.assertTrue((reach <= bounds[b, 3] * (1 + 1e-9)).all())
                m = int(inst["mesh"][k])
                runs = range(int(pack["mesh_run"][m]), int(pack["mesh_run"][m + 1]))
                self.assertGreater(len(runs), 1)
                for q, r in enumerate(runs):
                    faces = pack["faces"][pack["run_first"][r]:pack["run_end"][r]] - pack["mesh_vertex"][m]
                    sphere = bounds[b + 1 + q]
                    used = positions[np.unique(faces)]
                    self.assertTrue((np.linalg.norm(used - sphere[:3], axis=1) <= sphere[3] * (1 + 1e-9)).all())

    def test_weights_and_targets_checked(self):
        mesh = morphing_tube()
        obj = Object3D(mesh, morph_weights=[1.0, 0.0])  # (one too few)
        with self.assertRaises(ValueError):
            Renderer(10, 5).render([obj], CAMERA, SUN)
        obj.morph_weights = [np.nan, 0.0, np.inf]  # (numbers that aren't finite count as 0)
        plain = Object3D(Mesh(mesh.vertices, mesh.faces))
        self.assert_same_picture(self.render([obj, floor()]), self.render([plain, floor()]))
        with self.assertRaises(ValueError):
            Mesh(mesh.vertices, mesh.faces, morph_targets=mesh.morph_targets[:, :5]).check()
        with self.assertRaises(ValueError):
            Mesh(mesh.vertices, mesh.faces, morph_targets=mesh.morph_targets, morph_normals=np.zeros((1, 3))).check()
        broken = morphing_tube()
        broken.morph_targets[1, 4] = np.nan  # (an offset that isn't finite: none)
        obj = Object3D(broken, morph_weights=[0.0, 1.0, 0.0])
        self.assertTrue(np.isfinite(self.render([obj]).rgb).all())

    def test_world_bounds_bake_and_copies_as_morphed(self):
        root = Node(np.array([0.3, 0.2, 0.0]), quat_axis_angle((0, 1, 0), 0.7), 0.8)
        alone = Object3D(morphing_tube(), position=np.array([1.2, 0.0, 0.0]), parent=root, color=(220, 160, 120),
                         morph_weights=[1.0, 1.0, 0.0])
        joints = chain(3, parent=root)
        skinned = Object3D(morphing_tube(), skin=Skin(joints), parent=root, color=(220, 160, 120),
                           morph_weights=[0.0, -1.0, 0.5])
        model = Model(root, [alone, skinned])
        lo, hi = alone.world_bounds()
        posed = posing(alone).pose(alone.mesh, morph_weights=alone.morph_weights)[0]
        np.testing.assert_allclose((lo, hi), (posed.min(axis=0), posed.max(axis=0)))
        still = model.bake()
        for part in still.objects:
            part.simplify = False
        live, baked_ = (self.render([*model, floor()], shadow_fit="scene"),
                        self.render([*still, floor()], shadow_fit="scene"))
        live.ids = np.array([0, 1, 1, 2])[live.ids]  # (the two drawn as one object)
        self.assert_same_picture(live, baked_)
        twin = model.copy()
        twin.root.position = np.array([-2.0, 0.0, 0.0])
        copied = twin.objects[0]
        self.assertIs(posing(copied).joints[0], copied)  # (posed by itself, not by the original)
        copied.morph_weights[1] = 0.0  # (weights of its own)
        self.assertEqual(alone.morph_weights[1], 1.0)
        posed = posing(copied).pose(copied.mesh, morph_weights=copied.morph_weights)[0]
        np.testing.assert_allclose(copied.world_bounds(), (posed.min(axis=0), posed.max(axis=0)))
        np.testing.assert_allclose(posed - (-2.3, -0.2, 0.0), posing(alone).pose(alone.mesh, morph_weights=[1, 0, 0])[0],
                                   atol=1e-12)
        drawn = self.render([*model, *twin, floor()])
        self.assertTrue(set(np.unique(drawn.ids)) >= {1, 2, 3, 4})


class MorphClipTests(unittest.TestCase):
    def tracks(self, k):
        rng = np.random.default_rng(k)
        times = [0.0, 0.4, 1.1, 1.5]
        return [Track([(t, rng.uniform(-1, 2, k)) for t in times]),
                Track([(t, rng.uniform(-1, 2, k), e) for t, e in zip(times, ("linear", "ease_out", "step", "ease_in"))]),
                SplineTrack([(t, rng.uniform(-1, 2, k), rng.normal(size=k), rng.normal(size=k)) for t in times])]

    def test_played_by_the_kernel_as_track_at_gives_them(self):
        for k in (1, 3, 4, 5, 9):
            for track in self.tracks(k):
                targets = [Object3D(None), Object3D(None)]
                nodes = Node()
                moves = [Animation(targets[0], morph_weights=track),
                         Animation(nodes, position=Track([(0.0, (0, 0, 0)), (1.5, (1, 2, 3))])),
                         Animation(targets[1], morph_weights=track, rotation=None)]
                clip, slow = Clip(moves, loop="pingpong"), Clip(moves, loop="pingpong")
                slow._kernel = False
                self.assertTrue(clip._pack().ok)
                for t in np.linspace(-0.2, 3.7, 23):
                    clip.apply(t)
                    fast = [np.array(o.morph_weights) for o in targets]
                    slow.apply(t)
                    for o, w in zip(targets, fast):
                        self.assertEqual(w.shape, (k,))
                        np.testing.assert_array_equal(w, o.morph_weights)

    def test_fades_from_the_weights_it_finds(self):
        track = self.tracks(5)[0]
        for start in (None, np.full(5, 0.5)):
            target, other = Object3D(None, morph_weights=start), Object3D(None, morph_weights=start)
            clip, slow = Clip([Animation(target, morph_weights=track)]), Clip([Animation(other, morph_weights=track)])
            slow._kernel = False
            clip.start(fade=0.5)
            slow.start(fade=0.5)
            for _ in range(8):
                clip.update(0.1)
                slow.update(0.1)
                np.testing.assert_allclose(target.morph_weights, other.morph_weights, atol=1e-12)
            np.testing.assert_array_equal(target.morph_weights, track.at(clip.time))  # (faded in)

    def test_drawn_as_the_clip_poses_it(self):
        mesh = morphing_tube()
        obj = Object3D(mesh, morph_weights=np.zeros(3))
        clip = Clip([Animation(obj, morph_weights=Track([(0.0, (0.0, 0.0, 0.0)), (1.0, (1.0, 0.5, -0.5))]))])
        renderer = Renderer(40, 20)
        before = renderer.render([obj], CAMERA, SUN).ids.copy()
        clip.update(0.5)
        np.testing.assert_allclose(obj.morph_weights, (0.5, 0.25, -0.25))
        after = renderer.render([obj], CAMERA, SUN).ids
        self.assertFalse(np.array_equal(before, after))


if __name__ == "__main__":
    unittest.main()
