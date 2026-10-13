# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
"""Mistakes that would otherwise go unnoticed (a change that isn't drawn, an error far from its cause), made loud: an
error that says what is wrong, or a warning (some only in a playtest, where checking costs nothing that matters)."""
import tempfile
import unittest
import warnings

import numpy as np

from unicode3d.animation import Animation, Clip, SplineTrack, Track
from unicode3d.mesh import Mesh, make_box
from unicode3d.playtest import Playtest
from unicode3d.queries import Colliders
from unicode3d.scene import Camera, Light, Node, Object3D, Renderer, Skin

CAMERA = Camera(position=np.array([0.0, 0.0, 5.0]))


class TrackTests(unittest.TestCase):
    def test_keyframes_cannot_be_changed(self):
        # A Clip reads a track's keyframes once, so an edit would be ignored: it raises instead.
        track = Track([(0.0, (0.0, 0.0, 0.0)), (1.0, (1.0, 0.0, 0.0))])
        with self.assertRaises(ValueError):
            track.values[1][0] = 5.0
        with self.assertRaises(ValueError):
            track.times[1] = 2.0
        with self.assertRaises(TypeError):
            track.values[1] = np.array([5.0, 0.0, 0.0])
        for name in ("times", "values", "easings", "loop"):
            with self.assertRaises(AttributeError):
                setattr(track, name, getattr(track, name))
        spline = SplineTrack([(0.0, 0.0, 0.0, 1.0), (1.0, 1.0, 1.0, 0.0)])
        with self.assertRaises(AttributeError):
            spline.tangents = ()

    def test_a_new_track_plays(self):
        obj = Object3D(make_box())
        clip = Clip([Animation(obj, position=Track([(0.0, (0.0, 0.0, 0.0)), (1.0, (1.0, 0.0, 0.0))]))])
        clip.apply(1.0)
        clip.animations[0].tracks["position"] = Track([(0.0, (0.0, 0.0, 0.0)), (1.0, (5.0, 0.0, 0.0))])
        clip.apply(1.0)
        np.testing.assert_allclose(obj.position, (5.0, 0.0, 0.0))

    def test_given_arrays_stay_the_callers(self):
        value = np.array([1.0, 2.0, 3.0])
        Track([(0.0, value)])
        value[0] = 4.0  # (the track copied it)


class MessageTests(unittest.TestCase):
    def render(self, **kw):
        Renderer(20, 10).render([Object3D(make_box(), **kw)], CAMERA, Light())

    def test_wrong_shapes_say_what_is_wanted(self):
        for kw, words in [(dict(rotation=(0.0, 45.0, 0.0)), "quaternion"), (dict(position=(1.0, 2.0)), "three numbers"),
                          (dict(position="here"), "three numbers"), (dict(color="red"), "Color.RED")]:
            with self.subTest(kw=kw), self.assertRaisesRegex(ValueError, words):
                self.render(**kw)


def skinned():
    flat = Mesh(np.array([(-1, 0, 0), (1, 0, 0), (1, 2, 0), (-1, 2, 0)], float), np.array([(0, 1, 2), (0, 2, 3)]),
                joints=np.zeros((4, 4), np.int64), weights=np.tile([1.0, 0, 0, 0], (4, 1)))
    return Object3D(flat, skin=Skin([Node()]))


class ColliderTests(unittest.TestCase):
    def test_a_skinned_object_warns_once(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            solid = Colliders([skinned(), skinned(), Object3D(make_box())])
            solid.update()
        self.assertEqual(len(caught), 1)
        self.assertIn("bind", str(caught[0].message))
        self.assertEqual(caught[0].filename, __file__)  # (pointing at the program's line)

    def test_no_warning_without_one(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            Colliders([Object3D(make_box())]).update()


class Editor:
    """A program editing a mesh in place in frame 3: with or without invalidate(), for a renderer or Colliders."""

    def __init__(self, invalidate, colliders=False):
        self.renderer = Renderer(1, 1)
        self.box = Object3D(make_box())
        self.solid = Colliders([self.box]) if colliders else None
        self.invalidate, self.count = invalidate, 0

    def frame(self, screen, dt, events):
        self.count += 1
        if self.count == 3:
            self.box.mesh.vertices *= 2.0
            if self.invalidate:
                self.renderer.invalidate()
                if self.solid is not None:
                    self.solid.invalidate()
        if self.solid is not None:
            self.solid.update()
        rows, cols = screen.size()
        self.renderer.resize(cols, rows, screen.cell_pixels)
        screen.draw_frame(self.renderer.render([self.box], CAMERA, Light()))
        screen.refresh()


class Painter(Editor):
    """A textured box (a 1.5 MB picture: six slices of mesh.CHECK_BYTES) with three rows of the picture painted over in place at frame
    paint_at (None: never)."""

    def __init__(self, paint_at):
        super().__init__(invalidate=False)
        self.picture = np.full((256, 256, 3), 0.5)
        self.box = Object3D(make_box(1.0, [self.picture] * 6), color=(255, 255, 255))
        self.paint_at, self.count = paint_at, 10  # (past the frame Editor edits the vertices in)

    def frame(self, screen, dt, events):
        if self.count - 9 == self.paint_at:
            self.picture[100:103] = 0.0
        super().frame(screen, dt, events)


class EditCheckTests(unittest.TestCase):
    def play(self, game, frames=5, **kw):
        out = tempfile.TemporaryDirectory()
        self.addCleanup(out.cleanup)
        test = Playtest(game, size=(10, 30), out=out.name, compile=False, **kw)
        test.wait(frames)
        return test.report().warnings

    def test_an_edit_in_place_is_a_warning(self):
        for colliders in (False, True):
            with self.subTest(colliders=colliders):
                found = self.play(Editor(invalidate=False, colliders=colliders))
                messages = [m for _, m in found if "edited in place" in m]
                self.assertTrue(messages, found)
                self.assertIn("colliders.invalidate()" if colliders else "renderer.invalidate()", messages[0])
                self.assertIn("test_hazards.py", messages[0])  # (pointing at the program's line)
                self.assertEqual(found[0][0], 3)

    def test_a_few_rows_of_a_big_texture(self):
        # Checked a slice of rows a frame: the edit shows within as many frames as there are slices.
        for paint_at in (None, 3):
            with self.subTest(paint_at=paint_at):
                found = self.play(Painter(paint_at), frames=12)
                if paint_at is None:
                    self.assertEqual(found, [])
                else:
                    self.assertTrue(found and "edited in place" in found[0][1], found)
                    self.assertLessEqual(found[0][0], paint_at + 6)

    def test_not_with_invalidate(self):
        for colliders in (False, True):
            with self.subTest(colliders=colliders):
                self.assertEqual(self.play(Editor(invalidate=True, colliders=colliders)), [])

    def test_only_in_a_playtest_unless_asked(self):
        self.assertEqual(self.play(Editor(invalidate=False), check_edits=False), [])
        from unicode3d import mesh
        self.assertFalse(mesh.CHECK_EDITS)  # (put back after each frame)
        game = Editor(invalidate=False)
        renderer = Renderer(10, 10)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            for _ in range(3):
                renderer.render([game.box], CAMERA, Light())
                game.box.mesh.vertices *= 1.5


if __name__ == "__main__":
    unittest.main()
