# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
"""Ray and overlap queries (Colliders)."""
import unittest

import numpy as np

from unicode3d import Colliders, Mesh, Model, Node, Object3D, make_box
from unicode3d.queries import mesh_tree
from unicode3d.shapes import blob_mesh
from unicode3d.transforms import quat_axis_angle


def nearest_by_brute_force(objects, origin, direction):
    """(distance, object) of the nearest triangle a ray meets, testing every one in the world; (inf, None) if none."""
    best = (np.inf, None)
    for obj in objects:
        lin, pos, _ = obj.world_matrix()
        tri = (np.asarray(obj.mesh.vertices, float) @ lin.T + pos)[obj.mesh.faces]
        a, e1, e2 = tri[:, 0], tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
        p = np.cross(direction, e2)
        det = (e1 * p).sum(axis=1)
        with np.errstate(all="ignore"):
            s = origin - a
            u = (s * p).sum(axis=1) / det
            q = np.cross(s, e1)
            v = (q * direction).sum(axis=1) / det
            t = (e2 * q).sum(axis=1) / det
            hit = (np.abs(det) > 0) & (u >= 0) & (u <= 1) & (v >= 0) & (u + v <= 1) & (t >= 0)
        if hit.any() and t[hit].min() < best[0]:
            best = (t[hit].min(), obj)
    return best


def floor(size=10.0, y=0.0):
    """A square facing up."""
    return Mesh(np.array([(-size, y, -size), (-size, y, size), (size, y, size), (size, y, -size)], float),
                np.array([(0, 1, 2), (0, 2, 3)]))


class RayTests(unittest.TestCase):
    def test_nearest_hit(self):
        near = Object3D(make_box(), np.array([0.0, 0.0, -3.0]))
        far = Object3D(make_box(), np.array([0.0, 0.0, -6.0]))
        solid = Colliders([far, near])
        hit = solid.raycast((0.0, 0.2, 0.0), (0.0, 0.0, -2.0))  # (any length of direction)
        self.assertIs(hit.object, near)
        self.assertAlmostEqual(hit.distance, 2.5)
        np.testing.assert_allclose(hit.position, [0.0, 0.2, -2.5])
        np.testing.assert_allclose(hit.normal, [0.0, 0.0, 1.0], atol=1e-12)
        self.assertTrue(hit.front)
        self.assertTrue(0 <= hit.face < 12)
        np.testing.assert_allclose(make_box().vertices[make_box().faces[hit.face]][:, 2], 0.5)  # the +z face
        self.assertIsNone(solid.raycast((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)))  # behind
        self.assertIsNone(solid.raycast((0.0, 0.0, 0.0), (0.0, 0.0, -1.0), max_distance=2.0))  # too far
        self.assertIs(solid.raycast((0.0, 0.0, 0.0), (0.0, 0.0, -1.0), ignore=[near]).object, far)
        # From inside a box, its back face: the normal is on the side the ray came from.
        inside = solid.raycast((0.0, 0.0, -3.0), (1.0, 0.0, 0.0))
        self.assertFalse(inside.front)
        np.testing.assert_allclose(inside.normal, [-1.0, 0.0, 0.0], atol=1e-12)

    def test_every_hit(self):
        boxes = [Object3D(make_box(), np.array([0.0, 0.0, -3.0 * i])) for i in (1, 2)]
        hits = Colliders(boxes).raycast((0.0, 0.0, 0.0), (0.0, 0.0, -1.0), all=True)  # through edges of faces
        self.assertEqual([round(h.distance, 9) for h in hits], [2.5, 3.5, 5.5, 6.5])  # once each, nearest first
        self.assertEqual([h.front for h in hits], [True, False, True, False])
        self.assertEqual([h.object for h in hits], [boxes[0], boxes[0], boxes[1], boxes[1]])
        self.assertEqual(Colliders(boxes).raycast((0.0, 5.0, 0.0), (0.0, 0.0, -1.0), all=True), [])

    def test_same_as_testing_every_triangle(self):
        # Parents, scales along each axis, a mirroring, a turned box, and a soup of large crossing triangles.
        rng = np.random.default_rng(3)
        soup = Mesh(rng.normal(size=(600, 3)) * 3, rng.integers(0, 600, (400, 3)))
        parent = Node(position=np.array([1.0, 0.0, 0.0]), rotation=quat_axis_angle((0, 1, 0), 0.7), scale=(1, 1.5, 0.7))
        objects = [Object3D(soup),
                   Object3D(blob_mesh((1, 1, 1), rings=16, segments=24), np.array([0, 2.0, 1]), scale=(1, -2, 1),
                            parent=parent),
                   Object3D(make_box(), np.array([-3.0, 0, 0]), quat_axis_angle((1, 1, 0), 1.0), scale=(2, 0.5, 3))]
        solid = Colliders(objects)
        origins, directions = rng.normal(size=(400, 3)) * 8, rng.normal(size=(400, 3))
        many = solid.raycast_many(origins, directions)
        for r in range(len(origins)):
            d = directions[r] / np.linalg.norm(directions[r])
            distance, obj = nearest_by_brute_force(objects, origins[r], d)
            hit = solid.raycast(origins[r], directions[r])
            if obj is None:
                self.assertIsNone(hit)
                self.assertIsNone(many[0][r])
                self.assertEqual(many[1][r], np.inf)
                continue
            self.assertIs(hit.object, obj)
            self.assertAlmostEqual(hit.distance, distance, delta=1e-9 * (1 + distance))
            self.assertIs(many[0][r], obj)
            self.assertAlmostEqual(many[1][r], hit.distance, delta=1e-12 * (1 + distance))
            np.testing.assert_allclose(many[2][r], hit.position)
            np.testing.assert_allclose(many[3][r], hit.normal)
            self.assertEqual(many[4][r], hit.face)

    def test_many_rays(self):
        solid = Colliders([Object3D(floor())])
        xs = np.linspace(-12.0, 12.0, 25)
        objects, distances, positions, normals, faces = solid.raycast_many(
            np.stack([xs, np.full(25, 2.0), np.zeros(25)], axis=1), (0.0, -1.0, 0.0), max_distance=[3.0] * 24 + [1.0])
        hit = np.abs(xs) <= 10.0
        hit[-1] = False  # (beyond its max_distance, and off the floor anyway)
        self.assertEqual([o is not None for o in objects], hit.tolist())  # (the floor's edges count)
        np.testing.assert_array_equal(distances[hit], 2.0)
        self.assertTrue(np.isinf(distances[~hit]).all())
        np.testing.assert_allclose(normals[hit], [[0.0, 1.0, 0.0]] * hit.sum())
        np.testing.assert_array_equal(faces[~hit], -1)
        self.assertTrue(np.isnan(positions[~hit]).all())

    def test_models_parents_and_moving(self):
        part = Object3D(make_box())
        model = Model(root=Node(position=np.array([0.0, 0.0, -5.0])), objects=[part])
        part.parent = model.root
        solid = Colliders([model])
        self.assertIs(solid.raycast((0, 0, 0), (0, 0, -1)).object, part)
        self.assertAlmostEqual(solid.raycast((0, 0, 0), (0, 0, -1)).distance, 4.5)
        model.root.position = np.array([0.0, 0.0, -8.0])
        self.assertAlmostEqual(solid.raycast((0, 0, 0), (0, 0, -1)).distance, 4.5)  # until update()
        solid.update()
        self.assertAlmostEqual(solid.raycast((0, 0, 0), (0, 0, -1)).distance, 7.5)
        # Objects added to the list count from the next update; hidden ones count as solid.
        wall = Object3D(make_box(), np.array([0.0, 0.0, -3.0]), visible=False)
        solid.objects.append(wall)
        solid.update()
        self.assertIs(solid.raycast((0, 0, 0), (0, 0, -1)).object, wall)
        self.assertIsNone(solid.raycast((0, 0, 0), (0, 0, -1), ignore=[wall, model]))

    def test_trees_are_shared_and_rebuilt_when_meshes_change(self):
        mesh = make_box()
        a, b = Colliders([Object3D(mesh)]), Colliders([Object3D(mesh, np.array([3.0, 0.0, 0.0]))])
        self.assertIs(a._packed[0][0], b._packed[0][0])
        mesh.vertices = mesh.vertices * 2  # replaced: built afresh on the next update
        a.update()
        self.assertAlmostEqual(a.raycast((0, 0, 5), (0, 0, -1)).distance, 4.0)
        mesh.vertices *= 2  # edited in place: seen after invalidate()
        a.update()
        self.assertAlmostEqual(a.raycast((0, 0, 5), (0, 0, -1)).distance, 4.0)
        a.invalidate()
        self.assertAlmostEqual(a.raycast((0, 0, 5), (0, 0, -1)).distance, 3.0)

    def test_large_mesh(self):
        mesh = blob_mesh((1.0, 1.0, 1.0), rings=200, segments=160)  # 64,000 triangles
        tree = mesh_tree(mesh)
        self.assertLessEqual(tree.depth, 48)
        self.assertEqual(sorted(tree.face), list(range(len(mesh.faces))))  # every face, once
        hit = Colliders([Object3D(mesh)]).raycast((0.3, 0.2, 5.0), (0.0, 0.0, -1.0))
        self.assertAlmostEqual(hit.distance, 5.0 - np.sqrt(1 - 0.3 ** 2 - 0.2 ** 2), delta=1e-3)


class OverlapTests(unittest.TestCase):
    def test_sphere(self):
        box = Object3D(make_box(2.0))  # -1..1
        solid = Colliders([box])
        (contact,) = solid.overlap_sphere((0.0, 1.3, 0.2), 0.5)
        self.assertIs(contact.object, box)
        self.assertAlmostEqual(contact.depth, 0.2)
        np.testing.assert_allclose(contact.normal, [0.0, 1.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(contact.point, [0.0, 1.0, 0.2], atol=1e-12)
        self.assertEqual(solid.overlap_sphere((0.0, 1.6, 0.0), 0.5), [])
        # Over an edge: one contact, pushing away from the edge.
        (contact,) = solid.overlap_sphere((1.2, 1.2, 0.0), 0.5)
        np.testing.assert_allclose(contact.normal, [np.sqrt(0.5), np.sqrt(0.5), 0.0], atol=1e-12)
        self.assertAlmostEqual(contact.depth, 0.5 - np.sqrt(0.08))
        # In a corner of a room: one contact for each wall, deepest first.
        room = Object3D(Mesh(make_box(4.0).vertices, make_box(4.0).faces[:, ::-1]))  # faces turned inward
        contacts = Colliders([room]).overlap_sphere((1.8, -1.7, 0.0), 0.5)
        self.assertEqual(len(contacts), 2)
        np.testing.assert_allclose(contacts[0].normal, [-1.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(contacts[1].normal, [0.0, 1.0, 0.0], atol=1e-12)
        self.assertAlmostEqual(contacts[0].depth, 0.3)
        self.assertAlmostEqual(contacts[1].depth, 0.2)

    def test_capsule(self):
        solid = Colliders([Object3D(floor())])
        # Standing through the floor: out upwards, by the radius past the end below it.
        (contact,) = solid.overlap_capsule((0.0, -0.2, 0.0), (0.0, 1.5, 0.0), 0.3)
        np.testing.assert_allclose(contact.normal, [0.0, 1.0, 0.0])
        self.assertAlmostEqual(contact.depth, 0.5)
        # Lying just above it.
        (contact,) = solid.overlap_capsule((-1.0, 0.25, 0.0), (1.0, 0.25, 0.0), 0.3)
        self.assertAlmostEqual(contact.depth, 0.05)
        self.assertEqual(solid.overlap_capsule((-1.0, 0.35, 0.0), (1.0, 0.35, 0.0), 0.3), [])
        # Its side against a pole's edge, nowhere near the pole's corners.
        pole = Colliders([Object3D(make_box(), scale=(0.2, 4.0, 0.2))])
        (contact,) = pole.overlap_capsule((0.3, -1.0, 0.3), (0.3, 1.0, 0.3), 0.5)
        np.testing.assert_allclose(contact.normal, [np.sqrt(0.5), 0.0, np.sqrt(0.5)], atol=1e-9)
        self.assertAlmostEqual(contact.depth, 0.5 - np.sqrt(2) * 0.2)

    def test_box(self):
        solid = Colliders([Object3D(floor())])
        (contact,) = solid.overlap_box((0.0, 0.4, 0.0), 1.0)
        np.testing.assert_allclose(contact.normal, [0.0, 1.0, 0.0])
        self.assertAlmostEqual(contact.depth, 0.1)
        self.assertEqual(solid.overlap_box((0.0, 0.6, 0.0), 1.0), [])
        # Turned 45 degrees about z, its edge reaches down sqrt(2)/2.
        (contact,) = solid.overlap_box((0.0, 0.6, 0.0), (1.0, 1.0, 3.0), quat_axis_angle((0, 0, 1), np.pi / 4))
        self.assertAlmostEqual(contact.depth, np.sqrt(0.5) - 0.6)

    def test_push_out(self):
        room = Object3D(Mesh(make_box(4.0).vertices, make_box(4.0).faces[:, ::-1]))
        solid = Colliders([room, Object3D(make_box(), np.array([0.0, -1.5, 0.0]))])
        for centre in ((1.8, -1.7, 0.0), (1.9, -1.9, 1.9), (0.6, -1.2, 0.1), (0.0, 0.0, 0.0)):
            moved = solid.push_out(centre, 0.5)
            self.assertEqual(solid.overlap_sphere(np.add(centre, moved), 0.5 - 1e-6), [], centre)
        np.testing.assert_array_equal(solid.push_out((0.0, 0.0, 0.0), 0.5), 0.0)  # clear already
        moved = solid.push_out((0.0, -2.2, 1.0), 0.3, end=(0.0, 0.0, 1.0))  # a capsule through the floor
        np.testing.assert_allclose(moved, [0.0, 0.5, 0.0], atol=1e-6)


class StabilityTests(unittest.TestCase):
    def test_bad_numbers(self):
        # Poses with NaN or infinities, a scale of zero, NaN corners: never hit, and no exception or crash.
        box = make_box()
        holed = make_box()
        holed.vertices = holed.vertices.copy()
        holed.vertices[0] = np.nan
        objects = [Object3D(box, np.array([np.nan, 0.0, 0.0])), Object3D(box, scale=0.0),
                   Object3D(box, scale=(1.0, 0.0, 1.0)), Object3D(box, rotation=np.array([np.inf, 0, 0, 0])),
                   Object3D(holed, np.array([0.0, 0.0, -3.0])), Object3D(box, np.array([1e300, 0.0, 0.0]))]
        solid = Colliders(objects)
        hit = solid.raycast((0.0, 0.0, 0.0), (0.0, 0.0, -1.0))
        self.assertIs(hit.object, objects[4])  # the faces without a NaN corner
        for origin, direction in (((np.nan, 0, 0), (0, 0, -1)), ((0, 0, 0), (0, 0, 0)), ((0, 0, 0), (np.inf, 0, 0)),
                                  ((1e300, 0, 0), (0, 0, -1)), ((0, 0, 0), (1e-300, 0, 0))):
            solid.raycast(origin, direction)
            solid.raycast(origin, direction, all=True)
            solid.raycast_many([origin], [direction])
        self.assertIsNone(solid.raycast((0, 0, 0), (0, 0, -1), max_distance=np.nan))
        for centre, radius in (((np.nan, 0, 0), 1.0), ((0, 0, -3), np.nan), ((0, 0, -3), np.inf), ((1e300, 0, 0), 1.0)):
            solid.overlap_sphere(centre, radius)
            solid.overlap_box(centre, radius)
            solid.push_out(centre, radius)
        solid.overlap_box((0, 0, -3), 1.0, rotation=(0.0, 0.0, 0.0, 0.0))
        # Huge and tiny meshes, and one with every triangle in one place.
        for vertices in (np.random.default_rng(1).normal(size=(30, 3)) * 1e300,
                         np.random.default_rng(1).normal(size=(30, 3)) * 1e-300, np.zeros((30, 3))):
            mesh = Mesh(vertices, np.random.default_rng(2).integers(0, 30, (40, 3)))
            other = Colliders([Object3D(mesh)])
            other.raycast((0.0, 0.0, 5.0), (0.0, 0.0, -1.0))
            other.overlap_sphere((0.0, 0.0, 0.0), 1.0)
            other.raycast_many(np.zeros((3, 3)), np.eye(3))

    def test_empty_sets_and_bad_meshes(self):
        for solid in (Colliders(), Colliders([Object3D(Mesh(np.zeros((0, 3)), np.zeros((0, 3), int)))])):
            self.assertIsNone(solid.raycast((0, 0, 0), (0, 0, -1)))
            self.assertEqual(solid.raycast((0, 0, 0), (0, 0, -1), all=True), [])
            self.assertEqual(solid.overlap_sphere((0, 0, 0), 1.0), [])
            self.assertIsNone(solid.raycast_many(np.zeros((2, 3)), (0, 0, -1))[0][0])
        with self.assertRaises(ValueError):
            Colliders([Object3D(Mesh(np.zeros((3, 3)), np.array([[0, 1, 3]])))])  # a face beyond the vertices
        with self.assertRaises(ValueError):
            Colliders([Object3D(make_box())]).raycast((0, 0), (0, 0, -1))


if __name__ == "__main__":
    unittest.main()
