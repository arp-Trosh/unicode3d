# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
"""Skinned figures built in code, for the tests: a tube bent by a chain of joints."""
import numpy as np

from unicode3d.mesh import Mesh
from unicode3d.scene import Node, Object3D, Skin
from unicode3d.transforms import quat_axis_angle


def tube_mesh(bones=3, length=1.0, rings=6, segments=10, radius=0.2):
    """A tube up y, bones * length tall, its vertices weighted between the joints of a chain (see chain()): each
    ring to the joint it is in, and towards the next one along it, so that the tube bends smoothly at each joint.
    rings: rings of vertices to a bone (2 * bones * rings * segments faces)."""
    heights = np.linspace(0.0, bones * length, rings * bones + 1)
    turn = np.linspace(0.0, 2 * np.pi, segments, endpoint=False)
    vertices = np.array([(radius * np.cos(a), y, radius * np.sin(a)) for y in heights for a in turn])
    faces = []
    for r in range(len(heights) - 1):
        for k in range(segments):
            a, b = r * segments + k, r * segments + (k + 1) % segments
            faces += [(a, a + segments, b), (b, a + segments, b + segments)]
    along = np.clip(vertices[:, 1] / length, 0.0, bones - 1e-9)
    bone = np.floor(along).astype(np.int64)
    part = along - bone
    joints = np.stack([bone, np.minimum(bone + 1, bones - 1), np.zeros_like(bone), np.zeros_like(bone)], axis=1)
    weights = np.stack([1.0 - 0.5 * part, 0.5 * part, np.zeros_like(part), np.zeros_like(part)], axis=1)
    return Mesh(vertices, np.array(faces), joints=joints, weights=weights)


def chain(bones=3, length=1.0, parent=None, position=(0.0, 0.0, 0.0)):
    """Joints one above another, length apart, each the child of the one before (the first of `parent`)."""
    joints = []
    for i in range(bones):
        joints.append(Node(position=np.array(position, float) if i == 0 else np.array([0.0, length, 0.0]),
                           parent=joints[-1] if joints else parent))
    return joints


def figure(mesh=None, bones=3, position=(0.0, 0.0, 0.0), bend=0.0, **options):
    """(an Object3D showing a tube, skinned to a chain of its own standing at `position`, its joints) with each joint
    after the first turned `bend` about z."""
    mesh = tube_mesh(bones) if mesh is None else mesh
    joints = chain(bones)
    obj = Object3D(mesh, skin=Skin(joints), **options)  # (bound as the mesh stands, then moved)
    joints[0].position = np.array(position, float)
    for joint in joints[1:]:
        joint.rotation = quat_axis_angle((0.0, 0.0, 1.0), bend)
    return obj, joints


def baked(obj):
    """An Object3D showing obj's mesh as its skin poses it now, as a plain mesh in the world (to compare drawing
    with)."""
    positions, normals = obj.skin.pose(obj.mesh)
    mesh = Mesh(positions, obj.mesh.faces, normals=normals)
    return Object3D(mesh, color=obj.color, double_sided=obj.double_sided, simplify=False)  # (as skinned ones are)
