# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
"""Skinned figures built in code (the marionettes in skinned.py, the bench and the gallery): tubes through chains of
joints, and a simple figure of them (hips, spine, head, arms and legs), one mesh skinned to a skeleton of its own."""
import numpy as np

from ..mesh import Mesh
from ..scene import Node, Object3D, Skin
from ..transforms import quat_axis_angle, quat_mul


def tubes(joints, limbs, segments=8):
    """A mesh of tubes, bound to `joints` as they stand now (Skin(joints) draws it unchanged until they move).
    limbs: lists of rings (joint, before, along, radius), each a ring of `radius` around joint's y axis (indices
    into joints) at `along` on it, weighted half to joint and half to `before` (all to joint if they are the same);
    each ring is joined to the next."""
    world = Skin(joints).world_matrices()
    turn = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    circle = np.stack([np.cos(turn), np.zeros(segments), np.sin(turn), np.ones(segments)], axis=1)
    vertices, faces, joint_rows, weight_rows = [], [], [], []
    for rings in limbs:
        first = len(vertices)
        for joint, before, along, radius in rings:
            local = circle * (radius, 1.0, radius, 1.0) + (0.0, along, 0.0, 0.0)
            vertices += list((local @ world[joint].T)[:, :3])
            joint_rows += [(joint, before, 0, 0)] * segments
            weight_rows += [(1.0, 0.0, 0.0, 0.0) if before == joint else (0.5, 0.5, 0.0, 0.0)] * segments
        for r in range(len(rings) - 1):
            for k in range(segments):
                a, b = first + r * segments + k, first + r * segments + (k + 1) % segments
                faces += [(a, a + segments, b), (b, a + segments, b + segments)]
    return Mesh(np.array(vertices), np.array(faces), joints=np.array(joint_rows), weights=np.array(weight_rows))


def chain_rings(chain, radii, end):
    """The rings of a tube through the joints `chain` (each a child of the one before): one around each joint,
    half on it and half on the one before (the first on its own alone), and one `end` along the last, on it alone.
    radii: len(chain) + 1."""
    rings = [(chain[k], chain[k - 1] if k else chain[k], 0.0, radii[k]) for k in range(len(chain))]
    return rings + [(chain[-1], chain[-1], end, radii[-1])]


def _bone(parent, offset, rotation=None):
    return Node(np.array(offset, float), np.array([1.0, 0.0, 0.0, 0.0]) if rotation is None else rotation,
                parent=parent)


def skeleton():
    """A figure's joints, standing at the origin, about 1.7 tall: a dict of Nodes by name, "hips" the root. Each
    joint's y axis runs along its bone (the arms' down and a little out, the legs' down), and its z axis forward, so
    that bend(joint, (1, 0, 0), a) swings a limb forward by a and bend(joint, (0, 0, 1), a) to its side."""
    j = {"hips": _bone(None, (0.0, 0.95, 0.0))}
    j["spine"] = _bone(j["hips"], (0.0, 0.15, 0.0))
    j["chest"] = _bone(j["spine"], (0.0, 0.22, 0.0))
    j["neck"] = _bone(j["chest"], (0.0, 0.2, 0.0))
    j["head"] = _bone(j["neck"], (0.0, 0.08, 0.0))
    for side, name in ((1.0, "right"), (-1.0, "left")):
        j[f"{name} shoulder"] = _bone(j["chest"], (0.15 * side, 0.15, 0.0), quat_axis_angle((0, 0, 1), -side * 2.7))
        j[f"{name} elbow"] = _bone(j[f"{name} shoulder"], (0.0, 0.28, 0.0))
        j[f"{name} wrist"] = _bone(j[f"{name} elbow"], (0.0, 0.25, 0.0))
        j[f"{name} hip"] = _bone(j["hips"], (0.1 * side, 0.0, 0.0), quat_axis_angle((0, 0, 1), np.pi))
        j[f"{name} knee"] = _bone(j[f"{name} hip"], (0.0, 0.43, 0.0))
        j[f"{name} ankle"] = _bone(j[f"{name} knee"], (0.0, 0.42, 0.0))
    return j


def figure_rig(segments=8):
    """(mesh, inverse bind matrices) of the figure skeleton() makes, bound standing as it makes it: the mesh and
    matrices every figure() shares."""
    j = skeleton()
    at = {name: k for k, name in enumerate(j)}
    limbs = [chain_rings([at[n] for n in ("hips", "spine", "chest", "neck")], (0.12, 0.11, 0.15, 0.06, 0.05), 0.09),
             [(at["head"], at["neck"], 0.0, 0.05), (at["head"], at["head"], 0.06, 0.1),
              (at["head"], at["head"], 0.17, 0.105), (at["head"], at["head"], 0.26, 0.07),
              (at["head"], at["head"], 0.29, 0.0)]]
    for name in ("right", "left"):
        limbs.append(chain_rings([at[f"{name} {n}"] for n in ("shoulder", "elbow", "wrist")],
                                 (0.055, 0.045, 0.035, 0.03), 0.1))
        limbs.append(chain_rings([at[f"{name} {n}"] for n in ("hip", "knee", "ankle")],
                                 (0.075, 0.055, 0.045, 0.04), 0.08))
    joints = list(j.values())
    return tubes(joints, limbs, segments), np.linalg.inv(Skin(joints).world_matrices())


def figure(rig, position=(0.0, 0.0, 0.0), turn=0.0, **options):
    """(an Object3D showing the rig (figure_rig()) on a skeleton of its own, the skeleton's joints by name) standing
    at `position`, turned `turn` about y."""
    mesh, inverse_bind = rig
    j = skeleton()
    j["hips"].position = j["hips"].position + np.array(position, float)
    j["hips"].rotation = quat_axis_angle((0, 1, 0), turn)
    return Object3D(mesh, skin=Skin(list(j.values()), inverse_bind), **options), j


def bend(joint, axis, angle):
    """Turn a joint by `angle` about `axis` (in its own space), from the way skeleton() made it."""
    joint.rotation = quat_mul(joint.rotation, quat_axis_angle(axis, angle))
