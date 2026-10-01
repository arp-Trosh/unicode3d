# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
"""What a scene is made of: the camera, and objects placed in it (Object3D), grouped with Nodes.

Lights are in lights.py and the Renderer, which draws all this, in renderer.py; both are importable from here too.
"""
from dataclasses import dataclass, field

import numpy as np

from .lights import Light, PointLight, _vec3
from .mesh import Mesh
from .renderer import Pick, Renderer
from .transforms import look_at, quat_identity, quat_mul, quat_to_matrix, scale3

__all__ = ["Camera", "Light", "Model", "Node", "Object3D", "Pick", "PointLight", "Renderer"]


@dataclass
class Camera:
    position: np.ndarray = _vec3(0.0, 0.0, 5.0)
    target: np.ndarray = _vec3(0.0, 0.0, 0.0)
    up: np.ndarray = _vec3(0.0, 1.0, 0.0)
    fov: float = 50.0  # vertical, degrees
    near: float = 0.1
    far: float = 100.0

    def view_matrix(self):
        return look_at(self.position, self.target, self.up)


class _Placed:
    """Placement in a scene graph, shared by Node and Object3D: position, rotation and scale are
    relative to `parent` (a Node or Object3D), or to the world if there is none.

    scale is one number, or three (x, y, z) that stretch along the object's own axes: (1, 3, 1) makes a
    cube a tall box, before rotation turns it. A parent's scale stretches its children along the parent's
    axes, whichever way they are turned.
    """

    def world_matrix(self):
        """(linear (3, 3), position (3,), visible) in the world, through all parents: a point p of the mesh is
        at linear @ p + position. linear is the rotations and scales along the way, in one matrix; an object is
        visible only if it and all its parents are."""
        linear = quat_to_matrix(self.rotation) * scale3(self.scale)
        position, visible = np.asarray(self.position, dtype=float), bool(self.visible)
        node, seen = self.parent, 0
        while node is not None:
            turn = quat_to_matrix(node.rotation) * scale3(node.scale)
            position = np.asarray(node.position, dtype=float) + turn @ position
            linear = turn @ linear
            visible = visible and bool(node.visible)
            node, seen = node.parent, seen + 1
            if seen > 1000:
                raise ValueError("scene graph has a cycle: an object is its own ancestor")
        return linear, position, visible

    def world_transform(self):
        """(position (3,), rotation quaternion (4,), scale, visible) in the world, through all parents.

        A parent's scale scales its children's offsets and sizes; an object is visible only if it and all its
        parents are. scale is a number if every scale along the way is one; otherwise it is three, along the
        object's own axes, which is exact unless a parent stretching unevenly is turned against its child (a
        box leaning in a stretched group is sheared, which a rotation and a scale can't say): world_matrix()
        is exact whatever the scales.
        """
        _, position, visible = self.world_matrix()
        rotation, scale = np.asarray(self.rotation, dtype=float), self.scale
        node = self.parent
        while node is not None:
            rotation = quat_mul(node.rotation, rotation)
            if np.ndim(scale) == 0 and np.ndim(node.scale) == 0:
                scale = float(scale) * float(node.scale)
            else:
                scale = scale3(scale) * scale3(node.scale)
            node = node.parent
        return position, rotation, (float(scale) if np.ndim(scale) == 0 else scale), visible

    def to_world(self, point):
        """A point given in this node's own space, in world coordinates."""
        linear, position, _ = self.world_matrix()
        return position + linear @ np.asarray(point, dtype=float)


@dataclass
class Node(_Placed):
    """A transform with no mesh, for grouping: objects whose parent is a Node move, turn, scale and
    hide with it. Nodes can have Nodes as parents, and are not passed to render()."""
    position: np.ndarray = _vec3(0.0, 0.0, 0.0)
    rotation: np.ndarray = field(default_factory=quat_identity)
    scale: object = 1.0     # one number, or three: along x, y and z (see _Placed)
    visible: bool = True
    parent: object = None


@dataclass
class Object3D(_Placed):
    """A mesh placed in the scene. position, rotation (a quaternion, w first) and scale are relative to
    `parent` if it has one (a Node or another Object3D), otherwise to the world."""
    mesh: Mesh
    position: np.ndarray = _vec3(0.0, 0.0, 0.0)
    rotation: np.ndarray = field(default_factory=quat_identity)
    scale: object = 1.0     # one number, or three: along x, y and z (see _Placed)
    color: object = 0       # a named Color, or (r, g, b) sRGB as 0..255 ints or 0..1 floats
    visible: bool = True
    double_sided: bool = False  # draw back faces too (for meshes with inconsistent winding)
    parent: object = None
    emissive: float = 0.0   # light level the surface gives off itself, added to the lights': 1.0 shows its
                            # colour at full brightness whatever the lighting (a lamp, a screen, a glowing marker)
    cast_shadows: bool = True  # whether it blocks lights that have shadows (False for a lamp's own bulb, say)
    opacity: float = 1.0    # 1 is solid, less lets what is behind show through (0 is invisible); multiplies the
                            # alpha of the mesh's colours. See-through objects show their far side too.
    reflectivity: float = 0.0  # how much it reflects, 0..1 (1: a perfect mirror): a flat mesh reflects the
                               # scene, a curved one the background (see Renderer)
    specular: float = 1.0   # how strongly highlights show on it, times each light's `specular`: 0 for matte things
                            # (cloth, rubber, stone), more than 1 for glossier ones than the lights are set for
    shininess: float = None  # how tight its highlights are (the Blinn-Phong exponent): about 5 is broad and soft,
                             # 100 a pin-point, as on chrome; None takes each light's `shininess`


@dataclass
class Model:
    """A model loaded from a file (models.load_model): Object3Ds for its parts, all under `root`, so that placing,
    turning, scaling or hiding root does that to the whole model. render() takes the parts, and a Model unpacks
    into them: renderer.render([*model, floor], camera, lights).

    names: the parts by the names the file gives them (a list for each name: its groups', nodes', meshes' and
    materials'); materials: the materials the file defines, by name (models.Material); warnings: what in the file
    could not be loaded (a texture that isn't there, say), which was left out rather than stopping the loading.

    From glTF files also nodes: the file's nodes by name, each a Node, or the part itself where a node has one
    part (move, turn or hide one and what hangs from it goes too: a door, a wheel, an arm); and animations: its
    animations by name, each an animation.Clip moving those nodes (call its update(dt) each frame).
    """
    root: Node
    objects: list
    names: dict = field(default_factory=dict)
    materials: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    nodes: dict = field(default_factory=dict)
    animations: dict = field(default_factory=dict)

    def __iter__(self):
        return iter(self.objects)

    def __len__(self):
        return len(self.objects)

    def bounds(self):
        """The corners (low (3,), high (3,)) of the box around the model's parts, in root's own space (as if root
        were at the origin, unturned and unscaled)."""
        lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
        for obj in self.objects:
            linear, position = np.eye(3), np.zeros(3)
            node = obj
            while node is not None and node is not self.root:
                turn = quat_to_matrix(node.rotation) * scale3(node.scale)
                position = np.asarray(node.position, dtype=float) + turn @ position
                linear = turn @ linear
                node = node.parent
            v = np.asarray(obj.mesh.vertices, dtype=float).reshape(-1, 3)
            v = v[np.isfinite(v).all(axis=1)]
            if len(v):
                v = v @ linear.T + position
                lo, hi = np.minimum(lo, v.min(axis=0)), np.maximum(hi, v.max(axis=0))
        if not (lo <= hi).all():
            return np.zeros(3), np.zeros(3)
        return lo, hi

    def fit(self, size=2.0):
        """Scale root so the model's largest extent is `size`, and place it so that its centre is at root's
        parent's origin (the world's, if it has none). Returns self."""
        lo, hi = self.bounds()
        extent = float((hi - lo).max()) or 1.0
        self.root.scale = size / extent
        linear = quat_to_matrix(self.root.rotation) * scale3(self.root.scale)
        self.root.position = -(linear @ ((lo + hi) / 2.0))
        return self
