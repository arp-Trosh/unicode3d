# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
"""What a scene is made of: the camera, and objects placed in it (Object3D), grouped with Nodes.

Lights are in lights.py and the Renderer, which draws all this, in renderer.py; both are importable from here too.
"""
import copy as _copy
from dataclasses import dataclass, field

import numpy as np

from .animation import Animation, Clip
from .lights import Light, PointLight, _vec3
from .color import linear_to_srgb, srgb_to_linear, to_linear_rgb
from .mesh import Mesh, _srgb01
from .shapes import merge_meshes
from .texture import alpha_kind
from .renderer import Anchor, Pick, Renderer
from .transforms import look_at, quat_identity, quat_mul, place_boxes, quat_to_matrix, scale3, scene_poses, world_matrix

__all__ = ["Anchor", "Camera", "Light", "Model", "Node", "Object3D", "Pick", "PointLight", "Renderer", "union_bounds"]


ORTHO_BACK = 1e5  # how far back, in view sizes, the eye an orthographic view is drawn from sits (see Camera)


@dataclass
class Camera:
    """Where the scene is seen from: a camera at `position` looking at `target`, with `up` towards the top of the
    picture.

    projection "perspective" (the default) shows things smaller further off, in a view `fov` degrees tall;
    "ortho" (orthographic) shows them the same size however far off, in a view `size` units of the world tall, as
    in isometric and top-down games, board games and technical drawings. Either shows what is between `near` and
    `far` in front of the camera.
    """
    position: np.ndarray = _vec3(0.0, 0.0, 5.0)
    target: np.ndarray = _vec3(0.0, 0.0, 0.0)
    up: np.ndarray = _vec3(0.0, 1.0, 0.0)
    fov: float = 50.0  # vertical, degrees (perspective)
    near: float = 0.1
    far: float = 100.0
    projection: str = "perspective"  # or "ortho"
    size: float = 10.0  # how tall the view is in the world (ortho)

    def __post_init__(self):
        if self.projection not in ("perspective", "ortho"):
            raise ValueError(f'projection must be "perspective" or "ortho", not {self.projection!r}')

    def view_matrix(self):
        return look_at(self.position, self.target, self.up)

    def drawn_as(self):
        """(the perspective camera the renderer draws this one as, how far behind this one it is). An orthographic
        view is a perspective one seen from far back (ORTHO_BACK view sizes, and its depth as far again) through a
        narrow angle: what is size tall at `position` fills the picture, near and far are moved back with it, and
        the sizes of things differ with distance by a hundred-thousandth at most, far under a pixel. So every part
        of drawing works for both; only distances from the eye (fog, outlines, pick(), ray()) are measured from
        `position` instead. A perspective camera is drawn as it is, from 0 behind."""
        if self.projection != "ortho":
            return self, 0.0
        with np.errstate(all="ignore"):
            size, near, far = (float(v) for v in (self.size, self.near, self.far))
            back = ORTHO_BACK * (abs(size) + abs(far) + abs(near) + 1.0)
            position = np.asarray(self.position, dtype=float)
            forward = np.asarray(self.target, dtype=float) - position
            forward = forward / np.sqrt(forward @ forward)
            fov = float(np.degrees(2.0 * np.arctan(size / 2.0 / back)))
        return Camera(position - forward * back, self.target, self.up, fov, near + back, far + back), back


class _Placed:
    """Placement in a scene graph, shared by Node and Object3D: position, rotation and scale are
    relative to `parent` (a Node or Object3D), or to the world if there is none.

    Each is a thing of its own, equal only to itself (eq=False), like Model and Mesh: lists of them work with
    `in`, index() and remove() (comparing their numpy arrays would raise), and they can be dict keys and set
    members.

    scale is one number, or three (x, y, z) that stretch along the object's own axes: (1, 3, 1) makes a
    cube a tall box, before rotation turns it. A parent's scale stretches its children along the parent's
    axes, whichever way they are turned.
    """

    def world_matrix(self):
        """(linear (3, 3), position (3,), visible) in the world, through all parents: a point p of the mesh is
        at linear @ p + position. linear is the rotations and scales along the way, in one matrix; an object is
        visible only if it and all its parents are."""
        return world_matrix(self)

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


@dataclass(eq=False)
class Node(_Placed):
    """A transform with no mesh, for grouping: objects whose parent is a Node move, turn, scale and
    hide with it. Nodes can have Nodes as parents, and are not passed to render()."""
    position: np.ndarray = _vec3(0.0, 0.0, 0.0)
    rotation: np.ndarray = field(default_factory=quat_identity)
    scale: object = 1.0     # one number, or three: along x, y and z (see _Placed)
    visible: bool = True
    parent: object = None


@dataclass(eq=False)
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

    def world_bounds(self):
        """The corners (low (3,), high (3,)) of the box around the object's mesh as it stands in the world, through
        its parents: to put a label just above it, or frame a camera on it. None if there is nothing to box (no
        vertices, or a pose that isn't finite). Vertices at NaN or infinity are left out."""
        return union_bounds([self])


def _placed_mesh(part, linear, position):
    """A copy of the part's mesh moved by linear and position (p -> linear @ p + position), its normals turned to
    match and its faces wound the other way where linear mirrors (as the renderer draws such a part), with the
    part's colour multiplied into the mesh's colours (in linear light, as drawing does)."""
    mesh = part.mesh
    faces = np.asarray(mesh.faces, np.int64).reshape(-1, 3)
    c0, c1, c2 = linear[:, 0], linear[:, 1], linear[:, 2]
    cofactor = np.stack([np.cross(c1, c2), np.cross(c2, c0), np.cross(c0, c1)], axis=1)  # det * inverse transpose
    flip = float(np.linalg.det(linear)) < 0.0
    placed = Mesh(np.asarray(mesh.vertices, dtype=float).reshape(-1, 3) @ linear.T + position,
                  faces[:, [0, 2, 1]] if flip else faces, textures=list(mesh.textures),
                  normals=mesh.vertex_normals() @ cofactor.T * (-1.0 if flip else 1.0))
    if mesh.materials is not None and mesh.textures:
        uvs = np.asarray(mesh.uvs, dtype=float).reshape(-1, 3, 2)
        placed.uvs, placed.materials = (uvs[:, [0, 2, 1]] if flip else uvs), mesh.materials
    tint = to_linear_rgb(part.color)

    def tinted(colors):  # sRGB 0..1, opacity kept
        c = _srgb01(colors).reshape(len(colors), -1).copy()
        c[:, :3] = linear_to_srgb(np.clip(srgb_to_linear(np.clip(c[:, :3], 0.0, 1.0)) * tint, 0.0, 1.0))
        return c

    if mesh.vertex_colors is not None:
        placed.vertex_colors = tinted(mesh.vertex_colors)
    elif mesh.face_colors is not None:
        placed.face_colors = tinted(mesh.face_colors)
    else:
        placed.face_colors = np.broadcast_to(linear_to_srgb(np.clip(tint, 0.0, 1.0)), (len(faces), 3)).copy()
    return placed


def _finite_vertices(mesh):
    """The mesh's vertices that are finite, as floats (V, 3), cached like Mesh.bounds() (until vertices is
    replaced)."""
    if mesh is None:
        return np.zeros((0, 3))
    cached = mesh.__dict__.get("_finite_vertices")
    if cached is not None and cached[0] is mesh.vertices:
        return cached[1]
    v = np.asarray(mesh.vertices, dtype=float).reshape(-1, 3)
    v = v[np.isfinite(v).all(axis=1)]
    mesh._finite_vertices = (mesh.vertices, v)
    return v


@dataclass(eq=False)
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

    def copy(self):
        """Another of the same model, to place, pose, hide and animate on its own: ten goblins from one file loaded
        once. Its root, parts and nodes are new (posed as these are now), and so are its animations, with clocks of
        their own; meshes, textures, materials and animation keyframes are shared, so a copy costs little memory
        and no loading. Its root hangs from the same parent as this one's (move it, or set root.parent)."""
        made = {}  # id(node): its copy, for nodes under root (root's own parent and above are shared)

        def inside(node):
            while node is not None:
                if node is self.root:
                    return True
                node = node.parent
            return False

        def clone(node):
            if node is None or (id(node) not in made and not inside(node)):
                return node
            if id(node) not in made:
                twin = made[id(node)] = _copy.copy(node)
                twin.position = np.array(node.position, dtype=float)
                twin.rotation = np.array(node.rotation, dtype=float)
                if isinstance(node.scale, np.ndarray):
                    twin.scale = node.scale.copy()
                if node is not self.root:
                    twin.parent = clone(node.parent)
            return made[id(node)]

        animations = {}
        for name, clip in self.animations.items():
            moves = []
            for a in clip.animations:
                move = Animation(clone(a.target), speed=a.speed, **a.tracks)
                move.time = a.time
                moves.append(move)
            animations[name] = Clip(moves, name=clip.name, loop=clip.loop, speed=clip.speed)
            animations[name].time = clip.time
        return Model(clone(self.root), [clone(o) for o in self.objects],
                     {name: [clone(o) for o in parts] for name, parts in self.names.items()}, dict(self.materials),
                     list(self.warnings), {name: clone(n) for name, n in self.nodes.items()}, animations)

    def bake(self):
        """The model as it stands now, in as few parts as draw it the same: for things that stand still (scenery, a
        building, a tree), which draw quicker as a few big meshes than as dozens of small ones. Returns a new Model
        whose root is posed, parented and shown as this one's is, with its parts in root's space: those that look
        alike (the same specular, shininess, emissive, double_sided and cast_shadows) merged into one mesh
        (merge_meshes: textures and normals kept, each part's colour multiplied into its mesh's colours), and
        those that can't merge (see-through, with holes in their textures, or reflective) each moved into a mesh
        of its own. Hidden parts are left out; it has no nodes or animations (its parts no longer move apart).
        This model is left as it is; copy() the result for more of it."""
        parts = [o for o in self.objects if o.mesh is not None and len(o.mesh.faces)]
        rows, linear, position, visible = scene_poses(parts, top=self.root)
        groups = {}  # look: (part, linear, position)
        for part in parts:
            r = rows[id(part)]
            if not (visible[r] and np.isfinite(linear[r]).all() and np.isfinite(position[r]).all()):
                continue
            mesh = part.mesh
            corner = mesh.corner_colors()
            alone = (not part.opacity >= 1.0 or part.reflectivity != 0.0
                     or (corner is not None and bool((corner[:, :, 3] < 1.0).any()))
                     or (mesh.materials is not None and any(alpha_kind(mesh.mipmaps(m))
                                                            for m in range(len(mesh.textures)))))
            look = ((id(part),) if alone else ()) + (
                float(part.specular), part.shininess, float(part.emissive), bool(part.double_sided),
                bool(part.cast_shadows))
            groups.setdefault(look, []).append((part, linear[r], position[r]))
        root = Node(np.array(self.root.position, dtype=float), np.array(self.root.rotation, dtype=float),
                    _copy.copy(self.root.scale), self.root.visible, self.root.parent)
        objects = []
        for placed in groups.values():
            first = placed[0][0]
            objects.append(Object3D(merge_meshes([_placed_mesh(*p) for p in placed]), color=(1.0, 1.0, 1.0),
                                    double_sided=first.double_sided, parent=root, emissive=first.emissive,
                                    cast_shadows=first.cast_shadows, opacity=first.opacity,
                                    reflectivity=first.reflectivity, specular=first.specular,
                                    shininess=first.shininess))
        return Model(root, objects, materials=dict(self.materials), warnings=list(self.warnings))

    def world_bounds(self):
        """The corners (low (3,), high (3,)) of the box around the model's parts as they stand in the world (see
        Object3D.world_bounds); None if there is nothing to box."""
        return union_bounds(self.objects)

    def bounds(self):
        """The corners (low (3,), high (3,)) of the box around the model's parts, in root's own space (as if root
        were at the origin, unturned and unscaled)."""
        with np.errstate(all="ignore"):  # (numbers that aren't finite are left out, not warned about)
            return self._bounds()

    def _bounds(self):
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
        with np.errstate(all="ignore"):
            extent = float((hi - lo).max()) or 1.0
            self.root.scale = size / extent
            linear = quat_to_matrix(self.root.rotation) * scale3(self.root.scale)
            self.root.position = -(linear @ ((lo + hi) / 2.0))
        return self


def union_bounds(objects):
    """The corners (low (3,), high (3,)) of the box around several objects (Object3Ds or Models) as they stand in
    the world; None if there is nothing to box."""
    parts = [part for obj in objects for part in ([obj] if hasattr(obj, "mesh") else obj)]
    parts = [part for part in parts if len(_finite_vertices(part.mesh))]
    if not parts:
        return None
    rows, linear, position, _ = scene_poses(parts)  # (each shared parent once)
    runs, meshes = {}, []  # id(mesh): (first vertex, count), in vertices packed end to end
    first = 0
    for part in parts:
        if id(part.mesh) not in runs:
            v = _finite_vertices(part.mesh)
            runs[id(part.mesh)] = (first, len(v))
            meshes.append(v)
            first += len(v)
    run = np.array([runs[id(part.mesh)] for part in parts], np.int64).reshape(-1, 2)
    lo, hi = np.empty((len(parts), 3)), np.empty((len(parts), 3))
    place_boxes(np.ascontiguousarray(np.concatenate(meshes), np.float64), run[:, 0].copy(), run[:, 1].copy(),
                np.array([rows[id(part)] for part in parts], np.int64), linear, position, lo, hi)
    boxed = np.isfinite(lo).all(axis=1) & np.isfinite(hi).all(axis=1)  # (a pose that isn't finite: left out)
    if not boxed.any():
        return None
    return lo[boxed].min(axis=0), hi[boxed].max(axis=0)
