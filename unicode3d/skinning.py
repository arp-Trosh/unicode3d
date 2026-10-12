# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
"""Skinned meshes: vertices moved by a skeleton's joints (linear blend skinning), worked out once a frame.

A Mesh with `joints` and `weights` (V, 4) names up to four joints for each vertex and how much each moves it. An
Object3D showing it with a `skin` (a Skin: the joints, Nodes, and their inverse bind matrices) is drawn with each
vertex at the weighted sum of where its joints carry it: move, turn or animate the joints (a Clip does) and the
mesh bends with them. The object's own position, rotation and scale don't count (as in glTF: the joints place it);
its visibility does, and its joints' visibility doesn't.

How a frame takes them (Renderer._instances, through SkinPass): the joints' world poses come from scene_poses();
joint_palette() makes each joint's matrix (its world pose times its inverse bind matrix) and a bounding sphere
around the object and around each run of its faces from spheres around the joints (see joint_radii), which cull it
as a mesh's spheres do; skin() then blends the vertices of the objects whose matrices changed into world-space rows,
which raster.transform() and the shadow kernels read in place of the mesh's own vertices (inst_src), placed by the
identity. Objects whose matrices didn't change keep last frame's rows, so idle characters cost little.
"""
import weakref
from dataclasses import dataclass

import numpy as np
from numba import njit, prange

from .raster import VERTEX_CHUNK, _normal_matrix, stretch
from .transforms import scene_poses

__all__ = ["Skin"]

CULL_MARGIN = 1.01  # a skinned object is left out where its sphere, this much larger, is wholly out of view


@dataclass(eq=False)  # (equal only to itself: see scene._Placed)
class Skin:
    """The skeleton a skinned mesh (Mesh.joints, Mesh.weights) is posed by: Object3D(mesh, skin=Skin(...)).

    joints: the joints, Nodes (or Object3Ds) placed anywhere in the scene, usually in a tree of their own; a mesh's
    joint index k is joints[k]. inverse_bind: (J, 4, 4) matrices, one per joint, taking the mesh's own vertices into
    that joint's space as it stood when the mesh was bound to it (as glTF gives them); None takes the joints' poses
    now as that pose, so a mesh built where its skeleton stands now is drawn as it is until the joints move.

    Several objects may share a Skin (crowds of the same character move together then); Model.copy() gives each copy
    a Skin of its own, on its own copies of the joints."""
    joints: list
    inverse_bind: np.ndarray = None

    def __post_init__(self):
        self.joints = list(self.joints)
        if self.inverse_bind is None:
            with np.errstate(all="ignore"):
                world = self.world_matrices()
                try:
                    self.inverse_bind = np.linalg.inv(world)
                except np.linalg.LinAlgError:
                    raise ValueError("Skin: a joint's pose can't be inverted (a scale of 0?); give inverse_bind") from None
        else:
            self.inverse_bind = np.asarray(self.inverse_bind, dtype=np.float64)
        if self.inverse_bind.shape != (len(self.joints), 4, 4):
            raise ValueError(f"Skin.inverse_bind must be ({len(self.joints)}, 4, 4), one for each joint, not "
                             f"{self.inverse_bind.shape}")

    def world_matrices(self):
        """Each joint's pose in the world now, as (J, 4, 4) matrices."""
        rows, linear, place, _ = scene_poses(self.joints)
        r = np.array([rows[id(joint)] for joint in self.joints], np.int64)
        world = np.zeros((len(self.joints), 4, 4))
        world[:, :3, :3], world[:, :3, 3], world[:, 3, 3] = linear[r], place[r], 1.0
        return world

    def palette(self):
        """Each joint's matrix now (J, 4, 4): its pose in the world times its inverse bind matrix, which takes a
        vertex of the mesh to where that joint alone would carry it."""
        with np.errstate(all="ignore"):
            return self.world_matrices() @ self.inverse_bind

    def pose(self, mesh):
        """(positions (V, 3), unit normals (V, 3)) of a skinned mesh's vertices in the world, as this skin poses it
        now: what the renderer draws (worked out here in numpy, rather than in its kernels). A normal blended to
        nothing stays 0."""
        joints, weights = skin_arrays(mesh)
        _check_joints(joints, self)
        with np.errstate(all="ignore"):
            m = self.palette()[:, :3, :]
            normal = _normal_matrices(m[:, :, :3])
            blend = np.einsum("vk,vkij->vij", weights, m[joints])
            vertices = np.asarray(mesh.vertices, dtype=float).reshape(-1, 3)
            positions = np.einsum("vij,vj->vi", blend[:, :, :3], vertices) + blend[:, :, 3]
            normals = np.einsum("vij,vj->vi", np.einsum("vk,vkij->vij", weights, normal[joints]),
                                mesh.vertex_normals())
            length = np.linalg.norm(normals, axis=1, keepdims=True)
            normals = np.where(length > 0.0, normals / np.where(length > 0.0, length, 1.0), normals)
        return positions, normals

    def remapped(self, joint):
        """A Skin on joint(j) for each of its joints j (sharing its inverse bind matrices): for Model.copy()."""
        return Skin([joint(j) for j in self.joints], self.inverse_bind)


def _normal_matrices(linear):
    """What turns normals for each of linear (N, 3, 3), as joint_palette() makes them: the inverse transpose, scaled
    to stretch normals as much as linear stretches points (see there)."""
    c0, c1, c2 = linear[:, :, 0], linear[:, :, 1], linear[:, :, 2]
    cof = np.stack([np.cross(c1, c2), np.cross(c2, c0), np.cross(c0, c1)], axis=2)  # (det * inverse transpose)
    det = np.einsum("ni,ni->n", c0, cof[:, :, 0])
    size = np.abs(det) ** (1.0 / 3.0)
    scale = np.where(size > 0.0, np.sign(det) / np.where(size > 0.0, size, 1.0), 1.0)
    return cof * scale[:, None, None]


def skin_arrays(mesh, fresh=False):
    """A skinned mesh's (joints (V, 4) int64, weights (V, 4) float64), made safe for the kernels: weights that are
    negative or not finite as 0, each vertex's scaled to add up to 1 (a vertex with none follows its first joint
    alone). Cached on the mesh until either array is replaced (or fresh: worked out again, after edits made in
    place). Raises ValueError where they don't fit the mesh."""
    cached = mesh.__dict__.get("_skin_arrays")
    if (not fresh and cached is not None and cached[0] is mesh.joints and cached[1] is mesh.weights
            and cached[2] is mesh.vertices):
        return cached[3]
    mesh.check()
    if mesh.joints is None:
        raise ValueError("the mesh has no joints and weights to be skinned with (Mesh.joints, Mesh.weights)")
    joints = np.array(mesh.joints, dtype=np.int64)  # (copies: the mesh's own are left as they are)
    weights = np.array(mesh.weights, dtype=np.float64)
    weights[~(weights > 0.0) | ~np.isfinite(weights)] = 0.0  # (NaN fails the first test)
    total = weights.sum(axis=1)
    none = ~(total > 0.0) | ~np.isfinite(total)
    weights[~none] /= total[~none, None]
    weights[none] = (1.0, 0.0, 0.0, 0.0)
    joints[weights == 0.0] = 0  # (an unused slot names joint 0, which every skin has)
    mesh._skin_arrays = (mesh.joints, mesh.weights, mesh.vertices, (joints, weights))
    return joints, weights


def _check_joints(joints, skin):
    """Raise ValueError if a mesh's joint indices (as skin_arrays gives them) reach past the skin's joints."""
    most = int(joints.max()) if joints.size else -1
    if most >= len(skin.joints):
        raise ValueError(f"the mesh's vertices name joint {most}, but its skin has {len(skin.joints)} joints")


def joint_radii(mesh, inverse_bind):
    """How far from each joint (J,) the vertices it moves reach, in the joint's own space (|inverse bind @ p|), -1
    for joints moving none: each joint carries its vertices within that times its stretch of where it is. Cached on
    the mesh for each inverse_bind array."""
    cache = mesh.__dict__.setdefault("_joint_radii", {})
    cached = cache.get(id(inverse_bind))
    joints, weights = skin_arrays(mesh)
    if cached is not None and cached[0] is inverse_bind and cached[1] is joints:
        return cached[2]
    vertices = np.asarray(mesh.vertices, dtype=float).reshape(-1, 3)
    radius = np.full(len(inverse_bind), -1.0)
    with np.errstate(all="ignore"):
        for k in range(joints.shape[1]):
            moved = weights[:, k] > 0.0
            j = joints[moved, k]
            ibm = inverse_bind[j]
            local = np.einsum("vij,vj->vi", ibm[:, :3, :3], vertices[moved]) + ibm[:, :3, 3]
            reach = np.sqrt((local ** 2).sum(axis=1))
            reach = np.where(np.isnan(reach), np.inf, reach)  # (a NaN corner: never culled)
            np.maximum.at(radius, j, reach)
    if len(cache) > 8:
        cache.clear()
    cache[id(inverse_bind)] = (inverse_bind, joints, radius)
    return radius


def run_joints(joints, weights, run_vertex):
    """Which joints move the vertices of each run of a mesh's faces (run_vertex (R, 2): the vertices each uses,
    first and end): (first (R + 1,), joints (K,)), each run's joints[first[r]:first[r + 1]]."""
    lists = []
    for a, b in run_vertex:
        used = joints[a:b][weights[a:b] > 0.0]
        lists.append(np.unique(used))
    first = np.zeros(len(lists) + 1, np.int64)
    first[1:] = np.cumsum([len(x) for x in lists])
    return first, (np.concatenate(lists).astype(np.int64) if lists else np.zeros(0, np.int64))


# ----- kernels -------------------------------------------------------------------------------

@njit(cache=True, error_model="numpy", inline="always")
def _enclose(linear, place, joint_row, joint_radius, js, j0, out):
    """A sphere (out (4,): centre, radius) around the spheres of joints j0 + js[k] (each at its place, of its radius
    times its stretch); joints moving no vertex (a radius under 0) are left out. NaN where there are none, which
    culls nothing."""
    lo0 = lo1 = lo2 = np.inf
    hi0 = hi1 = hi2 = -np.inf
    some = False
    for k in range(js.shape[0]):
        j = j0 + js[k]
        if not joint_radius[j] >= 0.0:
            continue
        r = joint_row[j]
        reach = joint_radius[j] * stretch(linear[r])
        c = place[r]
        lo0, lo1, lo2 = min(lo0, c[0] - reach), min(lo1, c[1] - reach), min(lo2, c[2] - reach)
        hi0, hi1, hi2 = max(hi0, c[0] + reach), max(hi1, c[1] + reach), max(hi2, c[2] + reach)
        some = True
    if not some:
        out[0] = out[1] = out[2] = out[3] = np.nan
        return
    cx, cy, cz = (lo0 + hi0) / 2.0, (lo1 + hi1) / 2.0, (lo2 + hi2) / 2.0
    radius = 0.0
    for k in range(js.shape[0]):
        j = j0 + js[k]
        if not joint_radius[j] >= 0.0:
            continue
        r = joint_row[j]
        c = place[r]
        d = np.sqrt((c[0] - cx) ** 2 + (c[1] - cy) ** 2 + (c[2] - cz) ** 2) + joint_radius[j] * stretch(linear[r])
        if d > radius or d != d:  # (a NaN wins)
            radius = d
    out[0], out[1], out[2], out[3] = cx, cy, cz, radius


@njit(cache=True, error_model="numpy", parallel=True)
def joint_palette(linear, place, joint_row, inverse_bind, joint_radius, skin_joint, skin_runs, run_first, run_list,
                  skin_bound, palette, normal_palette, bounds, flip, finite):
    """Each skinned object's joint matrices and bounds, in parallel over the objects (each writes only its own rows).

    Object s has joints skin_joint[s]:skin_joint[s + 1] (rows of palette); joint j is row joint_row[j] of the world
    poses linear (N, 3, 3) and place (N, 3) (transforms.scene_poses), with inverse bind matrix inverse_bind[j]
    (3, 4: the top of the 4 x 4) and reach joint_radius[j] (joint_radii). Writes palette[j] (3, 4): the pose times
    the inverse bind matrix; normal_palette[j] (3, 3): what turns normals with it, its inverse transpose times the
    cube root of its determinant, which is the matrix itself for a turn and an even scale (so that normals blend
    as a GPU's skinning blends them, each joint's in proportion to its scale) and right for uneven ones;
    bounds[skin_bound[s]]: a sphere (centre, radius) around the object, and rows after it, one for each of the
    skin_runs[s, 1] runs of its mesh's faces, around those moved by run_list[run_first[q]:run_first[q + 1]] (joints
    counted from the object's first) for q from skin_runs[s, 0]; flip[s]: whether its matrices mirror, taken
    together (their determinants add up to less than 0), which turns its faces inside out; finite[s]: whether every
    matrix is finite (an object with one that isn't is left out)."""
    for s in prange(skin_joint.shape[0] - 1):
        j0, j1 = skin_joint[s], skin_joint[s + 1]
        ok = True
        dets = 0.0
        for j in range(j0, j1):
            r = joint_row[j]
            lin, t, b = linear[r], place[r], inverse_bind[j]
            for i in range(3):
                for k in range(4):
                    v = lin[i, 0] * b[0, k] + lin[i, 1] * b[1, k] + lin[i, 2] * b[2, k]
                    if k == 3:
                        v += t[i]
                    palette[j, i, k] = v
                    if not np.isfinite(v):
                        ok = False
            n = normal_palette[j]
            _normal_matrix(palette[j, :, :3], n)  # (|det| times the inverse transpose)
            p = palette[j]
            det = (p[0, 0] * (p[1, 1] * p[2, 2] - p[1, 2] * p[2, 1]) - p[0, 1] * (p[1, 0] * p[2, 2] - p[1, 2] * p[2, 0])
                   + p[0, 2] * (p[1, 0] * p[2, 1] - p[1, 1] * p[2, 0]))
            size = abs(det) ** (1.0 / 3.0)
            if size > 0.0 and size < np.inf:
                for i in range(3):
                    for k in range(3):
                        n[i, k] /= size
            dets += det
        flip[s] = dets < 0.0
        finite[s] = ok
        every = np.arange(j1 - j0)
        _enclose(linear, place, joint_row, joint_radius, every, j0, bounds[skin_bound[s]])
        for q in range(skin_runs[s, 1]):
            row = skin_runs[s, 0] + q
            _enclose(linear, place, joint_row, joint_radius, run_list[run_first[row]:run_first[row + 1]], j0,
                     bounds[skin_bound[s] + 1 + q])


@njit(cache=True, error_model="numpy", parallel=True)
def skin(vertices, normals, joints, weights, palette, normal_palette, chunk_vertex, chunk_weight, chunk_joint,
         chunk_row, chunk_count, out):
    """Skinned vertices into out (2, S, 3: positions, normals, in the world), in parallel over chunks of vertices
    (each writes only its own rows). Chunk c blends chunk_count[c] vertices: those from row chunk_vertex[c] of
    vertices and normals (the pack's, in the mesh's own space), with their 4 joints and weights from row
    chunk_weight[c] of joints and weights (joints counted from palette row chunk_joint[c]), into out from row
    chunk_row[c]. Each position is the sum over its joints of weight * palette[j] @ p, each normal the same with
    normal_palette (normalized afterwards, by raster.transform)."""
    for c in prange(chunk_count.shape[0]):
        v0, w0, j0, r0 = chunk_vertex[c], chunk_weight[c], chunk_joint[c], chunk_row[c]
        for v in range(chunk_count[c]):
            p, n = vertices[v0 + v], normals[v0 + v]
            px = py = pz = nx = ny = nz = 0.0
            for k in range(4):
                w = weights[w0 + v, k]
                if w == 0.0:
                    continue
                j = j0 + joints[w0 + v, k]
                m, q = palette[j], normal_palette[j]
                px += w * (m[0, 0] * p[0] + m[0, 1] * p[1] + m[0, 2] * p[2] + m[0, 3])
                py += w * (m[1, 0] * p[0] + m[1, 1] * p[1] + m[1, 2] * p[2] + m[1, 3])
                pz += w * (m[2, 0] * p[0] + m[2, 1] * p[1] + m[2, 2] * p[2] + m[2, 3])
                nx += w * (q[0, 0] * n[0] + q[0, 1] * n[1] + q[0, 2] * n[2])
                ny += w * (q[1, 0] * n[0] + q[1, 1] * n[1] + q[1, 2] * n[2])
                nz += w * (q[2, 0] * n[0] + q[2, 1] * n[1] + q[2, 2] * n[2])
            a, b = out[0, r0 + v], out[1, r0 + v]
            a[0], a[1], a[2] = px, py, pz
            b[0], b[1], b[2] = nx, ny, nz


# ----- the pass a frame takes ----------------------------------------------------------------

class _Plan:
    """What stays the same while the same skinned objects show the same meshes with the same skins: their joints in
    one list (for scene_poses), and the arrays joint_palette() and skin() take, but for the poses."""

    def __init__(self, key, objects, skins, meshes, runs_of, prepared):
        self.key = key
        self.objects = objects
        s_count = len(objects)
        self.joints = [joint for skin in skins for joint in skin.joints]
        sizes = np.array([len(skin.joints) for skin in skins], np.int64)
        self.skin_joint = np.zeros(s_count + 1, np.int64)
        np.cumsum(sizes, out=self.skin_joint[1:])
        self.inverse_bind = np.ascontiguousarray(
            np.concatenate([skin.inverse_bind[:, :3, :] for skin in skins]).reshape(-1, 3, 4), np.float64)
        weight_at, arrays, run_at, run_lists = {}, [], {}, []
        self.skin_weight = np.zeros(s_count, np.int64)
        self.skin_runs = np.zeros((s_count, 2), np.int64)
        self.vertices = np.zeros(s_count, np.int64)
        radii, total, n_runs = [], 0, 0
        for s, (skin_, mesh) in enumerate(zip(skins, meshes)):
            joints, weights = skin_arrays(mesh, fresh=mesh not in prepared)  # (afresh once after invalidate())
            prepared.add(mesh)
            _check_joints(joints, skin_)
            if id(mesh) not in weight_at:
                weight_at[id(mesh)] = total
                arrays.append((joints, weights))
                total += len(joints)
            if id(mesh) not in run_at:
                first, listed = run_joints(joints, weights, runs_of(mesh))
                run_at[id(mesh)] = (n_runs, len(first) - 1)
                run_lists.append((first, listed))
                n_runs += len(first) - 1
            self.skin_weight[s] = weight_at[id(mesh)]
            self.skin_runs[s] = run_at[id(mesh)]
            self.vertices[s] = len(joints)
            radii.append(joint_radii(mesh, skin_.inverse_bind))
        self.joint_radius = np.ascontiguousarray(np.concatenate(radii) if radii else np.zeros(0), np.float64)
        self.weight_joints = np.ascontiguousarray(np.concatenate([a[0] for a in arrays]).reshape(-1, 4), np.int64)
        self.weights = np.ascontiguousarray(np.concatenate([a[1] for a in arrays]).reshape(-1, 4), np.float64)
        # Every mesh's run lists end to end: run q's joints are run_list[run_first[q]:run_first[q + 1]].
        firsts, lists, at = [np.zeros(1, np.int64)], [], 0
        for first, listed in run_lists:
            firsts.append(first[1:] + at)
            lists.append(listed)
            at += len(listed)
        self.run_first = np.concatenate(firsts).astype(np.int64)
        self.run_list = np.concatenate(lists).astype(np.int64) if lists else np.zeros(0, np.int64)
        # Each object's rows in the bounds: its own sphere, then one for each run.
        rows = 1 + self.skin_runs[:, 1]
        self.skin_bound = np.zeros(s_count, np.int64)
        self.skin_bound[1:] = np.cumsum(rows)[:-1]
        self.n_bounds = int(rows.sum())
        self.order = None  # (scene_poses' order of the joints when joint_row was made)
        self.joint_row = None
        self.mesh_keys = [_skin_key(mesh) for mesh in meshes]


class _Slot:
    """An object's skinned rows (see SkinPass): where they start in the store and how many, and what they were last
    skinned from (key: its mesh's arrays, its palette's bits and its flip; None: not yet), with their serial."""
    __slots__ = ("start", "count", "mesh", "key", "serial")

    def __init__(self, start, count, mesh):
        self.start, self.count, self.mesh, self.key, self.serial = start, count, mesh, None, 0


class SkinPass:
    """A Renderer's skinning, from frame to frame: the plan for the skinned objects drawn (_Plan), and each one's
    rows of skinned vertices in a store of them (slots, by object), kept while its palette's bits stay the same.

    pose() works out the joints' matrices and the objects' bounds, and which to leave out (culled or not finite);
    skin() blends the vertices of those whose matrices changed (after the meshes are packed, whose vertices it
    reads) and says where each one's rows are."""

    def __init__(self):
        self._plan = None
        self._memo = {}  # (scene_poses' memo for the joints)
        self._slots = {}  # id(object): _Slot
        self._store = np.zeros((2, 0, 3))
        self._end = 0  # (rows of the store given out)
        self._serial = 0
        self._frame = None  # (palette, normal palette, bounds, flip) of the last pose()
        self._prepared = weakref.WeakSet()  # the meshes whose skin_arrays were worked out afresh for this pass

    def pose(self, nodes, skins, meshes, runs_of, cull=None):
        """For the objects `nodes` (rows of the instances; skins and meshes: each one's Skin, None if it has none,
        and Mesh): (each row's place among the skinned objects (N,), -1 for the rest; which rows to leave out (N,)
        bool, or None for none). runs_of(mesh): the vertices each run of a mesh's faces uses (renderer._runs).
        cull: None, or (planes (6, 4): the camera's view, each kept where a x + b y + c z + d >= 0; positions (L, 3)
        and ranges (L,) of the lights whose shadows an object out of view may cast): skinned objects wholly out of
        view and out of reach of those lights are left out, before they are skinned."""
        rows = [k for k, s in enumerate(skins) if s is not None]
        key = [x for k in rows for x in (nodes[k], skins[k], *_skin_key(meshes[k]), skins[k].inverse_bind,
                                         *skins[k].joints, None)]
        plan = self._plan
        if plan is None or not (len(plan.key) == len(key) and all(map(_is, plan.key, key))):
            plan = self._plan = _Plan(key, [nodes[k] for k in rows], [skins[k] for k in rows],
                                      [meshes[k] for k in rows], runs_of, self._prepared)
            self._allot(plan)
        poses, linear, place, _ = scene_poses(plan.joints, memo=self._memo)
        if plan.joint_row is None or plan.order is not self._memo.get("order"):
            plan.joint_row = np.array([poses[id(joint)] for joint in plan.joints], np.int64)
            plan.order = self._memo.get("order")
        n_skins, n_joints = len(plan.objects), len(plan.joints)
        palette, normal_palette = np.empty((n_joints, 3, 4)), np.empty((n_joints, 3, 3))
        bounds = np.empty((plan.n_bounds, 4))
        flip, finite = np.empty(n_skins, np.bool_), np.empty(n_skins, np.bool_)
        joint_palette(linear, place, plan.joint_row, plan.inverse_bind, plan.joint_radius, plan.skin_joint,
                      plan.skin_runs, plan.run_first, plan.run_list, plan.skin_bound, palette, normal_palette, bounds,
                      flip, finite)
        out = ~finite
        if cull is not None:
            planes, positions, ranges = cull
            spheres = bounds[plan.skin_bound]
            with np.errstate(all="ignore"):
                outside = (spheres[:, :3] @ planes[:, :3].T + planes[:, 3] < -CULL_MARGIN * spheres[:, 3:]).any(axis=1)
                if len(ranges):
                    gap = np.linalg.norm(spheres[None, :, :3] - positions[:, None, :], axis=2) - spheres[None, :, 3]
                    outside &= ~(gap < CULL_MARGIN * ranges[:, None]).any(axis=0)
            out |= outside
        self._frame = (palette, normal_palette, bounds, flip)
        skin_of = np.full(len(nodes), -1, np.int64)
        skin_of[rows] = np.arange(n_skins)
        return skin_of, (None if not out.any() else np.isin(skin_of, np.flatnonzero(out)))

    def _allot(self, plan):
        """Give each of the plan's objects its slot: the one it has if its mesh is the same, else rows after the
        others (and the store afresh, every object to be skinned again, when that runs out of room or is mostly
        let go). Objects no longer skinned let theirs go."""
        live = {id(obj) for obj in plan.objects}
        for gone in [k for k in self._slots if k not in live]:
            del self._slots[gone]
        wanted = []
        for obj, count, mesh in zip(plan.objects, plan.vertices, plan.mesh_keys):
            slot = self._slots.get(id(obj))
            if slot is None or slot.count != count or not all(map(_is, slot.mesh, mesh)):
                self._slots[id(obj)] = slot = _Slot(-1, int(count), mesh)
            wanted.append(slot)
        unique = {id(slot): slot for slot in wanted}.values()
        in_use = sum(slot.count for slot in unique)
        new = [slot for slot in unique if slot.start < 0]
        end = self._end + sum(slot.count for slot in new)
        if end > self._store.shape[1] or end > 2 * in_use + 4096:  # afresh, every slot packed from the start
            self._store = np.zeros((2, 2 * in_use + 1024, 3))
            self._end = 0
            new = list(unique)
            for slot in new:
                slot.key = None
        for slot in new:
            slot.start, self._end = self._end, self._end + slot.count
        self._slots = {id(obj): self._slots[id(obj)] for obj in plan.objects}

    def skin(self, pack, skin_of, pack_mesh):
        """Skin the objects drawn whose matrices changed since they last were (rows of the instances whose places
        among the skinned objects are skin_of (K,), showing meshes pack_mesh (K,) of `pack`, renderer._Packer's),
        into their slots. Returns (each row's first row in the store (K,), its first row in the bounds (K,), its
        serial (K,): a number that changes whenever its rows do, and whether it is drawn inside out (K,); the store
        (2, S, 3), the bounds (B, 4)): what raster.transform() takes as inst_src, inst_bound, skinned and bounds."""
        plan = self._plan
        palette, normal_palette, bounds, flip = self._frame
        chunks, row_of = [], {s: k for k, s in enumerate(skin_of.tolist())}
        for s in row_of:
            slot = self._slots[id(plan.objects[s])]
            j0, j1 = plan.skin_joint[s], plan.skin_joint[s + 1]
            key = palette[j0:j1].tobytes() + flip[s].tobytes()
            if slot.key is not None and slot.key == key:
                continue
            slot.key = key
            self._serial += 1
            slot.serial = self._serial
            m = int(pack_mesh[row_of[s]])
            first = int(pack["mesh_vertex"][m])
            if int(pack["mesh_vertex"][m + 1]) - first != slot.count:
                raise ValueError("a skinned mesh's vertices changed in place (call Renderer.invalidate())")
            for c in range(0, slot.count, VERTEX_CHUNK):
                chunks.append((first + c, plan.skin_weight[s] + c, j0, slot.start + c, min(VERTEX_CHUNK, slot.count - c)))
        if chunks:
            c = np.array(chunks, np.int64)
            skin(pack["vertices"], pack["normals"], plan.weight_joints, plan.weights, palette, normal_palette,
                 c[:, 0].copy(), c[:, 1].copy(), c[:, 2].copy(), c[:, 3].copy(), c[:, 4].copy(), self._store)
        slots = [self._slots[id(plan.objects[s])] for s in skin_of.tolist()]
        start = np.array([slot.start for slot in slots], np.int64)
        serial = np.array([slot.serial for slot in slots], np.int64)
        return start, plan.skin_bound[skin_of], serial, flip[skin_of], self._store, bounds


def _skin_key(mesh):
    """What a mesh's skinned rows depend on but the pose, compared by identity: the mesh and its arrays."""
    return mesh, mesh.vertices, mesh.faces, mesh.normals, mesh.joints, mesh.weights


def _is(a, b):
    return a is b
