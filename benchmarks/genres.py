# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
"""Benchmark scenes that load the engine as three kinds of game would (python -m benchmarks.bench --scene corridor):

  corridor  a first-person shooter: a level of 16 rooms loaded as a few big textured meshes (most of it out of
            sight), props, patrolling characters, shadowed lamps, a gun held in front of the camera, muzzle
            flashes and sparks, and a camera walking through it every frame
  arena     an isometric hack and slash: a walled courtyard in sunlight, 100 characters closing in on the player
            and dying in bursts of sparks, torches, smoke, health bars, and a camera following the player
  board     a board game: a table, a board, 30 standing pieces (a few fidgeting at any time, one moving now and
            then), towers, tokens, cards and dice, a still camera that pans now and then, and a side panel

Each is a game loop in miniature, with a fixed step of DT: update() moves the world (characters walk by Clips with
crossfades, as loaded models do; particles are Object3Ds coming and going, as programs make them today) and
returns what to draw, and hud() writes the text a game would write over the picture. Characters are figures of 17
parts sharing their meshes and keyframes (as Model.copy() makes them), about 1,200 triangles each, near what a
low-poly downloaded character has. Everything is seeded, so two runs draw the same frames.
"""
import math

import numpy as np

from unicode3d.animation import Animation, Clip, RotationTrack, Track
from unicode3d.background import Gradient
from unicode3d.color import Color
from unicode3d.examples.dice import make_die, orientation_showing
from unicode3d.mesh import Mesh, make_box
from unicode3d.scene import Camera, Light, Node, Object3D, PointLight
from unicode3d.shapes import blob_mesh, block_mesh, merge_meshes
from unicode3d.transforms import quat_axis_angle, quat_mul

DT = 1.0 / 30.0  # seconds a frame
X, Y = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)


# ------------------------------------------------------------------------------------------------- textures

def value_noise(rng, size, cells):
    """(size, size) smooth noise in 0..1, varying over about size / cells texels."""
    grid = rng.uniform(0.0, 1.0, (cells + 1, cells + 1))
    t = np.arange(size) * cells / size
    i = t.astype(int)
    f = t - i
    f = f * f * (3 - 2 * f)
    fx, fy = f[None, :], f[:, None]
    top = grid[i][:, i] * (1 - fx) + grid[i][:, i + 1] * fx
    bottom = grid[i + 1][:, i] * (1 - fx) + grid[i + 1][:, i + 1] * fx
    return top * (1 - fy) + bottom * fy


def ground_texture(rng, size, color, spread=0.35):
    """Grass, dirt or stone: a colour mottled at two scales."""
    shade = 1.0 - spread + spread * (0.6 * value_noise(rng, size, 8) + 0.4 * value_noise(rng, size, 32))
    return np.clip(np.asarray(color)[None, None, :] * shade[..., None], 0.0, 1.0)


def brick_texture(rng, size, color, mortar=(0.35, 0.33, 0.30), rows=8, cols=4):
    """Bricks in courses, each a slightly different shade, with mortar between."""
    y, x = np.mgrid[0:size, 0:size] / size
    row = np.floor(y * rows)
    across = x * cols + 0.5 * (row % 2)
    col = np.floor(across)
    tint = rng.uniform(0.75, 1.1, (rows, cols + 1))[row.astype(int), col.astype(int) % (cols + 1)]
    img = np.asarray(color)[None, None, :] * (tint * (0.85 + 0.15 * value_noise(rng, size, 32)))[..., None]
    joint = (np.minimum(y * rows - row, row + 1 - y * rows) < 0.06) | (np.minimum(across - col, col + 1 - across) < 0.03)
    img[joint] = mortar
    return np.clip(img, 0.0, 1.0)


def tile_texture(rng, size, color, tiles=4):
    """Floor tiles with grout lines."""
    y, x = np.mgrid[0:size, 0:size] / size * tiles
    tint = rng.uniform(0.8, 1.05, (tiles, tiles))[y.astype(int), x.astype(int)]
    img = np.asarray(color)[None, None, :] * (tint * (0.9 + 0.1 * value_noise(rng, size, 16)))[..., None]
    img[(np.minimum(y % 1, 1 - y % 1) < 0.03) | (np.minimum(x % 1, 1 - x % 1) < 0.03)] = (0.2, 0.2, 0.2)
    return np.clip(img, 0.0, 1.0)


def wood_texture(rng, size, color):
    """Wood grain: wavy stripes along the texture's rows."""
    y, x = np.mgrid[0:size, 0:size] / size
    grain = 0.5 + 0.5 * np.sin(40 * y + 6 * value_noise(rng, size, 6) + 2 * x)
    return np.clip(np.asarray(color)[None, None, :] * (0.7 + 0.3 * grain)[..., None], 0.0, 1.0)


def board_texture(rng, size):
    """A board's art: parchment, a 12 x 12 grid, and coloured rings round the middle (as Castle Panic's mat has)."""
    y, x = np.mgrid[0:size, 0:size] / size
    img = ground_texture(rng, size, (0.85, 0.78, 0.6), 0.25)
    r = np.hypot(x - 0.5, y - 0.5)
    for k, ring in enumerate(((0.75, 0.3, 0.25), (0.3, 0.55, 0.3), (0.3, 0.4, 0.7))):
        band = (r > 0.12 + 0.12 * k) & (r < 0.22 + 0.12 * k)
        img[band] = img[band] * 0.5 + np.asarray(ring) * 0.5
    img[(np.minimum((x * 12) % 1, 1 - (x * 12) % 1) < 0.02) | (np.minimum((y * 12) % 1, 1 - (y * 12) % 1) < 0.02)] *= 0.4
    return img


def card_texture(rng, size, color):
    """A card face: a border, a coloured field and a darker emblem."""
    y, x = np.mgrid[0:size, 0:size] / size
    img = np.empty((size, size, 3))
    img[:] = color
    img[np.hypot(x - 0.5, y - 0.45) < 0.2] *= 0.5
    img[(x < 0.06) | (x > 0.94) | (y < 0.06) | (y > 0.94)] = (0.95, 0.92, 0.85)
    return img


# --------------------------------------------------------------------------------------------------- meshes

def tiled(mesh, texture, tile):
    """The mesh with `texture` repeated every `tile` units across whichever two world axes each face lies along
    most, as a level's walls and floors are textured."""
    vertices, faces = np.asarray(mesh.vertices, float), np.asarray(mesh.faces)
    tri = vertices[faces]
    across = np.abs(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])).argmax(axis=1)
    pairs = np.array([(2, 1), (0, 2), (0, 1)])[across]  # facing x: (z, y); y: (x, z); z: (x, y)
    mesh.uvs = np.take_along_axis(tri, np.repeat(pairs[:, None, :], 3, axis=1), axis=2) / tile
    mesh.materials = np.zeros(len(faces), dtype=int)
    mesh.textures = [texture]
    return mesh


def grid_mesh(size, quads, height=None):
    """A square of quads x quads quads, size units across, centred on the origin, facing up, its corners raised by
    height(x, z) where given (terrain)."""
    t = np.linspace(-size / 2, size / 2, quads + 1)
    x, z = np.meshgrid(t, t)
    y = height(x, z) if height else np.zeros_like(x)
    vertices = np.c_[x.ravel(), y.ravel(), z.ravel()]
    i = np.arange(quads)
    a = (i[:, None] * (quads + 1) + i[None, :]).ravel()
    faces = np.r_[np.c_[a, a + quads + 1, a + quads + 2], np.c_[a, a + quads + 2, a + 1]]
    return Mesh(vertices, faces)


def limb_mesh(length, radius, rings=6, segments=8):
    """A rounded limb hanging down from its joint (the origin)."""
    return blob_mesh((radius, length / 2, radius), (0.0, -length / 2, 0.0), rings=rings, segments=segments)


def cylinder_mesh(radius, height, segments=12):
    """An upright cylinder standing on the origin (a barrel, a token, a tower)."""
    a = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    ring = np.c_[radius * np.cos(a), np.zeros(segments), radius * np.sin(a)]
    vertices = np.r_[ring, ring + (0, height, 0), [(0, 0, 0), (0, height, 0)]]
    i, j = np.arange(segments), (np.arange(segments) + 1) % segments
    bottom, top = 2 * segments, 2 * segments + 1
    faces = np.r_[np.c_[i, j + segments, j], np.c_[i, i + segments, j + segments],
                  np.c_[np.full(segments, top), j + segments, i + segments], np.c_[np.full(segments, bottom), i, j]]
    return Mesh(vertices, faces)


def tower_mesh(radius, height, segments=16):
    """A round tower with a crenellated top."""
    parts = [cylinder_mesh(radius, height, segments), cylinder_mesh(radius * 1.15, height * 0.12, segments)]
    parts[1].vertices = parts[1].vertices + (0, height * 0.88, 0)
    for k in range(8):
        a = 2 * np.pi * k / 8
        parts.append(block_mesh((radius * math.cos(a), height * 1.05, radius * math.sin(a)),
                                (radius * 0.35, height * 0.14, radius * 0.35)))
    return merge_meshes(parts)


# ------------------------------------------------------------------------------------------------ characters

def _turn(axis, angle):
    return quat_axis_angle(axis, angle)


class Kit:
    """What every figure shares: its parts' meshes and its clips' keyframes (as copies of a loaded model share
    theirs)."""

    def __init__(self):
        self.meshes = {
            "hips": blob_mesh((0.18, 0.1, 0.12), rings=6, segments=10),
            "torso": blob_mesh((0.22, 0.3, 0.14), (0.0, 0.3, 0.0), rings=8, segments=12),
            "head": blob_mesh((0.12, 0.14, 0.12), (0.0, 0.14, 0.0), rings=8, segments=12),
            "upper": limb_mesh(0.3, 0.06), "fore": limb_mesh(0.28, 0.05),
            "hand": block_mesh((0.0, -0.05, 0.0), (0.08, 0.1, 0.06)),
            "thigh": limb_mesh(0.45, 0.08), "shin": limb_mesh(0.45, 0.065),
            "foot": block_mesh((0.0, -0.03, 0.06), (0.1, 0.06, 0.22)),
            "sword": merge_meshes([block_mesh((0.0, -0.4, 0.0), (0.04, 0.7, 0.015)),
                                   block_mesh((0.0, -0.04, 0.0), (0.2, 0.03, 0.04)),
                                   block_mesh((0.0, 0.05, 0.0), (0.03, 0.12, 0.03))]),
            "shield": blob_mesh((0.03, 0.2, 0.17), rings=6, segments=10),
        }
        swing = lambda amount, period, phase=0.0: [(period * k / 4, _turn(X, amount * math.cos(2 * math.pi * (k / 4 + phase))))
                                                   for k in range(5)]
        bob = [(0.2 * k, (0.0, 0.95 + 0.04 * (k % 2), 0.0)) for k in range(5)]
        self.clips = {  # name: [(joint, attribute, track)]
            "walk": [("thigh_l", "rotation", swing(0.5, 0.8)), ("thigh_r", "rotation", swing(0.5, 0.8, 0.5)),
                     ("shin_l", "rotation", swing(0.35, 0.8, 0.25)), ("shin_r", "rotation", swing(0.35, 0.8, 0.75)),
                     ("upper_l", "rotation", swing(0.4, 0.8, 0.5)), ("upper_r", "rotation", swing(0.4, 0.8)),
                     ("fore_l", "rotation", swing(0.2, 0.8, 0.5)), ("fore_r", "rotation", swing(0.2, 0.8)),
                     ("hips", "position", bob)],
            "attack": [("upper_r", "rotation", [(0.0, _turn(X, -0.2)), (0.25, _turn(X, -2.2)), (0.4, _turn(X, 0.6)),
                                                (0.6, _turn(X, -0.2))]),
                       ("fore_r", "rotation", [(0.0, _turn(X, -0.3)), (0.25, _turn(X, -1.0)), (0.4, _turn(X, -0.1)),
                                               (0.6, _turn(X, -0.3))]),
                       ("torso", "rotation", [(0.0, _turn(Y, 0.0)), (0.25, _turn(Y, 0.3)), (0.4, _turn(Y, -0.4)),
                                              (0.6, _turn(Y, 0.0))]),
                       ("thigh_l", "rotation", [(0.0, _turn(X, 0.3)), (0.6, _turn(X, 0.3))]),
                       ("thigh_r", "rotation", [(0.0, _turn(X, -0.2)), (0.6, _turn(X, -0.2))])],
            "idle": [("torso", "rotation", [(0.0, _turn(X, 0.0)), (1.2, _turn(X, 0.06)), (2.4, _turn(X, 0.0))]),
                     ("head", "rotation", [(0.0, _turn(Y, -0.25)), (1.2, _turn(Y, 0.25)), (2.4, _turn(Y, -0.25))]),
                     ("upper_l", "rotation", [(0.0, _turn(X, 0.05)), (1.2, _turn(X, -0.08)), (2.4, _turn(X, 0.05))]),
                     ("upper_r", "rotation", [(0.0, _turn(X, -0.05)), (1.2, _turn(X, 0.08)), (2.4, _turn(X, -0.05))]),
                     ("hips", "position", [(0.0, (0.0, 0.95, 0.0)), (1.2, (0.0, 0.93, 0.0)), (2.4, (0.0, 0.95, 0.0))])],
        }
        self.tracks = {name: [(joint, attribute, (RotationTrack if attribute == "rotation" else Track)(keys))
                              for joint, attribute, keys in tracks] for name, tracks in self.clips.items()}


class Figure:
    """A jointed character of 17 parts (hips, torso, head, arms, hands, legs, feet, a sword and a shield) under a
    root Node, played by Clips (walk, attack, idle) that crossfade into each other."""

    def __init__(self, kit, palette, position, scale=1.0, heading=0.0):
        skin, cloth, metal = palette
        self.root = Node(position=np.array(position, float), rotation=_turn(Y, heading), scale=scale)
        j = self.joints = {}

        def part(name, mesh, parent, at, color, **options):
            j[name] = Object3D(kit.meshes[mesh], np.array(at, float), parent=parent, color=color, **options)
            return j[name]
        hips = part("hips", "hips", self.root, (0.0, 0.95, 0.0), cloth)
        torso = part("torso", "torso", hips, (0.0, 0.05, 0.0), cloth, specular=0.3)
        part("head", "head", torso, (0.0, 0.62, 0.0), skin, specular=0.2)
        for side, s in (("l", -1), ("r", 1)):
            upper = part("upper_" + side, "upper", torso, (0.28 * s, 0.55, 0.0), cloth)
            fore = part("fore_" + side, "fore", upper, (0.0, -0.3, 0.0), skin)
            part("hand_" + side, "hand", fore, (0.0, -0.28, 0.0), skin)
            thigh = part("thigh_" + side, "thigh", hips, (0.1 * s, -0.02, 0.0), cloth)
            shin = part("shin_" + side, "shin", thigh, (0.0, -0.45, 0.0), cloth)
            part("foot_" + side, "foot", shin, (0.0, -0.45, 0.0), (60, 45, 35), specular=0.1)
        self.sword = part("sword", "sword", j["hand_r"], (0.0, -0.08, 0.0), metal, specular=1.5, shininess=60.0)
        self.sword.rotation = _turn(X, math.pi / 2)
        part("shield", "shield", j["fore_l"], (-0.07, -0.15, 0.0), metal, specular=0.8)
        self.parts = list(j.values())
        self.clips = {name: Clip([Animation(j[joint], **{attribute: track}) for joint, attribute, track in tracks],
                                 name=name, loop="loop") for name, tracks in kit.tracks.items()}
        self.clip = None

    def play(self, name, fade=0.15, at=0.0):
        """Crossfade into the named clip (if not already playing it), from `at` seconds into it."""
        if self.clip is self.clips[name]:
            return
        self.clip = self.clips[name]
        self.clip.start(fade)
        if at:
            self.clip.time = at

    def update(self, dt):
        if self.clip is not None:
            self.clip.update(dt)

    def face(self, direction):
        self.root.rotation = _turn(Y, math.atan2(direction[0], direction[2]))

    def at(self, height=1.9):
        """A point above the figure's head (for a label or a health bar)."""
        return self.root.position + (0.0, height * self.root.scale, 0.0)


PALETTES = (((205, 160, 130), (150, 40, 40), (190, 190, 200)),
            ((110, 150, 90), (90, 70, 50), (150, 140, 120)),
            ((180, 170, 150), (60, 60, 90), (200, 170, 80)),
            ((150, 110, 90), (40, 90, 60), (170, 170, 175)))


class Particles:
    """Short-lived Object3Ds (sparks, smoke) moved in Python each frame, made and dropped as programs make them
    today (Castle Panic's fx.py)."""

    def __init__(self, rng):
        self.rng = rng
        self.spark = blob_mesh((1.0, 1.0, 1.0), rings=3, segments=4)
        self.puff = blob_mesh((1.0, 1.0, 1.0), rings=5, segments=8)
        self.items = []  # [object, velocity, seconds left, gravity, growth a second, fade a second]

    def burst(self, at, count, speed, life, color, gravity=-9.8, up=2.0):
        for _ in range(count):
            v = self.rng.normal(0.0, speed, 3) + (0.0, up, 0.0)
            obj = Object3D(self.spark, np.array(at, float), scale=0.05, color=color, emissive=1.0,
                           cast_shadows=False, specular=0.0)
            self.items.append([obj, v, life * self.rng.uniform(0.6, 1.0), gravity, 0.0, 0.0])

    def smoke(self, at):
        obj = Object3D(self.puff, np.array(at, float) + self.rng.normal(0.0, 0.05, 3), scale=0.12,
                       color=(120, 120, 120), opacity=0.45, cast_shadows=False, specular=0.0)
        self.items.append([obj, np.array([0.0, 0.8, 0.0]) + self.rng.normal(0.0, 0.1, 3), 2.0, 0.0, 0.15, 0.2])

    def update(self, dt):
        kept = []
        for item in self.items:
            obj, v, life, gravity, grow, fade = item
            item[2] = life = life - dt
            if life <= 0.0:
                continue
            v[1] += gravity * dt
            obj.position = obj.position + v * dt
            if grow:
                obj.scale = obj.scale + grow * dt
            if fade:
                obj.opacity = max(0.0, obj.opacity - fade * dt)
            kept.append(item)
        self.items = kept

    @property
    def objects(self):
        return [item[0] for item in self.items]


# -------------------------------------------------------------------------------------------- the scenes

class Game:
    """What bench.Frames drives: objects (the scene at the start, for the counts), renderer_options, frames (how
    many to time by default), update(dt) -> (objects, camera, lights) and hud(screen, renderer)."""
    frames = 300
    renderer_options = {}


class Corridor(Game):
    """A first-person shooter's level: 4 x 4 rooms of 5 x 5 cells joined by doorways, walls, floor and ceiling as
    three big meshes (as a level loaded from glTF often is), pillars, crates and barrels, a lamp with a shadowed
    light in every room (the game passes those near the player), 10 characters patrolling rooms, and the player
    walking round the outer rooms with a gun in front of the camera, firing twice a second."""
    renderer_options = dict(fog=0.45, simplify=1.0, background=(10, 10, 14))

    ROOMS, ROOM, CELL, HEIGHT = 4, 5, 2.5, 3.5

    def __init__(self):
        rng = self.rng = np.random.default_rng(11)
        n = self.ROOMS * (self.ROOM + 1) + 1
        wall = np.ones((n, n), bool)
        for i in range(self.ROOMS):
            for k in range(self.ROOMS):
                r, c = i * (self.ROOM + 1) + 1, k * (self.ROOM + 1) + 1
                wall[r:r + self.ROOM, c:c + self.ROOM] = False
                middle = self.ROOM // 2
                if k + 1 < self.ROOMS:
                    wall[r + middle, c + self.ROOM] = False
                if i + 1 < self.ROOMS:
                    wall[r + self.ROOM, c + middle] = False
        C, H = self.CELL, self.HEIGHT
        at = lambda row, col: np.array([(col + 0.5) * C, 0.0, (row + 0.5) * C])
        blocks, trims, floors = [], [], []
        for row in range(n):
            for col in range(n):
                p = at(row, col)
                if wall[row, col]:
                    blocks.append(block_mesh(p + (0, H / 2, 0), (C, H, C)))
                    continue
                floors.append(block_mesh(p + (0, -0.05, 0), (C, 0.1, C)))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):  # skirting along walls
                    if wall[row + dr, col + dc]:
                        trims.append(block_mesh(p + (dc * (C / 2 - 0.05), 0.15, dr * (C / 2 - 0.05)),
                                                (C if dc == 0 else 0.1, 0.3, C if dr == 0 else 0.1)))
        centres = [at(i * (self.ROOM + 1) + 1 + self.ROOM // 2, k * (self.ROOM + 1) + 1 + self.ROOM // 2)
                   for i in range(self.ROOMS) for k in range(self.ROOMS)]
        for room, centre in enumerate(centres):
            if room % 2 == 0:
                for dx in (-1, 1):
                    for dz in (-1, 1):
                        p = centre + (dx * 1.6 * C, 0, dz * 1.6 * C)
                        blocks += [block_mesh(p + (0, H / 2, 0), (0.5, H, 0.5)), block_mesh(p + (0, 0.15, 0), (0.8, 0.3, 0.8)),
                                   block_mesh(p + (0, H - 0.15, 0), (0.8, 0.3, 0.8))]
        size = n * C
        self.level = [Object3D(tiled(merge_meshes(blocks), brick_texture(rng, 512, (0.6, 0.5, 0.42)), 2.5),
                               color=(255, 255, 255), specular=0.1),
                      Object3D(tiled(merge_meshes(trims), ground_texture(rng, 256, (0.35, 0.3, 0.25)), 1.0),
                               color=(255, 255, 255), specular=0.3),
                      Object3D(tiled(merge_meshes(floors), tile_texture(rng, 512, (0.55, 0.55, 0.5)), 2.5),
                               color=(255, 255, 255), specular=0.5, shininess=40.0),
                      Object3D(tiled(grid_mesh(size, n), ground_texture(rng, 256, (0.5, 0.48, 0.45)), 2.5),
                               np.array([size / 2, H, size / 2]), _turn(X, math.pi), color=(255, 255, 255),
                               specular=0.0)]
        crate = make_box(1.0, [ground_texture(rng, 256, (0.55, 0.4, 0.22), 0.5)] * 6)
        barrel = cylinder_mesh(0.35, 1.0, 14)
        lamp = blob_mesh((0.25, 0.12, 0.25), rings=6, segments=12)
        self.props, self.lamps = [], []
        for centre in centres:
            for _ in range(rng.integers(4, 8)):
                p = centre + (rng.uniform(-2.0, 2.0) * C / 2, 0.0, rng.uniform(-2.0, 2.0) * C / 2)
                if rng.random() < 0.6:
                    s = rng.uniform(0.6, 1.1)
                    self.props.append(Object3D(crate, p + (0, s / 2, 0), _turn(Y, rng.uniform(0, 1.5)), scale=s,
                                               color=(255, 255, 255), specular=0.1))
                else:
                    self.props.append(Object3D(barrel, p, color=(110, 70, 40), specular=0.6))
            self.props.append(Object3D(lamp, centre + (0, H - 0.15, 0), color=(255, 220, 160), emissive=1.0,
                                       cast_shadows=False))
            self.lamps.append(PointLight(centre + (0, H - 0.5, 0), color=(255, 215, 170), diffuse=0.9, range=9.0,
                                         shadows=True))
        self.ambient = Light(direction=(0.2, -1.0, 0.1), ambient=0.12, diffuse=0.0, specular=0.0)
        self.flash_light = PointLight(np.zeros(3), color=(255, 200, 120), diffuse=1.0, range=6.0)

        kit = Kit()
        self.walkers = []
        for k, room in enumerate(rng.choice(len(centres), 10, replace=False)):
            a = centres[room] + (rng.uniform(-1, 1) * C, 0, -1.5 * C)
            b = centres[room] + (rng.uniform(-1, 1) * C, 0, 1.5 * C)
            fig = Figure(kit, PALETTES[k % 4], a, heading=0.0)
            fig.play("walk", 0.0, at=rng.uniform(0, 0.8))
            self.walkers.append([fig, a, b, rng.uniform(0, 1)])

        ring = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 3), (2, 3), (3, 3), (3, 2), (3, 1), (3, 0), (2, 0), (1, 0)]
        self.path = np.array([centres[i * self.ROOMS + k] for i, k in ring] + [centres[0]]) + (0, 1.6, 0)
        self.lengths = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(self.path, axis=0), axis=1))]
        self.camera = Camera(position=self.path[0].copy(), target=self.path[1].copy(), fov=70.0, far=60.0)
        gun = merge_meshes([block_mesh((0.0, 0.0, -0.2), (0.07, 0.09, 0.45)), block_mesh((0.0, -0.08, -0.05), (0.06, 0.16, 0.08)),
                            block_mesh((0.0, 0.07, -0.15), (0.03, 0.05, 0.2)),
                            blob_mesh((0.025, 0.025, 0.12), (0.0, 0.0, -0.45), rings=4, segments=8)])
        self.gun = Object3D(gun, np.array([0.22, -0.2, -0.45]), color=(70, 70, 75), specular=1.2, shininess=50.0,
                            parent=self.camera, cast_shadows=False)
        self.flash = Object3D(blob_mesh((0.06, 0.06, 0.1)), np.array([0.22, -0.2, -0.98]), color=(255, 220, 120),
                              emissive=1.0, cast_shadows=False, parent=self.camera, visible=False)
        self.particles = Particles(rng)
        self.figures = [w[0] for w in self.walkers]
        self.objects = self.level + self.props + [p for f in self.figures for p in f.parts] + [self.gun, self.flash]
        self.time, self.frame = 0.0, 0

    def _along(self, s):
        s %= self.lengths[-1]
        i = min(int(np.searchsorted(self.lengths, s, side="right")) - 1, len(self.path) - 2)
        f = (s - self.lengths[i]) / (self.lengths[i + 1] - self.lengths[i])
        return self.path[i] + (self.path[i + 1] - self.path[i]) * f

    def update(self, dt):
        self.time += dt
        self.frame += 1
        s = 3.5 * self.time
        eye, ahead = self._along(s), self._along(s + 3.0)
        sway = 0.25 * math.sin(0.7 * self.time)
        look = ahead - eye
        look = (look[0] * math.cos(sway) - look[2] * math.sin(sway), 0.0, look[0] * math.sin(sway) + look[2] * math.cos(sway))
        self.camera.position = eye + (0.0, 0.04 * math.sin(2 * math.pi * 1.6 * self.time), 0.0)
        self.camera.target = self.camera.position + np.array(look) + (0.0, -0.05, 0.0)
        for walker in self.walkers:
            fig, a, b, phase = walker
            walker[3] = phase = phase + dt * 1.2 / np.linalg.norm(b - a)
            f = phase % 2.0
            f = f if f < 1.0 else 2.0 - f
            fig.root.position = a + (b - a) * f
            fig.face(b - a if (phase % 2.0) < 1.0 else a - b)
            fig.update(dt)
        firing = self.frame % 15 < 2
        self.flash.visible = firing
        self.gun.position = np.array([0.22, -0.2 + 0.01 * math.sin(2 * math.pi * 1.6 * self.time),
                                      -0.45 + (0.03 if firing else 0.0)])
        if self.frame % 15 == 0:
            forward = self.camera.target - self.camera.position
            self.particles.burst(self.camera.position + forward / np.linalg.norm(forward) * 6.0, 12, 1.5, 0.4,
                                 (255, 210, 120))
        self.particles.update(dt)
        lights = [self.ambient] + [lamp for lamp in self.lamps if np.linalg.norm(lamp.position - eye) < 12.0]
        if firing:
            self.flash_light.position = self.flash.to_world((0.0, 0.0, 0.0))
            lights.append(self.flash_light)
        return self.objects + self.particles.objects, self.camera, lights

    def hud(self, screen, renderer):
        rows, cols = renderer.height, renderer.width
        screen.text(rows // 2, cols // 2, "+", Color.WHITE, bold=True)
        screen.text(rows - 2, 2, "HEALTH", (220, 220, 220))
        screen.bar(rows - 2, 9, 20, 0.8, Color.RED)
        screen.text(rows - 2, cols - 16, f"AMMO {30 - self.frame // 15 % 30:2d} / 120", (230, 210, 120))
        screen.text(0, 0, f"{1 / DT:.0f} fps  frame {self.frame}", Color.WHITE)


class Arena(Game):
    """An isometric hack and slash: a walled courtyard of rough ground in sunlight with rocks, pillars, trees and
    burning braziers (torch lights, smoke), the player walking a circle and swinging, 100 characters closing in,
    fighting when near, dying in bursts of sparks and coming back from the edge, health bars over the nearest,
    and the camera following the player from above."""
    renderer_options = dict(simplify=1.0, background=Gradient((60, 80, 110), (20, 25, 35)))

    def __init__(self, enemies=100):
        rng = self.rng = np.random.default_rng(12)
        size = 60.0
        height = lambda x, z: 0.15 * np.sin(0.4 * x) * np.cos(0.3 * z) + 0.1 * np.sin(1.3 * x + 0.7 * z)
        self.height = height
        ground = Object3D(tiled(grid_mesh(size, 48, height), ground_texture(rng, 512, (0.45, 0.5, 0.3)), 4.0),
                          color=(255, 255, 255), specular=0.05)
        bricks = brick_texture(rng, 512, (0.55, 0.52, 0.48))
        walls = []
        for k in range(48):
            a = 2 * np.pi * k / 48
            rot = np.array([[math.cos(a), 0, -math.sin(a)], [0, 1, 0], [math.sin(a), 0, math.cos(a)]])
            p = np.array([26 * math.cos(a), 1.5, 26 * math.sin(a)])
            walls.append(block_mesh(p, (1.0, 3.0, 3.6), rot))
            if k % 2 == 0:
                walls.append(block_mesh(p + (0, 1.8, 0), (1.1, 0.6, 1.2), rot))
        self.scenery = [ground, Object3D(tiled(merge_meshes(walls), bricks, 2.0), color=(255, 255, 255), specular=0.1)]
        rocks = [blob_mesh((0.6, 0.4, 0.5), rings=6, segments=10,
                           bump=lambda d, s=s: 1.0 + 0.2 * np.sin(5 * d[:, 0] + s) * np.cos(4 * d[:, 2] - s))
                 for s in range(3)]
        pillar = merge_meshes([block_mesh((0, 1.5, 0), (0.6, 3.0, 0.6)), block_mesh((0, 0.15, 0), (0.9, 0.3, 0.9)),
                               block_mesh((0, 3.0, 0), (0.9, 0.3, 0.9))])
        trunk, crown = cylinder_mesh(0.18, 1.6, 8), blob_mesh((1.0, 1.2, 1.0), rings=8, segments=12)
        flame = blob_mesh((0.2, 0.3, 0.2), rings=5, segments=8)
        brazier = merge_meshes([cylinder_mesh(0.1, 0.9, 8), cylinder_mesh(0.35, 0.25, 12)])
        brazier.vertices = brazier.vertices + np.r_[np.zeros((len(brazier.vertices) - 26, 3)),
                                                    np.tile((0.0, 0.9, 0.0), (26, 1))]
        spot = lambda r_lo, r_hi: (lambda r, a: np.array([r * math.cos(a), 0.0, r * math.sin(a)]))(
            rng.uniform(r_lo, r_hi), rng.uniform(0, 2 * np.pi))
        for _ in range(40):
            p = spot(10, 24)
            self.scenery.append(Object3D(rocks[rng.integers(3)], p, _turn(Y, rng.uniform(0, 6)),
                                         scale=rng.uniform(0.6, 1.6), color=(130, 125, 115), specular=0.1))
        for _ in range(16):
            p = spot(12, 24)
            self.scenery.append(Object3D(pillar, p, color=(180, 175, 160), specular=0.2))
        for _ in range(14):
            p = spot(14, 24)
            self.scenery += [Object3D(trunk, p, color=(90, 65, 40), specular=0.0),
                             Object3D(crown, p + (0, 2.4, 0), scale=rng.uniform(0.8, 1.3), color=(60, 110, 50),
                                      specular=0.1)]
        self.torches, self.braziers = [], []
        for k in range(8):
            a = 2 * np.pi * (k + 0.5) / 8
            p = np.array([9.0 * math.cos(a), 0.0, 9.0 * math.sin(a)])
            self.scenery += [Object3D(brazier, p, color=(70, 60, 55), specular=0.6),
                             Object3D(flame, p + (0, 1.3, 0),
                                      color=(255, 160, 60), emissive=1.0, cast_shadows=False)]
            self.braziers.append(p + (0, 1.4, 0))
            self.torches.append(PointLight(p + (0, 1.6, 0), color=(255, 170, 90), diffuse=0.9, range=7.0))
        self.sun = Light(direction=(0.45, -1.0, -0.35), ambient=0.28, diffuse=0.75, shadows=True)
        self.spell = PointLight(np.zeros(3), color=(120, 170, 255), diffuse=0.8, range=6.0)

        kit = Kit()
        self.player = Figure(kit, ((230, 190, 160), (40, 60, 140), (220, 220, 230)), (7.0, 0.0, 0.0), scale=1.1)
        self.player.play("walk", 0.0)
        self.enemies = []
        for k in range(enemies):
            fig = Figure(kit, PALETTES[k % 4], spot(6, 24), scale=0.9)
            fig.play("walk", 0.0, at=rng.uniform(0, 0.8))
            self.enemies.append([fig, rng.uniform(1.2, 2.0), 0.0])  # [figure, speed, seconds until back if dead]
        self.particles = Particles(rng)
        self.camera = Camera(fov=40.0, far=80.0)
        self.objects = self.scenery + self.player.parts + [p for e in self.enemies for p in e[0].parts]
        self.time, self.frame = 0.0, 0

    def update(self, dt):
        rng = self.rng
        self.time += dt
        self.frame += 1
        a = self.time * 3.0 / 7.0
        p = np.array([7.0 * math.cos(a), 0.0, 7.0 * math.sin(a)])
        self.player.root.position = p
        self.player.face((-math.sin(a), 0.0, math.cos(a)))
        if self.frame % 20 == 0:
            self.player.play("attack", 0.1)
        elif self.frame % 20 == 12:
            self.player.play("walk", 0.15)
        self.player.update(dt)
        if self.frame % 20 == 8:
            self.particles.burst(self.player.sword.to_world((0.0, -0.7, 0.0)), 15, 1.5, 0.5, (255, 230, 150), up=1.0)
        drawn = list(self.scenery) + self.player.parts
        for enemy in self.enemies:
            fig, speed, gone = enemy
            if gone > 0.0:
                enemy[2] = gone - dt
                if enemy[2] <= 0.0:
                    r, b = rng.uniform(18, 24), rng.uniform(0, 2 * np.pi)
                    fig.root.position = np.array([r * math.cos(b), 0.0, r * math.sin(b)])
                    fig.play("walk", 0.0)
                continue
            to = p - fig.root.position
            to[1] = 0.0
            d = float(np.linalg.norm(to))
            fig.face(to)
            if d > 1.6:
                fig.root.position = fig.root.position + to / d * speed * dt
                fig.play("walk")
            else:
                fig.play("attack")
                if rng.random() < 0.02:  # struck down: sparks where it stood, back from the edge in 2 s
                    self.particles.burst(fig.root.position + (0, 1.0, 0), 30, 2.0, 0.7, (255, 120, 60))
                    enemy[2] = 2.0
                    continue
            fig.update(dt)
            drawn += fig.parts
        if self.frame % 3 == 0:
            for b in self.braziers:
                self.particles.smoke(b)
        self.particles.update(dt)
        target = p + (0.0, 0.8, 0.0)
        self.camera.position = target + (0.0, 9.0, 6.5)
        self.camera.target = target
        self.spell.position = p + (0.0, 2.0, 0.0)
        lights = [self.sun, self.spell] + [t for t in self.torches if np.linalg.norm(t.position - p) < 20.0]
        return drawn + self.particles.objects, self.camera, lights

    def hud(self, screen, renderer):
        p = self.player.root.position
        near = sorted((e for e in self.enemies if e[2] <= 0.0),
                      key=lambda e: float(np.sum((e[0].root.position - p) ** 2)))[:20]
        for fig, speed, _ in near:
            anchor = renderer.anchor(fig.at(), fig.parts)
            if anchor is not None and not anchor.hidden:
                screen.bar(anchor.y - 1, anchor.x - 2, 5, 0.3 + 0.35 * speed, Color.RED)
        rows, cols = renderer.height, renderer.width
        screen.text(rows - 3, 2, "LIFE", (230, 80, 80))
        screen.bar(rows - 3, 7, 24, 0.7, Color.RED)
        screen.text(rows - 2, 2, "MANA", (90, 140, 255))
        screen.bar(rows - 2, 7, 24, 0.45, Color.BLUE)
        screen.text(rows - 2, cols // 2 - 12, "[1] Cleave [2] Leap [3] Nova", (220, 210, 180))
        screen.text(0, 0, f"Kills: {self.frame // 7}", Color.WHITE)


class Board(Game):
    """A board game on a table: a 1024-texel board, 30 standing pieces (6 fidgeting at any time, a different one
    starting each second; one moving to a nearby square every 4 s), 6 towers, 60 tokens, 12 cards and 2 dice rolled
    every 6 s, in sunlight with shadows; the camera still but for a pan in the last 2 s of every 10, and a side
    panel of about 100 lines of text with labels over a few pieces (as Castle Panic draws)."""
    renderer_options = dict(simplify=1.0, background=Gradient((40, 35, 30), (10, 8, 6)))

    def __init__(self):
        rng = self.rng = np.random.default_rng(13)
        table = make_box(1.0, [wood_texture(rng, 1024, (0.55, 0.36, 0.2))] * 6)
        board = make_box(1.0, [board_texture(rng, 1024)] * 6)
        self.still = [Object3D(table, np.array([0.0, -0.5, 0.0]), scale=(26.0, 1.0, 18.0), color=(255, 255, 255),
                               specular=0.4),
                      Object3D(board, np.array([0.0, 0.05, 0.0]), scale=(14.0, 0.1, 14.0), color=(255, 255, 255),
                               specular=0.2)]
        tower = tower_mesh(0.5, 1.6)
        for k in range(6):
            a = 2 * np.pi * k / 6
            self.still.append(Object3D(tower, np.array([1.6 * math.cos(a), 0.1, 1.6 * math.sin(a)]),
                                       color=(170, 165, 150), specular=0.2))
        token = cylinder_mesh(0.25, 0.08, 16)
        for _ in range(60):
            self.still.append(Object3D(token, np.array([rng.uniform(-6.5, 6.5), 0.1, rng.uniform(-6.5, 6.5)]),
                                       color=tuple(int(c) for c in rng.integers(60, 230, 3)), specular=0.8))
        cards = [make_box(1.0, [card_texture(rng, 128, c)] * 6) for c in ((0.7, 0.2, 0.2), (0.2, 0.4, 0.7), (0.3, 0.6, 0.3))]
        for k in range(12):
            self.still.append(Object3D(cards[k % 3], np.array([-4.5 + 0.9 * k, 0.02, 8.0]), _turn(Y, rng.uniform(-0.1, 0.1)),
                                       scale=(0.8, 0.02, 1.1), color=(255, 255, 255), specular=0.3))
        kit = Kit()
        squares = [(x, z) for x in range(-6, 7) for z in range(-6, 7) if math.hypot(x, z) > 2.8]
        chosen = rng.choice(len(squares), 30, replace=False)
        self.pieces = []
        for k, s in enumerate(chosen):
            x, z = squares[s]
            fig = Figure(kit, PALETTES[k % 4], (x + 0.0, 0.1, z + 0.0), scale=0.45, heading=math.atan2(-x, -z))
            self.pieces.append(fig)
        self.fidgeting = []  # [figure, seconds left]
        self.move = None  # (figure, from, to, seconds in)
        self.die = make_die()
        self.dice = [Object3D(self.die, np.array([5.0 + k, 0.35, 7.5]), orientation_showing(k, 0.0), scale=0.5,
                              color=(230, 230, 230)) for k in range(2)]
        self.sun = Light(direction=(0.35, -1.0, -0.45), ambient=0.3, diffuse=0.7, shadows=True)
        self.camera = Camera(position=np.array([0.0, 13.0, 11.0]), target=np.array([0.0, 0.0, 0.5]), fov=45.0)
        self.objects = self.still + [p for f in self.pieces for p in f.parts] + self.dice
        self.time, self.frame = 0.0, 0
        self.log = [f"Turn {k // 4 + 1}: player {k % 4 + 1} {('draws a card', 'moves a goblin', 'rolls 7', 'plays Brick')[k % 4]}"
                    for k in range(40)]

    def update(self, dt):
        rng = self.rng
        self.time += dt
        self.frame += 1
        if self.frame % 30 == 0:
            fig = self.pieces[rng.integers(len(self.pieces))]
            if all(f is not fig for f, _ in self.fidgeting):
                fig.play("idle", 0.3, at=rng.uniform(0, 2.4))
                self.fidgeting.append([fig, 6.0])
        kept = []
        for item in self.fidgeting:
            item[1] -= dt
            if item[1] > 0.0:
                item[0].update(dt)
                kept.append(item)
            else:
                item[0].clip = None
        self.fidgeting = kept
        if self.frame % 120 == 0 and self.move is None:
            fig = self.pieces[rng.integers(len(self.pieces))]
            start = fig.root.position.copy()
            self.move = (fig, start, start + (rng.choice((-1.0, 1.0)), 0.0, rng.choice((-1.0, 1.0))), 0.0)
            fig.play("walk", 0.15)
        if self.move is not None:
            fig, a, b, t = self.move
            t += dt
            f = min(t / 0.8, 1.0)
            fig.root.position = a + (b - a) * f + (0.0, 0.3 * math.sin(math.pi * f), 0.0)
            fig.face(b - a)
            fig.update(dt)
            self.move = (fig, a, b, t) if f < 1.0 else None
            if f >= 1.0:
                fig.clip = None
        roll = self.time % 6.0
        for k, die in enumerate(self.dice):
            if roll < 1.0:
                die.position = np.array([5.0 + k - 3.0 * roll, 0.35 + 1.2 * math.sin(math.pi * roll), 7.5 - roll])
                die.rotation = quat_mul(_turn((0.6, 0.3 + k, 0.2), 0.4), die.rotation)
        cycle = self.time % 10.0
        if cycle > 8.0:
            a = 0.35 * math.sin(math.pi * (cycle - 8.0) / 2.0)
            self.camera.position = np.array([17.0 * math.sin(a), 13.0, 11.0 * math.cos(a) + 0.0])
        return self.objects, self.camera, self.sun

    def hud(self, screen, renderer):
        rows, cols = renderer.height, renderer.width
        x = cols - 34
        screen.text(0, 0, f" Round 3  |  Player 2's turn  |  Phase: {('draw', 'trade', 'move', 'fight')[self.frame // 90 % 4]} ",
                    (0, 0, 0), bg=(200, 180, 120))
        screen.text(1, x, "┌" + "─" * 32 + "┐", (180, 160, 120))
        for r in range(2, rows - 1):
            screen.text(r, x, "│", (180, 160, 120))
            screen.text(r, x + 33, "│", (180, 160, 120))
        screen.text(rows - 1, x, "└" + "─" * 32 + "┘", (180, 160, 120))
        lines = (["Hand:"] + [f"  {name}" for name in ("Archer", "Knight", "Swordsman", "Brick", "Mortar")]
                 + ["", "Castle:"] + [f"  Tower {k + 1}: {'#' * (3 - k % 3)}" for k in range(6)] + ["", "Log:"]
                 + self.log[self.frame // 30 % 20:][:rows])
        for r, line in enumerate(lines[:rows - 3]):
            screen.text(r + 2, x + 2, line[:30], Color.WHITE if r % 7 else (240, 210, 120))
        for fig in self.pieces[:4]:
            screen.label(renderer, fig.at(), "Goblin", (230, 230, 230), owner=fig.parts)
        screen.text(rows - 1, 2, "Enter: end turn   Tab: next card   F1: help", (150, 150, 150))


GENRES = {"corridor": Corridor, "arena": Arena, "board": Board}
