# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
"""Walk around a courtyard showing off the engine: python -m unicode3d.examples.room

WASD walks, Q/E or Left/Right turn, Up/Down look up and down, Esc quits. The
crosshair names what it is on; click anything to name that. The panel switches
the lamp and the sun, picks the background (a sky, a starry sky box, a
gradient or none), sets how thick the fog is and whether it fades into the
background or into mist. There are shadows (the sun and the lamp), glass,
cut-outs (a tree, a trellis), stained glass, a mirror, a pool and a chrome ball
among the things to see; F5 and F6 switch shadows and reflections.

Newer things to find: a bench of balls in different materials (rubber, paint,
gold, pearl) beside an egg (a sphere, stretched), a table and bench built from
stretched cubes, a door that swings open as you come near (keyframe animation,
overshooting as it opens), dice bobbing in their case, and an orrery turning on
looping tracks.

Walking feels best in a terminal that reports key releases (kitty, foot,
Ghostty, WezTerm, Alacritty, iTerm2, Windows Terminal): elsewhere a held key is
worked out from its repeats, and a tap walks for about half a second.
"""
import argparse
import colorsys

import numpy as np

from .dice import make_die, orientation_showing
from .hud import StatusBar
from ..animation import Animation, RotationTrack, Track
from ..background import Fog, Gradient, Sky, SkyBox
from ..color import Color
from ..keys import Key, MouseEvent
from ..mesh import Mesh, make_box
from ..scene import Camera, Light, Node, Object3D, PointLight, Renderer
from ..shapes import blob_mesh, block_mesh, text_mesh
from ..terminal import add_display_args, display_options, run
from ..transforms import UP, quat_axis_angle, quat_mul
from ..ui import Choice, Panel, Slider, Toggle

HALF = 11.0          # the courtyard runs from -HALF to HALF in x and z
WALL_HEIGHT = 3.5
DIE_SIZE = 0.34      # of the dice in the glass case
EYE = 1.6
RADIUS = 0.35        # how close the walker gets to walls and things
WALK, TURN, LOOK = 3.0, 1.8, 1.2  # units a second, radians a second
MIST = (225, 228, 232)  # the fog's colour when it is mist rather than the background
DOOR_AT = np.array([0.85, 0.0, 1.0])  # the door's hinge; it is 1.1 wide, along +x
DOOR_NEAR = 2.3      # how close the walker comes before the door opens

FONT = {  # 5 rows a letter, for the sign on the north wall
    "U": ["#...#", "#...#", "#...#", "#...#", ".###."],
    "N": ["#...#", "##..#", "#.#.#", "#..##", "#...#"],
    "I": ["###", ".#.", ".#.", ".#.", "###"],
    "C": [".####", "#....", "#....", "#....", ".####"],
    "O": [".###.", "#...#", "#...#", "#...#", ".###."],
    "D": ["####.", "#...#", "#...#", "#...#", "####."],
    "E": ["#####", "#....", "####.", "#....", "#####"],
    "3": ["####.", "....#", ".###.", "....#", "####."],
}


# ----- textures ------------------------------------------------------------------------------

def tiles(res, a, b, n):
    y, x = np.mgrid[0:res, 0:res] * n // res
    return np.where(((x + y) % 2)[..., None], a, b).astype(float)


def plaster(rng, res=64):
    """A pale wall with a darker band along the bottom and a little grain."""
    tex = np.ones((res, res, 3)) * (0.86, 0.8, 0.7)
    tex[-res // 6:] = (0.45, 0.35, 0.3)
    tex[-res // 6 - 1] = (0.3, 0.22, 0.18)
    return np.clip(tex * rng.uniform(0.94, 1.03, (res, res, 1)), 0, 1)


def leaves(rng, res=128, count=110):
    """A clump of leaves on clear ground (alpha 0): ellipses in greens, their edges soft."""
    tex = np.zeros((res, res, 4))
    y, x = np.mgrid[0:res, 0:res] + 0.5
    for _ in range(count):
        cx, cy, r, turn = rng.uniform(6, res - 6), rng.uniform(6, res - 6), rng.uniform(4, 8), rng.uniform(0, np.pi)
        dx, dy = x - cx, y - cy
        along, across = dx * np.cos(turn) + dy * np.sin(turn), -dx * np.sin(turn) + dy * np.cos(turn)
        alpha = np.clip((1.0 - np.hypot(along / 1.7, across) / r) * r, 0.0, 1.0)
        green = (rng.uniform(0.15, 0.3), rng.uniform(0.4, 0.7), rng.uniform(0.08, 0.2))
        tex[..., :3] = np.where(alpha[..., None] > tex[..., 3:], green, tex[..., :3])
        tex[..., 3] = np.maximum(tex[..., 3], alpha)
    return tex


def lattice(res=128, bars=6, width=0.16):
    """Crossed wooden laths with square holes between them (alpha 0)."""
    tex = np.zeros((res, res, 4))
    tex[..., :3] = (0.62, 0.48, 0.32)
    u = (np.arange(res) + 0.5) / res * bars
    lath = np.abs(u - np.round(u)) < width
    frame = (np.arange(res) < res * 0.04) | (np.arange(res) >= res * 0.96)
    tex[..., 3] = (lath | frame)[None, :] | (lath | frame)[:, None]
    return tex


def stained_glass(rng, res=96, panes=14):
    """Panes of coloured glass (alpha 0.5) set in dark lead (solid), as a Voronoi pattern."""
    points = rng.uniform(0, res, (panes, 2))
    colours = np.array([(0.9, 0.15, 0.15), (0.15, 0.35, 0.95), (0.95, 0.8, 0.15), (0.2, 0.75, 0.3),
                        (0.7, 0.25, 0.85)])[rng.integers(0, 5, panes)]
    y, x = np.mgrid[0:res, 0:res] + 0.5
    d = np.sort(np.hypot(x[..., None] - points[:, 0], y[..., None] - points[:, 1]), axis=2)
    nearest = np.hypot(x[..., None] - points[:, 0], y[..., None] - points[:, 1]).argmin(axis=2)
    tex = np.concatenate([colours[nearest], np.full((res, res, 1), 0.5)], axis=2)
    lead = (d[..., 1] - d[..., 0] < 2.2) | (x < 3) | (x > res - 3) | (y < 3) | (y > res - 3)
    tex[lead] = (0.08, 0.07, 0.06, 1.0)
    return tex


def starry_sky_box(rng, res=128):
    """Six faces of a night sky: stars above a faint glow on the horizon, dark ground below."""
    def side():
        v = np.linspace(1.0, 0.0, res)[:, None, None]  # 1 at the top row
        tex = (0.02, 0.03, 0.08) + (1 - v) ** 3 * np.array([0.25, 0.15, 0.2])
        tex = np.broadcast_to(tex, (res, res, 3)).copy()
        stars(tex, top_only=True)
        return tex

    def stars(tex, top_only=False):
        n = res * res // 150
        ys, xs = rng.integers(0, res // 2 if top_only else res, n), rng.integers(0, res, n)
        tex[ys, xs] = rng.uniform(0.5, 1.0, (n, 1)) * (1.0, 0.95, 0.85)

    top = np.ones((res, res, 3)) * (0.02, 0.03, 0.08)
    stars(top)
    ground = np.ones((res, res, 3)) * 0.02
    return SkyBox([side(), side(), top, ground, side(), side()])


# ----- things to look at ------------------------------------------------------------------------

def inward_box(size, height, textures):
    """Walls and a floor of a courtyard, facing in (no ceiling): textures (walls, floor), each covering a whole side."""
    wall, floor = textures
    box = make_box(1.0, [wall, wall, wall, floor, wall, wall])
    verts = box.vertices * (2 * size, height, 2 * size) + (0, height / 2, 0)
    keep = np.repeat(np.arange(6) != 2, 2)  # no ceiling
    return Mesh(verts, box.faces[keep][:, ::-1].copy(), box.uvs[keep][:, ::-1].copy(), box.materials[keep],
                box.textures)


def picture(width, height, texture, turn=0.0):
    """A width x height rectangle standing upright on y = 0, turned `turn` about the vertical (facing +z at 0),
    showing all of `texture`."""
    c, s = np.cos(turn), np.sin(turn)
    right = np.array([c, 0.0, -s]) * width / 2
    corners = np.array([-right, right, right + (0, height, 0), -right + (0, height, 0)])
    uvs = np.array([[(0, 0), (1, 0), (1, 1)], [(0, 0), (1, 1), (0, 1)]], float)
    return Mesh(corners, np.array([(0, 1, 2), (0, 2, 3)]), uvs, np.zeros(2, int), [texture])


def rainbow_text(text):
    mesh, width = text_mesh(text, FONT)
    x = mesh.vertices[mesh.faces].mean(axis=1)[:, 0]
    hue = (x - x.min()) / max(np.ptp(x), 1e-9) * 0.85
    mesh.face_colors = np.array([colorsys.hsv_to_rgb(h, 0.8, 1.0) for h in hue])
    return mesh


def rainbow_blob():
    mesh = blob_mesh((1.0, 1.0, 1.0), rings=24, segments=32,
                     bump=lambda d: 1 + 0.12 * np.sin(5 * d[:, 0]) * np.sin(4 * d[:, 1]) * np.sin(5 * d[:, 2]))
    d = mesh.vertices / np.linalg.norm(mesh.vertices, axis=1, keepdims=True)
    hue = (np.arctan2(d[:, 2], d[:, 0]) / (2 * np.pi)) % 1.0
    mesh.vertex_colors = np.array([colorsys.hsv_to_rgb(h, 0.75, 1.0 - 0.3 * max(y, 0)) for h, y in zip(hue, d[:, 1])])
    return mesh


def turning(axis, period):
    """A RotationTrack turning once about `axis` every `period` seconds, for ever: keyframes a third of a turn apart
    (slerp takes the shortest way between two, so they must be less than half a turn apart)."""
    return RotationTrack([(period * k / 3, quat_axis_angle(axis, 2 * np.pi * k / 3)) for k in range(4)], loop="loop")


class Door:
    """A door on a hinge (a Node) that swings open when told the walker is near, and shut when not. Each swing is an
    Animation along a RotationTrack from wherever the door is: opening overshoots a little and settles
    (ease_out_back), as a door does against its stop; closing speeds up and slows down (ease_in_out)."""
    SHUT = quat_axis_angle(UP, 0.0)
    OPEN = quat_axis_angle(UP, 1.75)  # swung away from the start, about 100 degrees

    def __init__(self, hinge):
        self.hinge = hinge
        self.opening = False
        self.swing = None

    def update(self, dt, near):
        if near != self.opening:
            self.opening = near
            target, seconds, easing = (self.OPEN, 0.9, "ease_out_back") if near else (self.SHUT, 1.3, "ease_in_out")
            self.swing = Animation(self.hinge, rotation=RotationTrack([(0.0, self.hinge.rotation),
                                                                       (seconds, target, easing)]))
        if self.swing is not None and not self.swing.update(dt):
            self.swing = None


class Courtyard:
    def __init__(self, seed=None):
        rng = np.random.default_rng(seed)
        self.names = {}   # id(object) -> what the crosshair calls it
        self.objects = []
        self.obstacles = []  # (x, z, radius) the walker can't enter

        # Textures cover a whole wall or the floor, so their patterns repeat inside them: a stretch of
        # plaster four times along each wall, and metre-wide floor tiles.
        wall = np.tile(plaster(rng), (1, 4, 1))
        floor = tiles(256, (0.75, 0.72, 0.68), (0.4, 0.38, 0.36), int(2 * HALF))
        # Plaster and stone are matte (specular 0): no highlight sliding over them as you walk.
        self.add(Object3D(inward_box(HALF, WALL_HEIGHT, (wall, floor)), color=(255, 255, 255), specular=0.0),
                 "the courtyard")
        for x, z in ((-HALF + 0.6, -HALF + 0.6), (HALF - 0.6, -HALF + 0.6), (-HALF + 0.6, HALF - 0.6), (HALF - 0.6, HALF - 0.6)):
            self.add(Object3D(block_mesh((x, WALL_HEIGHT / 2 + 0.3, z), (0.8, WALL_HEIGHT + 0.6, 0.8)), color=(170, 160, 150),
                              specular=0.1), "a pillar", (x, z, 0.6))
        box = make_box()  # one cube, stretched (scale=(x, y, z)) into the table's parts, the bench, the door

        # Dice turning slowly on a pedestal.
        self.add(Object3D(block_mesh((-4.0, 0.5, -3.0), (1.2, 1.0, 1.2)), color=(90, 90, 110)), "a pedestal", (-4.0, -3.0, 0.9))
        # Small enough that, turning, they keep inside the glass case below and clear of each other.
        die = make_die(DIE_SIZE)
        self.dice = [Object3D(die, np.array([-4.0 + dx, 1.0 + DIE_SIZE / 2, -3.0 + dz]), orientation_showing(i, 0.0),
                              color=c, specular=1.4, shininess=40.0)
                     for i, (dx, dz, c) in enumerate(((-0.28, 0.0, Color.RED), (0.28, 0.2, Color.GREEN), (0.1, -0.3, Color.YELLOW)))]
        # They bob gently, each on a track rising and falling (ease in and out, back and forth), out of step.
        self.bobs = [Track([(0.0, d.position), (1.6, d.position + (0.0, 0.1, 0.0))], "ease_in_out", loop="pingpong")
                     for d in self.dice]
        for d in self.dice:
            self.add(d, "three dice")
        # In a glass case, which shows them through its near side and its far one, and casts a faint,
        # blue-tinted shadow.
        self.add(Object3D(block_mesh((-4.0, 1.45, -3.0), (1.1, 0.9, 1.1)), color=(200, 225, 255), opacity=0.15,
                          specular=1.5, shininess=60.0), "a glass case")

        # The orrery: a scene graph of nodes turning inside each other.
        self.orrery = Node(position=np.array([3.5, 1.4, -3.5]))
        self.planet_orbit = Node(parent=self.orrery)
        self.moon_orbit = Node(position=np.array([1.4, 0.0, 0.0]), parent=self.planet_orbit)
        self.add(Object3D(block_mesh((3.5, 0.35, -3.5), (0.9, 0.7, 0.9)), color=(120, 100, 80)), "the orrery's stand",
                 (3.5, -3.5, 1.0))
        self.add(Object3D(blob_mesh((0.3, 0.3, 0.3), rings=12, segments=16), color=(255, 190, 60), emissive=1.0,
                          parent=self.orrery), "the orrery's sun")
        self.add(Object3D(blob_mesh((0.16, 0.16, 0.16), rings=10, segments=14), color=Color.BLUE, parent=self.moon_orbit,
                          specular=1.2, shininess=30.0), "a planet")
        self.moon = Node(position=np.array([0.35, 0.0, 0.0]), parent=self.moon_orbit)
        self.add(Object3D(blob_mesh((0.06, 0.06, 0.06)), color=(200, 200, 200), parent=self.moon, specular=0.0), "its moon")
        # They turn on looping tracks: keyframes a third of a turn apart, blended at a steady rate (slerp).
        self.orbits = [Animation(self.planet_orbit, rotation=turning(UP, 12.6)),
                       Animation(self.moon_orbit, rotation=turning(UP, 3.7))]

        # A table, built as one node with its top and legs (the one cube, stretched), with stacked cubes of many
        # colours on it. Its top is lacquered: a strong, tight highlight.
        self.table = Node(position=np.array([-3.5, 0.0, 3.5]), rotation=quat_axis_angle(UP, 0.4))
        self.add(Object3D(box, np.array([0.0, 0.8, 0.0]), scale=(2.0, 0.1, 1.2), color=(140, 90, 50), specular=1.8,
                          shininess=45.0, parent=self.table), "a lacquered table", (-3.5, 3.5, 1.2))
        for lx in (-0.9, 0.9):
            for lz in (-0.5, 0.5):
                self.add(Object3D(box, np.array([lx, 0.4, lz]), scale=(0.08, 0.8, 0.08), color=(110, 70, 40),
                                  parent=self.table), "a lacquered table")
        cube = make_box(0.3)
        cube.face_colors = np.repeat([(230, 60, 60), (60, 200, 90), (70, 110, 240), (240, 210, 60), (220, 90, 220),
                                      (60, 210, 220)], 2, axis=0)
        for i, (x, y, z) in enumerate(((-0.5, 1.0, 0.0), (-0.1, 1.0, 0.1), (-0.3, 1.3, 0.05), (0.5, 1.0, -0.2))):
            self.add(Object3D(cube, np.array([x, y, z]), quat_axis_angle(UP, 0.7 * i), color=(255, 255, 255),
                              specular=1.3, shininess=30.0, parent=self.table), "coloured cubes (plastic)")

        # A slowly turning blob in rainbow vertex colours.
        self.blob = Object3D(rainbow_blob(), np.array([3.5, 1.3, 3.5]), scale=0.8, color=(255, 255, 255), specular=1.5,
                             shininess=35.0)
        self.add(Object3D(block_mesh((3.5, 0.2, 3.5), (1.0, 0.4, 1.0)), color=(80, 80, 80), specular=0.2), "a plinth",
                 (3.5, 3.5, 0.8))
        self.add(self.blob, "a rainbow blob")

        # A lamp by the west wall: a glowing bulb, and the light it gives.
        self.lamp_at = np.array([-6.5, 2.3, 0.0])
        self.add(Object3D(block_mesh((-6.5, 1.1, 0.0), (0.12, 2.2, 0.12)), color=(60, 60, 60)), "a lamp", (-6.5, 0.0, 0.4))
        self.bulb = Object3D(blob_mesh((0.18, 0.18, 0.18)), self.lamp_at, color=(255, 225, 160), emissive=1.0,
                             cast_shadows=False)  # it would shadow everything from the light inside it
        self.add(self.bulb, "a lamp")

        # A tree: a trunk and a crown of leaves, each clump a picture with holes (a cut-out texture), so
        # sunlight dapples the ground through it.
        tree = Node(position=np.array([5.5, 0.0, 0.2]))
        self.add(Object3D(block_mesh((0.0, 0.9, 0.0), (0.22, 1.8, 0.22)), color=(95, 70, 45), parent=tree,
                          specular=0.0), "a tree", (5.5, 0.2, 0.4))
        for i in range(3):
            self.add(Object3D(picture(2.4, 2.0, leaves(rng), i * np.pi / 3), np.array([0.0, 1.5, 0.0]),
                              color=(255, 255, 255), parent=tree), "a tree")
        crown = picture(2.4, 2.4, leaves(rng, count=140))  # and one lying flat on top
        crown.vertices = crown.vertices - (0, 1.2, 0)
        self.add(Object3D(crown, np.array([0.0, 2.9, 0.0]), quat_axis_angle((1, 0, 0), -np.pi / 2),
                          color=(255, 255, 255), parent=tree), "a tree")

        # A trellis between the lamp and the middle, whose lattice the lamp throws across the ground.
        self.add(Object3D(picture(2.6, 1.9, lattice(), np.pi / 2), np.array([-4.2, 0.0, 0.0]), color=(255, 255, 255)),
                 "a trellis")
        for z in (-1.0, 0.0, 1.0):
            self.obstacles.append((-4.2, z, 0.3))

        # A stained-glass panel in a frame, turned to face the sun (see Walk.frame): its colours fall on the
        # ground behind it.
        turn, at = -0.96, np.array([0.0, 0.0, -4.0])
        self.add(Object3D(picture(1.4, 1.8, stained_glass(rng), turn), at + (0, 0.25, 0), color=(255, 255, 255)),
                 "a stained-glass panel")
        for side in (-0.73, 0.73):
            post = at + side * np.array([np.cos(turn), 0.0, -np.sin(turn)])
            self.add(Object3D(block_mesh((post[0], 1.0, post[2]), (0.06, 2.0, 0.06)), color=(70, 55, 40)),
                     "a stained-glass panel", (post[0], post[2], 0.3))

        # A mirror on the south wall, in a frame: turn round at the start to see the courtyard behind you.
        mirror = Mesh(np.array([(3.0, 0.4, 0.0), (-3.0, 0.4, 0.0), (-3.0, 3.0, 0.0), (3.0, 3.0, 0.0)], float),
                      np.array([(0, 1, 2), (0, 2, 3)]))
        self.add(Object3D(mirror, np.array([0.0, 0.0, HALF - 0.08]), color=(180, 185, 190), reflectivity=0.9),
                 "a mirror")
        for x, y, w, h in ((0.0, 0.3, 6.4, 0.2), (0.0, 3.1, 6.4, 0.2), (-3.1, 1.7, 0.2, 2.6), (3.1, 1.7, 0.2, 2.6)):
            self.add(Object3D(block_mesh((x, y, HALF - 0.1), (w, h, 0.12)), color=(110, 80, 45)), "a mirror")

        # A still pool in a stone rim, which mirrors the sky and whatever stands round it.
        pool_at, pool = np.array([7.0, 0.0, 7.0]), 1.6
        water = Mesh(np.array([(-pool, 0.12, -pool), (-pool, 0.12, pool), (pool, 0.12, pool), (pool, 0.12, -pool)]),
                     np.array([(0, 1, 2), (0, 2, 3)]))
        self.add(Object3D(water, pool_at, color=(40, 70, 80), reflectivity=0.6), "a pool")
        for dx, dz, w, d in ((0, -pool - 0.1, 2 * pool + 0.4, 0.2), (0, pool + 0.1, 2 * pool + 0.4, 0.2),
                             (-pool - 0.1, 0, 0.2, 2 * pool), (pool + 0.1, 0, 0.2, 2 * pool)):
            self.add(Object3D(block_mesh((pool_at[0] + dx, 0.15, pool_at[2] + dz), (w, 0.3, d)), color=(150, 145, 135),
                              specular=0.1), "a pool")
        self.obstacles.append((*pool_at[[0, 2]], pool + 0.3))

        # A chrome ball on a plinth, reflecting the sky.
        self.add(Object3D(block_mesh((-7.0, 0.45, -7.0), (0.9, 0.9, 0.9)), color=(80, 80, 90)), "a plinth",
                 (-7.0, -7.0, 0.8))
        self.add(Object3D(blob_mesh((0.55, 0.55, 0.55), rings=24, segments=32), np.array([-7.0, 1.45, -7.0]),
                          color=(230, 230, 235), reflectivity=0.85, specular=2.5, shininess=150.0), "a chrome ball")

        # A bench (the cube, stretched) with four balls on it, one mesh in four materials: rubber (no highlight),
        # paint (a soft one), gold (a tight, strong one) and pearl (a broad sheen). Beside
        # it, an egg: a sphere stretched taller than it is wide.
        bench = Node(position=np.array([7.0, 0.0, -7.6]))
        self.add(Object3D(box, np.array([0.0, 0.55, 0.0]), scale=(2.8, 0.12, 0.7), color=(120, 85, 55), specular=0.3,
                          parent=bench), "a bench", (7.0, -7.6, 1.5))
        for x in (-1.25, 1.25):
            self.add(Object3D(box, np.array([x, 0.25, 0.0]), scale=(0.12, 0.5, 0.6), color=(90, 65, 40), parent=bench),
                     "a bench")
        ball = blob_mesh((0.25, 0.25, 0.25), rings=20, segments=28)
        for x, name, colour, specular, shininess, reflect in (
                (-0.975, "a rubber ball", (200, 60, 40), 0.0, None, 0.0),
                (-0.325, "a painted ball", (60, 110, 210), 1.0, 20.0, 0.0),
                (0.325, "a gold ball", (230, 165, 35), 2.5, 120.0, 0.0),
                (0.975, "a pearl", (240, 235, 225), 1.3, 6.0, 0.0)):
            self.add(Object3D(ball, np.array([x, 0.86, 0.0]), color=colour, specular=specular, shininess=shininess,
                              reflectivity=reflect, parent=bench), name)
        self.add(Object3D(box, np.array([8.9, 0.25, -7.6]), scale=0.5, color=(80, 80, 90), specular=0.2), "a plinth",
                 (8.9, -7.6, 0.5))
        self.add(Object3D(blob_mesh((1.0, 1.0, 1.0), rings=24, segments=32), np.array([8.9, 0.86, -7.6]),
                          scale=(0.26, 0.36, 0.26), color=(235, 225, 205), specular=0.8, shininess=15.0),
                 "an egg (a sphere, stretched)")

        # A door in a frame of its own in the middle of the courtyard, leading nowhere: it swings open as the
        # walker comes near and shuts behind them (see Door). Its handle is brass.
        hinge = Node(position=DOOR_AT.copy())
        for x, y, size in ((-0.07, 1.15, (0.12, 2.3, 0.16)), (1.17, 1.15, (0.12, 2.3, 0.16)),
                           (0.55, 2.36, (1.36, 0.12, 0.16))):
            self.add(Object3D(box, DOOR_AT + (x, y, 0.0), scale=size, color=(95, 70, 50), specular=0.3), "a door frame",
                     (DOOR_AT[0] + x, DOOR_AT[2], 0.12) if y < 2 else None)
        self.add(Object3D(box, np.array([0.55, 1.15, 0.0]), scale=(1.08, 2.28, 0.07), color=(150, 95, 55), specular=0.6,
                          shininess=20.0, parent=hinge), "a door (to nowhere)")
        for side in (-1, 1):
            self.add(Object3D(blob_mesh((0.05, 0.05, 0.05)), np.array([0.95, 1.05, side * 0.08]), color=(230, 180, 80),
                              specular=2.5, shininess=90.0, parent=hinge), "a brass handle")
        self.door = Door(hinge)

        # The sign on the north wall.
        sign = rainbow_text("UNICODE3D")
        self.add(Object3D(sign, np.array([0.0, 2.3, -HALF + 0.4]), scale=0.14, color=(255, 255, 255)), "the sign")

    def add(self, obj, name, obstacle=None):
        self.objects.append(obj)
        self.names[id(obj)] = name
        if obstacle is not None:
            self.obstacles.append(obstacle)

    def animate(self, t):
        """Everything that moves by itself, at time t seconds (the door moves with the walker: see Door)."""
        for i, (d, bob) in enumerate(zip(self.dice, self.bobs)):
            d.rotation = quat_mul(quat_axis_angle(UP, 0.4 * t + i), orientation_showing(i, 0.0))
            d.position = bob.at(t + 0.55 * i)
        for orbit in self.orbits:
            orbit.apply(t)
        self.blob.rotation = quat_axis_angle((0.3, 1.0, 0.1), 0.3 * t)

    def blocked(self, x, z):
        """Push a walker at (x, z) out of the walls and the things in the courtyard."""
        room = HALF - RADIUS
        x, z = min(max(x, -room), room), min(max(z, -room), room)
        for ox, oz, r in self.obstacles:
            dx, dz = x - ox, z - oz
            d = np.hypot(dx, dz)
            if d < r + RADIUS:
                if d < 1e-9:
                    dx, dz, d = 1.0, 0.0, 1.0
                x, z = ox + dx / d * (r + RADIUS), oz + dz / d * (r + RADIUS)
        return x, z


class Walk:
    def __init__(self, seed=None):
        self.court = Courtyard(seed)
        self.renderer = Renderer(1, 1)
        self.bar = StatusBar(self.renderer)
        self.lamp = Toggle("Lamp (l)", True, key="l")
        self.sun = Toggle("Sun (u)", True, key="u")
        self.backgrounds = {"sky": Sky(), "stars": starry_sky_box(np.random.default_rng(7)),
                            "gradient": Gradient((70, 90, 140), (20, 20, 30)), "none": None}
        self.background = Choice("Background (b)", tuple(self.backgrounds), key="b")
        self.fog = Slider("Fog", 0.3, 0.0, 0.9, step=0.05, keys="[]", length=8, fmt=lambda v: f"{v:.2f}")
        self.fog_into = Choice("fades into (f)", ("background", "mist"), key="f")
        self.panel = Panel([self.lamp, self.sun, self.background, self.fog, self.fog_into], keyboard=False)
        self.x, self.z, self.yaw, self.pitch = 0.0, 5.5, 0.0, 0.0
        self.t = 0.0
        self.clicked = ""

    def frame(self, screen, dt, events):
        dt = min(dt, 0.1)
        events = self.panel.handle(self.bar.handle(events, screen))
        rows, cols = screen.size()
        view_rows = max(rows - 2, 1)
        for ev in events:
            if ev == Key.ESC:
                return False
            if isinstance(ev, MouseEvent) and ev.pressed and ev.button == MouseEvent.LEFT and not ev.moved:
                hit = self.renderer.pick(ev.x, ev.y)
                self.clicked = f"clicked: {self.court.names.get(id(hit.object), '?')}" if hit else ""

        held = screen.held
        turn = ((Key.RIGHT in held or "e" in held) - (Key.LEFT in held or "q" in held)) * TURN * dt
        self.yaw += turn
        self.pitch = float(np.clip(self.pitch + ((Key.UP in held) - (Key.DOWN in held)) * LOOK * dt, -1.2, 1.2))
        ahead = ("w" in held) - ("s" in held)
        side = ("d" in held) - ("a" in held)
        fx, fz = np.sin(self.yaw), -np.cos(self.yaw)
        self.x, self.z = self.court.blocked(self.x + (fx * ahead - fz * side) * WALK * dt,
                                            self.z + (fz * ahead + fx * side) * WALK * dt)
        self.t += dt
        self.court.animate(self.t)
        near = np.hypot(self.x - DOOR_AT[0] - 0.55, self.z - DOOR_AT[2]) < DOOR_NEAR
        self.court.door.update(dt, near)

        eye = np.array([self.x, EYE, self.z])
        look = np.array([fx * np.cos(self.pitch), np.sin(self.pitch), fz * np.cos(self.pitch)])
        camera = Camera(position=eye, target=eye + look, fov=70.0, near=0.05, far=100.0)
        lights = [Light(direction=np.array([0.5, -1.0, -0.35]), ambient=0.3, diffuse=0.6, color=(255, 245, 230),
                        shadows=True)
                  if self.sun.value else Light(ambient=0.08, diffuse=0.0, specular=0.0)]
        objects = list(self.court.objects)
        self.court.bulb.visible = self.lamp.value
        if self.lamp.value:
            lights.append(PointLight(self.court.lamp_at, color=(255, 210, 150), diffuse=0.9, range=9.0, shadows=True))
        self.renderer.background = self.backgrounds[self.background.value]
        # Fog in the world, fading into the background (the sky's horizon, say) or into white mist: thicker further
        # up the slider.
        v = self.fog.value
        colour = MIST if self.fog_into.value == "mist" else None
        self.renderer.fog = Fog(start=2.0, end=6.0 + 60.0 * (1.0 - v) ** 2, color=colour) if v > 0 else 0.0

        self.renderer.resize(cols, view_rows, screen.cell_pixels)
        fb = self.renderer.render(objects, camera, lights)
        screen.erase()
        screen.draw_frame(fb)
        cy, cx = view_rows // 2, cols // 2
        screen.text(cy, cx, "+", Color.WHITE, bold=True)
        hit = self.renderer.pick(cx, cy)
        seeing = self.court.names.get(id(hit.object), "?") + f" ({hit.distance:.1f} m)" if hit else "the sky"
        self.panel.draw(screen, rows - 2, 1)
        keys = "exact" if held.exact else "estimated from repeats"
        self.bar.draw(screen, f"looking at {seeing}   {self.clicked}   [wasd] walk  [q/e/arrows] turn  [esc] quit   "
                              f"held keys: {keys}")
        screen.refresh()
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, help="random seed")
    parser.add_argument("--fps", type=int, default=30)
    add_display_args(parser)
    args = parser.parse_args()
    try:
        run(Walk(args.seed).frame, args.fps, mouse="drag", key_release=True, title="unicode3d room",
            **display_options(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
