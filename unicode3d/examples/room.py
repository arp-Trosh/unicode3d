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
looping tracks. You bump into things and slide along them: the walker is a
capsule pushed out of whatever it walks into (Colliders), the door too until it
has swung open. N puts names over the things nearby (labels anchored to points
in the scene, hidden behind whatever is in front of them).

In the north-east corner, a campfire: flames that are cut-out pictures turned to
face the camera (sprites from a sheet of wavering frames, each flickering through
them), sparks and smoke (Particles), a flickering light, and a string of lights
from a post by the fire to the corner pillar (a wire of Lines; the bulbs are
Particles that never move). Click anything to throw a puff of chips coloured as it is drawn there.
F9, or Debug on the panel, shows what the engine knows, drawn in Lines: a floor
grid, the world's axes, a box around each thing the walker bumps into, the
walker's capsule (look down, or in the mirror) and the ray query from the
walker's hand to what the crosshair is on; a pin floats over the last spot
clicked.

Before the mirror, the fighter walks a loop: a Sprite with eight sides, from a sheet
fighter.py bakes with the engine itself. The engine picks the side the camera
sees and the frame of the animation playing; this program decides the rest:
he faces the way he walks, stops to face the walker who comes near (a fighting
stance, a punch now and then), and, clicked, gathers a blast of light and
throws it, a ball flying on until a ray query says it meets something solid.

Walking feels best in a terminal that reports key releases (kitty, foot,
Ghostty, WezTerm, Alacritty, iTerm2, Windows Terminal): elsewhere a held key is
worked out from its repeats, and a tap walks for about half a second.
"""
import argparse
import colorsys

import numpy as np

from .dice import make_die, orientation_showing
from .fighter import fighter_sheet
from .hud import StatusBar
from ..animation import Animation, RotationTrack, Track
from ..background import Fog, Gradient, Sky, SkyBox
from ..color import Color, linear_to_srgb
from ..keys import Key, MouseEvent
from ..mesh import Mesh, make_box
from ..scene import Camera, Light, Lines, Node, Object3D, Particles, PointLight, Renderer, Sprite, SpriteSheet
from ..queries import Colliders
from ..shapes import blob_mesh, block_mesh, text_mesh
from ..terminal import add_display_args, display_options, run
from ..transforms import UP, quat_axis_angle, quat_mul, scene_poses
from ..ui import Choice, Panel, Slider, Toggle

HALF = 11.0          # the courtyard runs from -HALF to HALF in x and z
WALL_HEIGHT = 3.5
DIE_SIZE = 0.34      # of the dice in the glass case
EYE = 1.6
RADIUS = 0.35        # how close the walker gets to walls and things
KNEE = 0.45          # the height of the walker's lowest point, plus RADIUS: what they step over
WAIST = 1.0          # how high the debug view draws the walker's capsule
WALK, TURN, LOOK = 3.0, 1.8, 1.2  # units a second, radians a second
MIST = (225, 228, 232)  # the fog's colour when it is mist rather than the background
DOOR_AT = np.array([0.85, 0.0, 1.0])  # the door's hinge; it is 1.1 wide, along +x
DOOR_NEAR = 2.3      # how close the walker comes before the door opens
TAG_REACH = 10.0     # how near things are to have their names shown (the Names toggle)
TAG_LIFT = 0.15      # how far above the top of a thing its name is
TAG_GAP = 0.5        # objects of one name nearer each other than this are named once, as one thing
FIRE_AT = np.array([8.3, 0.0, -9.5])  # the campfire, in the north-east corner
SPARKS, PUFFS = 30.0, 5.0  # how many sparks and puffs of smoke the fire gives off a second
BREEZE = np.array([0.3, 0.0, 0.12])  # what carries them off, a little to the south-east
LOOP_AT, LOOP = np.array([0.0, 0.0, 8.2]), (2.6, 1.2)  # the fighter's walk: round an ellipse before the mirror
STRIDE = 1.3         # how fast he walks, units a second
HEED = 2.6           # how near the walker comes before he stops to face them

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


def flame(rng, res=64):
    """A tongue of flame on clear ground (a cut-out): round at the bottom, pointed and wavering at the top,
    white-yellow at its heart and red at its edges. Each call wavers differently: frames to swap between."""
    y, x = np.mgrid[0:res, 0:res] + 0.5
    v, u = 1.0 - y / res, x / res * 2.0 - 1.0  # (v 1 at the top; u -1..1 across)
    phase = rng.uniform(0, 2 * np.pi, 3)
    middle = v * (0.16 * np.sin(5.0 * v + phase[0]) + 0.07 * np.sin(13.0 * v + phase[1]))
    width = 0.8 * (1.0 - v) ** 0.9 * np.sqrt(np.clip(v / 0.22, 0.0, 1.0)) * (1.0 + 0.12 * np.sin(19.0 * v + phase[2]))
    r = np.abs(u - middle) / np.maximum(width, 1e-6)
    heat = np.clip((1.0 - r) * (1.15 - 0.8 * v), 0.0, 1.0)
    rgb = np.where((heat > 0.45)[..., None], (1.0, 0.93, 0.6),
                   np.where((heat > 0.2)[..., None], (1.0, 0.62, 0.12), (0.92, 0.25, 0.05)))
    return np.concatenate([rgb, (r < 1.0)[..., None]], axis=2).astype(float)


def pin(res=48):
    """A map pin on clear ground (a cut-out): a round yellow head with a dark rim, its point down."""
    y, x = np.mgrid[0:res, 0:res] + 0.5
    u, v = x / res - 0.5, 1.0 - y / res  # (v 1 at the top)
    head = np.hypot(u, v - 0.66) / 0.3
    point = (v < 0.66) & (np.abs(u) < 0.26 * (v / 0.66) ** 1.4)
    tex = np.zeros((res, res, 4))
    tex[..., :3] = np.where((head < 0.45)[..., None], (1.0, 1.0, 1.0), (1.0, 0.8, 0.1))
    tex[(head > 0.82) & (head < 1.0)] = (0.25, 0.15, 0.0, 1.0)
    tex[..., 3] = np.maximum(tex[..., 3], (head < 1.0) | point)
    return tex


def sagging(a, b, sag, step=0.15):
    """Points along a wire hung from a to b, sagging `sag` in the middle, about `step` apart."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    s = np.linspace(0.0, 1.0, max(int(np.linalg.norm(b - a) / step), 2) + 1)[:, None]
    return a + s * (b - a) - 4.0 * sag * s * (1.0 - s) * UP


CORNERS = np.array([(i & 1, i >> 1 & 1, i >> 2 & 1) for i in range(8)], float)  # (of a box from 0 to 1)
EDGES = np.array([(i, i | 1 << k) for i in range(8) for k in range(3) if not i & 1 << k])  # (12 pairs of corners)


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
        self.solid = Colliders()  # what the walker can't walk into (see blocked)
        self.groups = {}  # what the crosshair calls things -> the objects it calls that

        # Textures cover a whole wall or the floor, so their patterns repeat inside them: a stretch of
        # plaster four times along each wall, and metre-wide floor tiles.
        wall = np.tile(plaster(rng), (1, 4, 1))
        floor = tiles(256, (0.75, 0.72, 0.68), (0.4, 0.38, 0.36), int(2 * HALF))
        # Plaster and stone are matte (specular 0): no highlight sliding over them as you walk.
        self.ground = Object3D(inward_box(HALF, WALL_HEIGHT, (wall, floor)), color=(255, 255, 255), specular=0.0)
        self.add(self.ground, "the courtyard")
        for x, z in ((-HALF + 0.6, -HALF + 0.6), (HALF - 0.6, -HALF + 0.6), (-HALF + 0.6, HALF - 0.6), (HALF - 0.6, HALF - 0.6)):
            self.add(Object3D(block_mesh((x, WALL_HEIGHT / 2 + 0.3, z), (0.8, WALL_HEIGHT + 0.6, 0.8)), color=(170, 160, 150),
                              specular=0.1), "a pillar")
        box = make_box()  # one cube, stretched (scale=(x, y, z)) into the table's parts, the bench, the door

        # Dice turning slowly on a pedestal.
        self.add(Object3D(block_mesh((-4.0, 0.5, -3.0), (1.2, 1.0, 1.2)), color=(90, 90, 110)), "a pedestal")
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
        self.add(Object3D(block_mesh((3.5, 0.35, -3.5), (0.9, 0.7, 0.9)), color=(120, 100, 80)), "the orrery's stand")
        self.add(Object3D(blob_mesh((0.3, 0.3, 0.3), rings=12, segments=16), color=(255, 190, 60), emissive=1.0,
                          parent=self.orrery), "the orrery's sun")
        self.add(Object3D(blob_mesh((0.16, 0.16, 0.16), rings=10, segments=14), color=Color.BLUE, parent=self.moon_orbit,
                          specular=1.2, shininess=30.0), "a planet", solid=False)  # (it would sweep the walker aside)
        self.moon = Node(position=np.array([0.35, 0.0, 0.0]), parent=self.moon_orbit)
        self.add(Object3D(blob_mesh((0.06, 0.06, 0.06)), color=(200, 200, 200), parent=self.moon, specular=0.0), "its moon",
                 solid=False)
        # They turn on looping tracks: keyframes a third of a turn apart, blended at a steady rate (slerp).
        self.orbits = [Animation(self.planet_orbit, rotation=turning(UP, 12.6)),
                       Animation(self.moon_orbit, rotation=turning(UP, 3.7))]

        # A table, built as one node with its top and legs (the one cube, stretched), with stacked cubes of many
        # colours on it. Its top is lacquered: a strong, tight highlight.
        self.table = Node(position=np.array([-3.5, 0.0, 3.5]), rotation=quat_axis_angle(UP, 0.4))
        self.add(Object3D(box, np.array([0.0, 0.8, 0.0]), scale=(2.0, 0.1, 1.2), color=(140, 90, 50), specular=1.8,
                          shininess=45.0, parent=self.table), "a lacquered table")
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
        self.add(Object3D(block_mesh((3.5, 0.2, 3.5), (1.0, 0.4, 1.0)), color=(80, 80, 80), specular=0.2), "a plinth")
        self.add(self.blob, "a rainbow blob")

        # A lamp by the west wall: a glowing bulb, and the light it gives.
        self.lamp_at = np.array([-6.5, 2.3, 0.0])
        self.add(Object3D(block_mesh((-6.5, 1.1, 0.0), (0.12, 2.2, 0.12)), color=(60, 60, 60)), "a lamp")
        self.bulb = Object3D(blob_mesh((0.18, 0.18, 0.18)), self.lamp_at, color=(255, 225, 160), emissive=1.0,
                             cast_shadows=False)  # it would shadow everything from the light inside it
        self.add(self.bulb, "a lamp")

        # A tree: a trunk and a crown of leaves, each clump a picture with holes (a cut-out texture), so
        # sunlight dapples the ground through it.
        tree = Node(position=np.array([5.5, 0.0, 0.2]))
        self.add(Object3D(block_mesh((0.0, 0.9, 0.0), (0.22, 1.8, 0.22)), color=(95, 70, 45), parent=tree,
                          specular=0.0), "a tree")
        for i in range(3):
            self.add(Object3D(picture(2.4, 2.0, leaves(rng), i * np.pi / 3), np.array([0.0, 1.5, 0.0]),
                              color=(255, 255, 255), parent=tree), "a tree", solid=False)  # (leaves: walk through)
        crown = picture(2.4, 2.4, leaves(rng, count=140))  # and one lying flat on top
        crown.vertices = crown.vertices - (0, 1.2, 0)
        self.add(Object3D(crown, np.array([0.0, 2.9, 0.0]), quat_axis_angle((1, 0, 0), -np.pi / 2),
                          color=(255, 255, 255), parent=tree), "a tree", solid=False)

        # A trellis between the lamp and the middle, whose lattice the lamp throws across the ground.
        self.add(Object3D(picture(2.6, 1.9, lattice(), np.pi / 2), np.array([-4.2, 0.0, 0.0]), color=(255, 255, 255)),
                 "a trellis")

        # A stained-glass panel in a frame, turned to face the sun (see Walk.frame): its colours fall on the
        # ground behind it.
        turn, at = -0.96, np.array([0.0, 0.0, -4.0])
        self.add(Object3D(picture(1.4, 1.8, stained_glass(rng), turn), at + (0, 0.25, 0), color=(255, 255, 255)),
                 "a stained-glass panel")
        for side in (-0.73, 0.73):
            post = at + side * np.array([np.cos(turn), 0.0, -np.sin(turn)])
            self.add(Object3D(block_mesh((post[0], 1.0, post[2]), (0.06, 2.0, 0.06)), color=(70, 55, 40)),
                     "a stained-glass panel")

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
        # The rim is low enough to step over, so a box the walker can't walk into stands in for the pool: in the
        # Colliders, not drawn.
        self.solid.objects.append(Object3D(block_mesh(pool_at + (0.0, 0.5, 0.0), (2 * pool + 0.4, 1.0, 2 * pool + 0.4))))
        for dx, dz, w, d in ((0, -pool - 0.1, 2 * pool + 0.4, 0.2), (0, pool + 0.1, 2 * pool + 0.4, 0.2),
                             (-pool - 0.1, 0, 0.2, 2 * pool), (pool + 0.1, 0, 0.2, 2 * pool)):
            self.add(Object3D(block_mesh((pool_at[0] + dx, 0.15, pool_at[2] + dz), (w, 0.3, d)), color=(150, 145, 135),
                              specular=0.1), "a pool")

        # A chrome ball on a plinth, reflecting the sky.
        self.add(Object3D(block_mesh((-7.0, 0.45, -7.0), (0.9, 0.9, 0.9)), color=(80, 80, 90)), "a plinth")
        self.add(Object3D(blob_mesh((0.55, 0.55, 0.55), rings=24, segments=32), np.array([-7.0, 1.45, -7.0]),
                          color=(230, 230, 235), reflectivity=0.85, specular=2.5, shininess=150.0), "a chrome ball")

        # A bench (the cube, stretched) with four balls on it, one mesh in four materials: rubber (no highlight),
        # paint (a soft one), gold (a tight, strong one) and pearl (a broad sheen). Beside
        # it, an egg: a sphere stretched taller than it is wide.
        bench = Node(position=np.array([7.0, 0.0, -7.6]))
        self.add(Object3D(box, np.array([0.0, 0.55, 0.0]), scale=(2.8, 0.12, 0.7), color=(120, 85, 55), specular=0.3,
                          parent=bench), "a bench")
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
        self.add(Object3D(box, np.array([8.9, 0.25, -7.6]), scale=0.5, color=(80, 80, 90), specular=0.2), "a plinth")
        self.add(Object3D(blob_mesh((1.0, 1.0, 1.0), rings=24, segments=32), np.array([8.9, 0.86, -7.6]),
                          scale=(0.26, 0.36, 0.26), color=(235, 225, 205), specular=0.8, shininess=15.0),
                 "an egg (a sphere, stretched)")

        # A door in a frame of its own in the middle of the courtyard, leading nowhere: it swings open as the
        # walker comes near and shuts behind them (see Door). Its handle is brass.
        hinge = Node(position=DOOR_AT.copy())
        for x, y, size in ((-0.07, 1.15, (0.12, 2.3, 0.16)), (1.17, 1.15, (0.12, 2.3, 0.16)),
                           (0.55, 2.36, (1.36, 0.12, 0.16))):
            self.add(Object3D(box, DOOR_AT + (x, y, 0.0), scale=size, color=(95, 70, 50), specular=0.3), "a door frame")
        self.add(Object3D(box, np.array([0.55, 1.15, 0.0]), scale=(1.08, 2.28, 0.07), color=(150, 95, 55), specular=0.6,
                          shininess=20.0, parent=hinge), "a door (to nowhere)")
        for side in (-1, 1):
            self.add(Object3D(blob_mesh((0.05, 0.05, 0.05)), np.array([0.95, 1.05, side * 0.08]), color=(230, 180, 80),
                              specular=2.5, shininess=90.0, parent=hinge), "a brass handle")
        self.door = Door(hinge)

        # A campfire in the north-east corner, in a ring of stones: crossed logs on glowing embers, flames (cut-out
        # pictures that face the camera: sprites, each flickering through a sheet of wavering frames: see
        # animate), sparks and smoke (Particles: see burn), and a flickering light (fire_light).
        stone = blob_mesh((0.16, 0.1, 0.13), rings=6, segments=8)
        for k in range(9):
            a = 2 * np.pi * k / 9
            grey = int(rng.integers(110, 150))
            self.add(Object3D(stone, FIRE_AT + (0.55 * np.cos(a), 0.08, 0.55 * np.sin(a)),
                              quat_axis_angle(UP, rng.uniform(0, np.pi)), color=(grey, grey - 4, grey - 8),
                              specular=0.1), "a campfire")
        for k in range(3):
            self.add(Object3D(box, FIRE_AT + (0.0, 0.12, 0.0), quat_axis_angle(UP, 0.3 + k * np.pi / 3),
                              scale=(0.85, 0.1, 0.1), color=(75, 50, 32), specular=0.0), "a campfire")
        self.add(Object3D(blob_mesh((0.32, 0.04, 0.32)), FIRE_AT + (0.0, 0.03, 0.0), color=(255, 110, 40), emissive=1.0,
                          cast_shadows=False), "a campfire")
        # (six frames side by side: a sprite sheet of one direction, seen the same from everywhere)
        frames = SpriteSheet(np.concatenate([flame(rng) for _ in range(6)], axis=1), columns=6, directions=1,
                             animations={"flicker": (0, 6, 9.0)})
        self.flames = [Sprite(frames, position=FIRE_AT + (dx, 0.06, dz), scale=(w, h, 1.0), emissive=1.0,
                              billboard=True, outline=False, cast_shadows=False)
                       for w, h, dx, dz in ((0.7, 1.05, 0.0, 0.0), (0.45, 0.7, 0.14, 0.1), (0.4, 0.6, -0.15, -0.07))]
        for card in self.flames:
            self.add(card, "a campfire", solid=False)
        self.sparks = Particles(gravity=BREEZE + (0.0, 0.9, 0.0), drag=0.8, emissive=1.0, limit=200, seed=1)
        self.smoke = Particles(face_camera=True, gravity=BREEZE + (0.0, 0.1, 0.0), drag=0.4, limit=100, seed=2)
        self.add_effect(self.sparks, "sparks")
        self.add_effect(self.smoke, "smoke")
        self.puffing = [0.0, 0.0]  # sparks and puffs owed (see burn)

        # A string of lights from a hook on the north wall, over a post by the fire, to the corner pillar and on to a
        # hook on the east wall: a thin wire (Lines, in world units, so thinner further off) with glowing bulbs
        # hanging from it (Particles that are never moved, so stay where they are put).
        post = (6.6, -9.9)
        self.add(Object3D(block_mesh((post[0], 1.35, post[1]), (0.12, 2.7, 0.12)), color=(95, 70, 45), specular=0.0),
                 "a post")
        stops = [(4.6, 3.2, -HALF + 0.04), (post[0], 2.72, post[1]), (HALF - 1.0, 3.9, -HALF + 1.0),
                 (HALF - 0.04, 3.2, -6.4)]
        for hook in (stops[0], stops[-1]):
            self.add(Object3D(block_mesh(hook, (0.08, 0.08, 0.08)), color=(40, 40, 40)), "a string of lights",
                     solid=False)
        self.bulbs = Particles(emissive=1.0, limit=100, seed=3)
        wire = []
        for a, b, sag in zip(stops, stops[1:], (0.3, 0.45, 0.5)):
            points = sagging(a, b, sag)
            wire += list(zip(points[:-1], points[1:]))
            for i, at in enumerate(points[2:-2:3]):
                self.bulbs.emit(at - (0.0, 0.06, 0.0), life=1e9, size=0.09,
                                color=((255, 220, 140), (255, 80, 60), (90, 220, 120), (90, 140, 255))[i % 4])
        self.add_effect(Lines([s for s, e in wire], [e for s, e in wire], width=0.012, units="world",
                              color=(40, 35, 30)), "a string of lights")
        self.add_effect(self.bulbs, "a string of lights")

        # The sign on the north wall.
        sign = rainbow_text("UNICODE3D")
        self.add(Object3D(sign, np.array([0.0, 2.3, -HALF + 0.4]), scale=0.14, color=(255, 255, 255)), "the sign")

    def add(self, obj, name, solid=True):
        self.objects.append(obj)
        self.names[id(obj)] = name
        self.groups.setdefault(name, []).append(obj)
        if solid:
            self.solid.objects.append(obj)

    def add_effect(self, obj, name):
        """Particles or Lines: drawn and named, but neither solid nor labelled (they have no box to put a name on)."""
        self.objects.append(obj)
        self.names[id(obj)] = name

    def burn(self, dt):
        """The fire's sparks and smoke, dt seconds on: new ones given off, the others moved on."""
        self.puffing[0] += SPARKS * dt
        self.puffing[1] += PUFFS * dt
        sparks, puffs = int(self.puffing[0]), int(self.puffing[1])
        self.puffing = [self.puffing[0] - sparks, self.puffing[1] - puffs]
        # Sparks fly up out of the flames, yellow, dimming to red as they go; smoke rises above them, growing
        # and thinning, and drifts off on the breeze.
        self.sparks.emit(FIRE_AT + (0.0, 0.35, 0.0), sparks, velocity=(0.0, 1.0, 0.0), speed=(0.3, 1.2), radius=0.2,
                         life=(0.6, 1.6), size=(0.03, 0.05), size_end=0.01, color=(255, 230, 120),
                         color_end=(220, 40, 10))
        self.smoke.emit(FIRE_AT + (0.0, 0.95, 0.0), puffs, velocity=(0.0, 0.45, 0.0), speed=0.1, radius=0.12,
                        life=(3.0, 4.5), size=(0.25, 0.35), size_end=(1.0, 1.4), color=(120, 116, 112),
                        color_end=(190, 190, 195), opacity=0.4, opacity_end=0.0, spin=(-0.4, 0.4))
        self.sparks.update(dt)
        self.smoke.update(dt)

    def fire_light(self, t):
        """The fire's light at time t: flickering, as sines out of step with each other."""
        flicker = 0.75 + 0.12 * np.sin(11.0 * t) + 0.08 * np.sin(23.7 * t + 1.0) + 0.05 * np.sin(37.1 * t + 2.0)
        return PointLight(FIRE_AT + (0.0, 0.6 + 0.03 * np.sin(17.0 * t), 0.0), color=(255, 150, 70),
                          diffuse=1.1 * flicker, range=7.0)

    def animate(self, t):
        """Everything that moves by itself, at time t seconds (the door moves with the walker: see Door; the
        fire's sparks and smoke with each frame's time: see burn)."""
        for i, card in enumerate(self.flames):  # (each flickering about ten times a second, out of step)
            card.time = (t * (9.0 + 2.0 * i) + 2 * i) / 9.0
        for i, (d, bob) in enumerate(zip(self.dice, self.bobs)):
            d.rotation = quat_mul(quat_axis_angle(UP, 0.4 * t + i), orientation_showing(i, 0.0))
            d.position = bob.at(t + 0.55 * i)
        for orbit in self.orbits:
            orbit.apply(t)
        self.blob.rotation = quat_axis_angle((0.3, 1.0, 0.1), 0.3 * t)

    def name_tags(self, eye, reach):
        """(name, a point just above the thing, its objects) for each thing within reach of eye, to label: over the
        middle of the top of the box around its objects as they stand now. Objects of one name make one thing where
        their boxes are within TAG_GAP of each other (a table and its legs, three dice), and one each otherwise
        (the four pillars)."""
        tags = []
        for name, objects in self.groups.items():
            if name in ("the courtyard", "a mirror"):
                continue
            things = []  # [[low, high, objects]], merged as they come near each other
            for obj in objects:
                box = obj.world_bounds() if obj.world_matrix()[2] else None
                if box is None:
                    continue
                lo, hi, members = box[0], box[1], [obj]
                near = [(t[0] - TAG_GAP <= hi).all() and (lo <= t[1] + TAG_GAP).all() for t in things]
                for thing in [t for t, n in zip(things, near) if n]:
                    lo, hi, members = np.minimum(lo, thing[0]), np.maximum(hi, thing[1]), thing[2] + members
                # (kept by position: things.remove() would compare their arrays)
                things = [t for t, n in zip(things, near) if not n] + [[lo, hi, members]]
            for lo, hi, members in things:
                top = np.array([(lo[0] + hi[0]) / 2, hi[1] + TAG_LIFT, (lo[2] + hi[2]) / 2])
                if np.linalg.norm(top - eye) < reach:
                    tags.append((name, top, members))
        return tags

    def blocked(self, x, z):
        """Push a walker at (x, z) out of the walls and the things in the courtyard: a capsule from knee to eye
        height (clear of the floor), moved out of whatever it cuts into, along the ground. Things that move by
        themselves, such as the door, push the walker too."""
        self.solid.update()
        push = self.solid.push_out((x, KNEE, z), RADIUS, end=(x, EYE, z))
        room = HALF - RADIUS  # (and never outside the walls, whatever happens)
        return min(max(x + push[0], -room), room), min(max(z + push[2], -room), room)


class Fighter:
    """The fighter, a sprite with eight sides (fighter.py bakes his sheet), and what the game decides for him: he walks
    round a loop facing the way he goes, stops to face the walker when they come near (in a fighting stance,
    throwing a punch now and then), and, clicked, gathers a blast of light at his hip and throws it, a glowing
    ball flying on until it meets something solid (a ray query), where it bursts."""

    def __init__(self, court):
        self.sprite = Sprite(fighter_sheet(), animation="walk", position=LOOP_AT + (LOOP[0], 0.0, 0.0))
        # A box the walker bumps into, moved with him (in the Colliders only: not drawn).
        self.blocker = Object3D(block_mesh((0.0, 1.0, 0.0), (0.6, 1.8, 0.6)))
        court.add(self.sprite, "the fighter", solid=False)
        court.solid.objects.append(self.blocker)
        self.glow = Particles(face_camera=True, emissive=1.0, limit=400)  # (gathering at his hands, and bursting)
        self.ball = Object3D(blob_mesh((0.2, 0.2, 0.2), rings=6, segments=10), color=(170, 225, 255),
                             emissive=1.0, cast_shadows=False, outline=False, visible=False)
        court.add_effect(self.glow, "a blast of light")
        court.add(self.ball, "a blast of light", solid=False)
        self.solid = court.solid
        self.angle = 0.0      # how far round the loop he is
        self.heading = 0.0    # the way he faces: turned from +z towards +x
        self.state, self.clock = "walk", 0.0  # what he is doing, and for how long
        self.flight = None    # the ball's (start, direction, how far it goes, time out)

    def provoke(self):
        """Clicked: gather a blast (unless already at it)."""
        if self.state not in ("charge", "blast"):
            self.state, self.clock = "charge", 0.0
            self.sprite.play("charge")

    def update(self, dt, walker):
        """dt seconds on, with the walker at walker (x, z)."""
        s, at = self.sprite, self.sprite.position
        to = np.array([walker[0] - at[0], 0.0, walker[1] - at[2]])
        near = np.hypot(to[0], to[2]) < HEED
        self.clock += dt
        if self.state == "walk":
            if near:
                self.state, self.clock = "watch", 0.0
                s.play("idle")
            else:  # (along the ellipse, turning to face the way he goes)
                rx, rz = LOOP
                self.angle += STRIDE * dt / np.hypot(rx * np.sin(self.angle), rz * np.cos(self.angle))
                at = LOOP_AT + (rx * np.cos(self.angle), 0.0, rz * np.sin(self.angle))
                face = np.arctan2(-rx * np.sin(self.angle), rz * np.cos(self.angle))
                self.turn_to(face, dt)
                s.position = at
        elif self.state == "watch":  # (facing the walker; a punch every few seconds)
            self.turn_to(np.arctan2(to[0], to[2]), dt)
            if s.animation == "punch" and s.finished:
                s.play("idle")
            if self.clock > 2.5:
                self.clock = 0.0
                s.play("punch", loop=False, restart=True)
            if not near and s.animation == "idle":
                self.state = "walk"
                s.play("walk")
        elif self.state == "charge":  # (turning on the walker, light gathering at his hip, faster as it goes)
            self.turn_to(np.arctan2(to[0], to[2]), dt)
            hand = self.hand(0.15, 0.35, 0.95)  # (cupped at his right hip)
            self.glow.emit(hand, int(60 * dt * (1 + 3 * self.clock)) + 1, speed=(0.1, 0.4), radius=0.25,
                           life=(0.2, 0.4), size=(0.05, 0.1), size_end=0.02, color=(220, 245, 255),
                           color_end=(60, 140, 255), opacity=0.9)
            if self.clock > 1.4:
                self.state, self.clock = "blast", 0.0
                s.play("blast")
                ahead = np.array([np.sin(self.heading), 0.0, np.cos(self.heading)])
                start = self.hand(0.6, 0.0, 1.25)
                hit = self.solid.raycast(start, ahead, ignore=(self.blocker,))
                self.flight = (start, ahead, hit.distance if hit else 20.0, 0.0)
                self.ball.visible = True
        elif self.state == "blast":
            if self.clock > 0.9:
                self.state, self.clock = "watch" if near else "walk", 0.0
                s.play("idle" if near else "walk")
        self.fly(dt)
        s.rotation = quat_axis_angle(UP, self.heading)
        s.update(dt)
        self.glow.update(dt)
        self.blocker.position = s.position.copy()

    def turn_to(self, heading, dt, rate=7.0):
        """Turn towards heading, the short way round, at most rate radians a second."""
        gap = (heading - self.heading + np.pi) % (2 * np.pi) - np.pi
        self.heading += float(np.clip(gap, -rate * dt, rate * dt))

    def hand(self, ahead, right=0.0, up=1.05):
        """A point `up` above the floor, `ahead` in front of him and `right` to his right."""
        c, s = np.cos(self.heading), np.sin(self.heading)
        return self.sprite.position + (0.0, up, 0.0) + ahead * np.array([s, 0.0, c]) + right * np.array([-c, 0.0, s])

    def fly(self, dt):
        """The ball on its way: on along its line, bursting where its ray met something."""
        if self.flight is None:
            return
        start, ahead, reach, gone = self.flight
        gone += 6.0 * dt
        if gone >= reach:
            self.glow.emit(start + ahead * reach, 80, speed=(1.0, 3.0), life=(0.3, 0.7), size=(0.06, 0.14),
                           size_end=0.0, color=(230, 250, 255), color_end=(40, 110, 255))
            self.ball.visible, self.flight = False, None
            return
        self.ball.position = start + ahead * gone
        self.glow.emit(self.ball.position, 3, radius=0.1, life=(0.2, 0.35), size=(0.1, 0.16), size_end=0.0,
                       color=(200, 235, 255), color_end=(40, 110, 255), opacity=0.8)
        self.flight = (start, ahead, reach, gone)

    def light(self):
        """The glow's light: at the ball while it flies, at his hands while he gathers it, or None."""
        if self.flight is not None:
            return PointLight(self.ball.position, color=(120, 190, 255), diffuse=1.2, range=5.0)
        if self.state == "charge":
            return PointLight(self.hand(0.15, 0.35, 0.95), color=(120, 190, 255), diffuse=min(self.clock, 1.0),
                              range=4.0)
        return None


class DebugView:
    """What the engine knows, drawn over the scene in lines (F9): a floor grid a metre square, the world's axes
    (x red, y green, z blue) at the centre, a box around each thing the walker bumps into (the Colliders, the box
    standing in for the pool included), the walker's capsule up to the waist (higher, it would cross the picture
    from inside: look down, or in the mirror), and the ray query from the walker's hand to what the crosshair is on,
    with the surface's normal there."""
    def __init__(self):
        r = np.arange(-HALF, HALF + 0.5)
        n = len(r)
        across = np.c_[r, np.zeros(n), np.full(n, -HALF)], np.c_[r, np.zeros(n), np.full(n, HALF)]
        along = np.c_[np.full(n, -HALF), np.zeros(n), r], np.c_[np.full(n, HALF), np.zeros(n), r]
        self.grid = Lines(np.r_[across[0], along[0]], np.r_[across[1], along[1]], color=(110, 170, 210),
                          position=(0.0, 0.01, 0.0))
        self.axes = Lines(np.zeros((3, 3)), np.eye(3) * 1.5, colors=[(255, 60, 60), (60, 255, 60), (70, 110, 255)],
                          width=3, position=(0.0, 0.02, 0.0))
        self.boxes = Lines(color=(255, 170, 40))
        self.capsule = Lines(color=(120, 255, 120))
        self.ray = Lines(colors=np.zeros((0, 3)), width=2)
        self.lines = [self.grid, self.axes, self.boxes, self.capsule, self.ray]
        self.hit = None
        self.corners = {}  # id(mesh) -> the corners of the box around it, in its own space (8, 3)

    def show(self, on):
        for lines in self.lines:
            lines.visible = on

    def update(self, court, eye, look):
        """Follow the walker (at eye, looking along look) and whatever moved. court.solid has been updated."""
        # Each solid thing's box, turning with it: the box around its mesh, placed as the thing is (all at once).
        solid = [obj for obj in court.solid.objects if obj is not court.ground]
        for obj in solid:
            if id(obj.mesh) not in self.corners:
                lo, hi = obj.mesh.vertices.min(axis=0), obj.mesh.vertices.max(axis=0)
                self.corners[id(obj.mesh)] = lo + CORNERS * (hi - lo)
        rows, linear, position, _ = scene_poses(solid)
        at = [rows[id(obj)] for obj in solid]
        corners = np.array([self.corners[id(obj.mesh)] for obj in solid])  # (n, 8, 3)
        corners = np.einsum("nij,nkj->nki", linear[at], corners) + position[at][:, None]
        self.boxes.set(corners[:, EDGES[:, 0]].reshape(-1, 3), corners[:, EDGES[:, 1]].reshape(-1, 3))
        # The capsule (blocked) up to the waist: rings round it at knee and waist height, sides between them, and
        # half rings under its end.
        a = np.linspace(0.0, 2.0 * np.pi, 25)
        ring = np.c_[np.cos(a), np.zeros(25), np.sin(a)] * RADIUS
        half = np.linspace(0.0, np.pi, 13)
        arcs = [np.c_[np.cos(half) * np.cos(turn), -np.sin(half), np.cos(half) * np.sin(turn)] * RADIUS + (0, KNEE, 0)
                for turn in (0.0, np.pi / 2)]
        rings = [ring + (0.0, KNEE, 0.0), ring + (0.0, WAIST, 0.0)]
        sides = [np.array([p + (0.0, KNEE, 0.0), p + (0.0, WAIST, 0.0)]) for p in ring[:24:3]]
        feet = np.array([eye[0], 0.0, eye[2]])
        self.capsule.set(np.concatenate([c[:-1] for c in rings + arcs + sides]) + feet,
                         np.concatenate([c[1:] for c in rings + arcs + sides]) + feet)
        # The ray: from the walker's right hand to the nearest hit, and the surface's normal there.
        self.hit = court.solid.raycast(eye, look, max_distance=60.0)
        if self.hit is None:
            self.ray.set(np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)))
            return
        right = np.cross(look, UP)
        right /= max(np.linalg.norm(right), 1e-9)
        hand, at, normal = eye + 0.25 * right - (0.0, 0.3, 0.0) + 0.3 * look, self.hit.position, self.hit.normal
        self.ray.set([hand, at, at - (0.1, 0, 0), at - (0, 0.1, 0), at - (0, 0, 0.1)],
                     [at, at + 0.5 * normal, at + (0.1, 0, 0), at + (0, 0.1, 0), at + (0, 0, 0.1)],
                     [(255, 60, 60), (60, 255, 60), (255, 255, 255), (255, 255, 255), (255, 255, 255)])


class Walk:
    mouse = "drag"  # clicks and drags (run() and the playtest read it)
    key_release = True

    def __init__(self, seed=None):
        self.court = Courtyard(seed)
        self.renderer = Renderer()
        self.bar = StatusBar(self.renderer)
        self.lamp = Toggle("Lamp (l)", True, key="l")
        self.sun = Toggle("Sun (u)", True, key="u")
        self.backgrounds = {"sky": Sky(), "stars": starry_sky_box(np.random.default_rng(7)),
                            "gradient": Gradient((70, 90, 140), (20, 20, 30)), "none": None}
        self.background = Choice("Background (b)", tuple(self.backgrounds), key="b")
        self.fog = Slider("Fog", 0.3, 0.0, 0.9, step=0.05, keys="[]", length=8, fmt=lambda v: f"{v:.2f}")
        self.fog_into = Choice("fades into (f)", ("background", "mist"), key="f")
        self.tags = Toggle("Names (n)", False, key="n")
        self.debug = Toggle("Debug (F9)", False, key=Key.F9)
        self.panel = Panel([self.lamp, self.sun, self.background, self.fog, self.fog_into, self.tags, self.debug],
                           keyboard=False)
        self.view = DebugView()
        self.fighter = Fighter(self.court)
        # Clicking throws a puff of chips from the spot, coloured as it is drawn there, which bounce on the floor;
        # with the debug view on, a pin (a billboard) floats over the last spot clicked.
        self.chips = Particles(make_box(), gravity=(0.0, -9.8, 0.0), ground=0.0, bounce=0.35, emissive=0.5, limit=300)
        self.court.names[id(self.chips)] = "chips"
        self.pin = Object3D(picture(0.3, 0.42, pin()), color=(255, 255, 255), emissive=0.6, billboard=True,
                            cast_shadows=False, visible=False)
        self.stem = Lines(color=(255, 210, 40), visible=False)
        self.court.names[id(self.pin)] = self.court.names[id(self.stem)] = "the last spot clicked"
        self.pinned = None  # where the pin is over
        self.x, self.z, self.yaw, self.pitch = 0.0, 5.5, 0.0, 0.0
        self.t = 0.0
        self.clicked = ""
        self.drawn = []  # what the last frame drew

    def frame(self, screen, dt, events):
        events = self.panel.handle(self.bar.handle(events, screen))
        rows, cols = screen.size()
        view_rows = max(rows - 2, 1)
        for ev in events:
            if ev == Key.ESC:
                return False
            if isinstance(ev, MouseEvent) and ev.pressed and ev.button == MouseEvent.LEFT and not ev.moved:
                hit = self.pick(ev.x, ev.y)
                self.clicked = f"clicked: {self.court.names.get(id(hit.object), '?')}" if hit else ""
                if hit:
                    self.throw_chips(hit, ev.x, ev.y)
                    if hit.object is self.fighter.sprite:
                        self.fighter.provoke()

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
        self.court.burn(dt)
        self.chips.update(dt)
        self.fighter.update(dt, (self.x, self.z))
        near = np.hypot(self.x - DOOR_AT[0] - 0.55, self.z - DOOR_AT[2]) < DOOR_NEAR
        self.court.door.update(dt, near)

        eye = np.array([self.x, EYE, self.z])
        look = np.array([fx * np.cos(self.pitch), np.sin(self.pitch), fz * np.cos(self.pitch)])
        camera = Camera(position=eye, target=eye + look, fov=70.0, near=0.05, far=100.0)
        lights = [Light(direction=np.array([0.5, -1.0, -0.35]), ambient=0.3, diffuse=0.6, color=(255, 245, 230),
                        shadows=True)
                  if self.sun.value else Light(ambient=0.08, diffuse=0.0, specular=0.0)]
        objects = self.court.objects + [self.chips, self.pin, self.stem] + self.view.lines
        self.court.bulb.visible = self.lamp.value
        if self.lamp.value:
            lights.append(PointLight(self.court.lamp_at, color=(255, 210, 150), diffuse=0.9, range=9.0, shadows=True))
        lights.append(self.court.fire_light(self.t))
        # (a sprite is a flat picture turned to the camera, lit as one: from behind it, the sun leaves it in shade,
        # so he gives off a little light of his own, less by night)
        self.fighter.sprite.emissive = 0.45 if self.sun.value else 0.15
        if self.fighter.light() is not None:
            lights.append(self.fighter.light())
        self.view.show(self.debug.value)
        if self.debug.value:
            self.view.update(self.court, eye, look)  # (court.solid was updated by blocked, above)
        self.pin.visible = self.stem.visible = self.debug.value and self.pinned is not None
        if self.pin.visible:  # (bobbing over the spot, on a stem down to it)
            top = self.pinned + (0.0, 0.45 + 0.06 * np.sin(3.0 * self.t), 0.0)
            self.pin.position = top
            self.stem.set([self.pinned], [top])
        self.renderer.background = self.backgrounds[self.background.value]
        # Fog in the world, fading into the background (the sky's horizon, say) or into white mist: thicker further
        # up the slider.
        v = self.fog.value
        colour = MIST if self.fog_into.value == "mist" else None
        self.renderer.fog = Fog(start=2.0, end=6.0 + 60.0 * (1.0 - v) ** 2, color=colour) if v > 0 else 0.0

        screen.draw(self.renderer, objects, camera, lights, rows=view_rows)
        self.drawn = objects  # (what the framebuffer's ids number, from 1)
        if self.tags.value:  # a name over each thing nearby, hidden behind whatever is in front of it
            for name, point, objects in self.court.name_tags(eye, TAG_REACH):
                screen.label(self.renderer, point, name, Color.WHITE, owner=objects, dim=True)
        cy, cx = view_rows // 2, cols // 2
        screen.text(cy, cx, "+", Color.WHITE, bold=True)
        hit = self.pick(cx, cy)
        seeing = self.court.names.get(id(hit.object), "?") + f" ({hit.distance:.1f} m)" if hit else "the sky"
        self.panel.draw(screen, rows - 2, 1)
        keys = "exact" if held.exact else "estimated from repeats"
        self.bar.draw(screen, f"looking at {seeing}   {self.clicked}   [wasd] walk  [q/e/arrows] turn  [esc] quit   "
                              f"held keys: {keys}")
        return True

    def pick(self, x, y):
        """What is drawn in cell (x, y) of the last frame (a Pick, or a Hit), seeing through the debug view's lines
        and pin: where they are in the way, the ray query through the cell (which knows only the solid things)."""
        hit = self.renderer.pick(x, y)
        if hit is not None and hit.object in (self.pin, self.stem, *self.view.lines):
            origin, direction = self.renderer.ray(x + 0.5, y + 0.5)
            hit = self.court.solid.raycast(origin, direction)
        return hit

    def throw_chips(self, hit, x, y):
        """A puff of chips from where a click hit (of cell (x, y)), coloured as the thing hit is drawn there (or as
        it is coloured, if not drawn there), and the pin moved over it."""
        fb, (pw, ph) = self.renderer.framebuffer, self.renderer.cell_pixels
        cell = (slice(y * ph, (y + 1) * ph), slice(x * pw, (x + 1) * pw))
        drawn = np.isin(fb.ids[cell], [i + 1 for i, obj in enumerate(self.drawn) if obj is hit.object])
        if drawn.any():
            colour = tuple(linear_to_srgb(fb.rgb[cell][drawn].mean(axis=0)) * 255)
        else:
            colour = getattr(hit.object, "color", (200, 200, 200))
        out = self.renderer.ray(x + 0.5, y + 0.5)[1]  # (back towards the walker, and up)
        self.chips.emit(hit.position - 0.05 * out, 24, velocity=-1.5 * out + (0.0, 2.0, 0.0), speed=(0.8, 2.2),
                        life=(1.5, 2.5), size=(0.03, 0.07), color=colour, spin=(4.0, 12.0))
        self.pinned = hit.position.copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, help="random seed")
    parser.add_argument("--fps", type=int, default=30)
    add_display_args(parser)
    args = parser.parse_args()
    try:
        run(Walk(args.seed), args.fps, title="unicode3d room",
            **display_options(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
