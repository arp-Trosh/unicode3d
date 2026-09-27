# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
"""Walk around a courtyard of things to look at: python -m unicode3d.examples.room

WASD walks, Q/E or Left/Right turn, Up/Down look up and down, Esc quits. The
crosshair names what it is on; click anything to name that. The panel switches
the lamp and the sun, picks the background (a sky, a starry sky box, a
gradient or none) and sets the fog.

Walking feels best in a terminal that reports key releases (kitty, foot,
Ghostty, WezTerm, Alacritty, iTerm2, Windows Terminal): elsewhere a held key is
worked out from its repeats, and a tap walks for about half a second.
"""
import argparse
import colorsys

import numpy as np

from .dice import make_die, orientation_showing
from .hud import StatusBar
from ..background import Gradient, Sky, SkyBox
from ..color import Color
from ..keys import Key, MouseEvent
from ..mesh import Mesh, make_box
from ..scene import Camera, Light, Node, Object3D, PointLight, Renderer
from ..shapes import blob_mesh, block_mesh, text_mesh
from ..terminal import add_display_args, display_options, run
from ..transforms import UP, quat_axis_angle, quat_mul
from ..ui import Choice, Panel, Slider, Toggle

HALF = 8.0           # the courtyard runs from -HALF to HALF in x and z
WALL_HEIGHT = 3.5
EYE = 1.6
RADIUS = 0.35        # how close the walker gets to walls and things
WALK, TURN, LOOK = 3.0, 1.8, 1.2  # units a second, radians a second

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
        self.add(Object3D(inward_box(HALF, WALL_HEIGHT, (wall, floor)), color=(255, 255, 255)), "the courtyard")
        for x, z in ((-HALF + 0.6, -HALF + 0.6), (HALF - 0.6, -HALF + 0.6), (-HALF + 0.6, HALF - 0.6), (HALF - 0.6, HALF - 0.6)):
            self.add(Object3D(block_mesh((x, WALL_HEIGHT / 2 + 0.3, z), (0.8, WALL_HEIGHT + 0.6, 0.8)), color=(170, 160, 150)),
                     "a pillar", (x, z, 0.6))

        # Dice turning slowly on a pedestal.
        self.add(Object3D(block_mesh((-4.0, 0.5, -3.0), (1.2, 1.0, 1.2)), color=(90, 90, 110)), "a pedestal", (-4.0, -3.0, 0.9))
        die = make_die(0.45)
        self.dice = [Object3D(die, np.array([-4.0 + dx, 1.25, -3.0 + dz]), orientation_showing(i, 0.0), color=c)
                     for i, (dx, dz, c) in enumerate(((-0.28, 0.0, Color.RED), (0.28, 0.2, Color.GREEN), (0.1, -0.3, Color.YELLOW)))]
        for d in self.dice:
            self.add(d, "three dice")

        # The orrery: a scene graph of nodes turning inside each other.
        self.orrery = Node(position=np.array([3.5, 1.4, -3.5]))
        self.planet_orbit = Node(parent=self.orrery)
        self.moon_orbit = Node(position=np.array([1.4, 0.0, 0.0]), parent=self.planet_orbit)
        self.add(Object3D(block_mesh((3.5, 0.35, -3.5), (0.9, 0.7, 0.9)), color=(120, 100, 80)), "the orrery's stand",
                 (3.5, -3.5, 1.0))
        self.add(Object3D(blob_mesh((0.3, 0.3, 0.3), rings=12, segments=16), color=(255, 190, 60), emissive=1.0,
                          parent=self.orrery), "the orrery's sun")
        self.add(Object3D(blob_mesh((0.16, 0.16, 0.16), rings=10, segments=14), color=Color.BLUE, parent=self.moon_orbit),
                 "a planet")
        self.moon = Node(position=np.array([0.35, 0.0, 0.0]), parent=self.moon_orbit)
        self.add(Object3D(blob_mesh((0.06, 0.06, 0.06)), color=(200, 200, 200), parent=self.moon), "its moon")

        # A table, built as one node with its top and legs, with stacked cubes of many colours on it.
        self.table = Node(position=np.array([-3.5, 0.0, 3.5]), rotation=quat_axis_angle(UP, 0.4))
        self.add(Object3D(block_mesh((0, 0.8, 0), (2.0, 0.1, 1.2)), color=(140, 90, 50), parent=self.table), "a table",
                 (-3.5, 3.5, 1.2))
        for lx in (-0.9, 0.9):
            for lz in (-0.5, 0.5):
                self.add(Object3D(block_mesh((lx, 0.4, lz), (0.08, 0.8, 0.08)), color=(110, 70, 40), parent=self.table),
                         "a table")
        cube = make_box(0.3)
        cube.face_colors = np.repeat([(230, 60, 60), (60, 200, 90), (70, 110, 240), (240, 210, 60), (220, 90, 220),
                                      (60, 210, 220)], 2, axis=0)
        for i, (x, y, z) in enumerate(((-0.5, 1.0, 0.0), (-0.1, 1.0, 0.1), (-0.3, 1.3, 0.05), (0.5, 1.0, -0.2))):
            self.add(Object3D(cube, np.array([x, y, z]), quat_axis_angle(UP, 0.7 * i), color=(255, 255, 255),
                              parent=self.table), "coloured cubes")

        # A slowly turning blob in rainbow vertex colours.
        self.blob = Object3D(rainbow_blob(), np.array([3.5, 1.3, 3.5]), scale=0.8, color=(255, 255, 255))
        self.add(Object3D(block_mesh((3.5, 0.2, 3.5), (1.0, 0.4, 1.0)), color=(80, 80, 80)), "a plinth", (3.5, 3.5, 0.8))
        self.add(self.blob, "a rainbow blob")

        # A lamp by the west wall: a glowing bulb, and the light it gives.
        self.lamp_at = np.array([-6.5, 2.3, 0.0])
        self.add(Object3D(block_mesh((-6.5, 1.1, 0.0), (0.12, 2.2, 0.12)), color=(60, 60, 60)), "a lamp", (-6.5, 0.0, 0.4))
        self.bulb = Object3D(blob_mesh((0.18, 0.18, 0.18)), self.lamp_at, color=(255, 225, 160), emissive=1.0)
        self.add(self.bulb, "a lamp")

        # The sign on the north wall.
        sign = rainbow_text("UNICODE3D")
        self.add(Object3D(sign, np.array([0.0, 2.3, -HALF + 0.4]), scale=0.14, color=(255, 255, 255)), "the sign")

    def add(self, obj, name, obstacle=None):
        self.objects.append(obj)
        self.names[id(obj)] = name
        if obstacle is not None:
            self.obstacles.append(obstacle)

    def animate(self, t):
        for i, d in enumerate(self.dice):
            d.rotation = quat_mul(quat_axis_angle(UP, 0.4 * t + i), orientation_showing(i, 0.0))
        self.planet_orbit.rotation = quat_axis_angle(UP, 0.5 * t)
        self.moon_orbit.rotation = quat_axis_angle(UP, 1.7 * t)
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
        self.bar = StatusBar()
        self.lamp = Toggle("Lamp (l)", True, key="l")
        self.sun = Toggle("Sun (u)", True, key="u")
        self.backgrounds = {"sky": Sky(), "stars": starry_sky_box(np.random.default_rng(7)),
                            "gradient": Gradient((70, 90, 140), (20, 20, 30)), "none": None}
        self.background = Choice("Background (b)", tuple(self.backgrounds), key="b")
        self.fog = Slider("Fog", 0.3, 0.0, 0.9, step=0.05, keys="[]", length=8, fmt=lambda v: f"{v:.2f}")
        self.panel = Panel([self.lamp, self.sun, self.background, self.fog], keyboard=False)
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

        eye = np.array([self.x, EYE, self.z])
        look = np.array([fx * np.cos(self.pitch), np.sin(self.pitch), fz * np.cos(self.pitch)])
        camera = Camera(position=eye, target=eye + look, fov=70.0, near=0.05, far=100.0)
        lights = [Light(direction=np.array([0.5, -1.0, -0.35]), ambient=0.3, diffuse=0.6, color=(255, 245, 230))
                  if self.sun.value else Light(ambient=0.08, diffuse=0.0, specular=0.0)]
        objects = list(self.court.objects)
        self.court.bulb.visible = self.lamp.value
        if self.lamp.value:
            lights.append(PointLight(self.court.lamp_at, color=(255, 210, 150), diffuse=0.9, range=9.0))
        self.renderer.background = self.backgrounds[self.background.value]
        self.renderer.fog = self.fog.value

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
