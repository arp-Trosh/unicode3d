# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
"""3D maze screensaver, after the Windows 98 one: python -m unicode3d.examples.maze [--size N] [--seed N]

The camera walks through a random maze from the blue marker to the gold one,
keeping a hand on the right-hand wall (or taking the shortest way), spins round
at the exit and starts a new maze. The panel sets the maze's size, the walking
speed and the fog, and switches the headlamp, the textures (off: flat colours
per face), a polished floor that mirrors the maze, and a map.
"""
import argparse
from collections import deque

import numpy as np

from .hud import StatusBar
from ..background import Fog
from ..color import Color
from ..keys import Key
from ..mesh import Mesh
from ..scene import Camera, Light, Object3D, PointLight, Renderer
from ..terminal import add_display_args, display_options, run
from ..transforms import quat_axis_angle
from ..ui import Choice, Panel, Slider, Toggle

CELL, WALL_HEIGHT, EYE = 2.0, 2.0, 1.0
BLOCK = 6  # cells a side in each piece of the maze's mesh, so pieces out of view can be skipped whole
N, E, S, W = range(4)
DX, DZ = (0, 1, 0, -1), (-1, 0, 1, 0)  # north is -z, east +x
WALL_COLOR, FLOOR_COLOR, CEILING_COLOR = (150, 70, 50), (110, 110, 115), (205, 200, 185)


# ----- the maze --------------------------------------------------------------------------

def generate(width, height, rng):
    """A perfect maze (one way between any two cells), by random depth-first search.

    Returns walls (height, width, 4): whether cell (z, x) has a wall on side N, E, S, W."""
    walls = np.ones((height, width, 4), bool)
    seen = np.zeros((height, width), bool)
    stack = [(0, 0)]
    seen[0, 0] = True
    while stack:
        x, z = stack[-1]
        options = [d for d in range(4) if 0 <= x + DX[d] < width and 0 <= z + DZ[d] < height
                   and not seen[z + DZ[d], x + DX[d]]]
        if not options:
            stack.pop()
            continue
        d = options[rng.integers(len(options))]
        nx, nz = x + DX[d], z + DZ[d]
        walls[z, x, d] = walls[nz, nx, (d + 2) % 4] = False
        seen[nz, nx] = True
        stack.append((nx, nz))
    return walls


def wall_follower(walls, start, heading, goal):
    """Actions from `start` to `goal`, keeping a hand on the right-hand wall: "L", "R" (turn) and "F" (a cell on)."""
    (x, z), h, actions = start, heading, []
    while (x, z) != goal and len(actions) < 20 * walls.size:
        for turn in (1, 0, -1, 2):  # right, ahead, left, back
            d = (h + turn) % 4
            if not walls[z, x, d]:
                actions += {1: ["R"], 0: [], -1: ["L"], 2: ["R", "R"]}[turn] + ["F"]
                h, x, z = d, x + DX[d], z + DZ[d]
                break
    return actions


def shortest(walls, start, heading, goal):
    """Actions along the shortest way from `start` to `goal` (breadth-first search)."""
    came = {start: None}
    todo = deque([start])
    while todo:
        x, z = todo.popleft()
        for d in range(4):
            nxt = (x + DX[d], z + DZ[d])
            if not walls[z, x, d] and nxt not in came:
                came[nxt] = ((x, z), d)
                todo.append(nxt)
    dirs, cell = [], goal
    while came[cell] is not None:
        cell, d = came[cell]
        dirs.append(d)
    actions, h = [], heading
    for d in reversed(dirs):
        turn = (d - h) % 4
        actions += {0: [], 1: ["R"], 3: ["L"], 2: ["R", "R"]}[turn] + ["F"]
        h = d
    return actions


# ----- how it looks ------------------------------------------------------------------------

def brick_texture(rng, res=64):
    """Rows of bricks, each a slightly different red, with light mortar between (sRGB 0..1)."""
    tex = np.empty((res, res, 3))
    tex[:] = (0.72, 0.69, 0.62)
    rows, per_row, mortar = 4, 2, 2
    bh, bw = res // rows, res // per_row
    for r in range(rows):
        shift = (bw // 2) * (r % 2)
        for b in range(per_row + 1):
            x0 = b * bw - shift
            shade = rng.uniform(0.8, 1.1)
            colour = np.array([0.58, 0.22, 0.15]) * shade
            ys = slice(r * bh + mortar, (r + 1) * bh)
            for x in range(max(x0 + mortar, 0), min(x0 + bw, res)):
                tex[ys, x] = colour
    tex *= rng.uniform(0.92, 1.05, (res, res, 1))  # a little grain
    return np.clip(tex, 0, 1)


def floor_texture(rng, res=64, tiles=4):
    y, x = np.mgrid[0:res, 0:res] * tiles // res
    checker = (x + y) % 2
    tex = np.where(checker[..., None], (0.55, 0.55, 0.58), (0.35, 0.35, 0.38)) * rng.uniform(0.9, 1.05, (res, res, 1))
    return np.clip(tex, 0, 1)


def ceiling_texture(res=64, panels=2):
    tex = np.full((res, res, 3), 0.85)
    step = res // panels
    tex[::step] = tex[:, ::step] = 0.55
    tex[1::step] = tex[:, 1::step] = 0.65
    return tex


def maze_meshes(walls, textures, x0, z0, x1, z1, floor=False):
    """The textured and the flat-coloured mesh of cells x0..x1, z0..z1: walls facing into their cells and
    a ceiling square per cell, or with floor, just a floor square per cell (materials 0 wall, 1 floor,
    2 ceiling). The floor is kept apart so that it is flat, and can be a mirror."""
    verts, faces, uvs, materials = [], [], [], []
    corner_uv = np.array([(0, 0), (1, 0), (1, 1), (0, 1)], float)

    def quad(corners, material):
        base = len(verts)
        verts.extend(corners)
        faces.extend([(base, base + 1, base + 2), (base, base + 2, base + 3)])
        uvs.extend([corner_uv[[0, 1, 2]], corner_uv[[0, 2, 3]]])
        materials.extend([material, material])

    for z in range(z0, z1):
        for x in range(x0, x1):
            cx, cz = (x + 0.5) * CELL, (z + 0.5) * CELL
            a, b = x * CELL, (x + 1) * CELL
            c, d = z * CELL, (z + 1) * CELL
            if floor:
                quad([(a, 0, d), (b, 0, d), (b, 0, c), (a, 0, c)], 1)                              # facing up
                continue
            quad([(a, WALL_HEIGHT, c), (b, WALL_HEIGHT, c), (b, WALL_HEIGHT, d), (a, WALL_HEIGHT, d)], 2)  # ceiling
            for side in range(4):
                if not walls[z, x, side]:
                    continue
                fx, fz = DX[side], DZ[side]                     # towards the wall, from the cell's centre
                rx, rz = -fz, fx                                # right, seen facing the wall
                px, pz = cx + fx * CELL / 2, cz + fz * CELL / 2
                h = CELL / 2
                quad([(px - rx * h, 0, pz - rz * h), (px + rx * h, 0, pz + rz * h),
                      (px + rx * h, WALL_HEIGHT, pz + rz * h), (px - rx * h, WALL_HEIGHT, pz - rz * h)], 0)
    verts, faces = np.array(verts, float), np.array(faces)
    textured = Mesh(verts, faces, np.array(uvs), np.array(materials), textures)
    flat = Mesh(verts, faces, face_colors=np.array((WALL_COLOR, FLOOR_COLOR, CEILING_COLOR))[materials])
    return textured, flat


def gem_mesh():
    """An octahedron: the start and exit markers."""
    v = np.array([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)], float)
    f = [(0, 2, 4), (4, 2, 1), (1, 2, 5), (5, 2, 0), (4, 3, 0), (1, 3, 4), (5, 3, 1), (0, 3, 5)]
    return Mesh(v, np.array(f))


def smooth(t):
    return t * t * (3 - 2 * t)


# ----- the screensaver ---------------------------------------------------------------------

class Maze:
    def __init__(self, size, seed=None):
        self.rng = np.random.default_rng(seed)
        self.textures = [brick_texture(self.rng), floor_texture(self.rng), ceiling_texture()]
        self.renderer = Renderer(1, 1, fog=0.45)
        self.bar = StatusBar(self.renderer)
        self.size = Slider("Size", size, 3, 40, keys="[]", length=10, on_change=lambda v: self.new_maze())
        self.speed = Slider("Speed", 1.5, 0.5, 5.0, step=0.5, keys="-=", length=10, fmt=lambda v: f"{v:.1f}")
        self.fog = Slider("Fog", 0.45, 0.0, 0.9, step=0.05, keys=",.", length=8, fmt=lambda v: f"{v:.2f}")
        self.headlamp = Toggle("Headlamp (l)", True, key="l")
        self.textured = Toggle("Textures (t)", True, key="t")
        self.polished = Toggle("Polished floor (p)", False, key="p")
        self.show_map = Toggle("Map (m)", True, key="m")
        self.route = Choice("Route (r)", ("right-hand wall", "shortest way"), key="r", on_change=lambda v: self.new_maze())
        self.panel = Panel([self.size, self.speed, self.fog, self.headlamp, self.textured, self.polished, self.show_map,
                            self.route])
        # Glowing, see-through gems: the corridor shows through them.
        self.start_marker = Object3D(gem_mesh(), color=(60, 120, 255), emissive=0.8, scale=0.3, opacity=0.35)
        self.exit_marker = Object3D(gem_mesh(), color=(255, 200, 40), emissive=0.8, scale=0.3, opacity=0.35)
        self.paused = False
        self.new_maze()

    def new_maze(self):
        n = self.size.value
        self.walls = generate(n, n, self.rng)
        self.pieces = []
        for z0 in range(0, n, BLOCK):
            for x0 in range(0, n, BLOCK):
                textured, flat = maze_meshes(self.walls, self.textures, x0, z0, min(x0 + BLOCK, n), min(z0 + BLOCK, n))
                self.pieces.append((Object3D(textured, color=(255, 255, 255)), textured, flat))
        # The whole floor as one flat piece: one mirror when polished, rather than one for each block.
        textured, flat = maze_meshes(self.walls, self.textures, 0, 0, n, n, floor=True)
        self.floor = Object3D(textured, color=(255, 255, 255))
        self.pieces.append((self.floor, textured, flat))
        self.goal = (n - 1, n - 1)
        self.cell, self.heading = (0, 0), S if not self.walls[0, 0, S] else E
        plan = wall_follower if self.route.value == "right-hand wall" else shortest
        self.actions = deque(plan(self.walls, self.cell, self.heading, self.goal) + ["spin"])
        self.visited = {self.cell}
        self.action, self.t = None, 0.0
        self.start_marker.position = self.centre((0, 0), 1.0)
        self.exit_marker.position = self.centre(self.goal, 1.0)

    @staticmethod
    def centre(cell, y):
        return np.array([(cell[0] + 0.5) * CELL, y, (cell[1] + 0.5) * CELL])

    def advance(self, dt):
        """Move along the plan; returns the camera's (position, yaw) partway through the current action."""
        durations = {"F": 1.0, "L": 0.5, "R": 0.5, "spin": 2.5}
        self.t += dt * self.speed.value
        while self.action is None or self.t >= durations[self.action]:
            if self.action is not None:
                self.t -= durations[self.action]
                if self.action == "F":
                    self.cell = (self.cell[0] + DX[self.heading], self.cell[1] + DZ[self.heading])
                    self.visited.add(self.cell)
                elif self.action in "LR":
                    self.heading = (self.heading + (1 if self.action == "R" else -1)) % 4
                elif self.action == "spin":
                    self.new_maze()
                    return self.advance(0.0)
            self.action = self.actions.popleft() if self.actions else "spin"
        p = smooth(min(self.t / durations[self.action], 1.0))
        position = self.centre(self.cell, EYE)
        yaw = self.heading * np.pi / 2
        if self.action == "F":
            position = position + p * CELL * np.array([DX[self.heading], 0.0, DZ[self.heading]])
        elif self.action in "LR":
            yaw += (1 if self.action == "R" else -1) * p * np.pi / 2
        else:
            yaw += p * 2 * np.pi
        return position, yaw

    def draw_map(self, screen, top, right):
        """The maze from above in text: walls, the cells walked, the camera and the exit, if it fits."""
        n = self.size.value
        rows, cols = screen.size()
        if 2 * n + 1 > rows - top - 3 or 2 * n + 1 > cols // 3:
            return
        grid = np.full((2 * n + 1, 2 * n + 1), " ")
        grid[::2, ::2] = "+"
        for z in range(n):
            for x in range(n):
                y, c = 2 * z + 1, 2 * x + 1
                grid[y - 1, c] = "-" if self.walls[z, x, N] else " "
                grid[y + 1, c] = "-" if self.walls[z, x, S] else " "
                grid[y, c - 1] = "|" if self.walls[z, x, W] else " "
                grid[y, c + 1] = "|" if self.walls[z, x, E] else " "
                if (x, z) in self.visited:
                    grid[y, c] = "."
        gx, gz = self.goal
        grid[2 * gz + 1, 2 * gx + 1] = "E"
        cx, cz = self.cell
        grid[2 * cz + 1, 2 * cx + 1] = "^>v<"[self.heading]
        left = right - grid.shape[1]
        for i, line in enumerate(grid):
            screen.text(top + i, left, "".join(line), dim=True)
        screen.text(2 * gz + 1 + top, 2 * gx + 1 + left, "E", Color.YELLOW, bold=True)
        screen.text(2 * cz + 1 + top, 2 * cx + 1 + left, "^>v<"[self.heading], Color.CYAN, bold=True)

    def frame(self, screen, dt, events):
        dt = min(dt, 0.1)
        events = self.panel.handle(self.bar.handle(events, screen))
        for ev in events:
            if ev in (ord("q"), Key.ESC):
                return False
            if ev == ord(" "):
                self.paused = not self.paused
            elif ev == ord("n"):
                self.new_maze()
        position, yaw = self.advance(0.0 if self.paused else dt)
        forward = np.array([np.sin(yaw), 0.0, -np.cos(yaw)])
        camera = Camera(position=position, target=position + forward, fov=70.0, near=0.05, far=200.0)

        for obj, textured, flat in self.pieces:
            obj.mesh = textured if self.textured.value else flat
        self.floor.reflectivity = 0.35 if self.polished.value else 0.0
        spin = quat_axis_angle((0, 1, 0), 2.0 * (self.t + 0.37 * len(self.visited)))
        self.start_marker.rotation = self.exit_marker.rotation = spin
        if self.headlamp.value:
            # Carried a little to the right of and below the eye, so that its shadows show beside what casts them.
            lantern = position + 0.3 * np.array([np.cos(yaw), 0.0, np.sin(yaw)]) + np.array([0.0, -0.25, 0.0])
            lights = [Light(direction=np.array([0.3, -1.0, -0.6]), ambient=0.12, diffuse=0.25, specular=0.0),
                      PointLight(lantern, color=(255, 240, 215), diffuse=0.85, specular=0.2, range=5 * CELL,
                                 shadows=True)]
        else:
            lights = Light(direction=np.array([0.3, -1.0, -0.6]), ambient=0.4, diffuse=0.5, specular=0.1)
        # Fog in the world, fading into the dark: thicker further up the slider.
        v = self.fog.value
        self.renderer.fog = Fog(start=CELL, end=CELL * (3.0 + 24.0 * (1.0 - v) ** 2)) if v > 0 else 0.0

        rows, cols = screen.size()
        self.renderer.resize(cols, max(rows - 2, 1), screen.cell_pixels)
        objects = [obj for obj, _, _ in self.pieces] + [self.start_marker, self.exit_marker]
        fb = self.renderer.render(objects, camera, lights)
        screen.erase()
        screen.draw_frame(fb)
        if self.show_map.value:
            self.draw_map(screen, 1, cols - 2)
        self.panel.draw(screen, rows - 2, 1)
        self.bar.draw(screen, f"{self.size.value}x{self.size.value} maze, {len(self.visited)} cells walked"
                              f"{'  (paused)' if self.paused else ''}   [space] pause  [n] new maze  [tab] panel  [q] quit")
        screen.refresh()
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", type=int, default=12, help="cells along each side (3-40)")
    parser.add_argument("--seed", type=int, help="random seed")
    parser.add_argument("--fps", type=int, default=30)
    add_display_args(parser)
    args = parser.parse_args()
    demo = Maze(min(max(args.size, 3), 40), args.seed)
    try:
        run(demo.frame, args.fps, mouse="drag", title="unicode3d maze", **display_options(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
