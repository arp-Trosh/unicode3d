# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
"""An isometric tactics board, to show what an orthographic camera is for: python -m unicode3d.examples.tactics

A small island of terraced tiles with houses, trees and four pawns, seen the way strategy, tactics and board
games show their maps: through an orthographic camera (Camera(projection="ortho")) at the isometric angle.
Every tile is the same size wherever it is on the screen, the grid's lines stay parallel, and a pawn looks the
same on any tile. O switches to an ordinary perspective camera framed the same way, to compare: the near tiles
grow, the grid's lines converge, and houses lean outward towards the edges of the screen.

Click a pawn to pick it, then a tile to send it there: it hops along the shortest way, climbing at most one level
a step. The tile under the pointer lights up, found by a ray query (Colliders.raycast) along renderer.ray(), which
in an orthographic view gives parallel rays. Each pawn's name and health are over it (labels), dim where
something hides it, and at the edge of the screen, pointing, where it is out of view.

Q/E turn the view a quarter, +/- or the wheel zoom, WASD or the arrows pan, Tab picks the next pawn, Esc quits.
"""
import argparse
from collections import deque

import numpy as np

from .hud import StatusBar
from ..background import Gradient
from ..color import Color
from ..keys import Key, MouseEvent
from ..mesh import Mesh, make_box
from ..queries import Colliders
from ..scene import Camera, Light, Object3D, Renderer
from ..shapes import blob_mesh, block_mesh, merge_meshes
from ..terminal import add_display_args, display_options, run
from ..transforms import quat_axis_angle, quat_mul

N = 12               # tiles along each side of the board
STEP = 0.35          # how much higher each level of terrace is
WATER = 0.12         # the water's height
LEVEL_COLOURS = ((214, 196, 140), (110, 170, 80), (78, 138, 64), (150, 145, 140))  # sand, grass, deep grass, rock
PITCH = np.arctan(1.0 / np.sqrt(2.0))  # the isometric angle: a cube's three faces equally foreshortened
FOV = 60.0           # the perspective camera's, when O switches to it (as wide as a 3D game's)
HOP = 0.22           # seconds a pawn takes for each tile it hops along
PAWNS = (("Ash", Color.RED, 0.9), ("Bryn", Color.BLUE, 0.6), ("Cato", Color.YELLOW, 1.0), ("Dell", Color.MAGENTA, 0.35))


def island(rng):
    """Terrace levels (N, N): -1 for water, 0 (sand) to 3 (rock) for land, higher towards a few hills."""
    j, i = np.mgrid[0:N, 0:N]
    x, z = (i - (N - 1) / 2) / (N / 2), (j - (N - 1) / 2) / (N / 2)
    height = 1.2 - 1.3 * np.hypot(x, z) ** 2  # a dome, under water at the corners
    for _ in range(4):  # hills
        cx, cz, r = rng.uniform(-0.6, 0.6), rng.uniform(-0.6, 0.6), rng.uniform(0.25, 0.45)
        height += 1.1 * np.exp(-((x - cx) ** 2 + (z - cz) ** 2) / r ** 2)
    return np.clip(np.floor(height * 1.6), -1, 3).astype(int).T  # [i, j]


def tile_centre(i, j):
    """Where tile (i, j)'s middle is in the world, in x and z."""
    return i - (N - 1) / 2, j - (N - 1) / 2


def tile_top(level):
    return WATER if level < 0 else 0.25 + STEP * level


def terrain_mesh(levels):
    """The land's tiles as one mesh: a column for each, its top in its level's colour (in a checker, so the
    grid shows) and its sides darker."""
    parts = []
    for i in range(N):
        for j in range(N):
            if levels[i, j] < 0:
                continue
            x, z = tile_centre(i, j)
            top = tile_top(levels[i, j])
            column = block_mesh((x, (top - 0.4) / 2, z), (1.0, top + 0.4, 1.0))
            colour = np.array(LEVEL_COLOURS[levels[i, j]], float) * (0.9 if (i + j) % 2 else 1.0)
            v = column.vertices[column.faces]
            up = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])[:, 1] > 0.5
            column.face_colors = np.where(up[:, None], colour, colour * 0.62).astype(int)
            parts.append(column)
    return merge_meshes(parts)


def flat_square(size, y=0.0):
    h = size / 2
    return Mesh(np.array([(-h, y, h), (h, y, h), (h, y, -h), (-h, y, -h)], float), np.array([(0, 1, 2), (0, 2, 3)]))


def pawn_mesh():
    """A pawn standing on y = 0: a body and a head."""
    return merge_meshes([blob_mesh((0.2, 0.28, 0.2), (0.0, 0.28, 0.0), rings=10, segments=14),
                         blob_mesh((0.13, 0.13, 0.13), (0.0, 0.66, 0.0), rings=8, segments=12)])


class Pawn:
    def __init__(self, name, colour, health, tile, mesh, levels):
        self.name, self.colour, self.health, self.tile = name, colour, health, tile
        self.obj = Object3D(mesh, color=colour, specular=0.6, shininess=20.0)
        self.path = []     # tiles still to hop to
        self.hopped = 0.0  # how far through the hop to path[0], 0..1
        self.place(levels)

    def place(self, levels):
        """Stand on its tile, or partway through the hop to the next."""
        x, z = tile_centre(*self.tile)
        y = tile_top(levels[self.tile])
        if self.path:
            nx, nz = tile_centre(*self.path[0])
            ny, t = tile_top(levels[self.path[0]]), self.hopped
            x, z, y = x + (nx - x) * t, z + (nz - z) * t, y + (ny - y) * t + 0.3 * np.sin(np.pi * t)
        self.obj.position = np.array([x, y, z])

    def update(self, dt, levels):
        if self.path:
            self.hopped += dt / HOP
            while self.path and self.hopped >= 1.0:
                self.tile, self.hopped = self.path.pop(0), self.hopped - 1.0
            if not self.path:
                self.hopped = 0.0
        self.place(levels)


class Tactics:
    def __init__(self, seed=3):
        rng = np.random.default_rng(seed)
        self.levels = island(rng)
        self.terrain = Object3D(terrain_mesh(self.levels), color=(255, 255, 255), specular=0.1)
        sea = Object3D(flat_square(8 * N, WATER), color=(40, 90, 140), reflectivity=0.35, specular=1.5,
                       shininess=60.0)
        self.scenery, self.blocked = [], set()  # houses and trees, and the tiles they stand on
        land = [(i, j) for i in range(N) for j in range(N) if self.levels[i, j] >= 0]
        order = rng.permutation(len(land))
        box = make_box()
        for k in order[:5]:  # houses: a body, and a box turned on its edge for a roof
            i, j = land[k]
            x, z = tile_centre(i, j)
            top = tile_top(self.levels[i, j])
            turn = quat_axis_angle((0, 1, 0), np.pi / 2 * rng.integers(0, 2))
            body = Object3D(box, np.array([x, top + 0.25, z]), turn, scale=(0.7, 0.5, 0.6), color=(225, 215, 195))
            # (turned 45 degrees about its long side, its lower half inside the body: a gable roof)
            roof = Object3D(box, np.array([x, top + 0.5, z]), quat_mul(turn, quat_axis_angle((1, 0, 0), np.pi / 4)),
                            scale=(0.8, 0.44, 0.44), color=(170, 70, 50))
            self.scenery += [body, roof]
            self.blocked.add((i, j))
        crown = blob_mesh((0.28, 0.3, 0.28), rings=8, segments=12)
        for k in order[5:13]:  # trees
            i, j = land[k]
            x, z = tile_centre(i, j)
            top = tile_top(self.levels[i, j])
            self.scenery += [Object3D(box, np.array([x, top + 0.15, z]), scale=(0.1, 0.3, 0.1), color=(110, 80, 50)),
                             Object3D(crown, np.array([x, top + 0.55, z]), color=(60, 130, 60), specular=0.2)]
            self.blocked.add((i, j))
        mesh = pawn_mesh()
        free = [land[k] for k in order[13:]]
        self.pawns = [Pawn(name, colour, health, free[n], mesh, self.levels)
                      for n, (name, colour, health) in enumerate(PAWNS)]
        marker = flat_square(0.94)
        self.hover = Object3D(marker, color=(255, 240, 120), emissive=0.7, opacity=0.5, cast_shadows=False)
        self.chosen = Object3D(marker, color=(120, 230, 255), emissive=0.8, opacity=0.6, cast_shadows=False)
        self.objects = [self.terrain, sea, *self.scenery, *(p.obj for p in self.pawns), self.hover, self.chosen]
        self.solid = Colliders([self.terrain, *self.scenery, *(p.obj for p in self.pawns)])
        self.renderer = Renderer(1, 1, background=Gradient((70, 100, 140), (15, 20, 35)), fog=0.2)
        self.bar = StatusBar(self.renderer)
        self.light = Light(direction=np.array([-0.4, -1.0, -0.6]), ambient=0.35, diffuse=0.7, shadows=True)
        self.ortho = True
        self.turns = 0             # quarter turns of the view
        self.yaw = np.pi / 4       # as the view turns towards turns
        self.zoom = 11.0           # how tall the view is in the world, at the board
        self.pan = np.zeros(3)
        self.selected = self.pawns[0]
        self.pointer = None        # the cell the mouse is over
        self.hovered = None        # the tile under it
        self.camera = self.make_camera()

    # ----- the camera ----------------------------------------------------------------------------------------

    def make_camera(self):
        """The orthographic camera, or a perspective one framing the board about the same."""
        target = np.array([0.0, 0.5, 0.0]) + self.pan
        back = np.array([np.sin(self.yaw) * np.cos(PITCH), np.sin(PITCH), np.cos(self.yaw) * np.cos(PITCH)])
        if self.ortho:
            return Camera(target + back * 40.0, target, projection="ortho", size=self.zoom, near=1.0, far=120.0)
        distance = self.zoom / 2 / np.tan(np.radians(FOV) / 2)  # where the same height fills the view
        return Camera(target + back * distance, target, fov=FOV, near=0.3, far=120.0)

    def ground_axes(self):
        """The ground's directions to the right of the screen and away from the viewer."""
        return (np.array([np.cos(self.yaw), 0.0, -np.sin(self.yaw)]),
                np.array([-np.sin(self.yaw), 0.0, -np.cos(self.yaw)]))

    # ----- the board -----------------------------------------------------------------------------------------

    def tile_at(self, x, y):
        """(pawn, tile) under cell (x, y) of the frame: the pawn there or None, and the tile there or None."""
        ray = self.renderer.ray(x + 0.5, y + 0.5)
        hit = self.solid.raycast(*ray) if ray is not None else None
        if hit is None:
            return None, None
        pawn = next((p for p in self.pawns if p.obj is hit.object), None)
        if pawn is not None:
            return pawn, pawn.tile
        p = hit.position - hit.normal * 0.01  # (just inside what it hit: a side's tile, not its neighbour's)
        i, j = int(np.floor(p[0] + N / 2)), int(np.floor(p[2] + N / 2))
        return None, ((i, j) if 0 <= i < N and 0 <= j < N else None)

    def free(self, tile, pawn):
        i, j = tile
        return (0 <= i < N and 0 <= j < N and self.levels[i, j] >= 0 and tile not in self.blocked
                and not any(p is not pawn and (p.tile == tile or tile in p.path[-1:]) for p in self.pawns))

    def path_to(self, pawn, goal):
        """The tiles from pawn's to goal, shortest first (breadth-first), stepping to free tiles at most one level
        up or down; [] if there is no way."""
        start = tuple(pawn.path[-1]) if pawn.path else pawn.tile
        came = {start: None}
        queue = deque([start])
        while queue:
            tile = queue.popleft()
            if tile == goal:
                path = []
                while tile != start:
                    path.append(tile)
                    tile = came[tile]
                return path[::-1]
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                step = (tile[0] + di, tile[1] + dj)
                if (step not in came and self.free(step, pawn)
                        and abs(self.levels[step] - self.levels[tile]) <= 1):
                    came[step] = tile
                    queue.append(step)
        return []

    def click(self, x, y):
        pawn, tile = self.tile_at(x, y)
        if pawn is not None:
            self.selected = pawn
        elif tile is not None and self.selected is not None:
            self.selected.path += self.path_to(self.selected, tile)

    # ----- a frame -------------------------------------------------------------------------------------------

    def frame(self, screen, dt, events):
        dt = min(dt, 0.1)
        right, away = self.ground_axes()
        for ev in self.bar.handle(events, screen):
            if ev in (Key.ESC, 3):
                return False
            if isinstance(ev, MouseEvent):
                self.pointer = (ev.x, ev.y)
                if ev.button == MouseEvent.WHEEL_UP and ev.pressed:
                    self.zoom = max(self.zoom / 1.15, 4.0)
                elif ev.button == MouseEvent.WHEEL_DOWN and ev.pressed:
                    self.zoom = min(self.zoom * 1.15, 30.0)
                elif ev.button == MouseEvent.LEFT and ev.pressed and not ev.moved:
                    self.click(ev.x, ev.y)
                continue
            key = chr(ev) if isinstance(ev, int) and 32 <= ev < 127 else ev
            pan = {"w": away, Key.UP: away, "s": -away, Key.DOWN: -away, "d": right, Key.RIGHT: right,
                   "a": -right, Key.LEFT: -right}
            if key in pan:
                self.pan = np.clip(self.pan + pan[key] * 0.05 * self.zoom, -N / 2, N / 2)
            elif key == "q":
                self.turns -= 1
            elif key == "e":
                self.turns += 1
            elif key in ("+", "="):
                self.zoom = max(self.zoom / 1.15, 4.0)
            elif key in ("-", "_"):
                self.zoom = min(self.zoom * 1.15, 30.0)
            elif key == "o":
                self.ortho = not self.ortho
            elif key == Key.TAB:
                self.selected = self.pawns[(self.pawns.index(self.selected) + 1) % len(self.pawns)]
        # The view turns smoothly towards the quarter it was turned to.
        self.yaw += (np.pi / 4 + self.turns * np.pi / 2 - self.yaw) * min(1.0, 10.0 * dt)
        for pawn in self.pawns:
            pawn.update(dt, self.levels)
        self.solid.update()  # (the pawns move)
        self.camera = self.make_camera()

        rows, cols = screen.size()
        view_rows = max(rows - 1, 1)
        self.renderer.resize(cols, view_rows, screen.cell_pixels)
        # The markers go where the last frame's view puts the pointer and the chosen pawn.
        _, self.hovered = self.tile_at(*self.pointer) if self.pointer is not None else (None, None)
        for marker, tile in ((self.hover, self.hovered), (self.chosen, self.selected.tile)):
            marker.visible = tile is not None and self.levels[tile] >= 0
            if marker.visible:
                x, z = tile_centre(*tile)
                marker.position = np.array([x, tile_top(self.levels[tile]) + 0.02, z])
        fb = self.renderer.render(self.objects, self.camera, self.light)
        screen.erase()
        screen.draw_frame(fb)
        self.draw_labels(screen)
        mode, other = ("orthographic", "perspective") if self.ortho else ("perspective", "orthographic")
        # (the point of the demo: which view this is, and the key to compare with the other)
        screen.text(0, 1, f" {mode} view   [o] switch to {other} ", Color.WHITE, reverse=True)
        where = f"   tile {self.hovered[0]},{self.hovered[1]}" if self.hovered else ""
        self.bar.draw(screen, f"[o] to {other}  [click] pick a pawn, then a tile  [q/e] turn  [+/-] zoom  "
                              f"[wasd] pan  [tab] next  [esc] quit   {self.selected.name} chosen{where}")
        screen.refresh()
        return True

    def draw_labels(self, screen):
        """Each pawn's name over it and its health under that; dim where something is in front of it; at the
        edge of the screen, with an arrow, where it is out of view."""
        r = self.renderer
        arrows = "◀▶▲▼" if screen.unicode else "<>^v"
        for pawn in self.pawns:
            lo, hi = pawn.obj.world_bounds()
            anchor = r.anchor(((lo[0] + hi[0]) / 2, hi[1] + 0.1, (lo[2] + hi[2]) / 2), owner=pawn.obj, clamp=True)
            if anchor is None:
                continue
            bold = pawn is self.selected
            if anchor.edge:
                arrow = (arrows[0] if anchor.x == 0 else arrows[1] if anchor.x == r.width - 1 else
                         arrows[2] if anchor.y == 0 else arrows[3])
                text = f"{arrow} {pawn.name}" if arrow in (arrows[0], arrows[2]) else f"{pawn.name} {arrow}"
                x = min(max(anchor.x - len(text) // 2, 0), r.width - len(text))
                screen.text(anchor.y, x, text, pawn.colour, bold=bold)
                continue
            screen.text(anchor.y - 1, anchor.x - len(pawn.name) // 2, pawn.name, pawn.colour, bold=bold,
                        dim=anchor.hidden)
            screen.bar(anchor.y, anchor.x - 2, 5, pawn.health,
                       Color.GREEN if pawn.health > 0.5 else Color.YELLOW if pawn.health > 0.25 else Color.RED)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=3, help="which island (default 3)")
    parser.add_argument("--fps", type=int, default=30)
    add_display_args(parser)
    args = parser.parse_args()
    try:
        run(Tactics(args.seed).frame, args.fps, mouse="move", title="unicode3d tactics", **display_options(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
