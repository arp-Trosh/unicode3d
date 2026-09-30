# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
"""Try materials, scale, fog and animation by hand: python -m unicode3d.examples.workshop

An object on a turntable, and a panel of settings for it:

  material   specular (how strong its highlights are), shininess (how tight: "light's" takes each light's
             own), reflectivity and opacity, and a colour
  scale      along x, y and z separately; 0 squashes it flat, and below 0 mirrors it
  fog        off, or fading into the background, into mist or into night, from `start` to `end` units
             away (rows of pillars run into the distance to show it)
  animation  the Hop button (h or space) throws the object up and spins it once, the spin shaped by the
             easing curve chosen, which is drawn under the panel as the hop plays

Click, drag or use the mouse wheel on the panel, or Tab to a setting and change it with Left/Right.
a/d orbit the camera, w/s tilt it, z/x zoom; r resets everything; q or Esc quits. The object's
turntable turns on a looping animation track (p pauses it).
"""
import argparse

import numpy as np

from .dice import make_die
from .hud import StatusBar
from .room import rainbow_blob, rainbow_text, tiles
from ..animation import EASINGS, RotationTrack, Track
from ..background import Fog, Gradient, Sky
from ..keys import Key
from ..mesh import Mesh, make_box
from ..scene import Camera, Light, Node, Object3D, PointLight, Renderer
from ..shapes import blob_mesh
from ..terminal import add_display_args, display_options, run
from ..transforms import UP, quat_axis_angle
from ..ui import Button, Choice, Panel, Slider, Toggle

TABLE_TOP = 0.16          # the turntable's top, where the object stands
COLOURS = {"white": (235, 235, 235), "red": (210, 50, 40), "gold": (230, 165, 35), "blue": (50, 100, 215),
           "green": (60, 175, 80), "as painted": (255, 255, 255)}
FOGS = {"off": None, "background": None, "mist": (225, 228, 232), "night": (5, 6, 12)}
BACKDROPS = {"sky": Sky(), "gradient": Gradient((70, 90, 140), (20, 20, 30)), "none": None}
HOP_UP, HOP_TIME, SPIN_TIME = 1.2, 0.9, 1.4  # how high and how long the hop is, and how long the spin takes
GRAPH_W, GRAPH_H = 26, 9  # the easing curve's graph, in cells


def grid_floor(half, n, texture):
    """A floor 2 * half across, facing up, in n x n squares (so that each gets its own mip level), showing all of
    `texture` once."""
    t = np.linspace(0.0, 1.0, n + 1)
    u, v = (a.ravel() for a in np.meshgrid(t, t))
    vertices = np.stack([(u - 0.5) * 2 * half, np.zeros(u.size), (0.5 - v) * 2 * half], axis=1)
    faces = []
    for j in range(n):
        for i in range(n):
            a, b, c, d = j * (n + 1) + i, j * (n + 1) + i + 1, (j + 1) * (n + 1) + i + 1, (j + 1) * (n + 1) + i
            faces += [(a, b, c), (a, c, d)]
    faces = np.array(faces)
    return Mesh(vertices, faces, np.stack([u, v], axis=1)[faces], np.zeros(len(faces), int), [texture])


def shapes():
    """The objects to choose from, each about 1.2 across and centred on the origin, as (mesh, painted): painted
    meshes carry their own colours, shown as they are with the colour "as painted"."""
    sign = rainbow_text("3D").normalized(1.4)
    return {"sphere": (blob_mesh((0.65, 0.65, 0.65), rings=32, segments=48), False),
            "cube": (make_box(1.1), False),
            "die": (make_die(1.1), True),
            "sign": (sign, True),
            "blob": (rainbow_blob().normalized(1.3), True)}


class Workshop:
    def __init__(self):
        self.shapes = shapes()
        self.renderer = Renderer(1, 1)
        self.bar = StatusBar(self.renderer)
        floor = grid_floor(60.0, 40, tiles(1024, (0.72, 0.7, 0.66), (0.42, 0.4, 0.38), 120))
        self.scenery = [Object3D(floor, color=(255, 255, 255), specular=0.0)]
        for z in range(-6, -80, -6):  # two rows of pillars running into the distance, for the fog to take
            for x in (-4.0, 4.0):
                self.scenery.append(Object3D(make_box(), np.array([x, 1.25, float(z)]), scale=(0.6, 2.5, 0.6),
                                             color=(170, 120, 90) if x < 0 else (100, 130, 170), specular=0.2))
        # The turntable: a sphere squashed flat, turning on a looping track; the object stands on it.
        self.table = Node()
        self.scenery.append(Object3D(blob_mesh((1.0, 1.0, 1.0), rings=16, segments=48),
                                     np.array([0.0, TABLE_TOP / 2, 0.0]), scale=(1.5, TABLE_TOP / 2, 1.5),
                                     color=(90, 90, 100), specular=0.6, parent=self.table))
        self.turning = RotationTrack([(8.0 * k / 3, quat_axis_angle(UP, 2 * np.pi * k / 3)) for k in range(4)],
                                     loop="loop")
        self.obj = Object3D(self.shapes["sphere"][0], parent=self.table)

        self.shape = Choice("Object (o)", tuple(self.shapes), key="o")
        self.colour = Choice("Colour (c)", tuple(COLOURS), key="c")
        self.specular = Slider("Specular ", 1.0, 0.0, 3.0, step=0.1, length=12, fmt=lambda v: f"{v:.1f}")
        self.shininess = Slider("Shininess", 0, 0, 200, step=5, length=12,
                                fmt=lambda v: "light's" if v == 0 else f"{v:g}")
        self.reflect = Slider("Reflect  ", 0.0, 0.0, 1.0, step=0.05, length=12, fmt=lambda v: f"{v:.2f}")
        self.opacity = Slider("Opacity  ", 1.0, 0.05, 1.0, step=0.05, length=12, fmt=lambda v: f"{v:.2f}")
        self.sx, self.sy, self.sz = (Slider(f"Scale {a}  ", 1.0, -2.0, 2.0, step=0.1, length=12,
                                            fmt=lambda v: f"{v:+.1f}") for a in "xyz")
        self.fog = Choice("Fog (f)", tuple(FOGS), key="f")
        self.fog_start = Slider("Fog start", 6, 0, 40, step=1, length=12)
        self.fog_end = Slider("Fog end  ", 40, 1, 120, step=1, length=12)
        self.backdrop = Choice("Backdrop (b)", tuple(BACKDROPS), key="b")
        self.easing = Choice("Easing (e)", tuple(EASINGS), key="e", value="ease_out_back")
        self.spin = Toggle("Turntable (p)", True, key="p")
        self.panel = Panel([self.shape, self.colour, self.specular, self.shininess, self.reflect, self.opacity,
                            self.sx, self.sy, self.sz, self.fog, self.fog_start, self.fog_end, self.backdrop,
                            self.easing, self.spin, Button("Hop (h)", self.hop, key="h"),
                            Button("Reset (r)", self.reset, key="r")], vertical=True)
        self.defaults = [(w, w.value) for w in self.panel.widgets if not isinstance(w, Button)]
        self.yaw, self.pitch, self.distance = 0.2, 0.3, 5.0
        self.clock = 0.0     # the turntable's time, which stops while it is paused
        self.hopping = None  # (time into the hop, height track, spin track) while a hop plays

    def reset(self):
        for widget, value in self.defaults:
            widget.value = value
        self.yaw, self.pitch, self.distance = 0.2, 0.3, 5.0

    def hop(self):
        """Throw the object up (easing out as it rises, in as it falls) and spin it once about the vertical, the
        spin shaped by the chosen easing: two Tracks, played together."""
        height = Track([(0.0, 0.0), (HOP_TIME / 2, HOP_UP, "ease_out"), (HOP_TIME, 0.0, "ease_in")])
        spin = Track([(0.0, 0.0), (SPIN_TIME, 2 * np.pi)], easing=self.easing.value)
        self.hopping = [0.0, height, spin]

    def place(self, dt):
        """Set the object from the panel, and move it: the turntable, and a hop if one is playing."""
        mesh, painted = self.shapes[self.shape.value]
        obj = self.obj
        obj.mesh = mesh
        colour = self.colour.value
        obj.color = COLOURS[colour] if colour != "as painted" or painted else COLOURS["white"]
        obj.specular, obj.reflectivity, obj.opacity = self.specular.value, self.reflect.value, self.opacity.value
        obj.shininess = self.shininess.value or None
        scale = np.array([self.sx.value, self.sy.value, self.sz.value])
        obj.scale = scale
        # Stand it on the turntable, however tall it is scaled.
        lift = -mesh.vertices[:, 1].min() if scale[1] >= 0 else mesh.vertices[:, 1].max()
        height, angle = 0.0, 0.0
        if self.hopping is not None:
            self.hopping[0] += dt
            t, up, turn = self.hopping
            height, angle = up.at(t), turn.at(t)
            if t >= max(up.end, turn.end):
                self.hopping = None
        obj.position = np.array([0.0, TABLE_TOP + lift * abs(scale[1]) + height, 0.0])
        obj.rotation = quat_axis_angle(UP, angle)
        if self.spin.value:
            self.clock += dt
        self.table.rotation = self.turning.at(self.clock)

    def draw_graph(self, screen, y, x):
        """The chosen easing curve, from 0 (bottom) to 1 (top) as time runs left to right, with the spin's progress
        marked while a hop plays."""
        ease = EASINGS[self.easing.value]
        values = [ease(i / (GRAPH_W - 1)) for i in range(GRAPH_W)]
        lo, hi = min(0.0, *values), max(1.0, *values)  # 0 to 1, and any overshoot
        rows = [[" "] * GRAPH_W for _ in range(GRAPH_H)]
        for i, v in enumerate(values):
            r = int(round((hi - v) / (hi - lo) * (GRAPH_H - 1)))
            rows[r][i] = "*" if not screen.unicode else "•"
        for r, v in ((int(round((hi - 1.0) / (hi - lo) * (GRAPH_H - 1))), "1"),
                     (int(round((hi - 0.0) / (hi - lo) * (GRAPH_H - 1))), "0")):
            screen.text(y + 1 + r, x, v, dim=True)
        screen.text(y, x, f"{self.easing.value}:"[:GRAPH_W + 2], dim=True)
        for r, row in enumerate(rows):
            screen.text(y + 1 + r, x + 2, "".join(row))
        if self.hopping is not None:
            i = min(int(self.hopping[0] / SPIN_TIME * (GRAPH_W - 1)), GRAPH_W - 1)
            screen.text(y + 1 + GRAPH_H, x + 2 + i, "^")

    def frame(self, screen, dt, events):
        dt = min(dt, 0.1)
        for ev in self.panel.handle(self.bar.handle(events, screen)):
            if ev in (ord("q"), Key.ESC):
                return False
            if ev == ord(" "):
                self.hop()
            elif ev in (ord("a"), ord("d")):
                self.yaw += 0.1 if ev == ord("d") else -0.1
            elif ev in (ord("w"), ord("s")):
                self.pitch = float(np.clip(self.pitch + (0.06 if ev == ord("w") else -0.06), -0.1, 1.4))
            elif ev in (ord("z"), ord("x")):
                self.distance = float(np.clip(self.distance * (0.9 if ev == ord("z") else 1.1), 2.0, 20.0))
        self.place(dt)

        fog = self.fog.value
        start, end = self.fog_start.value, max(self.fog_end.value, self.fog_start.value + 1)
        self.renderer.fog = 0.0 if fog == "off" else Fog(start, end, FOGS[fog])
        self.renderer.background = BACKDROPS[self.backdrop.value]
        target = np.array([0.0, 0.8, 0.0])
        eye = target + self.distance * np.array([np.sin(self.yaw) * np.cos(self.pitch), np.sin(self.pitch),
                                                 np.cos(self.yaw) * np.cos(self.pitch)])
        camera = Camera(position=eye, target=target, fov=50.0, near=0.05, far=200.0)
        lights = [Light(direction=np.array([0.4, -1.0, -0.5]), ambient=0.25, diffuse=0.6, shadows=True),
                  PointLight(np.array([2.5, 3.0, 2.5]), color=(255, 225, 190), diffuse=0.5, range=12.0)]

        # The view takes the screen right of the panel (and above the status line).
        rows, cols = screen.size()
        left = min(self.panel.width + 3, max(cols - 10, 0))
        self.renderer.resize(max(cols - left, 1), max(rows - 1, 1), screen.cell_pixels)
        fb = self.renderer.render([*self.scenery, self.obj], camera, lights)
        screen.erase()
        screen.draw_frame(fb, 0, left)
        self.panel.draw(screen, 0, 1)
        if rows - 1 - self.panel.height >= GRAPH_H + 3:
            self.draw_graph(screen, self.panel.height + 1, 1)
        self.bar.draw(screen, "[tab/click] settings [h] hop [a/d w/s z/x] camera [r] reset [q] quit")
        screen.refresh()
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fps", type=int, default=30)
    add_display_args(parser)
    args = parser.parse_args()
    try:
        run(Workshop().frame, args.fps, mouse="drag", title="unicode3d workshop", **display_options(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
