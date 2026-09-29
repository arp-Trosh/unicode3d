# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
"""Frame-time benchmark: python benchmarks/bench.py [--size 180x50] [--frames 30] [--glyphs sextant] [--color truecolor]

Renders a few fixed, spinning scenes off-screen and reports the time each stage
of a frame takes:

  render          Renderer.render: rasterizing and shading the scene into pixels (and shadow maps)
  draw_frame      Screen.draw_frame: pixels into glyphs and terminal colours
  render_updates  Screen.render_updates: the changed cells as escape sequences

The first frames, which compile the renderer's kernels, are timed separately
and left out of the averages. Nothing is written to the terminal, so the time
the terminal takes to show a frame is not included.
"""
import argparse
import statistics
import time

import numpy as np

from unicode3d.color import Color
from unicode3d.examples.dice import make_die, orientation_showing
from unicode3d.scene import Camera, Light, Object3D, PointLight, Renderer
from unicode3d.mesh import Mesh
from unicode3d.shapes import blob_mesh
from unicode3d.terminal import Screen
from unicode3d.transforms import normalize, quat_axis_angle, quat_mul

STAGES = ("render", "draw_frame", "render_updates")
SPIN = quat_axis_angle([0.3, 1.0, 0.2], 0.05)


def dice_scene(count=3, shadows=None, glass=False, polished=False):
    """The dice demo's view: a row of textured dice seen from above and in front; with shadows ("sun" or
    "lamp"), on a floor (which keeps still) that they cast them onto, redrawing the shadow map every frame;
    with glass, see-through dice (casting tinted shadows); polished, on a mirror of a floor."""
    mesh = make_die()
    colors = (Color.GREEN, Color.YELLOW, Color.RED)
    xs = (np.arange(count) - (count - 1) / 2) * 1.8
    objects = [Object3D(mesh, np.array([x, 0.5, 0.0]), orientation_showing(i % 6, 0.4 * i), color=colors[i % 3],
                        opacity=0.45 if glass else 1.0) for i, x in enumerate(xs)]
    target = np.array([0.0, 0.4, 0.0])
    camera = Camera(position=target + normalize([0.0, 0.8, 0.6]) * 7.0, target=target, fov=35.0)
    if not shadows:
        return objects, camera, Light()
    top = Mesh(np.array([(-4, -0.5, -2.5), (-4, -0.5, 2.5), (4, -0.5, 2.5), (4, -0.5, -2.5)], float),
               np.array([(0, 1, 2), (0, 2, 3)]))
    floor = Object3D(top, color=(200, 200, 200), reflectivity=0.4 if polished else 0.0)
    light = (Light(shadows=True) if shadows == "sun" else
             [Light(ambient=0.2, diffuse=0.2), PointLight(np.array([0.5, 2.5, 1.0]), range=8.0, shadows=True)])
    return objects + [floor], camera, light, [floor]


def sphere_scene(rings, segments):
    """One smooth-shaded sphere, filling most of the view."""
    obj = Object3D(blob_mesh((1.0, 1.0, 1.0), rings=rings, segments=segments), color=(200, 80, 60))
    return [obj], Camera(position=np.array([0.0, 0.0, 4.0])), Light()


def balls_scene(count):
    """Many small smooth-shaded balls in different colours: the cost of each object."""
    rng = np.random.default_rng(1)
    mesh = blob_mesh((1.0, 1.0, 1.0), rings=12, segments=16)
    objects = [Object3D(mesh, rng.uniform(-4, 4, 3), scale=rng.uniform(0.2, 0.6),
                        color=tuple(int(c) for c in rng.integers(40, 255, 3))) for _ in range(count)]
    return objects, Camera(position=np.array([0.0, 2.0, 12.0]), fov=45.0), Light()


SCENES = {
    "dice": lambda: dice_scene(3),
    "dice-sun": lambda: dice_scene(3, shadows="sun"),
    "dice-lamp": lambda: dice_scene(3, shadows="lamp"),
    "dice-glass": lambda: dice_scene(3, shadows="sun", glass=True),
    "dice-mirror": lambda: dice_scene(3, shadows="sun", polished=True),
    "balls-400": lambda: balls_scene(400),
    "sphere-3k": lambda: sphere_scene(32, 48),
    "sphere-27k": lambda: sphere_scene(96, 144),
}


class Frames:
    """A scene, a renderer and an off-screen screen; step() spins the objects and draws one frame."""

    def __init__(self, scene, cols, rows, glyphs, color):
        self.objects, self.camera, self.light, *still = SCENES[scene]()
        self.moving = [obj for obj in self.objects if not any(obj is s for s in (still or [[]])[0])]
        self.screen = Screen(None, glyphs=glyphs, color=color, size=(rows, cols))
        self.renderer = Renderer(cols, rows, self.screen.cell_pixels)

    def step(self):
        for obj in self.moving:
            obj.rotation = quat_mul(SPIN, obj.rotation)
        t0 = time.perf_counter()
        fb = self.renderer.render(self.objects, self.camera, self.light)
        t1 = time.perf_counter()
        self.screen.draw_frame(fb)
        t2 = time.perf_counter()
        out = self.screen.render_updates()
        t3 = time.perf_counter()
        return (t1 - t0, t2 - t1, t3 - t2), len(out.encode())


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--size", default="180x50", help="terminal size in cells, COLSxROWS (default 180x50)")
    parser.add_argument("--frames", type=int, default=30, help="frames timed per scene (default 30)")
    parser.add_argument("--glyphs", default="sextant", choices=("half", "quad", "sextant", "ascii"))
    parser.add_argument("--color", default="truecolor", choices=("truecolor", "256", "16", "mono"))
    parser.add_argument("--scene", action="append", choices=SCENES, help="scene to run (default: all)")
    args = parser.parse_args()
    cols, rows = (int(v) for v in args.size.lower().split("x"))

    print(f"{cols}x{rows} cells, {args.glyphs} glyphs, {args.color}; median ms per frame")
    print(f"{'scene':12} {'first':>7} " + " ".join(f"{s:>15}" for s in STAGES) + f" {'total':>8} {'fps':>6} {'KB out':>7}")
    for name in args.scene or SCENES:
        frames = Frames(name, cols, rows, args.glyphs, args.color)
        t0 = time.perf_counter()
        frames.step()
        first = time.perf_counter() - t0
        for _ in range(2):
            frames.step()
        times, sizes = [], []
        for _ in range(args.frames):
            t, size = frames.step()
            times.append(t)
            sizes.append(size)
        stage_ms = [1000 * statistics.median(t[i] for t in times) for i in range(len(STAGES))]
        total = 1000 * statistics.median(sum(t) for t in times)
        print(f"{name:12} {1000 * first:7.0f} " + " ".join(f"{ms:15.2f}" for ms in stage_ms)
              + f" {total:8.2f} {1000 / total:6.1f} {statistics.median(sizes) / 1024:7.1f}")


if __name__ == "__main__":
    main()
