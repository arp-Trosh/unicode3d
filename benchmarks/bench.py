# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
"""Frame-time benchmark: python -m benchmarks.bench [--size 180x50] [--frames N] [--glyphs sextant] [--color truecolor]
[--threads N] [--scene NAME]

Renders scenes off-screen and reports the time each stage of a frame takes:
a few fixed, spinning set pieces (dice ... sphere-27k), and three that load the
engine as kinds of game would (corridor, arena, board: see genres.py), each
moving its world as a game loop does and writing text over the picture:

  render          Renderer.render: rasterizing and shading the scene into pixels (and shadow maps)
  draw_frame      Screen.draw_frame: pixels into glyphs and terminal colours
  render_updates  Screen.render_updates: the changed cells as escape sequences
  game            (the genre scenes) the game's own update (characters' clips, particles) and its text

Run the genre scenes at a usual window and at a full screen with a small font
(--size 480x130, say) too: they are where per-pixel costs show.

It starts with what it runs on (CPU, threads, versions), as times mean little
without it, and compile_kernels() (compiling the kernels, or loading them from
Numba's cache). Each scene's first frame, which sets up its buffers (and, in a
scene with shadows, draws the maps the first time), is timed separately and left
out of the medians. Nothing is written to the terminal, so the time the terminal
takes to show a frame is not included.
"""
import argparse
import os
import platform
import statistics
import subprocess
import sys
import time

import numba
import numpy as np

from unicode3d.color import Color
from unicode3d.examples.dice import make_die, orientation_showing
from unicode3d.scene import Camera, Light, Node, Object3D, PointLight, Renderer
from unicode3d.mesh import Mesh, make_box
from unicode3d.shapes import blob_mesh
from unicode3d import __version__
from unicode3d.terminal import Screen, compile_kernels
from unicode3d.transforms import normalize, quat_axis_angle, quat_mul

from .genres import DT, GENRES

STAGES = ("render", "draw_frame", "render_updates", "game")
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


def crowd_scene(count, parts=60, chain=6):
    """count jointed figures of `parts` small boxes each, in chains of `chain` hanging from each figure's root Node
    (as a glTF character's limbs do), every joint turning each frame: the cost of a deep scene graph."""
    mesh = make_box(0.25)
    objects = []
    for f in range(count):
        root = Node(position=np.array([(f % 5) * 2.0 - 4.0, 0.0, (f // 5) * 2.0 - 3.0]))
        for p in range(parts):
            parent = root if p % chain == 0 else objects[-1]
            objects.append(Object3D(mesh, np.array([0.0, 0.3, 0.0]) if p % chain else np.zeros(3),
                                    quat_axis_angle([np.cos(p), 0.5, np.sin(p)], 0.4 + 0.1 * (p // chain)),
                                    scale=0.95 if p % 2 else (1.0, 0.9, 1.0), color=(200, 120 + p, 60),
                                    parent=parent))
    return objects, Camera(position=np.array([0.0, 6.0, 10.0]), fov=50.0), Light()


def spawn_scene(count=60, textures=6):
    """count boxes showing a few 256x256 textures between them (as copies of a loaded model do), and a puff of dust
    (a new mesh each time) there one frame and gone the next: the cost of a render list that gains and loses
    objects, which repacks the meshes every frame."""
    rng = np.random.default_rng(2)
    images = [rng.uniform(0.0, 1.0, (256, 256, 3)) for _ in range(textures)]
    objects = [Object3D(make_box(0.8, [images[(b + k) % textures] for k in range(6)]),
                        np.array([(b % 10) * 1.2 - 5.4, 0.0, (b // 10) * 1.2 - 3.0]), color=(255, 255, 255))
               for b in range(count)]

    def churn(frame):  # (the objects to draw this frame)
        if frame % 2:
            return objects
        puff = Object3D(blob_mesh((0.3, 0.3, 0.3), rings=6, segments=8), np.array([0.0, 1.0, 0.0]),
                        color=(200, 200, 200))
        return objects + [puff]
    return objects, Camera(position=np.array([0.0, 7.0, 9.0]), fov=50.0), Light(), [], churn


SCENES = {
    "dice": lambda: dice_scene(3),
    "dice-sun": lambda: dice_scene(3, shadows="sun"),
    "dice-lamp": lambda: dice_scene(3, shadows="lamp"),
    "dice-glass": lambda: dice_scene(3, shadows="sun", glass=True),
    "dice-mirror": lambda: dice_scene(3, shadows="sun", polished=True),
    "balls-400": lambda: balls_scene(400),
    "crowd-20": lambda: crowd_scene(20),
    "spawn-60": lambda: spawn_scene(60),
    "sphere-3k": lambda: sphere_scene(32, 48),
    "sphere-27k": lambda: sphere_scene(96, 144),
    **GENRES,
}


def cpu_name():
    """The processor's model name, as the system gives it ("" if it can't be found)."""
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        elif sys.platform == "darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True,
                                  timeout=5).stdout.strip()
        elif sys.platform == "win32":
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor()


def machine():
    """Lines describing what the benchmark runs on (after the kernels have run, for the threading layer)."""
    try:
        layer = numba.threading_layer()
    except ValueError:
        layer = "not started"
    return [f"CPU:      {cpu_name() or 'unknown'} ({os.cpu_count()} logical cores)",
            f"Threads:  {numba.get_num_threads()} (Numba's {layer} threading layer)",
            f"System:   {platform.system()} {platform.release()} ({platform.machine()})",
            f"Software: Python {platform.python_version()}, NumPy {np.__version__}, Numba {numba.__version__}, "
            f"unicode3d {__version__}"]


class Frames:
    """A scene, a renderer and an off-screen screen; step() moves the scene on (spins the objects, or runs a genre
    scene's update) and draws one frame."""

    def __init__(self, scene, cols, rows, glyphs, color):
        spec = SCENES[scene]()
        self.game = None if isinstance(spec, tuple) else spec  # (a genres.Game)
        options = {}
        if self.game:
            self.objects, options, self.frames = spec.objects, spec.renderer_options, spec.frames
        else:
            self.objects, self.camera, self.light, *rest = spec
            still = rest[0] if rest else []
            self.churn = rest[1] if len(rest) > 1 else None  # (frame number: the objects to draw then)
            self.moving = [obj for obj in self.objects if not any(obj is s for s in still)]
            self.frames = 30
        self.frame = 0
        self.screen = Screen(None, glyphs=glyphs, color=color, size=(rows, cols))
        self.renderer = Renderer(cols, rows, self.screen.cell_pixels, **options)
        self.triangles = sum(len(obj.mesh.faces) for obj in self.objects)

    def step(self):
        if self.game:
            t0 = time.perf_counter()
            objects, camera, lights = self.game.update(DT)
            t1 = time.perf_counter()
            fb = self.renderer.render(objects, camera, lights)
            t2 = time.perf_counter()
            self.screen.draw_frame(fb)
            t3 = time.perf_counter()
            self.game.hud(self.screen, self.renderer)
            t4 = time.perf_counter()
            out = self.screen.render_updates()
            t5 = time.perf_counter()
            return (t2 - t1, t3 - t2, t5 - t4, (t1 - t0) + (t4 - t3)), len(out.encode())
        for obj in self.moving:
            obj.rotation = quat_mul(SPIN, obj.rotation)
        self.frame += 1
        objects = self.churn(self.frame) if self.churn else self.objects
        t0 = time.perf_counter()
        fb = self.renderer.render(objects, self.camera, self.light)
        t1 = time.perf_counter()
        self.screen.draw_frame(fb)
        t2 = time.perf_counter()
        out = self.screen.render_updates()
        t3 = time.perf_counter()
        return (t1 - t0, t2 - t1, t3 - t2, 0.0), len(out.encode())


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--size", default="180x50", help="terminal size in cells, COLSxROWS (default 180x50)")
    parser.add_argument("--frames", type=int, help="frames timed per scene (default 30; the genre scenes 300)")
    parser.add_argument("--glyphs", default="sextant", choices=("half", "quad", "sextant", "ascii"))
    parser.add_argument("--color", default="truecolor", choices=("truecolor", "256", "16", "mono"))
    parser.add_argument("--threads", type=int, help="threads the kernels run on (default: one per logical core)")
    parser.add_argument("--scene", action="append", choices=SCENES, help="scene to run (default: all)")
    args = parser.parse_args()
    cols, rows = (int(v) for v in args.size.lower().split("x"))
    if args.threads:
        numba.set_num_threads(args.threads)

    t0 = time.perf_counter()
    compile_kernels()
    ready = time.perf_counter() - t0
    print("\n".join(machine()))
    print(f"Kernels:  compiled or loaded from the cache in {ready:.1f} s")
    print(f"Frames:   {cols}x{rows} cells, {args.glyphs} glyphs, {args.color}; median of "
          f"{args.frames or '30 (genre scenes: 300)'} frames, ms\n")
    print(f"{'scene':12} {'objects':>7} {'tris':>7} {'first':>7} " + " ".join(f"{s:>15}" for s in STAGES)
          + f" {'total':>8} {'p90':>7} {'fps':>6} {'KB out':>7}")
    for name in args.scene or SCENES:
        frames = Frames(name, cols, rows, args.glyphs, args.color)
        t0 = time.perf_counter()
        frames.step()
        first = time.perf_counter() - t0
        for _ in range(2):
            frames.step()
        times, sizes = [], []
        for _ in range(args.frames or frames.frames):
            t, size = frames.step()
            times.append(t)
            sizes.append(size)
        stage_ms = [1000 * statistics.median(t[i] for t in times) for i in range(len(STAGES))]
        total = 1000 * statistics.median(sum(t) for t in times)
        p90 = 1000 * statistics.quantiles([sum(t) for t in times], n=10)[-1]
        print(f"{name:12} {len(frames.objects):7d} {frames.triangles:7d} {1000 * first:7.1f} "
              + " ".join(f"{ms:15.2f}" for ms in stage_ms)
              + f" {total:8.2f} {p90:7.2f} {1000 / total:6.1f} {statistics.median(sizes) / 1024:7.1f}")


if __name__ == "__main__":
    main()
