# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
"""Stress test, to run by hand before a release: python -m benchmarks.stress [--frames 2000] [--seed 0]

Two parts, both off-screen, stopping at the first exception with what reproduces it:

  walk    the room demo, walking and turning at random, pressing its keys (display settings too)
          and clicking, while the terminal changes size every 40 frames among sizes from a single
          cell to a full screen at a tiny font (--sizes)
  close   the courtyard drawn from random cameras, most of them right up against a surface,
          where clipping at the near plane makes the most extreme triangles

It prints the frame times and the peak memory. The unit tests cover each of these briefly;
this runs them for long enough to find rare trouble.
"""
import argparse
import statistics
import sys
import time
import traceback

import numpy as np

from unicode3d.background import Fog
from unicode3d.examples.room import HALF, Walk
from unicode3d.keys import Key, MouseEvent
from unicode3d.scene import Camera, Light, PointLight
from unicode3d.terminal import Screen, compile_kernels

try:
    import resource
except ImportError:  # Windows
    resource = None

DEFAULT_SIZES = "215x960,24x80,270x1100,1x1,2x3,540x1920,216x961,50x180"
KEYS = [Key.F2, Key.F3, Key.F5, Key.F6, ord("l"), ord("u"), ord("b"), ord("["), ord("]")]
HELD = ["w", "a", "s", "d", "q", "e", Key.LEFT, Key.RIGHT, Key.UP, Key.DOWN]


class Console:
    """A console of a given size that takes no input and throws the output away."""
    unicode, key_release, reports_releases = True, False, False

    def __init__(self):
        self.rows, self.cols = 24, 80

    def size(self):
        return self.cols, self.rows

    def read(self):
        return ""

    def key_is_down(self, key):
        return None

    def write(self, data):
        pass


def peak_mb():
    if resource is None:
        return float("nan")
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 if sys.platform != "darwin" else 1024 * 1024)


def walk(frames, rng, sizes):
    demo, console = Walk(int(rng.integers(1 << 30))), Console()
    screen = Screen(console, glyphs="sextant", color="truecolor")
    times = []
    for i in range(frames):
        if i % 40 == 0:
            console.rows, console.cols = sizes[(i // 40) % len(sizes)]
        screen.poll_size()
        rows, cols = screen.size()
        events = []
        if rng.random() < 0.05:
            events.append(KEYS[rng.integers(len(KEYS))])
        elif rng.random() < 0.05:
            events.append(MouseEvent(int(rng.integers(cols + 5)), int(rng.integers(rows + 5)), MouseEvent.LEFT, True))
        screen.held.exact = True
        screen.held.release_all()
        screen.held.update([ord(k) if isinstance(k, str) else k for k in
                            (HELD[j] for j in rng.integers(0, len(HELD), int(rng.integers(0, 3))))], 0.0)
        t = time.perf_counter()
        try:
            demo.frame(screen, float(rng.uniform(0.01, 0.1)), events)
        except Exception:
            traceback.print_exc()
            sys.exit(f"walk: frame {i} at {cols}x{rows} cells, {screen.mode}, {screen.color_mode}, events {events}, "
                     f"held {screen.held.keys()}, at x={demo.x!r} z={demo.z!r} yaw={demo.yaw!r} pitch={demo.pitch!r}")
        times.append((time.perf_counter() - t, cols * rows))
    return times


def close(renders, rng):
    demo = Walk(0)
    court, renderer = demo.court, demo.renderer
    screen = Screen(glyphs="sextant", color="truecolor", size=(215, 960))
    renderer.resize(960, 215, screen.cell_pixels)
    lights = [Light(direction=np.array([0.5, -1.0, -0.35]), shadows=True),
              PointLight(court.lamp_at, range=9.0, shadows=True)]
    points = np.concatenate([p + (linear @ obj.mesh.vertices.T).T
                             for obj in court.objects for linear, p, _ in [obj.world_matrix()]])
    times = []
    for i in range(renders):
        court.animate(float(rng.uniform(0, 100)))
        if rng.random() < 0.6:  # right up against a surface
            eye = points[rng.integers(len(points))] + rng.normal(size=3) * rng.choice([0.01, 0.05, 0.1, 0.3])
        else:
            eye = np.array([rng.uniform(-HALF, HALF), rng.uniform(0.1, 3.4), rng.uniform(-HALF, HALF)])
        look = rng.normal(size=3)
        camera = Camera(position=eye, target=eye + look, fov=70.0, near=0.05, far=100.0)
        renderer.background = demo.backgrounds[list(demo.backgrounds)[rng.integers(len(demo.backgrounds))]]
        if rng.random() < 0.5:  # depth cueing, or fog in the world (into the background, or a colour)
            renderer.fog = float(rng.uniform(0, 0.9))
        else:
            start = float(rng.uniform(0, 20))
            renderer.fog = Fog(start, start + float(rng.uniform(0, 40)),
                               None if rng.random() < 0.5 else tuple(int(c) for c in rng.integers(0, 256, 3)))
        renderer.shadows, renderer.reflections = bool(rng.random() < 0.8), bool(rng.random() < 0.8)
        t = time.perf_counter()
        try:
            screen.draw_frame(renderer.render(court.objects, camera, lights[:1 + int(rng.random() < 0.7)]))
            screen.render_updates()
            renderer.pick(480, 107)
        except Exception:
            traceback.print_exc()
            sys.exit(f"close: render {i}, camera at {eye.tolist()!r} looking along {look.tolist()!r}, "
                     f"shadows {renderer.shadows}, reflections {renderer.reflections}, fog {renderer.fog}, "
                     f"background {type(renderer.background).__name__}")
        times.append(time.perf_counter() - t)
    return times


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("--frames", type=int, default=2000, help="frames of each part (default 2000)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sizes", default=DEFAULT_SIZES, help=f"ROWSxCOLS,... (default {DEFAULT_SIZES})")
    parser.add_argument("--part", action="append", choices=("walk", "close"), help="part to run (default: both)")
    args = parser.parse_args()
    sizes = [tuple(int(v) for v in s.lower().split("x")) for s in args.sizes.split(",")]
    rng = np.random.default_rng(args.seed)
    compile_kernels()
    parts = args.part or ["walk", "close"]
    if "walk" in parts:
        t = time.perf_counter()
        times = walk(args.frames, rng, sizes)
        print(f"walk: {args.frames} frames in {time.perf_counter() - t:.0f} s")
        for cells in sorted({c for _, c in times}):
            ms = [1000 * s for s, c in times if c == cells]
            print(f"  {cells:>9} cells: median {statistics.median(ms):6.1f} ms, worst {max(ms):7.1f} ms")
    if "close" in parts:
        t = time.perf_counter()
        times = close(args.frames, rng)
        print(f"close: {args.frames} renders in {time.perf_counter() - t:.0f} s, median "
              f"{1000 * statistics.median(times):.1f} ms, worst {1000 * max(times):.1f} ms")
    print(f"peak memory {peak_mb():.0f} MB; no errors")


if __name__ == "__main__":
    main()
