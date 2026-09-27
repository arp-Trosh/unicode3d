# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
"""Dice rolling demo: python -m unicode3d.examples.demo [-n DICE] [--seed N]"""
import argparse

import numpy as np

from .dice import DIE_VALUES, RollAnimation, make_die, orientation_showing, top_face
from .hud import StatusBar
from ..scene import Camera, Light, Object3D, Renderer
from ..color import Color
from ..keys import Key
from ..terminal import add_display_args, display_options, run
from ..transforms import normalize

DIE_COLORS = (Color.GREEN, Color.YELLOW, Color.RED)
SPACING = 1.8
MAX_DICE = 6
HUD_ROWS = 2
CAMERA_DIR = normalize([0.0, 0.8, 0.6])  # from the table towards the camera


class DiceDemo:
    def __init__(self, count, seed=None):
        self.rng = np.random.default_rng(seed)
        self.mesh = make_die()
        self.renderer = Renderer(1, 1)
        self.camera = Camera(target=np.array([0.0, 0.4, 0.0]), fov=35.0)
        self.light = Light()
        self.bar = StatusBar()
        self.set_count(count)

    def set_count(self, n):
        xs = (np.arange(n) - (n - 1) / 2) * SPACING
        self.rests = [np.array([x, 0.5, 0.0]) for x in xs]
        self.dice = [
            Object3D(self.mesh, rest.copy(), orientation_showing(self.rng.integers(6), self.rng.uniform(0, 2 * np.pi)),
                     color=DIE_COLORS[i % len(DIE_COLORS)])
            for i, rest in enumerate(self.rests)
        ]
        self.anims = []
        self.t = 0.0

    def roll(self):
        self.anims = [RollAnimation(self.rng.integers(6), rest, self.rng) for rest in self.rests]
        self.t = 0.0

    def fit_camera(self, cols, rows):
        """Back the camera off until the whole row of dice fits the view."""
        tan_half = np.tan(np.radians(self.camera.fov) / 2)
        aspect = cols * self.renderer.cell_aspect / rows
        half_width = (len(self.dice) - 1) * SPACING / 2 + 1.0
        dist = max(1.8 / tan_half, 1.25 * half_width / (tan_half * aspect))
        self.camera.position = self.camera.target + CAMERA_DIR * dist

    def frame(self, screen, dt, keys):
        for k in self.bar.handle(keys, screen):
            if k in (ord("q"), ord("Q"), 27):
                return False
            if k in (ord(" "), ord("\n"), Key.ENTER):
                self.roll()
            elif k in (ord("+"), ord("=")):
                self.set_count(min(len(self.dice) + 1, MAX_DICE))
            elif k == ord("-"):
                self.set_count(max(len(self.dice) - 1, 1))

        self.t += dt
        for die, anim in zip(self.dice, self.anims):
            die.position, die.rotation = anim.pose(self.t)

        rows, cols = screen.size()
        view_rows = max(rows - HUD_ROWS, 1)
        self.renderer.resize(cols, view_rows, screen.cell_pixels)
        self.fit_camera(cols, view_rows)
        fb = self.renderer.render(self.dice, self.camera, self.light)

        screen.erase()
        screen.draw_frame(fb)
        if any(not a.done(self.t) for a in self.anims):
            status = "Rolling..."
        else:
            status = "Showing: " + "  ".join(str(DIE_VALUES[top_face(d.rotation)]) for d in self.dice)
        screen.text(rows - 2, 1, status, bold=True)
        self.bar.draw(screen, f"[space] roll   [+/-] dice ({len(self.dice)})   [q] quit")
        screen.refresh()
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", "--dice", type=int, default=3, help=f"number of dice (1-{MAX_DICE})")
    parser.add_argument("--seed", type=int, help="random seed")
    parser.add_argument("--fps", type=int, default=30)
    add_display_args(parser)
    args = parser.parse_args()
    demo = DiceDemo(min(max(args.dice, 1), MAX_DICE), args.seed)
    try:
        run(demo.frame, args.fps, mouse=True, title="unicode3d dice", **display_options(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
