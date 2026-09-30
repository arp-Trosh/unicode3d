# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
"""The pictures in how-it-works.md, from the gallery's scenes, and the README's screenshots of the demos:
python -m docs.make_figures [DIR]

Writes PNGs into DIR (default docs/images). Each shows the terminal cells as a terminal draws them
(benchmarks.gallery.cells_picture), except where the point is the pixels behind them.
"""
import dataclasses
import os
import sys

import numpy as np

from benchmarks.gallery import CELL, SCENES, cells_picture, checks, pixels_picture, render, write_png
from unicode3d.mesh import Mesh
from unicode3d.scene import Object3D
from unicode3d.terminal import Screen

GAP = 8  # pixels between panels
GREY = 128


def crop(frame, x0, y0, cols, rows):
    """The cells picture of a region of cells."""
    w, h = CELL
    return cells_picture(frame)[y0 * h:(y0 + rows) * h, x0 * w:(x0 + cols) * w]


def side_by_side(*images):
    height = max(im.shape[0] for im in images)
    parts = []
    for im in images:
        pad = np.full((height - im.shape[0], im.shape[1], 3), GREY, np.uint8)
        parts += [np.concatenate([im, pad]), np.full((height, GAP, 3), GREY, np.uint8)]
    return np.concatenate(parts[:-1], axis=1)


def stacked(*images):
    width = max(im.shape[1] for im in images)
    parts = []
    for im in images:
        pad = np.full((im.shape[0], width - im.shape[1], 3), GREY, np.uint8)
        parts += [np.concatenate([im, pad], axis=1), np.full((GAP, width, 3), GREY, np.uint8)]
    return np.concatenate(parts[:-1])


def zoom(image, k):
    return image.repeat(k, axis=0).repeat(k, axis=1)


def resize(image, height, width):
    """Nearest-neighbour resampling to height x width."""
    ys, xs = np.arange(height) * image.shape[0] // height, np.arange(width) * image.shape[1] // width
    return image[ys][:, xs]


def grid_floor(half, n, texture):
    """A floor 2 * half across, facing up, of n x n squares, showing all of `texture` once."""
    t = np.linspace(0.0, 1.0, n + 1)
    u, v = np.meshgrid(t, t)
    vertices = np.stack([(u.ravel() - 0.5) * 2 * half, np.zeros(u.size), (0.5 - v.ravel()) * 2 * half], axis=1)
    faces, uvs = [], []
    for j in range(n):
        for i in range(n):
            a, b, c, d = j * (n + 1) + i, j * (n + 1) + i + 1, (j + 1) * (n + 1) + i + 1, (j + 1) * (n + 1) + i
            for tri in ((a, b, c), (a, c, d)):
                faces.append(tri)
                uvs.append([(u.ravel()[k], v.ravel()[k]) for k in tri])
    return Mesh(vertices, np.array(faces), np.array(uvs), np.zeros(len(faces), int), [texture])


def shot_with(name, **settings):
    shot = SCENES[name]()
    return dataclasses.replace(shot, settings={**shot.settings, **settings})


def demo_view(demo, frames, cols=150, rows=48, dt=1 / 30):
    """Run a demo (anything with frame(screen, dt, events) and a renderer) off-screen for some frames, and picture
    its 3D view as the terminal shows it, without the demo's text (panels, status line) drawn over it."""
    screen = Screen(None, glyphs="sextant", color="truecolor", size=(rows, cols))
    screen.refresh = lambda: None
    for _ in range(frames):
        demo.frame(screen, dt, [])
    fb = demo.renderer.framebuffer
    view = Screen(None, glyphs="sextant", color="truecolor", size=(fb.height // 3, fb.width // 2))
    view.draw_frame(fb)
    return cells_picture({"chars": view.chars.view(np.uint32), "fg": view.fg, "bg": view.bg,
                          "glyphs": np.array("sextant")})


def screenshots(save):
    """The README's gallery: the demos at their best, 150x48 cells."""
    from unicode3d.examples.balls import Balls
    from unicode3d.examples.demo import DiceDemo
    from unicode3d.examples.maze import Maze
    from unicode3d.examples.room import Walk
    for name, x, z, yaw, pitch in (("screenshot-room", 1.0, 3.0, -0.75, -0.12),
                                   ("screenshot-room-garden", -1.0, -4.0, 2.6, -0.1),
                                   ("screenshot-room-bench", 7.1, -5.2, 0.08, -0.3)):
        walk = Walk(1)
        walk.x, walk.z, walk.yaw, walk.pitch = x, z, yaw, pitch
        save(name, demo_view(walk, 2))
    dice = DiceDemo(3, seed=3)
    dice.roll()
    dice.table.reflectivity = 0.5  # polished (m)
    save("screenshot-dice", demo_view(dice, 90))
    balls = Balls(160, seed=2)
    balls.glass.value = balls.mirror.value = True
    balls.room_size.value = 7.0
    save("screenshot-balls", demo_view(balls, 60))
    maze = Maze(10, seed=4)
    maze.show_map.value, maze.polished.value = False, True
    save("screenshot-maze", demo_view(maze, 165))  # a frame looking down a corridor
    from unicode3d.examples.workshop import Workshop
    shop = Workshop()
    shop.shape.value, shop.colour.value, shop.specular.value, shop.shininess.value = "die", "gold", 2.5, 120
    shop.sx.value, shop.sy.value = 1.5, 0.7
    shop.fog.value, shop.fog_start.value, shop.fog_end.value = "mist", 3, 30
    save("screenshot-workshop", demo_view(shop, 2))


def main(out):
    os.makedirs(out, exist_ok=True)
    frames = {name: render(SCENES[name]()) for name in SCENES}
    save = lambda name, image: write_png(os.path.join(out, name + ".png"), image)

    screenshots(save)

    # Whole scenes, as a terminal shows them.
    for name in ("courtyard", "glass", "sun-shadows", "lamp-shadows", "cutouts", "mirror", "facing-mirrors", "fog",
                 "materials", "stretched", "balls"):
        save(name, cells_picture(frames[name]))

    # The same dice in each glyph set.
    region = (18, 7, 64, 22)
    save("glyph-sets", stacked(side_by_side(crop(frames["die-half"], *region), crop(frames["die-quad"], *region)),
                               side_by_side(crop(frames["die"], *region), crop(frames["die-ascii"], *region))))

    # Pixels (2 x 3 to a cell in sextant mode, as tall as the terminal makes them) and the cells chosen for them,
    # close up, with the cells' borders drawn over the pixels.
    x0, y0, cols, rows = 50, 12, 14, 8
    die, k = frames["die"], 4
    w, h = CELL[0] * k, CELL[1] * k
    pixels = resize(pixels_picture(die)[y0 * 3:(y0 + rows) * 3, x0 * 2:(x0 + cols) * 2], rows * h, cols * w)
    pixels[::h, :] = GREY
    pixels[:, ::w] = GREY
    save("pixels-to-cells", side_by_side(pixels, zoom(crop(die, x0, y0, cols, rows), k)))

    # Truecolor, 256 and 16 colours.
    region = (30, 7, 40, 22)
    save("colours", side_by_side(*(crop(frames[n], *region) for n in ("die", "die-256", "die-16"))))

    # One sample a pixel against 4 (and 8 more at edges), in pixels, close up.
    plain = render(shot_with("cube", samples=1, edge_samples=0, outline=0.0))
    smooth = render(shot_with("cube", outline=0.0))
    rows, cols = slice(26, 62), slice(76, 120)
    save("antialiasing", side_by_side(zoom(pixels_picture(plain)[rows, cols], 8),
                                      zoom(pixels_picture(smooth)[rows, cols], 8)))

    # A finely checked floor to the horizon without mipmaps (always the finest level) and with them. The mip level
    # is chosen per triangle, so the floor is divided into many (the gallery's is two, which get one level each).
    fine = dataclasses.replace(SCENES["horizon"](), objects=[Object3D(grid_floor(40.0, 64, checks(1024, 160)),
                                                                     color=(255, 255, 255))])
    sharp = render(dataclasses.replace(fine, settings={**fine.settings, "lod_bias": -30.0}))
    region = (0, 10, 100, 16)
    save("mipmaps", stacked(crop(sharp, *region), crop(render(fine), *region)))
    print(f"figures written to {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "images"))
