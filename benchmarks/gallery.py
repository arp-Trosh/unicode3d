# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
"""Reference scenes, for seeing what a change does to the picture: python -m benchmarks.gallery COMMAND

  save DIR               render every scene into DIR: its framebuffer and terminal cells (NAME.npz), and a
                         picture of the cells as a terminal shows them (NAME.png)
  compare OLD NEW        compare two sets saved by `save`, scene by scene; pictures of the scenes that changed
                         (old, new, and the difference brightened) go to NEW/diff
  diff [REV]             render the scenes with the engine as of git revision REV (default HEAD, in a
                         temporary worktree) and as it is in this checkout (uncommitted changes included),
                         and compare them

Options: --scene NAME (repeatable) picks scenes; --tolerance N (for compare and diff, default 1) is how many
8-bit sRGB levels a colour may move before it counts as changed. compare and diff exit with status 1 if any
scene changed.

Every scene is drawn from fixed poses at a fixed size (100x36 cells), and frames are the same on any number of
threads, so two renders of the same engine match exactly. A change meant to keep the picture should come out
"identical" (or "within tolerance", for rounding); one meant to change it shows where. The engine as of an older
revision compiles its kernels afresh (about half a minute), and scenes using features it lacks are skipped.

The scenes: a cube; a textured die in each glyph set and colour mode; overlapping glass; sun and lamp shadows;
cut-out and stained-glass textures; a mirror; facing mirrors; a textured floor running to the horizon
(mipmapping); many small balls; a finely divided sphere; the room demo's courtyard; and, added after 0.4.1,
world fog, materials, and shapes stretched unevenly; and, after 0.5.0, a model loaded from an OBJ file on a
floor whose texture repeats, and fog over a starry sky box.
"""
import argparse
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import zlib
from dataclasses import dataclass, field

import numpy as np

from unicode3d.background import Gradient, Sky
from unicode3d.color import Color, linear_to_srgb, xterm_rgb
from unicode3d.glyphs import GLYPH_SETS
from unicode3d.mesh import Mesh, make_box
from unicode3d.scene import Camera, Light, Object3D, PointLight, Renderer
from unicode3d.shapes import blob_mesh, block_mesh
from unicode3d.terminal import Screen
from unicode3d.transforms import quat_axis_angle, quat_mul

SIZE = (100, 36)  # cells, columns x rows
CELL = (8, 16)    # pixels of a cell in the pictures, wide x high
UP = (0.0, 1.0, 0.0)
TERMINAL_FG, TERMINAL_BG = (204, 204, 204), (12, 12, 16)  # the terminal's own colours, in the pictures


@dataclass
class Shot:
    """A scene as drawn: objects, camera and lights, Renderer settings, and the screen's glyphs and colours."""
    objects: list
    camera: Camera
    lights: object
    settings: dict = field(default_factory=dict)
    glyphs: str = "sextant"
    color: str = "truecolor"


def camera(position, target=(0.0, 0.0, 0.0), fov=50.0):
    return Camera(position=np.array(position, float), target=np.array(target, float), fov=fov)


def floor(half, y=0.0, texture=None, repeat=1):
    """A square floor facing up, 2 * half across, textured `repeat` times across if given a texture."""
    verts = np.array([(-half, y, half), (half, y, half), (half, y, -half), (-half, y, -half)], float)
    mesh = Mesh(verts, np.array([(0, 1, 2), (0, 2, 3)]))
    if texture is not None:
        mesh.uvs = np.array([[(0, 0), (1, 0), (1, 1)], [(0, 0), (1, 1), (0, 1)]], float)
        mesh.materials = np.zeros(2, int)
        mesh.textures = [np.tile(texture, (repeat, repeat, 1))]
    return mesh


def panel(width, height, texture=None):
    """An upright rectangle facing +z, standing on y = 0, showing all of `texture` if given."""
    w = width / 2
    mesh = Mesh(np.array([(-w, 0, 0), (w, 0, 0), (w, height, 0), (-w, height, 0)], float),
                np.array([(0, 1, 2), (0, 2, 3)]))
    if texture is not None:
        mesh.uvs = np.array([[(0, 0), (1, 0), (1, 1)], [(0, 0), (1, 1), (0, 1)]], float)
        mesh.materials = np.zeros(2, int)
        mesh.textures = [texture]
    return mesh


def checks(res, n, a=(0.85, 0.83, 0.8), b=(0.25, 0.25, 0.3)):
    y, x = np.mgrid[0:res, 0:res] * n // res
    return np.where(((x + y) % 2)[..., None], a, b).astype(float)


def turned(angle, axis=UP):
    return quat_axis_angle(axis, angle)


# ----- scenes --------------------------------------------------------------------------------

def cube():
    box = Object3D(make_box(), rotation=quat_mul(turned(0.6), turned(0.45, (1, 0, 0))), color=(220, 120, 60))
    return Shot([box], camera((0, 0.3, 3.6)), Light(), {"background": Gradient()})


def die(glyphs="sextant", color="truecolor"):
    from unicode3d.examples.dice import make_die
    dice = [Object3D(make_die(), np.array([x, 0.0, 0.0]), quat_mul(turned(0.5 + x), turned(0.6, (1, 0, 0))), color=c)
            for x, c in ((-1.3, (240, 240, 230)), (0.0, Color.RED), (1.3, Color.GREEN))]
    return Shot(dice, camera((0, 0.8, 4.4), fov=45.0), Light(), {"background": Gradient()}, glyphs, color)


def glass():
    solid = Object3D(blob_mesh((0.6, 0.6, 0.6), (0.0, 0.6, -1.2), rings=24, segments=32), color=(230, 200, 60))
    panes = [Object3D(block_mesh((x, 0.7, z), (0.9, 1.4, 0.9)), color=c, opacity=0.4)
             for x, z, c in ((-0.7, 0.0, (255, 90, 90)), (0.0, 0.4, (90, 255, 120)), (0.7, 0.0, (90, 140, 255)))]
    ball = Object3D(blob_mesh((0.35, 0.35, 0.35), (0.0, 0.35, 1.2), rings=16, segments=24), color=(200, 230, 255),
                    opacity=0.3)
    ground = Object3D(floor(4.0), color=(170, 170, 170))
    sun = Light(direction=np.array([0.4, -1.0, -0.6]), shadows=True)
    return Shot([ground, solid, *panes, ball], camera((0.3, 2.2, 4.5), (0, 0.5, 0)), sun, {"background": Sky()})


def sun_shadows():
    things = [Object3D(block_mesh((-1.2, 0.5, 0.0), (0.6, 1.0, 0.6)), color=(200, 90, 70)),
              Object3D(blob_mesh((0.5, 0.5, 0.5), (0.2, 0.5, -0.6), rings=20, segments=28), color=(80, 160, 220)),
              Object3D(block_mesh((1.3, 0.9, 0.3), (0.25, 1.8, 0.25)), color=(220, 220, 220)),
              Object3D(block_mesh((0.6, 1.7, 0.3), (1.6, 0.12, 0.5)), color=(150, 110, 70))]
    sun = Light(direction=np.array([0.6, -1.0, -0.4]), shadows=True)
    return Shot([Object3D(floor(5.0), color=(180, 175, 165)), *things], camera((0.5, 3.0, 5.0), (0, 0.5, 0)), sun,
                {"background": Sky()})


def lamp_shadows():
    room = [Object3D(floor(4.0), color=(170, 160, 150)),
            Object3D(panel(8.0, 3.0), np.array([0.0, 0.0, -2.5]), color=(150, 150, 170))]
    pillars = [Object3D(block_mesh((np.cos(a) * 1.4, 0.6, np.sin(a) * 1.4 - 0.3), (0.3, 1.2, 0.3)),
                        color=(210, 120, 80)) for a in np.linspace(0, 2 * np.pi, 7)[:-1]]
    bulb = Object3D(blob_mesh((0.08, 0.08, 0.08), (0.0, 0.9, -0.3)), color=(255, 220, 160), emissive=1.0,
                    cast_shadows=False)
    lights = [Light(ambient=0.12, diffuse=0.0, specular=0.0),
              PointLight(np.array([0.0, 0.9, -0.3]), color=(255, 220, 160), diffuse=1.0, range=6.0, shadows=True)]
    return Shot(room + pillars + [bulb], camera((0.0, 2.6, 3.6), (0, 0.4, -0.3)), lights)


def cutouts():
    rng = np.random.default_rng(3)
    lattice = np.ones((64, 64, 4))
    lattice[..., :3] = (0.55, 0.4, 0.25)
    y, x = np.mgrid[0:64, 0:64]
    lattice[..., 3] = ((x % 16 < 4) | (y % 16 < 4)).astype(float)
    stained = np.ones((64, 64, 4))
    cells = rng.integers(0, 4, (4, 4))
    palette = np.array([(0.9, 0.2, 0.2), (0.2, 0.5, 0.9), (0.95, 0.8, 0.2), (0.3, 0.8, 0.4)])
    stained[..., :3] = palette[cells[y // 16, x // 16]]
    stained[..., 3] = np.where((x % 16 < 2) | (y % 16 < 2), 1.0, 0.45)
    stained[(x % 16 < 2) | (y % 16 < 2), :3] = 0.1
    objects = [Object3D(floor(4.0), color=(190, 190, 185)),
               Object3D(panel(1.6, 1.6, lattice), np.array([-0.9, 0.0, 0.0]), turned(0.3), color=(255, 255, 255)),
               Object3D(panel(1.6, 1.6, stained), np.array([0.9, 0.0, 0.0]), turned(-0.3), color=(255, 255, 255)),
               Object3D(block_mesh((0.0, 0.3, -1.2), (2.4, 0.6, 0.4)), color=(120, 130, 150))]
    sun = Light(direction=np.array([-0.3, -0.8, -1.0]), shadows=True)
    return Shot(objects, camera((0.0, 1.6, 3.8), (0, 0.6, -0.3)), sun, {"background": Sky()})


def mirror():
    glass_wall = Object3D(panel(3.6, 2.2), np.array([0.0, 0.0, -1.5]), color=(170, 180, 190), reflectivity=0.85)
    things = [Object3D(blob_mesh((0.4, 0.4, 0.4), (-0.7, 0.4, 0.0), rings=20, segments=28), color=(220, 80, 60)),
              Object3D(block_mesh((0.0, 0.35, 0.0), (0.5, 0.7, 0.5)), np.array([0.7, 0.0, 0.3]), turned(0.5),
                       color=(255, 255, 255))]
    things[1].mesh.face_colors = np.repeat([(255, 255, 255), (255, 200, 60), (255, 255, 255), (255, 255, 255),
                                            (90, 140, 255), (255, 80, 200)], 2, axis=0)
    return Shot([Object3D(floor(4.0, texture=checks(64, 8)), color=(255, 255, 255)), glass_wall, *things],
                camera((1.4, 1.4, 3.2), (0, 0.6, -0.5)), Light(direction=np.array([0.3, -1.0, -0.4]), shadows=True),
                {"background": Sky()})


def facing_mirrors():
    left = Object3D(panel(4.0, 2.0), np.array([-1.3, 0.0, 0.0]), turned(np.pi / 2), color=(160, 170, 180),
                    reflectivity=0.8)
    right = Object3D(panel(4.0, 2.0), np.array([1.3, 0.0, 0.0]), turned(-np.pi / 2), color=(160, 170, 180),
                     reflectivity=0.8)
    ball = Object3D(blob_mesh((0.35, 0.35, 0.35), (0.0, 0.5, 0.0), rings=20, segments=28), color=(240, 160, 40))
    return Shot([Object3D(floor(3.0, texture=checks(64, 8)), color=(255, 255, 255)), left, right, ball],
                camera((0.5, 1.2, 3.4), (-0.4, 0.5, 0.0)), Light(), {"background": Sky(), "mirror_bounces": 3})


def horizon():
    ground = Object3D(floor(40.0, texture=checks(512, 32)), color=(255, 255, 255))
    return Shot([ground], camera((0.0, 1.2, 6.0), (0.0, 0.6, -10.0), fov=60.0),
                Light(direction=np.array([0.2, -1.0, -0.3])), {"background": Sky(), "fog": 0.0})


def balls():
    rng = np.random.default_rng(1)
    mesh = blob_mesh((1.0, 1.0, 1.0), rings=12, segments=16)
    objects = [Object3D(mesh, rng.uniform(-4, 4, 3), scale=rng.uniform(0.2, 0.6),
                        color=tuple(int(c) for c in rng.integers(40, 255, 3))) for _ in range(400)]
    return Shot(objects, camera((0.0, 2.0, 12.0), fov=45.0), Light())


def sphere():
    ball = Object3D(blob_mesh((1.0, 1.0, 1.0), rings=96, segments=144), color=(200, 80, 60))
    return Shot([ball], camera((0.0, 0.0, 3.2)), [Light(), PointLight(np.array([-2.0, 1.5, 2.0]), color=Color.CYAN)])


def courtyard():
    from unicode3d.examples.room import Courtyard
    court = Courtyard(1)
    court.animate(2.0)
    lights = [Light(direction=np.array([0.5, -1.0, -0.35]), ambient=0.3, diffuse=0.6, color=(255, 245, 230),
                    shadows=True),
              PointLight(court.lamp_at, color=(255, 210, 150), diffuse=0.9, range=9.0, shadows=True)]
    eye = np.array([1.0, 1.6, 6.5])
    shot = camera(eye, eye + np.array([-0.35, -0.12, -1.0]), fov=70.0)
    shot.near = 0.05
    return Shot(list(court.objects), shot, lights, {"background": Sky()})


def fog():
    """Rows of pillars running into world fog that fades them into the sky (Fog, added after 0.4.1)."""
    from unicode3d.background import Fog
    pillars = [Object3D(block_mesh((x, 1.0, -z), (0.5, 2.0, 0.5)), color=(200, 110, 80) if x < 0 else (90, 150, 210))
               for z in range(0, 60, 4) for x in (-2.0, 2.0)]
    ground = Object3D(floor(40.0, texture=checks(256, 16)), color=(255, 255, 255))
    return Shot([ground, *pillars], camera((0.0, 1.6, 5.0), (0.0, 1.2, -10.0), fov=60.0), Light(),
                {"background": Sky(), "fog": Fog(start=4.0, end=40.0)})


def fog_stars():
    """World fog fading pillars into a starry sky box (SkyBox): into the sky blurred, so no star shows through a
    fogged pillar (fixed after 0.5.0)."""
    from unicode3d.background import Fog, SkyBox
    rng = np.random.default_rng(4)
    faces = []
    for k in range(6):
        night = np.full((64, 64, 3), (0.03, 0.04, 0.1))
        night[rng.integers(0, 64, 80), rng.integers(0, 64, 80)] = 1.0
        faces.append(night if k != 3 else np.full((64, 64, 3), 0.03))
    pillars = [Object3D(block_mesh((x, 1.5, -z), (1.0, 3.0, 1.0)), color=(170, 160, 150))
               for z in range(0, 40, 5) for x in (-2.5, 2.5)]
    return Shot([Object3D(floor(30.0), color=(90, 90, 100)), *pillars], camera((0.0, 1.4, 5.0), (0.0, 1.6, -10.0),
                                                                                    fov=60.0),
                Light(direction=np.array([0.3, -1.0, -0.5]), ambient=0.25), {"background": SkyBox(faces),
                                                                               "fog": Fog(start=3.0, end=35.0)})


def materials():
    """The same sphere matte, plain, glossy and chrome-tight (Object3D.specular and shininess, added after 0.4.1)."""
    ball = blob_mesh((0.55, 0.55, 0.55), rings=32, segments=48)
    spheres = [Object3D(ball, np.array([x, 0.0, 0.0]), color=(60, 110, 200), specular=spec, shininess=shine)
               for x, spec, shine in ((-1.95, 0.0, None), (-0.65, 1.0, None), (0.65, 2.0, 12.0), (1.95, 2.5, 120.0))]
    lights = [Light(direction=np.array([-0.5, -0.6, -1.0])), PointLight(np.array([1.5, 2.0, 2.5]), range=8.0)]
    return Shot(spheres, camera((0.0, 0.4, 4.6)), lights, {"background": Gradient()})


def stretched():
    """Shapes scaled unevenly (scale=(x, y, z), added after 0.4.1): a plank, a tall box, an egg, a mirror-image die,
    and a group stretched as a whole with a box turned inside it (sheared), lit so that wrong normals would show."""
    from unicode3d.examples.dice import make_die
    from unicode3d.scene import Node
    ball = blob_mesh((1.0, 1.0, 1.0), rings=24, segments=32)
    group = Node(position=np.array([1.6, 0.0, -0.4]), scale=(1.0, 2.2, 1.0))
    objects = [Object3D(floor(4.0), color=(170, 170, 170)),
               Object3D(make_box(), np.array([-1.5, 0.12, 0.6]), turned(0.3), scale=(1.8, 0.24, 0.5),
                        color=(170, 120, 70)),
               Object3D(make_box(), np.array([-1.6, 0.9, -0.8]), turned(0.5), scale=(0.5, 1.8, 0.5),
                        color=(90, 160, 90)),
               Object3D(ball, np.array([0.0, 0.55, 0.3]), turned(0.4, (0, 0, 1)), scale=(0.35, 0.55, 0.35),
                        color=(230, 220, 200)),
               Object3D(make_die(), np.array([0.2, 0.3, 1.5]), turned(0.7), scale=(-0.6, 0.6, 0.6),
                        color=(240, 240, 230)),
               Object3D(make_box(), np.array([0.0, 0.25, 0.0]), turned(0.8, (0, 0, 1)), scale=0.4,
                        color=(200, 90, 160), parent=group)]
    sun = Light(direction=np.array([0.5, -1.0, -0.5]), shadows=True)
    return Shot(objects, camera((0.3, 2.4, 4.6), (0, 0.5, 0)), sun, {"background": Sky()})


def _write_model(folder):
    """An OBJ model with its MTL and textures, written into `folder`: a column (its own normals, smooth across the
    seam where its texture wraps round), a crate whose texture repeats twice across each face (map_Kd -s), a
    glowing cap and a cut-out fence (map_d). Returns the OBJ's path."""
    from PIL import Image
    os.makedirs(os.path.join(folder, "textures"), exist_ok=True)
    stripes = np.zeros((32, 64, 3), np.uint8)
    stripes[:] = (200, 190, 170)
    stripes[:, ::8] = stripes[:, 1::8] = (90, 60, 40)
    Image.fromarray(stripes).save(os.path.join(folder, "textures", "stripes.png"))
    Image.fromarray((checks(32, 4, (0.75, 0.55, 0.3), (0.45, 0.3, 0.15)) * 255).astype(np.uint8)).save(
        os.path.join(folder, "textures", "crate.png"))
    y, x = np.mgrid[0:32, 0:32]
    Image.fromarray((((x % 8) < 2) | ((y % 8) < 2)).astype(np.uint8) * 255).save(
        os.path.join(folder, "textures", "fence.png"))
    with open(os.path.join(folder, "model.mtl"), "w") as f:
        f.write("newmtl stone\nKd 1 1 1\nKs 0.3 0.3 0.3\nNs 30\nmap_Kd textures/stripes.png\n"
                "newmtl crate\nKd 1 1 1\nKs 0.1 0.1 0.1\nmap_Kd -s 2 2 1 textures\\crate.png\n"
                "newmtl glow\nKd 1 0.8 0.4\nKe 1 0.8 0.4\nillum 1\n"
                "newmtl fence\nKd 0.35 0.45 0.3\nmap_d textures/fence.png\n")
    lines, n = ["mtllib model.mtl", "o column", "usemtl stone"], 24
    for i in range(n + 1):  # a ring of vertices at the bottom and the top, the first and last at the seam
        a = 2 * np.pi * i / n
        lines += [f"v {0.4 * np.cos(a) - 1.2:.5f} {h:.5f} {0.4 * np.sin(a):.5f}" for h in (0.0, 1.6)]
        lines += [f"vt {i / n:.5f} {h:.5f}" for h in (0.0, 1.0)]
        lines += [f"vn {np.cos(a):.5f} 0 {np.sin(a):.5f}"]
    for i in range(n):
        b, t, b2, t2 = 2 * i + 1, 2 * i + 2, 2 * i + 3, 2 * i + 4
        lines.append(f"f {b}/{b}/{i + 1} {t}/{t}/{i + 1} {t2}/{t2}/{i + 2} {b2}/{b2}/{i + 2}")
    lines += ["usemtl glow", "s off"]
    top = [2 * i + 2 for i in range(n)]
    lines.append("f " + " ".join(str(v) for v in reversed(top)))
    base, uv = 2 * (n + 1), 2 * (n + 1)
    lines += ["o crate", "usemtl crate", "s off", "vt 0 0", "vt 1 0", "vt 1 1", "vt 0 1"]
    corners = [(x, y, z) for x in (-0.5, 0.5) for y in (0.0, 1.0) for z in (-0.5, 0.5)]
    lines += [f"v {x + 0.4} {y} {z - 0.3}" for x, y, z in corners]
    for quad in ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (1, 5, 7, 3), (0, 2, 6, 4)):
        lines.append("f " + " ".join(f"{base + q + 1}/{uv + k + 1}" for k, q in enumerate(quad)))
    lines += ["o fence", "usemtl fence", "v 0.9 0 0.8", "v 2.1 0 0.2", "v 2.1 1.1 0.2", "v 0.9 1.1 0.8",
              "vt 0 0", "vt 3 0", "vt 3 2", "vt 0 2", "f -4/-4 -3/-3 -2/-2 -1/-1"]
    with open(os.path.join(folder, "model.obj"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return os.path.join(folder, "model.obj")


def model():
    """A model loaded from an OBJ file with materials and textures (models.load_model, added after 0.5.0), on a
    floor whose texture repeats beyond 0..1 (uv up to 24), and is sampled at a mip level per pixel."""
    from unicode3d.models import load_model
    with tempfile.TemporaryDirectory() as folder:
        loaded = load_model(_write_model(folder), double_sided=True)
    ground = floor(30.0, texture=checks(64, 2))
    ground.uvs = ground.uvs * 24.0
    sun = Light(direction=np.array([0.5, -1.0, -0.6]), shadows=True)
    return Shot([Object3D(ground, color=(255, 255, 255)), *loaded], camera((0.6, 1.9, 4.2), (0.2, 0.5, 0.0)),
                [sun, PointLight(np.array([-1.2, 2.2, 0.6]), color=(255, 200, 140), range=4.0)],
                {"background": Sky()})


def _write_gltf(path):
    """A glTF model (.glb) written to `path`: a textured crate, a smooth gold ball, a glowing lamp on an arm turned by
    its node (child of a turned, scaled post: the node hierarchy), a cut-out fence (alphaMode MASK) and a see-through
    pane (BLEND), and an animation swinging the arm (its rotation, CUBICSPLINE) and lifting the ball (STEP)."""
    import io
    import json
    import struct
    from PIL import Image
    from unicode3d.mesh import make_box
    from unicode3d.shapes import blob_mesh
    from unicode3d.transforms import quat_axis_angle
    blob, info = bytearray(), {"asset": {"version": "2.0"}, "bufferViews": [], "accessors": [], "meshes": [],
                               "materials": [], "images": [], "textures": [], "nodes": []}

    def add(kind, item):
        info.setdefault(kind, []).append(item)
        return len(info[kind]) - 1

    def view(data):
        blob.extend(b"\0" * (-len(blob) % 4))
        blob.extend(data)
        return add("bufferViews", {"buffer": 0, "byteOffset": len(blob) - len(data), "byteLength": len(data)})

    def accessor(values, integer=False):
        values = np.asarray(values)
        kind = "SCALAR" if values.ndim == 1 else f"VEC{values.shape[1]}"
        data = np.ascontiguousarray(values, "<u4" if integer else "<f4")
        return add("accessors", {"bufferView": view(data.tobytes()), "componentType": 5125 if integer else 5126,
                                 "count": len(values), "type": kind})

    def image(pixels):
        out = io.BytesIO()
        Image.fromarray(np.asarray(pixels, np.uint8)).save(out, "PNG")
        return add("textures", {"source": add("images", {"bufferView": view(out.getvalue()), "mimeType": "image/png"})})

    def mesh(m, material, uv=None):
        attributes = {"POSITION": accessor(m.vertices), "NORMAL": accessor(m.vertex_normals())}
        if uv is not None:
            attributes["TEXCOORD_0"] = accessor(uv)
        return add("meshes", {"primitives": [{"attributes": attributes, "indices": accessor(m.faces.reshape(-1), True),
                                              "material": material}]})

    def quat(axis, angle):  # glTF's order: x, y, z, w
        q = quat_axis_angle(axis, angle)
        return [*q[1:], q[0]]

    box = make_box(1.0)
    box_uv = np.zeros((len(box.vertices), 2))
    corners = np.array([(0, 1), (1, 1), (1, 0), (0, 0)], float)  # glTF's v runs down the image
    box_uv[:] = np.tile(corners, (len(box.vertices) // 4, 1))
    crate = (checks(32, 4, (0.75, 0.55, 0.3), (0.45, 0.3, 0.15)) * 255).astype(np.uint8)
    y, x = np.mgrid[0:32, 0:32]
    fence = np.zeros((32, 32, 4), np.uint8)
    fence[...] = (90, 115, 75, 0)
    fence[((x % 8) < 2) | ((y % 8) < 2), 3] = 255
    pane = type(box)(np.array([(-0.5, 0, 0), (0.5, 0, 0), (0.5, 1, 0), (-0.5, 1, 0)], float),
                     np.array([(0, 1, 2), (0, 2, 3)]))
    mats = [{"name": "crate", "pbrMetallicRoughness": {"baseColorTexture": {"index": image(crate)},
                                                       "metallicFactor": 0.0, "roughnessFactor": 0.8}},
            {"name": "gold", "pbrMetallicRoughness": {"baseColorFactor": [1.0, 0.6, 0.15, 1.0],
                                                      "metallicFactor": 1.0, "roughnessFactor": 0.35}},
            {"name": "post", "pbrMetallicRoughness": {"baseColorFactor": [0.2, 0.2, 0.25, 1.0], "metallicFactor": 0.0}},
            {"name": "glow", "emissiveFactor": [1.0, 0.8, 0.4],
             "pbrMetallicRoughness": {"baseColorFactor": [1.0, 0.8, 0.4, 1.0], "metallicFactor": 0.0}},
            {"name": "fence", "alphaMode": "MASK", "doubleSided": True,
             "pbrMetallicRoughness": {"baseColorTexture": {"index": image(fence)}, "metallicFactor": 0.0}},
            {"name": "glass", "alphaMode": "BLEND", "doubleSided": True,
             "pbrMetallicRoughness": {"baseColorFactor": [0.3, 0.6, 1.0, 0.4], "metallicFactor": 0.0,
                                      "roughnessFactor": 0.2}}]
    for m in mats:
        add("materials", m)
    pane_uv = np.array([(0, 2), (3, 2), (3, 0), (0, 0)], float)
    add("nodes", {"name": "Crate", "mesh": mesh(box, 0, box_uv), "translation": [0.5, 0.5, -0.4],
                  "rotation": quat([0, 1, 0], 0.4)})
    add("nodes", {"name": "Ball", "mesh": mesh(blob_mesh((0.35, 0.35, 0.35), rings=10, segments=16), 1),
                  "translation": [-0.5, 0.35, 0.3]})
    add("nodes", {"name": "Lamp", "mesh": mesh(blob_mesh((0.15, 0.15, 0.15), rings=6, segments=10), 3),
                  "translation": [0.0, 0.0, 0.8]})
    add("nodes", {"name": "Arm", "mesh": mesh(make_box(1.0), 2), "scale": [0.08, 0.08, 0.8],
                  "translation": [0.0, 0.0, 0.4]})
    add("nodes", {"name": "Swing", "children": [2, 3], "translation": [0.0, 1.6, 0.0]})
    add("nodes", {"name": "Post", "mesh": mesh(make_box(1.0), 2), "scale": [0.12, 1.6, 0.12],
                  "translation": [0.0, 0.8, 0.0]})
    add("nodes", {"name": "Stand", "children": [4, 5], "translation": [-1.4, 0.0, -0.6], "rotation": quat([0, 1, 0], 0.8)})
    add("nodes", {"name": "Fence", "mesh": mesh(pane, 4, pane_uv), "translation": [1.4, 0.0, 0.5],
                  "rotation": quat([0, 1, 0], -0.5), "scale": [1.2, 1.1, 1.0]})
    add("nodes", {"name": "Pane", "mesh": mesh(pane, 5), "translation": [0.2, 0.0, 1.0]})
    info["scenes"], info["scene"] = [{"nodes": [0, 1, 6, 7, 8]}], 0
    times = accessor(np.array([0.0, 1.0, 2.0]))
    swing = np.array([quat([0, 1, 0], a) for a in (-0.6, 0.6, -0.6) for _ in range(3)])
    swing[0::3] = swing[2::3] = 0.0  # tangents
    info["animations"] = [{"name": "Swing", "samplers": [
        {"input": times, "output": accessor(swing), "interpolation": "CUBICSPLINE"},
        {"input": times, "output": accessor(np.array([(-0.5, 0.35, 0.3), (-0.5, 0.75, 0.3), (-0.5, 0.35, 0.3)])),
         "interpolation": "STEP"}],
        "channels": [{"sampler": 0, "target": {"node": 4, "path": "rotation"}},
                     {"sampler": 1, "target": {"node": 1, "path": "translation"}}]}]
    blob.extend(b"\0" * (-len(blob) % 4))
    info["buffers"] = [{"byteLength": len(blob)}]
    text = json.dumps(info).encode()
    text += b" " * (-len(text) % 4)
    body = struct.pack("<II", len(text), 0x4E4F534A) + text + struct.pack("<II", len(blob), 0x004E4942) + bytes(blob)
    with open(path, "wb") as f:
        f.write(b"glTF" + struct.pack("<II", 2, 12 + len(body)) + body)
    return path


def gltf():
    """A glTF model (gltf.load_gltf, added after 0.6.0): textures, PBR materials as unicode3d's (gold reflecting the
    sky), cut-out and see-through parts, a node hierarchy, and its animation posed partway through."""
    from unicode3d.models import load_model
    with tempfile.TemporaryDirectory() as folder:
        loaded = load_model(_write_gltf(os.path.join(folder, "scene.glb")))
    loaded.animations["Swing"].apply(1.5)
    ground = floor(6.0, texture=checks(64, 4))
    sun = Light(direction=np.array([0.5, -1.0, -0.6]), shadows=True)
    return Shot([Object3D(ground, color=(255, 255, 255)), *loaded], camera((0.4, 2.2, 4.4), (0.0, 0.6, 0.0)),
                [sun, PointLight(loaded.nodes["Lamp"].to_world(np.zeros(3)), color=(255, 200, 140), range=3.0)],
                {"background": Sky()})


SCENES = {
    "cube": cube,
    "die": die,
    "die-quad": lambda: die("quad"),
    "die-half": lambda: die("half"),
    "die-ascii": lambda: die("ascii"),
    "die-256": lambda: die(color="256"),
    "die-16": lambda: die(color="16"),
    "glass": glass,
    "sun-shadows": sun_shadows,
    "lamp-shadows": lamp_shadows,
    "cutouts": cutouts,
    "mirror": mirror,
    "facing-mirrors": facing_mirrors,
    "horizon": horizon,
    "balls": balls,
    "sphere": sphere,
    "courtyard": courtyard,
    "fog": fog,
    "fog-stars": fog_stars,
    "materials": materials,
    "stretched": stretched,
    "model": model,
    "gltf": gltf,
}


# ----- rendering and pictures ----------------------------------------------------------------

def render(shot):
    """The shot drawn at SIZE: {rgb, alpha, depth, ids (the framebuffer), chars (code points), fg, bg (packed
    cell colours)}."""
    cols, rows = SIZE
    screen = Screen(None, glyphs=shot.glyphs, color=shot.color, size=(rows, cols))
    renderer = Renderer(cols, rows, screen.cell_pixels, **shot.settings)
    fb = renderer.render(shot.objects, shot.camera, shot.lights)
    screen.draw_frame(fb)
    return {"rgb": fb.rgb.copy(), "alpha": fb.alpha.copy(), "depth": fb.depth.copy(), "ids": fb.ids.copy(),
            "chars": screen.chars.view(np.uint32).copy(), "fg": screen.fg.copy(), "bg": screen.bg.copy(),
            "glyphs": np.array(shot.glyphs), "color": np.array(shot.color)}


def unpack_colors(packed, default):
    """sRGB 0..255 (..., 3) of packed cell colours (see color.quantize): truecolor, palette or the default."""
    out = np.empty(packed.shape + (3,), np.uint8)
    out[...] = default
    rgb = packed >= 0
    truecolor = rgb & (packed & (1 << 25) != 0)
    for k, shift in enumerate((16, 8, 0)):
        out[..., k] = np.where(truecolor, (packed >> shift) & 255, out[..., k])
    indexed = rgb & ~truecolor
    out[indexed] = xterm_rgb()[packed[indexed] & 255].astype(np.uint8)
    return out


# A 5x7 bitmap of each character of the ASCII ramp, drawn in the middle of the cell.
ASCII_BITMAPS = {
    ".": ["", "", "", "", "", "..#..", "..#.."],
    ":": ["", "..#..", "..#..", "", "..#..", "..#..", ""],
    "-": ["", "", "", "#####", "", "", ""],
    "=": ["", "", "#####", "", "#####", "", ""],
    "+": ["", "..#..", "..#..", "#####", "..#..", "..#..", ""],
    "*": ["", "#.#.#", ".###.", "#####", ".###.", "#.#.#", ""],
    "#": [".#.#.", "#####", ".#.#.", ".#.#.", "#####", ".#.#.", ""],
    "%": ["##..#", "##.#.", "..#..", ".#...", "#..##", "...##", ""],
    "@": [".###.", "#...#", "#.###", "#.#.#", "#.###", "#....", ".###."],
}


def _cell_masks(glyphs):
    """{character: (CELL[1], CELL[0]) bool, True where it shows the foreground}."""
    w, h = CELL
    if glyphs == "ascii":
        masks = {" ": np.zeros((h, w), bool)}
        for c, rows in ASCII_BITMAPS.items():
            mask = np.zeros((h, w), bool)
            for y, row in enumerate(rows):
                for x, bit in enumerate(row):
                    mask[2 * y + 1:2 * y + 3, x + 1] = bit == "#"
            masks[c] = mask
        return masks
    glyph_set = GLYPH_SETS[glyphs]
    pw, ph = glyph_set.cell_pixels
    xs, ys = np.arange(w) * pw // w, np.arange(h) * ph // h
    sub = ys[:, None] * pw + xs[None, :]  # which of the cell's pixels each picture pixel is in
    return {c: (bits >> sub & 1).astype(bool) for bits, c in enumerate(glyph_set.chars)}


def cells_picture(frame):
    """What a terminal shows for the saved cells, as (rows * CELL[1], cols * CELL[0], 3) uint8: block glyphs as
    their shapes, ASCII as small bitmaps."""
    chars = frame["chars"].view("<U1") if frame["chars"].dtype == np.uint32 else frame["chars"]
    fg, bg = unpack_colors(frame["fg"], TERMINAL_FG), unpack_colors(frame["bg"], TERMINAL_BG)
    masks = _cell_masks(str(frame["glyphs"]))
    rows, cols = chars.shape
    w, h = CELL
    out = np.empty((rows * h, cols * w, 3), np.uint8)
    blank = np.zeros((h, w), bool)
    for y in range(rows):
        for x in range(cols):
            mask = masks.get(chars[y, x], blank)
            out[y * h:(y + 1) * h, x * w:(x + 1) * w] = np.where(mask[..., None], fg[y, x], bg[y, x])
    return out


def pixels_picture(frame):
    """The framebuffer's pixels over black, 8-bit sRGB, (H, W, 3)."""
    return np.rint(linear_to_srgb(np.clip(frame["rgb"], 0.0, 1.0)) * 255).astype(np.uint8)


def write_png(path, image):
    """Save (H, W, 3) uint8 as an 8-bit RGB PNG."""
    h, w, _ = image.shape
    raw = b"".join(b"\x00" + image[y].tobytes() for y in range(h))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


# ----- commands ------------------------------------------------------------------------------

def save(directory, names):
    os.makedirs(directory, exist_ok=True)
    for name in names:
        t0 = time.perf_counter()
        try:
            frame = render(SCENES[name]())
        except Exception as e:  # a feature the engine being rendered lacks
            print(f"  {name:16} skipped: {type(e).__name__}: {e}")
            continue
        np.savez_compressed(os.path.join(directory, name + ".npz"), **frame)
        write_png(os.path.join(directory, name + ".png"), cells_picture(frame))
        print(f"  {name:16} {time.perf_counter() - t0:6.2f} s")


def _load(directory, name):
    path = os.path.join(directory, name + ".npz")
    if not os.path.exists(path):
        return None
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def _cell_change(old, new, tolerance):
    """How many cells differ: another character, or a colour further off than `tolerance` levels (truecolor) or
    another palette entry."""
    rgb_old, rgb_new = unpack_colors(old["fg"], TERMINAL_FG), unpack_colors(new["fg"], TERMINAL_FG)
    bg_old, bg_new = unpack_colors(old["bg"], TERMINAL_BG), unpack_colors(new["bg"], TERMINAL_BG)
    far = lambda a, b: (np.abs(a.astype(int) - b.astype(int)) > tolerance).any(axis=-1)
    if str(new["color"]) != "truecolor":
        far = lambda a, b: (a != b).any(axis=-1)
    changed = (old["chars"] != new["chars"]) | far(rgb_old, rgb_new) | far(bg_old, bg_new)
    return int(changed.sum())


def compare(old_dir, new_dir, names, tolerance, out_dir=None):
    """Print how each scene changed from old_dir to new_dir; pictures of the changes go to out_dir. Returns whether
    any scene changed by more than the tolerance."""
    out_dir = out_dir or os.path.join(new_dir, "diff")
    worse = False
    print(f"{'scene':16} {'status':18} {'max level':>9} {'pixels':>8} {'alpha':>7} {'depth':>8} {'ids':>6} {'cells':>6}")
    for name in names:
        old, new = _load(old_dir, name), _load(new_dir, name)
        if old is None or new is None:
            where = "in neither" if old is None and new is None else "only in new" if old is None else "only in old"
            print(f"{name:16} {where}")
            continue
        if old["rgb"].shape != new["rgb"].shape or old["chars"].shape != new["chars"].shape:
            print(f"{name:16} size changed: {old['rgb'].shape[:2]} -> {new['rgb'].shape[:2]}")
            worse = True
            continue
        if all(np.array_equal(old[k], new[k]) for k in ("rgb", "alpha", "depth", "ids", "chars", "fg", "bg")):
            print(f"{name:16} identical")
            continue
        a, b = pixels_picture(old).astype(int), pixels_picture(new).astype(int)
        levels = np.abs(a - b).max(axis=-1)
        pixels = int((levels > tolerance).sum())
        alpha = float(np.abs(old["alpha"] - new["alpha"]).max())
        depth = float((np.abs(old["depth"] - new["depth"]) / np.maximum(np.abs(old["depth"]), 1e-12)).max())
        ids = int((old["ids"] != new["ids"]).sum())
        cells = _cell_change(old, new, tolerance)
        changed = pixels > 0 or alpha * 255 > tolerance or cells > 0
        worse |= changed
        status = "CHANGED" if changed else "within tolerance"
        print(f"{name:16} {status:18} {int(levels.max()):9d} {pixels:8d} {alpha:7.3f} {depth:8.1e} {ids:6d} {cells:6d}")
        if changed:
            os.makedirs(out_dir, exist_ok=True)
            gap = np.full((a.shape[0], 4, 3), 128, np.uint8)
            diff = np.clip(np.abs(a - b) * 8, 0, 255).astype(np.uint8)
            write_png(os.path.join(out_dir, name + "-pixels.png"),
                      np.concatenate([a.astype(np.uint8), gap, b.astype(np.uint8), gap, diff], axis=1))
            ca, cb = cells_picture(old), cells_picture(new)
            gap = np.full((ca.shape[0], 8, 3), 128, np.uint8)
            write_png(os.path.join(out_dir, name + "-cells.png"), np.concatenate([ca, gap, cb], axis=1))
    print("\nmax level: the largest change of a pixel's colour, in 8-bit sRGB levels (pixels over black); pixels:\n"
          f"how many moved more than {tolerance}; alpha: the largest change in coverage; depth: the largest relative\n"
          "change in depth; ids: pixels showing another object; cells: terminal cells that look different.")
    if worse:
        print(f"Pictures of the changes (old | new | difference x8, and the cells old | new): {out_dir}")
    return worse


def _run_save(engine_root, directory, names):
    """Save the scenes rendered by the engine in engine_root, in a separate Python (so that its unicode3d is the
    one imported); this file does the rendering, whatever the revision."""
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([engine_root, os.environ.get("PYTHONPATH", "")]))
    cmd = [sys.executable, os.path.abspath(__file__), "save", directory] + [a for n in names for a in ("--scene", n)]
    subprocess.run(cmd, cwd=engine_root, env=env, check=True)


def diff(rev, names, tolerance):
    root = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=os.path.dirname(os.path.abspath(__file__)),
                          capture_output=True, text=True, check=True).stdout.strip()
    commit = subprocess.run(["git", "rev-parse", "--short", rev], cwd=root, capture_output=True, text=True,
                            check=True).stdout.strip()
    work = tempfile.mkdtemp(prefix="unicode3d-gallery-")
    tree = os.path.join(work, "engine")
    subprocess.run(["git", "worktree", "add", "--detach", "--quiet", tree, commit], cwd=root, check=True)
    try:
        print(f"{rev} ({commit}), compiling its kernels first:", flush=True)
        _run_save(tree, os.path.join(work, "old"), names)
        print("this checkout:", flush=True)
        _run_save(root, os.path.join(work, "new"), names)
    finally:
        subprocess.run(["git", "worktree", "remove", "--force", tree], cwd=root, check=False)
        shutil.rmtree(tree, ignore_errors=True)
    print()
    worse = compare(os.path.join(work, "old"), os.path.join(work, "new"), names, tolerance, os.path.join(work, "diff"))
    print(f"Both sets of scenes: {work}/old and {work}/new")
    return worse


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("command", choices=("save", "compare", "diff"))
    parser.add_argument("paths", nargs="*", help="save: DIR; compare: OLD NEW; diff: REV (default HEAD)")
    parser.add_argument("--scene", action="append", choices=SCENES, help="scene to use (default: all)")
    parser.add_argument("--tolerance", type=int, default=1, help="8-bit sRGB levels a colour may move (default 1)")
    args = parser.parse_args()
    names = args.scene or list(SCENES)
    if args.command == "save":
        if len(args.paths) != 1:
            parser.error("save takes one directory")
        save(args.paths[0], names)
        return 0
    if args.command == "compare":
        if len(args.paths) != 2:
            parser.error("compare takes two directories")
        return int(compare(*args.paths, names, args.tolerance))
    if len(args.paths) > 1:
        parser.error("diff takes at most one revision")
    return int(diff(args.paths[0] if args.paths else "HEAD", names, args.tolerance))


if __name__ == "__main__":
    sys.exit(main())
