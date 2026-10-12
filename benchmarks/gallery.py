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
floor whose texture repeats, and fog over a starry sky box; after 0.6.0, a glTF model; after 0.7.1, the
courtyard through an orthographic camera; after 0.17.0, sun shadows fitted to the view in a big field; and,
after 0.20.1, particles and a billboard sprite, lines, and sprites with eight sides; after 0.21.0, skinned
figures (from glTF and built in code); and, after 0.22.0, morph targets.
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
from unicode3d import pictures
from unicode3d.color import Color, linear_to_srgb
from unicode3d.pictures import TERMINAL_BG, TERMINAL_FG, unpack_colors
from unicode3d.mesh import Mesh, make_box
from unicode3d.scene import Camera, Light, Object3D, PointLight, Renderer
from unicode3d.shapes import blob_mesh, block_mesh
from unicode3d.terminal import Screen
from unicode3d.transforms import quat_axis_angle, quat_mul

SIZE = (100, 36)  # cells, columns x rows
UP = (0.0, 1.0, 0.0)


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


def view_shadows():
    """sun_shadows' things in a field 60 units across, with the sun's map fitted to what the camera sees."""
    shot = sun_shadows()
    rng = np.random.default_rng(3)
    posts = [Object3D(block_mesh((x, 0.6, z), (0.3, 1.2, 0.3)), color=(120, 150, 90))
             for x, z in rng.uniform(-30.0, 30.0, (60, 2)) if abs(x) > 4.0 or abs(z) > 4.0]
    shot.objects = [Object3D(floor(30.0), color=(180, 175, 165)), *shot.objects[1:], *posts]
    shot.settings["shadow_fit"] = "view"
    return shot


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


def _write_skinned_gltf(path):
    """A skinned figure in glTF (.glb) written to `path`: a tube bound standing straight to a chain of three joints
    (JOINTS_0 and WEIGHTS_0, inverse bind matrices), which the file's nodes leave bent, under a turned armature; its
    last joint has a mesh of its own (a box riding on it), and an animation turns its middle one (not played)."""
    import json
    from unicode3d.transforms import quat_axis_angle
    blob, info = bytearray(), {"asset": {"version": "2.0"}, "bufferViews": [], "accessors": [], "meshes": [],
                               "materials": [], "nodes": []}

    def add(kind, item):
        info.setdefault(kind, []).append(item)
        return len(info[kind]) - 1

    def accessor(values, component=5126, kind=None):
        values = np.asarray(values)
        data = np.ascontiguousarray(values, {5126: "<f4", 5125: "<u4", 5123: "<u2"}[component])
        blob.extend(b"\0" * (-len(blob) % 4))
        blob.extend(data.tobytes())
        view = add("bufferViews", {"buffer": 0, "byteOffset": len(blob) - data.nbytes, "byteLength": data.nbytes})
        kind = kind or ("SCALAR" if values.ndim == 1 else f"VEC{values.shape[1]}")
        return add("accessors", {"bufferView": view, "componentType": component, "count": len(values), "type": kind})

    def quat(axis, angle):  # glTF's order: x, y, z, w
        q = quat_axis_angle(axis, angle)
        return [*q[1:], q[0]]

    bones, rings, segments = 3, 8, 14
    heights = np.linspace(0.0, float(bones), rings * bones + 1)
    turn = np.linspace(0.0, 2 * np.pi, segments, endpoint=False)
    radius = 0.28 - 0.05 * heights / bones
    vertices = np.array([(r * np.cos(a), y, r * np.sin(a)) for y, r in zip(heights, radius) for a in turn])
    normals = np.c_[np.cos(np.tile(turn, len(heights))), np.zeros(len(vertices)), np.sin(np.tile(turn, len(heights)))]
    faces = []
    for r in range(len(heights) - 1):
        for k in range(segments):
            a, b = r * segments + k, r * segments + (k + 1) % segments
            faces += [(a, a + segments, b), (b, a + segments, b + segments)]
    along = np.clip(vertices[:, 1], 0.0, bones - 1e-9)
    bone, part = np.floor(along).astype(int), along - np.floor(along)
    near = np.where(part < 0.5, np.maximum(bone - 1, 0), np.minimum(bone + 1, bones - 1))  # (the joint it bends at)
    share = 0.5 * np.clip(1.0 - np.minimum(part, 1.0 - part) / 0.35, 0.0, 1.0)
    joints = np.c_[bone, near, np.zeros((len(bone), 2), int)]
    weights = np.c_[1.0 - share, share, np.zeros((len(bone), 2))]
    add("materials", {"name": "skin", "pbrMetallicRoughness": {"baseColorFactor": [0.85, 0.55, 0.4, 1.0],
                                                              "metallicFactor": 0.0, "roughnessFactor": 0.6}})
    add("materials", {"name": "hat", "pbrMetallicRoughness": {"baseColorFactor": [0.2, 0.35, 0.8, 1.0],
                                                             "metallicFactor": 0.0, "roughnessFactor": 0.4}})
    add("meshes", {"name": "Tube", "primitives": [{"attributes": {
        "POSITION": accessor(vertices), "NORMAL": accessor(normals), "JOINTS_0": accessor(joints, 5123),
        "WEIGHTS_0": accessor(weights)}, "indices": accessor(np.array(faces).reshape(-1), 5125), "material": 0}]})
    hat = make_box(0.4)
    add("meshes", {"name": "Hat", "primitives": [{"attributes": {
        "POSITION": accessor(hat.vertices + (0.0, 1.2, 0.0)), "NORMAL": accessor(hat.vertex_normals())},
        "indices": accessor(hat.faces.reshape(-1), 5125), "material": 1}]})
    # Bound standing straight: joint k at height k in the armature's space.
    inverse = np.array([np.linalg.inv(np.array([[1, 0, 0, 0], [0, 1, 0, k], [0, 0, 1, 0], [0, 0, 0, 1]], float))
                        for k in range(bones)])
    add("nodes", {"name": "Hip", "children": [1], "rotation": quat([1, 0, 0], 0.15)})
    add("nodes", {"name": "Knee", "children": [2], "translation": [0.0, 1.0, 0.0], "rotation": quat([0, 0, 1], 0.55)})
    add("nodes", {"name": "Ankle", "mesh": 1, "translation": [0.0, 1.0, 0.0],
                  "rotation": quat([0.3, 0.2, 1.0], 0.7), "scale": [1.1, 1.1, 1.1]})
    add("nodes", {"name": "Figure", "mesh": 0, "skin": 0, "translation": [5.0, 0.0, 0.0]})  # (its place: ignored)
    add("nodes", {"name": "Armature", "children": [0], "translation": [0.2, 0.0, -0.3],
                  "rotation": quat([0, 1, 0], 0.6), "scale": [1.1, 1.1, 1.1]})
    info["skins"] = [{"joints": [0, 1, 2], "inverseBindMatrices": accessor(inverse.transpose(0, 2, 1).reshape(-1, 16),
                                                                            kind="MAT4")}]
    info["scenes"], info["scene"] = [{"nodes": [4, 3]}], 0
    info["animations"] = [{"name": "Wave", "samplers": [
        {"input": accessor(np.array([0.0, 1.0])), "output": accessor(np.array([quat([0, 0, 1], 0.55),
                                                                               quat([0, 0, 1], -0.4)]))}],
        "channels": [{"sampler": 0, "target": {"node": 1, "path": "rotation"}}]}]
    blob.extend(b"\0" * (-len(blob) % 4))
    info["buffers"] = [{"byteLength": len(blob)}]
    text = json.dumps(info).encode()
    text += b" " * (-len(text) % 4)
    body = struct.pack("<II", len(text), 0x4E4F534A) + text + struct.pack("<II", len(blob), 0x004E4942) + bytes(blob)
    with open(path, "wb") as f:
        f.write(b"glTF" + struct.pack("<II", 2, 12 + len(body)) + body)
    return path


def gltf_skinned():
    """A skinned glTF figure as the file leaves it (added in 0.22.0; until then posed once as it loaded, now by its
    joints as it is drawn: the two should match), its shadows from the sun and a lamp."""
    from unicode3d.models import load_model
    with tempfile.TemporaryDirectory() as folder:
        loaded = load_model(_write_skinned_gltf(os.path.join(folder, "skinned.glb")))
    ground = floor(4.0, texture=checks(64, 4))
    sun = Light(direction=np.array([0.6, -1.0, -0.4]), shadows=True)
    lamp = PointLight(np.array([-1.2, 2.2, 1.4]), color=(255, 210, 160), range=5.0, shadows=True)
    return Shot([Object3D(ground, color=(255, 255, 255)), *loaded], camera((0.5, 1.9, 4.6), (0.3, 1.3, 0.0)),
                [sun, lamp], {"background": Sky()})


def skinned():
    """Figures built in code, each one mesh skinned to a skeleton of its own (added in 0.22.0), sharing the mesh:
    one standing as bound, one striding, one waving with its body turned, and a small one crouching; shadows from
    the sun and a lamp."""
    from unicode3d.examples.rigs import bend, figure, figure_rig
    rig = figure_rig()
    standing, _ = figure(rig, (-1.35, 0.0, -0.4), 0.3, color=(200, 170, 140))
    striding, j = figure(rig, (-0.45, 0.0, 0.1), -0.5, color=(120, 160, 210))
    for side, step in (("right", 1.0), ("left", -1.0)):
        bend(j[f"{side} hip"], (1, 0, 0), 0.45 * step)
        bend(j[f"{side} knee"], (1, 0, 0), -0.15 if step > 0 else -0.7)
        bend(j[f"{side} shoulder"], (1, 0, 0), -0.45 * step)
        bend(j[f"{side} elbow"], (1, 0, 0), 0.5)
    bend(j["spine"], (1, 0, 0), 0.1)
    waving, j = figure(rig, (0.55, 0.0, -0.2), -0.25, color=(210, 120, 110))
    bend(j["spine"], (0, 1, 0), 0.45)
    bend(j["neck"], (0, 0, 1), 0.2)
    bend(j["right shoulder"], (0, 0, 1), 2.1)
    bend(j["right elbow"], (0, 0, 1), 0.8)
    bend(j["left shoulder"], (1, 0, 0), 0.2)
    small, j = figure(rig, (1.45, 0.0, 0.4), -0.7, color=(150, 190, 120))
    j["hips"].scale = 0.6
    j["hips"].position = j["hips"].position * (1.0, 0.0, 1.0) + (0.0, 0.32, 0.0)
    for side in ("right", "left"):
        bend(j[f"{side} hip"], (1, 0, 0), 1.3)
        bend(j[f"{side} knee"], (1, 0, 0), -2.0)
        bend(j[f"{side} shoulder"], (1, 0, 0), 1.0)
    bend(j["spine"], (1, 0, 0), 0.5)
    ground = floor(4.0, texture=checks(64, 4))
    sun = Light(direction=np.array([0.6, -1.0, -0.4]), shadows=True)
    lamp = PointLight(np.array([-1.0, 2.4, 1.6]), color=(255, 210, 160), range=6.0, shadows=True)
    return Shot([Object3D(ground, color=(255, 255, 255)), standing, striding, waving, small],
                camera((0.1, 1.35, 3.5), (0.05, 0.8, 0.0)), [sun, lamp], {"background": Sky()})


def morph():
    """Morph targets (added in 0.23.0), built in code and set with constructor arguments: a ball, posed by itself
    and turned, both squashed and grown spiky; a flag rippling one way and creased the other, its normals turned
    with it; and a skinned figure whose head is swollen and whose body is thinned while it waves (a morph before
    the skin); shadows from the sun and a lamp."""
    from unicode3d.examples.rigs import bend, figure_rig, skeleton
    from unicode3d.scene import Skin
    rings, around = 16, 24
    lat = np.linspace(0.0, np.pi, rings + 1)
    lon = np.linspace(0.0, 2.0 * np.pi, around, endpoint=False)
    unit = np.array([(np.sin(a) * np.cos(b), np.cos(a), np.sin(a) * np.sin(b)) for a in lat for b in lon])
    faces = [(r * around + k, r * around + (k + 1) % around, (r + 1) * around + k) for r in range(rings)
             for k in range(around)]
    faces += [(r * around + (k + 1) % around, (r + 1) * around + (k + 1) % around, (r + 1) * around + k)
              for r in range(rings) for k in range(around)]
    squash = unit * (0.25, -0.45, 0.25)
    spikes = unit * (0.5 * np.outer(np.arange(rings + 1) % 4 == 2, np.arange(around) % 4 == 0).reshape(-1))[:, None]
    ball = Object3D(Mesh(unit * 0.45, np.array(faces), morph_targets=np.stack([squash, spikes])),
                    position=np.array([-1.0, 0.5, -0.7]), rotation=turned(0.5), color=(110, 170, 220),
                    morph_weights=[0.4, 0.8])
    x, y = np.meshgrid(np.linspace(0.0, 1.2, 25), np.linspace(0.0, 0.8, 17))
    flat = np.stack([x.ravel(), y.ravel(), np.zeros(x.size)], axis=1)
    cells = [(r * 25 + c, r * 25 + c + 1, (r + 1) * 25 + c + 1) for r in range(16) for c in range(24)]
    cells += [(r * 25 + c, (r + 1) * 25 + c + 1, (r + 1) * 25 + c) for r in range(16) for c in range(24)]
    ripple = np.zeros_like(flat)
    ripple[:, 2] = 0.12 * np.sin(flat[:, 0] * 9.0) * flat[:, 0]
    crease = np.zeros_like(flat)
    crease[:, 2] = 0.2 * np.abs(flat[:, 1] - 0.4) * flat[:, 0]
    turn = np.zeros((2, len(flat), 3))
    turn[0, :, 0] = -0.12 * 9.0 * np.cos(flat[:, 0] * 9.0) * flat[:, 0]
    flag = Object3D(Mesh(flat, np.array(cells), morph_targets=np.stack([ripple, crease]), morph_normals=turn),
                    position=np.array([0.9, 0.9, -0.6]), rotation=turned(-0.4), color=(220, 90, 80),
                    double_sided=True, morph_weights=[1.0, -0.8])
    pole = Object3D(make_box(), position=np.array([0.88, 0.85, -0.6]), scale=(0.04, 1.7, 0.04), color=(150, 130, 110))
    mesh, inverse_bind = figure_rig()
    v = mesh.vertices
    head = (v[:, 1] > 1.55)[:, None]
    swollen = (v - (0.0, 1.62, 0.0)) * 0.5 * head
    thin = v * (-0.3, 0.0, -0.3) * ~head
    j = skeleton()
    j["hips"].position = j["hips"].position + (-0.1, 0.0, 0.2)
    j["hips"].rotation = turned(-0.3)
    bend(j["right shoulder"], (0, 0, 1), 2.1)
    bend(j["right elbow"], (0, 0, 1), 0.8)
    bend(j["neck"], (0, 0, 1), 0.25)
    morphing = Mesh(v, mesh.faces, joints=mesh.joints, weights=mesh.weights, morph_targets=np.stack([swollen, thin]))
    waving = Object3D(morphing, skin=Skin(list(j.values()), inverse_bind), color=(200, 170, 140),
                      morph_weights=[1.0, 0.8])
    ground = floor(4.0, texture=checks(64, 4))
    sun = Light(direction=np.array([0.6, -1.0, -0.4]), shadows=True)
    lamp = PointLight(np.array([-1.0, 2.4, 1.6]), color=(255, 210, 160), range=6.0, shadows=True)
    return Shot([Object3D(ground, color=(255, 255, 255)), ball, flag, pole, waving],
                camera((0.1, 1.35, 3.5), (0.05, 0.8, 0.0)), [sun, lamp], {"background": Sky()})


def ortho():
    """The courtyard from above and to one side through an orthographic camera (added after 0.7.1), as in an
    isometric game: everything the same size however far off, with sun shadows, mirrors, outlines and fog
    measured from the camera's plane."""
    from unicode3d.examples.room import Courtyard
    from unicode3d.background import Fog
    court = Courtyard(1)
    court.animate(2.0)
    view = Camera(position=np.array([14.0, 16.0, 14.0]), target=np.zeros(3), projection="ortho", size=17.0, near=1.0)
    sun = Light(direction=np.array([0.5, -1.0, -0.35]), ambient=0.3, diffuse=0.6, shadows=True)
    return Shot(list(court.objects), view, sun,
                {"background": Sky(), "outline": 0.3, "fog": Fog(start=26.0, end=60.0, color=(200, 210, 230))})


def lettering():
    """3D lettering in the built-in font (added after 0.10.0): a sign extruded from merged runs and a line of
    stone blocks, a box per pixel each its own shade, hanging from the camera (so placed in its space) over a
    floor, with sun shadows."""
    from unicode3d.shapes import text_mesh
    view = Camera(position=np.array([2.0, 3.0, 9.0]), target=np.array([0.0, 0.5, 0.0]))
    sign, width = text_mesh("Hello, 3D!", depth=2.0)
    stones, across = text_mesh("Stone", depth=1.0, blocks=True, gap=0.2)
    shades = np.random.default_rng(3).uniform(0.55, 1.0, len(stones.faces) // 12)
    stones.face_colors = np.repeat(shades[:, None] * [0.8, 0.75, 0.68], 12, axis=0)
    tall = view.height_at(6.0)  # (of the view, 6 units in front of the camera)
    objects = [Object3D(block_mesh((0, -2.5, -2), (16, 0.1, 10)), color=(120, 110, 100)),
               Object3D(sign, np.array([0.0, tall * 0.2, -6.0]), color=(230, 180, 60), scale=tall * 1.3 / width,
                        parent=view),
               Object3D(stones, np.array([0.0, -tall * 0.2, -6.0]), scale=tall * 0.9 / across, parent=view,
                        color=(255, 255, 255))]
    sun = Light(direction=np.array([-0.3, -1.0, -0.5]), ambient=0.35, diffuse=0.7, shadows=True)
    return Shot(objects, view, sun, {"background": Gradient((40, 50, 80), (10, 10, 20))})


def detail():
    """Levels of detail (Renderer.simplify, added after 0.12.0): rows of knobbly balls and dice going off into the
    distance, the farther ones drawn from simpler copies of their meshes (detail.py) that differ by at most a pixel;
    shiny ones (the gold row) keep their shape. Shadows from the sun."""
    from unicode3d.detail import detail_levels
    from unicode3d.examples.dice import make_die
    knobbly = blob_mesh((1.0, 1.0, 1.0), rings=24, segments=32)
    knobbly.vertices = knobbly.vertices * (1.0 + 0.08 * np.sin(7.0 * knobbly.vertices[:, :1])
                                           * np.cos(5.0 * knobbly.vertices[:, 1:2]))
    die_mesh = make_die()
    detail_levels(knobbly)  # (made ahead, rather than a few a frame)
    detail_levels(die_mesh)
    objects = [Object3D(floor(30.0), color=(150, 150, 140))]
    for row, z in enumerate(np.arange(1.0, -24.0, -2.5)):
        objects += [Object3D(knobbly, np.array([-2.0, 0.6, z]), turned(0.3 * row), scale=0.6, color=(200, 90, 60)),
                    Object3D(die_mesh, np.array([0.0, 0.5, z]), turned(0.5 * row), scale=0.5, color=(240, 240, 230)),
                    Object3D(knobbly, np.array([2.0, 0.6, z]), turned(0.2 * row), scale=0.6, color=(230, 180, 60),
                             reflectivity=0.6)]
    sun = Light(direction=np.array([0.5, -1.0, -0.4]), shadows=True)
    return Shot(objects, camera((0.0, 2.2, 5.0), (0.0, 0.4, -6.0)), sun, {"background": Sky(), "simplify": 1.0})


def level():
    """A level as one big mesh (as loaded from glTF it often is): rooms of textured walls and pillars on a tiled
    floor, seen from inside, most of it out of view and left out in runs of its faces (raster.face_runs, added
    after 0.17.0, which keeps the picture the same), with a lamp's shadows and a mirror on a wall."""
    from unicode3d.shapes import merge_meshes
    walls, floors = [], []
    for x in range(-12, 13):
        for z in range(-12, 13):
            if x % 6 == 0 and z % 6 != 3 or z % 6 == 0 and x % 6 != 3 or (x % 6, z % 6) in ((2, 2), (4, 4)):
                walls.append(block_mesh((1.5 * x, 1.25, 1.5 * z), (1.5, 2.5, 1.5)))
            else:
                floors.append(block_mesh((1.5 * x, -0.05, 1.5 * z), (1.5, 0.1, 1.5)))
    brick, tile = checks(64, 8, (0.7, 0.45, 0.35), (0.5, 0.35, 0.3)), checks(64, 4)
    mesh = merge_meshes(walls + floors)
    n_walls = sum(len(w.faces) for w in walls)
    mesh.textures = [brick, tile]
    mesh.materials = (np.arange(len(mesh.faces)) >= n_walls).astype(np.int64)
    corners = mesh.vertices[mesh.faces]  # (texture coordinates from where each corner is: walls x + z and y)
    mesh.uvs = np.where(mesh.materials[:, None, None] == 0, np.stack([corners[..., 0] + corners[..., 2],
                                                                     corners[..., 1]], axis=-1),
                        corners[..., [0, 2]]) / 3.0
    mirror = Object3D(panel(1.6, 1.4), np.array([1.5, 0.5, 0.76]), color=(160, 160, 160), reflectivity=0.8)
    lights = [Light(ambient=0.3, diffuse=0.3), PointLight(np.array([3.0, 2.2, 4.5]), range=10.0, shadows=True)]
    return Shot([Object3D(mesh, color=(255, 255, 255)), mirror],
                camera((1.5, 1.4, 7.5), (4.5, 1.0, -4.5), fov=70.0), lights, {"background": (10, 10, 14)})


def particles():
    """Particles (added after 0.20.1): a fountain of glowing sparks bouncing on a floor and fading from yellow to red,
    chips of stone tumbling (boxes), smoke puffs facing the camera that grow and thin as they rise, and a cut-out
    sprite (Object3D.billboard) beside them, with sun shadows. Particles are moved a fixed number of steps from a
    seed, so the picture is the same every time."""
    from unicode3d.scene import Particles
    sparks = Particles(gravity=(0.0, -9.8, 0.0), ground=0.0, bounce=0.4, emissive=0.9, seed=1)
    chips = Particles(make_box(), gravity=(0.0, -9.8, 0.0), ground=0.0, bounce=0.2, specular=0.3, cast_shadows=True,
                      seed=2)
    smoke = Particles(face_camera=True, gravity=(0.3, 0.0, 0.0), seed=3)  # (drifting in a breeze)
    for step in range(45):
        sparks.emit((0.0, 0.3, 0.0), 40, velocity=(0.0, 4.5, 0.0), speed=(0.5, 2.0), life=(0.8, 1.4), size=0.07,
                    size_end=0.03, color=(255, 230, 120), color_end=(255, 60, 20), spin=(-6.0, 6.0))
        if step % 15 == 0:
            chips.emit((1.6, 0.3, 0.4), 12, velocity=(0.0, 3.0, 0.0), speed=(1.0, 2.5), life=3.0, size=(0.08, 0.16),
                       color=(140, 135, 125), spin=(-8.0, 8.0))
        if step % 3 == 0:
            smoke.emit((-1.4, 0.2, -0.6), 2, velocity=(0.0, 0.9, 0.0), speed=0.1, radius=0.15, life=(2.5, 3.0),
                       size=0.3, size_end=1.0,
                       color=(150, 150, 155), opacity=0.7, opacity_end=0.0, spin=(-0.6, 0.6))
        for p in (sparks, chips, smoke):
            p.update(1.0 / 30.0)
    y, x = np.mgrid[0:32, 0:32]
    star = np.zeros((32, 32, 4))
    star[..., :3] = (1.0, 0.85, 0.3)
    star[..., 3] = (np.abs(x - 15.5) + np.abs(y - 15.5) < 14).astype(float)
    sprite = Object3D(panel(1.0, 1.0, star), np.array([2.4, 1.2, -1.0]), color=(255, 255, 255), billboard=True)
    objects = [Object3D(floor(5.0), color=(110, 120, 100)), sparks, chips, smoke, sprite]
    sun = Light(direction=np.array([-0.4, -1.0, -0.6]), shadows=True)
    return Shot(objects, camera((1.0, 3.0, 6.5), (0.3, 1.2, 0.0)), sun, {"background": Sky()})


def lines():
    """Lines (added after 0.20.1): a floor grid 1 pixel wide running off to the horizon, axes 2 pixels wide, a wire
    box (its edges partly behind a solid box, partly in front), a laser beam in world units (thinner further off),
    and a path in steps of changing colour, all unshaded, with a lit box among them and its sun shadow."""
    from unicode3d.scene import Lines
    reach = np.arange(-30.0, 31.0)
    edge = np.full(len(reach), 30.0)
    zero = np.zeros(len(reach))
    grid = Lines(np.r_[np.c_[reach, zero, -edge], np.c_[-edge, zero, reach]],
                 np.r_[np.c_[reach, zero, edge], np.c_[edge, zero, reach]], color=(80, 95, 110))
    axes = Lines(np.zeros((3, 3)), 1.5 * np.eye(3), colors=[(255, 70, 70), (70, 255, 70), (90, 130, 255)], width=2)
    corners = np.array([(x, y, z) for x in (-0.7, 0.7) for y in (0.0, 1.4) for z in (-0.7, 0.7)])
    pairs = [(a, b) for a in range(8) for b in range(a + 1, 8)
             if np.isclose(np.abs(corners[a] - corners[b]).sum(), 1.4)]
    wire = Lines(corners[[a for a, _ in pairs]], corners[[b for _, b in pairs]], width=2, color=(255, 210, 90),
                 position=(1.8, 0.0, -0.4))
    beam = Lines([(-3.0, 1.2, 2.0)], [(4.0, 0.6, -12.0)], width=0.06, units="world", color=(255, 60, 170))
    t = np.linspace(0.0, 1.0, 25)
    path = np.c_[-2.5 + 3.0 * t, 0.02 + 0.6 * np.sin(np.pi * t), 1.5 - 4.0 * t]
    hue = (np.c_[255 * (1 - t), 120 + 100 * t, 255 * t]).astype(int)
    trail = Lines(path[:-1], path[1:], colors=hue[:-1], width=3)
    box = Object3D(make_box(), np.array([1.8, 0.5, -0.4]), color=(170, 140, 100))
    objects = [Object3D(floor(30.0), np.array([0.0, -0.01, 0.0]), color=(40, 44, 50)), box, grid, axes, wire, beam,
               trail]
    sun = Light(direction=np.array([-0.4, -1.0, -0.6]), shadows=True)
    return Shot(objects, camera((-1.5, 2.6, 6.0), (0.6, 0.4, -1.0)), sun, {"background": Sky()})


def numbered_sheet(**options):
    """A sprite sheet of 8 rows (directions) by 2 columns (frames): each cell a rounded figure in its row's hue with
    the row's number on it, its frames told apart by a band across the top; clear around."""
    from unicode3d.scene import SpriteSheet
    from unicode3d.shapes import text_bitmap
    w, h = 16, 24
    image = np.zeros((8 * h, 2 * w, 4))
    y, x = np.mgrid[0:h, 0:w]
    body = ((x - 7.5) / 7.5) ** 2 + ((y - 13.0) / 10.5) ** 4 < 1.0
    for row in range(8):
        hue = np.array([np.cos(row * np.pi / 4), np.cos(row * np.pi / 4 - 2.1), np.cos(row * np.pi / 4 + 2.1)])
        for frame in range(2):
            cell = image[row * h:(row + 1) * h, frame * w:(frame + 1) * w]
            cell[body] = (*(0.55 + 0.4 * hue), 1.0)
            cell[3 + 2 * frame:5 + 2 * frame, 3:13] = (0.1, 0.1, 0.1, 1.0)
            digit = text_bitmap(str(row))[:7]
            cell[9:16, 6:11][digit] = (1.0, 1.0, 1.0, 1.0)
    return SpriteSheet(image, columns=2, height=1.2, animations={"wave": (0, 2, 2.0)}, **options)


def sprites():
    """Sprites with eight sides (added after 0.20.1): eight standing in a ring, each turned to face outwards, so
    that the camera sees each from another side (each shows its row's number: 0 from the front, then on round
    clockwise); one in the middle of a mirrored sheet (rows 5-7 are 3-1 flipped), seen in a mirror behind them from
    its far side; an upright billboard and one turned square to the picture, from above; sun shadows."""
    from unicode3d.scene import Sprite
    sheet = numbered_sheet()
    objects = [Object3D(floor(6.0), color=(110, 120, 100))]
    for k in range(8):
        a = k * np.pi / 4
        sprite = Sprite(sheet, position=np.array([2.2 * np.sin(a), 0.0, 2.2 * np.cos(a) - 0.5]),
                        rotation=quat_axis_angle(UP, a))
        sprite.time = 0.5 * (k % 2)  # (every other one on its second frame)
        objects.append(sprite)
    objects.append(Sprite(numbered_sheet(mirrored=True, rows=8), position=np.array([0.0, 0.0, -0.5]),
                          rotation=quat_axis_angle(UP, 2.4)))
    card = panel(0.8, 1.2, np.dstack([np.full((8, 8, 3), (0.3, 0.7, 0.4)), np.ones((8, 8))]))
    objects += [Object3D(card, np.array([-3.4, 0.0, 1.2]), color=(255, 255, 255), billboard="upright"),
                Object3D(card, np.array([3.4, 0.0, 1.2]), color=(255, 255, 255), billboard=True),
                Object3D(Mesh(np.array([(-3.0, 0, -4.0), (3, 0, -4.0), (3, 2.5, -4.0), (-3, 2.5, -4.0)]),
                              np.array([(0, 1, 2), (0, 2, 3)])), color=(20, 20, 25), reflectivity=0.9)]
    sun = Light(direction=np.array([-0.4, -1.0, -0.6]), shadows=True)
    return Shot(objects, camera((0.6, 3.4, 6.5), (0.0, 0.5, -0.5)), sun, {"background": Sky()})


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
    "view-shadows": view_shadows,
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
    "gltf-skinned": gltf_skinned,
    "skinned": skinned,
    "morph": morph,
    "ortho": ortho,
    "lettering": lettering,
    "detail": detail,
    "level": level,
    "particles": particles,
    "lines": lines,
    "sprites": sprites,
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


def cells_picture(frame):
    """What a terminal shows for the saved cells, as (rows * 16, cols * 8, 3) uint8 (see
    unicode3d.pictures)."""
    return pictures.cells_picture(frame["chars"], frame["fg"], frame["bg"], frame.get("attrs"), str(frame["glyphs"]))


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
