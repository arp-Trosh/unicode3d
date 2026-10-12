# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
"""A sprite sheet baked with unicode3d itself: the fighter, a spiky-haired martial artist in an orange gi,
built from blobs and spikes, posed frame by frame and drawn from eight sides into a picture, eight rows of
directions by 18 columns of frames:

    idle    columns 0-3    a fighting stance, breathing
    walk    columns 4-9
    punch   columns 10-13  wind up, strike, hold, back
    charge  columns 14-15  both hands cupped at the hip, gathering a blast
    blast   columns 16-17  both hands thrust forward

fighter_sheet() bakes it (about a second) and gives the SpriteSheet; the room demo has him walking about.

    python -m unicode3d.examples.fighter [sheet.png]     # save the picture (fighter.png by default)
"""
import math
import sys

import numpy as np

from unicode3d import Camera, Light, Mesh, Node, Object3D, Renderer, SpriteSheet
from unicode3d.color import linear_to_srgb
from unicode3d.shapes import blob_mesh, block_mesh, merge_meshes
from unicode3d.transforms import quat_axis_angle, quat_between, quat_mul, quat_to_matrix

CELL = (72, 96)             # pixels a cell, across and down
SPAN = (1.65, 2.2)          # how much of the world a cell shows, across and up (the figure is about 2 tall;
                            # in the shape of CELL)
FLOOR = 0.04                # how far above a cell's bottom his feet stand, in the world
DIRECTIONS = 8
ANIMATIONS = {"idle": (0, 4, 4.0), "walk": (4, 6, 9.0), "punch": (10, 4, 10.0), "charge": (14, 2, 8.0),
              "blast": (16, 2, 8.0)}

GI = (250, 125, 25)         # orange
UNDER = (35, 70, 170)       # blue: undershirt, belt, wristbands, boots
SKIN = (250, 200, 160)
HAIR = (25, 22, 30)
EYES = (20, 20, 30)
X, Y, Z = np.eye(3)


def turn(x=0.0, y=0.0, z=0.0):
    """A rotation (quaternion) by x about the x axis, then z about z, then y about y (radians)."""
    q = quat_axis_angle(X, x)
    q = quat_mul(quat_axis_angle(Z, z), q)
    return quat_mul(quat_axis_angle(Y, y), q)


def spike(base, direction, length, radius, sides=5):
    """A cone from a ring of `radius` around base to a point `length` along direction."""
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    a = 2 * np.pi * np.arange(sides) / sides
    ring = np.c_[radius * np.cos(a), np.zeros(sides), radius * np.sin(a)]
    vertices = np.r_[ring, [(0.0, length, 0.0)], [(0.0, 0.0, 0.0)]] @ quat_to_matrix(quat_between(Y, d)).T + base
    faces = [(k, sides, (k + 1) % sides) for k in range(sides)] + [(k, (k + 1) % sides, sides + 1)
                                                                    for k in range(sides)]
    return Mesh(vertices, np.array(faces))


def part(meshes, colors, parent, at=(0.0, 0.0, 0.0)):
    return Object3D(merge_meshes(meshes, colors), position=np.array(at, float), parent=parent,
                    color=(255, 255, 255), specular=0.2)


def figure():
    """The figure's joints (Nodes, by name; "root" stands at his feet, facing +z) and parts (Object3Ds)."""
    j = {"root": Node()}

    def joint(name, parent, at):
        j[name] = Node(position=np.array(at, float), parent=j[parent])

    joint("hips", "root", (0.0, 0.9, 0.0))
    joint("chest", "hips", (0.0, 0.12, 0.0))
    joint("head", "chest", (0.0, 0.5, 0.0))
    for side, s in (("l", 1.0), ("r", -1.0)):
        joint("shoulder_" + side, "chest", (0.29 * s, 0.4, 0.0))
        joint("elbow_" + side, "shoulder_" + side, (0.0, -0.27, 0.0))
        joint("hip_" + side, "hips", (0.11 * s, 0.0, 0.0))
        joint("knee_" + side, "hip_" + side, (0.0, -0.43, 0.0))
    parts = []
    # Head: a face, eyes and brows, ears, and the hair: a cap and spikes up, out and back, bangs over the brow.
    face = [blob_mesh((0.16, 0.18, 0.16), (0, 0.17, 0), 8, 12), blob_mesh((0.075, 0.08, 0.07), (0, 0.0, 0), 4, 8)]
    for s in (1, -1):
        face.append(blob_mesh((0.03, 0.045, 0.03), (0.15 * s, 0.17, 0.0), 3, 6))
    eyes = [block_mesh((0.055 * s, 0.17, 0.138), (0.035, 0.045, 0.02)) for s in (1, -1)]
    brows = [block_mesh((0.06 * s, 0.225, 0.135), (0.06, 0.018, 0.02),
                        quat_to_matrix(quat_axis_angle(Z, -0.35 * s))) for s in (1, -1)]
    hair = [blob_mesh((0.165, 0.15, 0.16), (0, 0.25, -0.025), 6, 12)]
    for d, length, r in (((0, 1, -0.3), 0.36, 0.09), ((-0.55, 1, -0.25), 0.33, 0.085), ((0.55, 1, -0.25), 0.33, 0.085),
                         ((-1, 0.55, -0.2), 0.3, 0.08), ((1, 0.55, -0.2), 0.3, 0.08), ((-0.4, 0.9, 0.35), 0.24, 0.07),
                         ((0.4, 0.9, 0.35), 0.24, 0.07), ((0, 0.35, -1), 0.3, 0.09), ((-0.7, 0.1, -1), 0.26, 0.08),
                         ((0.7, 0.1, -1), 0.26, 0.08), ((0, -0.45, -1), 0.24, 0.08), ((-1, -0.1, -0.5), 0.2, 0.06),
                         ((1, -0.1, -0.5), 0.2, 0.06)):
        d = np.array(d, float)
        hair.append(spike(np.array((0, 0.25, -0.03)) + 0.1 * d / np.linalg.norm(d), d, length, r))
    for x, d in ((-0.07, (-0.5, -0.4, 1)), (0.0, (0.15, -0.5, 1)), (0.075, (0.6, -0.3, 1))):  # (bangs)
        hair.append(spike((x, 0.29, 0.1), d, 0.13, 0.045, 4))
    parts.append(part(face + eyes + brows + hair, [SKIN] * len(face) + [EYES] * 4 + [HAIR] * len(hair), j["head"]))
    # Body: the gi's top, the blue undershirt showing at the neck, the belt and its knot.
    body = [blob_mesh((0.24, 0.32, 0.15), (0, 0.28, 0), 8, 12), blob_mesh((0.31, 0.12, 0.16), (0, 0.5, 0), 6, 12),
            block_mesh((0, 0.6, 0.11), (0.12, 0.12, 0.05), quat_to_matrix(quat_axis_angle(Z, math.pi / 4))),
            blob_mesh((0.235, 0.075, 0.16), (0, 0.0, 0), 4, 12),
            blob_mesh((0.05, 0.05, 0.035), (0.09, 0.0, 0.15), 3, 6),
            block_mesh((0.11, -0.11, 0.155), (0.05, 0.14, 0.02))]
    parts.append(part(body, [GI, GI, UNDER, UNDER, UNDER, UNDER], j["chest"], (0, -0.12, 0)))
    parts.append(part([blob_mesh((0.2, 0.12, 0.13), (0, 0.0, 0), 6, 12)], [GI], j["hips"]))
    for side in "lr":
        # Arms: a blue sleeve, the forearm, a blue wristband and a fist.
        parts.append(part([blob_mesh((0.09, 0.11, 0.09), (0, -0.05, 0), 5, 8),
                           blob_mesh((0.075, 0.1, 0.075), (0, -0.17, 0), 5, 8)], [UNDER, SKIN], j["shoulder_" + side]))
        parts.append(part([blob_mesh((0.068, 0.11, 0.068), (0, -0.08, 0), 5, 8),
                           blob_mesh((0.075, 0.05, 0.075), (0, -0.19, 0), 3, 8),
                           blob_mesh((0.068, 0.068, 0.07), (0, -0.265, 0.01), 4, 8)], [SKIN, UNDER, SKIN],
                          j["elbow_" + side]))
        # Legs: baggy trousers, and boots with their toes forward.
        parts.append(part([blob_mesh((0.115, 0.25, 0.115), (0, -0.2, 0), 6, 10)], [GI], j["hip_" + side]))
        parts.append(part([blob_mesh((0.1, 0.2, 0.1), (0, -0.14, 0), 6, 10),
                           blob_mesh((0.08, 0.12, 0.085), (0, -0.33, 0.0), 5, 8),
                           blob_mesh((0.075, 0.055, 0.13), (0, -0.42, 0.05), 4, 8)], [GI, UNDER, UNDER],
                          j["knee_" + side]))
    return j, parts


def pose(j, animation, frame):
    """Set the joints for `frame` of `animation`."""
    for node in j.values():
        node.rotation = turn()
    j["hips"].position = np.array([0.0, 0.9, 0.0])
    set_ = lambda name, **angles: setattr(j[name], "rotation", turn(**angles))

    def stance(bob=0.0):  # (feet apart, knees bent, fists up)
        j["hips"].position = np.array([0.0, 0.85 - bob, 0.0])
        set_("hips", y=-0.35)
        set_("chest", y=0.2, x=0.08)
        set_("head", y=0.15)
        for s, sign in (("l", 1), ("r", -1)):
            set_("hip_" + s, x=-0.35 if s == "l" else 0.25, z=0.15 * sign)
            set_("knee_" + s, x=0.5 if s == "l" else 0.3)
            set_("shoulder_" + s, x=-0.55 - (0.3 if s == "l" else 0.0) - bob, z=0.25 * sign)
            set_("elbow_" + s, x=-1.9 + bob)

    if animation == "idle":
        stance(0.015 * (1 - math.cos(2 * math.pi * frame / 4)))
    elif animation == "walk":
        p = 2 * math.pi * frame / 6
        j["hips"].position = np.array([0.0, 0.9 - 0.035 * abs(math.sin(p)), 0.0])
        set_("chest", x=0.08, y=0.12 * math.sin(p))
        set_("hips", y=-0.1 * math.sin(p))
        for s, sign in (("l", 1), ("r", -1)):
            swing = 0.55 * math.sin(p) * sign
            set_("hip_" + s, x=-swing)
            set_("knee_" + s, x=0.15 + 0.8 * max(0.0, math.sin(p - 0.9) * -sign))
            set_("shoulder_" + s, x=0.5 * math.sin(p) * sign, z=0.12 * sign)
            set_("elbow_" + s, x=-0.5)
    elif animation == "punch":
        stance()
        reach = (0.3, 1.0, 1.0, 0.5)[frame]  # (how far into the punch)
        back = (1.0, 0.0, 0.0, 0.0)[frame]
        set_("hips", y=-0.35 + 0.5 * reach)
        set_("chest", y=0.2 + 0.35 * reach - 0.3 * back, x=0.08)
        set_("head", y=0.15 - 0.4 * reach)
        set_("shoulder_r", x=-0.4 - 1.15 * reach + 0.6 * back, z=-0.25 + 0.05 * reach, y=0.2 * reach)
        set_("elbow_r", x=-1.9 * (1 - reach) - 0.6 * back)
        set_("shoulder_l", x=-0.7 + 0.3 * reach, z=0.25)
        set_("elbow_l", x=-2.1)
    elif animation == "charge":
        shake = 0.03 * (1 if frame else -1)
        stance()
        set_("hips", y=0.5)
        set_("chest", y=0.45 + shake, x=0.15)
        set_("head", y=-0.6)
        for s, sign in (("l", 1), ("r", -1)):
            set_("shoulder_" + s, x=0.55 + shake, z=-0.1 * sign + (0.45 if s == "l" else 0.0), y=0.3)
            set_("elbow_" + s, x=-1.6)
    elif animation == "blast":
        shake = 0.04 * (1 if frame else -1)
        stance()
        set_("hips", y=-0.2)
        set_("chest", y=0.25 + shake, x=0.05)
        set_("head", y=-0.05)
        for s, sign in (("l", 1), ("r", -1)):
            set_("shoulder_" + s, x=-1.5 + shake, z=-0.32 * sign, y=0.25)
            set_("elbow_" + s, x=-0.1)


def edges(rgb, alpha):
    """rgb (drawn over nothing, so darkened where alpha says a pixel is partly covered) as the figure's own
    colours: undarkened at its edges, and spread a few pixels out over the clear ground around it, which a
    texture's filtering blends in at the edges (black there would ring the sprite with a dark line)."""
    rgb = np.where(alpha[..., None] > 0.0, rgb / np.maximum(alpha, 1e-6)[..., None], 0.0)
    known = alpha > 0.0
    for _ in range(3):  # (each clear pixel next to known ones takes their average)
        total, count = np.zeros_like(rgb), np.zeros(alpha.shape)
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            total += np.roll(rgb * known[..., None], (dy, dx), axis=(0, 1))
            count += np.roll(known, (dy, dx), axis=(0, 1))
        grow = ~known & (count > 0)
        rgb[grow] = total[grow] / count[grow][:, None]
        known = known | grow
    return rgb


def bake(cell=CELL, directions=DIRECTIONS):
    """The sheet's picture: (H, W, 4) 0..1 sRGB with alpha, a row for each direction (the front, then each
    turned a further 360 / directions degrees clockwise as seen from above) and a column for each frame."""
    j, parts = figure()
    columns = sum(frames for _, frames, _ in ANIMATIONS.values())
    w, h = cell
    picture = np.zeros((directions * h, columns * w, 4))
    renderer = Renderer(w, h, cell_pixels=(1, 1), cell_aspect=1.0, fog=0.0, outline=0.45)
    centre = np.array([0.0, SPAN[1] / 2 - FLOOR, 0.0])
    for name, (first, frames, _) in ANIMATIONS.items():
        for f in range(frames):
            pose(j, name, f)
            for row in range(directions):
                a = 2 * math.pi * row / directions  # (the camera that far round, counter-clockwise from above)
                way = np.array([math.sin(a), 0.12, math.cos(a)])
                camera = Camera(position=centre + 10 * way, target=centre, projection="ortho", size=SPAN[1])
                # (the light comes from above and to the left of the camera, wherever it is)
                left = np.array([-math.cos(a), 0.0, math.sin(a)])
                light = Light(direction=-(way + 0.9 * left + np.array([0.0, 1.2, 0.0])), ambient=0.45, diffuse=0.65,
                              specular=0.2)
                fb = renderer.render(parts, camera, light)
                y, x = row * h, (first + f) * w
                picture[y:y + h, x:x + w, :3] = linear_to_srgb(np.clip(edges(fb.rgb, fb.alpha), 0.0, 1.0))
                picture[y:y + h, x:x + w, 3] = fb.alpha
    return picture


_sheet = None


def fighter_sheet(height=1.9):
    """The fighter's SpriteSheet (baked on the first call), his feet at a sprite's position, `height` tall."""
    global _sheet
    if _sheet is None or _sheet[0] != height:
        picture = bake()
        columns = picture.shape[1] // CELL[0]
        cell_height = height * SPAN[1] / 2.0  # (the figure is about 2 of the cell's SPAN[1] tall)
        _sheet = (height, SpriteSheet(picture, columns=columns, directions=DIRECTIONS, height=cell_height,
                                      origin=(0.5, FLOOR / SPAN[1]),
                                      animations={k: (a, n, fps) for k, (a, n, fps) in ANIMATIONS.items()}))
    return _sheet[1]


def main():
    from PIL import Image
    out = sys.argv[1] if len(sys.argv) > 1 else "fighter.png"
    Image.fromarray(np.round(bake() * 255).astype(np.uint8), "RGBA").save(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
