# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
"""A marionette theatre, to show skinned meshes: python -m unicode3d.examples.skinned

A puppet hangs on strings from a control bar over the stage, and you work the bar: WASD move it about the stage
(away, left, back, right, as the view looks), , and . turn it (Left/Right too, from the front), R/F raise and
lower it, Q/E rock the leg bar (lifting one knee, then the other), Z/X pull the left and right hand strings,
M (held) sings.
Auto-walk (Space) rocks the leg bar by itself while the bar moves and turns the puppet to face where it goes.
C opens and closes the curtains, O switches between the view from the front (orthographic; the mouse wheel
zooms) and walking about the theatre yourself (Up/Down walk, Left/Right turn, drag the mouse to look; steps at
the side lead up onto the stage). The panel sets gravity ([ and ]), from the Moon's to twice the Earth's.
Esc quits.

The puppet isn't animated: it is a rag doll, a point for each of its joints held apart by rods, pulled down by
gravity, held up by its strings (which only pull when taut) and stopped by the stage floor (simulate, a Numba
kernel). Each frame its skeleton's joints are turned to point along the points, and its body, one mesh skinned to
them (Mesh.joints and Mesh.weights, Object3D(skin=Skin(joints))), bends smoothly between them. Its nose, eyes, hat
and hands are ordinary objects whose parents are joints. The curtains are skinned too: pleated sheets, each on a row of joints along the rail and a row along the
hem, gathered by moving and squeezing the joints (the hem's following the rail's a little late, so they sway), and
billowing by two morph targets (Mesh.morph_targets): a wave running across the cloth, weighted by the cosine and sine
of the time (Object3D.morph_weights), higher while they move and dying down to a draught after. Singing is a
morph target too, on the body (the jaw dropping) and on the mouth (an object of its own on the head joint, morphed
without a skin), weighted with a little vibrato. A second target squashes the head (the eyes, nose, mouth and hat,
on its joint, moved to match): a puppet landing on the stage after dangling plays a short squash and stretch, a Clip
of the weights; Enter makes the puppets gasp (the same target weighted below 0, stretching the head, the mouth open)
and turns them to stone at its height. The chorus does what the lead puppet is told a moment after the one before it,
singing included: a canon.
"""
import argparse
import time
from bisect import bisect_right

import numpy as np
from numba import njit

from .hud import StatusBar
from .rigs import figure, figure_rig, skeleton
from ..animation import Animation, Clip, Track
from ..background import Gradient
from ..keys import Key, MouseEvent
from ..mesh import Mesh, make_box
from ..scene import Camera, Light, Lines, Model, Node, Object3D, PointLight, Renderer, Skin
from ..shapes import blob_mesh, block_mesh, merge_meshes
from ..terminal import add_display_args, display_options, run
from ..transforms import quat_axis_angle, quat_from_matrix, quat_mul, quat_to_matrix
from ..ui import Choice, Panel, Slider, Toggle

# The puppet.
GRAVITY = 9.8
STEP = 1.0 / 120.0      # simulation step, seconds
ITERATIONS = 12         # constraint passes a step
HINGE_STIFFNESS = 0.3   # of a hinge's correction made each pass
FOLD = np.radians(135.0)  # how far knees and elbows bend
SPLAY = np.radians(45.0)  # how far a thigh swings out to the side
DAMPING = 2.4           # of the limbs' velocity apart from the whole puppet's, a second (stiff joints)
AIR = 0.4               # of the whole puppet's velocity, a second
FRICTION = 0.8          # of the floor: a point on it slides this much less a step than the floor pushes it up
BAR_HOME = 2.55         # the bar's height over the stage to start with
BAR_LOW, BAR_HIGH = 2.1, 3.3  # and the lowest and highest it goes
BAR_SPEED = 0.6         # units a second
BAR_TURN = 1.5          # radians a second
STRIDE = 0.9            # bar travel for one full rock of the leg bar when walking by itself
ROCK = 0.65             # the leg bar's rock, radians
ROCK_SPEED = 2.0        # how fast the leg bar rocks, radians a second
LEG_BAR = 0.3           # half its length
HAND_SLACK, HAND_PULL = 0.2, 0.8  # hand strings: slack at rest, how much shorter pulled in full
PULL_SPEED = 0.8        # how fast a hand string is pulled in or let out, units a second
SAG = 0.05              # how much longer the head, back and hand strings are than the puppet standing needs
WOOD = (225, 225, 225)  # (the wood texture's colours, a little darker)
HANDS = (200, 160, 110)
JAW = 0.055             # how far the jaw drops singing
SING_ATTACK, SING_RELEASE = 0.1, 0.2  # seconds to open the mouth and to close it
VIBRATO = 4.5           # wobbles a second, while singing
HEAD = 0.29             # the head's height, from its joint
SQUASH_DROP, SQUASH_WIDEN = 0.06, 0.25  # a squashed head's top comes down this far, and its middle this much wider
AIRBORNE = 0.4          # seconds with both feet off the stage (by CLEAR) that make touching it again a landing
CLEAR = 0.08            # (walking lifts a foot less than this, or not for as long)
GASP = 0.15             # seconds from Enter to the gasp's height, when the puppets turn to stone

# The chorus.
MAX_PUPPETS = 14        # in two rows of seven
LAG = 0.3               # seconds each puppet is behind the one before in doing what the lead does
HATS = [(150, 40, 45), (45, 90, 160), (50, 130, 70), (190, 150, 40), (120, 60, 150), (200, 100, 40), (40, 140, 150)]


def chorus_at(k):
    """Where puppet k of the chorus starts (x, z): the lead at the centre, the rest out to each side, then a row behind."""
    return (0.0, -0.8, 0.8, -1.6, 1.6, -2.4, 2.4)[k % 7], -1.8 - 1.1 * (k // 7)


# The theatre (x across, y up, z towards the audience; the stage's front edge at z = 0).
STAGE = 0.8             # the stage floor's height
STAGE_X, STAGE_BACK = 4.0, -4.3  # its half width and its back
OPENING_X, OPENING_Y = 3.6, 3.7  # the proscenium opening's half width and height over the stage
BAR_X, BAR_Z = 2.9, (-3.5, -0.6)  # where the bar may go
CURTAIN_Z = -0.35
HALL_X, HALL_BACK, HALL_Y = 6.5, 10.0, 6.0
ROWS_Z = (2.2, 3.3, 4.4, 5.5)  # rows of seats
SEATS_X = (0.9, 5.3)    # each side of the aisle
STEPS_X, STEPS_Z = (STAGE_X, 4.9), (-1.6, -0.4)  # steps up onto the stage, at the right
EYE = 1.6
WALK, TURN, LOOK = 2.2, 1.8, 0.006
REACH = 0.25            # the walker's radius


# ----- the rag doll --------------------------------------------------------------------------------------------

def rest_points():
    """(names, (N, 3) positions) of the simulated points, the figure standing at the origin as skeleton() makes it:
    its joints and a few more that fix which way the torso and head face, and where the hands and feet end."""
    j = skeleton()
    names = list(j)
    world = Skin(list(j.values())).world_matrices()
    at = {n: world[k, :3, 3] for k, n in enumerate(names)}
    axis = {n: world[k, :3, 1] for k, n in enumerate(names)}
    pts = dict(at)
    pts["head top"] = at["head"] + 0.29 * axis["head"]
    pts["head front"] = at["head"] + (0.0, 0.13, 0.1)
    pts["pelvis front"] = at["hips"] + (0.0, 0.0, 0.1)
    pts["chest front"] = at["chest"] + (0.0, 0.0, 0.1)
    pts["back"] = at["chest"] + (0.0, 0.05, -0.12)
    for side, name in ((1.0, "right"), (-1.0, "left")):
        pts[f"{name} ear"] = at["head"] + (0.1 * side, 0.13, 0.0)
        pts[f"{name} hand"] = at[f"{name} wrist"] + 0.1 * axis[f"{name} wrist"]
        pts[f"{name} foot"] = at[f"{name} ankle"] + 0.08 * axis[f"{name} ankle"]
    names = list(pts)
    return names, np.array([pts[n] for n in names])


class Doll:
    """A rag doll's points, rods, hinges and strings (the same for every puppet)."""

    def __init__(self):
        self.names, self.rest = rest_points()
        i = self.at = {n: k for k, n in enumerate(self.names)}
        rods = []  # (a, b, stiffness, kind, length): kind 0 keeps the length, 1 only stops them coming closer,
        # 2 parting

        def rod(a, b, k=1.0, kind=0, length=None):
            if length is None:
                length = np.linalg.norm(self.rest[i[b]] - self.rest[i[a]])
            rods.append((i[a], i[b], k, kind, length))

        for members in (["hips", "spine", "pelvis front", "right hip", "left hip"],  # rigid pelvis, chest and head
                        ["chest", "neck", "chest front", "back", "right shoulder", "left shoulder"],
                        ["head", "head top", "right ear", "left ear", "head front"]):
            for x in range(len(members)):
                for y in range(x + 1, len(members)):
                    rod(members[x], members[y])
        for a, b, k in (("spine", "chest", 1.0), ("hips", "chest", 0.3), ("pelvis front", "chest front", 0.3),
                        ("right hip", "right shoulder", 0.3), ("left hip", "left shoulder", 0.3),
                        ("neck", "head", 1.0), ("chest", "head", 0.5), ("chest", "head top", 0.4),
                        ("neck", "right ear", 0.5), ("neck", "left ear", 0.5),
                        ("chest front", "head front", 0.4), ("chest front", "right ear", 0.3),
                        ("chest front", "left ear", 0.3)):  # (the head follows the way the body faces)
            rod(a, b, k)
        for name in ("right", "left"):
            for a, b in (("shoulder", "elbow"), ("elbow", "wrist"), ("wrist", "hand"), ("elbow", "hand"),
                         ("hip", "knee"), ("knee", "ankle"), ("ankle", "foot"), ("knee", "foot")):
                rod(f"{name} {a}", f"{name} {b}")
            for a in ("elbow", "wrist", "hand"):  # arms kept out of the body
                rod(f"{name} {a}", "hips", kind=1, length=0.17)
                rod(f"{name} {a}", "chest", kind=1, length=0.17)
        for a in ("knee", "ankle", "foot"):  # and legs out of each other
            rod(f"right {a}", f"left {a}", kind=1, length=0.13)
        # knees and elbows fold no further than FOLD (a rod keeping the limb's ends apart)
        for name in ("right", "left"):
            for a, b, c in (("hip", "knee", "ankle"), ("shoulder", "elbow", "wrist")):
                u, w = (np.linalg.norm(self.rest[i[f"{name} {y}"]] - self.rest[i[f"{name} {x}"]])
                        for x, y in ((a, b), (b, c)))
                rod(f"{name} {a}", f"{name} {c}", kind=1, length=np.sqrt(u * u + w * w + 2.0 * u * w * np.cos(FOLD)))
        # and thighs swing out sideways no further than SPLAY (from the other hip, a knee is further off the more it
        # does, however far it swings forward or back)
        for name, other in (("right", "left"), ("left", "right")):
            hip, knee, far = (self.rest[i[n]] for n in (f"{name} hip", f"{name} knee", f"{other} hip"))
            out = hip + np.linalg.norm(knee - hip) * np.array(
                [np.sign((hip - far)[0]) * np.sin(SPLAY), -np.cos(SPLAY), 0.0])
            rod(f"{name} knee", f"{other} hip", kind=2, length=np.linalg.norm(out - far))
        self.rod_a = np.array([r[0] for r in rods], np.int64)
        self.rod_b = np.array([r[1] for r in rods], np.int64)
        self.rod_k = np.array([r[2] for r in rods], float)
        self.rod_kind = np.array([r[3] for r in rods], np.int64)
        self.rod_len = np.array([r[4] for r in rods], float)
        # Hinges (a, b, c, f0, f1, sign): b may stick out from the line a-c only one way (sign) about the axis f1
        # to f0. Knees only: elbows, loose as a marionette's are, bend any way (though no further than FOLD).
        self.hinges = np.array([(i[f"{n} hip"], i[f"{n} knee"], i[f"{n} ankle"], i["right hip"], i["left hip"], 1)
                                for n in ("right", "left")], np.int64)
        # (a hinge lets go when a and c are closer than this: a limb folded up)
        self.hinge_reach = 0.5 * np.linalg.norm(self.rest[self.hinges[:, 1]] - self.rest[self.hinges[:, 0]], axis=1)
        heavy = {"hips": 3.0, "spine": 2.0, "chest": 3.0, "head": 2.0, "pelvis front": 1.0, "chest front": 1.0}
        self.inv_mass = np.array([1.0 / heavy.get(n, 0.6) for n in self.names])
        self.radius = np.full(len(self.names), 0.04)
        self.radius[[i["hips"], i["spine"], i["chest"]]] = 0.12
        self.radius[i["head top"]] = 0.05
        # Strings: the point each holds, where it is tied on its bar (in the bar's space), and which bar (0 the main
        # one, 1 the leg bar, which rocks).
        self.string_point = np.array([i["right ear"], i["left ear"], i["back"], i["right hand"], i["left hand"],
                                      i["right knee"], i["left knee"]], np.int64)
        self.string_anchor = np.array([(0.1, 0.0, 0.0), (-0.1, 0.0, 0.0), (0.0, 0.0, -0.2), (0.35, 0.0, 0.08),
                                       (-0.35, 0.0, 0.08), (LEG_BAR, 0.0, 0.18), (-LEG_BAR, 0.0, 0.18)])
        self.string_bar = np.array([0, 0, 0, 0, 0, 1, 1], np.int64)
        self.hand_strings = np.array([3, 4])  # (right, left)
        self.feet = np.array([i[f"{n} {p}"] for n in ("right", "left") for p in ("ankle", "foot")], np.int64)


@njit(cache=True, error_model="numpy")
def simulate(p, prev, inv_mass, radius, rod_a, rod_b, rod_len, rod_k, rod_kind, hinges, hinge_reach, string_point,
             anchors, string_len, gravity, floor, dt, iterations, damping, friction):
    """One step of the rag dolls p (P, N, 3), prev (P, N, 3) their positions a step before, in place: Verlet
    integration, then `iterations` passes over rods, hinges, strings (anchors (P, S, 3), string_len (P, S)) and the
    floor (at height `floor`)."""
    keep = max(0.0, 1.0 - damping * dt)  # of each point's speed apart from the whole puppet's
    keep_all = max(0.0, 1.0 - AIR * dt)  # of the whole puppet's
    for q in range(p.shape[0]):
        mean = np.zeros(3)  # the puppet's own velocity (a step's movement), weighted by mass
        mass = 0.0
        for n in range(p.shape[1]):
            m = 1.0 / inv_mass[n]
            mass += m
            for d in range(3):
                mean[d] += m * (p[q, n, d] - prev[q, n, d])
        mean /= mass
        pressed = np.zeros(p.shape[1])  # how far the floor pushed each point up this step
        for n in range(p.shape[1]):
            for d in range(3):
                v = mean[d] * keep_all + (p[q, n, d] - prev[q, n, d] - mean[d]) * keep
                prev[q, n, d] = p[q, n, d]
                p[q, n, d] += v
            p[q, n, 1] -= gravity * dt * dt
        for _ in range(iterations):
            for c in range(rod_a.shape[0]):
                a, b = rod_a[c], rod_b[c]
                dx = p[q, b, 0] - p[q, a, 0]
                dy = p[q, b, 1] - p[q, a, 1]
                dz = p[q, b, 2] - p[q, a, 2]
                dist = np.sqrt(dx * dx + dy * dy + dz * dz)
                if not dist > 1e-9:
                    continue
                off = (dist - rod_len[c]) / dist
                if (rod_kind[c] == 1 and off > 0.0) or (rod_kind[c] == 2 and off < 0.0):
                    continue
                wa, wb = inv_mass[a], inv_mass[b]
                s = rod_k[c] * off / (wa + wb)
                p[q, a, 0] += wa * s * dx
                p[q, a, 1] += wa * s * dy
                p[q, a, 2] += wa * s * dz
                p[q, b, 0] -= wb * s * dx
                p[q, b, 1] -= wb * s * dy
                p[q, b, 2] -= wb * s * dz
            for h in range(hinges.shape[0]):  # b kept on its side of the line a-c (a knee out in front)
                a, b, c, f0, f1 = hinges[h, 0], hinges[h, 1], hinges[h, 2], hinges[h, 3], hinges[h, 4]
                sign = hinges[h, 5]
                lx = p[q, c, 0] - p[q, a, 0]
                ly = p[q, c, 1] - p[q, a, 1]
                lz = p[q, c, 2] - p[q, a, 2]
                ll = lx * lx + ly * ly + lz * lz
                reach = hinge_reach[h]
                # (folded up, which side is which means little: the hinge fades out as a and c come together)
                fade = min(1.0, max(0.0, (np.sqrt(ll) - reach) / reach))
                if fade <= 0.0:
                    continue
                t = ((p[q, b, 0] - p[q, a, 0]) * lx + (p[q, b, 1] - p[q, a, 1]) * ly
                     + (p[q, b, 2] - p[q, a, 2]) * lz) / ll
                t = min(1.0, max(0.0, t))
                ox = p[q, b, 0] - p[q, a, 0] - t * lx  # b's offset from the line
                oy = p[q, b, 1] - p[q, a, 1] - t * ly
                oz = p[q, b, 2] - p[q, a, 2] - t * lz
                # the way it may stick out: across the line from the hinge's axis (f1 to f0)
                ax = p[q, f0, 0] - p[q, f1, 0]
                ay = p[q, f0, 1] - p[q, f1, 1]
                az = p[q, f0, 2] - p[q, f1, 2]
                al = np.sqrt((ax * ax + ay * ay + az * az) * ll)
                if not al > 1e-12:
                    continue
                fx = sign * (ly * az - lz * ay) / al
                fy = sign * (lz * ax - lx * az) / al
                fz = sign * (lx * ay - ly * ax) / al
                ff = np.sqrt(fx * fx + fy * fy + fz * fz)  # (the sine between line and axis)
                # (with the line along the axis, which way is which means little: fade out there too)
                fade = fade * min(1.0, ff / 0.5)
                if not fade > 0.0:
                    continue
                fx, fy, fz = fx / ff, fy / ff, fz / ff
                o = ox * fx + oy * fy + oz * fz
                if o >= 0.0:
                    continue
                # the part of b's offset on the wrong side taken away, shared as a rod's push is: b one way, a and c
                # (by how near b is to each) the other
                cx, cy, cz = o * fx, o * fy, o * fz
                wa, wb, wc = inv_mass[a] * (1.0 - t) ** 2, inv_mass[b], inv_mass[c] * t * t
                k = HINGE_STIFFNESS * fade / (wa + wb + wc)
                p[q, b, 0] -= k * inv_mass[b] * cx
                p[q, b, 1] -= k * inv_mass[b] * cy
                p[q, b, 2] -= k * inv_mass[b] * cz
                ka, kc = k * inv_mass[a] * (1.0 - t), k * inv_mass[c] * t
                p[q, a, 0] += ka * cx
                p[q, a, 1] += ka * cy
                p[q, a, 2] += ka * cz
                p[q, c, 0] += kc * cx
                p[q, c, 1] += kc * cy
                p[q, c, 2] += kc * cz
            for s in range(string_point.shape[0]):
                n = string_point[s]
                dx = p[q, n, 0] - anchors[q, s, 0]
                dy = p[q, n, 1] - anchors[q, s, 1]
                dz = p[q, n, 2] - anchors[q, s, 2]
                dist = np.sqrt(dx * dx + dy * dy + dz * dz)
                if dist > string_len[q, s] and dist > 1e-9:  # taut: back onto the string's reach
                    f = string_len[q, s] / dist
                    p[q, n, 0] = anchors[q, s, 0] + dx * f
                    p[q, n, 1] = anchors[q, s, 1] + dy * f
                    p[q, n, 2] = anchors[q, s, 2] + dz * f
            for n in range(p.shape[1]):
                low = floor + radius[n]
                if p[q, n, 1] < low:
                    pressed[n] += low - p[q, n, 1]
                    p[q, n, 1] = low
        for n in range(p.shape[1]):  # friction: what the floor holds up slides less, the harder it presses
            if pressed[n] > 0.0:
                sx = p[q, n, 0] - prev[q, n, 0]
                sz = p[q, n, 2] - prev[q, n, 2]
                slide = np.sqrt(sx * sx + sz * sz)
                grip = friction * pressed[n]
                f = 1.0 if slide <= grip else grip / slide  # (of the slide taken away: all, or as much as grip)
                prev[q, n, 0] += sx * f
                prev[q, n, 2] += sz * f


# ----- from points to joints -----------------------------------------------------------------------------------

def _unit(v):
    return v / max(np.linalg.norm(v), 1e-9)


def _frame(y, z):
    """The rotation whose y axis is along y and whose z axis is as near z as it can be."""
    y = _unit(y)
    z = _unit(z - y * np.dot(z, y))
    return np.column_stack([np.cross(y, z), y, z])


def _arc(u, v):
    """The shortest rotation (3, 3) taking unit u onto unit v."""
    c = np.cross(u, v)
    d = float(np.dot(u, v))
    if d < -0.999999:
        axis = _unit(np.cross(u, (1.0, 0.0, 0.0)) if abs(u[0]) < 0.9 else np.cross(u, (0.0, 1.0, 0.0)))
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    k = np.array([[0.0, -c[2], c[1]], [c[2], 0.0, -c[0]], [-c[1], c[0], 0.0]])
    return np.eye(3) + k + k @ k / (1.0 + d)


@njit(cache=True, error_model="numpy")
def _unit3(a, b, c, out):
    n = max(np.sqrt(a * a + b * b + c * c), 1e-9)
    out[0], out[1], out[2] = a / n, b / n, c / n


@njit(cache=True, error_model="numpy")
def _frame_into(m, y, z):
    """m (3, 3) = the rotation whose y axis is along y and whose z axis is as near z as it can be."""
    u = np.empty(3)
    _unit3(y[0], y[1], y[2], u)
    d = z[0] * u[0] + z[1] * u[1] + z[2] * u[2]
    w = np.empty(3)
    _unit3(z[0] - u[0] * d, z[1] - u[1] * d, z[2] - u[2] * d, w)
    m[0, 0], m[1, 0], m[2, 0] = u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]
    for i in range(3):
        m[i, 1], m[i, 2] = u[i], w[i]


@njit(cache=True, error_model="numpy")
def _quat_into(q, m):
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = 2.0 * np.sqrt(trace + 1.0)
        q[0], q[1], q[2], q[3] = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(max(1.0 + m[0, 0] - m[1, 1] - m[2, 2], 1e-12))
        q[0], q[1], q[2], q[3] = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(max(1.0 + m[1, 1] - m[0, 0] - m[2, 2], 1e-12))
        q[0], q[1], q[2], q[3] = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(max(1.0 + m[2, 2] - m[0, 0] - m[1, 1], 1e-12))
        q[0], q[1], q[2], q[3] = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    n = np.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    if n > 1e-12:
        for i in range(4):
            q[i] /= n


@njit(cache=True, error_model="numpy")
def pose_all(p, torso, par, limb, limb_a, limb_b, rest, quats, hips_at):
    """Turn each puppet's joints to point along its points (see Puppet.pose): p (N, P, 3) the points; torso the
    joints' indices (hips, spine, chest, neck, head) and then the points fixing them (hips, spine, chest, neck, head,
    head top, left ear, right ear, pelvis front, chest front); par each joint's parent's index; limb, limb_a, limb_b
    each limb joint's index and its start and end points; rest (J, 3, 3) the joints' rest rotations. Writes quats
    (N, J, 4) and hips_at (N, 3)."""
    n_joints = rest.shape[0]
    world = np.empty((n_joints, 3, 3))
    pt = torso[5:]
    y, z, local, arc, unbent = np.empty(3), np.empty(3), np.empty((3, 3)), np.empty((3, 3)), np.empty((3, 3))
    for n in range(p.shape[0]):
        P = p[n]
        hips, spine, chest, neck, head = torso[0], torso[1], torso[2], torso[3], torso[4]
        for i in range(3):
            y[i] = P[pt[1], i] - P[pt[0], i]
            z[i] = P[pt[8], i] - P[pt[0], i]
        _frame_into(world[hips], y, z)
        for i in range(3):
            y[i] = P[pt[3], i] - P[pt[2], i]
            z[i] = P[pt[9], i] - P[pt[2], i]
        _frame_into(world[chest], y, z)
        for i in range(3):
            y[i] = P[pt[2], i] - P[pt[1], i]
            z[i] = world[hips][i, 2] + world[chest][i, 2]
        _frame_into(world[spine], y, z)
        for i in range(3):
            y[i] = P[pt[4], i] - P[pt[3], i]
            z[i] = world[chest][i, 2]
        _frame_into(world[neck], y, z)
        for i in range(3):
            y[i] = P[pt[5], i] - P[pt[4], i]
        _unit3(y[0], y[1], y[2], y)
        # (the head faces across the line between the ears, from the left one to the right)
        ex, ey, ez = P[pt[7], 0] - P[pt[6], 0], P[pt[7], 1] - P[pt[6], 1], P[pt[7], 2] - P[pt[6], 2]
        _unit3(ex, ey, ez, z)
        cx, cy, cz = z[1] * y[2] - z[2] * y[1], z[2] * y[0] - z[0] * y[2], z[0] * y[1] - z[1] * y[0]
        z[0], z[1], z[2] = cx, cy, cz
        _frame_into(world[head], y, z)
        # (the order the torso joints are written is the order of `torso`: hips, spine, chest, neck, head)
        for k in range(5):
            j = torso[k]
            if k == 0:
                for a in range(3):
                    for b in range(3):
                        local[a, b] = world[j][a, b]
            else:
                pj = par[j]
                for a in range(3):
                    for b in range(3):
                        s = 0.0
                        for c in range(3):
                            s += world[pj][c, a] * world[j][c, b]
                        local[a, b] = s
            _quat_into(quats[n, j], local)
        for k in range(limb.shape[0]):
            j = limb[k]
            pj = par[j]
            for a in range(3):
                for b in range(3):
                    s = 0.0
                    for c in range(3):
                        s += world[pj][a, c] * rest[j][c, b]
                    unbent[a, b] = s
            ux, uy, uz = unbent[0, 1], unbent[1, 1], unbent[2, 1]
            _unit3(P[limb_b[k], 0] - P[limb_a[k], 0], P[limb_b[k], 1] - P[limb_a[k], 1],
                   P[limb_b[k], 2] - P[limb_a[k], 2], y)
            # the shortest rotation arc taking u onto y
            cx, cy, cz = uy * y[2] - uz * y[1], uz * y[0] - ux * y[2], ux * y[1] - uy * y[0]
            d = ux * y[0] + uy * y[1] + uz * y[2]
            if d < -0.999999:
                if abs(ux) < 0.9:
                    _unit3(uy * 0.0 - uz * 0.0, uz * 1.0 - ux * 0.0, ux * 0.0 - uy * 1.0, z)
                else:
                    _unit3(uy * 0.0 - uz * 1.0, uz * 0.0 - ux * 0.0, ux * 1.0 - uy * 0.0, z)
                for a in range(3):
                    for b in range(3):
                        arc[a, b] = 2.0 * z[a] * z[b] - (1.0 if a == b else 0.0)
            else:
                f = 1.0 / (1.0 + d)
                kk = ((0.0, -cz, cy), (cz, 0.0, -cx), (-cy, cx, 0.0))
                for a in range(3):
                    for b in range(3):
                        s = 0.0
                        for c in range(3):
                            s += kk[a][c] * kk[c][b]
                        arc[a, b] = (1.0 if a == b else 0.0) + kk[a][b] + s * f
            for a in range(3):
                for b in range(3):
                    s = 0.0
                    for c in range(3):
                        s += arc[a, c] * unbent[c, b]
                    world[j][a, b] = s
            for a in range(3):
                for b in range(3):
                    s = 0.0
                    for c in range(3):
                        s += world[pj][c, a] * world[j][c, b]
                    local[a, b] = s
            _quat_into(quats[n, j], local)
        for i in range(3):
            hips_at[n, i] = P[pt[0], i]


def _turn_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _turn_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def normal_turns(mesh, targets, step=0.01):
    """How far each of `targets` (K, V, 3: morph targets for mesh) turns each vertex's normal at full weight, for
    Mesh.morph_normals: worked out from the mesh moved a little way (step) by each, so as a weight grows the normals
    turn as the surface does, at first."""
    flat = mesh.vertex_normals()
    return np.stack([(Mesh(mesh.vertices + step * t, mesh.faces).vertex_normals() - flat) / step for t in targets])


def squash_field(points):
    """How far the squash target moves points (N, 3) in the head joint's space (y up the head, z forward) at full
    weight: down, the more the higher (the top by SQUASH_DROP), and out from the head's axis, most at its middle.
    The parts on the head (eyes, nose, mouth, hat) move by it too, so they stay on it; a weight below 0 stretches
    the head instead, taller and narrower."""
    s = np.clip(points[:, 1] / HEAD, 0.0, 1.0)
    moved = np.zeros_like(points)
    moved[:, 1] = -SQUASH_DROP * s
    moved[:, [0, 2]] = (SQUASH_WIDEN * np.sin(np.pi * s))[:, None] * points[:, [0, 2]]
    return moved


def singing(mesh):
    """The figure's mesh (figure_rig()'s, bound as skeleton() stands) with two morph targets on the head: singing,
    the jaw (the head's lowest rings) dropped JAW and widened a little to the sides, the rings under it following
    part of the way; and squashing (squash_field)."""
    j = skeleton()
    names = list(j)
    head = names.index("head")
    world = Skin(list(j.values())).world_matrices()[head]
    up = (mesh.vertices - world[:3, 3]) @ world[:3, 1]  # (how far along the head each vertex is)
    on_head = (mesh.joints[:, 0] == head) | (mesh.joints[:, 1] == head)
    drop = np.where(on_head, np.interp(up, (-0.01, 0.0, 0.06, 0.12, 0.17), (0.0, 0.45, 1.0, 0.3, 0.0)), 0.0)
    target = np.zeros((2, len(mesh.vertices), 3))
    target[0, :, 1] = -JAW * drop
    across = (mesh.vertices - world[:3, 3]) @ world[:3, 0]
    target[0] += 0.15 * (drop * across)[:, None] * world[:3, 0]  # (wider, not deeper: the mouth stays in front)
    local = (mesh.vertices - world[:3, 3]) @ world[:3, :3]  # (in the head's space)
    target[1, on_head] = squash_field(local[on_head]) @ world[:3, :3].T
    mesh.morph_targets, mesh.morph_normals = target, normal_turns(mesh, target)
    return mesh


def mouth_mesh():
    """A dark mouth for the front of the head: a thin slot, with a morph target opening it (its lower edge down by
    JAW, as the jaw goes, and narrower, an O)."""
    mesh = block_mesh((0.0, 0.0, 0.0), (0.06, 0.014, 0.016))
    low = mesh.vertices[:, 1] < 0.0
    target = np.zeros((1, len(mesh.vertices), 3))
    target[0, low, 1] = -0.9 * JAW
    target[0, low, 0] = -0.3 * mesh.vertices[low, 0]
    mesh.morph_targets = target
    return mesh


def wood_texture(res=(64, 96)):
    """Light wood: pale boards with darker grain lines running along v (up the limbs), a few knots."""
    h, w = res
    v, u = np.mgrid[0:h, 0:w] / np.array([h, w])[:, None, None]
    rng = np.random.default_rng(7)
    grain = 0.5 + 0.5 * np.sin(2 * np.pi * (12 * u + 0.35 * np.sin(2 * np.pi * (1.5 * v + 2 * u))))
    grain = 0.75 * grain + 0.25 * rng.random((h, w))
    for cu, cv in ((0.25, 0.3), (0.7, 0.75)):  # knots
        grain -= 0.5 * np.exp(-(((u - cu) * 6) ** 2 + ((v - cv) * 2.5) ** 2) * 6)
    t = np.clip(grain, 0.0, 1.0)[..., None]
    return np.array([0.80, 0.60, 0.38]) * (1 - t) + np.array([0.96, 0.82, 0.58]) * t


def wooden(mesh):
    """`mesh` with a wood texture, projected from the front, so that grain runs up every limb of a figure
    standing as figure_rig() binds it."""
    corners = mesh.vertices[mesh.faces]  # (F, 3, 3)
    uvs = np.stack([3.0 * (corners[..., 0] + 0.5 * corners[..., 2]), 1.5 * corners[..., 1]], axis=-1)
    return Mesh(mesh.vertices, mesh.faces, uvs, np.zeros(len(mesh.faces), int), [wood_texture()], None,
                None, mesh.normals, mesh.joints, mesh.weights)


class Puppet:
    """A skinned figure (and the parts on its joints) posed from a Doll's points."""
    LIMBS = [(f"{s} {a}", f"{s} {b}") for s in ("right", "left")
             for a, b in (("shoulder", "elbow"), ("elbow", "wrist"), ("wrist", "hand"),
                          ("hip", "knee"), ("knee", "ankle"), ("ankle", "foot"))]

    def __init__(self, rig, doll, body=WOOD, hat=(150, 40, 45)):
        self.object, self.joints = figure(rig, color=body, specular=0.3, emissive=0.3)
        self.doll = doll
        self.parent_name = {n: next((m for m, other in self.joints.items() if other is node.parent), None)
                            for n, node in self.joints.items()}
        self.rest_local = {n: quat_to_matrix(node.rotation) for n, node in self.joints.items()}
        head, box = self.joints["head"], make_box()
        self.object.morph_weights = np.zeros(2)  # (singing, squashing; always an array, so the skinning plan stays put)
        self.clip = None  # (a squash or a gasp playing)
        self.mouth = Object3D(MOUTH, position=(0.0, 0.078, 0.097), color=(45, 15, 15), morph_weights=np.zeros(1),
                              parent=head)
        self.parts = [
            Object3D(box, position=(0.0, 0.13, 0.1), scale=(0.03, 0.04, 0.06), color=(190, 60, 50), parent=head),
            Object3D(box, position=(0.04, 0.17, 0.095), scale=(0.025, 0.025, 0.02), color=(25, 20, 20), parent=head),
            Object3D(box, position=(-0.04, 0.17, 0.095), scale=(0.025, 0.025, 0.02), color=(25, 20, 20),
                     parent=head),
            self.mouth,
            Object3D(box, position=(0.0, 0.265, 0.0), scale=(0.2, 0.02, 0.2), color=hat, parent=head),
            Object3D(box, position=(0.0, 0.33, 0.0), scale=(0.12, 0.13, 0.12), color=hat, parent=head),
        ] + [Object3D(HAND, position=(0.0, 0.07, 0.0), color=HANDS, specular=0.3, emissive=0.3, parent=self.joints[f"{s} wrist"])
             for s in ("right", "left")]
        self.face_parts = [p for p in self.parts if p.parent is head]
        self.face_home = np.array([p.position for p in self.face_parts], dtype=float)

    def objects(self):
        return [self.object] + self.parts

    def play(self, keys, fade=0.05):
        """Play keyframes (time, (jaw, squash)) on the head, as a Clip, easing in from where it is."""
        self.clip = Clip([Animation(self.object, morph_weights=Track(keys, easing="ease_in_out"))])
        self.clip.start(fade)

    def face(self, dt, voice):
        """The head's morph weights now: what the clip playing gives (then none), its jaw open at least `voice`
        (0..1, singing); the mouth opened as far as the jaw, and the parts on the head moved with its squash."""
        w = np.zeros(2)
        if self.clip is not None:
            playing = self.clip.update(dt)
            w = np.array(self.object.morph_weights, dtype=float)
            if not playing:
                self.clip = None
        w[0] = max(w[0], voice)
        self.object.morph_weights = w
        self.mouth.morph_weights = w[:1].copy()
        moved = squash_field(self.face_home) * w[1]
        for part, home, by in zip(self.face_parts, self.face_home, moved):
            part.position = home + by

    def triangles(self):
        return sum(len(o.mesh.faces) for o in self.objects())

    def spec(self):
        """The arguments of pose_all that describe this figure (the same for every one made from a rig)."""
        names = list(self.joints)
        at, doll = {n: k for k, n in enumerate(names)}, self.doll.at
        par = np.array([at[self.parent_name[n]] if self.parent_name[n] else -1 for n in names], np.int64)
        torso = [at[n] for n in ("hips", "spine", "chest", "neck", "head")] + [
            doll[n] for n in ("hips", "spine", "chest", "neck", "head", "head top", "left ear", "right ear",
                              "pelvis front", "chest front")]
        return (np.array(torso, np.int64), par, np.array([at[a] for a, b in self.LIMBS], np.int64),
                np.array([doll[a] for a, b in self.LIMBS], np.int64),
                np.array([doll[b] for a, b in self.LIMBS], np.int64), np.stack([self.rest_local[n] for n in names]))


def pose_puppets(puppets, points):
    """Turn the joints of each of `puppets` to point along its points (N, P, 3 for all of them): the torso's and
    head's from the points fixing them, each limb's by the shortest turn from where it points with the joint
    unbent."""
    torso, par, limb, limb_a, limb_b, rest = puppets[0].spec()
    quats = np.empty((len(puppets), len(par), 4))
    hips = np.empty((len(puppets), 3))
    pose_all(np.ascontiguousarray(points, dtype=np.float64), torso, par, limb, limb_a, limb_b, rest, quats, hips)
    for n, puppet in enumerate(puppets):
        puppet.joints["hips"].position = hips[n]
        for k, node in enumerate(puppet.joints.values()):
            node.rotation = quats[n, k]


HAND = blob_mesh((0.035, 0.06, 0.02))
MOUTH = mouth_mesh()


# ----- a marionette: the puppet, its bar and strings -----------------------------------------------------------

def slack(a, b, length, pieces=6):
    """Points (pieces + 1, 3) along a string of `length` from a to b: straight if taut, sagging if slack."""
    span = np.linalg.norm(b - a)
    sag = np.sqrt(max(0.375 * span * (length - span), 0.0))  # (a parabola's dip, for its length over the span)
    s = np.linspace(0.0, 1.0, pieces + 1)[:, None]
    return a + s * (b - a) - 4.0 * min(sag, 0.5 * length) * s * (1.0 - s) * np.array([0.0, 1.0, 0.0])


class Marionette:
    """One puppet, its control bar, and the strings between."""

    def __init__(self, rig, doll, at, yaw=0.0, **colours):
        self.doll, self.puppet = doll, Puppet(rig, doll, **colours)
        self.p = (doll.rest @ _turn_y(yaw).T + (at[0], STAGE + 0.03, at[1]))[None].copy()
        self.prev = self.p.copy()
        self.bar = np.array([at[0], BAR_HOME, at[1]])
        self.yaw, self.rock, self.phase = yaw, 0.0, 0.0
        self.pull = np.zeros(2)  # right, left
        self.voice, self.vibrato = 0.0, 0.0  # how far into a note (0..1), and the vibrato's phase
        self.air = 0.0  # how long both feet have been off the stage
        self.left_over = 0.0
        self.base_len = np.linalg.norm(self.p[0, doll.string_point] - self.anchors()[0], axis=1)
        self.base_len[doll.hand_strings] += HAND_SLACK
        self.base_len[doll.string_bar == 1] += 0.02
        self.base_len[doll.string_bar == 0] += SAG
        self.bar_obj = Object3D(BAR, color=(120, 80, 40))
        self.leg_obj = Object3D(LEG, color=(140, 95, 50))
        self.strings = Lines(color=(150, 150, 140), emissive=0.3)

    def anchors(self):
        """(1, S, 3) the strings' ends on the bars, in the world."""
        local = self.doll.string_anchor.copy()
        leg = self.doll.string_bar == 1
        local[leg] = local[leg] @ _turn_z(self.rock).T
        return (self.bar + (0.0, STAGE, 0.0) + local @ _turn_y(self.yaw).T)[None]

    def control(self, dt, move, turn, rise, rock, pulls, sing, auto):
        """Work the bar for dt seconds: move (x, z) a unit vector or zero (in the world), turn -1..1, rise -1..1,
        rock -1..1 (the leg bar), pulls (right, left) 0 or 1, sing 0 or 1; auto: rock the leg bar while moving, and
        turn to face the way it goes."""
        self.voice = float(np.clip(self.voice + (dt / SING_ATTACK if sing else -dt / SING_RELEASE), 0.0, 1.0))
        self.vibrato = (self.vibrato + 2.0 * np.pi * VIBRATO * dt) % (2.0 * np.pi) if self.voice else 0.0

        step = np.array([move[0], 0.0, move[1]]) * BAR_SPEED * dt
        self.bar = self.bar + step
        self.bar[0] = float(np.clip(self.bar[0], -BAR_X, BAR_X))
        self.bar[2] = float(np.clip(self.bar[2], *BAR_Z))
        self.bar[1] = float(np.clip(self.bar[1] + rise * BAR_SPEED * dt, BAR_LOW, BAR_HIGH))
        self.yaw += turn * BAR_TURN * dt
        target = ROCK * rock
        if auto and (move[0] or move[1]):
            self.phase += np.linalg.norm(step) / STRIDE * 2.0 * np.pi
            target += 0.7 * ROCK * np.sin(self.phase)
            if not turn:
                ahead = np.arctan2(move[0], move[1]) - self.yaw
                ahead = (ahead + np.pi) % (2.0 * np.pi) - np.pi
                self.yaw += float(np.clip(ahead, -BAR_TURN * dt, BAR_TURN * dt))
        self.rock += float(np.clip(target - self.rock, -ROCK_SPEED * dt, ROCK_SPEED * dt))
        rate = PULL_SPEED / HAND_PULL * dt
        self.pull += np.clip(np.asarray(pulls, float) - self.pull, -rate, rate)

    def update(self, dt, gravity):
        lens = self.base_len.copy()
        lens[self.doll.hand_strings] -= HAND_PULL * self.pull
        anchors = self.anchors()
        self.left_over += dt
        d = self.doll
        while self.left_over >= STEP:
            self.left_over -= STEP
            simulate(self.p, self.prev, d.inv_mass, d.radius, d.rod_a, d.rod_b, d.rod_len, d.rod_k, d.rod_kind,
                     d.hinges, d.hinge_reach, d.string_point, anchors, lens[None], float(gravity), STAGE, STEP,
                     ITERATIONS, DAMPING, FRICTION)
        if not np.isfinite(self.p).all():  # (never seen; but a puppet gone wrong stands up again)
            self.p[...] = self.doll.rest + (self.bar[0], STAGE + 0.03, self.bar[2])
            self.prev[...] = self.p
        # landing: touching the stage after a while off it, the head squashes (the harder, the more)
        low = d.feet[np.argmin(self.p[0, d.feet, 1] - d.radius[d.feet])]
        if self.p[0, low, 1] - d.radius[low] > STAGE + CLEAR:
            self.air += dt
        else:
            if self.air > AIRBORNE:
                speed = max(0.0, (self.prev[0, low, 1] - self.p[0, low, 1]) / STEP)
                a = float(np.clip(0.4 + 0.4 * speed, 0.4, 1.0))
                self.puppet.play([(0.0, (0.0, 0.0)), (0.07, (0.0, a)), (0.2, (0.0, -0.35 * a)),
                                  (0.34, (0.0, 0.12 * a)), (0.48, (0.0, 0.0))])
            self.air = 0.0
        o = self.voice * self.voice * (3.0 - 2.0 * self.voice)
        self.puppet.face(dt, o * (0.85 + 0.15 * np.sin(self.vibrato)))
        place = self.bar + (0.0, STAGE, 0.0)
        q = quat_axis_angle((0, 1, 0), self.yaw)
        self.bar_obj.position, self.bar_obj.rotation = place, q
        self.leg_obj.position = place + _turn_y(self.yaw) @ (0.0, 0.0, 0.18)
        self.leg_obj.rotation = quat_mul(q, quat_axis_angle((0, 0, 1), self.rock))
        points = self.p[0, d.string_point]
        wires = [slack(a, b, n) for a, b, n in zip(anchors[0], points, lens)]
        self.strings.set(np.concatenate([w[:-1] for w in wires]), np.concatenate([w[1:] for w in wires]))

    def objects(self):
        return self.puppet.objects() + [self.bar_obj, self.leg_obj, self.strings]


BAR = merge_meshes([block_mesh((0.0, 0.0, 0.0), (0.75, 0.04, 0.04)), block_mesh((0.0, 0.0, -0.05), (0.04, 0.04, 0.5))])
LEG = block_mesh((0.0, 0.0, 0.0), (2 * LEG_BAR + 0.05, 0.035, 0.035))


# ----- the curtains --------------------------------------------------------------------------------------------

class Curtains:
    """The two curtains, one pleated sheet (shared) skinned to a row of joints along the rail and a row along the
    hem, for each. Opening, the joints move to the side and squeeze up, gathering the cloth into deeper folds; the
    hem's joints follow the rail's on springs, so the cloth swings as it goes."""
    JOINTS = 6              # along each row
    FOLDS = 10
    WIDTH = OPENING_X + 0.2  # of each curtain, closed (meeting in the middle, a little over)
    HEIGHT = OPENING_Y + 0.05
    GATHERED = 0.55         # its width, open
    SPEED = 0.45            # of opening, a second (0..1)
    RIPPLE = 2.4            # the billow's wavelength across a curtain
    RIPPLE_SPEED = 1.8      # how fast it runs across, radians a second
    DRAUGHT = 0.04          # how far the hem billows at rest
    BILLOW = 0.15           # and at most, moving

    def __init__(self, color=(150, 25, 35)):
        self.mesh = self._sheet()
        self.rails, self.top, self.hem, self.objects = [], [], [], []
        k = np.arange(self.JOINTS)
        self.spread = k * self.WIDTH / (self.JOINTS - 1)
        self.bunched = k * self.GATHERED / (self.JOINTS - 1)
        for side in (-1.0, 1.0):  # each from its side of the opening, its +x towards the middle
            rail = Node(np.array([side * (OPENING_X + 0.15), STAGE + OPENING_Y + 0.05, CURTAIN_Z]),
                        quat_axis_angle((0, 1, 0), 0.0 if side < 0 else np.pi))
            top = [Node(np.array([x, 0.0, 0.0]), parent=rail) for x in self.spread]
            hem = [Node(np.array([x, -self.HEIGHT, 0.0]), parent=rail) for x in self.spread]
            skin = Skin(top + hem, np.linalg.inv(Skin(top + hem).world_matrices(top=rail)))
            self.rails.append(rail)
            self.top.append(top)
            self.hem.append(hem)
            self.objects.append(Object3D(self.mesh, skin=skin, color=color, specular=0.15, double_sided=True))
        self.open = 0.0  # 0 closed, 1 open
        self.hem_x = np.tile(self.spread, (2, 1))
        self.hem_v = np.zeros((2, self.JOINTS))
        self.phase = np.array([0.0, 2.0])  # (the two curtains out of step)
        self.billow = self.DRAUGHT
        self.settle(0.0)

    def settle(self, open):
        """Put the curtains `open` (0..1) at once, hanging still."""
        self.open = float(open)
        o = self.open * self.open * (3.0 - 2.0 * self.open)
        self.hem_x[:] = self.spread + (self.bunched - self.spread) * o
        self.hem_v[:] = 0.0
        self.update(0.0, self.open > 0.5)

    def _billow(self, mesh):
        """Two morph targets on the sheet: a wave across it, the cloth swinging out and in (z) and a little to and
        fro (x), none at the rail and most at the hem, one a quarter of a wave on from the other. Weighted
        (a cos t, a sin t) they make a wave of height a running across the cloth (sin(kx + t) = sin kx cos t +
        cos kx sin t). Their normals' turns are worked out from the sheet moved a little by each."""
        v = mesh.vertices
        down = (-v[:, 1] / self.HEIGHT) ** 1.5
        at = 2.0 * np.pi / self.RIPPLE * v[:, 0]
        targets = np.zeros((2, len(v), 3))
        targets[0, :, 0], targets[0, :, 2] = 0.4 * down * np.cos(at), down * np.sin(at)
        targets[1, :, 0], targets[1, :, 2] = -0.4 * down * np.sin(at), down * np.cos(at)
        return targets, normal_turns(mesh, targets)

    def _sheet(self):
        columns, rows = 4 * self.FOLDS + 1, 9
        x = np.linspace(0.0, self.WIDTH, columns)
        z = 0.06 * np.cos(2.0 * np.pi * self.FOLDS * x / self.WIDTH)
        y = -np.linspace(0.0, self.HEIGHT, rows)
        vertices = np.array([(x[c], y[r], z[c]) for r in range(rows) for c in range(columns)])
        along = np.minimum(x / self.WIDTH * (self.JOINTS - 1), self.JOINTS - 1 - 1e-9)
        k, t = np.floor(along).astype(int), along - np.floor(along)
        joints, weights = [], []
        for r in range(rows):
            s = r / (rows - 1)
            for c in range(columns):
                joints.append((k[c], k[c] + 1, self.JOINTS + k[c], self.JOINTS + k[c] + 1))
                weights.append(((1 - t[c]) * (1 - s), t[c] * (1 - s), (1 - t[c]) * s, t[c] * s))
        faces = []
        for r in range(rows - 1):
            for c in range(columns - 1):
                a = r * columns + c
                faces += [(a, a + columns, a + 1), (a + 1, a + columns, a + columns + 1)]
        mesh = Mesh(vertices, np.array(faces), joints=np.array(joints), weights=np.array(weights))
        mesh.morph_targets, mesh.morph_normals = self._billow(mesh)
        return mesh

    def update(self, dt, want_open):
        self.open = float(np.clip(self.open + (self.SPEED if want_open else -self.SPEED) * dt, 0.0, 1.0))
        o = self.open * self.open * (3.0 - 2.0 * self.open)  # (easing in and out)
        x = self.spread + (self.bunched - self.spread) * o
        squeeze = 1.0 + (self.GATHERED / self.WIDTH - 1.0) * o
        scale = np.array([squeeze, 1.0, 1.0 + 2.5 * o])  # (narrower, and the folds deeper)
        # the hem on springs, following the rail
        self.hem_v += (40.0 * (x - self.hem_x) - 6.0 * self.hem_v) * dt
        self.hem_x += self.hem_v * dt
        for c in range(2):
            for j in range(self.JOINTS):
                self.top[c][j].position = np.array([x[j], 0.0, 0.0])
                swing = self.hem_x[c, j] - x[j]
                self.hem[c][j].position = np.array([self.hem_x[c, j], -self.HEIGHT, 0.4 * abs(swing)])
                self.top[c][j].scale = self.hem[c][j].scale = scale
        # the billow: a wave running across the cloth, higher while it moves, dying down after
        moving = float(np.abs(self.hem_v).max()) + 2.0 * self.SPEED * (0.0 < self.open < 1.0)
        want = min(self.BILLOW, self.DRAUGHT + 0.15 * moving)
        self.billow += (want - self.billow) * min(1.0, (4.0 if want > self.billow else 0.8) * dt)
        self.phase += self.RIPPLE_SPEED * (1.0 + 2.0 * (self.billow - self.DRAUGHT) / self.BILLOW) * dt
        height = self.billow / scale[2]  # (the folds' depth scales it too)
        for c in range(2):
            self.objects[c].morph_weights = height * np.array([np.cos(self.phase[c]), np.sin(self.phase[c])])


# ----- the theatre ---------------------------------------------------------------------------------------------

def backdrop_texture(res=(96, 160)):
    """A painted backdrop: an evening sky over hills, with a moon and a few stars."""
    h, w = res
    v, u = np.mgrid[0:h, 0:w] / np.array([h - 1, w - 1])[:, None, None]
    up = 1.0 - v  # (row 0 is the top)
    sky = (np.array([0.95, 0.6, 0.35])[None, None] * (1 - up[..., None]) ** 2
           + np.array([0.15, 0.2, 0.45])[None, None] * (1 - (1 - up[..., None]) ** 2))
    rng = np.random.default_rng(4)
    for y, x in zip(rng.integers(0, h // 2, 30), rng.integers(0, w, 30)):
        sky[y, x] = (0.95, 0.95, 0.85)
    moon = (u - 0.72) ** 2 * (w / h) ** 2 + (up - 0.78) ** 2 < 0.08 ** 2
    sky[moon] = (0.98, 0.95, 0.8)
    far = 0.38 + 0.06 * np.sin(7.0 * u + 1.0) + 0.03 * np.sin(17.0 * u)
    near = 0.22 + 0.07 * np.sin(4.0 * u + 2.5) + 0.02 * np.sin(23.0 * u)
    sky[up < far] = (0.25, 0.35, 0.4)
    sky[up < near] = (0.18, 0.3, 0.2)
    for x in (0.12, 0.2, 0.55, 0.9):  # trees on the near hill
        top = 0.22 + 0.07 * np.sin(4.0 * x + 2.5) + 0.02 * np.sin(23.0 * x)
        tree = (np.abs(u - x) * w / h < 0.06 * (top + 0.16 - up) / 0.16) & (up > top - 0.02) & (up < top + 0.16)
        sky[tree] = (0.1, 0.2, 0.12)
    return sky


def picture(width, height, texture):
    """A width x height rectangle standing on y = 0, facing +z, showing all of `texture`."""
    corners = np.array([(-width / 2, 0, 0), (width / 2, 0, 0), (width / 2, height, 0), (-width / 2, height, 0)], float)
    uvs = np.array([[(0, 0), (1, 0), (1, 1)], [(0, 0), (1, 1), (0, 1)]], float)
    return Mesh(corners, np.array([(0, 1, 2), (0, 2, 3)]), uvs, np.zeros(2, int), [texture])


def boxes(parts):
    """One mesh of boxes (centre, size, colour) coloured per face."""
    return merge_meshes([block_mesh(c, s) for c, s, _ in parts], colors=[col for _, _, col in parts])


class Theatre:
    """The hall: the stage and its proscenium, the backdrop, the footlights, the seats, the walls and a window."""

    def __init__(self):
        wood, dark, gold, plush = (120, 75, 45), (60, 40, 30), (200, 160, 70), (150, 30, 40)
        top = STAGE + OPENING_Y
        stage = [((0.0, STAGE / 2, (STAGE_BACK + 0.0) / 2), (2 * STAGE_X, STAGE, -STAGE_BACK), wood),
                 ((0.0, STAGE / 2, 0.01), (2 * STAGE_X, STAGE - 0.04, 0.02), dark)]  # (its front, darker)
        for k in range(3):  # steps up at the right, the higher nearer the stage
            h = STAGE * (k + 1) / 4
            x1 = STEPS_X[0] + (STEPS_X[1] - STEPS_X[0]) * (3 - k) / 3
            stage.append((((STEPS_X[0] + x1) / 2, h / 2, sum(STEPS_Z) / 2),
                          (x1 - STEPS_X[0], h, STEPS_Z[1] - STEPS_Z[0]), wood))
        frame = [((side * (OPENING_X + 0.3), STAGE + (OPENING_Y + 0.6) / 2, -0.15), (0.6, OPENING_Y + 0.6, 0.3), plush)
                 for side in (-1, 1)]
        frame += [((0.0, top + 0.3, -0.15), (2 * OPENING_X + 1.2, 0.6, 0.3), plush),
                  ((0.0, (top + 0.6 + HALL_Y) / 2, -0.15), (2 * HALL_X, HALL_Y - top - 0.6, 0.3), (95, 80, 70)),
                  ((0.0, top + 0.02, -0.0), (2 * OPENING_X, 0.06, 0.04), gold)]
        frame += [((side * (OPENING_X + 0.02), STAGE + OPENING_Y / 2, 0.0), (0.06, OPENING_Y, 0.04), gold)
                  for side in (-1, 1)]
        lamps = [((x, STAGE + 0.03, -0.12), (0.12, 0.05, 0.06), (255, 215, 140)) for x in np.linspace(-3.0, 3.0, 9)]
        stage.append(((0.0, STAGE + 0.04, -0.04), (2 * OPENING_X, 0.08, 0.04), dark))  # (the footlights' shield)
        seats = []
        for z in ROWS_Z:
            for side in (-1, 1):
                for x in np.linspace(SEATS_X[0] + 0.3, SEATS_X[1] - 0.3, 6):
                    x = side * x
                    seats += [((x, 0.2, z), (0.5, 0.4, 0.45), dark), ((x, 0.46, z), (0.52, 0.12, 0.5), plush),
                              ((x, 0.75, z + 0.24), (0.52, 0.6, 0.08), plush)]
        walls = [((0.0, -0.05, (STAGE_BACK + HALL_BACK) / 2), (2 * HALL_X, 0.1, HALL_BACK - STAGE_BACK), (70, 30, 35))]
        for side in (-1, 1):
            walls.append(((side * (HALL_X + 0.1), HALL_Y / 2, (STAGE_BACK + HALL_BACK) / 2),
                          (0.2, HALL_Y, HALL_BACK - STAGE_BACK), (95, 80, 70)))
        walls += [((0.0, HALL_Y / 2, HALL_BACK + 0.1), (2 * HALL_X, HALL_Y, 0.2), (95, 80, 70)),
                  ((0.0, HALL_Y / 2, STAGE_BACK - 0.1), (2 * HALL_X, HALL_Y, 0.2), (40, 35, 35))]
        self.objects = [
            Object3D(boxes(stage), color=(255, 255, 255), specular=0.2),
            Object3D(boxes(frame), color=(255, 255, 255), specular=0.3),
            Object3D(boxes(seats), color=(255, 255, 255), specular=0.1),
            Object3D(boxes(walls), color=(255, 255, 255), specular=0.05),
            Object3D(boxes(lamps), color=(255, 255, 255), emissive=0.7, cast_shadows=False),
            Object3D(picture(2 * OPENING_X + 0.4, OPENING_Y + 0.3, backdrop_texture()),
                     position=(0.0, STAGE, STAGE_BACK + 0.15), color=(255, 255, 255), specular=0.0),
            Object3D(picture(1.6, 1.6, np.full((2, 2, 3), (0.75, 0.82, 0.95))), position=(-HALL_X + 0.01, 3.2, 4.0),
                     rotation=quat_axis_angle((0, 1, 0), np.pi / 2), color=(255, 255, 255), emissive=0.8,
                     cast_shadows=False),
        ]
        # Where the walker can't go (x0, x1, z0, z1): the seats, the proscenium's sides, the backdrop.
        self.blocks = [(x0, x1, z - 0.3, z + 0.35) for z in ROWS_Z
                       for x0, x1 in ((SEATS_X[0], SEATS_X[1]), (-SEATS_X[1], -SEATS_X[0]))]
        self.blocks += [(side * (OPENING_X + 0.3) - 0.3, side * (OPENING_X + 0.3) + 0.3, -0.3, 0.0)
                        for side in (-1, 1)]
        self.blocks += [(-HALL_X, HALL_X, -HALL_Y, STAGE_BACK + 0.25)]
        self.blocks += [(SLOTS_X[0] - 0.3, SLOTS_X[-1] + 0.3, SLOT_Z - 0.3, SLOT_Z + 0.3)]  # (the plinths)
        self.curtains = Curtains()

    def floor(self, x, z):
        """The height of the floor at (x, z): the stage's, a step's, or the hall's."""
        if abs(x) <= STAGE_X and STAGE_BACK <= z <= 0.0:
            return STAGE
        if STEPS_X[0] <= x <= STEPS_X[1] and STEPS_Z[0] <= z <= STEPS_Z[1]:
            return STAGE * min(3, int((STEPS_X[1] - x) / (STEPS_X[1] - STEPS_X[0]) * 3) + 1) / 4
        return 0.0

    def free(self, x, z, height, puppets=()):
        """Whether the walker, at `height` now, may stand at (x, z): inside the hall, clear of what blocks and of
        the puppets (at (x, z) each), and no more than a step up."""
        if abs(x) > HALL_X - REACH or not STAGE_BACK < z < HALL_BACK - REACH:
            return False
        if any(np.hypot(x - px, z - pz) < REACH + 0.25 for px, pz in puppets):
            return False
        if any(x0 - REACH < x < x1 + REACH and z0 - REACH < z < z1 + REACH for x0, x1, z0, z1 in self.blocks):
            return False
        return self.floor(x, z) - height < 0.3

    def lights(self, footlights=True):
        window = Light(direction=np.array([0.8, -0.45, -0.35]), ambient=0.14, diffuse=0.3, color=(190, 205, 255),
                       specular=0.1)
        if not footlights:
            return [window]
        # (and over the stage, a light without shadows, so the puppets' faces aren't lit only from below)
        return [window] + [PointLight(np.array([x, STAGE + 0.15, -0.1]), color=(255, 215, 160), diffuse=0.85,
                                      range=8.0, shadows=True) for x in (-1.8, 1.8)] + [
            PointLight(np.array([0.0, STAGE + OPENING_Y, -1.2]), color=(255, 235, 210), diffuse=0.5, range=6.0)]


# ----- the statues ---------------------------------------------------------------------------------------------

SLOTS_X, SLOT_Z = np.linspace(-2.8, 2.8, 8), -3.8  # eight plinths along the back of the stage
PLINTH, PLINTH_SIZE, STATUE_SCALE, LYING = 0.25, 0.4, 0.6, 0.1
STONE = (205, 205, 212)


def statue_mesh(objects):
    """The figure as it stands now, in stone: Model.bake() merges the parts in the pose their skin gives them (the
    body bent as it is, each part where its joint carries it) into a few meshes, which are one here, set with its
    feet on the origin."""
    baked = Model(Node(), objects).bake()
    mesh = merge_meshes([o.mesh for o in baked.objects])
    low, high = mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)
    centre = np.array([(low[0] + high[0]) / 2, low[1], (low[2] + high[2]) / 2])
    return Mesh(mesh.vertices - centre, mesh.faces)


class Statue:
    """A frozen puppet on its plinth, standing, tipping over (towards the audience) or lying down."""

    def __init__(self, mesh, slot):
        self.slot, self.state, self.angle, self.speed, self.drop = slot, "standing", 0.0, 0.0, 0.0
        self.object = Object3D(mesh, position=(SLOTS_X[slot], STAGE + PLINTH, SLOT_Z), scale=STATUE_SCALE, color=STONE,
                               specular=0.15, emissive=0.3)

    def knock(self):
        if self.state == "standing":
            self.state, self.angle, self.speed = "tipping", 0.03, 0.2

    def update(self, dt, gravity):
        if self.state == "tipping":  # (a rod tipping about its foot)
            height = STATUE_SCALE * 1.7
            self.speed += 1.5 * gravity / height * np.sin(self.angle) * dt
            self.angle += self.speed * dt
            if self.angle >= np.pi / 2:
                self.angle, self.state, self.speed = np.pi / 2, "falling", 0.0
        elif self.state == "falling":  # off the plinth, onto the stage
            self.speed += gravity * dt
            self.drop = min(self.drop + self.speed * dt, PLINTH - LYING)
            if self.drop >= PLINTH - LYING:
                self.state = "down"
        self.object.rotation = quat_axis_angle((1, 0, 0), self.angle)
        self.object.position = np.array([SLOTS_X[self.slot], STAGE + PLINTH - self.drop, SLOT_Z])


class Statues:
    """The row of plinths and what stands (or lies) at each."""

    def __init__(self):
        parts = [((x, PLINTH / 2, SLOT_Z), (PLINTH_SIZE, PLINTH, PLINTH_SIZE), (190, 185, 175)) for x in SLOTS_X]
        parts += [((x, PLINTH - 0.015, SLOT_Z), (PLINTH_SIZE + 0.06, 0.03, PLINTH_SIZE + 0.06), (215, 210, 200))
                  for x in SLOTS_X]
        self.plinths = Object3D(boxes(parts), position=(0.0, STAGE, 0.0), color=(255, 255, 255), specular=0.2)
        self.slots = [None] * len(SLOTS_X)
        self.age = []  # the slots in use, oldest first

    def statues(self):
        return [s for s in self.slots if s is not None]

    def objects(self):
        return [self.plinths] + [s.object for s in self.statues()]

    def triangles(self):
        return sum(len(s.object.mesh.faces) for s in self.statues())

    def freeze(self, figures):
        """Each of figures (lists of objects, a puppet's) as a statue, on a free plinth, or on the oldest statue's."""
        for objects in figures:
            free = [k for k, s in enumerate(self.slots) if s is None]
            slot = free[0] if free else self.age.pop(0)
            if slot in self.age:
                self.age.remove(slot)
            self.slots[slot] = Statue(statue_mesh(objects), slot)
            self.age.append(slot)

    def click(self, picked):
        """What happens when the object `picked` is clicked: a standing statue is knocked over, one lying down is
        swept away; a message, or None if it is not a statue's."""
        for s in self.statues():
            if s.object is picked:
                if s.state == "standing":
                    s.knock()
                    return "Timber!"
                if s.state == "down":
                    self.slots[s.slot] = None
                    self.age.remove(s.slot)
                    return "Swept away."
                return None
        return None

    def update(self, dt, gravity):
        for s in self.statues():
            s.update(dt, gravity)


# ----- the show ------------------------------------------------------------------------------------------------

class Show:
    def __init__(self):
        self.renderer = Renderer()
        self.renderer.background = Gradient((30, 22, 28), (8, 6, 10))
        self.bar = StatusBar(self.renderer)
        self.gravity = Slider("Gravity", GRAVITY, 1.6, 19.6, step=0.2, keys="[]", length=10, fmt=self._gravity)
        self.crowd = Slider("Puppets (-/=)", 1, 1, MAX_PUPPETS, step=1, keys="-=", length=8, fmt=lambda v: f"{v:2d}")
        self.auto = Toggle("Auto-walk (space)", True, key=" ")
        self.open = Toggle("Curtains open (c)", True, key="c")
        self.view = Choice("View (o)", ("front", "walk"), key="o")
        self.panel = Panel([self.gravity, self.crowd, self.auto, self.open, self.view], keyboard=False)
        self.theatre = Theatre()
        self.theatre.curtains.settle(1.0)
        self.doll = Doll()
        mesh, inverse_bind = figure_rig()
        self.rig = (singing(wooden(mesh)), inverse_bind)
        self.marionettes = []
        self.clock, self.times, self.inputs = 0.0, [], []  # what the lead was told to do, and when
        self.cast()
        self.statues = Statues()
        self.note, self.note_time = "", 0.0
        self.freezing = None  # (seconds till the gasping puppets turn to stone)
        self.x, self.z, self.yaw, self.pitch, self.eye_y = 0.0, 7.5, 0.0, 0.0, EYE
        self.zoom = 1.0
        self.drag = None
        self.frame_ms = 0.0

    def cast(self):
        """Add puppets to the chorus, or take them off the end, till there are as many as the slider says."""
        while len(self.marionettes) < self.crowd.value:
            k = len(self.marionettes)
            self.marionettes.append(Marionette(self.rig, self.doll, chorus_at(k), hat=HATS[k % len(HATS)]))
        del self.marionettes[self.crowd.value:]

    def told(self, k):
        """What puppet k of the chorus does now: what the lead was told to do k * LAG seconds ago (nothing, before)."""
        i = bisect_right(self.times, self.clock - k * LAG) - 1
        return self.inputs[i] if i >= 0 else (np.zeros(2), 0.0, 0.0, 0.0, (0.0, 0.0), 0.0)

    @staticmethod
    def _gravity(g):
        name = "Moon" if g < 2.5 else "Mars" if g < 5.0 else "Earth" if g < 12.0 else "heavy"
        return f"{g:4.1f} {name:5}"

    def objects(self):
        return (self.theatre.objects + self.theatre.curtains.objects + self.statues.objects()
                + [o for m in self.marionettes for o in m.objects()])

    def ahead(self):
        """The way the view looks, flat (x, z), unit."""
        if self.view.value == "front":
            return np.array([0.0, -1.0])
        return np.array([np.sin(self.yaw), -np.cos(self.yaw)])

    def control(self, dt, held):
        walking = self.view.value == "walk"
        fx, fz = self.ahead()
        ahead = ("w" in held) - ("s" in held)
        side = ("d" in held) - ("a" in held)
        move = np.array([fx * ahead - fz * side, fz * ahead + fx * side], float)
        if move.any():
            move /= np.linalg.norm(move)
        turn = ("," in held) - ("." in held)
        if not walking:
            turn += (Key.LEFT in held) - (Key.RIGHT in held)
        self.clock += dt
        self.times.append(self.clock)
        self.inputs.append((move, float(np.clip(turn, -1, 1)), ("r" in held) - ("f" in held),
                            ("e" in held) - ("q" in held), (float("x" in held), float("z" in held)),
                            float("m" in held)))
        keep = bisect_right(self.times, self.clock - MAX_PUPPETS * LAG - 1.0)
        del self.times[:keep], self.inputs[:keep]
        for k, m in enumerate(self.marionettes):
            m.control(dt, *self.told(k), self.auto.value)
        if walking:
            self.yaw += ((Key.RIGHT in held) - (Key.LEFT in held)) * TURN * dt
            step = ((Key.UP in held) - (Key.DOWN in held)) * WALK * dt
            if step:
                fx, fz = self.ahead()
                floor = self.eye_y - EYE
                hips = self.doll.at["hips"]
                puppets = [m.p[0, hips, ::2] for m in self.marionettes]
                if self.theatre.free(self.x + fx * step, self.z, floor, puppets):
                    self.x += fx * step
                if self.theatre.free(self.x, self.z + fz * step, floor, puppets):
                    self.z += fz * step
            want = self.theatre.floor(self.x, self.z) + EYE
            self.eye_y += (want - self.eye_y) * min(1.0, 10.0 * dt)

    def advance(self, dt, held):
        """Move the show on by dt seconds, with the keys `held`."""
        self.cast()
        self.control(dt, held)
        for m in self.marionettes:
            m.update(dt, self.gravity.value)
        pose_puppets([m.puppet for m in self.marionettes], np.stack([m.p[0] for m in self.marionettes]))
        if self.freezing is not None:
            self.freezing -= dt
            if self.freezing <= 0.0:
                self.statues.freeze([m.puppet.objects() for m in self.marionettes[:len(SLOTS_X)]])
                self.freezing = None
        self.statues.update(dt, self.gravity.value)
        self.note_time -= dt
        self.theatre.curtains.update(dt, self.open.value)

    def camera(self, cols, rows, cell_pixels):
        if self.view.value == "walk":
            eye = np.array([self.x, self.eye_y, self.z])
            fx, fz = self.ahead()
            look = np.array([fx * np.cos(self.pitch), np.sin(self.pitch), fz * np.cos(self.pitch)])
            return Camera(position=eye, target=eye + look, fov=70.0, near=0.05, far=60.0)
        aspect = cols * cell_pixels[0] / max(rows * cell_pixels[1], 1)
        target = np.array([0.0, STAGE + 1.6, -1.5])
        back = np.array([0.0, np.sin(0.3), np.cos(0.3)])
        size = max(OPENING_Y + 1.2, (2 * OPENING_X + 1.4) / aspect) / self.zoom
        # (from beyond the back wall of the hall, seeing only what is nearer the stage than all of it)
        position = target + 30.0 * back
        wall = np.dot(position - (0.0, 0.0, HALL_BACK), back)
        return Camera(position=position, target=target, projection="ortho", size=size, near=wall + 0.3, far=60.0)

    def handle(self, events, screen):
        for ev in self.panel.handle(self.bar.handle(events, screen)):
            if ev == Key.ESC:
                return False
            if ev == Key.ENTER and self.freezing is None:  # (a gasp, then stone at its height)
                self.freezing = GASP
                for m in self.marionettes[:len(SLOTS_X)]:
                    m.puppet.play([(0.0, (0.0, 0.0)), (GASP, (0.8, -0.9)), (GASP + 0.4, (0.8, -0.9)),
                                   (GASP + 0.8, (0.0, 0.0))], fade=0.0)
                self.say("Freeze!")
            if isinstance(ev, MouseEvent):
                if ev.button in (MouseEvent.WHEEL_UP, MouseEvent.WHEEL_DOWN) and self.view.value == "front":
                    self.zoom = float(np.clip(self.zoom * (1.15 if ev.button == MouseEvent.WHEEL_UP else 1 / 1.15),
                                              1.0, 3.0))
                elif ev.button == MouseEvent.LEFT and ev.pressed and not ev.moved:
                    picked = self.renderer.pick(ev)  # (what was drawn there, puppets bent as they stood)
                    said = self.statues.click(picked.object) if picked is not None else None
                    if said:
                        self.say(said)
                    self.drag = (ev.x, ev.y)
                elif ev.button == MouseEvent.LEFT and ev.pressed:
                    if ev.moved and self.drag is not None and self.view.value == "walk":
                        self.yaw += (ev.x - self.drag[0]) * 0.04
                        self.pitch = float(np.clip(self.pitch - (ev.y - self.drag[1]) * 0.06, -1.2, 1.2))
                    self.drag = (ev.x, ev.y)
                elif not ev.pressed:
                    self.drag = None
        return True

    def say(self, note):
        self.note, self.note_time = note, 2.0

    def frame(self, screen, dt, events):
        start = time.perf_counter()
        if not self.handle(events, screen):
            return False
        self.advance(dt, screen.held)
        rows, cols = screen.size()
        view_rows = max(rows - 2, 1)
        camera = self.camera(cols, view_rows, screen.cell_pixels)
        screen.draw(self.renderer, self.objects(), camera, self.theatre.lights(), rows=view_rows)
        ms = 1000.0 * (time.perf_counter() - start)
        self.frame_ms += (ms - self.frame_ms) * 0.1
        n = len(self.marionettes)
        joints = sum(len(m.puppet.object.skin.joints) for m in self.marionettes)
        triangles = sum(m.puppet.triangles() for m in self.marionettes) + self.statues.triangles()
        screen.text(0, 1, f"{n} puppet{'s' if n != 1 else ''}  {joints} joints  {triangles} triangles  "
                          f"{self.frame_ms:4.1f} ms a frame", dim=True)
        if self.note_time > 0.0:
            screen.text(1, 1, self.note)
        self.panel.draw(screen, rows - 2, 1)
        if self.view.value == "walk":
            walk = "[up/down] walk  [left/right] turn  [drag] look  "
            turn = "[,/.] turn"
        else:
            walk = "[wheel] zoom  "
            turn = "[,/. or left/right] turn"
        self.bar.draw(screen, f"[wasd] move the bar  {turn}  [r/f] up/down  [q/e] legs  [z/x] hands  [m] sing  [enter] freeze  "
                              f"[click] a statue  [-/=] puppets  {walk}[esc] quit")
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fps", type=int, default=30)
    add_display_args(parser)
    args = parser.parse_args()
    try:
        run(Show().frame, args.fps, mouse="drag", key_release=True, title="unicode3d marionettes",
            **display_options(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
