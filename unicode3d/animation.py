# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
"""Animation: easing curves, keyframe tracks, and Animation, which moves an object along them.

    door = Animation(door_obj, rotation=RotationTrack([(0.0, closed), (0.8, open_, ease_out)]))
    ...
    door.update(dt)          # each frame; or door.apply(t) for a time of your own

A Track holds keyframes (time, value) and gives the value at any time between them, blending each pair
linearly (lerp) after easing the fraction of the way between them; a RotationTrack blends quaternions
along the arc (quat_slerp) instead. A keyframe's own easing, given as a third item, shapes the stretch
leading up to it; the track's `easing` shapes the others. A SplineTrack curves through its keyframes
along tangents given with each (cubic Hermite splines, as glTF models' animations have).

A Clip plays several Animations on one clock: what models.load_model() gives for each animation in a
glTF file (Model.animations), and a way to group your own. It samples all of its tracks in one kernel
(_sample), with the same arithmetic as Track.at, so a track's keyframes are read once, the first time a clip
plays it: make a new Track rather than changing one in place.
"""
import math

import numpy as np
from numba import njit

from .transforms import normalize, quat_slerp


def lerp(a, b, t):
    """The value a fraction t of the way from a to b (numbers or arrays)."""
    return a + (np.asarray(b, dtype=float) - a) * t if isinstance(a, np.ndarray) else a + (b - a) * t


# Easing curves: each maps 0..1 onto 0..1 (0 to 0, 1 to 1), setting how a move speeds up and slows down.

def linear(t):
    return t


def ease_in(t):
    """Starts slowly and speeds up (cubic)."""
    return t * t * t


def ease_out(t):
    """Starts fast and slows to a stop (cubic)."""
    return 1.0 - (1.0 - t) ** 3


def ease_in_out(t):
    """Speeds up, then slows to a stop (cubic)."""
    return 4.0 * t * t * t if t < 0.5 else 1.0 - (-2.0 * t + 2.0) ** 3 / 2.0


def ease_out_back(t, overshoot=1.70158):
    """Goes a little past the end and settles back (a door swinging open against its stop)."""
    u = t - 1.0
    return 1.0 + (overshoot + 1.0) * u * u * u + overshoot * u * u


def ease_out_bounce(t):
    """Drops to the end and bounces there, less each time (something landing)."""
    n, d = 7.5625, 2.75
    if t < 1.0 / d:
        return n * t * t
    if t < 2.0 / d:
        t -= 1.5 / d
        return n * t * t + 0.75
    if t < 2.5 / d:
        t -= 2.25 / d
        return n * t * t + 0.9375
    t -= 2.625 / d
    return n * t * t + 0.984375


def step(t):
    """Holds the value of the keyframe before, then jumps to the next one's as it arrives."""
    return 0.0 if t < 1.0 else 1.0


EASINGS = {f.__name__: f for f in (linear, ease_in, ease_out, ease_in_out, ease_out_back, ease_out_bounce, step)}

LOOPS = ("once", "loop", "pingpong")


def _check_loop(loop):
    if loop not in LOOPS:
        raise ValueError(f"loop must be one of {LOOPS}, not {loop!r}")
    return loop


def _wrap(t, start, end, loop):
    """Time t brought within start..end, as `loop` says (see Track)."""
    span = end - start
    if span <= 0.0 or loop == "once":
        return min(max(t, start), end)
    u = (t - start) % (2.0 * span if loop == "pingpong" else span)
    return start + (2.0 * span - u if u > span else u)


class Track:
    """Keyframes (time, value) or (time, value, easing), in any order, and the value at any time between them.

    Values are numbers or arrays (a position, a scale, a colour); easing is a function of 0..1 or the name of one
    in EASINGS. loop: "once" holds the first value before the first key and the last after the last; "loop"
    starts again from the first key; "pingpong" plays forwards, then backwards.
    """

    def __init__(self, keys, easing=linear, loop="once"):
        if not keys:
            raise ValueError("a track needs at least one keyframe")
        _check_loop(loop)
        keys = sorted(keys, key=lambda k: k[0])
        self.times = np.array([float(k[0]) for k in keys])
        self.values = [self._value(k[1]) for k in keys]
        default = EASINGS[easing] if isinstance(easing, str) else easing
        self.easings = [EASINGS[k[2]] if len(k) > 2 and isinstance(k[2], str) else (k[2] if len(k) > 2 else default)
                        for k in keys]
        self.loop = loop

    @staticmethod
    def _value(value):
        return np.array(value, dtype=float) if np.ndim(value) else float(value)

    @property
    def start(self):
        return float(self.times[0])

    @property
    def end(self):
        return float(self.times[-1])

    @property
    def duration(self):
        return self.end - self.start

    def blend(self, a, b, t):
        return lerp(a, b, t)

    def _local(self, t):
        """t brought within start..end, as the loop setting says."""
        return _wrap(t, self.start, self.end, self.loop)

    def at(self, t):
        """The value at time t."""
        t = self._local(float(t))
        i = int(np.searchsorted(self.times, t, side="right"))
        if i <= 0 or i >= len(self.times):  # before the first key or after the last: that key's value, a copy
            value = self.values[0 if i <= 0 else -1]
            return value.copy() if isinstance(value, np.ndarray) else value
        t0, t1 = self.times[i - 1], self.times[i]
        f = (t - t0) / (t1 - t0) if t1 > t0 else 1.0
        return self.between(i, f)

    def between(self, i, f):
        """The value a fraction f of the way from keyframe i - 1 to keyframe i."""
        return self.blend(self.values[i - 1], self.values[i], self.easings[i](f))


class RotationTrack(Track):
    """A Track of rotations (quaternions, w first), turning between keyframes along the shortest arc."""

    @staticmethod
    def _value(value):
        return np.array(value, dtype=float).reshape(4)

    def blend(self, a, b, t):
        return quat_slerp(a, b, t)


class SplineTrack(Track):
    """Keyframes (time, value, in_tangent, out_tangent), and a smooth curve through them: between two keys it
    leaves the first along its out_tangent and arrives at the second along its in_tangent (a cubic Hermite
    spline; tangents are rates of change per second). rotation=True takes the values as quaternions (w first),
    and keeps what it gives a rotation (unit length). loop as for Track."""

    def __init__(self, keys, loop="once", rotation=False):
        self.rotation = rotation
        keys = sorted(keys, key=lambda k: k[0])
        super().__init__([(k[0], k[1]) for k in keys], loop=loop)
        self.tangents = [(self._value(k[2]), self._value(k[3])) for k in keys]

    def between(self, i, f):
        span = self.times[i] - self.times[i - 1]
        f2, f3 = f * f, f * f * f
        value = ((2 * f3 - 3 * f2 + 1) * self.values[i - 1] + (f3 - 2 * f2 + f) * span * self.tangents[i - 1][1]
                 + (-2 * f3 + 3 * f2) * self.values[i] + (f3 - f2) * span * self.tangents[i][0])
        if self.rotation:
            return normalize(value)
        return value if isinstance(value, np.ndarray) else float(value)

    def at(self, t):
        value = super().at(t)
        return normalize(value) if self.rotation and isinstance(value, np.ndarray) else value


class Animation:
    """Moves an object (an Object3D, a Node, or anything with those attributes) by tracks: position and scale
    Tracks and a rotation RotationTrack, any of them None to leave that alone. Plays from time 0: call
    update(dt) each frame, or apply(t) for a time of your own. speed scales time (2 plays twice as fast)."""

    def __init__(self, target, position=None, rotation=None, scale=None, speed=1.0):
        self.target = target
        self.tracks = {name: track for name, track in (("position", position), ("rotation", rotation),
                                                        ("scale", scale)) if track is not None}
        self.speed = speed
        self.time = 0.0

    @property
    def duration(self):
        """When the last track ends (inf if one loops)."""
        ends = [np.inf if track.loop != "once" else track.end for track in self.tracks.values()]
        return max(ends, default=0.0)

    def done(self):
        return self.time >= self.duration

    def apply(self, t):
        """Put the target where the tracks have it at time t."""
        self.time = float(t)
        for name, track in self.tracks.items():
            value = track.at(self.time)
            setattr(self.target, name, value.copy() if isinstance(value, np.ndarray) else value)

    def update(self, dt):
        """Move on by dt seconds (times speed) and apply; returns whether it is still playing."""
        self.apply(self.time + dt * self.speed)
        return not self.done()


# ----------------------------------------------------------------------------- Clips' tracks sampled in a kernel
#
# A Clip packs its tracks into arrays (each track's keyframes once, kept on the track and so shared by a model's
# copies) and samples them all in one call of _sample, with the arithmetic of Track.at, between and blend, and of
# lerp, quat_slerp, normalize and the easing curves, in the same order. What the kernel can't do exactly (a Track
# subclass or easing function of the program's own, values not all numbers or all 3- or 4-vectors, two tracks
# moving the same attribute, an Animation subclass) leaves the clip on Track.at, as before.

_LERP, _SLERP, _SPLINE, _SPLINE_ROTATION = 0, 1, 2, 3  # kinds of track
_LOOP_CODES = {"once": 0, "loop": 1, "pingpong": 2}
_EASING_CODES = {f: i for i, f in enumerate((linear, ease_in, ease_out, ease_in_out, ease_out_back, ease_out_bounce,
                                             step))}
_WIDTHS = (1, 3, 4)  # the values of each group: numbers, 3-vectors (positions, scales), 4-vectors (rotations)


@njit(cache=True, error_model="numpy")
def _ease(code, t):
    """Easing curve `code` (see _EASING_CODES) at t, as the Python functions compute it (** 3.0: libm's pow, as
    Python's ** 3 uses)."""
    if code == 0:
        return t
    if code == 1:
        return t * t * t
    if code == 2:
        return 1.0 - (1.0 - t) ** 3.0
    if code == 3:
        return 4.0 * t * t * t if t < 0.5 else 1.0 - (-2.0 * t + 2.0) ** 3.0 / 2.0
    if code == 4:
        u = t - 1.0
        return 1.0 + (1.70158 + 1.0) * u * u * u + 1.70158 * u * u
    if code == 5:
        n, d = 7.5625, 2.75
        if t < 1.0 / d:
            return n * t * t
        if t < 2.0 / d:
            t -= 1.5 / d
            return n * t * t + 0.75
        if t < 2.5 / d:
            t -= 2.25 / d
            return n * t * t + 0.9375
        t -= 2.625 / d
        return n * t * t + 0.984375
    return 0.0 if t < 1.0 else 1.0


@njit(cache=True, error_model="numpy")
def _wrap_time(t, start, end, loop):
    """_wrap, with `loop` as a code (_LOOP_CODES); min and max as Python's (NaN stays NaN)."""
    span = end - start
    if span <= 0.0 or loop == 0:
        t = start if start > t else t
        return end if end < t else t
    u = (t - start) % (2.0 * span if loop == 2 else span)
    return start + (2.0 * span - u if u > span else u)


@njit(cache=True, error_model="numpy")
def _normalize4(v):
    """normalize() of a 4-vector, in place."""
    n = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3])
    if n > 1e-12:
        for c in range(4):
            v[c] = v[c] / n


@njit(cache=True, error_model="numpy")
def _slerp4(a, b, t, out, na, nb):
    """quat_slerp(a, b, t) into out (na, nb: scratch 4-vectors)."""
    for c in range(4):
        na[c], nb[c] = a[c], b[c]
    _normalize4(na)
    _normalize4(nb)
    d = na[0] * nb[0] + na[1] * nb[1] + na[2] * nb[2] + na[3] * nb[3]
    if d < 0.0:
        for c in range(4):
            nb[c] = -nb[c]
        d = -d
    if d > 0.9995:
        for c in range(4):
            out[c] = na[c] + t * (nb[c] - na[c])
        _normalize4(out)
        return
    angle = math.acos(1.0 if 1.0 < d else d)
    s0, s1, s = math.sin((1.0 - t) * angle), math.sin(t * angle), math.sin(angle)
    for c in range(4):
        out[c] = (s0 * na[c] + s1 * nb[c]) / s


@njit(cache=True, error_model="numpy")
def _sample(t, w, kinds, loops, firsts, counts, groups, rows, fade_slerp, times, values, tangents, easings,
            start1, start3, start4, out1, out3, out4):
    """Every track of a clip at time t, as Track.at gives it, into out1 (numbers), out3 or out4 (vectors) at the
    track's row; and while a fade runs (w < 1), blended from its starting pose (start1/3/4) as Clip.apply does.
    tangents[k] holds keyframe k's in and out tangents (spline tracks)."""
    v, s, na, nb = np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4)
    for k in range(kinds.shape[0]):
        first, n, kind = firsts[k], counts[k], kinds[k]
        u = _wrap_time(t, times[first], times[first + n - 1], loops[k])
        lo, hi = 0, n  # np.searchsorted(times, u, side="right"), NaN after everything as numpy has it
        while lo < hi:
            mid = (lo + hi) >> 1
            m = times[first + mid]
            if u < m or (m != m and u == u):
                hi = mid
            else:
                lo = mid + 1
        width = (1, 3, 4)[groups[k]]
        if lo <= 0 or lo >= n:  # before the first key or after the last: that key's value
            j = first if lo <= 0 else first + n - 1
            for c in range(width):
                v[c] = values[j, c]
            if kind == 3:
                _normalize4(v)
        else:
            a, b = first + lo - 1, first + lo
            t0, t1 = times[a], times[b]
            f = (u - t0) / (t1 - t0) if t1 > t0 else 1.0
            if kind == 0:
                e = _ease(easings[b], f)
                for c in range(width):
                    v[c] = values[a, c] + (values[b, c] - values[a, c]) * e
            elif kind == 1:
                _slerp4(values[a], values[b], _ease(easings[b], f), v, na, nb)
            else:
                span = t1 - t0
                f2 = f * f
                f3 = f * f * f
                h0, h1, h2, h3 = 2 * f3 - 3 * f2 + 1, (f3 - 2 * f2 + f) * span, -2 * f3 + 3 * f2, (f3 - f2) * span
                for c in range(width):
                    v[c] = h0 * values[a, c] + h1 * tangents[a, 1, c] + h2 * values[b, c] + h3 * tangents[b, 0, c]
                if kind == 3:  # SplineTrack.between normalizes, and at() again
                    _normalize4(v)
                    _normalize4(v)
        row, group = rows[k], groups[k]
        if w < 1.0:
            if group == 0:
                v[0] = start1[row] + (v[0] - start1[row]) * w
            elif group == 2 and fade_slerp[k]:
                for c in range(4):
                    s[c] = v[c]
                _slerp4(start4[row], s, w, v, na, nb)
            else:
                start = start3[row] if group == 1 else start4[row]
                for c in range(width):
                    v[c] = start[c] + (v[c] - start[c]) * w
        if group == 0:
            out1[row] = v[0]
        elif group == 1:
            for c in range(3):
                out3[row, c] = v[c]
        else:
            for c in range(4):
                out4[row, c] = v[c]


def _width(values):
    """1, 3 or 4 if every value is a number, or every one a 3- or 4-vector; else None."""
    if all(isinstance(value, float) for value in values):
        return 1
    shapes = {np.shape(value) for value in values if isinstance(value, np.ndarray)}
    if len(shapes) == 1 and all(isinstance(value, np.ndarray) for value in values):
        shape = shapes.pop()
        return shape[0] if shape in ((3,), (4,)) else None
    return None


def _track_piece(track):
    """track's keyframes as _sample takes them, worked out once and kept on the track: (kind, width, loop code,
    times (k,), values (k, 4), tangents (k, 2, 4), easing codes (k,)), or None if the kernel can't play it."""
    piece = track.__dict__.get("_piece", False)
    if piece is not False:
        return piece
    piece = None
    kind = {Track: _LERP, RotationTrack: _SLERP, SplineTrack: _SPLINE}.get(type(track))
    if kind == _SPLINE and track.rotation:
        kind = _SPLINE_ROTATION
    values = list(track.values)
    if kind == _SPLINE:
        values += [t for pair in track.tangents for t in pair]
    width = _width(values) if kind is not None else None
    easings = [_EASING_CODES.get(e) for e in track.easings]
    if kind in (_SPLINE, _SPLINE_ROTATION):
        easings = [0] * len(easings)  # (not used)
    if (width is not None and None not in easings and track.loop in _LOOP_CODES
            and (width == 4 or kind in (_LERP, _SPLINE)) and len(track.times) == len(track.values)):
        k = len(track.times)
        packed = np.zeros((k, 4))
        packed[:, :width] = np.reshape(np.array(track.values, dtype=float), (k, -1))
        tangents = np.zeros((k, 2, 4))
        if kind in (_SPLINE, _SPLINE_ROTATION):
            tangents[:, :, :width] = np.reshape(np.array(track.tangents, dtype=float), (k, 2, -1))
        piece = (kind, width, _LOOP_CODES[track.loop], np.array(track.times, dtype=float), packed, tangents,
                 np.array(easings, dtype=np.int8))
    track._piece = piece
    return piece


class _PackedClip:
    """A Clip's tracks as _sample takes them (ok False: some can't be, and Track.at plays the clip), with what it
    was made from, so the clip can tell when that changes."""

    def __init__(self, animations):
        self.animations = list(animations)
        self.targets = [a.target for a in animations]
        self.tracks = [dict(a.tracks) for a in animations]
        self.duration = max((track.end for a in animations for track in a.tracks.values()), default=0.0)
        self.ok = False
        self.slots = ([], [], [])  # (target, attribute) of each row of each group
        self.where = {}            # (id(target), attribute): (group, row)
        columns = [[] for _ in range(8)]  # kinds, loops, firsts, counts, groups, rows, fade_slerp, pieces
        for a in animations:
            if type(a) is not Animation:
                return
            for name, track in a.tracks.items():
                piece = _track_piece(track)
                key = (id(a.target), name)
                if piece is None or key in self.where:
                    return
                group = _WIDTHS.index(piece[1])
                self.where[key] = (group, len(self.slots[group]))
                self.slots[group].append((a.target, name))
                columns[0].append(piece[0])
                columns[1].append(piece[2])
                columns[2].append(sum(len(p[3]) for p in columns[7]))
                columns[3].append(len(piece[3]))
                columns[4].append(group)
                columns[5].append(self.where[key][1])
                columns[6].append(name == "rotation")
                columns[7].append(piece)
        pieces = columns[7]
        if not pieces:
            return
        self.kinds, self.loops = np.array(columns[0], np.int8), np.array(columns[1], np.int8)
        self.firsts, self.counts = np.array(columns[2], np.int64), np.array(columns[3], np.int64)
        self.groups, self.rows = np.array(columns[4], np.int8), np.array(columns[5], np.int64)
        self.fade_slerp = np.array(columns[6], np.bool_)
        self.times = np.ascontiguousarray(np.concatenate([p[3] for p in pieces]))
        self.values = np.ascontiguousarray(np.concatenate([p[4] for p in pieces]))
        self.tangents = np.ascontiguousarray(np.concatenate([p[5] for p in pieces]))
        self.easings = np.ascontiguousarray(np.concatenate([p[6] for p in pieces]))
        self.no_start = (np.zeros(0), np.zeros((0, 3)), np.zeros((0, 4)))
        self.ok = True

    def current(self, animations):
        """Whether it was made from these animations, with the same targets and tracks."""
        return (animations == self.animations and [a.target for a in animations] == self.targets
                and [a.tracks for a in animations] == self.tracks)

    def starts(self, pose):
        """A fade's starting pose (see Clip.start) as _sample's start1/3/4, or None if it doesn't fit the tracks."""
        starts = (np.zeros(len(self.slots[0])), np.zeros((len(self.slots[1]), 3)), np.zeros((len(self.slots[2]), 4)))
        if len(pose) != len(self.kinds):
            return None
        for target, name, start in pose:
            where = self.where.get((id(target), name))
            if where is None or start.shape != ((), (3,), (4,))[where[0]]:
                return None
            starts[where[0]][where[1]] = start
        return starts

    def sample(self, t, w, starts):
        """Every track at time t (blended from starts while w < 1), set on its target."""
        start1, start3, start4 = starts if starts is not None else self.no_start
        out1, out3, out4 = np.empty(len(self.slots[0])), np.empty((len(self.slots[1]), 3)), np.empty(
            (len(self.slots[2]), 4))
        _sample(t, w, self.kinds, self.loops, self.firsts, self.counts, self.groups, self.rows, self.fade_slerp,
                self.times, self.values, self.tangents, self.easings, start1, start3, start4, out1, out3, out4)
        for (target, name), value in zip(self.slots[0], out1.tolist()):
            setattr(target, name, value)
        for (target, name), value in zip(self.slots[1], out3):  # each its own row of this frame's array
            setattr(target, name, value)
        for (target, name), value in zip(self.slots[2], out4):
            setattr(target, name, value)


class Clip:
    """Animations played together on one clock, from time 0 to the end of the longest: an animation clip, like
    those a glTF model carries (a door opening, a robot waving its arm), moving several parts at once.

        wave = model.animations["Wave"]
        wave.loop = "loop"
        ...
        wave.update(dt)          # each frame; or wave.apply(t)

    loop: "once" holds the last pose once it ends, "loop" starts again, "pingpong" plays it back and forth
    (each Animation's own tracks are best left "once": the clip does the looping). speed scales time.
    """

    def __init__(self, animations, name="", loop="once", speed=1.0):
        self.animations = list(animations)
        self.name = name
        self.loop = _check_loop(loop)
        self.speed = speed
        self.time = 0.0
        self._fade = None  # [[(target, attribute, value at the start), ...], seconds faded, seconds to fade, starts]
        self._packed = None  # its tracks as the kernel takes them (_PackedClip)
        self._kernel = True  # (False: Track.at plays it, for comparing)

    def _pack(self):
        """Its tracks packed for _sample, made again when its animations, their targets or tracks change."""
        if self._packed is None or not self._packed.current(self.animations):
            self._packed = _PackedClip(self.animations)
        return self._packed

    @property
    def duration(self):
        """When the last track of its animations ends (they start at time 0)."""
        return self._pack().duration

    @property
    def targets(self):
        """What it moves: each animation's target."""
        return [a.target for a in self.animations]

    def done(self):
        return self.loop == "once" and self.time >= self.duration

    def start(self, fade=0.0):
        """Play from the beginning (time 0). With fade, in seconds, ease from the pose its targets have now (left by
        another clip, or by code) into the clip's over that long, rather than jumping: a crossfade from Walk into
        Attack. The fade runs on update()'s dt (not scaled by speed); starting again, or another clip moving the
        same targets, takes over from wherever they are."""
        self.time = 0.0
        self._fade = None
        fade = float(fade)
        if fade > 0.0 and math.isfinite(fade):
            pose = [(a.target, name, np.array(getattr(a.target, name), dtype=float))
                    for a in self.animations for name in a.tracks]
            self._fade = [pose, 0.0, fade, None]
        self.apply(0.0)

    @property
    def fading(self):
        """Whether it is still easing in from where start(fade) found its targets."""
        return self._fade is not None

    def apply(self, t):
        """Put every target where the clip has it at time t (looped, if it loops), blended with where it started
        from while it fades in (see start)."""
        self.time = float(t)
        packed = self._pack()
        local = _wrap(self.time, 0.0, packed.duration, _check_loop(self.loop))
        w, starts = 1.0, None
        if self._fade is not None:
            pose, faded, fade, starts = self._fade
            w = ease_in_out(max(faded, 0.0) / fade) if faded < fade else 1.0  # (a NaN dt ends it)
            if w >= 1.0:
                self._fade = None
            elif starts is None and packed.ok:
                starts = self._fade[3] = packed.starts(pose)
        if packed.ok and self._kernel and (w >= 1.0 or starts is not None):
            for animation in self.animations:
                animation.time = local
            packed.sample(local, w, starts if w < 1.0 else None)
            return
        for animation in self.animations:
            animation.apply(local)
        if w < 1.0:
            for target, name, start in pose:
                now = getattr(target, name)
                if name == "rotation":
                    value = quat_slerp(start, now, w)
                else:
                    value = start + (np.asarray(now, dtype=float) - start) * w
                    value = float(value) if np.ndim(now) == 0 and value.shape == () else value
                setattr(target, name, value)

    def update(self, dt):
        """Move on by dt seconds (times speed) and apply; returns whether it is still playing."""
        if self._fade is not None:
            self._fade[1] += dt
        self.apply(self.time + dt * self.speed)
        return not self.done()
