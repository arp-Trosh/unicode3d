# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
"""Animation: easing curves, keyframe tracks, and Animation, which moves an object along them.

    door = Animation(door_obj, rotation=RotationTrack([(0.0, closed), (0.8, open_, ease_out)]))
    ...
    door.update(dt)          # each frame; or door.apply(t) for a time of your own

A Track holds keyframes (time, value) and gives the value at any time between them, blending each pair
linearly (lerp) after easing the fraction of the way between them; a RotationTrack blends quaternions
along the arc (quat_slerp) instead. A keyframe's own easing, given as a third item, shapes the stretch
leading up to it; the track's `easing` shapes the others.
"""
import numpy as np

from .transforms import quat_slerp


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


EASINGS = {f.__name__: f for f in (linear, ease_in, ease_out, ease_in_out, ease_out_back, ease_out_bounce)}

LOOPS = ("once", "loop", "pingpong")


class Track:
    """Keyframes (time, value) or (time, value, easing), in any order, and the value at any time between them.

    Values are numbers or arrays (a position, a scale, a colour); easing is a function of 0..1 or the name of one
    in EASINGS. loop: "once" holds the first value before the first key and the last after the last; "loop"
    starts again from the first key; "pingpong" plays forwards, then backwards.
    """

    def __init__(self, keys, easing=linear, loop="once"):
        if not keys:
            raise ValueError("a track needs at least one keyframe")
        if loop not in LOOPS:
            raise ValueError(f"loop must be one of {LOOPS}, not {loop!r}")
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
        span = self.duration
        if span <= 0.0 or self.loop == "once":
            return min(max(t, self.start), self.end)
        u = (t - self.start) % (2.0 * span if self.loop == "pingpong" else span)
        return self.start + (2.0 * span - u if u > span else u)

    def at(self, t):
        """The value at time t."""
        t = self._local(float(t))
        i = int(np.searchsorted(self.times, t, side="right"))
        if i <= 0 or i >= len(self.times):  # before the first key or after the last: that key's value, a copy
            value = self.values[0 if i <= 0 else -1]
            return value.copy() if isinstance(value, np.ndarray) else value
        t0, t1 = self.times[i - 1], self.times[i]
        f = (t - t0) / (t1 - t0) if t1 > t0 else 1.0
        return self.blend(self.values[i - 1], self.values[i], self.easings[i](f))


class RotationTrack(Track):
    """A Track of rotations (quaternions, w first), turning between keyframes along the shortest arc."""

    @staticmethod
    def _value(value):
        return np.array(value, dtype=float).reshape(4)

    def blend(self, a, b, t):
        return quat_slerp(a, b, t)


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
