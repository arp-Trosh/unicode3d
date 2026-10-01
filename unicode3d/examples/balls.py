# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
"""Bouncing balls in a room: python -m unicode3d.examples.balls [--balls N] [--seed N]

Balls of many colours drift around a room seen from outside (the walls nearest
the camera are see-through, since only their insides are drawn). The panel on
the left sets how many balls there are, their size, the room's size and their
speed, and switches collisions, gravity, squashing on bounces, mixed materials, glass balls, a mirror
floor, the lamp and the camera's orbit (F5 switches shadows, F6 reflections).
Click a ball to make it glow. Tab moves through the panel; arrows then change
the focused setting, otherwise they turn the camera.
"""
import argparse
import colorsys

import numpy as np
from numba import njit

from .hud import StatusBar
from ..background import Gradient
from ..keys import Key, MouseEvent
from ..mesh import Mesh, make_box
from ..scene import Camera, Light, Object3D, PointLight, Renderer
from ..shapes import blob_mesh
from ..terminal import add_display_args, display_options, run
from ..transforms import quat_identity
from ..ui import Panel, Slider, Toggle

GLASS_OPACITY = 0.3  # of the glass balls
# Squash and stretch (s): a bump that changes a ball's velocity by dv (units a second) squashes it along dv, by
# SQUASH * |dv| (at most MAX_SQUASH; bumps under MIN_BUMP, like resting on the floor, are ignored), and it wobbles
# back to round in about half a second.
SQUASH, MAX_SQUASH, MIN_BUMP = 0.12, 0.4, 0.6
WOBBLE_DECAY, WOBBLE_RATE = 7.0, 22.0  # per second; radians a second
# Mixed materials (t): (specular, shininess, reflectivity), one of these picked at random for each ball.
FINISHES = ((0.0, None, 0.0),   # rubber
            (1.0, None, 0.0),   # plastic
            (2.5, 60.0, 0.5),   # metal, reflecting the gradient behind the room
            (0.8, 4.0, 0.1))    # pearl
MAX_BALLS = 500   # about 15 ms a frame at 180x50 cells on a 6-core machine: room to push, not to freeze
BASE_SPEED = 1.5  # units a second at speed 1
IDENTITY = quat_identity()


def grid_texture(res, base, line, cells=8):
    """A tiled texture: `base` colour with `line` coloured grid lines, `cells` tiles across (sRGB 0..1)."""
    t = np.ones((res, res, 3)) * base
    step = res // cells
    t[::step] = line
    t[:, ::step] = line
    return t


def room_mesh():
    """A unit box with its faces turned inwards, tiled: (walls and a ceiling, the floor), the floor apart so
    that it is flat, and can be a mirror."""
    wall = grid_texture(64, (0.75, 0.8, 0.9), (0.45, 0.5, 0.65))
    floor = grid_texture(64, (0.85, 0.72, 0.55), (0.55, 0.42, 0.3))
    ceiling = grid_texture(64, (0.9, 0.9, 0.88), (0.7, 0.7, 0.68), cells=4)
    box = make_box(1.0, [wall, wall, ceiling, floor, wall, wall])
    # Reversed winding faces each side inwards; its corners' uvs go with it.
    floor = box.materials == 3
    return tuple(Mesh(box.vertices, box.faces[keep][:, ::-1].copy(), box.uvs[keep][:, ::-1].copy(), box.materials[keep],
                      box.textures) for keep in (~floor, floor))


@njit(cache=True)
def collide(pos, vel, radii):
    """Elastic bounces between touching balls (heavier for bigger), pushing each pair apart; in place.

    Serial, pair by pair: a pair's change must be seen by the next pair that shares a ball."""
    n = pos.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy, dz = pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1], pos[i, 2] - pos[j, 2]
            reach = radii[i] + radii[j]
            d2 = dx * dx + dy * dy + dz * dz
            if d2 >= reach * reach:
                continue
            d = max(np.sqrt(d2), 1e-9)
            nx, ny, nz = dx / d, dy / d, dz / d
            mi, mj = radii[i] ** 3, radii[j] ** 3
            approach = (vel[i, 0] - vel[j, 0]) * nx + (vel[i, 1] - vel[j, 1]) * ny + (vel[i, 2] - vel[j, 2]) * nz
            if approach < 0:
                impulse = -2 * approach / (1 / mi + 1 / mj)
                vel[i, 0] += impulse / mi * nx
                vel[i, 1] += impulse / mi * ny
                vel[i, 2] += impulse / mi * nz
                vel[j, 0] -= impulse / mj * nx
                vel[j, 1] -= impulse / mj * ny
                vel[j, 2] -= impulse / mj * nz
            push = (reach - d) / 2
            pos[i, 0] += push * nx
            pos[i, 1] += push * ny
            pos[i, 2] += push * nz
            pos[j, 0] -= push * nx
            pos[j, 1] -= push * ny
            pos[j, 2] -= push * nz


class Balls:
    def __init__(self, count, seed=None):
        self.rng = np.random.default_rng(seed)
        # Levels of detail: balls small on screen get fewer triangles, which look the same at that size.
        self.lods = [(4.0, blob_mesh((1.0, 1.0, 1.0), rings=6, segments=8)),
                     (10.0, blob_mesh((1.0, 1.0, 1.0), rings=8, segments=12)),
                     (np.inf, blob_mesh((1.0, 1.0, 1.0), rings=12, segments=16))]
        # The room has a ceiling, which would put all of it in shadow; only the balls cast shadows.
        walls, floor = room_mesh()
        self.room = Object3D(walls, color=(255, 255, 255), cast_shadows=False)
        self.floor = Object3D(floor, color=(255, 255, 255), cast_shadows=False)
        self.bulb = Object3D(blob_mesh((1.0, 1.0, 1.0), rings=8, segments=12), color=(255, 235, 190), emissive=1.0,
                             cast_shadows=False)
        self.renderer = Renderer(1, 1, background=Gradient((22, 26, 40), (4, 4, 8)))
        self.bar = StatusBar(self.renderer)
        self.count = Slider("Balls", count, 1, MAX_BALLS, keys="[]", length=14)
        self.size = Slider("Size ", 0.4, 0.1, 1.5, step=0.05, keys="-=", length=14, fmt=lambda v: f"{v:.2f}")
        self.room_size = Slider("Room ", 10.0, 3.0, 30.0, step=0.5, keys=",.", length=14, fmt=lambda v: f"{v:.1f}")
        self.speed = Slider("Speed", 1.0, 0.0, 3.0, step=0.1, keys=";'", length=14, fmt=lambda v: f"{v:.1f}")
        self.collide = Toggle("Collisions (c)", True, key="c")
        self.gravity = Toggle("Gravity (g)", False, key="g")
        self.squash = Toggle("Squash (s)", True, key="s")
        self.mixed = Toggle("Materials (t)", False, key="t")
        self.mirror = Toggle("Mirror floor (m)", False, key="m")
        self.glass = Toggle("Glass (x)", False, key="x")
        self.lamp = Toggle("Lamp (l)", True, key="l")
        self.orbit = Toggle("Orbit (o)", True, key="o")
        self.panel = Panel([self.count, self.size, self.room_size, self.speed, self.collide, self.gravity,
                            self.squash, self.mixed, self.glass, self.mirror, self.lamp, self.orbit], vertical=True)
        self.pos = np.zeros((0, 3))
        self.vel = np.zeros((0, 3))
        self.radius_factor = np.zeros(0)
        self.finish = np.zeros(0, np.int64)  # an index into FINISHES, for mixed materials
        # Each ball's latest squash: along which direction (unit), how much at first, and how long ago.
        self.squash_dir, self.squash_amount, self.squash_age = np.zeros((0, 3)), np.zeros(0), np.zeros(0)
        self.balls = []
        self.yaw, self.pitch, self.paused = 0.6, 0.45, False

    # ----- the balls ---------------------------------------------------------------------

    def sync_count(self):
        """Add or remove balls to match the slider; new ones start at random places, heading anywhere."""
        n, have = self.count.value, len(self.balls)
        if n < have:
            del self.balls[n:]
            self.pos, self.vel, self.radius_factor = self.pos[:n], self.vel[:n], self.radius_factor[:n]
            self.finish, self.squash_dir = self.finish[:n], self.squash_dir[:n]
            self.squash_amount, self.squash_age = self.squash_amount[:n], self.squash_age[:n]
        elif n > have:
            k = n - have
            half = self.room_size.value / 2
            self.pos = np.r_[self.pos, self.rng.uniform(-half * 0.8, half * 0.8, (k, 3))]
            direction = self.rng.normal(size=(k, 3))
            direction /= np.linalg.norm(direction, axis=1, keepdims=True)
            self.vel = np.r_[self.vel, direction * self.rng.uniform(0.5, 1.0, (k, 1))]
            self.radius_factor = np.r_[self.radius_factor, self.rng.uniform(0.7, 1.3, k)]
            self.finish = np.r_[self.finish, self.rng.integers(len(FINISHES), size=k)]
            self.squash_dir = np.r_[self.squash_dir, np.tile([1.0, 0.0, 0.0], (k, 1))]
            self.squash_amount, self.squash_age = np.r_[self.squash_amount, np.zeros(k)], np.r_[self.squash_age, np.zeros(k)]
            for _ in range(k):
                r, g, b = colorsys.hsv_to_rgb(self.rng.uniform(), self.rng.uniform(0.55, 0.95), self.rng.uniform(0.75, 1.0))
                self.balls.append(Object3D(self.lods[-1][1], color=(r, g, b)))

    def step(self, dt):
        radii = self.size.value * self.radius_factor
        half = self.room_size.value / 2
        if self.gravity.value:
            self.vel[:, 1] -= 4.0 * dt / max(self.speed.value, 0.1)
        before = self.vel.copy()
        self.pos += self.vel * (BASE_SPEED * self.speed.value * dt)
        # Walls: reflect whatever has gone through one, and turn it back inwards.
        room = np.maximum(half - radii, 0.0)[:, None]
        over, under = self.pos > room, self.pos < -room
        self.pos = np.where(over, 2 * room - self.pos, np.where(under, -2 * room - self.pos, self.pos))
        self.pos = np.clip(self.pos, -room, room)
        self.vel = np.where(over, -np.abs(self.vel), np.where(under, np.abs(self.vel), self.vel))
        if self.collide.value and len(self.pos) > 1:
            collide(self.pos, self.vel, radii)
        # Bumps (off a wall or another ball) squash the ball along the change in its velocity.
        bump = self.vel - before
        size = np.linalg.norm(bump, axis=1)
        hard = size * (BASE_SPEED * self.speed.value)  # in units a second
        hit = hard > MIN_BUMP
        self.squash_age += dt
        if hit.any():
            self.squash_dir[hit] = bump[hit] / size[hit, None]
            self.squash_amount[hit] = np.minimum(SQUASH * hard[hit], MAX_SQUASH)
            self.squash_age[hit] = 0.0

    def squashes(self):
        """How much each ball is squashed now along its squash_dir (negative: stretched), wobbling back to 0."""
        age = self.squash_age
        return self.squash_amount * np.exp(-WOBBLE_DECAY * age) * np.cos(WOBBLE_RATE * age)

    def shape(self, ball, position, radius, squash, direction):
        """Place a ball: round, or squashed by `squash` along `direction` and bulging the other ways, its centre
        moved back the way it was squashed so that it stays touching what it hit."""
        if abs(squash) < 0.01:
            ball.position, ball.rotation, ball.scale = position, IDENTITY, radius
            return
        d = direction if direction[0] >= 0 else -direction  # (squashing is the same either way along it)
        q = np.array([1.0 + d[0], 0.0, -d[2], d[1]])         # turns x onto d
        ball.rotation = q / np.linalg.norm(q)
        ball.scale = radius * np.array([1.0 - squash, 1.0 + squash / 2, 1.0 + squash / 2])
        ball.position = position - direction * radius * max(squash, 0.0)

    # ----- a frame -----------------------------------------------------------------------

    def frame(self, screen, dt, events):
        dt = min(dt, 0.05)
        events = self.panel.handle(self.bar.handle(events, screen))
        for ev in events:
            if ev in (ord("q"), Key.ESC):
                return False
            if ev == ord(" "):
                self.paused = not self.paused
            elif ev == Key.LEFT:
                self.yaw -= 0.1
            elif ev == Key.RIGHT:
                self.yaw += 0.1
            elif ev == Key.UP:
                self.pitch = min(self.pitch + 0.08, 1.4)
            elif ev == Key.DOWN:
                self.pitch = max(self.pitch - 0.08, -0.2)
            elif isinstance(ev, MouseEvent) and ev.pressed and ev.button == MouseEvent.LEFT and not ev.moved:
                hit = self.renderer.pick(ev.x, ev.y)
                if hit is not None and any(hit.object is b for b in self.balls):
                    hit.object.emissive = 0.0 if hit.object.emissive else 0.7

        self.sync_count()
        if not self.paused:
            self.step(dt)
            if self.orbit.value:
                self.yaw += 0.15 * dt

        # Place everything.
        s = self.room_size.value
        half = s / 2
        self.room.scale = self.floor.scale = s
        self.floor.reflectivity = 0.55 if self.mirror.value else 0.0
        radii = self.size.value * self.radius_factor
        squashes = self.squashes() if self.squash.value else np.zeros(len(self.balls))
        for i, (ball, p, r) in enumerate(zip(self.balls, self.pos, radii)):
            self.shape(ball, p, r, squashes[i], self.squash_dir[i])
            ball.opacity = GLASS_OPACITY if self.glass.value and i % 3 == 0 else 1.0  # every third one glass
            ball.specular, ball.shininess, ball.reflectivity = FINISHES[self.finish[i] if self.mixed.value else 1]
        lights = [Light(direction=np.array([0.4, -1.0, -0.3]), ambient=0.25, diffuse=0.45 if self.lamp.value else 0.7,
                        shadows=True)]
        objects = [self.room, self.floor, *self.balls]
        if self.lamp.value:
            lamp_at = np.array([0.0, half * 0.8, 0.0])
            self.bulb.position, self.bulb.scale = lamp_at, 0.04 * s
            lights.append(PointLight(lamp_at, color=(255, 225, 170), diffuse=0.9, range=1.6 * s, shadows=True))
            objects.append(self.bulb)

        distance = s * 1.9 + 2
        target = np.zeros(3)
        camera = Camera(position=target + distance * np.array([np.sin(self.yaw) * np.cos(self.pitch), np.sin(self.pitch),
                                                               np.cos(self.yaw) * np.cos(self.pitch)]),
                        target=target, fov=45.0, near=0.1, far=10 * s + 20)

        rows, cols = screen.size()
        self.renderer.resize(cols, max(rows - 1, 1), screen.cell_pixels)
        # Each ball's radius in pixels picks its level of detail.
        pixels_per_unit = self.renderer.drawn_size[1] / (2 * np.tan(np.radians(camera.fov) / 2))
        if len(self.balls):
            on_screen = radii * pixels_per_unit / np.maximum(np.linalg.norm(self.pos - camera.position, axis=1), 1e-6)
            for ball, px in zip(self.balls, on_screen):
                ball.mesh = next(mesh for limit, mesh in self.lods if px < limit)
        fb = self.renderer.render(objects, camera, lights)
        screen.erase()
        screen.draw_frame(fb)
        self.panel.draw(screen, 0, 1)
        tris = sum(len(ball.mesh.faces) for ball in self.balls)
        self.bar.draw(screen, f"{len(self.balls)} balls, {tris:,} triangles{'  (paused)' if self.paused else ''}   "
                              f"[space] pause  [arrows] turn  [click] a ball glows  [tab] panel  [q] quit")
        screen.refresh()
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-n", "--balls", type=int, default=40, help=f"number of balls to start with (1-{MAX_BALLS})")
    parser.add_argument("--seed", type=int, help="random seed")
    parser.add_argument("--fps", type=int, default=30)
    add_display_args(parser)
    args = parser.parse_args()
    demo = Balls(min(max(args.balls, 1), MAX_BALLS), args.seed)
    try:
        run(demo.frame, args.fps, mouse="drag", title="unicode3d balls", **display_options(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
