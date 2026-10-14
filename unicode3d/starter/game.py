"""GAME NAME: take the gems before the time runs out.

Arrow keys or WASD move, Esc goes back to the title, q quits. Started from unicode3d's starter
(python -m unicode3d.new); CLAUDE.md says how to work on it, and docs/making-a-game.md is the guide.

The game is three screens, Title, Play and GameOver. Each has a frame(screen, dt, events) that returns
the screen to show next: itself to stay, another to change, None to quit. Game holds what they share
and passes each frame on to the current one. run() erases the screen before each frame and sends what
was drawn after it.
"""
import argparse
import math
import random

import numpy as np

from unicode3d import (Button, Camera, Color, Gradient, Key, Light, Mesh, Object3D, Panel, Renderer, add_display_args,
                       display_options, make_box, run)
from unicode3d.shapes import blob_mesh
from unicode3d.transforms import UP, quat_axis_angle

TITLE = "GAME NAME"
ROUND_TIME = 30.0   # seconds a round lasts
ARENA = 5.0         # the floor runs from -ARENA to ARENA in x and z
SPEED = 4.0         # the player's speed, units a second
REACH = 0.8         # how near the player must come to take a gem
GEMS = 3            # gems on the floor at once
MOVES = {           # held keys and the way each moves the player (x, z)
    Key.LEFT: (-1, 0), Key.RIGHT: (1, 0), Key.UP: (0, -1), Key.DOWN: (0, 1),
    "a": (-1, 0), "d": (1, 0), "w": (0, -1), "s": (0, 1),
}


class Game:
    """What the screens share (the renderer, the light, meshes, the best score) and the current screen."""
    mouse = True                                # clicks, for the buttons: run() and the playtest both read it

    def __init__(self, seed=None):
        self.random = random.Random(seed)       # a seed makes a round the same every time (tests)
        self.renderer = Renderer(background=Gradient())   # screen.draw() sizes it to fit, every frame
        self.light = Light(direction=(0.3, -1.0, -0.5), shadows=True)
        self.box = make_box()
        stone = blob_mesh((0.3, 0.45, 0.3), rings=2, segments=6)        # a six-sided double pyramid
        corners = stone.vertices[stone.faces].reshape(-1, 3)              # each face its own corners: flat facets
        self.gem = Mesh(corners, np.arange(len(corners)).reshape(-1, 3))
        self.best = 0
        self.current = Title(self)

    def frame(self, screen, dt, events):
        showing = self.current
        self.current = self.current.frame(screen, dt, events)
        if self.current is not None and self.current is not showing:
            screen.erase()                      # a new screen draws at once, this frame, over a clean one
            self.current = self.current.frame(screen, 0.0, [])
        return self.current is not None         # False stops run()


def centred(screen, y, text, color=Color.WHITE, **style):
    """Write text in the middle of row y."""
    screen.text(y, max((screen.size().cols - len(text)) // 2, 0), text, color, **style)


class Title:
    """A gem turning, the game's name, and Play and Quit buttons (Enter and q, or a click)."""

    def __init__(self, game):
        self.game = game
        self.next = self
        self.panel = Panel([Button("Play", self.play, key=Key.ENTER), Button("Quit", self.quit, key="q")])
        self.stone = Object3D(game.gem, scale=2.0, color=Color.YELLOW, emissive=0.2)
        self.camera = Camera(position=(0.0, 0.6, 3.0), target=(0.0, 0.0, 0.0))
        self.angle = 0.0

    def play(self):
        self.next = Play(self.game)

    def quit(self):
        self.next = None

    def frame(self, screen, dt, events):
        events = self.panel.handle(events)      # takes the clicks and keys the buttons use
        if Key.ESC in events:
            self.next = None
        self.angle += 1.5 * dt
        self.stone.rotation = quat_axis_angle(UP, self.angle)

        rows, cols = screen.size()
        screen.draw(self.game.renderer, [self.stone], self.camera, self.game.light)
        centred(screen, 1, TITLE, Color.YELLOW, bold=True)
        centred(screen, 2, "take the gems before the time runs out")
        if self.game.best:
            centred(screen, 3, f"best {self.game.best}", Color.CYAN)
        self.panel.draw(screen, rows - 2, (cols - self.panel.width) // 2)
        return self.next


class Play:
    """A round: the player walks the floor taking gems until the time is up."""

    def __init__(self, game):
        self.game = game
        self.floor = Object3D(game.box, position=(0.0, -0.1, 0.0), scale=(2 * ARENA + 1, 0.2, 2 * ARENA + 1),
                              color=(70, 80, 90))
        self.player = Object3D(game.box, position=(0.0, 0.4, 0.0), scale=(0.6, 0.8, 0.6), color=Color.CYAN)
        self.gems = [Object3D(game.gem, position=self.free_spot(), color=Color.YELLOW, emissive=0.3)
                     for _ in range(GEMS)]
        self.camera = Camera(position=(0.0, 7.0, 7.0), target=(0.0, 0.0, 0.0))
        self.score = 0
        self.time_left = ROUND_TIME
        self.time = 0.0                          # seconds since the round began
        self.popups = []                         # [position, seconds left] of "+1"s over taken gems

    def free_spot(self):
        """A place for a new gem, away from the player."""
        while True:
            x, z = (self.game.random.uniform(-ARENA, ARENA) for _ in range(2))
            if math.dist((x, z), (self.player.position[0], self.player.position[2])) > 2 * REACH:
                return (x, 0.5, z)

    def frame(self, screen, dt, events):
        for ev in events:
            if ev == "q":                       # a key equals its character (or a Key: Key.ESC, Key.UP, ...)
                return None
            if ev == Key.ESC:
                return Title(self.game)

        # Move: keys held down, so walking is smooth whatever the terminal's key repeat.
        dx = sum(MOVES[k][0] for k in MOVES if k in screen.held)
        dz = sum(MOVES[k][1] for k in MOVES if k in screen.held)
        length = math.hypot(dx, dz)
        if length:
            x, y, z = self.player.position
            step = SPEED * dt / length
            self.player.position = (min(max(x + dx * step, -ARENA), ARENA), y,
                                    min(max(z + dz * step, -ARENA), ARENA))
            self.player.rotation = quat_axis_angle(UP, math.atan2(-dx, -dz))   # face the way it walks

        # Take the gems within reach; each goes somewhere new.
        self.time += dt
        px, _, pz = self.player.position
        for gem in self.gems:
            if math.dist((px, pz), (gem.position[0], gem.position[2])) < REACH:
                self.score += 1
                self.popups.append([gem.position, 0.8])
                gem.position = self.free_spot()
            gem.rotation = quat_axis_angle(UP, self.time * 2.0)
        for popup in self.popups:
            popup[1] -= dt
        self.popups = [p for p in self.popups if p[1] > 0]

        self.time_left -= dt
        if self.time_left <= 0:
            return GameOver(self.game, self.score)

        # Draw: the camera follows the player; the last row is the status line.
        self.camera.position = (px, 7.0, pz + 7.0)
        self.camera.target = (px, 0.0, pz)
        rows, cols = screen.size()
        screen.draw(self.game.renderer, [self.floor, self.player, *self.gems], self.camera, self.game.light,
                    rows=-1)                    # all but the last row
        for position, left in self.popups:
            rise = -1 - int(2 * (0.8 - left))   # rows above the spot: it floats up as it fades
            screen.label(self.game.renderer, position, "+1", Color.YELLOW, bold=True, dy=rise, hide=False)
        screen.text(rows - 1, 0, f"score {self.score}  time {math.ceil(self.time_left):2d} ", Color.WHITE)
        screen.bar(rows - 1, 18, 20, self.time_left / ROUND_TIME, Color.GREEN)
        screen.text(rows - 1, 40, "[arrows] move [esc] title [q] quit", Color.DEFAULT, dim=True)
        return self


class GameOver:
    """The score, and Again and Title buttons (Enter and t, or a click)."""

    def __init__(self, game, score):
        self.game = game
        self.score = score
        self.best = score > game.best
        game.best = max(game.best, score)
        self.next = self
        self.panel = Panel([Button("Again", self.again, key=Key.ENTER), Button("Title", self.title, key="t")])

    def again(self):
        self.next = Play(self.game)

    def title(self):
        self.next = Title(self.game)

    def frame(self, screen, dt, events):
        events = self.panel.handle(events)
        if "q" in events or Key.ESC in events:
            return None
        rows, cols = screen.size()
        centred(screen, rows // 2 - 2, "GAME OVER", Color.RED, bold=True)
        centred(screen, rows // 2, f"score {self.score}", Color.YELLOW)
        if self.best:
            centred(screen, rows // 2 + 1, "a new best!", Color.CYAN)
        self.panel.draw(screen, rows // 2 + 3, (cols - self.panel.width) // 2)
        return self.next


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fps", type=int, default=30)
    add_display_args(parser)                    # --glyphs, --color, --ascii
    args = parser.parse_args()
    try:
        run(Game(), args.fps, title=TITLE, **display_options(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
