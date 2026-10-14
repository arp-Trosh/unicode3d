# Making a game with unicode3d

This is the guide for building a game on unicode3d, written for AI agents first and people second. It
covers the shape of a program, the loop for checking your work without a terminal, the conventions
that are easy to get backwards, an API cheat sheet and a list of gotchas. The
[README](../../README.md) has the full reference for each part.

unicode3d is a graphics engine with terminal input and a few UI widgets. It draws 3D scenes and text
in a terminal, and reads keys and the mouse. It has no game logic: rules, turns, physics, scoring,
saving and sound are up to your game. It does give you ray and overlap queries (`Colliders`) for
walls, the ground and clicking the world, and animation tracks for moving things smoothly.

**Read this first.** Keys reach your game as **ints**, never strings. `ord("q")` is the q key and
`Key.UP` the up arrow, so `if key == "q"` is never true, and nothing says so. Write
`if key == ord("q")`, or `if ev in (ord("q"), Key.ESC)`. (`"w" in screen.held`, for held keys,
does take a string.) The [gotchas](#gotchas) list the rest.

## Setting up

Python 3.10 or later. Install unicode3d in a virtual environment in the game's folder, then let it
write a starter game there:

```sh
mkdir mygame && cd mygame
python -m venv .venv && . .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install "unicode3d @ git+https://github.com/arp-Trosh/unicode3d@v0.25.1"
python -m unicode3d.new .
python -m unittest                              # the starter's tests: they play it off-screen
```

The first run compiles the renderer, about 10 seconds; later runs start at once. Use v0.24.0 or
later: it has the playtest tool and the starter this guide relies on. `python -m unicode3d.new`
never overwrites a file, and writes:

- `game.py`: a small working game to grow, with title, play and game-over screens, a camera that
  follows the player, held keys, buttons, a label in the scene and a status line. Adapting it is
  quicker and safer than starting from nothing.
- `tests/test_game.py`: tests that play it with the playtest, one for each feature.
- `CLAUDE.md`: for an agent working on the game: how to run, check and test it, and the rules that
  matter most. ([game-CLAUDE.md](game-CLAUDE.md) is its template.)
- `docs/making-a-game.md`: this guide, for the installed version. After upgrading unicode3d,
  `python -m unicode3d.new --guide .` replaces it.
- `requirements.txt`, with unicode3d pinned to its release tag
  (`unicode3d @ git+https://github.com/arp-Trosh/unicode3d@v0.25.1`), for `pip install -r
  requirements.txt` on another machine.
- `.gitignore`, leaving out `.venv/`, `__pycache__/` and `playtest/` (the playtests' pictures).

## The shape of a game

A game is a class holding its state, with a `frame(screen, dt, events)` method. `run()` takes over
the terminal and calls it about 30 times a second. Here is a whole game, a box on a floor that the
arrow keys turn:

```python
"""A box on a floor: arrows turn it, space changes its colour, q quits."""
import argparse

from unicode3d import (Camera, Color, Key, Light, Object3D, Renderer, add_display_args, display_options, make_box,
                       run)
from unicode3d.transforms import UP, quat_axis_angle

COLOURS = [Color.CYAN, Color.MAGENTA, Color.YELLOW]
TURN_SPEED = 2.0  # radians a second


class Game:
    def __init__(self):
        self.renderer = Renderer(1, 1)          # sized in frame(), once the terminal's size is known
        self.camera = Camera(position=(0.0, 2.0, 4.0), target=(0.0, 0.0, 0.0))
        self.light = Light(direction=(0.4, -1.0, -0.6), shadows=True)
        self.floor = Object3D(make_box(), position=(0.0, -0.6, 0.0), scale=(6.0, 0.2, 6.0), color=(90, 90, 90))
        self.box = Object3D(make_box(), color=COLOURS[0])
        self.heading = 0.0
        self.colour = 0

    def frame(self, screen, dt, events):
        dt = min(dt, 0.1)                       # a pause (a dragged window) mustn't jump the game ahead
        for ev in events:
            if ev in (ord("q"), Key.ESC):       # keys are ints: ord("q"), never "q"
                return False                    # stops run()
            if ev == ord(" "):
                self.colour = (self.colour + 1) % len(COLOURS)
                self.box.color = COLOURS[self.colour]
        if Key.LEFT in screen.held:             # held keys: smooth turning, not the terminal's key repeat
            self.heading += TURN_SPEED * dt
        if Key.RIGHT in screen.held:
            self.heading -= TURN_SPEED * dt
        self.box.rotation = quat_axis_angle(UP, self.heading)

        rows, cols = screen.size()
        self.renderer.resize(cols, rows - 1, screen.cell_pixels)   # every frame: the terminal can change size
        screen.erase()
        screen.draw_frame(self.renderer.render([self.floor, self.box], self.camera, self.light))
        screen.text(rows - 1, 0, f"heading {self.heading:+.2f}   [left/right] turn  [space] colour  [q] quit",
                    Color.YELLOW)
        screen.refresh()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fps", type=int, default=30)
    add_display_args(parser)                    # --glyphs, --color, --ascii
    args = parser.parse_args()
    try:
        run(Game().frame, args.fps, title="box", **display_options(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
```

What each part is for:

- **`__init__` builds everything once:** the renderer, meshes, objects, lights, loaded models,
  widgets. Making meshes or loading models inside `frame()` costs time every frame.
- **`frame()` does three things in order.** It reads the events, moves the game on by `dt` seconds,
  then draws: `resize`, `erase`, `draw_frame`, text, and `refresh` last. Returning `False` quits;
  returning anything else, or nothing, carries on.
- **`events`** holds key presses (ints), `MouseEvent`s (only if `run()` was given `mouse=True`), and
  `KeyRelease`s (only with `key_release=True`). Use the events for things that happen once, such as
  menus, firing and toggles. Use `screen.held` for things that last while a key is down, such as
  walking and turning.
- **The screen is a grid of cells that keeps what was written to it.** `erase()` clears it, and
  `draw_frame()` puts the 3D picture on it (at `top`, `left` if given). `text()` writes over it, and
  `refresh()` sends the changes to the terminal. Without `refresh()` the terminal shows nothing.
- **`main()` runs only from the command line.** The `if __name__ == "__main__"` guard lets tests and
  the playtest import the file without starting the game.

Bigger games split this up as you would any program: a class per screen (title, play, game over)
with its own `frame()`, and the game's `frame()` passing on to the current one. The starter's
`game.py` is built that way. The demos in
`unicode3d/examples/` are games of this shape. `workshop.py` and `tactics.py` are good ones to read.

## The playtest loop: change, playtest, look, fix

An agent can't see a terminal, and `run()` refuses to start without one. The playtest plays your
`frame()` off-screen instead. It sends keys, clicks and held keys, then saves what a terminal would
have shown, as pictures and as text, and reports frame times, warnings and blank screens. After
every change that affects the screen:

1. **Playtest:** play the game with a script of inputs.
2. **Look:** read the report, look at the pictures (or the text dumps), and check the words on the
   screen.
3. **Fix**, and play again.

### From the command line

```sh
python -m unicode3d.playtest play game.py:Game "wait 5; shot start; hold left 30; shot turned; press q"
```

The target is `file.py:Name` or `package.module:Name`. Name is a class (made with no arguments), an
object with `frame()`, or a frame function; arguments go in as Python literals, as in
`game.py:Game(seed=1)`. The script is commands separated by `;` or new lines (`-f script.txt` reads
them from a file):

| command | does |
|---------|------|
| `wait [N]` | N frames (default 1) with no new input |
| `press KEY...` | one frame with the keys pressed, let go with the next frame |
| `type TEXT` | each character pressed in a frame of its own |
| `hold KEY... N` | the keys held for N frames (repeating as a terminal repeats them), then a frame letting go |
| `down KEY...` / `up [KEY...]` | press and keep holding / let go |
| `click X Y [BUTTON]` | a click on column X, row Y (from 0); `left`, `middle`, `right`, `wheel_up`, `wheel_down` |
| `click TEXT` | a click on the middle of TEXT where it is on the screen (a button's label) |
| `move X Y` | the mouse moved to column X, row Y |
| `resize COLSxROWS` | the terminal resized |
| `glyphs MODE` / `color MODE` | switch the glyph set / colour mode |
| `shot NAME` | save the screen as `playtest/NAME.png` and `playtest/NAME.txt` |
| `sheet NAME N [SHOTS]` | play N frames and save SHOTS pictures (default 6) through them as one contact sheet: for motion and animation |
| `expect TEXT` / `expect_not TEXT` | stop with an error unless TEXT is (is not) on the screen |
| `dump [words]` | print the screen as text (`words`: without the picture's block characters) |

Keys are a character (`a`, `A`, `+`) or a name: `enter`, `esc`, `space`, `tab`, `backspace`, `up`,
`down`, `left`, `right`, `home`, `end`, `page_up`, `page_down`, `delete`, `f1` to `f12`,
`ctrl+c`, and so on.

Options: `--size 120x40` (columns x rows; the default), `--glyphs sextant|quad|half|ascii`,
`--color truecolor|256|16|mono`, `--dt 0.0333` (seconds per frame), `--no-unicode`,
`--no-releases` (a terminal that reports no key releases, as many don't: held keys are then
guessed from repeats), `--key-release`, `--no-check-edits` and `--out DIR`. The screen at the end is
always saved as `last.png` and `last.txt`.

The report it prints at the end looks like this:

```text
38 frames (1.27 s of play) at 80x24 cells, sextant glyphs, truecolor
frame time: mean 1.6 ms, 95% 2.4 ms, slowest 4.0 ms (frame 34); first frame 5 ms
output: 1.9 KB a frame on average, 2.9 KB at most
quit: frame() returned False at frame 38
saved playtest/start.png
...
```

**What frame times to expect.** On a recent desktop, the example game below takes about 2 ms a frame
at 80x24, and a first-person level with a lantern (the [recipe](#a-first-person-level-from-a-text-map))
about 2.5 ms at 160x48. Point lights casting shadows cost the most: a few milliseconds each. A laptop
takes two to three times as long. Under about 30 ms a frame plays smoothly; if the report's mean is
far above that, find out what costs it before adding more.

Lines starting `BLANK SCREEN` or `WARNING` are problems to fix. A warning names the line in your
game that caused it. If `frame()` raises, the playtest stops, prints the traceback and saves the
screen as it was (`crash-frame-N.png`). The exit status is 0 if the script finished, 1 if the game
raised or an `expect` failed, and 2 for a bad script or target.

**Looking at the results.** With vision, open the `.png` files: they show exactly what the terminal
would. Without vision, read the `.txt` files and `dump words` for the text on screen. Play with
`--glyphs ascii` to get the 3D picture as shading characters (`.:-=+*#%@`, dark to bright; a space is nothing drawn):
enough to see where things are and whether anything is drawn.

**Models.** `python -m unicode3d.playtest preview models/goblin.glb` saves a contact sheet of each
of a model's animations, and prints its clips with their lengths, its size and any warnings from
loading it. Use it to check a model before putting it in the game.

### From Python, as tests

The same tool as a class, for the game's unit tests:

```python
import tempfile
import unittest

from unicode3d.playtest import Playtest

from game import Game


class GameTests(unittest.TestCase):
    def play(self):
        out = tempfile.TemporaryDirectory()
        self.addCleanup(out.cleanup)
        return Playtest(Game(), size=(24, 80), out=out.name)   # size as (rows, cols)

    def test_arrows_turn_the_box(self):
        test = self.play()
        test.wait(5)
        test.hold("left", frames=30)       # 30 frames held at 1/30 s each: one second
        test.expect("heading +2.00")

    def test_q_quits(self):
        test = self.play()
        test.press("q")
        self.assertTrue(test.quit)

    def test_plays_cleanly(self):
        test = self.play()
        test.wait(10)
        report = test.report()
        self.assertEqual(report.warnings, [])
        self.assertEqual(report.blank_frames, [])
```

`Playtest(game, size=(40, 120), glyphs="sextant", color="truecolor", dt=1/30, releases=True,
key_release=False, out="playtest")` has the same commands as methods: `wait(n)`, `press(*keys)`,
`type(text)`, `hold(*keys, frames=n)`, `down`/`up`, `click(x, y)`, `click_text(text)`,
`move(x, y)`, `resize(rows, cols)`, `step(events)` for events of your own, `shot(name)` and
`sheet(name, frames, shots=6, columns=3)`. To look at the screen, it has `text(picture=False)`,
`find(text)` (a `Found(row, col)` or `None`), `expect(text)`, `expect_not(text)` and `picture()` (an
RGB array). `quit` says whether `frame()` returned `False`. `report()` gives a `Report` with
`frames`, `seconds`, `mean_ms`, `p95_ms`, `max_ms`, `first_ms`, `slowest_frame`, `mean_bytes`,
`max_bytes`, `blank_frames` (a list of frame numbers), `warnings` (a list of `(frame, message)`),
`shots` and `quit_at`; `print(report)` shows it as the command line does. Keep the `Game()` you passed in, and a test can check its state too.

A playtest is the same every time: `dt` is fixed, and so are the glyphs and colours. It plays
every frame without waiting, so ten seconds of play take as long as drawing 300 frames. It also
checks for mistakes that real runs can't afford to check: a mesh's arrays edited in place without
`invalidate()` is reported as a warning.

## Conventions

| what | convention |
|------|------------|
| the world | y is up, right-handed: with the default camera at (0, 0, 5) looking at the origin, x is right and +z towards the viewer. Units are yours (metres suit most games). |
| positions, directions, scales | three numbers `(x, y, z)`, as a tuple or a numpy array; `scale` may be one number |
| rotations | quaternions, **w first**: `(w, x, y, z)`. Make them with `quat_axis_angle(axis, radians)`; combine with `quat_mul(a, b)` (b first, then a); blend with `quat_slerp(a, b, t)`. Never three angles. |
| angles | radians (`Camera.fov` is the exception: degrees) |
| `Light(direction=...)` | the way the light travels, so a sun overhead is `(0, -1, 0)` |
| colours | a named `Color` (`WHITE`, `RED`, `GREEN`, `YELLOW`, `BLUE`, `MAGENTA`, `CYAN`; `DEFAULT` for text in the terminal's own colour) or `(r, g, b)`, 0..255 ints or 0..1 floats |
| meshes | a face's corners go **counter-clockwise seen from the side that shows**; `uvs` is `(F, 3, 2)`, a `(u, v)` for each corner of each face; `materials` `(F,)` picks each face's texture from `textures` |
| textures | 0..1 arrays, `(H, W)` brightness or `(H, W, 3)` colour (`(H, W, 4)` with opacity); `v` runs from the image's bottom (0) to its top (1) |
| time | `dt` is in seconds; speeds are units a second, turns radians a second |
| screen size | `screen.size()` gives **(rows, cols)**; `renderer.resize(cols, rows, screen.cell_pixels)` takes columns first |
| writing to the screen | `screen.text(y, x, ...)`, `screen.draw_frame(fb, top, left)` and `screen.bar(y, x, ...)` take the **row first** |
| clicks and picking | `MouseEvent.x` is the column, `.y` the row; `renderer.pick(x, y)` and `renderer.ray(x, y)` take a cell relative to where the frame was drawn, so subtract `top` and `left` |
| playtest sizes | `Playtest(size=(rows, cols))` and `resize(rows, cols)` in Python; `--size COLSxROWS` and `resize COLSxROWS` in scripts and the command line; `click X Y` column first everywhere |

## A first-person level from a text map

Dungeon crawlers, mazes and shooters start the same way: a level drawn as text, walls you can't walk
through, and a camera at eye height. This is a whole one to start from (`python -m unicode3d.playtest
play crawl.py:Crawl "hold w 30; shot ahead"`):

```python
"""A first-person walk through a level made from a text map: W/S walk, A/D step aside, left/right turn, q quits."""
import math

import numpy as np

from unicode3d import Camera, Colliders, Color, Key, Mesh, Object3D, PointLight, Renderer, make_box

LEVEL = [
    "##########",
    "#....#...#",
    "#.##.#.#.#",
    "#.#....#.#",
    "#.#.####.#",
    "#.......@#",
    "##########",
]
WALK, TURN = 2.5, 2.0       # units a second, radians a second
RADIUS, EYE = 0.4, 0.5      # the player's width (for walls) and eye height; a cell is 1 across, walls 1 high


def bricks(size=32):
    """A brick pattern as a brightness texture, 0..1, row 0 at the top."""
    y, x = np.mgrid[0:size, 0:size]
    mortar = (y % 8 == 0) | ((x + 8 * (y // 8 % 2)) % 16 == 0)
    return np.where(mortar, 0.45, 1.0)


def wall_mesh(level, height=1.0):
    """The walls of a text map as one mesh. Cell (row r, column c) is the square around x = c, z = r. Only the
    sides of '#' cells that face open floor are made: one mesh with nothing hidden draws fastest."""
    def solid(r, c):
        return not (0 <= r < len(level) and 0 <= c < len(level[r])) or level[r][c] == "#"

    up = np.array([0.0, height, 0.0])
    vertices, uvs = [], []
    for r, row in enumerate(level):
        for c, cell in enumerate(row):
            if cell != "#":
                continue
            for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                if solid(r + dr, c + dc):
                    continue
                out = np.array([dc, 0.0, dr])                     # from the wall towards the floor
                half = np.cross(-out, (0.0, 1.0, 0.0)) / 2        # half a cell to the right, seen from the floor
                left, right = np.array([c, 0.0, r]) + out / 2 - half, np.array([c, 0.0, r]) + out / 2 + half
                # Two triangles, counter-clockwise seen from the floor (the side that shows); each corner's
                # (u, v) runs left to right and bottom (0) to top (1) of the texture.
                vertices += [left, right, right + up, left, right + up, left + up]
                uvs += [[(0, 0), (1, 0), (1, 1)], [(0, 0), (1, 1), (0, 1)]]
    faces = np.arange(len(vertices)).reshape(-1, 3)
    return Mesh(np.array(vertices), faces, uvs=np.array(uvs, dtype=float), materials=np.zeros(len(faces), int),
                textures=[bricks()])


class Crawl:
    def __init__(self):
        self.renderer = Renderer(1, 1)
        rows, cols = len(LEVEL), len(LEVEL[0])
        self.walls = Object3D(wall_mesh(LEVEL), color=(210, 180, 150))
        self.floor = Object3D(make_box(), position=((cols - 1) / 2, -0.05, (rows - 1) / 2), scale=(cols, 0.1, rows),
                              color=(90, 80, 70))
        self.solid = Colliders([self.walls])                  # the same mesh, for walking into
        self.z, self.x = next((r, row.index("@")) for r, row in enumerate(LEVEL) if "@" in row)
        self.yaw = 0.0                                        # 0 looks along -z, the map's top
        self.camera = Camera(near=0.05)
        # Held in the camera's own space (x right, y up, -z ahead), so it moves and turns with the view.
        self.sword = Object3D(make_box(), position=(0.18, -0.13, -0.28), scale=(0.03, 0.03, 0.25),
                              color=(200, 200, 215), parent=self.camera)
        self.lantern = PointLight(color=(255, 190, 120), range=5.0, ambient=0.15)
        self.time = 0.0

    def frame(self, screen, dt, events):
        dt = min(dt, 0.1)
        self.time += dt
        if any(ev in (ord("q"), Key.ESC) for ev in events):
            return False
        held = screen.held
        self.yaw += TURN * dt * ((Key.LEFT in held) - (Key.RIGHT in held))
        ahead = np.array([-math.sin(self.yaw), 0.0, -math.cos(self.yaw)])
        right = np.array([-ahead[2], 0.0, ahead[0]])
        step = ahead * (("w" in held) - ("s" in held)) + right * (("d" in held) - ("a" in held))
        eye = np.array([self.x, EYE, self.z]) + step * WALK * dt
        eye += self.solid.push_out(eye, RADIUS)               # out of any wall walked into: it slides along
        self.x, self.z = eye[0], eye[2]

        self.camera.position, self.camera.target = eye, eye + ahead
        self.lantern.position = eye + (0.0, 0.3, 0.0)
        self.lantern.diffuse = 0.8 + 0.1 * math.sin(self.time * 13.0) * math.sin(self.time * 7.0)  # a flicker
        rows, cols = screen.size()
        self.renderer.resize(cols, rows - 1, screen.cell_pixels)
        screen.erase()
        screen.draw_frame(self.renderer.render([self.walls, self.floor, self.sword], self.camera, [self.lantern]))
        screen.text(rows - 1, 0, f"at {self.x:.2f}, {self.z:.2f}   [WASD] walk  [left/right] turn  [q] quit",
                    Color.YELLOW)
        screen.refresh()
```

How it works, and what to keep when you change it:

- **One mesh for all the walls.** Each object costs time every frame, so a level of hundreds of
  wall boxes is slow; one mesh of only the sides that face the floor draws in a fraction of that.
  Doors, pickups and enemies, which move or vanish, are objects of their own.
- **Faces show from one side.** Corners go counter-clockwise seen from the side that shows (here,
  from the floor). A wall that comes out invisible has its corners the other way round.
- **Textures:** `uvs` gives each corner of each face its `(u, v)`: `u` 0 to 1 left to right across
  the image, `v` 0 to 1 from its **bottom** to its top. `materials` gives each face's index in
  `textures`. Brightness textures (2D arrays) tint with the object's colour.
- **Walls block by the same mesh:** `Colliders([walls])` and `push_out(eye, RADIUS)` each frame
  after moving. Pushing out of the wall you walked into, rather than refusing the move, is what lets
  the player slide along it.
- **Held things go on the camera:** an object whose `parent` is the camera stays put in the view
  (x right, y up, -z ahead), as the sword does.
- **Light:** in the dark, a `PointLight` carried with the player (its `ambient` lights whatever
  faces away from it) and the renderer's fog (`Renderer(fog=...)`, 0.3 by default) give depth.
  Torches are more `PointLight`s on the walls; with `shadows=True` each costs a few milliseconds a
  frame, so let only the nearest one or two cast shadows. Change a light's fields (`position`,
  `diffuse`) every frame to make it flicker.
- **Facing:** `yaw` 0 looks along -z, and turning left adds to it. The step ahead is
  `(-sin(yaw), 0, -cos(yaw))` and to the right `(cos(yaw), 0, -sin(yaw))`.

Line of sight (an enemy seeing the player) is `solid.raycast(enemy_eye, player_eye - enemy_eye,
np.linalg.norm(player_eye - enemy_eye))`: `None` (no wall in between) means it can see them.

## API cheat sheet

Everything below imports from `unicode3d` unless another module is named. The [README](../../README.md)
has each in full.

**Running.**
`run(frame_fn, fps=30, mouse=False, key_release=False, title=None, glyphs=None, color=None)`.
`frame_fn(screen, dt, events)` returns `False` to stop. Pass `mouse=True` for clicks and the wheel,
`"drag"` for moves while a button is held (sliders), and `"move"` for every move (hover);
`key_release=True` for `KeyRelease` events.
`add_display_args(parser)` and `display_options(args)` give the `--glyphs`, `--color` and `--ascii`
flags.

**Screen.**
- `screen.size()` gives `(rows, cols)`, and `screen.cell_pixels` the pixels in a cell.
- `screen.erase()` clears it; `screen.draw_frame(fb, top=0, left=0)` puts a picture on it.
- `screen.text(y, x, s, color, bold=False, reverse=False, dim=False, bg=None)` writes text.
- `screen.bar(y, x, width, fraction, color)` draws a meter.
- `screen.label(renderer, point, text, color, top=0, left=0, owner=None, clamp=False)` writes text
  over a point in the scene.
- `screen.refresh()` sends the changes. Call it last in every frame.
- `screen.held` is the keys held down: `"w" in screen.held`, `Key.LEFT in screen.held`.
- `screen.fps` is the target frame rate and `screen.measured_fps` the one achieved.

**Input.** Key presses are ints: `ord("a")` and `Key.ENTER`, `Key.ESC`, `Key.TAB`, `Key.BACK_TAB`,
`Key.BACKSPACE`, `Key.UP`, `Key.DOWN`, `Key.LEFT`, `Key.RIGHT`, `Key.HOME`, `Key.END`,
`Key.PAGE_UP`, `Key.PAGE_DOWN`, `Key.DELETE`, `Key.INSERT`, `Key.F1` to `Key.F12`. Shifted letters
come as capitals (`ord("Q")`). A click is `MouseEvent(x, y, button, pressed, moved)`, with `button`
one of `MouseEvent.LEFT`, `MIDDLE`, `RIGHT`, `WHEEL_UP`, `WHEEL_DOWN` or `NONE`. `KeyRelease(key)`
comes with `run(key_release=True)`.

**Scene.**
- `Renderer(1, 1, background=None, fog=0.3)`; then `renderer.resize(cols, rows, screen.cell_pixels)`
  and `fb = renderer.render(objects, camera, lights)` every frame.
- `Camera(position, target, up=(0, 1, 0), fov=50, near=0.1, far=100)`, or
  `projection="ortho", size=10` for isometric and top-down views.
- `Light(direction, ambient=0.3, diffuse=0.7, shadows=False)` is the sun.
  `PointLight(position, color, range=10, shadows=False)` is a lamp.
- `Object3D(mesh, position, rotation, scale, color, visible=True, parent=None)`. It also takes
  `emissive`, `opacity`, `reflectivity`, `specular`, `shininess`, `cast_shadows`, `billboard`,
  `outline` and `simplify`.
- `Node(position, rotation, scale, parent)` groups objects, which move with their parent. A
  `Camera` can be a parent too: its children stay in place in the view (a held weapon, a cockpit).
- `obj.world_bounds()` gives the corners `(low, high)`; `union_bounds(objects)`.

**Meshes.**
- `make_box(size=1.0, textures=None)`.
- `unicode3d.shapes` has `blob_mesh(radii)` (a sphere or an ellipsoid), `block_mesh(center, size)`,
  `text_mesh(text)` (returns `(mesh, width)`), `pillow_mesh(shape)` and
  `merge_meshes(meshes, colors)`.
- `Mesh(vertices, faces, uvs=None, materials=None, textures=[], vertex_colors=None, face_colors=None)`;
  see [Conventions](#conventions) for the order of corners and `uvs`, and the
  [recipe](#a-first-person-level-from-a-text-map) for a mesh built from a map.
- `load_image(path)`.

**Models.**
- `model = load_model("ship.glb").fit(2.0)` loads glTF or OBJ.
- Render its parts with `renderer.render([*model, ...])`, and move it all by `model.root`.
- `model.nodes["Door"]` is one part; `model.animations["Walk"]` is a `Clip`.
- `model.copy()` gives another (cheap); `model.bake()` gives a still model that draws faster.
- Check `model.warnings` after loading.

**Animation.**
- `Track([(time, value), ...], easing, loop="once"|"loop"|"pingpong")` and `RotationTrack` (for
  quaternions). `easing` is a function or its name: `"linear"` (the default), `"ease_in"`,
  `"ease_out"`, `"ease_in_out"`, `"ease_out_back"`, `"ease_out_bounce"` or `"step"`.
- `Animation(obj, position=track, rotation=track, scale=track)`; call `anim.update(dt)` every frame.
- `Clip([animations], loop=...)` plays several on one clock: `clip.update(dt)`,
  `clip.start(fade=0.2)`, `clip.done()`.
- The easing functions (`ease_in_out(t)`, ...) and `lerp(a, b, t)` import from `unicode3d`;
  `unicode3d.transforms` has `quat_slerp`.

**Queries.**
- `solid = Colliders(objects)`. Call `solid.update()` once a frame after moving things.
- `solid.raycast(origin, direction, max_distance, ignore=[...])` gives a `Hit` (`object`,
  `position`, `normal`, `distance`) or `None`.
- `solid.overlap_sphere(centre, radius)`, `overlap_capsule(a, b, radius)` and
  `overlap_box(centre, size)` give contacts.
- `solid.push_out(centre, radius)` is the move out of walls.
- `renderer.pick(x, y)` tells what the last frame drew in a cell (`.object`, `.position`).
- `renderer.ray(x, y)` gives a line of sight `(origin, direction)` for clicking the world:
  `solid.raycast(*renderer.ray(ev.x, ev.y - top))`.

**Effects.**
- `Particles(gravity, ground, face_camera, seed=None)` (`seed` an int, for the same particles each run) with `emit(point, count, velocity, speed, life, size,
  color, ...)` and `update(dt)`, put in the render list.
- `Lines(starts, ends, color, width)`.
- `SpriteSheet(picture, columns, animations={...})` and `Sprite(sheet, position)`, with
  `play(name)` and `update(dt)`.
- `Sky()`, `Gradient(top, bottom)` and `SkyBox(textures)` as `renderer.background`;
  `Fog(start, end, color)` as `renderer.fog`.

**Widgets.**
- `Button(label, action, key)`, `Toggle(label, value, key)`, `Slider(label, value, lo, hi, step)` and
  `Choice(label, options, key)` go in a `Panel([...], vertical=False)`.
- In each frame, `events = panel.handle(events)` takes what the panel uses, and
  `panel.draw(screen, y, x)` draws it.
- `DisplayControls(renderer=renderer)` is the F2 to F8 display settings bar:
  `events = controls.handle(events, screen)` and `controls.draw(screen, y, x)`.
- Run with `mouse="drag"` for sliders.

**Starter.** `python -m unicode3d.new FOLDER` writes a starter game (see [Setting up](#setting-up));
`--guide` refreshes the copy of this guide.

**Playtest.** `python -m unicode3d.playtest play file.py:Name "SCRIPT"` and `preview MODEL`;
`unicode3d.playtest.Playtest` from Python (see above). `python -m unicode3d.playtest play --help`
lists every command.

## Gotchas

Mistakes that look right and fail quietly, or fail far from their cause. The engine makes most of
its own mistakes loud (a wrong-shaped position, rotation or colour is a clear `ValueError`, an edited
animation track raises, the playtest warns of mesh edits it can't see), but these it can't:

1. **Keys are ints.** `key == "q"` is never true; write `key == ord("q")`. Only `screen.held` takes
   strings.
2. **`frame()` returning `False` quits**, and only `False`: returning `None` or `0` carries on.
   Don't return `False` by accident from a helper whose result you pass on.
3. **`dt` is in seconds**, and under `run()` it is the real time since the last frame. After a pause
   (the window dragged, a breakpoint) it can be large, so clamp it (`dt = min(dt, 0.1)`) before
   moving things by it. In a playtest it is fixed (1/30 by default).
4. **Clicks need `run(mouse=True)`** (`"drag"` for sliders, `"move"` for hover), and `KeyRelease`
   events need `run(key_release=True)` (`screen.held` doesn't: `run()` always asks for releases). The playtest sends clicks and releases to `frame()`
   whatever `run()` is told, so a game can pass its playtests and get no clicks in a terminal.
   Check what `main()` passes to `run()`.
5. **Held keys need `screen.held`, not events.** A held key arrives as a press, a half-second
   pause, then repeats, so moving a player once per key event stutters. Move by
   `speed * dt` while `"w" in screen.held`. It is exact in terminals that report releases (kitty,
   foot, Ghostty, Alacritty, iTerm2, Windows); in the others it guesses, and a tap holds a key for
   up to half a second. Playtest with `--no-releases` to see that case.
6. **Resize the renderer every frame:** `renderer.resize(cols, rows, screen.cell_pixels)`. The
   terminal can change size, and `cell_pixels` changes with the glyph set (F2). Without the resize,
   the picture stays the size the renderer was made (one cell, for `Renderer(1, 1)`); with the wrong
   `cell_pixels`, `draw_frame` raises.
7. **Erase or overwrite.** The screen keeps what was written to it, so text from an earlier frame
   stays until something covers it. `screen.erase()` at the start of drawing is simplest.
8. **Call `screen.refresh()`** at the end of every frame, or the terminal shows nothing (the
   playtest's report says so).
9. **Coordinates go in both orders.** `screen.text(y, x)` and `draw_frame(fb, top, left)` take the
   row first, while `pick(x, y)`, `MouseEvent.x/.y` and `click X Y` take the column first.
   `screen.size()` is `(rows, cols)` and `renderer.resize()` takes `(cols, rows)`. See
   [Conventions](#conventions).
10. **`pick()` and `ray()` take cells relative to the frame**, not the screen: subtract the
    `top` and `left` you drew it at (`renderer.pick(ev.x - left, ev.y - top)`).
11. **Rotations are quaternions, w first.** `rotation=(0, 45, 0)` is not 45 degrees about y: it is
    refused with an error. Use `quat_axis_angle((0, 1, 0), np.radians(45))`.
12. **A new object starts at the origin.** An `Object3D` made in the middle of play without a
    position shows for a frame at (0, 0, 0) before your code moves it. Give it its position when you
    make it, or make it with `visible=False` until it is placed.
13. **Mesh arrays edited in place need `invalidate()`.** The renderer and `Colliders` keep packed
    copies of meshes, so after changing `mesh.vertices`, `faces`, colours or a texture array in place,
    call `renderer.invalidate()` (and `colliders.invalidate()`). The playtest warns of an edit made
    without it. Moving objects needs nothing: change `obj.position`, `rotation` or `scale` freely.
14. **Animation tracks can't be edited.** A track's keyframes are read-only once it is made (an edit
    raises): give the animation a new `Track` instead. A `Clip`'s `loop` and `speed` can be changed.
15. **`Colliders` don't see skinned poses.** A skinned character is seen by queries in its bind
    pose (a set holding one warns of it). Stand an invisible box or capsule in for it, moved with it,
    in the set instead.
16. **`print()` while `run()` is going scribbles over the picture.** Put debugging output in a file,
    or look at it in a playtest, where `print()` is harmless.
17. **Don't start the game from an agent's shell.** `run()` needs a terminal and raises without one;
    play it with the playtest. Keep `run()` inside `main()` behind
    `if __name__ == "__main__":` so the playtest and tests can import the file.
18. **Build things once.** Make meshes, load models and textures, and build `Renderer`s and
    `Colliders` in `__init__` (or when a level loads), not in `frame()`. A new `Camera` or light
    each frame is fine, and so is showing, hiding or moving objects.
19. **Frozen programs and custom launchers.** The first run compiles in several processes at once.
    PyInstaller programs are handled; a launcher of your own that can't start Python processes should
    set `UNICODE3D_NO_PRECOMPILE=1`.

## Where to look next

- The [README](../../README.md): the reference for every part (each section is collapsed; the headings
  open them).
- The starter's `game.py` (`python -m unicode3d.new`), a game of the shape above with several screens.
- The demos, which are games of the shape above (`python -m unicode3d.examples.NAME`, the source in
  `unicode3d/examples/`): `workshop` (a panel of widgets, materials and animation), `tactics` (an
  orthographic board, clicking tiles with ray queries, labels), `room` (walking with collisions,
  held keys, particles, sprites, picking), `balls` (many objects, sliders), `maze` (animation
  tracks, fog), `viewer` (loading models and playing their animations) and `skinned` (skinned
  characters).
- [How unicode3d works](../../docs/how-it-works.md), for what happens inside.
