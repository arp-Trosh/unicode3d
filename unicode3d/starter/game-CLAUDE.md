# GAME NAME: notes for working on this game

A game for the terminal, built on [unicode3d](https://github.com/arp-Trosh/unicode3d) {{version}} (pinned in
`requirements.txt`). unicode3d draws the 3D scene and text and reads keys and the mouse. Everything else
(rules, state, physics, scoring, saving) is this game's own code. Read unicode3d's guide,
[docs/making-a-game.md](docs/making-a-game.md) (a copy for this version), before changing anything; its
[README](https://github.com/arp-Trosh/unicode3d/blob/v{{version}}/README.md) is the reference for each part,
and `python -m unicode3d.playtest play --help` works offline.

## Layout

- `game.py`: `Game` (what the screens share, and the current screen), a class for each screen (`Title`,
  `Play`, `GameOver`), each with a `frame(screen, dt, events)` that returns the screen to show next, and
  `main()`, which calls `run()`. It began as unicode3d's starter (`python -m unicode3d.new`): rename,
  replace and add to anything in it.
- `tests/`: unit tests, which play the game with `unicode3d.playtest.Playtest`.
- `docs/making-a-game.md`: the guide. After upgrading unicode3d, `python -m unicode3d.new --guide .`
  replaces it with the new version's.
- `playtest/`: pictures and text saved by playtests. It is in `.gitignore`; never commit it.

## Running and checking

You can't see a terminal, and `run()` won't start without one, so never start the game from the shell. Play
it with the playtest instead:

```sh
python -m unicode3d.playtest play game.py:Game "wait 10; shot title; press enter; hold right 20; shot play; press q"
python -m unittest
```

After every change that affects what is on screen:

1. **Playtest** it with a script that reaches the part you changed.
2. **Look** at the result:
   - Read the report. `WARNING`, `BLANK SCREEN`, `DARK SCENE` and `EMPTY SCENE` lines are bugs: fix them,
     don't ignore them.
   - Open the `.png` shots. Without vision, read the `.txt` dumps, play with `--glyphs ascii` to see
     the 3D picture as shading characters, and tune lights by the brightness the report gives each shot.
   - Use `sheet NAME N` for anything that moves.
3. **Fix** what is wrong and play again. Only call a change done when you have seen it work.
4. **Hand over** what a playtest can't judge. It plays perfectly: exact taps, a fixed `dt`, a terminal
   that reports key releases. Play with `--no-releases` and `--jitter 0.5` too, and when you report a change
   as done, list under "For a person to check" what only playing by hand shows (how far a tap turns, how
   fast things move, whether it's fun).

Each feature you add gets a test in `tests/` that plays it: keys in, then `expect` on the words it puts
on screen, or a check of the game's state.

## Rules

- **Keys** equal their characters (`ev == "q"`, `ev in ("q", Key.ESC)`), but not as set members or dict
  keys: look up `chr(ev)` in a dict of strings. Use `screen.held` (`"w" in screen.held`) for keys held down,
  such as walking or turning; use the events for presses, such as menus or firing.
- **Quitting:** `run()` stops when `Game.frame()` returns `False` (which it does when a screen's `frame()`
  returns `None`), and only then.
- **Time:** `dt` is seconds (at most 0.1 under `run()`); move things by `speed * dt`.
- **`print()`** output shows only after the game ends; in a playtest it shows at once.
- **Every frame:** draw the scene with `screen.draw(renderer, objects, camera, lights)` (`rows=-1` leaves
  the last row for a status line), then write text over it. `run()` erases the screen before each frame and
  sends what was drawn after it.
- **The mouse:** the `Game` class declares `mouse = True` (or `"drag"` / `"move"`), which `run(Game())` and
  the playtests both read; a playtest click the game didn't ask for is an error.
- **Build once:** make meshes, models, renderers and `Colliders` in `__init__` or when a level loads,
  never in `frame()`.
- **Coordinates:** `screen.text(y, x)` takes the row first; clicks take the column first. Pass the click
  itself to `renderer.pick(ev)` and `renderer.ray(ev)`, which know where the frame was drawn.
  Rotations are quaternions, w first (`quat_axis_angle(axis, radians)`).
- **Mesh edits:** a mesh's arrays are read-only once drawn; change one through `mesh.edit("vertices")`
  (a writable copy the mesh then uses), or give the mesh a new array.
- **Engine bugs and gaps:** don't edit or monkey-patch unicode3d from here. Work around the problem in the
  game if you must, and write it down in `ENGINE-NOTES.md`: what you were doing, what went wrong or what
  was missing, and what you did instead.
