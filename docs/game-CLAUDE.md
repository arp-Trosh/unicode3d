# GAME NAME: notes for working on this game

A game for the terminal, built on [unicode3d](https://github.com/arp-Trosh/unicode3d) (the version is
pinned in `requirements.txt`). unicode3d draws the 3D scene and text and reads keys and the mouse. Everything
else (rules, state, physics, scoring, saving) is this game's own code. Read unicode3d's guide,
[Making a game](https://github.com/arp-Trosh/unicode3d/blob/main/docs/making-a-game.md), before changing
anything; its README is the reference for each part, and `python -m unicode3d.playtest play --help` works
offline.

## Layout

- `game.py`: the `Game` class (state, and `frame(screen, dt, events)`) and `main()`, which calls `run()`.
- `tests/`: unit tests, which play the game with `unicode3d.playtest.Playtest`.
- `playtest/`: pictures and text saved by playtests. It is in `.gitignore`; never commit it.

## Running and checking

You can't see a terminal, and `run()` won't start without one, so never start the game from the shell. Play
it with the playtest instead:

```sh
python -m unicode3d.playtest play game.py:Game "wait 10; shot start; press enter; hold w 20; shot walk; press q"
python -m unittest
```

After every change that affects what is on screen:

1. **Playtest** it with a script that reaches the part you changed.
2. **Look** at the result:
   - Read the report. `WARNING` and `BLANK SCREEN` lines are bugs: fix them, don't ignore them.
   - Open the `.png` shots. Without vision, read the `.txt` dumps, and play with `--glyphs ascii` to see
     the 3D picture as shading characters.
   - Use `sheet NAME N` for anything that moves.
3. **Fix** what is wrong and play again. Only call a change done when you have seen it work.

Each feature you add gets a test in `tests/` that plays it: keys in, then `expect` on the words it puts
on screen, or a check of the game's state.

## Rules

- **Keys are ints.** Use `ord("q")` and `Key.UP`, never `"q"`. Use `screen.held` (`"w" in screen.held`) for
  keys held down, such as walking or turning; use the events for presses, such as menus or firing.
- **Quitting:** `frame()` returns `False` to quit, and nothing else does.
- **Time:** `dt` is seconds; clamp it (`min(dt, 0.1)`) and move things by `speed * dt`.
- **Every frame:** call `renderer.resize(cols, rows, screen.cell_pixels)`, then `screen.erase()`, then draw,
  then `screen.refresh()` last.
- **The mouse:** `run()` in `main()` must have `mouse=True` (or `"drag"` / `"move"`) for clicks to arrive
  in a real terminal. Playtests send clicks either way.
- **Build once:** make meshes, models, renderers and `Colliders` in `__init__` or when a level loads,
  never in `frame()`.
- **Coordinates:** `screen.text(y, x)` takes the row first; `pick(x, y)` and clicks take the column first.
  Rotations are quaternions, w first (`quat_axis_angle(axis, radians)`).
- **Mesh edits:** after editing a mesh's arrays in place, call `renderer.invalidate()` (and
  `colliders.invalidate()`).
- **Engine bugs and gaps:** don't edit or monkey-patch unicode3d from here. Work around the problem in the
  game if you must, and write it down in `ENGINE-NOTES.md`: what you were doing, what went wrong or what
  was missing, and what you did instead.
