# unicode3d

A 3D renderer for the terminal, written in Python with only numpy. It draws with Unicode block
characters in 24-bit colour, falls back to 256 or 16 colours and to ASCII on terminals that need
it, and runs in Windows Terminal and in Linux/macOS terminals (no curses). It started as a Python
port, in spirit, of [ShakedAp/ASCII-renderer](https://github.com/ShakedAp/ASCII-renderer).

Used by [Zombie Dice](https://github.com/arp-Trosh/zombieDice).

## Install

Python 3.10 or later. In a game, pin a released version (a git tag) in `requirements.txt`:

```text
unicode3d @ git+https://github.com/arp-Trosh/unicode3d@v0.1.0
```

To work on the engine and a game together, install your local copy in the game's virtual
environment in editable mode, so the game always runs your latest engine code:

```sh
cd ~/Documents/Claude/someGame
python -m venv .venv && . .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e ../unicode3d                      # replaces the pinned copy with your working tree
```

## Display options

The display is detected automatically (see [Display detection](#engine-reference)); programs that
call `add_display_args` get these flags to override it:

- `--glyphs sextant|quad|half|ascii`: how finely cells are divided. `sextant` shows the most detail
  but needs the Unicode 13 "legacy computing" block symbols, so it is the default only in terminals
  known to show them: Windows Terminal (its font, Cascadia, has them), kitty, WezTerm, foot and
  Ghostty. Everywhere else the default is `quad`, which works with any font. If you see boxes or
  question marks, use `--glyphs quad`.
- `--color truecolor|256|16|mono`: colour depth.
- `--ascii`: plain characters, for terminals without Unicode.
- `--fps N`: the frame rate to aim for (default 30).

The environment variables `UNICODE3D_GLYPHS` and `UNICODE3D_COLOR` set the same things.

## Demos

```sh
python3 -m unicode3d [-n DICE] [--seed N]                # dice roll demo: space rolls, +/- dice count, q quits
python3 -m unicode3d.examples.viewer [model.obj]         # spinning model viewer (WASD/arrows, q/e spin, Esc)
python3 -m unittest                                      # the tests
```

Both demos take the display flags below, `--fps` included; the viewer shows a die when given no
model, and `--double-sided` draws back faces for meshes with inconsistent winding. A small font and
a large terminal give the best detail.

### How it draws

- **Sub-cell pixels:** each terminal cell covers a small grid of pixels: 2x2 with quadrant blocks
  (`▘▝▖▗▚▞▙▟…`), 2x3 with sextants, or 1x2 with half blocks (`▀▄`). A cell shows only two colours,
  so for each cell the renderer tries every way of splitting its pixels in two and picks the glyph
  and colour pair with the least error (the approach [chafa](https://hpjansson.org/chafa/) uses).
  Edges land on the right sub-pixel while flat areas stay solid.
- **Colour:** everything is computed in linear light and converted to sRGB at the end. Truecolor
  terminals get exact 24-bit colour. On 256- or 16-colour terminals, colours are matched in the OKLab
  colour space (so a shaded green stays green) with ordered dithering instead of banding.
- **Shading:** per-pixel Blinn-Phong lighting with interpolated vertex normals (smooth where a mesh
  shares vertices, flat where it doesn't, like cube faces), plus a white specular highlight. Light
  levels are perceived brightness, so a level of 0.5 looks half as bright.
- **Antialiasing:** 4 samples per pixel in a rotated-grid pattern, so near-vertical and
  near-horizontal edges get four coverage steps instead of two. Pixels whose samples disagree
  (silhouettes, creases, overlaps) get 8 more. Each pixel is shaded once per triangle, as GPUs do with
  multisampling, and partly covered pixels blend by coverage in linear light.
- **Textures:** mipmapped with trilinear filtering, so a 48x48 face texture a dozen pixels across
  stays steady instead of shimmering as a die turns.
- **Depth cues:** surfaces dim with distance across the scene (fog), and where one surface passes in
  front of another the far side gets a dark outline.
- **Output:** the screen is a grid of cells, and each refresh sends only the cells that changed, as
  VT escape sequences, wrapped in synchronized-output markers where the terminal supports them. There
  is no curses: on Windows the console is put in VT mode, and keys and mouse clicks are read as
  console input records.
- **Speed:** each object's triangles are rasterized together, for all sample positions at once,
  with numpy rather than one at a time. Triangles crossing the camera's near plane are clipped
  rather than dropped.

## Layout

The tests, in `tests/test_renderer.py`, use only the public API. The `examples/` programs show the
engine in use; games may build on them, as [Zombie Dice](https://github.com/arp-Trosh/zombieDice)
does with `examples/dice.py`.

| module          | role |
|-----------------|------|
| `transforms.py` | projection/view matrices, quaternions |
| `mesh.py`       | `Mesh` with vertex normals and cached mipmaps, OBJ loader, textured `make_box` |
| `texture.py`    | mipmap chains and trilinear sampling |
| `raster.py`     | `FrameBuffer` (linear RGB premultiplied by coverage, alpha, depth, object ids), vectorized multi-sample z-buffered rasterizer, perspective-correct interpolation, near-plane clipping |
| `scene.py`      | `Camera`, `Light`, `Object3D`, `Renderer` (transform, cull, lighting, multisampling with extra edge samples, fog, outlines) |
| `color.py`      | sRGB/linear conversion, named `Color`s, OKLab palette matching, dithering, SGR colour codes |
| `glyphs.py`     | glyph sets (half, quad, sextant, ascii) and matching pixels to cells |
| `keys.py`       | `Key` codes, `MouseEvent`, the VT input decoder |
| `console.py`    | raw terminal I/O for POSIX (termios) and Windows (console API), `WindowsInput` (console key and mouse records to VT sequences), colour and glyph detection |
| `terminal.py`   | `Screen` (cell grid, text, frames, diffed output), `run`, command-line display flags |
| `shapes.py`     | mesh builders: `text_mesh` (extruded text in any bitmap font), `bitmap_mesh`, `blob_mesh` (ellipsoid), `block_mesh`, `pillow_mesh` (a 2D shape puffed into a cushion), `merge_meshes` |
| `examples/dice.py`   | pip-textured die, `orientation_showing`, `top_face`, `RollAnimation` (result chosen first, then animated to land on it); Zombie Dice builds its dice on it |
| `examples/demo.py`   | the dice roll demo (`python3 -m unicode3d`) |
| `examples/viewer.py` | the model viewer |

### Using it in a game

```python
from unicode3d import Camera, Color, Key, Light, Object3D, Renderer, make_box, run

cube = Object3D(make_box(), color=Color.CYAN)   # or color=(r, g, b)
renderer, camera, light = Renderer(1, 1), Camera(), Light()

def frame(screen, dt, keys):
    if ord("q") in keys or Key.ESC in keys:
        return False
    rows, cols = screen.size()
    renderer.resize(cols, rows - 1, screen.cell_pixels)  # cell_pixels depends on the glyph set
    screen.erase()
    screen.draw_frame(renderer.render([cube], camera, light))
    screen.text(rows - 1, 0, "q quits", Color.YELLOW)
    screen.refresh()

run(frame, fps=30)
```

Textures (for `make_box`, or any mesh with `uvs` and `materials`) are 2D arrays of brightness
multipliers or `(H, W, 3)` colour arrays, both 0..1 in sRGB, where 1.0 leaves the object's colour
unchanged and 0.0 is black.

### Engine reference

**Display detection.** `Screen` picks a glyph set and colour depth unless told otherwise (arguments,
the `--glyphs`/`--color` flags from `add_display_args`, or `UNICODE3D_GLYPHS`/`UNICODE3D_COLOR`):

| setting | chosen when |
|---------|-------------|
| `truecolor` | `COLORTERM=truecolor` or `24bit`, Windows Terminal (`WT_SESSION`), any Windows console, or a known truecolor terminal (kitty, WezTerm, Alacritty, foot, Ghostty, iTerm2, VS Code, ...) |
| `256` | `TERM` contains `256` |
| `mono` | `NO_COLOR` is set or `TERM=dumb` (glyphs then default to `ascii`, which still shows shading) |
| `16` | anything else |
| `sextant` glyphs | a terminal known to show sextants whatever the font: Windows Terminal (`WT_SESSION`), kitty, WezTerm, foot or Ghostty (by `TERM`, `TERM_PROGRAM` or their own variables) |
| `quad` glyphs | any other terminal with a UTF-8 locale (always on Windows) |
| `half` glyphs | the Linux text console (`TERM=linux`), whose fonts lack quadrants |
| `ascii` glyphs | no UTF-8 |

A program can't ask a terminal which characters its font has (a missing one still takes a cell,
drawn as a box), so sextants are picked by terminal, never by guessing at fonts.
`console.shows_sextants()` holds the list.

**`Renderer(width, height, cell_pixels=(1, 2), ...)` options:**

| option | default | effect |
|--------|---------|--------|
| `cell_pixels` | `(1, 2)` | pixels per cell; pass `screen.cell_pixels` to `resize()` every frame, since it follows the glyph set |
| `cell_aspect` | `0.5` | a cell's width divided by its height |
| `samples` | `4` | samples per pixel: 1, 4, 8 or 16 |
| `edge_samples` | `8` | extra samples in pixels whose samples disagree: 0, 4, 8 or 16 |
| `fog` | `0.3` | how much the farthest surfaces are dimmed |
| `outline` | `0.55` | how much the far side of a depth edge is darkened |
| `lod_bias` | `-0.5` | added to texture mip levels: lower is sharper, higher is softer |

`render()` returns the renderer's own `FrameBuffer`, which the next render reuses; `copy()` it to
cache a frame. `fb.ids` tells you which object (its index in the render
list plus one) covers each pixel, which is handy for mouse picking. `renderer.project(point)` gives
the cell a world point landed on, for placing text labels. An unchanged scene isn't drawn again:
`render()` hands back the last frame, and `renderer.draws` counts only the renders that rasterized.

**Shapes.** `unicode3d.shapes` builds meshes to use with `Object3D`: `text_mesh(text, font)` extrudes
text in a bitmap font you supply (`{char: ["#..#", ...]}`, every glyph the same height) and returns
`(mesh, width)`; `bitmap_mesh(cells)` does the same for any boolean grid; `blob_mesh(radii, center)`
is an ellipsoid with an optional bump function; `block_mesh(center, size, rotation)` a box;
`pillow_mesh(shape)` puffs a 2D inside/outside function into a cushion with texture coordinates
that line up with the shape; and `merge_meshes(meshes)` joins them into one.

**Colours and light.** `Object3D.color` takes a named `Color` or an `(r, g, b)` triple (0..255 ints
or 0..1 floats). `Light` levels (`ambient`, `diffuse`, `specular`) are perceived brightness from 0
to 1; the highlight is always white, so lower `specular` for large flat faces that would otherwise
wash out.

**`Screen` and `run()`.** `run(frame_fn, fps=30, glyphs=None, color=None, mouse=False,
background=None, title=None)` takes over the terminal (naming its window `title`, if given) and
restores it however the loop ends (Ctrl-C raises `KeyboardInterrupt`). `frame_fn(screen, dt, keys)` returns `False` to stop. `keys` holds ints (a
character's code, or a `Key` such as `Key.UP`, `Key.ENTER`, `Key.ESC`) and, with `mouse=True`,
`MouseEvent(x, y, button, pressed)` values. `screen.text()` draws in the terminal's own ANSI colours,
so text follows the user's theme. Characters that aren't exactly one cell wide are shown as `?`.
`background=(r, g, b)` fills the screen with a known colour so anti-aliased edges blend into it
exactly; by default, edges blend toward black over the terminal's own background.
`Screen(size=(rows, cols))` with no console gives an off-screen grid for tests;
`render_updates()` returns the escape sequences a refresh would send. `screen.set_glyphs(name)` and
`screen.set_color(mode)` switch modes while running (`screen.glyph_modes` lists the glyph sets the
terminal can take), and `screen.fps` is the target frame rate, which `run()` re-reads every frame;
`screen.measured_fps` is the rate it achieved over the last second.

**Performance.** Rendering a frame and building its screen update takes about 6-7 ms for a
60x15-cell view of three rolling dice and 10-12 ms for an 80x16 view of extruded voxel text, from half blocks to
sextants. Cost grows with the pixel count, so large views in `sextant` mode are the most expensive:
a 150x45 view of three rolling dice takes about 26 ms, against 20 ms in `quad` and 13 ms in `half`. A scene that hasn't changed since the last `render()` (same objects,
poses, camera, light and size) isn't drawn again, so still frames cost a few milliseconds; after
editing a mesh's arrays in place, call `renderer.invalidate()`. Use `quad` if frames drop.

**Windows.** Needs Windows 10 or later (for VT sequences in the console) and only numpy. Windows
Terminal is recommended, and gets sextants by default; the classic console works too, with quadrants
(its default fonts may lack sextants).

## Versions

Releases are git tags (`v0.1.0`, ...) following [semantic versioning](https://semver.org): a patch
release (`0.1.1`) fixes bugs, a minor release (`0.2.0`) adds features, and before 1.0 a minor release
may also change the API. `unicode3d.__version__` holds the version. To release: bump
`__version__` in `unicode3d/__init__.py`, commit, then `git tag v0.2.0 && git push --tags`.

## License

Copyright (C) 2026 arp-Trosh.

unicode3d is free software: you can redistribute it and/or modify it under the terms of the GNU
Lesser General Public License as published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version ([`COPYING.LESSER`](COPYING.LESSER), which builds on
the GNU General Public License in [`COPYING`](COPYING)). It is distributed in the hope that it will be
useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
A PARTICULAR PURPOSE.

In short: any program, open or closed, may use unicode3d. If you distribute a modified unicode3d,
your changes to it must be released under the LGPL too. A program that ships unicode3d should
include these two license files and say that it uses unicode3d, and must let its users swap in their
own version of unicode3d (with Python source files, that's automatic).
