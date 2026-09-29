# unicode3d

A 3D renderer for the terminal, written in Python with numpy and [Numba](https://numba.pydata.org). It draws with Unicode block
characters in 24-bit colour, falls back to 256 or 16 colours and to ASCII on terminals that need
it, and runs in Windows Terminal and in Linux/macOS terminals (no curses). It started as a Python
port, in spirit, of [ShakedAp/ASCII-renderer](https://github.com/ShakedAp/ASCII-renderer).

Used by [Zombie Dice](https://github.com/arp-Trosh/zombieDice).

## Install

Python 3.10 or later. In a game, pin a released version (a git tag) in `requirements.txt`:

```text
unicode3d @ git+https://github.com/arp-Trosh/unicode3d@v0.4.0
```

To work on the engine and a game together, install your local copy of the engine in the game's
virtual environment in editable mode, so the game always runs your latest engine code. Here the
game is in `~/your-game` and the engine in `~/unicode3d`; use your own paths:

```sh
cd ~/your-game
python -m venv .venv && . .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e ~/unicode3d                       # replaces the pinned copy with your working tree
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
python3 -m unicode3d.examples.balls [-n BALLS]           # balls bouncing around a room, with a settings panel
python3 -m unicode3d.examples.maze [--size N]            # the Windows 98 maze screensaver
python3 -m unicode3d.examples.room                       # walk around a courtyard of things to look at (WASD)
python3 -m unittest                                      # the tests
```

Every demo takes the display flags below, `--fps` included, and shows the display settings at the
bottom right: F2 cycles the glyph set, F3 the colours, F4 the target frame rate (shown as
achieved/target), F5 switches shadows on and off and F6 reflections, and each can be clicked too. A
small font and a large terminal give the best detail.

- **dice** (`python3 -m unicode3d`): textured dice that tumble onto a table, casting shadows, and
  land on a chosen face; g makes them glass, m polishes the table so that it mirrors them.
- **viewer**: spins an OBJ model, or a die when given none; c makes it chrome, reflecting a sky;
  `--double-sided` draws back faces for meshes with inconsistent winding.
- **balls**: up to 500 balls of many colours drift and bounce around a room seen from outside (the
  near walls are see-through, since only the insides of the walls are drawn). The panel on the left
  sets the number of balls, their size, the room's size and their speed, and switches collisions,
  gravity, glass (every third ball see-through), a mirror floor, a lamp (a point light and a glowing
  bulb) and the camera's orbit; the sun and the lamp both cast shadows, tinted through the glass. Drag the
  sliders, click the toggles, use their keys (shown in the panel), or Tab through them. Click a ball
  to make it glow (picking). Balls small on screen use meshes with fewer triangles, so hundreds stay
  fast. Shows: many objects, shadows, transparency, a mirror, point lights, emissive objects,
  picking, widgets.
- **maze**: the camera walks a random maze, keeping a hand on the right-hand wall (or taking the
  shortest way), from the blue marker to the gold one (glowing see-through gems), spins round at the
  exit and starts a new maze. The panel sets the size (3 to 40 cells a side), the speed and the fog,
  and switches the headlamp (a lantern carried beside the camera, casting shadows), the textures
  (off: flat colours per face), a polished floor that mirrors the maze, and a map. Space pauses, n
  starts a new maze. Shows: textures, a point light moving with the camera, shadows, transparency, a
  mirror, fog, per-face colours, parts of the maze out of view skipped whole.
- **room**: an engine showcase to walk around: an open courtyard, WASD to walk, Q/E or Left/Right to
  turn, Up/Down to look up and down. Inside are dice turning on a pedestal in a glass case, an
  orrery (a planet and its moon circling a glowing sun, built as a scene graph), a table with
  coloured cubes on it, a rainbow blob in vertex colours, a tree whose leaves dapple the ground with
  light, a trellis the lamp throws across the floor at night, a stained-glass panel casting coloured
  light, a mirror on the south wall (turn round at the start), a still pool, a chrome ball, a lamp,
  and a sign. The crosshair names whatever it is on; clicking names what you clicked. The sun and
  the lamp cast shadows. The panel switches the lamp and the sun, picks the background (a sky, a
  starry sky box, a gradient or none) and sets the fog. Walking is smooth in terminals that report
  key releases (see [Input](#input)); elsewhere a tap walks for about half a second. Esc quits.

### How it draws

- **Sub-cell pixels:** each terminal cell covers a small grid of pixels: 2x2 with quadrant blocks
  (`▘▝▖▗▚▞▙▟…`), 2x3 with sextants, or 1x2 with half blocks (`▀▄`). A cell shows only two colours,
  so for each cell the renderer tries every way of splitting its pixels in two and picks the glyph
  and colour pair with the least error (the approach [chafa](https://hpjansson.org/chafa/) uses).
  Edges land on the right sub-pixel while flat areas stay solid.
- **Colour output:** everything is computed in linear light and converted to sRGB at the end. Truecolor
  terminals get exact 24-bit colour. On 256- or 16-colour terminals, colours are matched in the OKLab
  colour space (so a shaded green stays green) with ordered dithering instead of banding.
- **Shading:** per-pixel Blinn-Phong lighting with interpolated vertex normals (smooth where a mesh
  shares vertices, flat where it doesn't, like cube faces), plus a specular highlight. Any number of
  lights add up: directional ones (like the sun) and point lights (lamps, torches) that fade out
  with distance, each in its own colour, and objects can glow by themselves. Light levels are
  perceived brightness, so a level of 0.5 looks half as bright.
- **Shadows:** lights can cast shadows, from shadow maps (the scene's depth as seen from the light;
  six of them, a cube, all round a point light) with edges softened by percentage-closer filtering,
  over at least a pixel on screen so they look as smooth as the edges of shapes.
- **Colour:** one colour per object, or colours per face or per vertex (blended smoothly across
  each face) in the mesh itself, and textures on top.
- **Transparency:** see-through objects (glass, water, ghosts), and see-through parts of meshes or
  textures, blend in depth order whatever order they are listed in, up to 4 layers deep in each
  pixel. They show their far side through their near one, keep their highlights, grow more opaque
  at a slant as glass does, and cast shadows tinted by their colour. Textures with holes (leaves,
  fences) are cut out with smooth edges, and cast shadows with holes in them.
- **Reflections:** flat objects can be mirrors (a wall mirror, a polished floor, still water),
  showing the scene reflected in them, lit and shadowed like the rest, mirrors in mirrors included if
  asked for. Curved shiny objects (chrome, lacquer) reflect the sky or background.
- **Antialiasing:** 4 samples per pixel in a rotated-grid pattern, so near-vertical and
  near-horizontal edges get four coverage steps instead of two. Pixels whose samples disagree
  (silhouettes, creases, overlaps) get 8 more. Each pixel is shaded once per triangle, as GPUs do with
  multisampling, and partly covered pixels blend by coverage in linear light.
- **Textures:** mipmapped with trilinear filtering, so a 48x48 face texture a dozen pixels across
  stays steady instead of shimmering as a die turns.
- **Depth cues:** surfaces dim with distance across the scene (fog), and where one surface passes in
  front of another the far side gets a dark outline.
- **Backgrounds:** behind the scene, a colour, a vertical gradient, a sky that follows the camera
  (zenith, horizon and ground colours), or a sky box of six pictures.
- **Output:** the screen is a grid of cells, and each refresh sends only the cells that changed, as
  VT escape sequences, wrapped in synchronized-output markers where the terminal supports them. There
  is no curses: on Windows the console is put in VT mode, and keys and mouse clicks are read as
  console input records.
- **Speed:** the per-pixel work (projecting and clipping triangles, rasterizing, shading, fog and
  outlines, matching glyphs, encoding the output) is plain Python loops that Numba compiles to
  machine code, split across CPU cores where it pays. All objects go through one kernel call rather
  than one each, meshes shared by many objects are stored once, and objects wholly outside the view
  are skipped before any of their triangles are touched. The renderer keeps its working arrays from
  frame to frame rather than allocating new ones. Triangles crossing the camera's near plane are
  clipped rather than dropped.

## Layout

The tests, in `tests/test_renderer.py`, use only the public API. The conventions for engine code (Numba
kernels, and the one-writer rule that keeps multithreaded rendering deterministic) are in `CLAUDE.md`. The `examples/` programs show the
engine in use; games may build on them, as [Zombie Dice](https://github.com/arp-Trosh/zombieDice)
does with `examples/dice.py`.

| module          | role |
|-----------------|------|
| `transforms.py` | projection/view matrices, quaternions |
| `mesh.py`       | `Mesh` with vertex normals, per-vertex or per-face colours, bounding sphere and cached mipmaps, OBJ loader, textured `make_box` |
| `texture.py`    | mipmap chains and trilinear sampling |
| `raster.py`     | `FrameBuffer` (linear RGB premultiplied by coverage, alpha, depth, object ids), projection of all objects at once with view culling and near-plane clipping, multi-sample z-buffered rasterizer |
| `scene.py`      | `Camera`, `Light`, `PointLight`, `Node`, `Object3D`, `Renderer` (transform, cull, lighting, shadow maps, transparency, mirrors, multisampling with extra edge samples, fog, outlines, picking) |
| `background.py` | what is drawn behind the scene: `Gradient`, `Sky`, `SkyBox` |
| `color.py`      | sRGB/linear conversion, named `Color`s, OKLab palette matching, dithering, SGR colour codes |
| `glyphs.py`     | glyph sets (half, quad, sextant, ascii) and matching pixels to cells |
| `keys.py`       | `Key` codes, `KeyRelease`, `MouseEvent`, the VT and kitty-protocol input decoder, `HeldKeys` |
| `console.py`    | raw terminal I/O for POSIX (termios) and Windows (console API), `WindowsInput` (console key and mouse records to VT sequences), key state on Windows, colour and glyph detection |
| `terminal.py`   | `Screen` (cell grid, text, frames, diffed output, held keys), `run`, command-line display flags |
| `ui.py`         | widgets: `Button`, `Toggle`, `Slider`, `Choice`, laid out in a `Panel`; `DisplayControls` (glyphs, colours, frame rate, shadows, reflections on F2-F6) |
| `shapes.py`     | mesh builders: `text_mesh` (extruded text in any bitmap font), `bitmap_mesh`, `blob_mesh` (ellipsoid), `block_mesh`, `pillow_mesh` (a 2D shape puffed into a cushion), `merge_meshes` |
| `examples/dice.py`   | pip-textured die, `orientation_showing`, `top_face`, `RollAnimation` (result chosen first, then animated to land on it); Zombie Dice builds its dice on it |
| `examples/hud.py`    | the status line the demos share: help text and `DisplayControls` |
| `examples/demo.py`   | the dice roll demo (`python3 -m unicode3d`) |
| `examples/viewer.py` | the model viewer |
| `examples/balls.py`  | balls in a room, with a settings panel |
| `examples/maze.py`   | the maze screensaver |
| `examples/room.py`   | the walk-around courtyard |

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

#### Display detection

`Screen` picks a glyph set and colour depth unless told otherwise (arguments,
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

#### Renderer

`Renderer(width, height, cell_pixels=(1, 2), ...)` options:

| option | default | effect |
|--------|---------|--------|
| `cell_pixels` | `(1, 2)` | pixels per cell; pass `screen.cell_pixels` to `resize()` every frame, since it follows the glyph set |
| `cell_aspect` | `0.5` | a cell's width divided by its height |
| `samples` | `4` | samples per pixel: 1, 4, 8 or 16 |
| `edge_samples` | `8` | extra samples in pixels whose samples disagree: 0, 4, 8 or 16 |
| `fog` | `0.3` | how much the farthest surfaces are dimmed |
| `outline` | `0.55` | how much the far side of a depth edge is darkened |
| `lod_bias` | `-0.5` | added to texture mip levels: lower is sharper, higher is softer |
| `background` | `None` | drawn behind the scene (see [Backgrounds](#backgrounds)); `None` leaves it empty, so the screen's background shows |
| `shadow_size` | `1024` | texels across the shadow map of each `Light` (see [Shadows](#shadows)) |
| `point_shadow_size` | `256` | texels across each of the six faces of a `PointLight`'s shadow map |
| `shadow_softness` | `1.5` | how far shadow edges are blurred, in shadow-map texels either way |
| `shadows` | `True` | `False` draws no shadows whatever the lights say (a graphics setting) |
| `transparency_layers` | `4` | see-through surfaces each pixel can show in front of the solid ones (see [Transparency](#transparency)) |
| `reflections` | `True` | `False` draws no reflections whatever the objects' reflectivity (see [Reflections](#reflections)) |
| `mirror_bounces` | `1` | how deep mirrors show each other, at most 4 |

`render(objects, camera, lights)` takes one light or a list of them. It returns the renderer's own
`FrameBuffer`, which the next render reuses; `copy()` it to cache a frame. `fb.ids` tells you which
object (its index in the render list plus one) covers each pixel. `renderer.project(point)` gives
the cell a world point landed on, for placing text labels. An unchanged scene isn't drawn again:
`render()` hands back the last frame, and `renderer.draws` counts only the renders that rasterized.

#### Picking

`renderer.pick(x, y)` tells what the last render drew in cell `(x, y)` of its frame (cells count
from the frame's top-left, so subtract where you drew it, e.g. from a `MouseEvent`): a `Pick` with
`object` (the `Object3D` itself), `position` (the point on its surface, in the world) and
`distance` from the camera, or `None` where nothing was drawn. `renderer.ray(x, y)` gives the line
of sight through a cell as `(origin, direction)`, for aiming at things that aren't drawn, such as an
imaginary floor plane:

```python
for ev in events:
    if isinstance(ev, MouseEvent) and ev.pressed and ev.button == MouseEvent.LEFT:
        hit = renderer.pick(ev.x, ev.y - top)
        if hit:
            selected = hit.object
```

#### Scene graph

An `Object3D`'s `position`, `rotation` and `scale` are relative to its `parent`, if it has one: a
`Node` (a transform with no mesh, for grouping) or another `Object3D`. Children move, turn, scale
and hide with their parent, through any number of levels. Only the objects to draw go in the render
list; their parents are followed automatically. `obj.world_transform()` gives `(position, rotation,
scale, visible)` in the world, and `obj.to_world(point)` places a point given in the object's own
space.

```python
from unicode3d import Node, Object3D
from unicode3d.shapes import blob_mesh, block_mesh
from unicode3d.transforms import UP, quat_axis_angle

car = Node(position=np.array([0.0, 0.0, 0.0]))
body = Object3D(block_mesh((0, 0.5, 0), (2, 0.6, 1)), color=Color.RED, parent=car)
wheel = blob_mesh((0.3, 0.3, 0.12))
wheels = [Object3D(wheel, np.array([x, 0.3, z]), parent=car) for x in (-0.7, 0.7) for z in (-0.55, 0.55)]
car.rotation = quat_axis_angle(UP, heading)   # the whole car turns
renderer.render([body, *wheels], camera, light)
```

#### Many objects

Draw as many objects as you like in one render list: objects that share a `Mesh` share its packed
copy, all objects are projected in one parallel kernel call, and those wholly outside the view are
skipped. 400 small balls (140,000 triangles) take about 15 ms at 180x50 cells. For thousands of
static pieces, merging them into one mesh with `merge_meshes` (keeping each part's colour) is
cheaper still.

#### Shapes

`unicode3d.shapes` builds meshes to use with `Object3D`: `text_mesh(text, font)` extrudes
text in a bitmap font you supply (`{char: ["#..#", ...]}`, every glyph the same height) and returns
`(mesh, width)`; `bitmap_mesh(cells)` does the same for any boolean grid; `blob_mesh(radii, center)`
is an ellipsoid with an optional bump function; `block_mesh(center, size, rotation)` a box;
`pillow_mesh(shape)` puffs a 2D inside/outside function into a cushion with texture coordinates
that line up with the shape; and `merge_meshes(meshes, colors=None)` joins them into one, keeping
their colours or giving each part the colour listed for it.

#### Colours

`Object3D.color` takes a named `Color` or an `(r, g, b)` triple (0..255 ints or 0..1 floats). A mesh
can carry its own colours: `mesh.face_colors` (one per face) or `mesh.vertex_colors` (one per
vertex, blended smoothly across each face, in linear light). Like textures, they multiply the
object's colour, so give the object `color=(255, 255, 255)` to show them as they are; the default
colour is a light grey. A fourth column is opacity (see [Transparency](#transparency)).

```python
terrain = Mesh(vertices, faces)
terrain.vertex_colors = np.where(vertices[:, 1:2] > 2.0, (240, 240, 250), (60, 140, 50))  # snow above 2
```

#### Lights

Pass `render()` a list of lights and their light adds up.

- `Light(direction, ambient=0.3, diffuse=0.7, specular=0.35, shininess=24, color=(255, 255, 255),
  shadows=False)` is light from far away, the same everywhere (the sun). See [Shadows](#shadows).
- `PointLight(position, color=(255, 255, 255), diffuse=0.8, specular=0.35, shininess=24,
  range=10, ambient=0, shadows=False)` spreads from a point and fades smoothly to nothing at `range`.
- `Object3D.emissive` is light a surface gives off itself: `1.0` shows its colour at full brightness
  whatever the lighting (a lamp's bulb, a screen, a glowing marker). It lights nothing else; put a
  `PointLight` beside it for that.

Levels (`ambient`, `diffuse`, `specular`) are perceived brightness from 0 to 1, and a light's colour
scales them channel by channel. The highlight takes the light's colour, so lower `specular` for
large flat faces that would otherwise wash out. Each light costs a little shading time per pixel;
point lights cost nothing where they are out of range.

```python
lights = [Light(ambient=0.15, diffuse=0.3), PointLight(np.array([0.0, 2.5, 0.0]), color=(255, 200, 140), range=8)]
bulb = Object3D(blob_mesh((0.15, 0.15, 0.15)), np.array([0.0, 2.5, 0.0]), color=(255, 230, 180), emissive=1.0)
```

#### Shadows

`shadows=True` on a `Light` or a `PointLight` makes objects block that light from whatever lies
behind them, where only its `ambient` light still reaches. Every object casts shadows unless it
has `cast_shadows=False`: use that for a lamp's own bulb (which would otherwise shadow
everything from the light inside it), or for a room with a ceiling, which would keep the sun
out. Surfaces facing away from the light are in their own shadow.

Each shadowed light draws the scene once more, as seen from the light, into a shadow map. A
`Light`'s is `shadow_size` x `shadow_size` texels and covers every object (larger is sharper and
slower; spread over a big scene, texels get coarser). A `PointLight`'s is a cube of six
`point_shadow_size` x `point_shadow_size` faces looking every way from it, holding whatever is in
its range. A map is only redrawn when an object or its light moves, so walking the camera around
a still scene costs little. At 180x50 cells, a shadowed light adds about 1.1-1.3 ms to a frame
when something moves (a sun or a lamp alike) and about 0.4-0.6 ms when nothing but the camera does.
`shadow_softness` blurs edges further; they are always smoothed over at least a pixel on screen.
`renderer.shadows = False` switches all shadows off, as F5 does in the demos
(`DisplayControls(renderer=...)`).

```python
sun = Light(direction=np.array([0.5, -1.0, -0.3]), shadows=True)
room = Object3D(room_mesh, cast_shadows=False)  # lets the sun in; what is inside still casts shadows
lamp = PointLight(np.array([0.0, 2.5, 0.0]), range=8, shadows=True)
bulb = Object3D(blob_mesh((0.15, 0.15, 0.15)), lamp.position, emissive=1.0, cast_shadows=False)
```

#### Transparency

`Object3D(..., opacity=0.3)` makes an object see-through: 1 (the default) is solid, 0 is not drawn
at all. A mesh's `vertex_colors` or `face_colors` can also carry opacity, as a fourth column
(0..1 or 0..255, like the colour, but not gamma-encoded), which multiplies the object's: a pane
that fades out towards one edge, or a model with some faces clear.

See-through surfaces blend over what is behind them in order of depth, whatever order the render
list has, including where they cross each other. Each pixel shows up to `transparency_layers`
(4) of them in front of the solid surface there, the nearest ones if there are more. A see-through
object always shows its back faces, so the far side of a glass box shows through its near side.
Highlights stay at full strength however clear the surface is, and surfaces grow more opaque when
seen at a slant, as glass does; below an opacity of 0.25 both fade too, so that an object faded to
0 disappears. What a pixel shows first is what `pick()` finds, so a click on a window picks the
window.

With shadows, light through a see-through object is dimmed and tinted by it: through clear glass
nearly all of it gets through, through red glass red light does, and through nearly solid glass
little does.

Textures can carry opacity too, as a fourth channel: `(H, W, 4)` arrays, alpha 0..1 (0 is a hole).
What it is for is worked out from the texture:

- **Cut-outs**, mostly solid or clear with at most soft edges between (leaves, a fence, lettering, a
  window frame): drawn as solid surfaces with holes, each sample of a pixel testing the texture, so
  the edges of the holes are smoothed like the edges of shapes, and far away, where a texel is
  smaller than a pixel, the holes thin out gradually rather than flickering. They cost no
  transparency layers, and cast shadows with holes in them.
- **Translucent textures**, where more than a tenth is partly see-through (stained glass): drawn with
  the see-through surfaces, the texture's alpha multiplying the object's, and casting light tinted
  pane by pane.

Shrunk (mipmapped), a texture's colour is averaged over what is there, not over its holes, so the
edge of a leaf stays green rather than darkening. Anything with holes or see-through parts shows
its far side through them.

Scenes without see-through objects cost nothing extra. Otherwise the cost grows with the screen area
that see-through surfaces cover: three glass dice, covering about an eighth of a 180x50 view, add
about 2 ms.

```python
glass = Object3D(make_box(), color=(200, 225, 255), opacity=0.15)  # a faintly blue glass case
pane.mesh.vertex_colors = [(255, 255, 255, 255), (255, 255, 255, 0), ...]  # solid on one side, clear on the other
```

#### Reflections

`Object3D(..., reflectivity=0.8)` makes an object reflect, from 0 (not at all, the default) to 1 (a
perfect mirror). What it reflects depends on its shape:

- **Flat meshes** (all faces on one plane: a wall mirror, a polished floor or table top, still
  water) are mirrors. The renderer draws the scene again from the camera reflected in the mirror's
  plane, into just the pixels the mirror covers, and blends it in by the reflectivity: everything in
  front of the mirror, lit, shadowed, see-through and cut out as usual, with the background beyond.
  Only solid objects are mirrors; a flat see-through one reflects the background, like a curved one.
- **Curved meshes** (chrome, polished metal, lacquer) reflect the background (the sky, sky box,
  gradient or colour) in the direction each point reflects the view, but not other objects.
- **See-through surfaces** reflect the background at a slant anyway, as glass and water do, whatever
  their reflectivity: the extra opacity they gain seen edge-on shows the sky.

`mirror_bounces` sets how deep mirrors show each other. With 1 (the default), a mirror seen in a
mirror shows the background reflected in it rather than the scene: enough unless strong mirrors face
each other. Up to 4 draws deeper images, each an extra pass, but stops early where an image covers
fewer than 50 pixels or its reflectivities multiply to less than 5%, so even an infinity mirror
costs only a few passes. `renderer.reflections = False` switches all reflections off, as F6 does
in the demos.

Each mirror on screen costs a pass: about 1.5–2 ms at 180x50 cells, plus the drawing of the pixels
it covers (a floor mirror under three dice adds about 3 ms in all). Curved shiny objects cost
almost nothing extra. A mirror off screen, or hidden, costs nothing.

```python
mirror = Object3D(flat_mesh, reflectivity=0.9)       # flat: shows the scene
pool = Object3D(water_mesh, color=(40, 70, 80), reflectivity=0.6)
chrome = Object3D(blob_mesh((1, 1, 1), rings=24, segments=32), reflectivity=0.85)  # curved: the sky
renderer = Renderer(80, 24, background=Sky(), mirror_bounces=2)
```

#### Backgrounds

`Renderer(background=...)`, or `renderer.background = ...` at any time:

- a colour: fills everything the scene leaves empty;
- `Gradient(top, bottom)`: fixed on screen, top row to bottom row;
- `Sky(zenith, horizon, ground)`: by the direction each pixel looks in, so it moves as the camera
  turns: the horizon colour at eye level, fading up to the zenith and quickly down to the ground;
- `SkyBox(textures)`: six pictures (in `+X, -X, +Y, -Y, +Z, -Z` order, `(H, W)` brightness or
  `(H, W, 3)` colour) on the inside of a box that moves with the camera, so it never gets nearer.
  The sides are upright; the top continues upward from the top edge of the `-Z` picture and the
  bottom downward from its bottom edge (`background.SKYBOX_FACES` has the exact axes).

The background fills in behind edges in proportion to how much of each pixel the scene leaves
uncovered, so antialiased silhouettes blend into it. It is not fogged or outlined.

#### Screen and run()

`run(frame_fn, fps=30, glyphs=None, color=None, mouse=False,
background=None, title=None, key_release=False)` takes over the terminal (naming its window
`title`, if given) and restores it however the loop ends (Ctrl-C raises `KeyboardInterrupt`).
`frame_fn(screen, dt, keys)` returns `False` to stop. `keys` holds ints (a character's code, or a
`Key` such as `Key.UP`, `Key.ENTER`, `Key.ESC`), `MouseEvent`s and, with `key_release=True`,
`KeyRelease`s (see [Input](#input)). `screen.text()` draws in the terminal's own ANSI colours,
so text follows the user's theme. Characters that aren't exactly one cell wide are shown as `?`.
`background=(r, g, b)` fills the screen with a known colour so anti-aliased edges blend into it
exactly; by default, edges blend toward black over the terminal's own background.
`Screen(size=(rows, cols))` with no console gives an off-screen grid for tests;
`render_updates()` returns the escape sequences a refresh would send. `screen.set_glyphs(name)` and
`screen.set_color(mode)` switch modes while running (`screen.glyph_modes` lists the glyph sets the
terminal can take), and `screen.fps` is the target frame rate, which `run()` re-reads every frame;
`screen.measured_fps` is the rate it achieved over the last second.

#### Input

**Mouse.** `run(mouse=True)` reports clicks and the wheel as `MouseEvent(x, y, button, pressed,
moved)`; `mouse="drag"` adds moves while a button is held (`moved=True`, `pressed=True`), which
dragging a slider needs; `mouse="move"` reports every move (with `button=MouseEvent.NONE` when no
button is held), for hover effects.

**Holding keys.** Most terminals send only key presses, and a held key repeats: once, a pause of
about half a second, then steadily. That is fine for menus but makes "walk while W is held" stutter.
`screen.held` (a `HeldKeys`) tells which keys are down, and `"w" in screen.held` is the way to ask:

```python
def frame(screen, dt, keys):
    if "w" in screen.held:
        player.walk(speed * dt)
```

With `run(key_release=True)` the engine asks the terminal to report releases too, and passes them
on as `KeyRelease(key)` events:

| terminal | held keys |
|----------|-----------|
| kitty, foot, Ghostty, Alacritty, iTerm2, Rio; WezTerm with `enable_kitty_keyboard = true` | exact: presses and releases come through the kitty keyboard protocol |
| Windows (Windows Terminal and the console) | exact: the engine asks Windows for the state of each held key |
| everything else | estimated: a key counts as held for half a second after its press, and then for as long as its repeats keep coming. A tap moves a player for up to half a second, and on many systems holding a second key stops the first one repeating |

`screen.held.exact` says which you have. Letters are tracked without case, so Shift+W still holds
W. With the kitty protocol on, Ctrl-C arrives as a key, and `screen.keys()` raises
`KeyboardInterrupt` for it as usual.

#### Widgets

`unicode3d.ui` has text widgets that work with the mouse and keyboard: `Button(label, action,
key=None)`, `Toggle(label, value, key=None)`, `Slider(label, value, lo, hi, step=1, keys="[]")` and
`Choice(label, options, key=None)`. A `Panel` lays them out in a row (or a column, with
`vertical=True`) and handles a frame's events: clicks, drags, the mouse wheel, each widget's own key,
and Tab/Shift-Tab to move the focus followed by Left/Right/Enter/Space. It returns the events it
didn't use, so the rest of the program sees only those. Each widget calls `on_change(value)` when
the user changes it, or you can read `widget.value` every frame.

`DisplayControls()` is the display settings panel from Zombie Dice: glyphs on F2, colours on F3 and
the frame rate (achieved/target) on F4, each also clickable; `DisplayControls(renderer=renderer)`
adds shadows on F5 and reflections on F6. Draw it every frame; its `width` stays
fixed as the values change.

```python
from unicode3d.ui import Button, DisplayControls, Panel, Slider, Toggle

count = Slider("Balls", 20, 1, 400, keys="[]")
spin = Toggle("Spin", True, key="s")
panel = Panel([count, spin, Button("Reset", reset, key="r")])
controls = DisplayControls()

def frame(screen, dt, events):
    events = controls.handle(panel.handle(events), screen)
    ...
    rows, cols = screen.size()
    panel.draw(screen, rows - 1, 1)
    controls.draw(screen, rows - 1, cols - controls.width - 1)
    screen.refresh()

run(frame, mouse="drag")
```

#### Performance

Rendering a frame and building its screen update takes about 1.5 ms for a
60x15-cell view of three rolling dice, and about 3.5 ms for a 180x50 view in `sextant` mode. Cost
grows with the pixel count and, more slowly, the triangle count: a 27,000-triangle sphere filling a
180x50 view takes about 5 ms, and 400 separate balls about 15 ms. Shadows, see-through surfaces and
mirrors cost more where they are used (see [Shadows](#shadows), [Transparency](#transparency) and
[Reflections](#reflections)); the room demo, with all of them, draws a frame in about 12 ms. A
scene that hasn't changed since the last `render()` (same objects, poses, camera, lights,
background and size) isn't drawn again, so still frames cost almost nothing; after editing a mesh's
arrays in place, call `renderer.invalidate()`. The terminal showing the frame usually takes longer
than drawing it. `python benchmarks/bench.py` times each stage of a frame.

#### First run

Numba compiles the renderer the first time it is used, which takes about 30 seconds;
the result is cached (in `__pycache__` beside the code, or a user cache folder if that can't be
written), so later runs start in a fraction of a second. Upgrading unicode3d or Numba, or moving to
another CPU, compiles it again. `run()` compiles before the first frame and shows "First run
compile, please wait..." while it does; programs that drive a `Screen` themselves can call
`compile_kernels()` at a moment of their choosing.

#### Threads

The renderer uses every CPU core Numba finds, about twice as fast as one core at
typical sizes. Set `NUMBA_NUM_THREADS`, or call `numba.set_num_threads()`, to leave cores for other
work. Frames come out identical whatever the thread count.

#### Windows

Needs Windows 10 or later (for VT sequences in the console), numpy and Numba. Windows
Terminal is recommended, and gets sextants by default; the classic console works too, with quadrants
(its default fonts may lack sextants).

## Versions

Releases are git tags (`v0.1.0`, ...) following [semantic versioning](https://semver.org): a patch
release (`0.1.1`) fixes bugs, a minor release (`0.2.0`) adds features, and before 1.0 a minor release
may also change the API. `unicode3d.__version__` holds the version. To release: bump
`__version__` in `unicode3d/__init__.py`, commit, then `git tag v0.4.0 && git push --tags`.

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

---

*Disclaimer: This project was created with Claude Code.*
