# unicode3d: notes for working on the engine

A 3D renderer for the terminal in Python. Numba is a required dependency: per-pixel and per-triangle work is
written as `@njit(cache=True, error_model="numpy")` kernels (plain loops over numpy arrays), and Python code only
prepares arrays and calls them. Keep it that way: new rendering features (shadows, transparency, more lights,
materials, ...) go into kernels, not into vectorized numpy over large temporary arrays, which is several times
slower.

## The one-writer rule (parallel kernels)

Kernels declared with `parallel=True` split a `prange` loop across CPU cores. **Every array element a parallel
kernel writes must be written by exactly one iteration of its `prange` loop**, so which thread runs what can never
change the result.

- Split the work by what is written, not by what is read. `raster.rasterize` is the pattern: it loops in parallel
  over bands of pixel rows, and each band goes through all the triangles in order, touching only its own rows.
  Looping in parallel over triangles instead would let two threads update the same pixel at once.
- No `+=`, min/max or "if nearer, replace" on a shared element from inside a `prange` loop. For a reduction over
  the whole frame (such as the depth range in `shading.post_effects`), loop serially, or have each iteration write
  its own slot and combine the slots serially afterwards.
- Scratch space written inside a `prange` loop must be either indexed by that iteration (like `sample_rgb[c]` in
  `shading.resolve`) or allocated inside the loop body, once per iteration of the outer loop (like `feat` in
  `glyphs._match`), never shared.
- Output whose length depends on the work (like the triangles `raster.project` makes, which clipping can
  split) is written in two passes: count what each iteration will write, turn the counts into offsets
  serially, then have each iteration write only from its own offset (`raster.transform` counts,
  `raster.project` writes).
- Read-only inputs can be shared freely.
- A kernel may read an element that another iteration writes only if it runs in a separate pass afterwards.

Breaking the rule gives frames that differ with the thread count or from run to run, for example rare flickering
pixels. `test_same_frame_on_any_number_of_threads` compares whole frames drawn on one thread and on all of them.
It catches most violations, but races show up by chance, so passing it is not proof: check new kernels against the
rule by reading them. Add anything a new kernel draws to that test's scenes.

Programs may also draw from threads of their own, each with its own Renderer and Screen. Python code that launches
parallel kernels holds `threads.kernel_lock()` (as `Renderer.render`, `match_cells` and `linear_to_srgb` do): Numba's
fallback `workqueue` threading layer aborts the process when two threads launch at once, and the lock makes them
take turns there (`test_threads_on_the_workqueue_layer`).

## Other kernel conventions

- **One set of argument types.** Numba compiles a kernel separately for each combination of argument types and
  array layouts, and a compile takes seconds. `terminal.compile_kernels()` compiles everything before the first
  frame (while `run()` shows "First run compile, please wait..."), so a kernel must see the same types then as in
  every later frame: convert inputs with `np.ascontiguousarray(x, dtype)` before the call (see
  `Renderer._geometry`), and extend `compile_kernels()`'s scene when adding a kernel or a new code path.
- **Keep `kernel_signatures.py` current.** On a first run, `precompile.py` compiles the kernels
  `compile_kernels()` calls in several Pythons at once (about 10 s instead of 40), from the argument types listed
  in `kernel_signatures.py`. After changing a kernel's arguments or adding one, run
  `python -m unicode3d.precompile --update` (about 40 s); `test_kernel_signatures_are_current` fails until then.
  A kernel missing from the list still works, compiled on its own after the others.
- **No per-frame allocation of big arrays.** Frame-sized working arrays come from `Renderer._buffers`, which keeps
  them from frame to frame. Allocating them afresh each frame made frame times double depending on what the
  program did beforehand (page faults from the allocator).
- **No small allocations in per-pixel loops** (`np.empty` inside the innermost loop costs more than the work).
  Use scalars or scratch space allocated outside the loop.

## Bad numbers must not crash (stability)

A frame can hold NaN or infinities (a mesh or a pose gone wrong, a camera at a degenerate spot, a division by a
near-zero), and a terminal can be tiny or huge (a full screen at font size 2 is about 960x215 cells). Neither may
stop the program: frames may be slower or wrong for a moment, never an exception or a crash.

- **Every kernel takes `error_model="numpy"`.** With Numba's default, a float division by zero raises
  ZeroDivisionError in serial code, and SystemError from a helper called inside a `prange` loop, which ends the
  program. `test_every_kernel_uses_the_numpy_error_model` checks it.
- **Floats become indices only once they are in range.** Converting NaN, an infinity or anything beyond int64 to an
  int gives an undefined value, and Numba doesn't check bounds, so an index or loop bound from one writes outside
  the array (a segfault, or quietly corrupted memory). Clamp as floats first, with comparisons that are false for
  NaN (Numba's `max(nan, 0.0)` is NaN): `texture.clamp_index` for an index, `raster.pixel_range` for a span of
  pixels, and `raster.drawable(area)` to skip triangles with non-finite corners.
- **`Renderer.render()` runs under `np.errstate(all="ignore")`**, as a warning would be printed over the picture;
  objects with a non-finite pose are skipped in `_instances`. Other numbers are made finite where they enter (NaN
  as 0: `_instances`, `light_rows`, `background_args`, `Mesh.corner_colors`, `build_mipmaps`), and kernels clamp
  as `min(1.0, max(0.0, x))`, which turns NaN into 0 (`max(nan, 0.0)` is NaN, `max(0.0, nan)` is 0.0).
- **Mesh arrays are checked before kernels index them** (`Mesh.check()`, from `_pack_meshes`): a face index, a colour
  row or a material beyond the end of its array is a ValueError, not a read outside it (one such crashed Python).
- **No matrix inverses of the view** (singular for a camera at its target, near = 0, ...): pixel directions come
  from `transforms.view_axes`.
- **Memory is bounded by `Renderer.max_pixels`.** Beyond it the scene is drawn into `Renderer._fb`, a smaller
  framebuffer, and `raster.upscale` stretches it into `Renderer.framebuffer`. Code that draws uses `self._fb`
  (its size is what rasterizing, shading and buffers work with); only the result and `pick()` use
  `self.framebuffer`.
- `run()` writes a line per run and any traceback (and, through faulthandler, any crash of Python itself) to
  `terminal.crash_log_path()`. Ask for that file when a user reports a crash.

## Where things are

`scene.py` holds what a scene is made of (`Camera`, `Node`, `Object3D`) and re-exports the rest; `renderer.py`
the `Renderer`, which inherits its shadow-map methods from `shadows.ShadowMaps` and its mirror passes from
`mirrors.Mirrors`; `shading.py` the shading kernels; `raster.py` projection and rasterizing; `lights.py` the
lights; `models.py` and `gltf.py` loading models (OBJ and glTF; no kernels); `kernel_cache.py` keeps Numba's
cache in step with helpers in other modules; `queries.py` the ray and overlap queries (`Colliders`, with a bounding volume hierarchy per mesh built in a kernel and
shared through `mesh_tree`); `threads.py` has the lock that keeps threads from launching
parallel kernels at once where Numba can't take that, and `prefer_sleeping_workers()` (run on import: OpenMP
preferred, its workers sleeping between kernels; compare threading layers with `NUMBA_THREADING_LAYER=tbb`). Instances reach the kernels as `inst` dicts of arrays
(`Renderer._instances`): each has a `lin` (3, 3) matrix (rotation and per-axis scale, parents included) and `pos`,
and `flip` where the matrix mirrors.

## Checking a change

- `python -m unittest` from the repository root. The tests use the public API. `tests/test_stability.py` covers
  huge terminals, resizing while running, bad numbers, and a real pseudo-terminal (on Unix).
- `NUMBA_BOUNDSCHECK=1 NUMBA_CACHE_DIR=$(mktemp -d) python -m unittest` (about a minute) runs the suite with every
  kernel checking its indices, so a read or write outside an array raises (inside a parallel kernel, a SystemError)
  rather than quietly corrupting memory. It needs a cache of its own: Numba's cache key leaves the flag out, so a
  warm cache would load the unchecked kernels. CI runs it too (the `bounds` job).
- Before a release, `python -m benchmarks.stress` (a few minutes): the room demo walked through sizes from one
  cell to 1920x540 while resizing, and drawn from thousands of random close-up cameras.
- `python -m benchmarks.bench` times each stage of a frame; compare it before and after a change, run back to
  back (the machine's speed drifts by 10-20% over a session, so numbers from earlier in the day mislead). It
  starts with the machine and versions; `compile_kernels()` runs first, so its `first` column is only each
  scene's first frame.
- `python -m benchmarks.gallery diff` renders the reference scenes with `HEAD` (in a temporary worktree) and with
  the working tree, and compares framebuffers (`rgb`, `alpha`, `depth`, `ids`) and terminal cells. A change meant
  to keep the picture the same should come out "identical" or "within tolerance" for every scene; a change meant
  to alter it shows where (pictures in the `diff` folder it names). Add a scene for a new feature (built with
  constructor arguments, so that older revisions without the feature skip it).
- Editing a module makes Numba recompile its kernels on the next run (the cache is keyed on the source file), so
  the first run after an edit is slow. That is expected. Numba alone wouldn't recompile kernels in *other* modules
  that call a helper you edited (`texture.sample` from `shading.py`, say); `kernel_cache.refresh()`, run when
  `unicode3d` is imported, does: it fingerprints the modules each kernel module calls into (and the numbers and
  arrays it imports) in `__pycache__/<module>.deps`, and drops that module's cache when they change. A kernel
  module must reach its helpers through names it imports (`from .texture import sample`, or `from . import
  texture`), which is how `refresh()` finds them. `test_kernel_cache.py` shows the stale case and the fix.
