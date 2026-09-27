# unicode3d: notes for working on the engine

A 3D renderer for the terminal in Python. Numba is a required dependency: per-pixel and per-triangle work is
written as `@njit(cache=True)` kernels (plain loops over numpy arrays), and Python code only prepares arrays and
calls them. Keep it that way: new rendering features (shadows, transparency, more lights, materials, ...) go into
kernels, not into vectorized numpy over large temporary arrays, which is several times slower.

## The one-writer rule (parallel kernels)

Kernels declared with `parallel=True` split a `prange` loop across CPU cores. **Every array element a parallel
kernel writes must be written by exactly one iteration of its `prange` loop**, so which thread runs what can never
change the result.

- Split the work by what is written, not by what is read. `raster.rasterize` is the pattern: it loops in parallel
  over bands of pixel rows, and each band goes through all the triangles in order, touching only its own rows.
  Looping in parallel over triangles instead would let two threads update the same pixel at once.
- No `+=`, min/max or "if nearer, replace" on a shared element from inside a `prange` loop. For a reduction over
  the whole frame (such as the depth range in `scene._post_effects`), loop serially, or have each iteration write
  its own slot and combine the slots serially afterwards.
- Scratch space written inside a `prange` loop must be either indexed by that iteration (like `sample_rgb[c]` in
  `scene._resolve`) or allocated inside the loop body, once per iteration of the outer loop (like `feat` in
  `glyphs._match`), never shared.
- Read-only inputs can be shared freely.
- A kernel may read an element that another iteration writes only if it runs in a separate pass afterwards.

Breaking the rule gives frames that differ with the thread count or from run to run, for example rare flickering
pixels. `test_same_frame_on_any_number_of_threads` compares whole frames drawn on one thread and on all of them.
It catches most violations, but races show up by chance, so passing it is not proof: check new kernels against the
rule by reading them. Add anything a new kernel draws to that test's scenes.

## Other kernel conventions

- **One set of argument types.** Numba compiles a kernel separately for each combination of argument types and
  array layouts, and a compile takes seconds. `terminal.compile_kernels()` compiles everything before the first
  frame (while `run()` shows "First run compile, please wait..."), so a kernel must see the same types then as in
  every later frame: convert inputs with `np.ascontiguousarray(x, dtype)` before the call (see
  `Renderer._geometry`), and extend `compile_kernels()`'s scene when adding a kernel or a new code path.
- **No per-frame allocation of big arrays.** Frame-sized working arrays come from `Renderer._buffers`, which keeps
  them from frame to frame. Allocating them afresh each frame made frame times double depending on what the
  program did beforehand (page faults from the allocator).
- **No small allocations in per-pixel loops** (`np.empty` inside the innermost loop costs more than the work).
  Use scalars or scratch space allocated outside the loop.

## Checking a change

- `python -m unittest` from the repository root. The tests use the public API.
- `python -m benchmarks.bench` times each stage of a frame; compare it before and after a change. Its `first`
  column includes compiling or loading the kernels.
- For a change meant to keep the picture the same, compare framebuffers (`rgb`, `alpha`, `depth`, `ids`) and
  `Screen.render_updates()` output before and after, exactly or to within rounding.
- Editing a module makes Numba recompile its kernels on the next run (the cache is keyed on the source file), so
  the first run after an edit is slow. That is expected.
