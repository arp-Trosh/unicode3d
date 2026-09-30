# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
"""Compiling the kernels on several cores at once, for the first run after installing or upgrading.

Numba compiles one function at a time, so compiling every kernel takes about half a minute even on a fast
machine. Each kernel it compiles goes into its cache, though, and a separate Python can put it there: precompile()
starts a few Pythons (`python -m unicode3d.precompile KERNEL...`), shares the kernels out among them, and waits;
compile_kernels() then loads them all from the cache. Compiling a kernel before anything calls it needs its
argument types, which kernel_signatures.py lists (with how long each took to compile, for sharing the work out).

Each index file in the cache must have one writer at a time (Numba reads it, adds an entry and writes it back), so
a worker writes only the kernels it was given: the helper functions those call, which several workers compile
along the way, are compiled without their cache.

Whatever goes wrong here (no cores to spare, a frozen program, a Numba whose types read differently, a worker
that fails), compile_kernels() compiles what is missing itself, as it would anyway.

After changing a kernel's arguments, or adding a kernel: python -m unicode3d.precompile --update
"""
import importlib
import os
import subprocess
import sys
import time

MAX_WORKERS = 6   # at most this many Pythons compiling at once (each takes about 300 MB while it does)
MIN_MISSING = 4   # fewer kernels than this to compile aren't worth starting other Pythons for
TIMEOUT = 600     # seconds to wait for the workers
SIGNATURES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernel_signatures.py")


def _dispatcher(module, name):
    return getattr(importlib.import_module(f"unicode3d.{module}"), name)


def _types(text):
    """A signature as kernel_signatures.py spells it (repr of Numba's types) back as a tuple of types."""
    from numba.core import types
    names = dict(vars(types), bool=types.boolean)
    return tuple(eval(text, {"__builtins__": {}}, names))


def _cached(dispatcher, sig):
    """Whether the kernel is in Numba's cache for these argument types (None if that can't be told)."""
    try:
        cache = dispatcher._cache
        key = cache._index_key(sig, dispatcher.targetctx.codegen())
        return key in cache._cache_file._load_index()
    except Exception:
        return None


def missing_kernels():
    """[(module, name, seconds, [signature text])] of the listed kernels not yet in the cache; None if that can't
    be told (no list, or Numba's cache works differently)."""
    try:
        from .kernel_signatures import KERNELS
    except ImportError:
        return None
    missing = []
    for module, name, seconds, signatures in KERNELS:
        try:
            dispatcher = _dispatcher(module, name)
            states = [_cached(dispatcher, _types(s)) for s in signatures]
        except Exception:
            return None
        if None in states:
            return None
        if not all(states):
            missing.append((module, name, seconds, signatures))
    return missing


def share_out(kernels, workers):
    """The kernels [(module, name, seconds, ...)] in `workers` groups of about equal compiling time: longest first,
    each to the group with the least so far."""
    groups = [[] for _ in range(workers)]
    load = [0.0] * workers
    for kernel in sorted(kernels, key=lambda k: -k[2]):
        i = load.index(min(load))
        groups[i].append(kernel)
        load[i] += kernel[2]
    return [g for g in groups if g]


def precompile(max_workers=MAX_WORKERS):
    """Compile the kernels missing from the cache in other Pythons, several at once; returns how many Pythons it
    started (0 if it didn't: all cached already, too few missing, or it can't here)."""
    if getattr(sys, "frozen", False) or not sys.executable or os.environ.get("UNICODE3D_NO_PRECOMPILE"):
        return 0
    missing = missing_kernels()
    if not missing or len(missing) < MIN_MISSING:
        return 0
    workers = max(min(max_workers, (os.cpu_count() or 1), len(missing)), 1)
    if workers < 2:
        return 0
    groups = share_out(missing, workers)
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)  # on Windows, no console window for each
    # The workers import this copy of unicode3d, wherever it came from (installed, or put on sys.path by a program).
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in (here, os.environ.get("PYTHONPATH")) if p))
    procs = []
    for group in groups:
        cmd = [sys.executable, "-m", "unicode3d.precompile"] + [f"{m}.{n}" for m, n, *_ in group]
        try:
            procs.append(subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                          stderr=subprocess.DEVNULL, creationflags=flags, env=env))
        except OSError:
            break
    deadline = time.monotonic() + TIMEOUT
    for proc in procs:
        try:
            proc.wait(timeout=max(deadline - time.monotonic(), 0.1))
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    return len(procs)


def _compile(names):
    """In a worker: compile the kernels named `module.name`, with their cache; everything else they call, without."""
    from numba.core.caching import NullCache
    from numba.core.registry import CPUDispatcher
    from .kernel_signatures import KERNELS
    wanted = {f"{m}.{n}": signatures for m, n, _, signatures in KERNELS}
    mine = {id(_dispatcher(*name.split("."))) for name in names}
    import pkgutil
    from . import __path__ as package_path
    for info in pkgutil.iter_modules(package_path):
        if info.ispkg or info.name in ("__main__", "precompile"):
            continue
        module = importlib.import_module(f"unicode3d.{info.name}")
        for value in vars(module).values():
            if isinstance(value, CPUDispatcher) and id(value) not in mine:
                value._cache = NullCache()
    for name in names:
        dispatcher = _dispatcher(*name.split("."))
        for text in wanted[name]:
            dispatcher.compile(_types(text))


# ----- keeping kernel_signatures.py up to date ------------------------------------------------------

def _record():
    """Run compile_kernels() and print what it compiled or loaded directly (the kernels Python calls): lines of
    module, name, seconds, signature, tab-separated."""
    from numba.core import event
    from numba.core.registry import CPUDispatcher
    from .terminal import compile_kernels

    class Timer(event.Listener):
        def __init__(self):
            self.stack, self.seconds = [], {}

        def on_start(self, ev):
            self.stack.append(time.perf_counter())

        def on_end(self, ev):
            start = self.stack.pop()
            if not self.stack:  # compiled for a call from Python, not as a helper of another kernel
                key = (id(ev.data["dispatcher"]), tuple(ev.data["args"]))
                self.seconds[key] = time.perf_counter() - start

    timer = Timer()
    os.environ["UNICODE3D_NO_PRECOMPILE"] = "1"
    with event.install_listener("numba:compile", timer):
        compile_kernels()
    import pkgutil
    from . import __path__ as package_path
    for info in pkgutil.iter_modules(package_path):
        if info.ispkg or info.name in ("__main__", "precompile"):
            continue
        module = importlib.import_module(f"unicode3d.{info.name}")
        for name, value in vars(module).items():
            if not isinstance(value, CPUDispatcher) or value.__module__ != module.__name__:
                continue
            for sig in {**value._cache_hits, **value._cache_misses}:
                seconds = timer.seconds.get((id(value), tuple(sig)))
                if seconds is None and sig not in value._cache_hits:
                    continue  # compiled only as a helper of another kernel
                text = repr(tuple(sig))
                if _types(text) != tuple(sig):
                    raise ValueError(f"{info.name}.{name}: {text} doesn't read back as the same types")
                print(f"{info.name}\t{name}\t{seconds or 0.0:.2f}\t{text}")


def update():
    """Rewrite kernel_signatures.py from compile_kernels() run afresh (compiling everything, in a new cache)."""
    import tempfile
    import numba
    with tempfile.TemporaryDirectory() as cache:
        env = dict(os.environ, NUMBA_CACHE_DIR=cache)
        out = subprocess.run([sys.executable, "-m", "unicode3d.precompile", "--record"], env=env, check=True,
                             capture_output=True, text=True).stdout
    kernels = {}
    for line in out.splitlines():
        module, name, seconds, text = line.split("\t")
        entry = kernels.setdefault((module, name), [0.0, []])
        entry[0] += float(seconds)
        entry[1].append(text)
    lines = ["# SPDX-License-Identifier: LGPL-3.0-or-later",
             "# Copyright (C) 2026 arp-Trosh",
             '"""The kernels compile_kernels() calls directly, with their argument types and about how many seconds',
             'each took to compile, for precompile.py. Made by `python -m unicode3d.precompile --update`: don\'t edit."""',
             f'NUMBA = "{numba.__version__}"',
             "KERNELS = ["]
    for (module, name), (seconds, texts) in sorted(kernels.items(), key=lambda kv: -kv[1][0]):
        lines.append(f"    ({module!r}, {name!r}, {seconds:.2f}, [")
        lines += [f"        {text!r}," for text in sorted(texts)]
        lines.append("    ]),")
    lines.append("]")
    with open(SIGNATURES_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    total = sum(s for s, _ in kernels.values())
    print(f"{len(kernels)} kernels, {total:.1f} s of compiling, written to {SIGNATURES_FILE}")


def main(args):
    if args == ["--update"]:
        update()
    elif args == ["--record"]:
        _record()
    elif args and not any(a.startswith("-") for a in args):
        _compile(args)
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
