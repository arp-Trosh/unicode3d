# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
"""Keeping Numba's cache of compiled kernels in step with the helpers they call.

Numba keys each cached kernel on its own source file only. A kernel in shading.py compiles texture.sample into
itself, so after texture.py changes (an upgrade, or an edit while working on the engine) the cached shading
kernels would go on running the old sample, with no error. refresh() stops that: for each module with cached
kernels it keeps a fingerprint of everything compiled into them from elsewhere (the source of the modules whose
kernels they call, through any number of steps, and the numbers and arrays the module imports), in a file next to
the cache (`<module>.deps`). When the fingerprint differs, or there is none yet, that module's cache entries are
dropped, and its kernels compile afresh on first use (on several cores, through compile_kernels()).

unicode3d calls refresh() once its modules are imported, before any kernel runs. Whatever goes wrong here (a cache
that can't be written, a Numba that works differently) leaves the cache as it is.
"""
import hashlib
import os
import sys
import types

import numpy as np

STAMP_SUFFIX = ".deps"


def _dispatchers(module):
    """The kernels a module defines that Numba caches."""
    from numba.core.caching import NullCache
    from numba.core.registry import CPUDispatcher
    return [value for value in vars(module).values()
            if isinstance(value, CPUDispatcher) and getattr(value.py_func, "__module__", None) == module.__name__
            and not isinstance(getattr(value, "_cache", None), NullCache)]


def _uses(module, package):
    """The package's other modules whose kernels a module's code can call: those it imports kernels from, and the
    modules it imports whole."""
    from numba.core.registry import CPUDispatcher
    used = set()
    for value in vars(module).values():
        if isinstance(value, CPUDispatcher):
            name = getattr(value.py_func, "__module__", None)
        elif isinstance(value, types.ModuleType):
            name = value.__name__
        else:
            continue
        if name and name != module.__name__ and (name == package or name.startswith(package + ".")):
            used.add(name)
    return used


def _constants(module):
    """The module's numbers and arrays, as bytes: kernels compile in the values of the globals they read, imported
    ones included (which its own source doesn't show)."""
    parts = []
    for name, value in sorted(vars(module).items()):
        if isinstance(value, (bool, int, float, complex, np.number, np.bool_)):
            parts.append(f"{name}={value!r};".encode())
        elif isinstance(value, np.ndarray) and value.dtype != object:
            parts.append(f"{name}:{value.dtype.str}{value.shape};".encode())
            parts.append(np.ascontiguousarray(value).tobytes())
        elif isinstance(value, tuple) and all(isinstance(v, (bool, int, float)) for v in value):
            parts.append(f"{name}={value!r};".encode())
    return b"".join(parts)


def fingerprints(package):
    """{module name: hex digest} for each of the package's imported modules with cached kernels: a hash of the
    sources of the modules it uses (see _uses, followed through any number of steps; not its own, which Numba
    checks itself) and of its and their numbers and arrays."""
    modules = {name: module for name, module in list(sys.modules.items())
               if isinstance(module, types.ModuleType) and (name == package or name.startswith(package + "."))}
    uses = {name: _uses(module, package) for name, module in modules.items()}
    result = {}
    for name, module in modules.items():
        if not _dispatchers(module):
            continue
        seen, todo = set(), list(uses[name])
        while todo:
            other = todo.pop()
            if other in seen or other == name or other not in modules:
                continue
            seen.add(other)
            todo += uses[other]
        digest = hashlib.sha256(_constants(module))
        for other in sorted(seen):
            digest.update(f"\0{other}\0".encode())
            path = getattr(modules[other], "__file__", None)
            if path:
                with open(path, "rb") as f:
                    digest.update(f.read())
            digest.update(_constants(modules[other]))
        result[name] = digest.hexdigest()
    return result


def refresh(package=None):
    """Drop the cached kernels of each of the package's modules (unicode3d's by default) whose fingerprint changed
    since they were cached; returns the names of the modules whose kernels were dropped."""
    package = package or __name__.rpartition(".")[0]
    dropped = []
    try:
        prints = fingerprints(package)
    except Exception:
        return dropped
    for name, digest in prints.items():
        try:
            kernels = _dispatchers(sys.modules[name])
            stamp = os.path.join(kernels[0]._cache.cache_path, name.rpartition(".")[2] + STAMP_SUFFIX)
            try:
                with open(stamp, encoding="ascii") as f:
                    if f.read().strip() == digest:
                        continue
            except (OSError, ValueError):
                pass
            for kernel in kernels:
                kernel._cache.flush()
            os.makedirs(os.path.dirname(stamp), exist_ok=True)
            # Written whole, then put in place, so that another program starting meanwhile never reads half of it
            # (and drops the cache again, as it would for a changed fingerprint).
            temp = f"{stamp}.{os.getpid()}.tmp"
            with open(temp, "w", encoding="ascii") as f:
                f.write(digest + "\n")
            os.replace(temp, stamp)
            dropped.append(name)
        except Exception:
            continue
    return dropped
