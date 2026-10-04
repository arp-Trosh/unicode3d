# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
"""Drawing from more than one thread at once.

Numba runs parallel kernels on one of three threading layers: TBB, OpenMP, or its own workqueue, which it falls back
to where neither of the others is installed (often so on macOS and Windows). The workqueue can't take parallel
kernels launched from two threads at once: it aborts the whole process. kernel_lock() takes turns where that is the
layer, so that a program drawing from several threads (a server drawing for each client, a render thread beside
the one writing to the terminal) is safe everywhere; on the other layers the threads run side by side.

Between kernels a frame runs Python, and OpenMP's and TBB's workers keep spinning through it (OpenMP for about 5 ms
after each of a frame's ~20 kernels), which keeps every core busy, holds a laptop's chip at its power limit and
takes CPU from the terminal. prefer_sleeping_workers() has them sleep instead.
"""
import contextlib
import os
import threading

import numba

_LOCK = threading.RLock()
_FREE = contextlib.nullcontext()


def prefer_sleeping_workers():
    """Have Numba's workers sleep between kernels rather than spin: OpenMP told to wait passively (and Intel's or
    LLVM's OpenMP to stop spinning at once), and the OpenMP layer preferred over TBB, whose workers can't be told.
    Measured on Castle Panic at 4 cores / 8 threads: 6% faster than TBB, 3.4 logical CPUs busy instead of 4.9 (TBB)
    or 6.4 (spinning OpenMP). Run when unicode3d is imported, before any parallel kernel: Numba picks its layer and
    OpenMP reads its settings at the first parallel launch. Anything the user set is left alone, and so is the
    layer on Windows, where whether MSVC's OpenMP (vcomp140) waits passively is not known."""
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
    os.environ.setdefault("KMP_BLOCKTIME", "0")
    if os.name == "nt" or "NUMBA_THREADING_LAYER" in os.environ or "NUMBA_THREADING_LAYER_PRIORITY" in os.environ:
        return
    # In the environment rather than numba.config alone: Numba reloads its config from the environment when that
    # changes, and the precompile workers (and other Pythons started from here) should choose alike.
    os.environ["NUMBA_THREADING_LAYER_PRIORITY"] = "omp tbb workqueue"
    numba.config.reload_config()


def kernel_lock():
    """What to hold (`with kernel_lock(): ...`) while running parallel kernels: a lock shared by every thread
    where Numba's threading layer is the workqueue, or not chosen yet (before any parallel kernel has run), and
    nothing elsewhere. A thread may take it again while it holds it."""
    try:
        layer = numba.threading_layer()
    except ValueError:  # none chosen yet
        return _LOCK
    return _LOCK if layer == "workqueue" else _FREE
