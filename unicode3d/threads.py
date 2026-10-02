# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
"""Drawing from more than one thread at once.

Numba runs parallel kernels on one of three threading layers: TBB, OpenMP, or its own workqueue, which it falls back
to where neither of the others is installed (often so on macOS and Windows). The workqueue can't take parallel
kernels launched from two threads at once: it aborts the whole process. kernel_lock() takes turns where that is the
layer, so that a program drawing from several threads (a server drawing for each client, a render thread beside
the one writing to the terminal) is safe everywhere; on the other layers the threads run side by side.
"""
import contextlib
import threading

import numba

_LOCK = threading.RLock()
_FREE = contextlib.nullcontext()


def kernel_lock():
    """What to hold (`with kernel_lock(): ...`) while running parallel kernels: a lock shared by every thread
    where Numba's threading layer is the workqueue, or not chosen yet (before any parallel kernel has run), and
    nothing elsewhere. A thread may take it again while it holds it."""
    try:
        layer = numba.threading_layer()
    except ValueError:  # none chosen yet
        return _LOCK
    return _LOCK if layer == "workqueue" else _FREE
