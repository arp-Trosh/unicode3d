# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
"""Numba's cache and helpers in other modules: kernel_cache.refresh() recompiles kernels whose helpers changed."""
import os
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest

import unicode3d
from unicode3d import kernel_cache

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(unicode3d.__file__)))

HELPERS = """\
from numba import njit


@njit(cache=True, error_model="numpy")
def helper(x):
    return x * {factor}
"""

KERNELS = """\
from numba import njit

from .helpers import helper
from .settings import OFFSET


@njit(cache=True, error_model="numpy")
def kernel(x):
    return helper(x) + OFFSET
"""


class KernelCacheTests(unittest.TestCase):
    def make_package(self, folder, name, refresh):
        """A package whose kernel (in kernels.py) calls a helper in another module (helpers.py)."""
        path = os.path.join(folder, name)
        os.makedirs(path)
        init = "from . import helpers, kernels\n"
        if refresh:
            init += "from unicode3d.kernel_cache import refresh\nrefresh(__name__)\n"
        with open(os.path.join(path, "__init__.py"), "w") as f:
            f.write(init)
        with open(os.path.join(path, "kernels.py"), "w") as f:
            f.write(KERNELS)
        self.write(path, "helpers.py", HELPERS.format(factor=2))
        self.write(path, "settings.py", "OFFSET = 1\n")
        return path

    @staticmethod
    def write(path, name, text):
        file = os.path.join(path, name)
        old = os.stat(file).st_mtime_ns if os.path.exists(file) else 0
        with open(file, "w") as f:
            f.write(text)
        if os.stat(file).st_mtime_ns == old:  # a filesystem with coarse times: make sure Numba sees a change
            os.utime(file, ns=(old + 10 ** 9, old + 10 ** 9))

    @staticmethod
    def run_kernel(folder, name):
        # No .pyc files: Python checks those by the second and the size, and the edits here come faster than that.
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([folder, ROOT]), NUMBA_CACHE_DIR=os.path.join(folder, "cache"),
                   PYTHONDONTWRITEBYTECODE="1")
        out = subprocess.run([sys.executable, "-c", f"import {name}; print({name}.kernels.kernel(10))"], env=env, check=True,
                             capture_output=True, text=True, timeout=300)
        return out.stdout.strip()

    def test_a_changed_helper_recompiles_the_kernels_calling_it(self):
        with tempfile.TemporaryDirectory() as folder:
            fixed = self.make_package(folder, "fixed_pkg", refresh=True)
            stale = self.make_package(folder, "stale_pkg", refresh=False)
            self.assertEqual(self.run_kernel(folder, "fixed_pkg"), "21")
            self.assertEqual(self.run_kernel(folder, "stale_pkg"), "21")
            time.sleep(0.01)
            for path in (fixed, stale):
                self.write(path, "helpers.py", HELPERS.format(factor=3))
            # Without refresh(), Numba keeps the kernel compiled with the old helper: the bug this guards against.
            self.assertEqual(self.run_kernel(folder, "stale_pkg"), "21")
            self.assertEqual(self.run_kernel(folder, "fixed_pkg"), "31")
            # So does a number the kernel imports from a module without kernels (Numba compiles its value in).
            self.write(fixed, "settings.py", "OFFSET = 5\n")
            self.assertEqual(self.run_kernel(folder, "fixed_pkg"), "35")
            self.assertEqual(self.run_kernel(folder, "fixed_pkg"), "35")

    def test_fingerprints_follow_the_helpers(self):
        prints = kernel_cache.fingerprints("unicode3d")
        # Kernels that call helpers elsewhere, and those that call none, are all fingerprinted.
        for name in ("shading", "raster", "texture", "background", "terminal"):
            self.assertIn(f"unicode3d.{name}", prints)
        self.assertEqual(kernel_cache.fingerprints("unicode3d"), prints)  # the same each time
        self.assertEqual(kernel_cache.refresh(), [])  # imported unicode3d refreshed already: nothing to drop
        uses = kernel_cache._uses(sys.modules["unicode3d.shading"], "unicode3d")
        self.assertTrue({"unicode3d.texture", "unicode3d.raster"} <= uses)


if __name__ == "__main__":
    unittest.main()
