# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
"""The code in docs/making-a-game.md, as written: its example game played, and its example tests run."""
import os
import re
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GUIDE = os.path.join(ROOT, "docs", "making-a-game.md")


class GuideTests(unittest.TestCase):
    def setUp(self):
        with open(GUIDE, encoding="utf-8") as f:
            self.game, self.tests = re.findall(r"```python\n(.*?)```", f.read(), re.S)
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = folder.name
        os.makedirs(os.path.join(self.folder, "tests"))
        for name, text in [("game.py", self.game), ("tests/__init__.py", ""), ("tests/test_game.py", self.tests)]:
            with open(os.path.join(self.folder, name), "w", encoding="utf-8") as f:
                f.write(text)

    def python(self, *args):
        env = dict(os.environ, PYTHONPATH=ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""))
        return subprocess.run([sys.executable, *args], cwd=self.folder, env=env, capture_output=True, text=True,
                              timeout=300)

    def test_the_example_game_plays(self):
        done = self.python("-m", "unicode3d.playtest", "play", "game.py:Game",
                           "wait 5; shot start; hold left 30; expect 'heading +2.00'; press q")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("quit: frame() returned False", done.stdout)
        self.assertNotIn("WARNING", done.stdout)
        self.assertNotIn("BLANK", done.stdout)

    def test_the_example_tests_pass(self):
        done = self.python("-m", "unittest", "-v")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("Ran 3 tests", done.stderr)


if __name__ == "__main__":
    unittest.main()
