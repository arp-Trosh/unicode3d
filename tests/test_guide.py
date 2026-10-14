# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
"""The code in the guide (unicode3d/starter/making-a-game.md), as written: its example game played, and its example tests run."""
import os
import re
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GUIDE = os.path.join(ROOT, "unicode3d", "starter", "making-a-game.md")


class GuideTests(unittest.TestCase):
    def setUp(self):
        with open(GUIDE, encoding="utf-8") as f:
            self.game, self.tests, self.crawl = re.findall(r"```python\n(.*?)```", f.read(), re.S)
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = folder.name
        os.makedirs(os.path.join(self.folder, "tests"))
        for name, text in [("game.py", self.game), ("tests/__init__.py", ""), ("tests/test_game.py", self.tests),
                           ("crawl.py", self.crawl)]:
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


    def test_the_first_person_recipe_walks_and_stops_at_walls(self):
        # From the start, at the end of a corridor facing -z: walk to its far wall, turn left, walk to the next.
        done = self.python("-c", "from crawl import Crawl\n"
                                 "from unicode3d.playtest import Playtest\n"
                                 "game = Crawl()\n"
                                 "test = Playtest(game, size=(48, 160), out='playtest')\n"
                                 "test.hold('w', frames=60); print('A', round(game.x, 2), round(game.z, 2))\n"
                                 "test.hold('left', frames=24); test.hold('w', frames=60)\n"
                                 "print('B', round(game.x, 2), round(game.z, 2))\n"
                                 "report = test.report(); print('W', report.warnings, report.blank_frames)\n")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        a, b = (line.split()[1:] for line in done.stdout.splitlines()[:2])
        self.assertEqual(a, ["8.0", "0.9"])                      # stopped RADIUS short of the wall at z = 0.5
        self.assertAlmostEqual(float(b[0]), 5.9, delta=0.02)     # and of the one at x = 5.5, sliding along
        self.assertIn("W [] []", done.stdout)


if __name__ == "__main__":
    unittest.main()
