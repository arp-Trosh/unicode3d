# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
"""python -m unicode3d.new: a game made from the starter passes its own tests and plays cleanly, the guide's links
work from the game, files are never overwritten, and the starter ships with the package."""
import os
import re
import subprocess
import sys
import tempfile
import unittest

from unicode3d import __version__, new

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def python(*args, cwd):
    env = dict(os.environ, PYTHONPATH=ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""))
    return subprocess.run([sys.executable, *args], cwd=cwd, env=env, capture_output=True, text=True, timeout=300)


class NewGameTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = folder.name
        self.game = os.path.join(self.folder, "gem run")

    def read(self, path):
        with open(os.path.join(self.game, path), encoding="utf-8") as f:
            return f.read()

    def test_a_new_game_passes_its_tests_and_plays_cleanly(self):
        done = python("-m", "unicode3d.new", self.game, cwd=self.folder)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        for path, _ in new.FILES:
            self.assertTrue(os.path.isfile(os.path.join(self.game, path)), path)
        self.assertIn('TITLE = "gem run"', self.read("game.py"))
        self.assertTrue(self.read("CLAUDE.md").startswith("# gem run: notes"))
        self.assertIn(f"@v{__version__}", self.read("requirements.txt"))
        self.assertNotIn("{{version}}", self.read("CLAUDE.md") + self.read("requirements.txt"))
        self.assertNotIn("GAME NAME", self.read("game.py") + self.read("CLAUDE.md"))

        tests = python("-m", "unittest", "-v", cwd=self.game)
        self.assertEqual(tests.returncode, 0, tests.stdout + tests.stderr)
        self.assertIn("OK", tests.stderr)

        # The playtest that the game's CLAUDE.md and new's own output suggest.
        script = re.search(r'play game\.py:Game "([^"]+)"', self.read("CLAUDE.md")).group(1)
        self.assertIn(script, done.stdout)
        play = python("-m", "unicode3d.playtest", "play", "game.py:Game", script, cwd=self.game)
        self.assertEqual(play.returncode, 0, play.stdout + play.stderr)
        self.assertIn("quit: frame() returned False", play.stdout)
        self.assertNotIn("WARNING", play.stdout)
        self.assertNotIn("BLANK", play.stdout)
        self.assertTrue(os.path.isfile(os.path.join(self.game, "playtest", "play.png")))

    def test_the_guides_links_work_from_the_game(self):
        new.make_game(self.game)
        guide = self.read(new.GUIDE)
        self.assertIn(f"the guide for unicode3d {__version__}", guide)
        links = re.findall(r"\]\(([^)\s]+)\)", guide)
        self.assertTrue(links)
        prefix = f"{new.REPO}/blob/v{__version__}/"
        for link in links:
            if link.startswith("#") or link.startswith("https://") and not link.startswith(new.REPO):
                continue
            if link == "../CLAUDE.md":
                continue
            self.assertTrue(link.startswith(prefix), link)
            path = link[len(prefix):].split("#")[0]
            self.assertTrue(os.path.isfile(os.path.join(ROOT, path)), f"{link}: no {path} in the repository")

    def test_never_overwrites(self):
        os.makedirs(os.path.join(self.game, ".venv"))      # where unicode3d went: fine
        with open(os.path.join(self.game, "game.py"), "w") as f:
            f.write("mine")
        done = python("-m", "unicode3d.new", self.game, cwd=self.folder)
        self.assertEqual(done.returncode, 2)
        self.assertIn("game.py", done.stderr)
        self.assertIn("nothing was written", done.stderr)
        self.assertEqual(self.read("game.py"), "mine")
        self.assertEqual(sorted(os.listdir(self.game)), [".venv", "game.py"])

    def test_guide_refreshes_only_the_guide(self):
        new.make_game(self.game)
        with open(os.path.join(self.game, "game.py"), "w") as f:
            f.write("mine")
        with open(os.path.join(self.game, new.GUIDE), "w") as f:
            f.write("an old guide")
        done = python("-m", "unicode3d.new", "--guide", ".", cwd=self.game)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertTrue(self.read(new.GUIDE).startswith("# Making a game with unicode3d"))
        self.assertEqual(self.read("game.py"), "mine")

    def test_names_that_would_break_game_py_are_refused(self):
        for name in ['say "hi"', "back\\slash", "new\nline"]:
            with self.assertRaises(ValueError):
                new.make_game(self.game, name=name)
        self.assertFalse(os.path.exists(self.game))

    def test_the_starter_ships_with_the_package(self):
        try:
            import tomllib
        except ImportError:
            self.skipTest("tomllib is Python 3.11 and later")
        from fnmatch import fnmatch
        with open(os.path.join(ROOT, "pyproject.toml"), "rb") as f:
            patterns = tomllib.load(f)["tool"]["setuptools"]["package-data"]["unicode3d.starter"]
        for _, source in new.FILES:
            if source is not None and not source.endswith(".py"):   # .py files go in as modules
                self.assertTrue(any(fnmatch(source, p) for p in patterns), f"{source} isn't package data")
        self.assertFalse(os.path.exists(os.path.join(new.STARTER, "__init__.py")),
                         "starter/ has no __init__.py, so unittest discovery in the engine skips its test_game.py")


if __name__ == "__main__":
    unittest.main()
