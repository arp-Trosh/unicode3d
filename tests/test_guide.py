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


# What the guide calls an object by (screen.draw(...), solid.raycast(...)), the methods it names alone (Playtest's,
# listed after it, and others), and names in its code that aren't unicode3d's.
INSTANCES = {"screen": "Screen", "renderer": "Renderer", "solid": "Colliders", "mesh": "Mesh", "model": "Model",
             "anim": "Animation", "clip": "Clip", "panel": "Panel", "controls": "DisplayControls", "obj": "Object3D",
             "test": "Playtest"}
METHODS = {"emit": "Particles", "play": "Sprite", "push_out": "Colliders", "invalidate": "Renderer", **{name: "Playtest" for name in (
    "wait", "press", "type", "hold", "down", "up", "click", "click_text", "move", "resize", "step", "shot", "sheet",
    "text", "find", "expect", "expect_not", "picture", "report")}}
NOT_CALLS = {"ord", "chr", "print", "range", "len", "min", "max", "sin", "cos", "frame", "update", "Game", "Crawl"}


def _split_arguments(text):
    """The arguments of a call as written, split at the commas outside brackets and quotes."""
    parts, depth, quote, start = [], 0, None, 0
    for i, ch in enumerate(text):
        if quote:
            quote = None if ch == quote else quote
        elif ch in "\"'":
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(text[start:i].strip())
            start = i + 1
    parts.append(text[start:].strip())
    return [p for p in parts if p]


def _resolve(name):
    """The function or class a name in the guide means, or None."""
    import unicode3d
    from unicode3d import playtest, shapes, transforms
    owner, _, attr = name.rpartition(".")
    if not owner:
        if name in METHODS:
            return getattr(getattr(unicode3d, METHODS[name]), name)
        for module in (unicode3d, shapes, transforms, playtest):
            if hasattr(module, name):
                return getattr(module, name)
        return None
    cls = INSTANCES.get(owner)
    if cls is not None:
        holder = getattr(playtest, cls, None) or getattr(unicode3d, cls, None)
        if holder is None:
            from unicode3d import models
            holder = getattr(models, cls)
        return getattr(holder, attr, None)
    for module in (unicode3d, shapes, transforms, playtest):
        target = module
        for part in name.split("."):
            target = getattr(target, part, None)
        if target is not None:
            return target
    return None


def _defaults(target):
    """Each argument's default, with a dataclass's default_factory called (inspect shows those as <factory>)."""
    import dataclasses
    import inspect
    defaults = {name: p.default for name, p in inspect.signature(target).parameters.items()}
    if isinstance(target, type) and dataclasses.is_dataclass(target):
        for field in dataclasses.fields(target):
            if field.default_factory is not dataclasses.MISSING:
                defaults[field.name] = field.default_factory()
    return defaults


def signature_problems(markdown):
    """What the calls written in a markdown file's inline code (not its code blocks) get wrong about the functions
    they name: functions or arguments that don't exist, or too many arguments by position. A call written with three
    or more arguments, one at least with a default (`Camera(position, target, up=(0, 1, 0), fov=50)`), is read as a
    signature: its arguments by position must have the function's names, and its defaults the function's values
    (`...` for one says nothing). Returns a list of strings."""
    import inspect

    import numpy as np

    import unicode3d
    text = re.sub(r"```.*?```", "", markdown, flags=re.S)
    namespace = {**vars(unicode3d), "np": np}
    problems = []
    for code in re.findall(r"`([^`]+)`", text):
        code = " ".join(code.split())  # (inline code may wrap onto the next line)
        for call in re.finditer(r"(?<![\w.])([A-Za-z_][\w.]*)\(([^()]*(?:\([^()]*\)[^()]*)*)\)", code):
            name, arguments = call.group(1), call.group(2)
            if name in NOT_CALLS or name.split(".")[0] in ("np", "self", "os", "sys"):
                continue
            target = _resolve(name)
            if target is None:
                owner = name.rpartition(".")[0]
                if not owner or owner in INSTANCES or hasattr(unicode3d, owner.split(".")[0]):
                    problems.append(f"{call.group(0)}: no such function")
                continue  # (a method of the game's own, such as MOVES.get)
            try:
                sig = inspect.signature(target)
            except (TypeError, ValueError):
                continue
            params = [p for p in sig.parameters.values() if p.name not in ("self", "cls")]
            names = {p.name: p for p in params}
            defaults = _defaults(target)
            takes_any = any(p.kind == p.VAR_KEYWORD for p in params)
            written_arguments = _split_arguments(arguments)
            signature = len(written_arguments) >= 3 and any(re.match(r"\w+=", a) for a in written_arguments)
            index = 0
            for argument in written_arguments:
                if argument in ("...", "…") or argument.startswith("*"):
                    break
                key, eq, value = argument.partition("=")
                if eq and re.fullmatch(r"\w+", key.strip()) and not value.startswith("="):
                    key, value = key.strip(), value.strip()
                    if key not in names:
                        if not takes_any:
                            problems.append(f"{call.group(0)}: {name} has no argument {key!r}")
                        continue
                    default = defaults[key]
                    if not signature or default is inspect.Parameter.empty or "|" in value or "..." in value:
                        continue
                    try:
                        written = eval(value, namespace)
                    except Exception:
                        continue
                    try:
                        same = bool(np.all(np.asarray(written) == np.asarray(default))) and \
                            np.shape(written) == np.shape(default)
                    except Exception:
                        same = written == default
                    if not same:
                        problems.append(f"{call.group(0)}: {name}'s {key} defaults to {default!r}, not {value}")
                    continue
                positional = [p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
                if index >= len(positional):
                    if not any(p.kind == p.VAR_POSITIONAL for p in params):
                        problems.append(f"{call.group(0)}: {name} takes {len(positional)} arguments by position")
                    break
                if re.fullmatch(r"[a-z_]\w*", argument) and (signature or argument in names) \
                        and argument != positional[index].name:
                    problems.append(f"{call.group(0)}: {name}'s argument {index + 1} is {positional[index].name}, "
                                    f"not {argument}")
                index += 1
    return problems


class SignatureTests(unittest.TestCase):
    def test_the_guides_calls_match_the_functions(self):
        for path in (GUIDE, os.path.join(ROOT, "unicode3d", "starter", "game-CLAUDE.md")):
            with open(path, encoding="utf-8") as f:
                problems = signature_problems(f.read())
            self.assertEqual(problems, [], os.path.basename(path))
