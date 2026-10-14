# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
"""A new game from the starter: a small working game to grow, with its tests, and what an agent needs to work on it.

    python -m unicode3d.new mygame

writes into the folder mygame (made if missing):

    game.py                 the game: title, play and game-over screens, a camera that follows the player, held keys,
                            buttons, a label in the scene and a status line (main() calls run())
    tests/test_game.py      tests that play it off-screen with the playtest (python -m unittest)
    CLAUDE.md               how an agent runs, checks and tests the game
    docs/making-a-game.md   the guide to building a game on unicode3d, for this version
    requirements.txt        unicode3d pinned to this version's tag
    .gitignore              .venv/, __pycache__/ and playtest/ (the playtests' pictures)

It never overwrites a file: if any of these is already there it writes nothing and says which. The folder may
already hold other things, such as the .venv unicode3d was installed into. After upgrading unicode3d,
python -m unicode3d.new --guide mygame brings docs/making-a-game.md up to date (and only that).
"""
import argparse
import os
import posixpath
import re
import sys

from . import __version__

REPO = "https://github.com/arp-Trosh/unicode3d"
STARTER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "starter")
STARTER_IN_REPO = "unicode3d/starter"   # where the starter's files are in unicode3d's repository, for the guide's links
GUIDE = "docs/making-a-game.md"
FILES = [  # (in the new game, from the starter: None for an empty file)
    ("game.py", "game.py"),
    ("tests/__init__.py", None),
    ("tests/test_game.py", "test_game.py"),
    ("CLAUDE.md", "game-CLAUDE.md"),
    (GUIDE, "making-a-game.md"),
    ("requirements.txt", "requirements.txt"),
    (".gitignore", "gitignore"),
]
PLACEHOLDER = "GAME NAME"


def _link(match, version):
    """A relative link in the guide, made to work from the game's docs/ folder: to the game's own CLAUDE.md, or to
    the file in unicode3d's repository at this version's tag."""
    target, anchor = match.group(1), match.group(2) or ""
    path = posixpath.normpath(posixpath.join(STARTER_IN_REPO, target))
    if path == f"{STARTER_IN_REPO}/game-CLAUDE.md":
        return f"](../CLAUDE.md{anchor})"
    return f"]({REPO}/blob/v{version}/{path}{anchor})"


def fill(name, source, text, version=__version__):
    """A starter file's text as it goes into a game called name."""
    if source == "making-a-game.md":
        text = re.sub(r"\]\(([^)#:\s]+)(#[^)\s]*)?\)", lambda m: _link(m, version), text)
        title, rest = text.split("\n", 1)
        return (f"{title}\n\n*This copy is the guide for unicode3d {version}, the version in `requirements.txt`. "
                f"After upgrading unicode3d, `python -m unicode3d.new --guide .` brings it up to date.*\n{rest}")
    return text.replace(PLACEHOLDER, name).replace("{{version}}", version)


def check_name(name):
    if not name or not name.isprintable() or any(c in name for c in "\"\\"):
        raise ValueError(f"{name!r} won't do as a game's name: it goes in game.py as a string, so no quotes, "
                         "backslashes or control characters (give one with --name)")
    return name


def make_game(folder, name=None, guide_only=False):
    """Write the starter into folder (made if missing) as a game called name (the folder's name if None); with
    guide_only, only docs/making-a-game.md, replacing it. Returns the paths written, relative to folder.
    Raises FileExistsError, before writing anything, if any of the files is already there (unless guide_only)."""
    folder = os.path.abspath(folder)
    name = check_name(name if name is not None else os.path.basename(folder))
    files = [(path, source) for path, source in FILES if not guide_only or path == GUIDE]
    there = [path for path, _ in files if os.path.lexists(os.path.join(folder, path))]
    if there and not guide_only:
        raise FileExistsError(f"{folder} already has {', '.join(there)}; nothing was written (unicode3d.new never "
                              "overwrites files: pick another folder, or move these out of the way)")
    for path, source in files:
        text = ""
        if source is not None:
            with open(os.path.join(STARTER, source), encoding="utf-8") as f:
                text = fill(name, source, f.read())
        target = os.path.join(folder, *path.split("/"))
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
    return [path for path, _ in files]


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m unicode3d.new", description=__doc__.split("\n\n")[0],
                                     epilog="See the module's docstring (python -m pydoc unicode3d.new) for the files "
                                            "it writes.")
    parser.add_argument("folder", help="where to put the game (made if missing; . for the current folder)")
    parser.add_argument("--name", help="the game's name, for its title and CLAUDE.md (default: the folder's name)")
    parser.add_argument("--guide", action="store_true",
                        help=f"write only {GUIDE}, replacing it with the guide for unicode3d {__version__}")
    args = parser.parse_args(argv)
    try:
        written = make_game(args.folder, args.name, args.guide)
    except (ValueError, FileExistsError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if args.guide:
        print(f"wrote {os.path.join(args.folder, GUIDE)}, the guide for unicode3d {__version__}")
        return 0
    shown = os.path.relpath(os.path.abspath(args.folder))
    print(f"made a game in {shown} (unicode3d {__version__}): {', '.join(written)}\n")
    print("next, in that folder:")
    print("  python -m unittest                    # play it off-screen: the tests in tests/")
    print('  python -m unicode3d.playtest play game.py:Game "wait 10; shot title; press enter; hold right 20; '
          'shot play; press q"')
    print("  python game.py                        # play it yourself, in a terminal")
    print("then read CLAUDE.md and docs/making-a-game.md, and make it your own game")
    return 0


if __name__ == "__main__":
    sys.exit(main())
