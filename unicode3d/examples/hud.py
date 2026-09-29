# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
"""The status line the demos share: help text on the left, display settings on the right.

The display settings (glyphs F2, colours F3, frame rate F4, shadows F5, reflections
F6) are ui.DisplayControls, so they work by key or by click on any screen of any demo.
"""
from ..ui import DisplayControls


class StatusBar:
    def __init__(self, renderer=None):
        self.controls = DisplayControls(renderer=renderer)

    def handle(self, events, screen):
        """Act on the display keys and clicks; returns the other events."""
        return self.controls.handle(events, screen)

    def draw(self, screen, text="", row=None):
        """Draw the bar on `row` (default: the bottom one): `text` on the left, cut short before the controls."""
        rows, cols = screen.size()
        y = rows - 1 if row is None else row
        x = max(cols - self.controls.width - 1, 0)
        screen.text(y, 0, " " * cols)
        screen.text(y, 1, text[:max(x - 3, 0)])
        self.controls.draw(screen, y, x)
