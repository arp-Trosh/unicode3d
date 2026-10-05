# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
"""The status line the demos share: help text on the left, display settings on the right.

The display settings (glyphs F2, colours F3, frame rate F4, shadows F5, reflections F6,
detail F7) are ui.DisplayControls, so they work by key or by click on any screen of any demo.
"""
from ..ui import DisplayControls


HELP_ROOM = 40  # columns of help text kept before detail (F7) is left out of the bar


class StatusBar:
    def __init__(self, renderer=None):
        self.wide = DisplayControls(renderer=renderer)
        # Where all of them would leave too little room for the help text: all but detail drawn, F7 still
        # switching it. (Both read and write the same screen and renderer, so they always agree.)
        self.narrow = DisplayControls(renderer=renderer, show=[n for n in DisplayControls.SETTINGS if n != "detail"])
        self.controls = self.wide

    def handle(self, events, screen):
        """Act on the display keys and clicks; returns the other events."""
        return self.controls.handle(events, screen)

    def draw(self, screen, text="", row=None):
        """Draw the bar on `row` (default: the bottom one): `text` on the left, cut short before the controls."""
        rows, cols = screen.size()
        room = cols - self.wide.width - 4
        self.controls = self.wide if room >= min(len(text), HELP_ROOM) else self.narrow
        y = rows - 1 if row is None else row
        x = max(cols - self.controls.width - 1, 0)
        screen.text(y, 0, " " * cols)
        screen.text(y, 1, text[:max(x - 3, 0)])
        self.controls.draw(screen, y, x)
