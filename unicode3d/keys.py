# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
"""Keyboard and mouse input, decoded from the VT escape sequences terminals send.

Every supported terminal (and the Windows console, see console.py) delivers
input as text in which special keys and mouse clicks are escape sequences.
InputDecoder turns that text into key codes and MouseEvents.

Keys come out as ints: the character's code for ordinary keys (so `ord("q")`,
13 for Enter, 9 for Tab, 27 for Esc, 127 or 8 for Backspace) and Key values for
the rest.
"""
from dataclasses import dataclass
from enum import IntEnum

ESC_TIMEOUT = 0.03  # a lone Esc is only reported once this long passes without a sequence following it


class Key(IntEnum):
    """Codes for keys that are not characters. The numbers match curses' KEY_* constants."""
    TAB = 9
    ENTER = 13
    ESC = 27
    BACKSPACE = 127
    DOWN = 258
    UP = 259
    LEFT = 260
    RIGHT = 261
    HOME = 262
    F1 = 265
    F2 = 266
    F3 = 267
    F4 = 268
    F5 = 269
    F6 = 270
    F7 = 271
    F8 = 272
    F9 = 273
    F10 = 274
    F11 = 275
    F12 = 276
    DELETE = 330
    INSERT = 331
    PAGE_DOWN = 338
    PAGE_UP = 339
    BACK_TAB = 353
    END = 360


@dataclass(frozen=True)
class MouseEvent:
    """A mouse button press or release (or wheel step) at cell (x, y), counted from 0.

    button: 0 left, 1 middle, 2 right, 64 wheel up, 65 wheel down.
    """
    x: int
    y: int
    button: int
    pressed: bool

    LEFT = 0
    MIDDLE = 1
    RIGHT = 2
    WHEEL_UP = 64
    WHEEL_DOWN = 65


_CSI_LETTER = {"A": Key.UP, "B": Key.DOWN, "C": Key.RIGHT, "D": Key.LEFT, "H": Key.HOME, "F": Key.END,
               "Z": Key.BACK_TAB, "P": Key.F1, "Q": Key.F2, "R": Key.F3, "S": Key.F4}
_CSI_TILDE = {1: Key.HOME, 2: Key.INSERT, 3: Key.DELETE, 4: Key.END, 5: Key.PAGE_UP, 6: Key.PAGE_DOWN,
              7: Key.HOME, 8: Key.END, 11: Key.F1, 12: Key.F2, 13: Key.F3, 14: Key.F4, 15: Key.F5,
              17: Key.F6, 18: Key.F7, 19: Key.F8, 20: Key.F9, 21: Key.F10, 23: Key.F11, 24: Key.F12}


class InputDecoder:
    """Incremental decoder: feed it text as it arrives, get keys and mouse events back."""

    def __init__(self):
        self.pending = ""
        self.pending_since = 0.0

    def feed(self, text, now):
        """Decode text received at time `now` (seconds). A trailing partial sequence is held back."""
        if not self.pending:
            self.pending_since = now
        self.pending += text
        events, i, buf = [], 0, self.pending
        while i < len(buf):
            if buf[i] != "\x1b":
                events.append(ord(buf[i]))
                i += 1
                continue
            end, event = self._escape(buf, i)
            if end is None:  # incomplete: wait for the rest
                break
            if event is not None:
                events.append(event)
            i = end
        self.pending = buf[i:]
        if self.pending and i:
            self.pending_since = now
        return events

    def flush(self, now):
        """Give up on a partial sequence that has waited too long: a lone Esc is the Esc key."""
        if self.pending and now - self.pending_since >= ESC_TIMEOUT:
            events = [Key.ESC] + [ord(c) for c in self.pending[1:]]
            self.pending = ""
            return events
        return []

    @staticmethod
    def _escape(buf, i):
        """(index after the sequence starting at buf[i], event or None); index None if incomplete."""
        if i + 1 >= len(buf):
            return None, None
        kind = buf[i + 1]
        if kind == "O":  # SS3: arrows in application mode, F1-F4
            if i + 2 >= len(buf):
                return None, None
            return i + 3, _CSI_LETTER.get(buf[i + 2])
        if kind != "[":
            return i + 1, Key.ESC  # Esc then another key (Alt+key): the key follows as itself
        j = i + 2
        while j < len(buf) and not "\x40" <= buf[j] <= "\x7e":  # parameter bytes, up to the final byte
            j += 1
        if j >= len(buf):
            return None, None
        params, final = buf[i + 2:j], buf[j]
        if params.startswith("<") and final in "Mm":
            return j + 1, _sgr_mouse(params[1:], final == "M")
        if final == "~":
            try:
                return j + 1, _CSI_TILDE.get(int(params.split(";")[0]))
            except ValueError:
                return j + 1, None
        return j + 1, _CSI_LETTER.get(final)


def _sgr_mouse(params, pressed):
    try:
        b, x, y = (int(p) for p in params.split(";"))
    except ValueError:
        return None
    if b & 32:  # motion while a button is held: not reported
        return None
    button = 64 + (b & 1) if b & 64 else b & 3
    return MouseEvent(x - 1, y - 1, button, pressed or button >= 64)
