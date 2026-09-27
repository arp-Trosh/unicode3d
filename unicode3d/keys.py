# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
"""Keyboard and mouse input, decoded from the VT escape sequences terminals send.

Every supported terminal (and the Windows console, see console.py) delivers
input as text in which special keys and mouse clicks are escape sequences.
InputDecoder turns that text into key codes, KeyReleases and MouseEvents.

Keys come out as ints: the character's code for ordinary keys (so `ord("q")`,
13 for Enter, 9 for Tab, 27 for Esc, 127 or 8 for Backspace) and Key values for
the rest. A held key repeats: the terminal sends it again and again after a
short pause.

Most terminals send only presses. Terminals that speak the kitty keyboard
protocol (kitty, foot, Ghostty, WezTerm with enable_kitty_keyboard, Alacritty,
iTerm2, Rio) also report releases once a program asks for them (see
KITTY_KEYBOARD_ON), which HeldKeys uses to tell exactly which keys are down; the
Windows console reports them too (see console.py).
"""
from dataclasses import dataclass
from enum import IntEnum

ESC_TIMEOUT = 0.03  # a lone Esc is only reported once this long passes without a sequence following it

# Kitty keyboard protocol flags: disambiguate (1), report event types, i.e. repeats and releases (2), report
# every key as an escape code, which releases of plain letters need (8), and the text a key types (16).
KITTY_KEYBOARD_ON = "\x1b[>27u"   # push the flags onto the terminal's stack
KITTY_KEYBOARD_OFF = "\x1b[<u"    # pop them, restoring whatever was there before


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
class KeyRelease:
    """A key let go. `key` is the code its press came as (a character's code, or a Key)."""
    key: int


@dataclass(frozen=True)
class MouseEvent:
    """A mouse button press or release (or wheel step) at cell (x, y), counted from 0, or a move.

    button: 0 left, 1 middle, 2 right, 3 none (a move with no button held), 64 wheel up, 65 wheel down.
    moved: the pointer moved, with `button` held down (pressed) or with none held (button NONE).
    Moves are only reported when the program asks for them (run(mouse="drag") or "move").
    """
    x: int
    y: int
    button: int
    pressed: bool
    moved: bool = False

    LEFT = 0
    MIDDLE = 1
    RIGHT = 2
    NONE = 3
    WHEEL_UP = 64
    WHEEL_DOWN = 65


_CSI_LETTER = {"A": Key.UP, "B": Key.DOWN, "C": Key.RIGHT, "D": Key.LEFT, "H": Key.HOME, "F": Key.END,
               "Z": Key.BACK_TAB, "P": Key.F1, "Q": Key.F2, "R": Key.F3, "S": Key.F4}
_CSI_TILDE = {1: Key.HOME, 2: Key.INSERT, 3: Key.DELETE, 4: Key.END, 5: Key.PAGE_UP, 6: Key.PAGE_DOWN,
              7: Key.HOME, 8: Key.END, 11: Key.F1, 12: Key.F2, 13: Key.F3, 14: Key.F4, 15: Key.F5,
              17: Key.F6, 18: Key.F7, 19: Key.F8, 20: Key.F9, 21: Key.F10, 23: Key.F11, 24: Key.F12}
# Kitty protocol codes (CSI code u) of keys that are not characters. Keypad keys come as what they type or do;
# modifier and lock keys on their own (57358-57363, 57441-57452) are not reported.
_KITTY_KEYS = {27: Key.ESC, 13: Key.ENTER, 9: Key.TAB, 127: Key.BACKSPACE, 57414: Key.ENTER,
               57409: ord("."), 57410: ord("/"), 57411: ord("*"), 57412: ord("-"), 57413: ord("+"), 57415: ord("="),
               57417: Key.LEFT, 57418: Key.RIGHT, 57419: Key.UP, 57420: Key.DOWN, 57421: Key.PAGE_UP,
               57422: Key.PAGE_DOWN, 57423: Key.HOME, 57424: Key.END, 57425: Key.INSERT, 57426: Key.DELETE,
               **{57399 + d: ord("0") + d for d in range(10)}}
KITTY_SHIFT, KITTY_ALT, KITTY_CTRL = 1, 2, 4  # modifier bits (the protocol sends 1 + the bits)
PRESS, REPEAT, RELEASE = 1, 2, 3            # kitty event types


class InputDecoder:
    """Incremental decoder: feed it text as it arrives, get keys and mouse events back."""

    def __init__(self):
        self.pending = ""
        self.pending_since = 0.0
        self.kitty = False  # the terminal has sent kitty keyboard protocol events, so releases are reported
        self._typed = {}    # kitty key code -> what its press came as, so its release names the same key

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
            if isinstance(event, list):
                events += event
            elif event is not None:
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

    def _escape(self, buf, i):
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
        if params[:1] in ("?", ">", "<", "="):  # replies to queries, not keys
            return j + 1, None
        fields = [f.split(":") for f in params.split(";")]
        try:
            mods = int(fields[1][0] or 1) - 1 if len(fields) > 1 else 0
            kind = int(fields[1][1]) if len(fields) > 1 and len(fields[1]) > 1 else PRESS
            if final == "u":
                self.kitty = True
                code = int(fields[0][0])
                text = "".join(chr(int(c)) for c in fields[2] if c) if len(fields) > 2 else ""
                return j + 1, self._kitty_key(code, mods, kind, text)
            key = _CSI_TILDE.get(int(fields[0][0])) if final == "~" else _CSI_LETTER.get(final)
        except ValueError:
            return j + 1, None
        if key is None:
            return j + 1, None
        if kind != PRESS:  # only terminals speaking the kitty protocol send repeats and releases this way
            self.kitty = True
        return j + 1, KeyRelease(key) if kind == RELEASE else key

    def _kitty_key(self, code, mods, kind, text):
        """The events for a kitty protocol key (CSI code ; modifiers:event ; text u)."""
        if code in _KITTY_KEYS:
            typed = [_KITTY_KEYS[code]]
        elif 57344 <= code <= 63743:  # other keys in the private use area: modifiers, locks, media keys, F13+
            return None
        elif mods & KITTY_CTRL and ord("a") <= code <= ord("z"):
            typed = [code & 0x1F]  # Ctrl+letter, as the control character a legacy terminal sends
        elif text:
            typed = [ord(c) for c in text]
        elif mods & KITTY_SHIFT and ord("a") <= code <= ord("z"):
            typed = [code - 32]
        else:
            typed = [code]
        if kind == RELEASE:
            return KeyRelease(self._typed.pop(code, typed[0]))
        self._typed[code] = typed[0]
        return typed


def _sgr_mouse(params, pressed):
    try:
        b, x, y = (int(p) for p in params.split(";"))
    except ValueError:
        return None
    if b & 32:  # a move: with a button held (drag), or with none (3)
        return MouseEvent(x - 1, y - 1, b & 3, b & 3 != MouseEvent.NONE, True)
    button = 64 + (b & 1) if b & 64 else b & 3
    return MouseEvent(x - 1, y - 1, button, pressed or button >= 64)


class HeldKeys:
    """Which keys are held down, for games that move while a key is held (see Screen.held).

    Where the terminal reports releases (`exact`: the kitty keyboard protocol, or
    the Windows console), a key is down from its press to its release. Elsewhere
    there are only presses and the auto-repeats of a held key, which start after
    a pause of about half a second, so a key counts as down for `first_hold`
    seconds after its press and, once repeats come, until they stop for longer
    than 1.5 of their intervals. That is only an estimate: a tap moves a player
    for up to `first_hold`, and on many systems holding a second key stops the
    first one repeating, so it looks released.

    Letters are tracked without case: `"w" in held` is true while W is held, with or without Shift.
    """

    def __init__(self, first_hold=0.5, min_repeat_hold=0.06):
        self.first_hold, self.min_repeat_hold = first_hold, min_repeat_hold
        self.exact = False  # releases are reported (set by Screen when the input source reports them)
        self._down = {}     # key -> [time of the press, time of the latest press or repeat, interval between them]
        self._now = 0.0

    @staticmethod
    def _norm(key):
        if isinstance(key, str):
            key = ord(key)
        return key + 32 if ord("A") <= key <= ord("Z") else key

    def update(self, events, now):
        """Take in a frame's events (ints, KeyReleases, anything else is ignored) at time `now`, in seconds."""
        self._now = now
        for e in events:
            if isinstance(e, KeyRelease):
                self._down.pop(self._norm(e.key), None)
            elif isinstance(e, int) and not isinstance(e, bool):
                k = self._norm(e)
                d = self._down.get(k)
                if d is None or not self._held(d):
                    self._down[k] = [now, now, None]
                else:
                    d[2], d[1] = now - d[1], now
        if not self.exact:
            for k in [k for k, d in self._down.items() if not self._held(d)]:
                del self._down[k]

    def _held(self, d):
        if self.exact:
            return True
        if d[2] is None:
            return self._now - d[1] < self.first_hold
        return self._now - d[1] < max(1.5 * d[2], self.min_repeat_hold)

    def __contains__(self, key):
        d = self._down.get(self._norm(key))
        return d is not None and self._held(d)

    def keys(self):
        """The keys held now (letters in lower case)."""
        return [k for k, d in self._down.items() if self._held(d)]

    def release_all(self):
        """Forget every held key (e.g. when the game loses track of the keyboard)."""
        self._down.clear()
