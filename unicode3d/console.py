# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
"""Talking to the terminal directly, on Linux/macOS and on Windows.

Output is plain VT escape sequences, which Windows Terminal, the Windows 10+
console and every Unix terminal understand, so drawing is the same everywhere.
Only setup and input differ: termios raw mode on Unix, console modes and input
records on Windows. Nothing here needs curses, so there is nothing extra to
install on Windows.
"""
import codecs
import locale
import os
import select
import sys

from .keys import KITTY_KEYBOARD_OFF, KITTY_KEYBOARD_ON, Key

ENTER_SEQ = "\x1b[?1049h\x1b[?25l\x1b[?7l\x1b[2J"  # alternate screen, hide cursor, no auto-wrap, clear
EXIT_SEQ = "\x1b[0m\x1b[?7h\x1b[?25h\x1b[?1049l"
# Mouse reporting, in SGR encoding: clicks (1000), plus moves while a button is held (1002), or all moves (1003).
MOUSE_MODES = {True: "1000", "click": "1000", "drag": "1002", "move": "1003"}
TITLE_PUSH, TITLE_POP = "\x1b[22;0t", "\x1b[23;0t"  # save and restore the window title, where supported

TRUECOLOR_TERMS = ("kitty", "ghostty", "alacritty", "foot", "wezterm", "contour", "iterm", "rio")
TRUECOLOR_PROGRAMS = ("iTerm.app", "WezTerm", "vscode", "Hyper", "ghostty", "Tabby", "rio")
SEXTANT_TERMS = ("kitty", "ghostty", "foot", "wezterm")  # TERM values of terminals that draw sextants themselves
SEXTANT_PROGRAMS = ("WezTerm", "ghostty")
SEXTANT_VARS = ("KITTY_WINDOW_ID", "WEZTERM_PANE", "GHOSTTY_RESOURCES_DIR", "WT_SESSION")


def detect_color_mode(env=None, windows=None):
    """Best guess at the terminal's colour depth: truecolor, 256, 16 or mono.

    UNICODE3D_COLOR overrides it. Windows Terminal and the Windows 10+ console both
    do 24-bit colour; elsewhere COLORTERM, TERM and TERM_PROGRAM are the hints.
    """
    env = os.environ if env is None else env
    windows = os.name == "nt" if windows is None else windows
    forced = env.get("UNICODE3D_COLOR", "").lower()
    if forced in ("truecolor", "256", "16", "mono"):
        return forced
    if "NO_COLOR" in env:
        return "mono"
    term = env.get("TERM", "").lower()
    if env.get("COLORTERM", "").lower() in ("truecolor", "24bit") or "WT_SESSION" in env or windows:
        return "truecolor"
    if any(t in term for t in TRUECOLOR_TERMS) or env.get("TERM_PROGRAM", "") in TRUECOLOR_PROGRAMS:
        return "truecolor"
    if "256" in term:
        return "256"
    if term in ("dumb",):
        return "mono"
    return "16"


def shows_sextants(env=None):
    """Whether the terminal is one known to show sextants whatever font is set.

    A program can't ask a terminal which characters its font has (a missing one
    still takes a cell, drawn as a box), so this goes by the terminal: kitty,
    WezTerm, foot and Ghostty draw block symbols themselves, and Windows
    Terminal's own font, Cascadia, has them. Their variables survive tmux and
    WSL, and over ssh TERM still names the terminal, which draws the characters
    either way.
    """
    env = os.environ if env is None else env
    term = env.get("TERM", "").lower()
    return (any(t in term for t in SEXTANT_TERMS) or env.get("TERM_PROGRAM", "") in SEXTANT_PROGRAMS
            or any(v in env for v in SEXTANT_VARS))


def detect_glyphs(env=None, unicode_ok=True):
    """Best guess at the richest glyph set the terminal can show: sextant, quad, half or ascii.

    UNICODE3D_GLYPHS overrides it. Sextants are picked only where shows_sextants()
    vouches for them; elsewhere quadrants, which every font with half blocks has.
    """
    env = os.environ if env is None else env
    forced = env.get("UNICODE3D_GLYPHS", "").lower()
    if forced in ("half", "quad", "sextant", "ascii"):
        return forced
    if not unicode_ok:
        return "ascii"
    if env.get("TERM", "") == "linux":  # the Linux text console's fonts have half blocks but not quadrants
        return "half"
    if shows_sextants(env):
        return "sextant"
    return "quad"


def mouse_sequences(mouse):
    """(on, off) escape sequences for a mouse setting: False, True or "click", "drag", "move"."""
    if not mouse:
        return "", ""
    if mouse not in MOUSE_MODES:
        raise ValueError(f"mouse must be False, True, 'click', 'drag' or 'move', not {mouse!r}")
    mode = MOUSE_MODES[mouse]
    return f"\x1b[?{mode}h\x1b[?1006h", f"\x1b[?{mode}l\x1b[?1006l"


class Console:
    """Raw terminal I/O. Use as a context manager: entering switches to a full-screen raw mode.

    mouse: False, True (clicks), "drag" (clicks, and moves while a button is held) or "move" (every move).
    key_release: ask the terminal to report key releases (the kitty keyboard protocol); terminals
    without it ignore the request.
    """

    unicode = True
    reports_releases = False  # whether key_is_down() can tell when a key is let go

    def __init__(self, mouse=False, title=None, key_release=False):
        self.mouse = mouse
        self.title = title
        self.key_release = key_release
        self._mouse_on, self._mouse_off = mouse_sequences(mouse)

    def __enter__(self):
        self._setup()
        title = f"{TITLE_PUSH}\x1b]0;{self.title}\x07" if self.title else ""
        # The keyboard flags go on the alternate screen's own stack, so they are popped before leaving it.
        self.write(title + ENTER_SEQ + self._mouse_on + (KITTY_KEYBOARD_ON if self.key_release else ""))
        return self

    def __exit__(self, *exc):
        try:
            self.write((KITTY_KEYBOARD_OFF if self.key_release else "") + self._mouse_off + EXIT_SEQ
                       + (TITLE_POP if self.title else ""))
        finally:
            self._restore()

    def key_is_down(self, key):
        """Whether `key` is physically held right now; None where that can't be asked (see WindowsConsole)."""
        return None

    def size(self):
        """(columns, rows) of the terminal window."""
        try:
            cols, rows = os.get_terminal_size(self._size_fd())
        except OSError:
            cols, rows = 80, 24
        return max(cols, 1), max(rows, 1)

    def _size_fd(self):
        return sys.__stdout__.fileno()


class PosixConsole(Console):
    def __init__(self, mouse=False, title=None, key_release=False):
        super().__init__(mouse, title, key_release)
        import termios
        self._termios = termios
        self.fd_in = sys.stdin.fileno()
        self.fd_out = sys.stdout.fileno()
        try:
            locale.setlocale(locale.LC_CTYPE, "")
        except locale.Error:
            pass
        encoding = (locale.getpreferredencoding(False) or "ascii").lower().replace("-", "").replace("_", "")
        self.unicode = encoding == "utf8"
        self._decoder = codecs.getincrementaldecoder("utf-8" if self.unicode else "latin-1")(errors="replace")
        self._saved = None

    def _setup(self):
        t = self._termios
        self._saved = t.tcgetattr(self.fd_in)
        attrs = t.tcgetattr(self.fd_in)
        attrs[0] &= ~(t.IXON | t.ICRNL | t.INLCR)          # iflag: no flow control, keep Enter as \r
        attrs[3] &= ~(t.ICANON | t.ECHO | t.IEXTEN)         # lflag: no line editing or echo; Ctrl-C still interrupts
        attrs[6] = list(attrs[6])
        attrs[6][t.VMIN], attrs[6][t.VTIME] = 0, 0
        try:
            attrs[6][t.VSUSP] = os.fpathconf(self.fd_in, "PC_VDISABLE")  # Ctrl-Z would leave the screen garbled
        except (OSError, ValueError, AttributeError):
            pass
        t.tcsetattr(self.fd_in, t.TCSANOW, attrs)

    def _restore(self):
        if self._saved is not None:
            self._termios.tcsetattr(self.fd_in, self._termios.TCSAFLUSH, self._saved)

    def read(self):
        """Everything typed since the last call, as text (never blocks)."""
        chunks = []
        while select.select([self.fd_in], [], [], 0)[0]:
            data = os.read(self.fd_in, 65536)
            if not data:
                break
            chunks.append(self._decoder.decode(data))
        return "".join(chunks)

    def write(self, text):
        """Write text, or bytes already encoded for this terminal (UTF-8, or ASCII without Unicode)."""
        data = text if isinstance(text, bytes) else text.encode("utf-8" if self.unicode else "ascii", "replace")
        while data:
            try:
                n = os.write(self.fd_out, data)
            except BlockingIOError:
                select.select([], [self.fd_out], [])
                continue
            data = data[n:]

    def _size_fd(self):
        return self.fd_out


class WindowsInput:
    """Turns Windows console key and mouse records into the sequences a VT terminal sends.

    Pure Python, so it runs (and is tested) on any platform; the records only need the
    fields ctypes gives KEY_EVENT_RECORD and MOUSE_EVENT_RECORD.
    """

    MOUSE_MOVED, MOUSE_WHEELED = 0x1, 0x4

    # Virtual-key codes of keys with no character, and the sequence a VT terminal sends for each.
    VK_SEQ = {0x26: "\x1b[A", 0x28: "\x1b[B", 0x27: "\x1b[C", 0x25: "\x1b[D", 0x24: "\x1b[H", 0x23: "\x1b[F",
              0x21: "\x1b[5~", 0x22: "\x1b[6~", 0x2D: "\x1b[2~", 0x2E: "\x1b[3~",
              0x70: "\x1bOP", 0x71: "\x1bOQ", 0x72: "\x1bOR", 0x73: "\x1bOS", 0x74: "\x1b[15~", 0x75: "\x1b[17~",
              0x76: "\x1b[18~", 0x77: "\x1b[19~", 0x78: "\x1b[20~", 0x79: "\x1b[21~", 0x7A: "\x1b[23~",
              0x7B: "\x1b[24~"}

    def __init__(self, mouse=True):
        self.moves = mouse in ("drag", "move")  # report moves: while a button is held, or ("move") always
        self.all_moves = mouse == "move"
        self.buttons = 0  # mouse buttons held, as dwButtonState bits: presses and releases are the changes

    def key(self, ev):
        if not ev.bKeyDown:
            return ""
        repeat = max(ev.wRepeatCount, 1)
        if ev.uChar:
            return chr(ev.uChar) * repeat  # a UTF-16 unit: surrogate halves are paired up in read()
        return self.VK_SEQ.get(ev.wVirtualKeyCode, "") * repeat

    def mouse(self, ev):
        x, y = ev.dwMousePosition.X + 1, ev.dwMousePosition.Y + 1
        if ev.dwEventFlags & self.MOUSE_WHEELED:
            up = ev.dwButtonState >> 16 < 0x8000  # the high word is the signed wheel delta
            return f"\x1b[<{64 if up else 65};{x};{y}M"
        if ev.dwEventFlags == self.MOUSE_MOVED:
            state = ev.dwButtonState & 0x7
            button = next((b for bit, b in ((1, 0), (4, 1), (2, 2)) if state & bit), 3)
            if self.moves and (button != 3 or self.all_moves):
                return f"\x1b[<{32 + button};{x};{y}M"
            return ""
        if ev.dwEventFlags:  # double clicks, horizontal wheel
            return ""
        state, out = ev.dwButtonState & 0x7, []
        for bit, button in ((1, 0), (4, 1), (2, 2)):  # left, middle, right
            if (state ^ self.buttons) & bit:
                out.append(f"\x1b[<{button};{x};{y}{'M' if state & bit else 'm'}")
        self.buttons = state
        return "".join(out)


class WindowsConsole(Console):
    """The Windows console (Windows Terminal, or conhost on Windows 10 and later).

    Output uses VT processing. Input is read as console input records without
    blocking; with VT input mode on, special keys and mouse clicks already arrive
    as escape sequences, and anything that arrives as a raw key or mouse record
    is translated to the same sequences, so one decoder serves every platform.
    """

    STD_INPUT_HANDLE, STD_OUTPUT_HANDLE = -10, -11
    ENABLE_PROCESSED_INPUT = 0x1
    ENABLE_MOUSE_INPUT = 0x10
    ENABLE_EXTENDED_FLAGS = 0x80  # with ENABLE_QUICK_EDIT_MODE (0x40) left out, clicks reach the program
    ENABLE_VIRTUAL_TERMINAL_INPUT = 0x200
    ENABLE_PROCESSED_OUTPUT = 0x1
    ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x4
    DISABLE_NEWLINE_AUTO_RETURN = 0x8
    KEY_EVENT, MOUSE_EVENT = 0x1, 0x2

    reports_releases = True  # key_is_down() asks Windows directly

    # Virtual-key codes of keys that are not characters (characters are looked up with VkKeyScanW).
    VK_OF_KEY = {Key.ENTER: 0x0D, 10: 0x0D, Key.TAB: 0x09, Key.ESC: 0x1B, Key.BACKSPACE: 0x08, 8: 0x08, 32: 0x20,
                 Key.UP: 0x26, Key.DOWN: 0x28, Key.LEFT: 0x25, Key.RIGHT: 0x27, Key.HOME: 0x24, Key.END: 0x23,
                 Key.PAGE_UP: 0x21, Key.PAGE_DOWN: 0x22, Key.INSERT: 0x2D, Key.DELETE: 0x2E,
                 **{Key.F1 + i: 0x70 + i for i in range(12)}}

    def __init__(self, mouse=False, title=None, key_release=False):
        super().__init__(mouse, title, key_release)
        import ctypes
        from ctypes import wintypes as wt

        # Fixed-width fields, so the layout matches the Win32 one whatever ctypes' platform types are.
        u16, i16, u32, i32 = ctypes.c_uint16, ctypes.c_int16, ctypes.c_uint32, ctypes.c_int32

        class COORD(ctypes.Structure):
            _fields_ = [("X", i16), ("Y", i16)]

        class KEY_EVENT_RECORD(ctypes.Structure):
            _fields_ = [("bKeyDown", i32), ("wRepeatCount", u16), ("wVirtualKeyCode", u16),
                        ("wVirtualScanCode", u16), ("uChar", u16), ("dwControlKeyState", u32)]

        class MOUSE_EVENT_RECORD(ctypes.Structure):
            _fields_ = [("dwMousePosition", COORD), ("dwButtonState", u32),
                        ("dwControlKeyState", u32), ("dwEventFlags", u32)]

        class EVENT(ctypes.Union):
            _fields_ = [("KeyEvent", KEY_EVENT_RECORD), ("MouseEvent", MOUSE_EVENT_RECORD), ("pad", u32 * 4)]

        class INPUT_RECORD(ctypes.Structure):
            _fields_ = [("EventType", u16), ("Event", EVENT)]

        self._ctypes, self._wt, self._INPUT_RECORD = ctypes, wt, INPUT_RECORD
        k = self._k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k.GetStdHandle.restype = wt.HANDLE
        k.GetStdHandle.argtypes = [wt.DWORD]
        k.GetConsoleMode.argtypes = [wt.HANDLE, ctypes.POINTER(wt.DWORD)]
        k.SetConsoleMode.argtypes = [wt.HANDLE, wt.DWORD]
        k.GetNumberOfConsoleInputEvents.argtypes = [wt.HANDLE, ctypes.POINTER(wt.DWORD)]
        k.ReadConsoleInputW.argtypes = [wt.HANDLE, ctypes.POINTER(INPUT_RECORD), wt.DWORD, ctypes.POINTER(wt.DWORD)]
        k.WriteConsoleW.argtypes = [wt.HANDLE, wt.LPCWSTR, wt.DWORD, ctypes.POINTER(wt.DWORD), wt.LPVOID]
        self.h_in = k.GetStdHandle(self.STD_INPUT_HANDLE & 0xFFFFFFFF)
        self.h_out = k.GetStdHandle(self.STD_OUTPUT_HANDLE & 0xFFFFFFFF)
        u = self._u32 = ctypes.WinDLL("user32")
        u.GetAsyncKeyState.restype = ctypes.c_short
        u.GetAsyncKeyState.argtypes = [ctypes.c_int]
        u.VkKeyScanW.restype = ctypes.c_short
        u.VkKeyScanW.argtypes = [ctypes.c_wchar]
        self._saved = None
        self._input = WindowsInput(mouse)

    def _mode(self, handle):
        mode = self._wt.DWORD()
        if not self._k32.GetConsoleMode(handle, self._ctypes.byref(mode)):
            raise OSError("not running in a Windows console (stdin/stdout are redirected?)")
        return mode.value

    def _setup(self):
        in_mode, out_mode = self._mode(self.h_in), self._mode(self.h_out)
        self._saved = (in_mode, out_mode)
        out = out_mode | self.ENABLE_PROCESSED_OUTPUT | self.ENABLE_VIRTUAL_TERMINAL_PROCESSING
        if not self._k32.SetConsoleMode(self.h_out, out | self.DISABLE_NEWLINE_AUTO_RETURN):
            if not self._k32.SetConsoleMode(self.h_out, out):
                raise OSError("this console does not understand VT sequences (Windows 10 or later is needed)")
        new_in = self.ENABLE_PROCESSED_INPUT | self.ENABLE_EXTENDED_FLAGS | (self.ENABLE_MOUSE_INPUT if self.mouse else 0)
        if not self._k32.SetConsoleMode(self.h_in, new_in | self.ENABLE_VIRTUAL_TERMINAL_INPUT):
            self._k32.SetConsoleMode(self.h_in, new_in)  # older consoles: raw records only, translated in read()

    def _restore(self):
        if self._saved is not None:
            self._k32.SetConsoleMode(self.h_in, self._saved[0])
            self._k32.SetConsoleMode(self.h_out, self._saved[1])

    def key_is_down(self, key):
        """Whether `key` (a character's code or a Key) is held right now, as the keyboard itself reports it.

        Windows keeps the state of every key, so a release is seen even when the
        console doesn't pass one on (and even when the window has lost focus).
        None for keys with no virtual-key code.
        """
        vk = self.VK_OF_KEY.get(key)
        if vk is None:
            if not 32 < key < 0x10000:
                return None
            scan = self._u32.VkKeyScanW(chr(key))
            if scan == -1:
                return None
            vk = scan & 0xFF
        return bool(self._u32.GetAsyncKeyState(vk) & 0x8000)

    def read(self):
        ct, wt = self._ctypes, self._wt
        parts = []
        count = wt.DWORD()
        while self._k32.GetNumberOfConsoleInputEvents(self.h_in, ct.byref(count)) and count.value:
            records = (self._INPUT_RECORD * count.value)()
            got = wt.DWORD()
            if not self._k32.ReadConsoleInputW(self.h_in, records, count.value, ct.byref(got)):
                break
            for rec in records[:got.value]:
                if rec.EventType == self.KEY_EVENT:
                    parts.append(self._input.key(rec.Event.KeyEvent))
                elif rec.EventType == self.MOUSE_EVENT:
                    parts.append(self._input.mouse(rec.Event.MouseEvent))
        # Characters outside the BMP arrive as two records, one surrogate each: pair them up.
        return "".join(parts).encode("utf-16-le", "surrogatepass").decode("utf-16-le", "replace")

    def write(self, text):
        """Write text, or UTF-8 bytes."""
        if isinstance(text, bytes):
            text = text.decode("utf-8", "replace")
        ct, wt = self._ctypes, self._wt
        written = wt.DWORD()
        for s in range(0, len(text), 8192):
            chunk = text[s:s + 8192]
            units = len(chunk.encode("utf-16-le")) // 2  # characters outside the BMP take two
            self._k32.WriteConsoleW(self.h_out, chunk, units, ct.byref(written), None)


def open_console(mouse=False, title=None, key_release=False):
    """The console for this platform; `title` names the terminal window while it is open (see Console)."""
    cls = WindowsConsole if os.name == "nt" else PosixConsole
    return cls(mouse, title, key_release)
