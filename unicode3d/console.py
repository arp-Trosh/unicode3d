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

ENTER_SEQ = "\x1b[?1049h\x1b[?25l\x1b[?7l\x1b[2J"  # alternate screen, hide cursor, no auto-wrap, clear
EXIT_SEQ = "\x1b[0m\x1b[?7h\x1b[?25h\x1b[?1049l"
MOUSE_ON = "\x1b[?1000h\x1b[?1006h"   # report clicks, in SGR encoding
MOUSE_OFF = "\x1b[?1000l\x1b[?1006l"
TITLE_PUSH, TITLE_POP = "\x1b[22;0t", "\x1b[23;0t"  # save and restore the window title, where supported

TRUECOLOR_TERMS = ("kitty", "ghostty", "alacritty", "foot", "wezterm", "contour", "iterm", "rio")
TRUECOLOR_PROGRAMS = ("iTerm.app", "WezTerm", "vscode", "Hyper", "ghostty", "Tabby", "rio")


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


def detect_glyphs(env=None, unicode_ok=True):
    """Best guess at the richest glyph set the terminal's font can show: quad, half or ascii.

    UNICODE3D_GLYPHS overrides it. Sextants are never picked automatically, because
    older fonts lack them; ask for them explicitly.
    """
    env = os.environ if env is None else env
    forced = env.get("UNICODE3D_GLYPHS", "").lower()
    if forced in ("half", "quad", "sextant", "ascii"):
        return forced
    if not unicode_ok:
        return "ascii"
    if env.get("TERM", "") == "linux":  # the Linux text console's fonts have half blocks but not quadrants
        return "half"
    return "quad"


class Console:
    """Raw terminal I/O. Use as a context manager: entering switches to a full-screen raw mode."""

    unicode = True

    def __init__(self, mouse=False, title=None):
        self.mouse = mouse
        self.title = title

    def __enter__(self):
        self._setup()
        title = f"{TITLE_PUSH}\x1b]0;{self.title}\x07" if self.title else ""
        self.write(title + ENTER_SEQ + (MOUSE_ON if self.mouse else ""))
        return self

    def __exit__(self, *exc):
        try:
            self.write((MOUSE_OFF if self.mouse else "") + EXIT_SEQ + (TITLE_POP if self.title else ""))
        finally:
            self._restore()

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
    def __init__(self, mouse=False, title=None):
        super().__init__(mouse, title)
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
        data = text.encode("utf-8" if self.unicode else "ascii", "replace")
        while data:
            try:
                n = os.write(self.fd_out, data)
            except BlockingIOError:
                select.select([], [self.fd_out], [])
                continue
            data = data[n:]

    def _size_fd(self):
        return self.fd_out


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
    MOUSE_WHEELED = 0x4

    # Virtual-key codes of keys with no character, and the sequence a VT terminal sends for each.
    VK_SEQ = {0x26: "\x1b[A", 0x28: "\x1b[B", 0x27: "\x1b[C", 0x25: "\x1b[D", 0x24: "\x1b[H", 0x23: "\x1b[F",
              0x21: "\x1b[5~", 0x22: "\x1b[6~", 0x2D: "\x1b[2~", 0x2E: "\x1b[3~",
              0x70: "\x1bOP", 0x71: "\x1bOQ", 0x72: "\x1bOR", 0x73: "\x1bOS", 0x74: "\x1b[15~", 0x75: "\x1b[17~",
              0x76: "\x1b[18~", 0x77: "\x1b[19~", 0x78: "\x1b[20~", 0x79: "\x1b[21~", 0x7A: "\x1b[23~",
              0x7B: "\x1b[24~"}

    def __init__(self, mouse=False, title=None):
        super().__init__(mouse, title)
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
        self._saved = None
        self._buttons = 0

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
                    parts.append(self._key(rec.Event.KeyEvent))
                elif rec.EventType == self.MOUSE_EVENT:
                    parts.append(self._mouse(rec.Event.MouseEvent))
        # Characters outside the BMP arrive as two records, one surrogate each: pair them up.
        return "".join(parts).encode("utf-16-le", "surrogatepass").decode("utf-16-le", "replace")

    def _key(self, ev):
        if not ev.bKeyDown:
            return ""
        repeat = max(ev.wRepeatCount, 1)
        if ev.uChar:
            return chr(ev.uChar) * repeat  # a UTF-16 unit: surrogate halves are paired up in read()
        return self.VK_SEQ.get(ev.wVirtualKeyCode, "") * repeat

    def _mouse(self, ev):
        x, y = ev.dwMousePosition.X + 1, ev.dwMousePosition.Y + 1
        if ev.dwEventFlags & self.MOUSE_WHEELED:
            up = ev.dwButtonState >> 16 < 0x8000  # the high word is the signed wheel delta
            return f"\x1b[<{64 if up else 65};{x};{y}M"
        if ev.dwEventFlags:  # movement, double clicks, horizontal wheel
            return ""
        state, out = ev.dwButtonState & 0x7, []
        for bit, button in ((1, 0), (4, 1), (2, 2)):  # left, middle, right
            if (state ^ self._buttons) & bit:
                out.append(f"\x1b[<{button};{x};{y}{'M' if state & bit else 'm'}")
        self._buttons = state
        return "".join(out)

    def write(self, text):
        ct, wt = self._ctypes, self._wt
        written = wt.DWORD()
        for s in range(0, len(text), 8192):
            chunk = text[s:s + 8192]
            units = len(chunk.encode("utf-16-le")) // 2  # characters outside the BMP take two
            self._k32.WriteConsoleW(self.h_out, chunk, units, ct.byref(written), None)


def open_console(mouse=False, title=None):
    """The console for this platform; `title` names the terminal window while it is open."""
    return WindowsConsole(mouse, title) if os.name == "nt" else PosixConsole(mouse, title)
