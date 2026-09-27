# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
"""3D renderer for the terminal (numpy only, no curses).

Draws with Unicode block characters in 24-bit, 256 or 16 colours, whatever the
terminal supports, with an ASCII fallback; runs in Windows Terminal and Unix
terminals alike. Ported in spirit from https://github.com/ShakedAp/ASCII-renderer.
"""
__version__ = "0.1.0"

from .color import Color
from .keys import Key, MouseEvent
from .mesh import Mesh, load_obj, make_box
from .raster import FrameBuffer
from .scene import Camera, Light, Object3D, Renderer
from .terminal import Screen, add_display_args, display_options, frame_to_text, run
