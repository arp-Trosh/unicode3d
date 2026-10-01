# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
"""3D renderer for the terminal (numpy and Numba, no curses).

Draws with Unicode block characters in 24-bit, 256 or 16 colours, whatever the
terminal supports, with an ASCII fallback; runs in Windows Terminal and Unix
terminals alike. Ported in spirit from https://github.com/ShakedAp/ASCII-renderer.
"""
__version__ = "0.6.0"

from .animation import Animation, RotationTrack, Track
from .background import Fog, Gradient, Sky, SkyBox
from .color import Color
from .keys import HeldKeys, Key, KeyRelease, MouseEvent
from .mesh import Mesh, make_box
from .models import Material, load_model, load_mtl, load_obj
from .raster import FrameBuffer
from .scene import Camera, Light, Model, Node, Object3D, Pick, PointLight, Renderer
from .terminal import Screen, add_display_args, compile_kernels, display_options, frame_to_text, run
from .texture import load_image
from .ui import Button, Choice, DisplayControls, Panel, Slider, Toggle
