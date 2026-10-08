# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
"""3D renderer for the terminal (numpy and Numba, no curses).

Draws with Unicode block characters in 24-bit, 256 or 16 colours, whatever the
terminal supports, with an ASCII fallback; runs in Windows Terminal and Unix
terminals alike. Ported in spirit from https://github.com/ShakedAp/ASCII-renderer.
"""
__version__ = "0.18.0"

from .threads import prefer_sleeping_workers as _prefer_sleeping_workers

_prefer_sleeping_workers()  # before any kernel runs (see threads.py)
del _prefer_sleeping_workers

from .animation import Animation, Clip, RotationTrack, SplineTrack, Track
from .background import Fog, Gradient, Sky, SkyBox
from .color import Color
from .keys import HeldKeys, Key, KeyRelease, MouseEvent
from .mesh import Mesh, make_box
from .gltf import load_gltf
from .models import Material, load_model, load_mtl, load_obj
from .queries import Colliders, Contact, Hit
from .raster import FrameBuffer
from .scene import Anchor, Camera, Light, Model, Node, Object3D, Pick, PointLight, Renderer, union_bounds
from .terminal import Screen, add_display_args, compile_kernels, display_options, frame_to_text, run
from .texture import load_image
from .ui import Button, Choice, DisplayControls, Panel, Slider, Toggle

from .kernel_cache import refresh as _refresh_kernel_cache

_refresh_kernel_cache()  # drop cached kernels that were compiled with helpers since changed (see kernel_cache.py)
del _refresh_kernel_cache
