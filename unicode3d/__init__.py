"""3D renderer for the terminal (curses + numpy).

Draws with Unicode half blocks in 256 colours where the terminal allows, with an
ASCII fallback. Ported in spirit from https://github.com/ShakedAp/ASCII-renderer.
"""
from .mesh import Mesh, load_obj, make_box
from .raster import FrameBuffer
from .scene import Camera, Light, Object3D, Renderer
from .terminal import Color, Screen, frame_to_text, init_locale, run_loop
