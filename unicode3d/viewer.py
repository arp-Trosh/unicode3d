"""Model viewer, after the original C renderer: python -m unicode3d.viewer [model.obj]

WASD moves the camera, arrow keys look around, q/e slow down / speed up the
spin, Esc or Ctrl-C quits.
"""
import argparse
import curses

import numpy as np

from .dice import make_die
from .mesh import load_obj
from .scene import Camera, Light, Object3D, Renderer
from .terminal import Color, init_locale, run_loop
from .transforms import UP, quat_axis_angle, quat_mul


class Viewer:
    def __init__(self, mesh, double_sided):
        self.obj = Object3D(mesh, color=Color.CYAN, double_sided=double_sided)
        self.camera = Camera(position=np.array([0.0, 0.0, 4.0]))
        self.light = Light(direction=np.array([0.0, 0.0, -1.0]))
        self.renderer = Renderer(1, 1)
        self.yaw = self.pitch = 0.0
        self.spin = 0.6  # radians per second
        self.axis = np.array([1.0, 1.0, 0.3])

    def frame(self, screen, dt, keys):
        fwd = np.array([np.sin(self.yaw) * np.cos(self.pitch), np.sin(self.pitch), -np.cos(self.yaw) * np.cos(self.pitch)])
        right = np.cross(fwd, UP)
        right /= np.linalg.norm(right)
        step, turn = 0.1, 0.05
        for k in keys:
            if k in (27, 3):
                return False
            moves = {ord("w"): fwd * step, ord("s"): -fwd * step, ord("d"): right * step, ord("a"): -right * step}
            if k in moves:
                self.camera.position = self.camera.position + moves[k]
            elif k == curses.KEY_UP:
                self.pitch = min(self.pitch + turn, 1.5)
            elif k == curses.KEY_DOWN:
                self.pitch = max(self.pitch - turn, -1.5)
            elif k == curses.KEY_LEFT:
                self.yaw -= turn
            elif k == curses.KEY_RIGHT:
                self.yaw += turn
            elif k == ord("q"):
                self.spin /= 2
            elif k == ord("e"):
                self.spin = min(self.spin * 2, 20.0)
        self.camera.target = self.camera.position + fwd
        self.obj.rotation = quat_mul(quat_axis_angle(self.axis, self.spin * dt), self.obj.rotation)

        rows, cols = screen.size()
        self.renderer.resize(cols, max(rows - 1, 1))
        fb = self.renderer.render([self.obj], self.camera, self.light)
        screen.erase()
        screen.draw_frame(fb)
        p = self.camera.position
        screen.text(rows - 1, 1, f"{len(self.obj.mesh.faces)} tris   cam {p[0]:.2f} {p[1]:.2f} {p[2]:.2f}   "
                                 f"spin {self.spin:.2f}   [wasd/arrows] move  [q/e] spin  [esc] quit")
        screen.refresh()
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", nargs="?", help="Wavefront .obj file (default: a die)")
    parser.add_argument("--double-sided", action="store_true", help="draw back faces (for meshes with bad winding)")
    parser.add_argument("--ascii", action="store_true", help="draw with ASCII characters instead of Unicode blocks")
    args = parser.parse_args()
    init_locale()
    mesh = load_obj(args.model).normalized() if args.model else make_die(1.5)
    try:
        curses.wrapper(run_loop, Viewer(mesh, args.double_sided).frame, 30, "ascii" if args.ascii else None)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
