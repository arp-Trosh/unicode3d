# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
"""Model viewer, after the original C renderer: python -m unicode3d.examples.viewer [model.obj|.gltf|.glb]

Shows a Wavefront .obj model with its materials and textures (from the .mtl files it names), or a glTF model
with its own, playing its animations. WASD moves the camera, arrow keys look around, q/e slow down / speed up
the spin, c makes the model chrome (reflecting a sky), o switches between perspective and an orthographic view
(everything the same size however far off), n plays the model's next animation, Esc or Ctrl-C quits.
"""
import argparse
import sys

import numpy as np

from .dice import make_die
from .hud import StatusBar
from ..background import Sky
from ..models import load_model
from ..scene import Camera, Light, Model, Node, Object3D, Renderer
from ..color import Color
from ..keys import Key
from ..terminal import add_display_args, display_options, run
from ..transforms import UP, quat_axis_angle, quat_mul


class Viewer:
    def __init__(self, model):
        self.model = model
        self.reflectivity = [obj.reflectivity for obj in model]
        self.camera = Camera(position=np.array([0.0, 0.0, 4.0]))
        self.light = Light(direction=np.array([0.0, 0.0, -1.0]), shadows=True)
        self.renderer = Renderer(1, 1)
        self.yaw = self.pitch = 0.0
        self.spin = 0.6  # radians per second
        self.axis = np.array([1.0, 1.0, 0.3])
        self.bar = StatusBar(self.renderer)
        # Its animations, looping, the first playing; n moves to the next, and after the last to none.
        self.clips = list(model.animations.values())
        for clip in self.clips:
            clip.loop = "loop"
        self.playing = 0 if self.clips else None

    def frame(self, screen, dt, keys):
        fwd = np.array([np.sin(self.yaw) * np.cos(self.pitch), np.sin(self.pitch), -np.cos(self.yaw) * np.cos(self.pitch)])
        right = np.cross(fwd, UP)
        right /= np.linalg.norm(right)
        step, turn = 0.1, 0.05
        for k in self.bar.handle(keys, screen):
            if k in (27, 3):
                return False
            moves = {ord("w"): fwd * step, ord("s"): -fwd * step, ord("d"): right * step, ord("a"): -right * step}
            if k in moves:
                self.camera.position = self.camera.position + moves[k]
            elif k == Key.UP:
                self.pitch = min(self.pitch + turn, 1.5)
            elif k == Key.DOWN:
                self.pitch = max(self.pitch - turn, -1.5)
            elif k == Key.LEFT:
                self.yaw -= turn
            elif k == Key.RIGHT:
                self.yaw += turn
            elif k == ord("q"):
                self.spin /= 2
            elif k == ord("e"):
                self.spin = min(self.spin * 2, 20.0)
            elif k == ord("n") and self.clips:
                self.playing = None if self.playing == len(self.clips) - 1 else (
                    0 if self.playing is None else self.playing + 1)
                if self.playing is not None:
                    self.clips[self.playing].apply(0.0)
            elif k == ord("o"):  # orthographic, showing the model at the size it has now, or perspective again
                ortho = self.camera.projection != "ortho"
                self.camera.projection = "ortho" if ortho else "perspective"
                if ortho:
                    distance = float(np.linalg.norm(self.camera.position))
                    self.camera.size = 2.0 * distance * np.tan(np.radians(self.camera.fov) / 2)
            elif k == ord("c"):  # chrome, with a sky to reflect
                chrome = self.renderer.background is None
                for obj, own in zip(self.model, self.reflectivity):
                    obj.reflectivity = 0.85 if chrome else own
                self.renderer.background = Sky() if chrome else None
        self.camera.target = self.camera.position + fwd
        root = self.model.root
        root.rotation = quat_mul(quat_axis_angle(self.axis, self.spin * dt), root.rotation)
        if self.playing is not None:
            self.clips[self.playing].update(dt)

        rows, cols = screen.size()
        self.renderer.resize(cols, max(rows - 1, 1), screen.cell_pixels)
        fb = self.renderer.render([*self.model], self.camera, self.light)
        screen.erase()
        screen.draw_frame(fb)
        p = self.camera.position
        playing = ""
        if self.clips:
            name = "none" if self.playing is None else self.clips[self.playing].name
            playing = f"[n] animation: {name} ({0 if self.playing is None else self.playing + 1}/{len(self.clips)})   "
        view = "ortho" if self.camera.projection == "ortho" else "perspective"
        self.bar.draw(screen, f"[wasd/arrows] move  [q/e] spin  [c] chrome  [o] {view}  {playing}[esc] quit   "
                              f"{sum(len(obj.mesh.faces) for obj in self.model)} tris   "
                              f"cam {p[0]:.2f} {p[1]:.2f} {p[2]:.2f}   spin {self.spin:.2f}")
        screen.refresh()
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", nargs="?", help="a Wavefront .obj or glTF .gltf/.glb file (default: a die)")
    parser.add_argument("--still", action="store_true", help="don't spin the model")
    parser.add_argument("--double-sided", action="store_true", help="draw back faces (for meshes with bad winding)")
    parser.add_argument("--groups", action="store_true", help="load each group of the model as a part of its own")
    parser.add_argument("--fps", type=int, default=30)
    add_display_args(parser)
    args = parser.parse_args()
    if args.model:
        model = load_model(args.model, split_groups=args.groups, double_sided=args.double_sided).fit()
        for warning in model.warnings:
            print(f"{args.model}: {warning}", file=sys.stderr)
        if not model.materials:  # no colours of its own: cyan, as the original viewer drew it
            for obj in model:
                obj.color = Color.CYAN
    else:
        root = Node()
        model = Model(root, [Object3D(make_die(1.5), parent=root, double_sided=args.double_sided)])
    try:
        viewer = Viewer(model)
        if args.still:
            viewer.spin = 0.0
        run(viewer.frame, args.fps, mouse=True, title="unicode3d viewer",
            **display_options(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
