"""Camera, lights, scene objects and the render pipeline."""
from dataclasses import dataclass, field

import numpy as np

from .mesh import Mesh
from .raster import PIXELS_PER_CELL, FrameBuffer, clip_near, perspective_interpolate, rasterize
from .transforms import look_at, normalize, perspective, quat_identity, quat_to_matrix


def _vec3(*v):
    return field(default_factory=lambda: np.array(v, dtype=float))


@dataclass
class Camera:
    position: np.ndarray = _vec3(0.0, 0.0, 5.0)
    target: np.ndarray = _vec3(0.0, 0.0, 0.0)
    up: np.ndarray = _vec3(0.0, 1.0, 0.0)
    fov: float = 50.0  # vertical, degrees
    near: float = 0.1
    far: float = 100.0

    def view_matrix(self):
        return look_at(self.position, self.target, self.up)


@dataclass
class Light:
    direction: np.ndarray = _vec3(0.3, -1.0, -0.5)  # the way the light travels
    ambient: float = 0.2
    diffuse: float = 0.8
    specular: float = 0.35  # strength of the Blinn-Phong highlight
    shininess: float = 24.0


@dataclass
class Object3D:
    mesh: Mesh
    position: np.ndarray = _vec3(0.0, 0.0, 0.0)
    rotation: np.ndarray = field(default_factory=quat_identity)
    scale: float = 1.0
    color: int = 0          # colour id, see terminal.Color
    visible: bool = True
    double_sided: bool = False  # draw back faces too (for meshes with inconsistent winding)


class Renderer:
    """Renders objects into a FrameBuffer of pixels, PIXELS_PER_CELL to a terminal cell.

    width and height are in terminal cells; cell_aspect is a cell's width divided
    by its height (terminal cells are roughly twice as tall as they are wide).

    supersample: samples per pixel along each axis, averaged for smooth edges.
    fog: how much the farthest surfaces are dimmed relative to the nearest (depth cueing).
    outline: how much to darken the far side of depth edges, where one surface
    passes in front of another, so overlapping shapes stay distinct.
    """

    def __init__(self, width, height, cell_aspect=0.5, supersample=2, fog=0.3, outline=0.55):
        self.cell_aspect = cell_aspect
        self.supersample = supersample
        self.fog = fog
        self.outline = outline
        self.width = self.height = 0
        self.samples = FrameBuffer(0, 0)
        self.framebuffer = FrameBuffer(0, 0)
        self.view_proj = None
        self.resize(width, height)

    def resize(self, width, height):
        if (width, height) == (self.width, self.height):
            return
        self.width, self.height = width, height
        ss = self.supersample
        self.samples.resize(width * ss, height * PIXELS_PER_CELL * ss)
        self.framebuffer.resize(width, height * PIXELS_PER_CELL)

    def project(self, point):
        """Cell coordinates (x, y) of a world point as of the last render, or None if behind the camera."""
        if self.view_proj is None:
            return None
        clip = self.view_proj @ np.array([*point, 1.0])
        if clip[3] <= 1e-6:
            return None
        return (clip[0] / clip[3] + 1.0) * 0.5 * self.width, (1.0 - clip[1] / clip[3]) * 0.5 * self.height

    def render(self, objects, camera, light):
        sb = self.samples
        sb.clear()
        if self.width < 1 or self.height < 1:
            self.framebuffer.clear()
            return self.framebuffer
        aspect = self.width * self.cell_aspect / self.height
        self.view_proj = perspective(np.radians(camera.fov), aspect, camera.near, camera.far) @ camera.view_matrix()
        for obj in objects:
            if obj.visible and len(obj.mesh.faces):
                self._draw(obj, camera, light)
        return self._resolve()

    # ----- geometry and shading ---------------------------------------------------------

    def _draw(self, obj, camera, light):
        sb, mesh = self.samples, obj.mesh
        rot = quat_to_matrix(obj.rotation)
        world = (mesh.vertices * obj.scale) @ rot.T + obj.position
        normals = mesh.vertex_normals() @ rot.T
        faces = mesh.faces
        textured = mesh.materials is not None and bool(mesh.textures)

        tri = world[faces]
        face_n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        cam_pos = np.asarray(camera.position, dtype=float)
        facing = np.einsum("ij,ij->i", face_n, cam_pos - tri[:, 0]) > 0
        flip = np.zeros(len(faces), bool)
        if obj.double_sided:
            flip, facing = ~facing, np.ones_like(facing)
        keep = np.flatnonzero(facing)
        if not len(keep):
            return

        # Per-corner attributes: clip xyzw | world xyz | normal xyz | uv
        n = normals[faces[keep]] * np.where(flip[keep], -1.0, 1.0)[:, None, None]
        uv = mesh.uvs[keep] if textured else np.zeros((len(keep), 3, 2))
        clip = np.concatenate([tri[keep], np.ones((len(keep), 3, 1))], axis=2) @ self.view_proj.T
        corners, src = clip_near(np.concatenate([clip, tri[keep], n, uv], axis=2), camera.near)
        if not len(corners):
            return
        src = keep[src]
        w = corners[:, :, 3]
        inv_w = 1.0 / w
        xs = (corners[:, :, 0] * inv_w + 1.0) * 0.5 * sb.width
        ys = (1.0 - corners[:, :, 1] * inv_w) * 0.5 * sb.height
        pix, t, bary = rasterize(sb, xs, ys, inv_w)
        if not len(pix):
            return

        attrs = perspective_interpolate(corners[:, :, 4:], t, bary, inv_w)
        pos, nrm, uvf = attrs[:, 0:3], attrs[:, 3:6], attrs[:, 6:8]
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
        to_light = -normalize(light.direction)
        to_eye = cam_pos - pos
        to_eye /= np.maximum(np.linalg.norm(to_eye, axis=1, keepdims=True), 1e-12)
        half = to_light + to_eye
        half /= np.maximum(np.linalg.norm(half, axis=1, keepdims=True), 1e-12)
        diffuse = light.ambient + light.diffuse * np.clip(nrm @ to_light, 0, None)
        spec = light.specular * np.clip(np.einsum("ij,ij->i", nrm, half), 0, None) ** light.shininess
        if textured:
            albedo = np.ones(len(pix))
            mats = mesh.materials[src[t]]
            for m in np.unique(mats):
                sel = mats == m
                tex = mesh.textures[m]
                th, tw = tex.shape
                col = np.clip((uvf[sel, 0] * tw).astype(int), 0, tw - 1)
                row = np.clip(((1.0 - uvf[sel, 1]) * th).astype(int), 0, th - 1)
                albedo[sel] = tex[row, col]
            diffuse = diffuse * albedo
        sb.shade.reshape(-1)[pix] = np.clip(diffuse + spec, 0.0, 1.0)
        sb.color.reshape(-1)[pix] = obj.color

    # ----- supersample resolve and post effects -------------------------------------------

    def _resolve(self):
        sb, fb, ss = self.samples, self.framebuffer, self.supersample
        H, W = fb.height, fb.width

        def blocks(a):
            return a.reshape(H, ss, W, ss).transpose(0, 2, 1, 3).reshape(H, W, ss * ss)

        drawn = blocks(sb.shade >= 0)
        coverage = drawn.mean(axis=2)
        shade = np.where(drawn, blocks(sb.shade), 0.0).sum(axis=2) / np.maximum(drawn.sum(axis=2), 1)
        depth = blocks(sb.depth)
        nearest = depth.argmax(axis=2)[..., None]
        fb.depth[:] = np.take_along_axis(depth, nearest, axis=2)[..., 0]
        fb.color[:] = np.take_along_axis(blocks(sb.color), nearest, axis=2)[..., 0]
        # Pixels mostly outside the shape are dropped; partly covered edge pixels fade.
        visible = coverage >= 0.25
        shade = shade * np.sqrt(coverage)
        fb.shade[:] = np.where(visible, shade, -1.0)
        fb.depth[~visible] = 0.0
        if visible.any():
            self._fog(fb, visible)
            self._outline(fb, visible)
        return fb

    def _fog(self, fb, visible):
        """Dim pixels in proportion to how far back they sit within the scene's depth range."""
        if not self.fog:
            return
        dist = 1.0 / np.where(visible, fb.depth, 1.0)
        near, far = dist[visible].min(), dist[visible].max()
        span = max(far - near, 0.25 * near)
        fb.shade[visible] *= 1.0 - self.fog * (dist[visible] - near) / span

    def _outline(self, fb, visible):
        """Darken pixels just behind a depth edge.

        1/w is linear across any plane on screen, so its second difference is ~0 on
        flat and gently curved surfaces and large where one surface passes in front
        of another. It is positive on the far side of the edge, which is the side
        darkened, so shapes in front keep their full size.
        """
        if not self.outline:
            return
        d = fb.depth
        edge = np.zeros_like(visible)
        for axis in (0, 1):
            lap = np.zeros_like(d)
            ok = np.zeros_like(visible)
            sl = [slice(None)] * 2
            mid, lo, hi = list(sl), list(sl), list(sl)
            mid[axis], lo[axis], hi[axis] = slice(1, -1), slice(None, -2), slice(2, None)
            mid, lo, hi = tuple(mid), tuple(lo), tuple(hi)
            lap[mid] = d[lo] + d[hi] - 2 * d[mid]
            ok[mid] = visible[lo] & visible[hi] & visible[mid]
            edge |= ok & (lap > 0.08 * d)
        fb.shade[edge] *= 1.0 - self.outline
