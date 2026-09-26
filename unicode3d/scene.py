"""Camera, lights, scene objects and the render pipeline."""
from dataclasses import dataclass, field

import numpy as np

from .color import srgb_to_linear, to_linear_rgb
from .mesh import Mesh
from .raster import FrameBuffer, barycentric, clip_near, perspective_interpolate, rasterize
from .texture import sample as sample_texture
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
    ambient: float = 0.3    # light levels are perceived brightness, 0..1
    diffuse: float = 0.7
    specular: float = 0.35  # strength of the Blinn-Phong highlight, which is white whatever the surface colour
    shininess: float = 24.0


@dataclass
class Object3D:
    mesh: Mesh
    position: np.ndarray = _vec3(0.0, 0.0, 0.0)
    rotation: np.ndarray = field(default_factory=quat_identity)
    scale: float = 1.0
    color: object = 0       # a named Color, or (r, g, b) sRGB as 0..255 ints or 0..1 floats
    visible: bool = True
    double_sided: bool = False  # draw back faces too (for meshes with inconsistent winding)




# Sample positions within a pixel (x, y from its top-left corner): the standard
# multisample patterns, where no two samples share a row or column, so near-vertical
# and near-horizontal edges get as many coverage steps as there are samples.
SAMPLE_PATTERNS = {
    1: ((0.5, 0.5),),
    4: ((0.375, 0.125), (0.875, 0.375), (0.125, 0.625), (0.625, 0.875)),
    8: ((0.5625, 0.3125), (0.4375, 0.6875), (0.8125, 0.5625), (0.3125, 0.1875),
        (0.1875, 0.8125), (0.0625, 0.4375), (0.6875, 0.9375), (0.9375, 0.0625)),
    16: ((0.5625, 0.5625), (0.4375, 0.3125), (0.3125, 0.625), (0.75, 0.4375),
         (0.1875, 0.375), (0.625, 0.8125), (0.8125, 0.6875), (0.6875, 0.1875),
         (0.375, 0.875), (0.5, 0.0625), (0.25, 0.125), (0.125, 0.75),
         (0.0, 0.5), (0.9375, 0.25), (0.875, 0.9375), (0.0625, 0.0)),
}
EDGE_CONTRAST = 0.03  # linear-light spread among a pixel's first samples that marks it for more


@dataclass
class _Prepared:
    """An object's triangles in screen space, ready to rasterize at any sample offset."""
    obj: Object3D
    ident: int
    xs: np.ndarray
    ys: np.ndarray
    inv_w: np.ndarray
    attrs: np.ndarray      # per-corner world xyz | normal xyz | uv
    src: np.ndarray        # source face of each triangle
    albedo: np.ndarray     # linear RGB
    lod: np.ndarray | None  # mip level of each triangle, if textured


class Renderer:
    """Renders objects into a FrameBuffer whose pixels are finer than terminal cells.

    width and height are in terminal cells; cell_pixels is the (columns, rows)
    of pixels in a cell, which comes from the screen's glyph set
    (Screen.cell_pixels). cell_aspect is a cell's width divided by its height
    (terminal cells are roughly twice as tall as they are wide).

    samples: samples per pixel (1, 4, 8 or 16), averaged for smooth edges. Each
    pixel is shaded once per triangle covering it, like hardware multisampling;
    samples only measure coverage, and texture detail is smoothed by mipmaps.
    edge_samples: extra samples (0, 4, 8 or 16) taken only in pixels whose first
    samples disagree (silhouettes, creases, overlaps), where they matter.
    fog: how much the farthest surfaces are dimmed relative to the nearest (depth cueing).
    outline: how much to darken the far side of depth edges, where one surface
    passes in front of another, so overlapping shapes stay distinct.
    lod_bias: added to every texture's mip level; negative is sharper, positive softer.
    """

    def __init__(self, width, height, cell_pixels=(1, 2), cell_aspect=0.5, samples=4, edge_samples=8,
                 fog=0.3, outline=0.55, lod_bias=-0.5):
        for n in (samples, edge_samples):
            if n not in SAMPLE_PATTERNS and n != 0:
                raise ValueError(f"sample counts must be one of {sorted(SAMPLE_PATTERNS)}, not {n}")
        self.cell_aspect = cell_aspect
        self.samples = samples
        self.edge_samples = edge_samples
        self.fog = fog
        self.outline = outline
        self.lod_bias = lod_bias
        self.width = self.height = 0
        self.cell_pixels = tuple(cell_pixels)
        self.framebuffer = FrameBuffer(0, 0, cell_pixels)
        self.view_proj = None
        self.resize(width, height)

    def resize(self, width, height, cell_pixels=None):
        """Set the size in cells, and optionally the pixels per cell (e.g. Screen.cell_pixels)."""
        cell_pixels = self.cell_pixels if cell_pixels is None else tuple(cell_pixels)
        if (width, height, cell_pixels) == (self.width, self.height, self.cell_pixels):
            return
        self.width, self.height, self.cell_pixels = width, height, cell_pixels
        self.framebuffer.cell_pixels = cell_pixels
        self.framebuffer.resize(width * cell_pixels[0], height * cell_pixels[1])

    def project(self, point):
        """Cell coordinates (x, y) of a world point as of the last render, or None if behind the camera."""
        if self.view_proj is None:
            return None
        clip = self.view_proj @ np.array([*point, 1.0])
        if clip[3] <= 1e-6:
            return None
        return (clip[0] / clip[3] + 1.0) * 0.5 * self.width, (1.0 - clip[1] / clip[3]) * 0.5 * self.height

    def render(self, objects, camera, light):
        """Draw the objects; returns the framebuffer (reused by the next render: copy it to keep it)."""
        fb = self.framebuffer
        fb.clear()
        if self.width < 1 or self.height < 1:
            return fb
        aspect = self.width * self.cell_aspect / self.height
        self.view_proj = perspective(np.radians(camera.fov), aspect, camera.near, camera.far) @ camera.view_matrix()
        prepared = [p for i, obj in enumerate(objects) if obj.visible and len(obj.mesh.faces)
                    for p in [self._prepare(obj, i + 1, camera)] if p is not None]
        if not prepared:
            return fb

        n = self.samples
        base = self._accumulate(prepared, SAMPLE_PATTERNS[n], camera, light)
        shape = (fb.height, fb.width)
        rgb, cover = base["rgb"], base["cover"].astype(float)
        count = np.full(len(cover), float(n))
        fb.depth[:], fb.ids[:] = base["depth"].reshape(shape), base["ids"].reshape(shape)
        if self.edge_samples and n > 1:
            edges = ((cover > 0) & (cover < n)) | base["mixed"] | (base["spread"] > EDGE_CONTRAST)
            if edges.any():
                # A different pattern from the base one: the same one turned a quarter.
                pattern = tuple((1.0 - y, x) for x, y in SAMPLE_PATTERNS[self.edge_samples])
                extra = self._accumulate(prepared, pattern, camera, light, np.flatnonzero(edges))
                px = extra["pixels"]
                rgb[px] += extra["rgb"]
                cover[px] += extra["cover"]
                count[px] += self.edge_samples
        fb.rgb[:] = (rgb / count[:, None]).reshape(shape + (3,))
        fb.alpha[:] = (cover / count).reshape(shape)
        visible = fb.alpha > 0
        if visible.any():
            self._fog(fb, visible)
            self._outline(fb, visible)
        return fb

    # ----- geometry --------------------------------------------------------------------

    def _prepare(self, obj, ident, camera):
        fb, mesh = self.framebuffer, obj.mesh
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
            return None

        # Per-corner attributes: clip xyzw | world xyz | normal xyz | uv
        n = normals[faces[keep]] * np.where(flip[keep], -1.0, 1.0)[:, None, None]
        uv = mesh.uvs[keep] if textured else np.zeros((len(keep), 3, 2))
        clip = np.concatenate([tri[keep], np.ones((len(keep), 3, 1))], axis=2) @ self.view_proj.T
        corners, src = clip_near(np.concatenate([clip, tri[keep], n, uv], axis=2), camera.near)
        if not len(corners):
            return None
        src = keep[src]
        inv_w = 1.0 / corners[:, :, 3]
        xs = (corners[:, :, 0] * inv_w + 1.0) * 0.5 * fb.width
        ys = (1.0 - corners[:, :, 1] * inv_w) * 0.5 * fb.height
        lod = self._mip_levels(mesh, src, xs, ys, corners[:, :, 10:12]) if textured else None
        return _Prepared(obj, ident, xs, ys, inv_w, corners[:, :, 4:], src, to_linear_rgb(obj.color), lod)

    def _mip_levels(self, mesh, src, xs, ys, uv):
        """Mip level for each triangle: log2 of how many texels span one pixel across it."""
        px_area = np.abs((xs[:, 1] - xs[:, 0]) * (ys[:, 2] - ys[:, 0]) - (ys[:, 1] - ys[:, 0]) * (xs[:, 2] - xs[:, 0]))
        d1, d2 = uv[:, 1] - uv[:, 0], uv[:, 2] - uv[:, 0]
        uv_area = np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])
        texels = np.array([t.shape[0] * t.shape[1] for t in mesh.textures], dtype=float)[mesh.materials[src]]
        ratio = uv_area * texels / np.maximum(px_area, 1e-9)
        return 0.5 * np.log2(np.maximum(ratio, 1e-9)) + self.lod_bias

    # ----- sampling and shading ---------------------------------------------------------

    def _accumulate(self, prepared, pattern, camera, light, pixels=None):
        """Render every sample position in `pattern`, in all pixels or just the given flat pixel indices.

        Returns, per pixel: rgb (summed colour of the covered samples), cover (how
        many samples were covered), spread (largest colour difference between
        samples), mixed (whether they hit different objects), and the depth and
        object id of the nearest sample.
        """
        fb = self.framebuffer
        slots = None
        if pixels is None:
            pixels = np.arange(fb.width * fb.height)
        else:
            slots = np.full(fb.width * fb.height, -1)
            slots[pixels] = np.arange(len(pixels))
        n, m = len(pattern), len(pixels)
        depth = np.zeros((n, m))
        rgb = np.zeros((n * m, 3))
        ids = np.zeros(n * m, np.int32)
        for p in prepared:
            idx, t, _ = rasterize(depth, fb.width, fb.height, p.xs, p.ys, p.inv_w, pattern, slots)
            if not len(idx):
                continue
            # Shade once per pixel and triangle, at the pixel centre, like hardware multisampling:
            # the samples only decide coverage. (Texture detail is smoothed by mipmapping instead.)
            key, inverse = np.unique((idx % m) * len(p.xs) + t, return_inverse=True)
            col, tri = key // len(p.xs), key % len(p.xs)
            px = pixels[col]
            bary = barycentric(p.xs, p.ys, tri, px % fb.width + 0.5, px // fb.width + 0.5)
            rgb[idx] = self._shade(p, tri, bary, camera, light)[inverse]
            ids[idx] = p.ident
        rgb, ids = rgb.reshape(n, m, 3), ids.reshape(n, m)
        nearest = depth.argmax(axis=0)
        return {"pixels": pixels, "rgb": rgb.sum(axis=0), "cover": (ids > 0).sum(axis=0),
                "spread": (rgb.max(axis=0) - rgb.min(axis=0)).max(axis=1), "mixed": (ids != ids[0]).any(axis=0),
                "depth": depth.max(axis=0), "ids": ids[nearest, np.arange(m)]}

    def _shade(self, p, t, bary, camera, light):
        """Linear RGB of fragments: Blinn-Phong with a white highlight, textured if the mesh is."""
        attrs = perspective_interpolate(p.attrs, t, bary, p.inv_w)
        pos, nrm, uvf = attrs[:, 0:3], attrs[:, 3:6], attrs[:, 6:8]
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
        to_light = -normalize(light.direction)
        to_eye = np.asarray(camera.position, dtype=float) - pos
        to_eye /= np.maximum(np.linalg.norm(to_eye, axis=1, keepdims=True), 1e-12)
        half = to_light + to_eye
        half /= np.maximum(np.linalg.norm(half, axis=1, keepdims=True), 1e-12)
        diffuse = light.ambient + light.diffuse * np.clip(nrm @ to_light, 0, None)
        spec = light.specular * np.clip(np.einsum("ij,ij->i", nrm, half), 0, None) ** light.shininess
        albedo = np.broadcast_to(p.albedo, (len(t), 3))
        if p.lod is not None:
            mesh = p.obj.mesh
            albedo = albedo.copy()
            mats = mesh.materials[p.src[t]]
            for m in np.unique(mats):
                sel = mats == m
                albedo[sel] *= sample_texture(mesh.mipmaps(m), uvf[sel, 0], uvf[sel, 1], p.lod[t[sel]])
        # Light levels are perceived brightness (0.5 looks half as bright), as artists tune them, so they
        # are decoded like any sRGB value; everything after this point works in linear light.
        return np.clip(albedo * srgb_to_linear(np.clip(diffuse, 0.0, 1.0))[:, None] + spec[:, None], 0.0, 1.0)

    # ----- post effects ---------------------------------------------------------------------

    def _fog(self, fb, visible):
        """Dim pixels in proportion to how far back they sit within the scene's depth range."""
        if not self.fog:
            return
        dist = 1.0 / np.where(visible, fb.depth, 1.0)
        near, far = dist[visible].min(), dist[visible].max()
        span = max(far - near, 0.25 * near)
        fb.rgb[visible] *= (1.0 - self.fog * (dist[visible] - near) / span)[:, None]

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
        fb.rgb[edge] *= 1.0 - self.outline
