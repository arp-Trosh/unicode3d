# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
"""Loading glTF 2.0 models (.gltf with its .bin and image files, or one .glb): meshes, materials, textures, the
node hierarchy and animations. models.load_model() calls load_gltf() for these files.

What is read:
- Meshes: positions, normals (flat, as glTF asks, where a mesh has none), texture coordinates, vertex colours,
  triangles, strips and fans (points and lines are skipped), sparse accessors, and morph targets at their
  default weights. Skinned meshes are posed by their joints at load (the pose the file leaves them in).
- Materials (PBR, mapped approximately onto an Object3D's): base colour and its texture to color and the texture;
  roughness to highlights (specular, shininess); metallic, much less for rough metal, to reflectivity; emissive to
  emissive; alphaMode OPAQUE, MASK and BLEND to solid, cut-out and see-through; doubleSided. Extensions:
  KHR_texture_transform, KHR_materials_emissive_strength, KHR_materials_unlit (glowing at full colour),
  KHR_mesh_quantization and EXT_texture_webp.
- Nodes: each becomes a Node, or an Object3D where it has a mesh of one part, under the Model's root, with its
  translation, rotation and scale (or matrix): Model.nodes has them by name.
- Animations: translation, rotation and scale channels, with LINEAR, STEP or CUBICSPLINE interpolation, each an
  animation.Clip in Model.animations.

Not read: cameras, lights, normal, occlusion and metallic-roughness textures (which a terminal's pixels can't show
much of), skinned and morph animation (those meshes keep the pose they load in), and compressed meshes and
textures (Draco, meshopt, KTX2/Basis): a file that requires those is refused with a ValueError rather than loaded
wrong. What else can't be loaded (an image that isn't there, a broken accessor) is left out with a warning.
"""
import base64
import json
import os
import struct
import urllib.parse

import numpy as np

from .animation import Animation, Clip, RotationTrack, SplineTrack, Track, step
from .color import linear_to_srgb
from .mesh import Mesh
from .scene import Model, Node, Object3D
from .texture import MAX_TEXTURE, load_image
from .transforms import quat_from_matrix, quat_to_matrix, scale3

__all__ = ["load_gltf"]

GLB_MAGIC, JSON_CHUNK, BIN_CHUNK = b"glTF", 0x4E4F534A, 0x004E4942
COMPONENTS = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
WIDTHS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}
TRIANGLES, TRIANGLE_STRIP, TRIANGLE_FAN = 4, 5, 6
# Extensions read (at least as far as the module's notes say); a file that requires any other is refused.
SUPPORTED = {"KHR_texture_transform", "KHR_materials_emissive_strength", "KHR_materials_unlit",
             "KHR_mesh_quantization", "EXT_texture_webp"}
COMPRESSION = {"KHR_draco_mesh_compression": "Draco-compressed meshes",
               "EXT_meshopt_compression": "meshopt-compressed data", "KHR_meshopt_compression": "meshopt-compressed data",
               "KHR_texture_basisu": "KTX2/Basis textures"}
MAX_ZEROS = 1 << 24  # the most numbers an accessor without data of its own may give
MAX_NODES = 100_000  # deeper or larger node trees than this are taken as broken (a cycle)
# What a broken part of a file raises as it is read: that part is left out, with a warning.
BROKEN = (KeyError, ValueError, TypeError, IndexError, AttributeError)


def _name(value, default):
    """A name from the file, if it is one."""
    return value if isinstance(value, str) and value else default


def _problem(e):
    return e.args[0] if e.args else type(e).__name__


def _find(uri, folder):
    """The file a uri names, relative to `folder` (percent-escapes decoded); also looked for by its name alone in
    the folder, and relative to the folder above (a kit's models often share a Textures folder beside theirs)."""
    name = urllib.parse.unquote(uri).replace("\\", "/")
    for candidate in (name, os.path.basename(name), os.path.join("..", name)):
        path = candidate if os.path.isabs(candidate) else os.path.join(folder, candidate)
        if os.path.isfile(path):
            return path
    return None


def _data_uri(uri):
    """The bytes a data: uri holds."""
    head, _, data = uri.partition(",")
    return base64.b64decode(data) if head.endswith(";base64") else urllib.parse.unquote_to_bytes(data)


def _srgb(linear):
    """Linear colour components (0..1, any shape) as sRGB."""
    return linear_to_srgb(np.clip(np.asarray(linear, dtype=float), 0.0, 1.0))


class _File:
    """A glTF file's JSON and its buffers, read as they are needed."""

    def __init__(self, path):
        self.folder = os.path.dirname(os.path.abspath(path))
        with open(path, "rb") as f:
            data = f.read()
        self.glb = None
        if data[:4] == GLB_MAGIC:
            version, length = struct.unpack_from("<II", data, 4)
            if version != 2:
                raise ValueError(f"{path}: glTF version {version}; only 2 can be loaded")
            chunks, at = {}, 12
            while at + 8 <= min(length, len(data)):
                size, kind = struct.unpack_from("<II", data, at)
                chunks.setdefault(kind, data[at + 8:at + 8 + size])
                at += 8 + size + (-size % 4)
            if JSON_CHUNK not in chunks:
                raise ValueError(f"{path}: a .glb without its JSON")
            text, self.glb = chunks[JSON_CHUNK], chunks.get(BIN_CHUNK)
        else:
            text = data
        try:
            self.json = json.loads(text.decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise ValueError(f"{path}: not a glTF file ({e})") from None
        if not isinstance(self.json, dict):
            raise ValueError(f"{path}: not a glTF file")
        version = str(self.json.get("asset", {}).get("version", "2.0"))
        if not version.startswith("2"):
            raise ValueError(f"{path}: glTF version {version}; only 2.x can be loaded")
        refused = [e for e in self.json.get("extensionsRequired", []) if e not in SUPPORTED]
        if refused:
            what = ", ".join(f"{e} ({COMPRESSION[e]})" if e in COMPRESSION else e for e in refused)
            raise ValueError(f"{path}: needs glTF extensions unicode3d doesn't support: {what}")
        self.warnings = []
        self._buffers = {}

    def get(self, kind, index):
        """Item `index` of the top-level list `kind` (a dict); KeyError if it isn't there."""
        items = self.json.get(kind, [])
        if not isinstance(items, list) or not isinstance(index, int) or not 0 <= index < len(items):
            raise KeyError(f"{kind} {index} doesn't exist")
        if not isinstance(items[index], dict):
            raise KeyError(f"{kind} {index} isn't an object")
        return items[index]

    def buffer(self, index):
        if index not in self._buffers:
            info = self.get("buffers", index)
            uri = info.get("uri")
            if uri is None:
                if self.glb is None:
                    raise KeyError(f"buffer {index} has no data")
                data = self.glb
            elif uri.startswith("data:"):
                data = _data_uri(uri)
            else:
                path = _find(uri, self.folder)
                if path is None:
                    raise KeyError(f"buffer file {uri!r} not found")
                with open(path, "rb") as f:
                    data = f.read()
            self._buffers[index] = data
        return self._buffers[index]

    def view_bytes(self, index):
        """The bytes of a bufferView, and its byteStride (None if packed)."""
        view = self.get("bufferViews", index)
        data = self.buffer(view.get("buffer", 0))
        start, length = int(view.get("byteOffset", 0)), int(view.get("byteLength", 0))
        if start < 0 or length < 0 or start + length > len(data):
            raise ValueError(f"bufferView {index} runs past its buffer")
        return memoryview(data)[start:start + length], view.get("byteStride")

    def _read(self, view, offset, count, dtype, width, stride=None):
        """count elements of `width` components of `dtype` from a bufferView, `offset` bytes in."""
        data, view_stride = self.view_bytes(view)
        size = dtype.itemsize * width
        stride = int(stride or view_stride or size)
        offset = int(offset)
        if count <= 0:
            return np.zeros((0, width), dtype)
        if offset < 0 or stride < size or offset + stride * (count - 1) + size > len(data):
            raise ValueError(f"accessor runs past bufferView {view}")
        array = np.ndarray((count, width), dtype, buffer=data, offset=offset, strides=(stride, dtype.itemsize))
        return array.copy()

    def accessor(self, index, as_float=True):
        """An accessor's elements, (count, components): floats (normalized integers scaled to 0..1 or -1..1),
        or with as_float=False, the integers as they are (indices, joints)."""
        acc = self.get("accessors", index)
        kind = acc.get("type", "SCALAR")
        if acc.get("componentType") not in COMPONENTS or kind not in WIDTHS:
            raise ValueError(f"accessor {index}: unknown component type or type")
        dtype, width, count = np.dtype(COMPONENTS[acc["componentType"]]).newbyteorder("<"), WIDTHS[kind], int(acc["count"])
        if kind in ("MAT2", "MAT3") and dtype.itemsize < 4:
            raise ValueError(f"accessor {index}: padded {kind} of small integers isn't supported")
        if count < 0 or count > 1 << 28:
            raise ValueError(f"accessor {index}: count {count}")
        if "bufferView" in acc:
            values = self._read(acc["bufferView"], acc.get("byteOffset", 0), count, dtype, width)
        else:  # all zeros (unless sparse changes some): as many as the file says, within reason
            if count * width > MAX_ZEROS:
                raise ValueError(f"accessor {index}: {count} elements without data")
            values = np.zeros((count, width), dtype)
        sparse = acc.get("sparse")
        if sparse:
            n, ind, val = int(sparse["count"]), sparse["indices"], sparse["values"]
            itype = np.dtype(COMPONENTS[ind["componentType"]]).newbyteorder("<")
            where = self._read(ind["bufferView"], ind.get("byteOffset", 0), n, itype, 1)[:, 0].astype(np.int64)
            changed = self._read(val["bufferView"], val.get("byteOffset", 0), n, dtype, width)
            if (where < 0).any() or (where >= count).any():
                raise ValueError(f"accessor {index}: sparse indices out of range")
            values[where] = changed
        if not as_float:
            return values.astype(np.int64)
        if acc.get("normalized") and dtype.kind in "iu":
            scale = float(np.iinfo(dtype).max)
            return np.maximum(values.astype(float) / scale, -1.0)
        return values.astype(float)

    def image(self, index, max_texture):
        """A texture's image (see texture.load_image), or None, with a warning, if it can't be read."""
        try:
            texture = self.get("textures", index)
            source = texture.get("source")
            webp = texture.get("extensions", {}).get("EXT_texture_webp", {}).get("source")
            source = webp if webp is not None else source
            if source is None:
                raise KeyError("no image it can show (a compressed format?)")
            image = self.get("images", source)
            if "bufferView" in image:
                data = bytes(self.view_bytes(image["bufferView"])[0])
            elif image.get("uri", "").startswith("data:"):
                data = _data_uri(image["uri"])
            elif "uri" in image:
                path = _find(image["uri"], self.folder)
                if path is None:
                    raise KeyError(f"image file {image['uri']!r} not found")
                data = path
            else:
                raise KeyError("image has no data")
            return load_image(data, max_texture)
        except BROKEN + (OSError,) as e:
            self.warnings.append(f"texture {index}: {_problem(e)}")
            return None


# ----- materials ---------------------------------------------------------------------------

def _uv_transform(info):
    """The 3x3 matrix KHR_texture_transform puts on a texture's coordinates (None if it has none)."""
    t = info.get("extensions", {}).get("KHR_texture_transform")
    if not t:
        return None
    (ox, oy), (sx, sy), r = t.get("offset", (0.0, 0.0)), t.get("scale", (1.0, 1.0)), float(t.get("rotation", 0.0))
    c, s = np.cos(r), np.sin(r)
    return (np.array([[1.0, 0.0, ox], [0.0, 1.0, oy], [0.0, 0.0, 1.0]])
            @ np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]]) @ np.diag([sx, sy, 1.0]))


class _Material:
    """A glTF material as the Object3D settings, texture and texture coordinates it gives."""

    def __init__(self, file, index, max_texture):
        from .models import Material
        info = file.get("materials", index) if index is not None else {}
        pbr = info.get("pbrMetallicRoughness", {})
        ext = info.get("extensions", {})
        factor = np.asarray(pbr.get("baseColorFactor", (1.0, 1.0, 1.0, 1.0)), dtype=float).reshape(4)
        metal = float(np.clip(pbr.get("metallicFactor", 1.0), 0.0, 1.0))
        rough = float(np.clip(pbr.get("roughnessFactor", 1.0), 0.0, 1.0))
        self.alpha_mode = info.get("alphaMode", "OPAQUE")
        self.cutoff = float(info.get("alphaCutoff", 0.5))
        m = Material(_name(info.get("name"), f"material {index}" if index is not None else "default"))
        m.color = tuple(float(c) for c in _srgb(factor[:3]))
        # Highlights from roughness: none on rough things, tight and strong on smooth ones (metal more so), with
        # the usual match between a roughness and a Blinn-Phong exponent, 2 / roughness^4 - 2.
        m.specular = (1.0 - rough) * (1.0 + metal)
        m.shininess = float(np.clip(2.0 / max(rough, 0.03) ** 4 - 2.0, 2.0, 1000.0))
        # Metal reflects, less as it gets rougher; squared, as reflections here take no tint from the metal's colour,
        # and would wash out gold or copper that is anything but polished.
        m.reflectivity = metal * (1.0 - rough) ** 2
        emissive = np.asarray(info.get("emissiveFactor", (0.0, 0.0, 0.0)), dtype=float).reshape(3)
        strength = float(ext.get("KHR_materials_emissive_strength", {}).get("emissiveStrength", 1.0))
        m.emissive = float(max(emissive.max(), 0.0) * strength)
        if m.emissive > 0 and "emissiveTexture" in info:  # glowing only in places: as much as it does on average
            glow = file.image(info["emissiveTexture"].get("index"), 64)
            m.emissive *= float(glow[..., :3].mean()) if glow is not None else 1.0
        if "KHR_materials_unlit" in ext:
            m.emissive, m.specular, m.reflectivity = 1.0, 0.0, 0.0
        m.opacity = float(np.clip(factor[3], 0.0, 1.0)) if self.alpha_mode == "BLEND" else 1.0
        self.texcoord, self.uv_matrix = 0, None
        base = pbr.get("baseColorTexture")
        if base is not None:
            image = file.image(base.get("index"), max_texture)
            if image is not None:
                m.texture = self._alpha(image, factor[3])
                self.texcoord = int(base.get("extensions", {}).get("KHR_texture_transform", {})
                                    .get("texCoord", base.get("texCoord", 0)))
                self.uv_matrix = _uv_transform(base)
        if self.alpha_mode == "MASK" and m.texture is None and factor[3] < self.cutoff:
            m.opacity = 0.0  # cut out everywhere
        self.material = m
        self.double_sided = bool(info.get("doubleSided", False))

    def _alpha(self, image, factor_alpha):
        """The base colour texture with its alpha as the alpha mode says: none (OPAQUE), all or nothing at the
        cutoff (MASK), or as it is (BLEND)."""
        if image.shape[2] < 4 or self.alpha_mode == "OPAQUE":
            return image[..., :3]
        if self.alpha_mode == "MASK":
            image = image.copy()
            image[..., 3] = (image[..., 3] * factor_alpha >= self.cutoff).astype(float)
        return image

    def options(self):
        options = self.material.object_options()
        options["double_sided"] = self.double_sided
        return options


# ----- meshes ------------------------------------------------------------------------------

def _triangles(indices, mode):
    """Triangles (F, 3) from a primitive's vertex indices, by its mode."""
    n = len(indices)
    if mode == TRIANGLES:
        return indices[:n - n % 3].reshape(-1, 3)
    if n < 3:
        return np.zeros((0, 3), np.int64)
    i = np.arange(n - 2)
    if mode == TRIANGLE_STRIP:  # every other one turned round, so they all wind the same way
        odd = i % 2
        return np.stack([indices[i], indices[i + 1 + odd], indices[i + 2 - odd]], axis=1)
    return np.stack([indices[i + 1], indices[i + 2], np.full(n - 2, indices[0])], axis=1)  # a fan


def _primitive(file, prim, material, weights, skin):
    """A primitive as a Mesh (None if it has no triangles). weights: the mesh's morph target weights; skin: None, or
    (joint matrices (J, 4, 4), which posed the vertices)."""
    mode = prim.get("mode", TRIANGLES)
    if mode not in (TRIANGLES, TRIANGLE_STRIP, TRIANGLE_FAN):
        return None
    attributes = prim.get("attributes", {})
    if "POSITION" not in attributes:
        raise ValueError("no positions")
    positions = file.accessor(attributes["POSITION"])[:, :3]
    count = len(positions)

    def attribute(name, width):
        if name not in attributes:
            return None
        values = file.accessor(attributes[name])
        if len(values) != count or values.shape[1] < width:
            file.warnings.append(f"{name} ignored: {len(values)} values for {count} vertices")
            return None
        return values

    normals = attribute("NORMAL", 3)
    normals = None if normals is None else normals[:, :3]
    targets = prim.get("targets", [])
    for target, weight in zip(targets, weights or []):  # morph targets, at their default weights
        if weight:
            if "POSITION" in target:
                positions = positions + weight * file.accessor(target["POSITION"])[:count, :3]
            if "NORMAL" in target and normals is not None:
                normals = normals + weight * file.accessor(target["NORMAL"])[:count, :3]
    if skin is not None:
        joints, weights4 = attribute("JOINTS_0", 4), attribute("WEIGHTS_0", 4)
        if joints is not None and weights4 is not None:
            joints = np.clip(joints[:, :4].astype(np.int64), 0, len(skin) - 1)
            blend = np.einsum("vk,vkij->vij", weights4[:, :4], skin[joints])  # (V, 4, 4)
            positions = np.einsum("vij,vj->vi", blend[:, :3, :3], positions) + blend[:, :3, 3]
            if normals is not None:
                normals = np.einsum("vij,vj->vi", blend[:, :3, :3], normals)
    colors = attribute("COLOR_0", 3)
    if colors is not None:
        rgb = _srgb(colors[:, :3])
        keep_alpha = colors.shape[1] == 4 and material.alpha_mode == "BLEND"
        colors = np.concatenate([rgb, np.clip(colors[:, 3:4], 0.0, 1.0)], axis=1) if keep_alpha else rgb
    uv = attribute(f"TEXCOORD_{material.texcoord}", 2) if material.material.texture is not None else None

    indices = (file.accessor(prim["indices"], as_float=False)[:, 0] if "indices" in prim
               else np.arange(count, dtype=np.int64))
    faces = _triangles(indices, mode)
    ok = ((faces >= 0) & (faces < count)).all(axis=1)
    if not ok.all():
        file.warnings.append(f"{int((~ok).sum())} triangles with missing vertices left out")
        faces = faces[ok]
    if not len(faces):
        return None
    if normals is None:  # flat, as glTF asks: each triangle with corners of its own
        corners = faces.reshape(-1)
        positions, faces = positions[corners], np.arange(len(corners), dtype=np.int64).reshape(-1, 3)
        colors = None if colors is None else colors[corners]
        uv = None if uv is None else uv[corners]
    used, local = np.unique(faces, return_inverse=True)  # only the vertices the triangles use
    faces = local.reshape(-1, 3).astype(np.int64)
    mesh = Mesh(positions[used], faces, vertex_colors=None if colors is None else colors[used],
                normals=None if normals is None else normals[used])
    if uv is not None:
        uv = uv[used][:, :2]
        if material.uv_matrix is not None:
            uv = uv @ material.uv_matrix[:2, :2].T + material.uv_matrix[:2, 2]
        uv = np.stack([uv[:, 0], 1.0 - uv[:, 1]], axis=1)  # glTF's v runs down the image, unicode3d's up it
        mesh.uvs = uv[faces]
        mesh.materials = np.zeros(len(faces), np.int64)
        mesh.textures = [material.material.texture]
    return mesh


# ----- nodes ---------------------------------------------------------------------------------

def _local(info):
    """A node's (position, rotation (w first), scale: a number, or three) from its TRS or its matrix."""
    if "matrix" in info:
        m = np.asarray(info["matrix"], dtype=float).reshape(4, 4).T  # column-major
        linear = m[:3, :3]
        scale = np.linalg.norm(linear, axis=0)
        if np.linalg.det(linear) < 0:
            scale[0] = -scale[0]
        rotation = quat_from_matrix(linear / np.where(np.abs(scale) > 1e-12, scale, 1.0))
        position = m[:3, 3].copy()
    else:
        position = np.asarray(info.get("translation", (0.0, 0.0, 0.0)), dtype=float).reshape(3)
        x, y, z, w = np.asarray(info.get("rotation", (0.0, 0.0, 0.0, 1.0)), dtype=float).reshape(4)
        rotation = np.array([w, x, y, z])
        scale = np.asarray(info.get("scale", (1.0, 1.0, 1.0)), dtype=float).reshape(3)
    norm = np.linalg.norm(rotation)
    rotation = rotation / norm if np.isfinite(norm) and norm > 1e-12 else np.array([1.0, 0.0, 0.0, 0.0])
    return position, rotation, (float(scale[0]) if np.all(scale == scale[0]) else scale)


def _matrix(position, rotation, scale):
    m = np.eye(4)
    m[:3, :3] = quat_to_matrix(rotation) * scale3(scale)
    m[:3, 3] = position
    return m


def _tracks(file, sampler, path):
    """A Track for an animation sampler moving `path` (translation, rotation or scale)."""
    times = file.accessor(sampler["input"])[:, 0]
    values = file.accessor(sampler["output"])
    method = sampler.get("interpolation", "LINEAR")
    width = 4 if path == "rotation" else 3
    values = values[:, :width]
    if path == "rotation":
        values = values[:, [3, 0, 1, 2]]  # glTF's x, y, z, w as w, x, y, z
    if not len(times):
        raise ValueError("no keyframes")
    if method == "CUBICSPLINE":
        if len(values) != 3 * len(times):
            raise ValueError(f"{len(values)} values for {len(times)} spline keyframes")
        keys = [(t, values[3 * k + 1], values[3 * k], values[3 * k + 2]) for k, t in enumerate(times)]
        if path == "scale" or path == "translation":
            return SplineTrack(keys)
        return SplineTrack(keys, rotation=True)
    if len(values) != len(times):
        raise ValueError(f"{len(values)} values for {len(times)} keyframes")
    keys = list(zip(times, values))
    easing = step if method == "STEP" else "linear"
    return RotationTrack(keys, easing=easing) if path == "rotation" else Track(keys, easing=easing)


def load_gltf(path, max_texture=MAX_TEXTURE, double_sided=False, scene=None):
    """A glTF 2.0 file (.gltf or .glb) as a Model: its nodes as Nodes and Object3Ds under the model's root, its
    materials (models.Material) and its animations (animation.Clip). See the module's notes for what is read.

    max_texture: textures are shrunk to at most this many texels across (None: as they are).
    double_sided: draw the back of every face, whatever its material says.
    scene: which of the file's scenes (an index; None: the one it names as the default, or else the first).

    Raises ValueError for a file that isn't glTF 2, is broken in its structure, or needs an extension unicode3d
    doesn't support (such as Draco compression); OSError if it can't be read. Smaller problems are listed in
    Model.warnings.
    """
    try:
        return _load(path, max_texture, double_sided, scene)
    except BROKEN as e:
        if isinstance(e, ValueError) and str(e).startswith(str(path)):
            raise
        raise ValueError(f"{path}: not a well-formed glTF file ({type(e).__name__}: {_problem(e)})") from e


def _load(path, max_texture, double_sided, scene):
    file = _File(path)
    nodes = file.json.get("nodes", [])
    scenes = file.json.get("scenes", [])
    index = scene if scene is not None else file.json.get("scene", 0)
    if scenes:
        if not isinstance(index, int) or not 0 <= index < len(scenes):
            raise ValueError(f"{path}: scene {index} doesn't exist")
        roots = list(scenes[index].get("nodes", []))
    else:  # no scenes: every node that is no other's child
        children = {c for n in nodes for c in n.get("children", [])}
        roots = [i for i in range(len(nodes)) if i not in children]

    root = Node()
    model = Model(root, [])
    materials, meshes = {}, {}

    def material(i):
        if i not in materials:
            try:
                materials[i] = _Material(file, i, max_texture)
            except BROKEN as e:
                file.warnings.append(f"material {i}: {_problem(e)}; left plain")
                materials[i] = _Material(file, None, max_texture)
        return materials[i]

    # The nodes, parents first: each one's (Node or Object3D, world matrix at rest).
    built, world, order = {}, {}, []
    todo = [(i, None, np.eye(4)) for i in reversed(roots)]
    while todo:
        i, parent, parent_world = todo.pop()
        if i in built or not isinstance(i, int) or not 0 <= i < len(nodes) or len(built) >= MAX_NODES:
            if i in built:
                file.warnings.append(f"node {i} appears twice in the tree; the second left out")
            continue
        info = nodes[i]
        position, rotation, scale = _local(info)
        world[i] = parent_world @ _matrix(position, rotation, scale)
        built[i] = Node(position, rotation, scale, parent=parent)
        order.append(i)
        todo += [(c, built[i], world[i]) for c in reversed(info.get("children", []))]

    skinned = set()  # joints that move skinned meshes, which animations can't
    for i in order:
        info, node = nodes[i], built[i]
        name = _name(info.get("name"), f"node {i}")
        if "mesh" not in info:
            model.nodes.setdefault(name, node)
            continue
        try:
            mesh_info = file.get("meshes", info["mesh"])
        except KeyError as e:
            file.warnings.append(f"node {name}: {e.args[0]}")
            continue
        weights = info.get("weights", mesh_info.get("weights"))
        skin, parent = None, node
        if "skin" in info:
            try:
                skin_info = file.get("skins", info["skin"])
                joints = skin_info["joints"]
                inverse = (file.accessor(skin_info["inverseBindMatrices"]).reshape(-1, 4, 4).transpose(0, 2, 1)
                           if "inverseBindMatrices" in skin_info else np.tile(np.eye(4), (len(joints), 1, 1)))
                skin = np.stack([world.get(j, np.eye(4)) @ inverse[k] for k, j in enumerate(joints)])
                skinned.update(joints)
                parent = root  # posed by its joints, in the scene's space: the node's own place doesn't count
            except BROKEN as e:
                file.warnings.append(f"node {name}: skin left out ({_problem(e)})")
        parts = []
        for p, prim in enumerate(mesh_info.get("primitives", [])):
            mat = material(prim.get("material"))
            key = (info["mesh"], p, repr(info.get("skin")), repr(weights))
            try:
                if key not in meshes or skin is not None:
                    meshes[key] = _primitive(file, prim, mat, weights, skin)
            except BROKEN as e:
                file.warnings.append(f"mesh {_name(mesh_info.get('name'), info['mesh'])} part {p}: {_problem(e)}")
                meshes[key] = None
            if meshes[key] is not None:
                parts.append((meshes[key], mat))
        if len(parts) == 1 and parent is node:  # the node is the part: one object, not a node and an object
            mesh, mat = parts[0]
            obj = Object3D(mesh, node.position, node.rotation, node.scale, parent=node.parent, **mat.options())
            built[i] = obj
            for c in info.get("children", []):
                if c in built and built[c].parent is node:
                    built[c].parent = obj
            objects = [obj]
        else:
            objects = [Object3D(mesh, parent=parent, **mat.options()) for mesh, mat in parts]
        if double_sided:
            for obj in objects:
                obj.double_sided = True
        model.nodes.setdefault(name, built[i])
        model.objects += objects
        for label in (name, _name(mesh_info.get("name"), None)):
            if label:
                model.names.setdefault(label, []).extend(objects)
        for obj, (_, mat) in zip(objects, parts):
            model.names.setdefault(mat.material.name, []).append(obj)
    for i in order:
        if built[i].parent is None:
            built[i].parent = root
    model.materials = {m.material.name: m.material for m in materials.values() if m.material.name != "default"}

    for a, info in enumerate(file.json.get("animations", [])):
        name = _name(info.get("name"), f"animation {a}")
        tracks, morph, joints = {}, False, False
        for channel in info.get("channels", []):
            try:
                target = channel["target"]
                node, attr = target.get("node"), target.get("path")
                if attr == "weights":
                    morph = True
                    continue
                if not isinstance(node, int) or node not in built or attr not in ("translation", "rotation", "scale"):
                    continue
                joints = joints or node in skinned
                sampler = info["samplers"][channel["sampler"]]
                tracks.setdefault(node, {})[{"translation": "position"}.get(attr, attr)] = _tracks(file, sampler, attr)
            except BROKEN as e:
                file.warnings.append(f"animation {name}: a channel left out ({_problem(e)})")
        if joints:
            file.warnings.append(f"animation {name}: moves the joints of a skinned mesh, which stays as it loaded "
                                 "(skinned animation isn't supported)")
        if morph:
            file.warnings.append(f"animation {name}: morph target weights aren't animated")
        if tracks:
            model.animations[name] = Clip([Animation(built[n], **t) for n, t in tracks.items()], name=name)
    model.warnings = file.warnings
    return model
