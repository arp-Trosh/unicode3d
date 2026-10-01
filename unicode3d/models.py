# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
"""Loading models from files: Wavefront OBJ, with its MTL materials and their textures, and glTF 2.0 (gltf.py).

load_model() gives a Model (scene.py): an Object3D for each material, all under one Node, with the material's
colour, texture, highlights, glow and opacity. load_obj() gives the whole file as one Mesh, keeping the colours
and textures but not the rest of the materials (a Mesh can only carry those).

What is read from an OBJ: positions (with colours, if each `v` has an r g b after x y z), texture coordinates,
normals, faces (polygons are split into triangles; negative indices count back from the last), `usemtl` and
`mtllib`, groups (`o` and `g`) and smoothing groups (`s`: faces in none are flat, if the file has no normals).
Lines, points and curves are skipped.

From an MTL: Kd colour, Ks and Ns highlights, Ke glow, d or Tr opacity, illum 0 and 1 (no highlights),
map_Kd texture (with -o and -s, which move and scale it), map_d cut-outs and, from the PBR extension, Pm and Pr
(metal reflects; rough metal less). Texture paths are taken relative to the MTL file.
"""
import os
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from .mesh import Mesh
from .scene import Model, Node, Object3D
from .texture import MAX_TEXTURE, load_image

__all__ = ["Material", "load_model", "load_mtl", "load_obj"]


@dataclass
class Material:
    """A material from an MTL file, as an Object3D takes it."""
    name: str
    color: tuple = (0.8, 0.8, 0.8)  # Kd, 0..1 sRGB
    specular: float = 1.0           # Ks (its average), or 0 for illum 0 and 1
    shininess: float = None         # Ns (None: the lights' own)
    emissive: float = 0.0           # Ke (its brightest channel)
    opacity: float = 1.0            # d, or 1 - Tr
    reflectivity: float = 0.0       # Pm, less for rough metal (Pr)
    texture: np.ndarray = None      # map_Kd, with map_d's alpha: (H, W, 3) or (H, W, 4), 0..1 sRGB
    uv_scale: tuple = (1.0, 1.0)    # map_Kd -s
    uv_offset: tuple = (0.0, 0.0)   # map_Kd -o
    files: list = field(default_factory=list)  # the image files it was loaded from

    def object_options(self):
        """Its settings as keyword arguments for Object3D."""
        return {"color": tuple(float(c) for c in self.color), "specular": float(self.specular),
                "shininess": self.shininess, "emissive": float(self.emissive), "opacity": float(self.opacity),
                "reflectivity": float(self.reflectivity)}


# How many arguments each option of a map_ statement takes (-o, -s and -t take one to three numbers).
_MAP_OPTIONS = {"-blendu": 1, "-blendv": 1, "-bm": 1, "-boost": 1, "-cc": 1, "-clamp": 1, "-imfchan": 1,
                "-mm": 2, "-texres": 1, "-type": 1, "-o": 3, "-s": 3, "-t": 3}


def _map_statement(args):
    """A map_ statement's arguments as (file name, options {name: [values]})."""
    options, i = {}, 0
    while i < len(args) and args[i].lower() in _MAP_OPTIONS:
        name, n = args[i].lower(), _MAP_OPTIONS[args[i].lower()]
        i += 1
        values = []
        while len(values) < n and i < len(args) - 1:  # (the file name is always left)
            if n == 3:
                try:
                    values.append(float(args[i]))
                except ValueError:
                    break
            else:
                values.append(args[i])
            i += 1
        options[name] = values
    return " ".join(args[i:]), options


def _find_file(name, folder):
    """The file a material or model refers to, relative to `folder`; None if it can't be found. Files made on
    Windows often use backslashes, and some name a path that only existed where they were made: then the file
    is looked for by its name alone, in the folder, and relative to the folder above (a kit's models often share
    a Textures folder beside theirs)."""
    name = name.strip().strip('"')
    if not name:
        return None
    candidates = [name, name.replace("\\", "/")]
    candidates += [os.path.basename(candidates[-1]), os.path.join("..", candidates[-1])]
    for c in candidates:
        path = c if os.path.isabs(c) else os.path.join(folder, c)
        if os.path.isfile(path):
            return path
    return None


def _floats(args, n, default):
    """The first n numbers of args (repeating the first if fewer are given, as MTL does)."""
    try:
        values = [float(a) for a in args[:n]]
    except ValueError:
        return default
    if not values:
        return default
    return tuple(values + [values[0]] * (n - len(values)))


def load_mtl(path, max_texture=MAX_TEXTURE, warnings=None):
    """The materials in an MTL file, {name: Material}, with their textures loaded (shrunk to at most max_texture
    texels across). What can't be loaded (a texture that isn't there, a line that makes no sense) is skipped,
    with a message added to `warnings` (a list) if given."""
    warnings = [] if warnings is None else warnings
    folder = os.path.dirname(os.path.abspath(path))
    materials, raw = {}, {}
    current = None
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.split()
            if not parts or parts[0].startswith("#"):
                continue
            key, args = parts[0], parts[1:]
            if key == "newmtl":
                current = raw[" ".join(args)] = {}
            elif current is not None:
                if key.lower().startswith("map_") or key.lower() in ("bump", "disp", "decal", "refl", "norm"):
                    current[key.lower()] = args
                else:
                    current[key] = args
    for name, statements in raw.items():
        m = Material(name)

        def number(key, default=None):
            value = _floats(statements.get(key, []), 1, None)
            return default if value is None else value[0]

        if "Kd" in statements:
            m.color = tuple(float(np.clip(c, 0.0, 1.0)) for c in _floats(statements["Kd"], 3, m.color))
        if "Ks" in statements:
            m.specular = float(np.mean(np.clip(_floats(statements["Ks"], 3, (1.0,) * 3), 0.0, None)))
        illum = number("illum")
        if illum is not None and illum < 2:
            m.specular = 0.0
        if "Ns" in statements:
            m.shininess = float(np.clip(number("Ns", 1.0), 1.0, 1000.0))
        if "Ke" in statements:
            m.emissive = float(max(max(_floats(statements["Ke"], 3, (0.0,) * 3)), 0.0))
        if "d" in statements:
            m.opacity = float(np.clip(number("d", 1.0), 0.0, 1.0))
        elif "Tr" in statements:
            m.opacity = float(np.clip(1.0 - number("Tr", 0.0), 0.0, 1.0))
        metal, rough = number("Pm"), number("Pr")
        if metal is not None:
            m.reflectivity = float(np.clip(metal * (1.0 - np.clip(rough or 0.0, 0.0, 1.0)), 0.0, 1.0))
            if m.shininess is None and rough is not None:
                # The usual match between a roughness and a Blinn-Phong exponent: 2 / roughness^4 - 2.
                m.shininess = float(np.clip(2.0 / max(rough, 0.03) ** 4 - 2.0, 1.0, 1000.0))
        _load_maps(m, statements, folder, max_texture, warnings)
        materials[name] = m
    return materials


def _load_maps(m, statements, folder, max_texture, warnings):
    """Material m's texture: map_Kd for colour, and map_d for its alpha (holes, or see-through parts)."""
    colour = alpha = None
    for key in ("map_kd", "map_d"):
        if key not in statements:
            continue
        name, options = _map_statement(statements[key])
        path = _find_file(name, folder)
        if path is None:
            warnings.append(f"material {m.name}: {key} {name!r} not found")
            continue
        try:
            image = load_image(path, max_texture)
        except (OSError, ValueError) as e:
            warnings.append(f"material {m.name}: {key} {name!r} could not be read ({e})")
            continue
        m.files.append(path)
        if key == "map_kd":
            colour = image
            scale, offset = options.get("-s", []), options.get("-o", [])
            m.uv_scale = tuple((scale + [1.0, 1.0])[:2]) if scale else (1.0, 1.0)
            m.uv_offset = tuple((offset + [0.0, 0.0])[:2]) if offset else (0.0, 0.0)
        else:  # its alpha channel, or its brightness if it has none
            alpha = image[..., 3] if image.shape[2] == 4 else image[..., :3].mean(axis=2)
    if alpha is not None:
        if colour is None:
            colour = np.ones(alpha.shape + (3,))
        elif alpha.shape != colour.shape[:2]:
            img = Image.fromarray(np.round(alpha * 255).astype(np.uint8))
            alpha = np.asarray(img.resize((colour.shape[1], colour.shape[0]), Image.Resampling.BILINEAR)) / 255.0
        colour = np.concatenate([colour[..., :3], alpha[..., None]], axis=2)
    m.texture = colour


@dataclass
class _Parsed:
    """An OBJ file's contents: vertices (deduplicated by position and normal), triangles, and what they use."""
    positions: np.ndarray       # (V, 3) of the vertices
    normals: np.ndarray         # (V, 3), NaN where the file gives none
    colors: np.ndarray          # (V, 3) or None
    faces: np.ndarray           # (F, 3) into the vertices
    uvs: np.ndarray             # (F, 3, 2), 0 where the file gives none
    part: np.ndarray            # (F,) index into part_keys
    part_keys: list             # (group name, material name) of each part
    materials: dict             # name: Material, from the MTL files
    warnings: list


def _index(token, count):
    """An OBJ index (1-based, or negative counting back from the last) as 0-based; None if missing or out of range."""
    if not token:
        return None
    i = int(token)
    i = i - 1 if i > 0 else count + i
    return i if 0 <= i < count else None


def _parse(path, max_texture, split_groups):
    folder = os.path.dirname(os.path.abspath(path))
    v, vc, vt, vn = [], [], [], []
    keys, vertex_pos, vertex_norm = {}, [], []
    faces, face_uv, face_part = [], [], []
    part_of, part_keys = {}, []
    materials, warnings = {}, []
    material, group, smoothing, smoothing_seen = None, None, 1, False
    with open(path, encoding="utf-8", errors="replace") as f:
        pending = ""
        for number, line in enumerate(f, 1):
            line = pending + line
            if line.rstrip().endswith("\\"):
                pending = line.rstrip()[:-1] + " "
                continue
            pending = ""
            parts = line.split()
            if not parts:
                continue
            key = parts[0]
            try:
                if key == "v":
                    v.append((float(parts[1]), float(parts[2]), float(parts[3])))
                    if len(parts) >= 7:
                        vc.append((float(parts[4]), float(parts[5]), float(parts[6])))
                elif key == "vt":
                    vt.append((float(parts[1]), float(parts[2]) if len(parts) > 2 else 0.0))
                elif key == "vn":
                    vn.append((float(parts[1]), float(parts[2]), float(parts[3])))
                elif key == "f":
                    corners = []
                    for token in parts[1:]:
                        fields = token.split("/")
                        pi = _index(fields[0], len(v))
                        if pi is None:
                            raise ValueError(f"vertex {fields[0]} doesn't exist")
                        ti = _index(fields[1], len(vt)) if len(fields) > 1 else None
                        ni = _index(fields[2], len(vn)) if len(fields) > 2 else None
                        if ni is not None:
                            k = (pi, ni)
                        elif smoothing or not smoothing_seen:
                            k = (pi, -1, smoothing)
                        else:  # smoothing off: a vertex of its own, so the face is flat
                            k = (pi, -1, -1 - len(faces))
                        if k not in keys:
                            keys[k] = len(vertex_pos)
                            vertex_pos.append(pi)
                            vertex_norm.append(-1 if ni is None else ni)
                        corners.append((keys[k], -1 if ti is None else ti))
                    if len(corners) < 3:
                        continue
                    pk = (group if split_groups else None, material)
                    if pk not in part_of:
                        part_of[pk] = len(part_keys)
                        part_keys.append(pk)
                    for j in range(1, len(corners) - 1):  # a fan of triangles
                        a, b, c = corners[0], corners[j], corners[j + 1]
                        faces.append((a[0], b[0], c[0]))
                        face_uv.append((a[1], b[1], c[1]))
                        face_part.append(part_of[pk])
                elif key == "usemtl":
                    material = " ".join(parts[1:])
                elif key == "mtllib":
                    rest = " ".join(parts[1:])
                    names = [rest] if _find_file(rest, folder) else parts[1:]
                    for name in names:
                        found = _find_file(name, folder)
                        if found is None:
                            warnings.append(f"mtllib {name!r} not found")
                            continue
                        materials.update(load_mtl(found, max_texture, warnings))
                elif key in ("o", "g"):
                    group = " ".join(parts[1:]) or None
                elif key == "s":
                    smoothing_seen = True
                    value = parts[1].lower() if len(parts) > 1 else "off"
                    smoothing = 0 if value in ("off", "0") else (int(value) if value.isdigit() else 1)
            except (ValueError, IndexError) as e:
                warnings.append(f"line {number}: {key} skipped ({e})")
    missing = sorted({k[1] for k in part_keys if k[1] is not None and k[1] not in materials})
    for name in missing:
        warnings.append(f"material {name!r} not defined in any mtllib")
    positions = np.array(v, dtype=float).reshape(-1, 3)
    normals = np.vstack([np.array(vn, dtype=float).reshape(-1, 3), np.full((1, 3), np.nan)])
    vertex_pos, vertex_norm = np.array(vertex_pos, np.int64), np.array(vertex_norm, np.int64)
    uv_table = np.vstack([np.array(vt, dtype=float).reshape(-1, 2), np.zeros((1, 2))])
    face_uv = np.array(face_uv, np.int64).reshape(-1, 3)
    colors = None
    if vc and len(vc) == len(v):
        colors = np.array(vc, dtype=float)[vertex_pos]
    return _Parsed(positions[vertex_pos].reshape(-1, 3), normals[vertex_norm].reshape(-1, 3), colors,
                   np.array(faces, np.int64).reshape(-1, 3), uv_table[face_uv], np.array(face_part, np.int64),
                   part_keys, materials, warnings)


def _uvs(parsed, faces, material):
    """The faces' texture coordinates, moved and scaled as the material's map_Kd says."""
    uvs = parsed.uvs[faces]
    if material is not None and (material.uv_scale != (1.0, 1.0) or material.uv_offset != (0.0, 0.0)):
        uvs = uvs * np.asarray(material.uv_scale) + np.asarray(material.uv_offset)
    return uvs


def _submesh(parsed, faces):
    """A Mesh of just the given faces, with only the vertices they use."""
    used, local = np.unique(parsed.faces[faces], return_inverse=True)
    normals = parsed.normals[used]
    return Mesh(parsed.positions[used], local.reshape(-1, 3).astype(np.int64),
                vertex_colors=None if parsed.colors is None else parsed.colors[used],
                normals=None if np.isnan(normals).all() else normals)


def load_model(path, split_groups=False, max_texture=MAX_TEXTURE, double_sided=False):
    """A model file as a Model (scene.py): Wavefront .obj, or glTF 2.0 (.gltf or .glb, see gltf.load_gltf, which
    also gives the file's nodes and animations).

    From an OBJ: an Object3D for each material, with the material's colour, texture, highlights, glow, opacity
    and reflectivity (see the module's notes on what is read), all with the Model's root as their parent. Each
    part's mesh keeps the model's own coordinates, so the parts fit together.

    split_groups (OBJ): an Object3D for each group (`o` or `g`) and material, rather than for each material (so the
    parts can move on their own: a door, a wheel), named by their group in Model.names. A file with thousands
    of groups is slower to draw split. (A glTF file is in parts already: its nodes.)
    max_texture: textures are shrunk to at most this many texels across (None: as they are).
    double_sided: draw the back of every face too (for models whose faces don't all wind the same way).

    Problems that leave part of the model out (a missing texture, a bad line) are listed in Model.warnings; a
    file that can't be read at all raises OSError, and one in another format (or needing glTF extensions that
    aren't supported) ValueError.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in (".gltf", ".glb"):
        from .gltf import load_gltf
        return load_gltf(path, max_texture=max_texture, double_sided=double_sided)
    if ext != ".obj":
        raise ValueError(f"{path}: only Wavefront .obj and glTF (.gltf, .glb) models can be loaded")
    parsed = _parse(path, max_texture, split_groups)
    root = Node()
    model = Model(root, [], {}, parsed.materials, parsed.warnings)
    for p, (group, name) in enumerate(parsed.part_keys):
        faces = np.flatnonzero(parsed.part == p)
        if not len(faces):
            continue
        material = parsed.materials.get(name)
        mesh = _submesh(parsed, faces)
        options = material.object_options() if material is not None else {"color": (0.8, 0.8, 0.8)}
        if material is not None and material.texture is not None:
            mesh.uvs = _uvs(parsed, faces, material)
            mesh.materials = np.zeros(len(faces), np.int64)
            mesh.textures = [material.texture]
        obj = Object3D(mesh, parent=root, double_sided=double_sided, **options)
        model.objects.append(obj)
        for label in (group, name):
            if label is not None:
                model.names.setdefault(label, []).append(obj)
    return model


def load_obj(path, max_texture=MAX_TEXTURE):
    """A Wavefront OBJ file as one Mesh: its vertices, faces, normals (worked out where the file has none),
    vertex colours if it has them, and, if it uses materials, their colours (as face_colors, with their
    opacity) and textures. The rest of a material (highlights, glow, reflectivity) needs an Object3D for each
    material: load_model() makes those."""
    parsed = _parse(path, max_texture, split_groups=False)
    mesh = Mesh(parsed.positions, parsed.faces, vertex_colors=parsed.colors,
                normals=None if np.isnan(parsed.normals).all() else parsed.normals)
    materials = [parsed.materials.get(name) for _, name in parsed.part_keys]
    if any(m is not None for m in materials):
        rgba = np.array([(*m.color, m.opacity) if m is not None else (0.8, 0.8, 0.8, 1.0) for m in materials])
        mesh.face_colors = rgba[parsed.part].reshape(-1, 4)
    if any(m is not None and m.texture is not None for m in materials):
        mesh.uvs = np.empty((len(parsed.faces), 3, 2))
        mesh.textures, table = [], np.zeros(len(materials), np.int64)
        for p, m in enumerate(materials):
            faces = parsed.part == p
            if m is not None and m.texture is not None:
                table[p] = len(mesh.textures)
                mesh.textures.append(m.texture)
            else:
                table[p] = -1
            mesh.uvs[faces] = _uvs(parsed, np.flatnonzero(faces), m)
        if (table < 0).any():  # untextured materials: plain white, which leaves their colour as it is
            table[table < 0] = len(mesh.textures)
            mesh.textures.append(np.ones((1, 1, 3)))
        mesh.materials = table[parsed.part]
    return mesh
