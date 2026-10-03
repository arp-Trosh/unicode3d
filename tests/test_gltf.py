# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
"""Loading glTF 2.0 models (.glb and .gltf): meshes, materials, nodes and animations; and the animation pieces
they use (Clip, SplineTrack, the step easing)."""
import base64
import io
import json
import os
import struct
import tempfile
import unittest
import warnings

import numpy as np
from PIL import Image

from unicode3d import (Animation, Camera, Clip, Light, Node, Object3D, Renderer, SplineTrack, Track, load_gltf,
                       load_model)
from unicode3d.animation import step
from unicode3d.color import linear_to_srgb
from unicode3d.texture import BLEND, CUTOUT, alpha_kind, build_mipmaps
from unicode3d.transforms import quat_axis_angle, quat_to_matrix

FLOAT, UBYTE, USHORT, UINT = 5126, 5121, 5123, 5125
WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
DTYPE = {FLOAT: "<f4", UBYTE: "u1", USHORT: "<u2", UINT: "<u4"}

# A unit quad facing +z, in the xy plane: corners (-1, -1), (1, -1), (1, 1), (-1, 1).
QUAD = np.array([(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)], float)
QUAD_UV = np.array([(0, 1), (1, 1), (1, 0), (0, 0)], float)  # glTF's v runs down the image
QUAD_FACES = np.array([0, 1, 2, 0, 2, 3])


def png(pixels):
    """PNG bytes of an (H, W, 3 or 4) uint8 image."""
    out = io.BytesIO()
    Image.fromarray(np.asarray(pixels, np.uint8)).save(out, "PNG")
    return out.getvalue()


def _png_chunk(kind, data):
    """A PNG chunk: length, kind, data and checksum."""
    import zlib
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


class Builder:
    """Writes small glTF files: JSON plus one binary buffer, as a .glb or as a .gltf with the buffer beside it
    (or inside it, as a data: uri)."""

    def __init__(self):
        self.json = {"asset": {"version": "2.0"}}
        self.bin = bytearray()

    def add(self, kind, item):
        self.json.setdefault(kind, []).append(item)
        return len(self.json[kind]) - 1

    def view(self, data, stride=None):
        self.bin += b"\0" * (-len(self.bin) % 4)
        view = {"buffer": 0, "byteOffset": len(self.bin), "byteLength": len(data)}
        if stride:
            view["byteStride"] = stride
        self.bin += data
        return self.add("bufferViews", view)

    def accessor(self, values, component=FLOAT, kind=None, **extra):
        values = np.asarray(values)
        kind = kind or {1: "SCALAR", 2: "VEC2", 3: "VEC3", 4: "VEC4"}[1 if values.ndim == 1 else values.shape[1]]
        data = np.ascontiguousarray(values, DTYPE[component])
        acc = {"bufferView": self.view(data.tobytes()), "componentType": component, "count": len(values),
               "type": kind, **extra}
        if kind == "VEC3" and component == FLOAT:
            acc["min"], acc["max"] = values.min(axis=0).tolist(), values.max(axis=0).tolist()
        return self.add("accessors", acc)

    def mesh(self, positions=QUAD, faces=QUAD_FACES, uv=QUAD_UV, normals=True, material=None, name=None, **attrs):
        attributes = {"POSITION": self.accessor(positions)}
        if normals:
            attributes["NORMAL"] = self.accessor(np.tile([0.0, 0.0, 1.0], (len(positions), 1)))
        if uv is not None:
            attributes["TEXCOORD_0"] = self.accessor(uv)
        attributes.update(attrs)
        prim = {"attributes": attributes}
        if faces is not None:
            prim["indices"] = self.accessor(faces, USHORT)
        if material is not None:
            prim["material"] = material
        return self.add("meshes", {"primitives": [prim], **({"name": name} if name else {})})

    def texture(self, pixels):
        image = self.add("images", {"bufferView": self.view(png(pixels)), "mimeType": "image/png"})
        return self.add("textures", {"source": image})

    def node(self, children=(), **info):
        if children:
            info["children"] = list(children)
        return self.add("nodes", info)

    def scene(self, *roots):
        self.json["scenes"] = [{"nodes": list(roots)}]
        self.json["scene"] = 0

    def glb(self, path):
        self.bin += b"\0" * (-len(self.bin) % 4)
        self.json["buffers"] = [{"byteLength": len(self.bin)}]
        text = json.dumps(self.json).encode()
        text += b" " * (-len(text) % 4)
        body = struct.pack("<II", len(text), 0x4E4F534A) + text + struct.pack("<II", len(self.bin), 0x004E4942) + self.bin
        with open(path, "wb") as f:
            f.write(b"glTF" + struct.pack("<II", 2, 12 + len(body)) + body)
        return path

    def gltf(self, path, embed=False, bin_name="data.bin"):
        if embed:
            uri = "data:application/octet-stream;base64," + base64.b64encode(bytes(self.bin)).decode()
        else:
            uri = bin_name.replace(" ", "%20")
            with open(os.path.join(os.path.dirname(path), bin_name), "wb") as f:
                f.write(self.bin)
        self.json["buffers"] = [{"byteLength": len(self.bin), "uri": uri}]
        with open(path, "w") as f:
            json.dump(self.json, f)
        return path


class GltfTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)

    def path(self, name):
        return os.path.join(self.dir.name, name)

    def test_a_textured_mesh(self):
        b = Builder()
        texels = np.zeros((4, 4, 3), np.uint8)
        texels[:2] = (255, 0, 0)  # the image's top half red, its bottom half black
        mat = b.add("materials", {"name": "paint", "pbrMetallicRoughness": {
            "baseColorFactor": [0.5, 0.25, 1.0, 1.0], "baseColorTexture": {"index": b.texture(texels)},
            "metallicFactor": 0.0, "roughnessFactor": 0.5}})
        b.scene(b.node(mesh=b.mesh(material=mat, name="sign"), name="Sign"))
        model = load_model(b.glb(self.path("sign.glb")))
        self.assertEqual(model.warnings, [])
        self.assertEqual(len(model), 1)
        obj = model.objects[0]
        self.assertIs(model.nodes["Sign"], obj)  # a node with one part is that part
        self.assertEqual(model.names["sign"], [obj])
        self.assertEqual(model.names["paint"], [obj])
        self.assertIs(obj.parent, model.root)
        np.testing.assert_allclose(obj.mesh.vertices, QUAD)
        self.assertEqual(obj.mesh.faces.tolist(), [[0, 1, 2], [0, 2, 3]])
        np.testing.assert_allclose(obj.mesh.vertex_normals(), np.tile([0, 0, 1.0], (4, 1)))
        # glTF's linear colour factor as sRGB; roughness 0.5 sets highlights; not metal, so no reflection.
        np.testing.assert_allclose(obj.color, linear_to_srgb(np.array([0.5, 0.25, 1.0])))
        self.assertEqual((obj.specular, obj.shininess, obj.reflectivity), (0.5, 30.0, 0.0))
        self.assertEqual(obj.mesh.textures[0].shape, (4, 4, 3))
        # v turned over: the quad's top corners (glTF v = 0, the image's top row) at unicode3d's v = 1.
        np.testing.assert_allclose(obj.mesh.uvs[0], [(0, 0), (1, 0), (1, 1)])
        self.assertEqual(model.materials["paint"].files, [])
        # It draws: the top of the quad red, the bottom black.
        fb = Renderer(40, 20).render([*model], Camera(position=np.array([0.0, 0.0, 3.0])), Light(ambient=1.0))
        self.assertTrue(fb.drawn.any())
        rows = np.flatnonzero(fb.drawn.any(axis=1))
        top, bottom = fb.rgb[rows[2]][fb.drawn[rows[2]]].mean(axis=0), fb.rgb[rows[-3]][fb.drawn[rows[-3]]].mean(axis=0)
        self.assertGreater(top[0], 0.05)
        self.assertLess(bottom.max(), 0.02)

    def test_gltf_with_files_beside_it_or_inside(self):
        b = Builder()
        Image.fromarray(np.full((2, 2, 3), 200, np.uint8)).save(self.path("wood grain.png"))
        image = b.add("images", {"uri": "wood%20grain.png"})
        tex = b.add("textures", {"source": image})
        mat = b.add("materials", {"pbrMetallicRoughness": {"baseColorTexture": {"index": tex}}})
        b.scene(b.node(mesh=b.mesh(material=mat)))
        for embed in (False, True):
            model = load_gltf(b.gltf(self.path("crate.gltf"), embed=embed, bin_name="crate data.bin"))
            self.assertEqual(model.warnings, [])
            self.assertEqual(model.objects[0].mesh.textures[0].shape, (2, 2, 3))
            np.testing.assert_allclose(model.materials["material 0"].color, (1.0, 1.0, 1.0))
        # A texture that isn't there: a warning, and the model still loads, untextured.
        os.remove(self.path("wood grain.png"))
        model = load_gltf(self.path("crate.gltf"))
        self.assertIn("wood%20grain.png", model.warnings[0])
        self.assertFalse(model.objects[0].mesh.textures)

    def test_flat_normals_where_a_mesh_has_none(self):
        b = Builder()
        tent = np.array([(-1, 0, 0), (0, 1, 0), (1, 0, 0), (0, 1, -1)], float)
        b.scene(b.node(mesh=b.mesh(tent, np.array([0, 2, 1, 1, 2, 3]), uv=None, normals=False)))
        mesh = load_gltf(b.glb(self.path("tent.glb"))).objects[0].mesh
        self.assertEqual(len(mesh.vertices), 6)  # each triangle with corners of its own
        normals = mesh.vertex_normals()
        np.testing.assert_allclose(normals[:3], np.tile([0, 0, 1.0], (3, 1)), atol=1e-12)
        self.assertTrue(np.allclose(normals[3:], normals[3]))

    def test_strips_fans_sparse_interleaved_and_quantized(self):
        b = Builder()
        # Positions and 8-bit normalized colours interleaved in one buffer view, 16 bytes a vertex.
        grid = np.array([(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0), (2, 0, 0)], np.float32)
        colors = np.array([(255, 0, 0, 255)] * 5, np.uint8)
        packed = b"".join(p.tobytes() + c.tobytes() for p, c in zip(grid, colors))
        view = b.view(packed, stride=16)
        pos = b.add("accessors", {"bufferView": view, "componentType": FLOAT, "count": 5, "type": "VEC3"})
        col = b.add("accessors", {"bufferView": view, "byteOffset": 12, "componentType": UBYTE, "count": 5,
                                  "type": "VEC4", "normalized": True})
        # Sparse: vertex 4 moved up to (2, 1, 0).
        moved = b.add("accessors", {"componentType": FLOAT, "count": 5, "type": "VEC3", "bufferView": b.view(grid.tobytes()),
                                    "sparse": {"count": 1, "indices": {"bufferView": b.view(np.array([4], "<u2").tobytes()),
                                                                        "componentType": USHORT},
                                               "values": {"bufferView": b.view(np.array([2, 1, 0], "<f4").tobytes())}}})
        strip = {"attributes": {"POSITION": pos, "COLOR_0": col}, "indices": b.accessor([0, 1, 2, 3, 4], UBYTE),
                 "mode": 5}
        fan = {"attributes": {"POSITION": moved}, "mode": 6}  # no indices: the vertices in order
        lines = {"attributes": {"POSITION": pos}, "mode": 1}
        mesh = b.add("meshes", {"primitives": [strip, fan, lines]})
        b.scene(b.node(mesh=mesh, name="Strip"))
        model = load_gltf(b.glb(self.path("strip.glb")))
        self.assertIsInstance(model.nodes["Strip"], Node)  # two parts (the lines are skipped): a node over them
        strip_obj, fan_obj = model.objects
        self.assertTrue(all(o.parent is model.nodes["Strip"] for o in model.objects))
        tri = strip_obj.mesh.vertices[strip_obj.mesh.faces]
        # Strip triangles all wind the same way (counter-clockwise, facing +z).
        normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        self.assertEqual(len(tri), 3)
        self.assertTrue((normals[:, 2] < 0).all() or (normals[:, 2] > 0).all())
        np.testing.assert_allclose(strip_obj.mesh.vertex_colors, np.tile([1.0, 0, 0], (9, 1)))  # flat: unshared
        self.assertEqual(len(fan_obj.mesh.faces), 3)
        self.assertIn([2.0, 1.0, 0.0], fan_obj.mesh.vertices.tolist())

    def test_alpha_modes(self):
        b = Builder()
        texels = np.full((4, 4, 4), 255, np.uint8)
        texels[:, :2, 3] = 100  # the left half partly clear
        tex = b.texture(texels)
        modes = [{"alphaMode": "OPAQUE"}, {"alphaMode": "MASK", "alphaCutoff": 0.5}, {"alphaMode": "BLEND"}]
        for k, mode in enumerate(modes):
            mat = b.add("materials", {"name": mode["alphaMode"], "doubleSided": k == 2, **mode, "pbrMetallicRoughness": {
                "baseColorTexture": {"index": tex}, "baseColorFactor": [1, 1, 1, 0.5]}})
            b.node(mesh=b.mesh(material=mat), name=mode["alphaMode"])
        b.scene(0, 1, 2)
        model = load_gltf(b.glb(self.path("alpha.glb")))
        opaque, mask, blend = (model.nodes[n] for n in ("OPAQUE", "MASK", "BLEND"))
        self.assertEqual(opaque.mesh.textures[0].shape, (4, 4, 3))  # alpha ignored
        self.assertEqual(opaque.opacity, 1.0)
        self.assertEqual(alpha_kind(build_mipmaps(mask.mesh.textures[0])), CUTOUT)
        cut = mask.mesh.textures[0][..., 3]
        self.assertEqual((cut[:, :2].max(), cut[:, 2:].min()), (0.0, 1.0))  # 100/255 * 0.5 is under the cutoff
        self.assertEqual(mask.opacity, 1.0)
        self.assertEqual(alpha_kind(build_mipmaps(blend.mesh.textures[0])), BLEND)
        self.assertEqual(blend.opacity, 0.5)
        self.assertEqual((opaque.double_sided, blend.double_sided), (False, True))

    def test_materials_metal_glow_and_extensions(self):
        b = Builder()
        mats = [{"name": "chrome", "pbrMetallicRoughness": {"metallicFactor": 1.0, "roughnessFactor": 0.0}},
                {"name": "lamp", "emissiveFactor": [1.0, 0.5, 0.0],
                 "extensions": {"KHR_materials_emissive_strength": {"emissiveStrength": 2.0}}},
                {"name": "flat", "extensions": {"KHR_materials_unlit": {}}}]
        for m in mats:
            b.node(mesh=b.mesh(material=b.add("materials", m)), name=m["name"])
        b.scene(0, 1, 2)
        b.json["extensionsUsed"] = b.json["extensionsRequired"] = ["KHR_materials_unlit"]
        model = load_gltf(b.glb(self.path("mats.glb")))
        chrome, lamp, flat = (model.nodes[n] for n in ("chrome", "lamp", "flat"))
        self.assertEqual((chrome.reflectivity, chrome.specular, chrome.shininess), (1.0, 2.0, 1000.0))
        self.assertEqual(lamp.emissive, 2.0)
        self.assertEqual((lamp.reflectivity, lamp.specular), (0.0, 0.0))  # the defaults: rough metal
        self.assertEqual((flat.emissive, flat.specular), (1.0, 0.0))

    def test_unsupported_files_are_refused(self):
        b = Builder()
        b.scene(b.node(mesh=b.mesh()))
        b.json["extensionsRequired"] = ["KHR_draco_mesh_compression"]
        with self.assertRaisesRegex(ValueError, "Draco"):
            load_gltf(b.glb(self.path("draco.glb")))
        b.json["extensionsRequired"] = []
        b.json["asset"]["version"] = "1.0"
        with self.assertRaisesRegex(ValueError, "version"):
            load_gltf(b.gltf(self.path("old.gltf")))
        with open(self.path("junk.glb"), "wb") as f:
            f.write(b"not a model at all")
        with self.assertRaises(ValueError):
            load_gltf(self.path("junk.glb"))
        # A broken accessor leaves its part out, with a warning.
        b.json["asset"]["version"] = "2.0"
        # A node that isn't one: the file is refused, saying so, rather than failing somewhere inside.
        b.json["nodes"].append(5)
        b.json["scenes"][0]["nodes"].append(1)
        with self.assertRaisesRegex(ValueError, "well-formed"):
            load_gltf(b.glb(self.path("bad-node.glb")))
        b.json["nodes"].pop()
        b.json["scenes"][0]["nodes"].pop()
        # A material that isn't one is left plain, with a warning.
        b.json["materials"] = ["shiny"]
        b.json["meshes"][0]["primitives"][0]["material"] = 0
        model = load_gltf(b.glb(self.path("bad-material.glb")))
        self.assertEqual(len(model), 1)
        self.assertIn("material 0", model.warnings[0])
        b.json["accessors"][0]["count"] = 10_000
        model = load_gltf(b.glb(self.path("broken.glb")))
        self.assertEqual(len(model), 0)
        self.assertTrue(any("past" in w for w in model.warnings))
        # A .glb cut short within its header is refused as a ValueError too (not struct's own error).
        for data in (b"glTF", b"glTF\x02\x00\x00\x00", b"glTF\x02\x00\x00\x00\x40\x00"):
            with open(self.path("short.glb"), "wb") as f:
                f.write(data)
            with self.assertRaises(ValueError):
                load_gltf(self.path("short.glb"))

    def test_broken_parts_are_left_out_with_warnings(self):
        # Positions of two numbers each (they need three) leave their part out, rather than making a mesh that
        # can't be drawn; a texture too large to decode safely (a decompression bomb) leaves the part untextured.
        b = Builder()
        flat = b.mesh(positions=QUAD[:, :2], normals=False, uv=None)
        bomb = (b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", 30000, 30000, 8, 2, 0, 0, 0))
                + _png_chunk(b"IDAT", b"") + _png_chunk(b"IEND", b""))
        image = b.add("images", {"bufferView": b.view(bomb), "mimeType": "image/png"})
        material = b.add("materials", {"pbrMetallicRoughness": {"baseColorTexture": {"index": b.add(
            "textures", {"source": image})}}})
        b.scene(b.node(mesh=flat), b.node(mesh=b.mesh(material=material)))
        model = load_gltf(b.glb(self.path("broken-parts.glb")))
        self.assertEqual(len(model), 1)
        self.assertIsNone(model.objects[0].mesh.uvs)
        self.assertTrue(any("3" in w and "positions" in w for w in model.warnings), model.warnings)
        self.assertTrue(any("too large" in w for w in model.warnings), model.warnings)
        Renderer(20, 10).render(list(model), Camera(), Light())

    def test_animations_and_numbers_that_cannot_be_used(self):
        # A translation or scale keyframed with one number each: the channel is left out (played, the node got a
        # position of one number, and the next render() raised).
        b = Builder()
        door = b.node(mesh=b.mesh(), name="Door")
        b.scene(door)
        times = b.accessor([0.0, 1.0])
        b.add("animations", {"name": "Bad", "samplers": [{"input": times, "output": b.accessor([1.0, 2.0])},
                                                         {"input": times, "output": b.accessor(np.zeros((2, 3)))}],
                             "channels": [{"sampler": 0, "target": {"node": door, "path": "translation"}},
                                          {"sampler": 0, "target": {"node": door, "path": "scale"}},
                                          {"sampler": 1, "target": {"node": door, "path": "translation"}}]})
        model = load_gltf(b.glb(self.path("narrow.glb")))
        self.assertEqual(sum("channel left out" in w for w in model.warnings), 2, model.warnings)
        model.animations["Bad"].apply(0.5)
        self.assertEqual(np.shape(model.nodes["Door"].position), (3,))
        Renderer(20, 10).render(list(model), Camera(), Light())
        # Numbers that can be drawn around are loaded without warnings (which would be printed over the picture);
        # a length of infinity leaves its part out, with a warning (it raised OverflowError).
        b.json["nodes"][door]["matrix"] = [1e308] * 16
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            Renderer(20, 10).render(list(load_gltf(b.glb(self.path("huge.glb")))), Camera(), Light())
        b.json["bufferViews"][0]["byteLength"] = float("inf")  # (Python's json writes Infinity, and reads it)
        model = load_gltf(b.glb(self.path("endless.glb")))
        self.assertEqual(len(model), 0)
        self.assertTrue(any("mesh 0" in w and "infinity" in w for w in model.warnings), model.warnings)

    def test_node_hierarchy_and_matrices(self):
        b = Builder()
        quad = b.mesh()
        turn = quat_axis_angle([0, 1, 0], np.pi / 2)  # w first; glTF writes x, y, z, w
        child = b.node(mesh=quad, name="Child", translation=[1, 0, 0], rotation=[*turn[1:], turn[0]])
        # A matrix: scale -2 along x (mirrored), then up 3.
        matrix = np.diag([-2.0, 1.0, 1.0, 1.0])
        matrix[:3, 3] = (0, 3, 0)
        mirrored = b.node(mesh=quad, name="Mirrored", matrix=matrix.T.reshape(-1).tolist())
        parent = b.node([child, mirrored], name="Parent", translation=[0, 0, -5], scale=[2, 2, 2])
        b.node(mesh=quad, name="Elsewhere")  # in no scene: not loaded
        b.scene(parent)
        model = load_gltf(b.glb(self.path("tree.glb")))
        self.assertEqual(set(model.nodes), {"Child", "Mirrored", "Parent"})
        self.assertEqual(len(model), 2)
        self.assertIs(model.nodes["Child"].parent, model.nodes["Parent"])
        self.assertIs(model.nodes["Parent"].parent, model.root)
        # The child's corner (1, 1, 0): turned about y onto (0, 1, -1), moved +x, then doubled and moved back 5.
        np.testing.assert_allclose(model.nodes["Child"].to_world([1.0, 1.0, 0.0]), (2.0, 2.0, -7.0), atol=1e-12)
        linear, position, _ = model.nodes["Mirrored"].world_matrix()
        np.testing.assert_allclose(linear, np.diag([-4.0, 2.0, 2.0]), atol=1e-12)
        np.testing.assert_allclose(position, (0, 6, -5), atol=1e-12)
        # Two nodes using one mesh share it.
        self.assertIs(model.objects[0].mesh, model.objects[1].mesh)
        model.fit(2.0)  # and the whole model fits
        lo, hi = model.bounds()
        self.assertAlmostEqual(float((hi - lo).max()) * model.root.scale, 2.0)

    def test_skins_and_morph_targets_load_posed(self):
        b = Builder()
        joints = b.accessor(np.zeros((4, 4)), USHORT)
        weights = b.accessor(np.tile([1.0, 0, 0, 0], (4, 1)))
        mesh = b.mesh(JOINTS_0=joints, WEIGHTS_0=weights)
        bone = b.node(name="Bone", translation=[0, 1, 0])
        skin = b.add("skins", {"joints": [bone],
                               "inverseBindMatrices": b.accessor(np.eye(4).reshape(1, 16), kind="MAT4")})
        skinned = b.node(name="Skinned", mesh=mesh, skin=skin, translation=[50, 0, 0])  # ignored: the joints place it
        morph = b.add("meshes", {"weights": [0.5], "primitives": [{
            "attributes": {"POSITION": b.accessor(QUAD)},
            "targets": [{"POSITION": b.accessor(np.tile([0.0, 0.0, 2.0], (4, 1)))}],
            "indices": b.accessor(QUAD_FACES, USHORT)}]})
        b.node(name="Morph", mesh=morph)
        b.scene(bone, skinned, 2)
        rotate = b.add("animations", {"channels": [{"sampler": 0, "target": {"node": bone, "path": "translation"}},
                                                   {"sampler": 0, "target": {"node": 2, "path": "weights"}}],
                                      "samplers": [{"input": b.accessor([0.0, 1.0]),
                                                    "output": b.accessor(np.zeros((2, 3)))}]})
        model = load_gltf(b.glb(self.path("skin.glb")))
        skinned_obj = model.names["Skinned"][0]
        self.assertIs(skinned_obj.parent, model.root)
        np.testing.assert_allclose(skinned_obj.mesh.vertices, QUAD + (0, 1, 0))  # moved up by its bone
        np.testing.assert_allclose(model.nodes["Morph"].mesh.vertices[:, 2], 1.0)  # half way to its target
        self.assertEqual(rotate, 0)
        self.assertTrue(any("skinned" in w for w in model.warnings))
        self.assertTrue(any("morph" in w for w in model.warnings))

    def test_animations(self):
        b = Builder()
        door = b.node(mesh=b.mesh(), name="Door")
        lamp = b.node(mesh=b.mesh(), name="Lamp")
        b.scene(door, lamp)
        times = b.accessor([0.0, 1.0, 2.0])
        half = quat_axis_angle([0, 1, 0], np.pi / 2)
        rotations = np.array([(0, 0, 0, 1), (*half[1:], half[0]), (*half[1:], half[0])])
        spline = np.array([(0, 0, 0), (0, 0, 0), (0, 0, 0),    # in-tangent, value, out-tangent at t = 0
                           (0, 0, 0), (0, 4, 0), (0, 0, 0),
                           (0, 0, 0), (0, 0, 0), (0, 0, 0)], float)
        samplers = [{"input": times, "output": b.accessor(rotations)},
                    {"input": times, "output": b.accessor(np.array([(0, 0, 0), (2, 0, 0), (4, 0, 0)], float)),
                     "interpolation": "STEP"},
                    {"input": times, "output": b.accessor(spline), "interpolation": "CUBICSPLINE"},
                    {"input": times, "output": b.accessor(np.array([[1.0, 1, 1], [2, 2, 2], [1, 1, 1]]))}]
        b.add("animations", {"name": "Open", "samplers": samplers, "channels": [
            {"sampler": 0, "target": {"node": door, "path": "rotation"}},
            {"sampler": 1, "target": {"node": door, "path": "translation"}},
            {"sampler": 2, "target": {"node": lamp, "path": "translation"}},
            {"sampler": 3, "target": {"node": lamp, "path": "scale"}}]})
        b.add("animations", {"samplers": [samplers[1]], "channels": [
            {"sampler": 0, "target": {"node": lamp, "path": "translation"}}]})
        model = load_gltf(b.glb(self.path("anim.glb")))
        self.assertEqual(set(model.animations), {"Open", "animation 1"})
        clip = model.animations["Open"]
        self.assertIsInstance(clip, Clip)
        self.assertEqual(clip.duration, 2.0)
        door_obj, lamp_obj = model.nodes["Door"], model.nodes["Lamp"]
        self.assertEqual(set(map(id, clip.targets)), {id(door_obj), id(lamp_obj)})
        clip.apply(0.5)
        quarter = quat_axis_angle([0, 1, 0], np.pi / 4)
        np.testing.assert_allclose(np.abs(door_obj.rotation), np.abs(quarter), atol=1e-9)  # slerped, as w, x, y, z
        np.testing.assert_allclose(door_obj.position, (0, 0, 0))  # STEP: still the first key's
        np.testing.assert_allclose(lamp_obj.position, (0, 2, 0))  # the spline, flat at both ends: halfway up
        np.testing.assert_allclose(lamp_obj.scale, (1.5, 1.5, 1.5))
        clip.apply(1.0)
        np.testing.assert_allclose(door_obj.position, (2, 0, 0))
        np.testing.assert_allclose(lamp_obj.position, (0, 4, 0))
        # Played once it holds the end; looped it starts again.
        self.assertFalse(clip.update(5.0))
        np.testing.assert_allclose(door_obj.position, (4, 0, 0))
        clip.loop = "loop"
        clip.apply(2.5)
        self.assertFalse(clip.done())
        np.testing.assert_allclose(lamp_obj.position, (0, 2, 0))
        # A Model's parts draw wherever the animation has put them.
        fb = Renderer(40, 20).render([*model], Camera(position=np.array([0.0, 2.0, 12.0])), Light())
        self.assertTrue(fb.drawn.any())
        # A copy shares meshes and keyframes, but has parts, nodes and clocks of its own.
        table = Node()
        model.root.parent = table
        twin = model.copy()
        self.assertIs(twin.root.parent, table)
        self.assertIsNot(twin.root, model.root)
        twin_door = twin.nodes["Door"]
        self.assertIsNot(twin_door, door_obj)
        self.assertIs(twin_door.mesh, door_obj.mesh)
        self.assertEqual({id(o) for o in twin}, {id(o) for o in twin.nodes.values()})
        self.assertTrue(all(o.parent is twin.root for o in twin))
        self.assertEqual(set(map(id, twin.animations["Open"].targets)), set(map(id, twin.nodes.values())))
        self.assertIs(twin.animations["Open"].animations[0].tracks["rotation"],
                      clip.animations[0].tracks["rotation"])
        self.assertEqual(twin.animations["Open"].time, clip.time)
        np.testing.assert_allclose(twin_door.position, door_obj.position)
        before = door_obj.position.copy()
        twin.animations["Open"].apply(1.0)
        np.testing.assert_allclose(twin_door.position, (2, 0, 0))
        np.testing.assert_allclose(door_obj.position, before)
        self.assertEqual(clip.time, 2.5)
        twin.root.position = np.array([3.0, 0.0, 0.0])
        np.testing.assert_allclose(twin_door.world_matrix()[1] - door_obj.world_matrix()[1],
                                   np.array([3.0, 0.0, 0.0]) + (2, 0, 0) - before)
        camera = Camera(position=np.array([0.0, 2.0, 12.0]))
        drawn = [Renderer(40, 20).render([*m], camera, Light()).drawn.copy() for m in (model, twin)]
        self.assertTrue(drawn[1].any())
        self.assertFalse((drawn[0] == drawn[1]).all())


class AnimationPieceTests(unittest.TestCase):
    def test_step_easing(self):
        track = Track([(0.0, 1.0), (1.0, 2.0), (2.0, 5.0)], easing=step)
        self.assertEqual([track.at(t) for t in (0.0, 0.5, 0.999, 1.0, 1.5, 2.0, 9.0)], [1, 1, 1, 2, 2, 5, 5])
        self.assertEqual(Track([(0, 0.0), (1, 1.0)], easing="step").at(0.7), 0.0)

    def test_spline_track(self):
        # Matching tangents make a straight line; tangents of 0 a smooth step; values stay on the keys.
        line = SplineTrack([(0.0, 0.0, 1.0, 1.0), (2.0, 2.0, 1.0, 1.0)])
        self.assertAlmostEqual(line.at(0.5), 0.5)
        ease = SplineTrack([(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0)])
        self.assertAlmostEqual(ease.at(0.25), 0.15625)
        self.assertEqual((ease.at(-1), ease.at(1.0)), (0.0, 1.0))
        turn = quat_axis_angle([0, 0, 1], 1.0)
        spin = SplineTrack([(0.0, [1, 0, 0, 0], [0] * 4, [0] * 4), (1.0, turn, [0] * 4, [0] * 4)], rotation=True)
        self.assertAlmostEqual(float(np.linalg.norm(spin.at(0.3))), 1.0)
        looped = SplineTrack([(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0)], loop="loop")
        self.assertAlmostEqual(looped.at(1.25), 0.15625)

    def test_clip_plays_animations_together(self):
        a, b = Object3D(None), Node()
        clip = Clip([Animation(a, position=Track([(0, (0, 0, 0)), (1, (1, 0, 0))])),
                     Animation(b, scale=Track([(0, 1.0), (3, 4.0)]))], name="grow", loop="pingpong")
        self.assertEqual(clip.duration, 3.0)
        clip.update(1.5)
        np.testing.assert_allclose(a.position, (1, 0, 0))
        self.assertEqual(b.scale, 2.5)
        clip.update(3.0)  # 4.5 s: on the way back
        self.assertEqual(b.scale, 2.5)
        self.assertFalse(clip.done())
        with self.assertRaises(ValueError):
            Clip([], loop="sometimes")
        self.assertEqual(Clip([]).duration, 0.0)
        self.assertTrue(Clip([]).done())


if __name__ == "__main__":
    unittest.main()
