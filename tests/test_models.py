# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
"""Loading models (OBJ and MTL, with textures), authored normals, repeating textures and mip levels per pixel."""
import os
import tempfile
import unittest
import warnings

import numpy as np
from PIL import Image

from unicode3d import Camera, Light, Mesh, Model, Node, Object3D, Renderer, load_image, load_model, load_obj
from unicode3d.color import luminance
from unicode3d.texture import CUTOUT, alpha_kind, build_mipmaps

CHECKS = np.indices((16, 16)).sum(axis=0) % 2 * 255  # a 16x16 checkerboard, 0..255

# A box's front (wood, a texture), back (wood again), bottom (glow) and, as group "lid", its top (fence: holes
# from map_d) and right side (a material no MTL file defines); a line, which is skipped; and some bad lines.
CRATE_OBJ = """# a crate
mtllib crate.mtl
o box
v -1 -1 1
v 1 -1 1
v 1 1 1
v -1 1 1
v -1 -1 -1
v 1 -1 -1
v 1 1 -1
v -1 1 -1
vt 0 0
vt 1 0
vt 1 1
vt 0 1
vn 0 0 1
vn 0 0 -1
usemtl wood
s off
f 1/1/1 2/2/1 3/3/1 4/4/1
f -2/1/2 -3/2/2 -4/3/2 -1/4/2
usemtl glow
f 1/1 5/2 6/3 2/4
o lid
usemtl fence
s 1
f 4 3 7 8
usemtl missing
f 2 6 7 3
l 1 2
f 1 2 99
v nonsense
"""

CRATE_MTL = """newmtl wood
Kd 1 0.5 0.25
Ks 0.2 0.2 0.2
Ns 20
map_Kd -s 2 2 1 -o 0.5 0 0 tex\\checks.png
newmtl glow
Kd 1 0.5 0
Ke 0.5 0.8 0
illum 1
newmtl fence
Kd 0.5 1 0.5
Tr 0.25
map_d tex/hole.png
newmtl chrome
Pm 1
Pr 0.5
map_Kd nowhere.png
"""


class ModelFiles(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        os.mkdir(self.path("tex"))
        Image.fromarray(CHECKS.astype(np.uint8)).convert("RGB").save(self.path("tex", "checks.png"))
        hole = np.full((8, 8), 255, np.uint8)
        hole[2:6, 2:6] = 0
        Image.fromarray(hole).save(self.path("tex", "hole.png"))
        self.write("crate.obj", CRATE_OBJ)
        self.write("crate.mtl", CRATE_MTL)

    def path(self, *parts):
        return os.path.join(self.dir.name, *parts)

    def write(self, name, text):
        with open(self.path(name), "w") as f:
            f.write(text)
        return self.path(name)


class LoadModelTests(ModelFiles):
    def test_a_part_for_each_material(self):
        model = load_model(self.path("crate.obj"))
        self.assertIsInstance(model, Model)
        self.assertEqual(len(model), 4)  # wood, glow, fence, and the undefined one
        wood, glow, fence, missing = model.names["wood"][0], model.names["glow"][0], model.names["fence"][0], \
            model.names["missing"][0]
        self.assertTrue(all(obj.parent is model.root for obj in model))
        self.assertEqual([len(o.mesh.faces) for o in model], [4, 2, 2, 2])  # quads, split into triangles
        # Kd, Ks, Ns; the texture, moved and scaled by -o and -s, found though its path has a backslash.
        self.assertEqual(wood.color, (1.0, 0.5, 0.25))
        self.assertAlmostEqual(wood.specular, 0.2)
        self.assertEqual(wood.shininess, 20.0)
        self.assertEqual(wood.mesh.textures[0].shape, (16, 16, 3))
        self.assertEqual(wood.mesh.uvs[..., 0].min(), 0.5)
        self.assertEqual(wood.mesh.uvs[..., 0].max(), 2.5)
        # Ke (its brightest channel), and illum 1: no highlights.
        self.assertEqual((glow.emissive, glow.specular), (0.8, 0.0))
        # Tr is the opposite of d; map_d makes holes.
        self.assertEqual(fence.opacity, 0.75)
        self.assertEqual(fence.mesh.textures[0].shape, (8, 8, 4))
        self.assertEqual(alpha_kind(build_mipmaps(fence.mesh.textures[0])), CUTOUT)
        # Metal reflects (less when rough), and roughness sets the highlights; a missing texture is skipped.
        chrome = model.materials["chrome"]
        self.assertEqual(chrome.reflectivity, 0.5)
        self.assertAlmostEqual(chrome.shininess, 30.0)
        self.assertIsNone(chrome.texture)
        self.assertEqual(missing.color, (0.8, 0.8, 0.8))
        warnings = "\n".join(model.warnings)
        for problem in ("nowhere.png", "'missing'", "line 31: f", "line 32: v"):
            self.assertIn(problem, warnings)

    def test_normals_from_the_file_and_from_smoothing_groups(self):
        model = load_model(self.path("crate.obj"))
        wood, glow, fence = (model.names[n][0] for n in ("wood", "glow", "fence"))
        # The file's normals, one vertex for each position and normal.
        self.assertEqual(len(wood.mesh.vertices), 8)
        np.testing.assert_array_equal(np.unique(wood.mesh.vertex_normals(), axis=0), [[0, 0, -1], [0, 0, 1]])
        # No normals, smoothing off ("s off"): flat. Smoothing on ("s 1"): worked out, shared between faces.
        self.assertIsNone(glow.mesh.normals)
        np.testing.assert_allclose(glow.mesh.vertex_normals(), [[0, -1, 0]] * 4, atol=1e-12)
        self.assertEqual(len(fence.mesh.vertices), 4)
        # Without any "s" statement, faces share vertices and shade smoothly (as load_obj always has).
        self.write("plain.obj", "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 1\nf 1 2 3\nf 1 3 4\n")
        self.assertEqual(len(load_model(self.path("plain.obj")).objects[0].mesh.vertices), 4)

    def test_groups_split_into_parts(self):
        model = load_model(self.path("crate.obj"), split_groups=True)
        self.assertEqual(sorted(model.names), ["box", "fence", "glow", "lid", "missing", "wood"])
        self.assertEqual(len(model.names["box"]), 2)
        self.assertEqual(len(model.names["lid"]), 2)

    def test_fit_and_draw(self):
        model = load_model(self.path("crate.obj"))
        model.root.rotation = np.array([np.cos(0.3), 0.0, np.sin(0.3), 0.0])
        lo, hi = model.fit(1.0).bounds()
        np.testing.assert_allclose((lo, hi), ([-1] * 3, [1] * 3))  # in root's own space
        self.assertEqual(model.root.scale, 0.5)
        np.testing.assert_allclose(model.root.position, 0.0, atol=1e-12)
        fb = Renderer(40, 20).render([*model], Camera(position=np.array([1.5, 1.5, 2.5])), Light(shadows=True))
        self.assertGreater(fb.drawn.mean(), 0.1)
        self.assertEqual(set(np.unique(fb.ids)) - {0}, {1, 2, 3, 4} & set(np.unique(fb.ids)))
        model.root.visible = False
        self.assertFalse(Renderer(40, 20).render([*model], Camera(), Light()).drawn.any())

    def test_only_obj_and_gltf_files(self):
        with self.assertRaises(ValueError):
            load_model(self.path("crate.fbx"))
        for name in ("nothing.obj", "nothing.glb"):
            with self.assertRaises(OSError):
                load_model(self.path(name))

    def test_missing_mtllib_still_loads(self):
        self.write("lost.obj", "mtllib lost.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nusemtl red\nf 1 2 3\n")
        model = load_model(self.path("lost.obj"))
        self.assertEqual(len(model), 1)
        self.assertIn("lost.mtl", model.warnings[0])
        self.assertEqual(model.materials, {})


class BadNumberFileTests(ModelFiles):
    def test_numbers_beyond_any_use(self):
        # Loaded without an exception (a roughness of 1e308 raised OverflowError) or a warning (printed over the
        # picture), and drawn without one.
        self.write("huge.mtl", "newmtl a\nKs 1e308 1e308 1e308\nPm inf\nPr 1e308\nKe nan 1 1\n"
                               "map_Kd -s 1e308 1e308 1 tex/checks.png\nnewmtl b\nPm 1\nPr nan\n")
        self.write("huge.obj", "mtllib huge.mtl\nv 0 0 0\nv 1 10 0\nv 0 0 -1e308\nv 1 0 0\nvt 0 0\nvt -inf 1e308\n"
                               "usemtl a\nf 1/1 2/2 3/1\nusemtl b\nf 1 4 2\n")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            model = load_model(self.path("huge.obj"))
            mesh = load_obj(self.path("huge.obj"))
            self.assertEqual(model.materials["b"].shininess, 1000.0)
            for objects in (list(model), [Object3D(mesh)]):
                fb = Renderer(20, 10).render(objects, Camera(position=np.array([0.0, 1.0, 4.0])), Light())
                self.assertTrue(np.isfinite(fb.rgb).all() and np.isfinite(fb.alpha).all())


class LoadObjTests(ModelFiles):
    def test_one_mesh_with_colours_and_textures(self):
        mesh = load_obj(self.path("crate.obj"))
        self.assertEqual(mesh.faces.shape, (10, 3))
        # Two textures (wood's and fence's), and a white one for the untextured materials.
        self.assertEqual([t.shape for t in mesh.textures], [(16, 16, 3), (8, 8, 4), (1, 1, 3)])
        np.testing.assert_array_equal(mesh.materials, [0, 0, 0, 0, 2, 2, 1, 1, 2, 2])
        np.testing.assert_allclose(mesh.face_colors[0], (1.0, 0.5, 0.25, 1.0))
        np.testing.assert_allclose(mesh.face_colors[6], (0.5, 1.0, 0.5, 0.75))
        self.assertEqual(mesh.uvs[0, :, 0].max(), 2.5)

    def test_vertex_colours(self):
        self.write("tri.obj", "v 0 0 0 1 0 0\nv 1 0 0 0 1 0\nv 0 1 0 0 0 1\nf 1 2 3\n")
        mesh = load_obj(self.path("tri.obj"))
        np.testing.assert_array_equal(mesh.vertex_colors, np.eye(3))
        self.assertIsNone(mesh.face_colors)
        fb = Renderer(30, 15).render([Object3D(mesh, color=(255, 255, 255))], Camera(position=np.array([0.3, 0.3, 2])),
                                     Light())
        rgb = fb.colour()[fb.alpha == 1]
        self.assertTrue(all((rgb.argmax(axis=1) == k).any() for k in range(3)))  # red, green and blue corners


class ImageTests(ModelFiles):
    def test_formats(self):
        rgba = np.zeros((4, 6, 4), np.uint8)
        rgba[..., 0], rgba[:, :3, 3] = 255, 255
        Image.fromarray(rgba).save(self.path("a.png"))
        image = load_image(self.path("a.png"))
        self.assertEqual(image.shape, (4, 6, 4))
        np.testing.assert_array_equal(image[0, :, 3], [1, 1, 1, 0, 0, 0])
        Image.fromarray(rgba).convert("RGB").save(self.path("a.jpg"))
        self.assertEqual(load_image(self.path("a.jpg")).shape, (4, 6, 3))
        palette = Image.fromarray(CHECKS.astype(np.uint8)).convert("P")
        palette.info["transparency"] = 0
        palette.save(self.path("p.png"), transparency=0)
        self.assertEqual(load_image(self.path("p.png")).shape, (16, 16, 4))
        grey = load_image(self.path("tex", "hole.png"))
        self.assertEqual((grey.shape, grey.min(), grey.max()), ((8, 8, 3), 0.0, 1.0))

    def test_broken_images(self):
        # Files Pillow can't read raise OSError, and one too large to decode safely ValueError (Pillow's own
        # DecompressionBombError is neither): the model loaders take both as warnings and leave the texture out.
        import io
        import struct
        import zlib

        def chunk(kind, data):
            return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

        out = io.BytesIO()
        Image.fromarray(CHECKS.astype(np.uint8)).save(out, "PNG")
        good = out.getvalue()
        for data in (good[:len(good) // 2], good[:20], b"", b"\x00" * 100):
            with self.assertRaises(OSError):
                load_image(data)
        bomb = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 30000, 30000, 8, 2, 0, 0, 0))
                + chunk(b"IDAT", b"") + chunk(b"IEND", b""))
        with self.assertRaisesRegex(ValueError, "too large"):
            load_image(bomb)
        with open(self.path("tex", "checks.png"), "wb") as f:
            f.write(bomb)
        with open(self.path("tex", "hole.png"), "wb") as f:
            f.write(good[:30])
        model = load_model(self.path("crate.obj"))
        self.assertTrue(any("too large" in w for w in model.warnings), model.warnings)
        self.assertTrue(any("could not be read" in w for w in model.warnings), model.warnings)
        Renderer(20, 10).render(list(model), Camera(), Light())

    def test_big_images_are_shrunk(self):
        Image.new("RGB", (300, 100), (10, 20, 30)).save(self.path("big.png"))
        self.assertEqual(load_image(self.path("big.png"), max_size=150).shape, (50, 150, 3))
        self.assertEqual(load_image(self.path("big.png"), max_size=None).shape, (100, 300, 3))


def quad(uv_scale=1.0, texture=None, depth=1.0, width=1.0):
    """A floor-like quad lying on y = -0.5 from z = 0 back to z = -depth (or, with depth=None, a square facing +z),
    textured with uv from 0 to uv_scale."""
    if depth is None:
        verts = np.array([(-width, -1, 0), (width, -1, 0), (width, 1, 0), (-width, 1, 0)], float)
    else:
        verts = np.array([(-width, -0.5, 0), (width, -0.5, 0), (width, -0.5, -depth), (-width, -0.5, -depth)], float)
    uv = np.array([(0, 0), (1, 0), (1, 1), (0, 1)], float) * uv_scale
    faces = np.array([(0, 1, 2), (0, 2, 3)])
    return Mesh(verts, faces, uvs=uv[faces], materials=np.zeros(2, int),
                textures=[CHECKS / 255.0 if texture is None else texture])


class NormalTests(unittest.TestCase):
    def render(self, mesh):
        fb = Renderer(30, 15, fog=0, outline=0).render([Object3D(mesh, color=(255, 255, 255))], Camera(),
                                                       Light(direction=np.array([0.0, 0.0, -1.0])))
        return luminance(fb.colour())[fb.alpha == 1]

    def test_authored_normals_are_used(self):
        flat = Mesh(np.array([(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)], float), np.array([(0, 1, 2), (0, 2, 3)]))
        facing = self.render(flat).mean()
        flat.normals = np.array([(1.0, 0.0, 1.0)] * 4)  # turned 45 degrees away from the light (and not unit length)
        np.testing.assert_allclose(np.linalg.norm(flat.vertex_normals(), axis=1), 1.0)
        self.assertLess(self.render(flat).mean(), 0.9 * facing)
        # Unusable rows are worked out from the faces; the wrong number of rows is an error.
        flat.normals = np.array([(0.0, 0.0, 0.0), (np.nan, 0.0, 1.0), (0.0, 1.0, 0.0), (0.0, 1.0, 0.0)])
        np.testing.assert_allclose(flat.vertex_normals()[:2], [(0, 0, 1)] * 2)
        np.testing.assert_allclose(flat.vertex_normals()[2:], [(0, 1, 0)] * 2)
        flat.normals = np.zeros((3, 3))
        with self.assertRaises(ValueError):
            flat.vertex_normals()

    def test_normals_follow_replaced_faces(self):
        # Worked out afresh when faces is replaced, as when the vertices are (they were kept from the old faces).
        mesh = Mesh(np.array([(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)], float), np.array([(0, 1, 2), (0, 2, 3)]))
        np.testing.assert_allclose(mesh.vertex_normals(), [(0, 0, 1)] * 4)
        mesh.faces = mesh.faces[:, ::-1].copy()  # turned to face the other way
        np.testing.assert_allclose(mesh.vertex_normals(), [(0, 0, -1)] * 4)

    def test_texture_seams_need_not_be_shading_seams(self):
        # Two faces with their own vertices (as a loader splits them where texture coordinates change) shade as
        # one smooth surface when given the normals of the surface they belong to.
        v = np.array([(-1, -1, 0), (0, -1, -0.5), (0, 1, -0.5), (-1, 1, 0), (0, -1, -0.5), (1, -1, 0), (1, 1, 0),
                      (0, 1, -0.5)], float)
        mesh = Mesh(v, np.array([(0, 1, 2), (0, 2, 3), (4, 5, 6), (4, 6, 7)]))
        seam = mesh.vertex_normals()
        self.assertFalse(np.allclose(seam[1], seam[4]))  # worked out: a crease at x = 0
        mesh.normals = np.array([(-0.4, 0, 1), (0, 0, 1), (0, 0, 1), (-0.4, 0, 1), (0, 0, 1), (0.4, 0, 1),
                                 (0.4, 0, 1), (0, 0, 1)])
        np.testing.assert_allclose(mesh.vertex_normals()[1], mesh.vertex_normals()[4])


class TextureSamplingTests(unittest.TestCase):
    def test_textures_repeat_beyond_0_to_1(self):
        # Red on the left half of the texture, blue on the right, shown twice across: red, blue, red, blue.
        texture = np.zeros((4, 4, 3))
        texture[:, :2, 0], texture[:, 2:, 2] = 1.0, 1.0
        fb = Renderer(40, 20, fog=0, outline=0).render(
            [Object3D(quad(2.0, texture, depth=None), color=(255, 255, 255), emissive=1.0)],
            Camera(position=np.array([0.0, 0.0, 2.2])), Light())
        row = fb.colour()[fb.height // 2]
        row = row[fb.alpha[fb.height // 2] == 1]
        reds = (row[:, 0] > row[:, 2]).astype(int)
        self.assertEqual(list(reds[np.flatnonzero(np.diff(reds)) + 1]), [0, 1, 0])  # changes: blue, red, blue

    def test_mip_level_follows_each_pixel(self):
        # A long floor of two triangles, a checkerboard tiled over it: crisp near the camera, and smoothly grey
        # towards the horizon instead of aliasing. One mip level for the whole floor did neither: it was grey
        # close by and flickered in the distance.
        floor = Object3D(quad(4.0, depth=60.0, width=4.0), color=(255, 255, 255), emissive=1.0)
        r = Renderer(80, 40, cell_pixels=(1, 1), cell_aspect=1.0, samples=1, edge_samples=0, fog=0, outline=0,
                     lod_bias=0.0)
        fb = r.render([floor], Camera(position=np.array([0.0, 0.0, 0.5]), target=np.array([0.0, -0.1, -1.0]),
                                      fov=60), Light())
        shade = luminance(fb.colour())
        rows = [shade[y][fb.alpha[y] == 1] for y in range(fb.height) if (fb.alpha[y] == 1).sum() > 20]
        self.assertGreater(max(row.std() for row in rows[-4:]), 0.2)  # the checks are there close by
        self.assertLess(max(row.std() for row in rows[:2]), 0.03)      # and averaged out far away

if __name__ == "__main__":
    unittest.main()
