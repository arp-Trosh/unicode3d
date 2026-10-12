import math
import unittest

import numpy as np

from unicode3d import Camera, Light, Mesh, Node, Object3D, Renderer, Sprite, SpriteSheet
from unicode3d.color import to_linear_rgb
from unicode3d.transforms import quat_axis_angle

HUES = [(255, 0, 0), (255, 128, 0), (255, 255, 0), (0, 255, 0), (0, 255, 255), (0, 0, 255), (128, 0, 255),
        (255, 0, 255)]  # (a colour for each row of a sheet)


def sheet_of(rows, columns=3, cell=4, **options):
    """A sheet whose cells are each one colour: row r's hue, darker in each column after the first."""
    image = np.zeros((rows * cell, columns * cell, 3))
    for r in range(rows):
        for c in range(columns):
            image[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = np.array(HUES[r]) / 255.0 * (1.0 - 0.12 * c)
    options.setdefault("height", 2.0)
    options.setdefault("origin", (0.5, 0.5))
    return SpriteSheet(image, columns=columns, **options)


def around(angle, distance=5.0, height=0.0, **options):
    """A camera `angle` degrees round from +z, counter-clockwise as seen from above, looking at the origin."""
    a = math.radians(angle)
    return Camera(position=np.array([distance * math.sin(a), height, distance * math.cos(a)]), target=np.zeros(3),
                  **options)


def middle(objects, camera, renderer=None):
    fb = (renderer or Renderer(40, 40)).render(objects, camera, [])
    return fb.rgb[fb.rgb.shape[0] // 2, fb.rgb.shape[1] // 2].copy()


def shown(colour, columns=3):
    """The (row, column) of the sheet_of cell drawn in this colour, or None for none of them (an edge)."""
    cells = [(r, c) for r in range(len(HUES)) for c in range(columns)]
    error = [np.abs(to_linear_rgb(np.array(HUES[r]) * (1.0 - 0.12 * c)) - colour).sum() for r, c in cells]
    return cells[int(np.argmin(error))] if min(error) < 0.05 else None


class SpritesTest(unittest.TestCase):
    def test_a_row_for_each_side(self):
        # Facing +z: from in front row 0; going round counter-clockwise (from above) it is seen turned clockwise.
        sheet = sheet_of(8)
        sprite = Sprite(sheet, emissive=1.0)
        for k in range(8):
            for off in (-20, 0, 20):
                self.assertEqual(shown(middle([sprite], around(45 * k + off))), (k, 0), (k, off))
        # Turned (its own rotation, or a parent's): the rows go with it.
        sprite.rotation = quat_axis_angle(np.array([0.0, 1.0, 0.0]), math.radians(90))  # (facing +x)
        self.assertEqual(shown(middle([sprite], around(90))), (0, 0))
        self.assertEqual(shown(middle([sprite], around(0))), (6, 0))
        group = Node(rotation=quat_axis_angle(np.array([0.0, 1.0, 0.0]), math.radians(-90)))
        sprite.parent = group
        self.assertEqual(shown(middle([sprite], around(0))), (0, 0))
        # Counter-clockwise sheets; four directions; one.
        backwards = Sprite(sheet_of(8, clockwise=False), emissive=1.0)
        self.assertEqual(shown(middle([backwards], around(90))), (6, 0))
        four = Sprite(sheet_of(4, directions=4), emissive=1.0)
        self.assertEqual([shown(middle([four], around(a)))[0] for a in (0, 40, 90, 180, 270, 300)], [0, 0, 1, 2, 3, 3])
        one = Sprite(sheet_of(1, directions=1), emissive=1.0)
        self.assertEqual({shown(middle([one], around(a)))[0] for a in range(0, 360, 30)}, {0})

    def test_mirrored_sheets(self):
        # Five rows, front to back; the other three sides are rows 3, 2, 1 flipped left to right.
        image = np.zeros((5 * 4, 4, 3))
        for r in range(5):
            image[r * 4:(r + 1) * 4, :2] = np.array(HUES[r]) / 255.0
            image[r * 4:(r + 1) * 4, 2:] = (1.0, 1.0, 1.0)  # (its right half white)
        sprite = Sprite(SpriteSheet(image, directions=8, mirrored=True, height=2.0, origin=(0.5, 0.5)),
                        emissive=1.0)
        for angle, row, flipped in ((0, 0, False), (90, 2, False), (180, 4, False), (225, 3, True), (270, 2, True)):
            fb = Renderer(40, 40).render([sprite], around(angle), [])
            xs = np.flatnonzero(fb.ids[40] > 0)
            left, right = fb.rgb[40, xs[2]], fb.rgb[40, xs[-3]]
            white = np.ones(3)
            self.assertTrue(np.allclose(right if flipped else left, to_linear_rgb(HUES[row]), atol=0.02), angle)
            self.assertTrue(np.allclose(left if flipped else right, white, atol=0.02), angle)
        with self.assertRaises(ValueError):
            SpriteSheet(image, directions=7, mirrored=True)

    def test_upright_stays_upright(self):
        # Looked down on, an upright billboard stays standing (foreshortened); one turned to the picture doesn't.
        card = Mesh(np.array([(-1.0, 0, 0), (1, 0, 0), (1, 2, 0), (-1, 2, 0)]), np.array([(0, 1, 2), (0, 2, 3)]))
        above = Camera(position=np.array([0.0, 6.0, 4.0]), target=np.zeros(3))
        tall = {}
        for kind in (True, "upright"):
            ids = Renderer(40, 40).render([Object3D(card, billboard=kind, emissive=1.0)], above, []).ids
            rows = np.flatnonzero((ids > 0).any(axis=1))
            tall[kind] = len(rows)
            wide = np.flatnonzero((ids > 0).any(axis=0))
            self.assertGreater(len(wide), 10)
        self.assertLess(tall["upright"], 0.8 * tall[True])
        # Turned to the camera however it is turned itself; straight above, edge on, with no errors.
        turned = Object3D(card, billboard="upright", emissive=1.0,
                          rotation=quat_axis_angle(np.array([1.0, 1.0, 0.0]), 1.0))
        side = Renderer(40, 40).render([turned], around(60, height=0.0), []).ids
        self.assertGreater(len(np.flatnonzero((side > 0).any(axis=0))), 10)
        Renderer(40, 40).render([turned], Camera(position=np.array([0.0, 6.0, 0.0]), target=np.zeros(3)), [])

    def test_animations(self):
        sheet = sheet_of(8, columns=6, animations={"idle": (0, 1, 2.0), "walk": (1, 4, 8.0), "wave": (5, 1, 1.0)})
        sprite = Sprite(sheet, emissive=1.0)
        self.assertEqual(sprite.animation, "idle")
        renderer = Renderer(40, 40)
        sprite.play("walk")
        seen = []
        for _ in range(6):
            seen.append(sprite.frame)
            self.assertEqual(shown(middle([sprite], around(90), renderer), 6), (2, 1 + min(sprite.frame, 3)))
            sprite.update(1 / 8)
        self.assertEqual(seen, [0, 1, 2, 3, 0, 1])
        sprite.play("walk")  # (already playing: carries on)
        self.assertEqual(sprite.frame, 2)
        sprite.play("walk", loop=False, restart=True)
        sprite.speed = 2.0
        sprite.update(0.2)  # (3.2 frames)
        self.assertEqual((sprite.frame, sprite.finished), (3, False))
        sprite.update(0.1)
        self.assertEqual((sprite.frame, sprite.finished), (3, True))
        with self.assertRaises(ValueError):
            sprite.play("run")
        with self.assertRaises(ValueError):
            sheet_of(8, columns=6, animations={"long": (4, 4, 8.0)})  # (off the sheet)
        # Stacked blocks of rows: an animation from row 8 down.
        stacked = SpriteSheet(np.zeros((64, 8, 3)), columns=2, rows=16, animations={"a": (0, 2, 1.0),
                                                                                     "b": (0, 2, 1.0, 8)})
        self.assertEqual(stacked.cell("b", 1, 3), ((8 + 3) * 2 + 1, False))

    def test_cutouts_picking_and_many(self):
        image = np.zeros((8 * 4, 4, 4))
        image[..., :3] = (1.0, 0.0, 0.0)
        image[:, :, 3] = 1.0
        image[:, :2, 3] = 0.0  # (the left half of every cell clear)
        sheet = SpriteSheet(image, height=2.0, origin=(0.5, 0.5))
        sprite = Sprite(sheet, emissive=1.0)
        behind = Object3D(Mesh(np.array([(-5.0, -5, -2), (5, -5, -2), (5, 5, -2), (-5, 5, -2)]),
                               np.array([(0, 1, 2), (0, 2, 3)])), color=(0, 0, 255), emissive=1.0)
        renderer = Renderer(40, 40)
        fb = renderer.render([behind, sprite], around(0), [])
        self.assertEqual(fb.ids[40, 13], 1)  # (through the clear half)
        self.assertGreater(fb.rgb[40, 13, 2], 0.5)
        self.assertGreater(fb.rgb[40, 27, 0], 0.9)
        self.assertIs(renderer.pick(27, 20).object, sprite)
        # Many sprites of one sheet, in different frames and sides, beside a plain object showing one of its cells.
        sheet = sheet_of(8, columns=3, animations={"a": (0, 3, 1.0)}, origin=(0.5, 0.0), height=0.5)
        crowd = [Sprite(sheet, position=np.array([x, 0.0, z]), emissive=1.0) for x in range(-4, 5) for z in (-2, 0)]
        for k, s in enumerate(crowd):
            s.time = k % 3
            s.rotation = quat_axis_angle(np.array([0.0, 1.0, 0.0]), k * 0.7)
        still = Object3D(sheet.meshes[4], position=np.array([0.0, 1.0, 0.0]), color=(255, 255, 255), emissive=1.0)
        camera = Camera(position=np.array([0.0, 3.0, 9.0]), target=np.zeros(3))
        a = renderer.render(crowd + [still], camera, []).rgb.copy()
        crowd[9].time += 1.0  # (the one in the middle)
        b = renderer.render(crowd + [still], camera, []).rgb
        self.assertFalse(np.array_equal(a, b))
        np.testing.assert_array_equal(b, Renderer(40, 40).render(crowd + [still], camera, []).rgb)

    def test_seen_in_a_mirror_from_its_other_side(self):
        sprite = Sprite(sheet_of(8), emissive=1.0, position=np.array([0.0, 0.0, 0.0]))
        mirror = Object3D(Mesh(np.array([(-20.0, -20, -4), (20, -20, -4), (20, 20, -4), (-20, 20, -4)]),
                               np.array([(0, 1, 2), (0, 2, 3)])), color=(0, 0, 0), reflectivity=1.0, specular=0.0)
        camera = Camera(position=np.array([0.0, 1.5, 6.0]), target=np.array([0.0, 0.0, -4.0]))
        fb = Renderer(60, 40).render([mirror, sprite], camera, [])
        ids, rgb = fb.ids, fb.rgb
        # (in the mirror, the light it passes on is dimmer: compared by hue)
        in_mirror = {shown(rgb[y, x] / rgb[y, x].max()) for y, x in zip(*np.nonzero(ids == 1)) if rgb[y, x].max() > 0.2}
        self.assertIn((4, 0), in_mirror)  # (its back)
        self.assertNotIn((0, 0), in_mirror)
        self.assertEqual({shown(rgb[y, x] / rgb[y, x].max()) for y, x in zip(*np.nonzero(ids == 2))} - {None}, {(0, 0)})

    def test_bad_numbers_and_odd_cases_do_not_crash(self):
        sheet = sheet_of(8)
        sprites = [Sprite(sheet), Sprite(sheet, rotation=np.array([np.nan, 0, 0, 0])),
                   Sprite(sheet, scale=0.0), Sprite(sheet, scale=(0.0, 1.0, 1.0)), Sprite(sheet, position=(0, 5, 0)),
                   Sprite(sheet, billboard=False), Sprite(sheet, billboard=True)]
        sprites[0].update(np.nan)
        sprites[0].update(np.inf)
        self.assertEqual(sprites[0].frame, 0)
        sprites[2].time = np.nan
        sprites[3].time, sprites[3].speed = 1e300, 1e300
        sprites[3].update(1e300)
        for camera in (around(0), Camera(position=np.zeros(3), target=np.zeros(3)), around(0, distance=0.0),
                       Camera(position=np.array([0.0, 5.0, 0.0]), target=np.zeros(3)),
                       Camera(position=np.array([0.0, np.nan, 0.0])), Camera(projection="ortho"),
                       Camera(position=np.array([0.0, 0.0, 5.0]), up=np.array([1.0, 0.0, 0.0])),
                       Camera(near=0.0)):
            Renderer(40, 40).render(sprites, camera, Light())
        with self.assertRaises(ValueError):
            SpriteSheet(np.zeros((4, 4)), columns=1)
        with self.assertRaises(ValueError):
            SpriteSheet(np.zeros((4, 4, 3)), columns=8)  # (cells less than a pixel)
        with self.assertRaises(ValueError):
            SpriteSheet(np.zeros((8, 8, 3)), columns=1, directions=0)


if __name__ == "__main__":
    unittest.main()
