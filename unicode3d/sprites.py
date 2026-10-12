# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
"""Sprite sheets: a picture cut into cells, a character drawn from several directions (one row of cells each)
in the frames of its animations (one column each), shown by a scene.Sprite. Each cell is a flat rectangle with
its own texture (its own mipmaps, so that a small sprite never blurs into the cells next to it on the sheet)."""
import math

import numpy as np

from .mesh import Mesh
from .texture import load_image


class SheetCell(Mesh):
    """A cell's rectangle (a Mesh): `sheet` is the SpriteSheet it belongs to, `index` its place in sheet.meshes."""
    sheet = None
    index = 0


def _clip(value, name):
    """A sheet's animation as (first column, frames, frames a second, first row)."""
    first, frames, fps, row = (tuple(value) + (0,))[:4] if len(tuple(value)) == 3 else tuple(value)
    first, frames, row, fps = int(first), int(frames), int(row), float(fps)
    if frames < 1 or first < 0 or row < 0 or not math.isfinite(fps) or fps < 0.0:
        raise ValueError(f"animation {name!r} must be (first column, frames >= 1, frames a second >= 0"
                         f"[, first row]), not {value!r}")
    return first, frames, fps, row


class SpriteSheet:
    """A picture cut into a grid of cells: a character drawn from `directions` sides, one row each, and in the
    frames of its animations, one column each. Shown by Sprite objects, which pick the cell for the way the
    camera sees them (see Sprite):

        sheet = SpriteSheet("knight.png", columns=8, height=1.8,
                            animations={"idle": (0, 1, 4.0),      # (first column, frames, frames a second)
                                        "walk": (1, 4, 8.0),
                                        "attack": (5, 3, 10.0)})

    image: a file name (see load_image), or an (H, W, 3) or (H, W, 4) array, 0..1 sRGB floats or 0..255 ints.
        Alpha makes clear parts (cut-outs), as in any texture.
    columns, rows: how many cells across and down the picture is cut into. rows is by default as many as there
        are rows of directions (`directions`, or directions // 2 + 1 when mirrored) below the lowest
        animation's first row.
    directions: how many sides the character is drawn from, evenly around it. Row 0 is its front (facing the
        viewer); each row after it shows it turned a further 360 / directions degrees clockwise as seen from above
        (counter-clockwise with clockwise=False). With 8 directions: front, front-left, left, back-left, back, and
        so on (left and right as the picture shows them). 1: the same row seen from anywhere.
    mirrored: the sheet holds only the first half of the turn (directions // 2 + 1 rows: front to back) and the
        rest are those rows flipped left to right, as in classic sheets of five rows of eight directions.
    height: how tall a cell stands in the world (its width follows from the cell's shape in pixels).
    origin: the point of a cell, as fractions of its width from the left and of its height from the bottom, that
        stands at the sprite's position: (0.5, 0.0) puts a character's feet there; (0.5, 0.5) its middle.
    animations: {name: (first column, frames, frames a second[, first row])}: each a run of frames along the rows,
        from its first row of directions (0 by default; a sheet may stack several blocks of rows). None: one
        animation, "still", the first cell of each row.
    margin, spacing: pixels around the edge of the picture, and between cells, left out.

    meshes holds every cell's rectangle, rows * columns of them (row by row), each with its cell's texture, as
    SheetCell meshes; a Sprite shows the first, and the renderer swaps in the one it needs."""

    def __init__(self, image, columns=1, rows=None, directions=8, mirrored=False, clockwise=True, height=1.0,
                 origin=(0.5, 0.0), animations=None, margin=0, spacing=0):
        picture = load_image(image) if not isinstance(image, np.ndarray) else image
        picture = np.asarray(picture)
        picture = picture / 255.0 if np.issubdtype(picture.dtype, np.integer) else picture.astype(np.float64)
        if picture.ndim != 3 or picture.shape[2] not in (3, 4):
            raise ValueError(f"a sprite sheet's image must be (H, W, 3) or (H, W, 4), not {picture.shape}")
        picture = np.nan_to_num(np.clip(picture, 0.0, 1.0))
        directions = int(directions)
        if directions < 1 or mirrored and directions % 2:
            raise ValueError(f"directions must be at least 1 (and even when mirrored), not {directions}")
        held = directions // 2 + 1 if mirrored else directions  # (rows of directions on the sheet)
        clips = {name: _clip(value, name) for name, value in (animations or {"still": (0, 1, 0.0)}).items()}
        if not clips:
            raise ValueError("a sprite sheet needs at least one animation")
        columns = int(columns)
        rows = max(row for _, _, _, row in clips.values()) + held if rows is None else int(rows)
        margin, spacing = int(margin), int(spacing)
        high, wide = picture.shape[:2]
        cell_w = (wide - 2 * margin - (columns - 1) * spacing) // columns if columns > 0 else 0
        cell_h = (high - 2 * margin - (rows - 1) * spacing) // rows if rows > 0 else 0
        if columns < 1 or rows < 1 or cell_w < 1 or cell_h < 1 or margin < 0 or spacing < 0:
            raise ValueError(f"can't cut a {wide}x{high} picture into {columns}x{rows} cells "
                             f"(margin {margin}, spacing {spacing})")
        for name, (first, frames, _, row) in clips.items():
            if first + frames > columns or row + held > rows:
                raise ValueError(f"animation {name!r} runs off the sheet's {columns} columns and {rows} rows")
        self.columns, self.rows, self.directions, self.held = columns, rows, directions, held
        self.mirrored, self.clockwise, self.animations = bool(mirrored), bool(clockwise), clips
        self.cell_size = (cell_w, cell_h)
        height = float(height)
        width = height * cell_w / cell_h
        ox, oy = (float(v) for v in origin)
        corners = np.array([(-ox * width, -oy * height, 0.0), ((1 - ox) * width, -oy * height, 0.0),
                            ((1 - ox) * width, (1 - oy) * height, 0.0), (-ox * width, (1 - oy) * height, 0.0)])
        faces = np.array([(0, 1, 2), (0, 2, 3)])
        # (half a texel in from the edges: sampling at an edge would blend in the cell's opposite one)
        u0, u1, v0, v1 = 0.5 / cell_w, 1 - 0.5 / cell_w, 0.5 / cell_h, 1 - 0.5 / cell_h
        uvs = np.array([[(u0, v0), (u1, v0), (u1, v1)], [(u0, v0), (u1, v1), (u0, v1)]])
        materials = np.zeros(2, np.int64)
        self.meshes = []
        for r in range(rows):
            for c in range(columns):
                x, y = margin + c * (cell_w + spacing), margin + r * (cell_h + spacing)
                cell = SheetCell(corners, faces, uvs, materials,
                                 [np.ascontiguousarray(picture[y:y + cell_h, x:x + cell_w])])
                cell.sheet, cell.index = self, len(self.meshes)
                self.meshes.append(cell)

    def cell(self, animation, frame, direction):
        """Index into meshes of `animation`'s frame (from 0) seen from `direction` (0: from the front, then on
        around as the rows go), and whether that cell is shown flipped (mirrored sheets)."""
        first, frames, _, row = self.animations[animation]
        direction %= self.directions
        flip = self.mirrored and direction >= self.held
        if flip:
            direction = self.directions - direction
        return (row + direction) * self.columns + first + min(max(int(frame), 0), frames - 1), flip
