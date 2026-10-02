# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
"""Lights: Light (the sun, from one direction) and PointLight (a lamp), and their packing for the kernels."""
from dataclasses import dataclass, field

import numpy as np

from .color import cached_linear_rgb, to_srgb
from .transforms import normalize


def _vec3(*v):
    return field(default_factory=lambda: np.array(v, dtype=float))



@dataclass
class Light:
    """Light from far away, arriving from one direction everywhere (like the sun).

    Light levels are perceived brightness, 0..1: a level of 0.5 looks half as bright. A surface gets
    ambient everywhere plus diffuse where it faces the light, from every light in the scene, summed.
    """
    direction: np.ndarray = _vec3(0.3, -1.0, -0.5)  # the way the light travels
    ambient: float = 0.3    # light levels are perceived brightness, 0..1
    diffuse: float = 0.7
    specular: float = 0.35  # strength of the Blinn-Phong highlight, in the light's colour whatever the surface's
    shininess: float = 24.0
    color: object = (255, 255, 255)  # a named Color, or (r, g, b) sRGB: it scales the light's levels channel by channel
    shadows: bool = False   # objects block this light from what lies behind them (see Renderer for the settings)


@dataclass
class PointLight:
    """Light spreading from a point (a lamp, a torch), fading to nothing at `range` from it.

    Its levels (see Light) are full at the light and fall off smoothly with distance. ambient is
    light it gives surfaces nearby whichever way they face.
    """
    position: np.ndarray = _vec3(0.0, 2.0, 0.0)
    color: object = (255, 255, 255)
    diffuse: float = 0.8
    specular: float = 0.35
    shininess: float = 24.0
    range: float = 10.0
    ambient: float = 0.0
    shadows: bool = False   # objects block this light, in every direction (see Renderer for the settings)


LIGHT_COLUMNS = 16  # kind, direction or position (3), colour as levels (3), linear colour (3), ambient, diffuse,
                    # specular, shininess, range, first shadow map (-1 for none)


def light_rows(lights, shadows=True):
    """The lights as a (L, LIGHT_COLUMNS) array for shading._shade; shadows=False leaves out every light's shadows."""
    rows = np.zeros((len(lights), LIGHT_COLUMNS))
    rows[:, 15] = -1
    maps = 0
    for i, light in enumerate(lights):
        lin = cached_linear_rgb(light.color)
        if isinstance(light, PointLight):
            rows[i, 0], rows[i, 1:4], rows[i, 14] = 1, np.asarray(light.position, float), float(light.range)
        else:
            rows[i, 1:4] = -normalize(light.direction)
        if light.shadows and shadows:  # its first shadow map: one for a Light, six (a cube) for a PointLight
            rows[i, 15], maps = maps, maps + (6 if isinstance(light, PointLight) else 1)
        rows[i, 4:7], rows[i, 7:10] = to_srgb(light.color), lin
        rows[i, 10:14] = light.ambient, light.diffuse, light.specular, light.shininess
    # NaN as 0 (a light that adds nothing there) and infinities as the largest numbers (a range of inf still fades
    # out nowhere), as either could reach the picture as NaN.
    return rows if np.isfinite(rows).all() else np.nan_to_num(rows)


def as_lights(lights):
    """A light or a sequence of lights, as a list."""
    return [lights] if isinstance(lights, (Light, PointLight)) else list(lights)
