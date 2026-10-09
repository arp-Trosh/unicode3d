# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 arp-Trosh
"""Mirrors: flat, solid, reflective objects show the scene reflected in their plane, drawn in extra passes
(Renderer's Mirrors part, and its kernels)."""
import numpy as np
from numba import njit, prange

from .background import sky_colour
from .raster import CUT, SAMPLE_PATTERNS, SOLID

MAX_BOUNCES = 4          # the deepest Renderer.mirror_bounces goes
MIRROR_MIN_PIXELS = 50   # a mirror seen in a mirror smaller than this shows no reflection
MIRROR_MIN_WEIGHT = 0.05  # nor one whose reflection would count for less of the frame's colour than this


@njit(cache=True, error_model="numpy", parallel=True)
def _mirror_cover(tris, tri_inst, ident, mirror, cover):
    """How many of each pixel's samples (rows of tris: rasterize()'s triangle at each) landed on the object
    whose id is `mirror`, into cover."""
    for c in prange(tris.shape[0]):
        n = 0
        for s in range(tris.shape[1]):
            t = tris[c, s]
            n += t >= 0 and ident[tri_inst[t]] == mirror
        cover[c] = n


@njit(cache=True, error_model="numpy", parallel=True)
def _gather_pass(pixels, rgb, cover, depth, ids, n_samples, out_rgb, out_alpha, out_depth, out_ids):
    """A pass drawn in only some pixels (rows of rgb, cover, depth, ids, one for each of `pixels`, flat indices)
    into whole-frame arrays: colour premultiplied by coverage, coverage, depth and id."""
    for j in prange(pixels.shape[0]):
        c = pixels[j]
        for k in range(3):
            out_rgb[c, k] = rgb[j, k] / n_samples
        out_alpha[c], out_depth[c], out_ids[c] = cover[j] / n_samples, depth[j], ids[j]


@njit(cache=True, error_model="numpy", parallel=True)
def _fill_sky(pixels, rgb, alpha, width, height, axes, sky, sky_colors, sky_faces, sky_texels, sky_levels, sky_first,
              sky_lod):
    """Fill what the scene leaves uncovered in each of `pixels` (flat indices of whole-frame rgb, alpha) with
    the background seen along the ray through the pixel of a view whose pixels look along `axes` (see
    transforms.view_axes; a reflected view: what a mirror shows beyond everything), and make the pixel opaque."""
    for j in prange(pixels.shape[0]):
        c = pixels[j]
        a = alpha[c]
        if a >= 1.0:
            continue
        nx, ny = (c % width + 0.5) / width * 2.0 - 1.0, 1.0 - (c // width + 0.5) / height * 2.0
        dx = axes[0, 0] + nx * axes[1, 0] + ny * axes[2, 0]
        dy = axes[0, 1] + nx * axes[1, 1] + ny * axes[2, 1]
        dz = axes[0, 2] + nx * axes[1, 2] + ny * axes[2, 2]
        dl = max(np.sqrt(dx * dx + dy * dy + dz * dz), 1e-30)
        r, g, b = sky_colour(sky, sky_colors, sky_faces, sky_texels, sky_levels, sky_first, sky_lod, dx / dl, dy / dl,
                             dz / dl)
        rgb[c, 0] += (1.0 - a) * r
        rgb[c, 1] += (1.0 - a) * g
        rgb[c, 2] += (1.0 - a) * b
        alpha[c] = 1.0


@njit(cache=True, error_model="numpy", parallel=True)
def _mix_mirror(target, pixels, rows, tris, sample_rgb, extra_rows, extra_tris, extra_rgb, tri_inst, ident, mirror,
                reflectivity, seen):
    """Show what a mirror reflects: in each of `pixels` (flat indices of whole-frame colours `target`,
    premultiplied by coverage), the share of the pixel's samples that fell on the mirror (id `mirror`)
    changes from the mirror's own colour towards `seen` (whole-frame, opaque) by `reflectivity`.

    The samples are row rows[j] of tris and sample_rgb (the pass the pixel was drawn in), and, where the
    pixel took extra samples at an edge (extra_rows[c] >= 0, indexed by flat pixel), that row of
    extra_tris and extra_rgb too, as resolve() averages them into the frame."""
    n_base, n_extra = tris.shape[1], extra_tris.shape[1]
    for j in prange(pixels.shape[0]):
        c, row, extra = pixels[j], rows[j], extra_rows[pixels[j]]
        own_r = own_g = own_b = 0.0
        n = 0
        for s in range(n_base):
            t = tris[row, s]
            if t >= 0 and ident[tri_inst[t]] == mirror:
                n += 1
                own_r, own_g, own_b = own_r + sample_rgb[row, s, 0], own_g + sample_rgb[row, s, 1], own_b + sample_rgb[
                    row, s, 2]
        total = n_base
        if extra >= 0:
            total += n_extra
            for s in range(n_extra):
                t = extra_tris[extra, s]
                if t >= 0 and ident[tri_inst[t]] == mirror:
                    n += 1
                    own_r, own_g, own_b = (own_r + extra_rgb[extra, s, 0], own_g + extra_rgb[extra, s, 1],
                                           own_b + extra_rgb[extra, s, 2])
        part = n / total
        # (At least 0: the frame's colours are float32, so taking the mirror's own share back out of them can leave
        # a rounding error either way. max(0.0, NaN) is 0.0.)
        target[c, 0] = max(0.0, target[c, 0] + reflectivity * (part * seen[c, 0] - own_r / total))
        target[c, 1] = max(0.0, target[c, 1] + reflectivity * (part * seen[c, 1] - own_g / total))
        target[c, 2] = max(0.0, target[c, 2] + reflectivity * (part * seen[c, 2] - own_b / total))


class Mirrors:
    """The Renderer's mirrors (a part of Renderer, in its own module)."""

    def _mirrors(self, inst):
        """Which instances are mirrors: flat, solid (not see-through, no holes) and reflective (Object3D
        .reflectivity), as a boolean array; each shows the scene reflected in its plane (see _reflect)."""
        pack = inst["pack"]
        return ((inst["shine"] > 0.0) & (inst["alpha"] >= 1.0) & ~pack["clear"][inst["mesh"]]
                & np.isfinite(pack["planes"][inst["mesh"], 0]))

    def _reflects(self, inst, level):
        """Whether a pass `level` mirrors deep (0: the frame itself) showing instances `inst` has mirrors to show
        the scene in (see _reflect), which then need its samples' colours."""
        return bool(self.reflections and level < min(int(self.mirror_bounces), MAX_BOUNCES)
                    and self._mirrors(inst).any())

    def _pass_shine(self, inst, level):
        """Each instance's reflectivity of the background (see _shade) in a pass `level` mirrors deep (0: the
        frame itself): mirrors reflect the scene instead (their own pass) until mirror_bounces runs out,
        and then the background."""
        shine = inst["shine"].copy()
        if self.reflections and level < min(int(self.mirror_bounces), MAX_BOUNCES):
            shine[self._mirrors(inst)] = 0.0
        return shine

    def _plane(self, inst, i):
        """Instance i's mirror plane in the world: (unit normal (3,), d), n . x + d = 0 on it; None if it is squashed
        flat (a scale of 0)."""
        normal, d = inst["pack"]["planes"][inst["mesh"][i], :3], inst["pack"]["planes"][inst["mesh"][i], 3]
        lin = inst["lin"][i]
        point = inst["pos"][i] + lin @ (-d * normal)
        # Normals turn by the inverse transpose of lin: its cofactors, up to a factor (see raster._normal_matrix).
        cof = np.array([np.cross(lin[:, 1], lin[:, 2]), np.cross(lin[:, 2], lin[:, 0]), np.cross(lin[:, 0], lin[:, 1])])
        n = cof.T @ normal
        length = np.linalg.norm(n)
        if not length > 1e-300:
            return None
        n = n / length
        return n, -float(n @ point)

    def _reflect(self, scene, drawn, pixels, rows, target, level, weight):
        """Show what each mirror drawn in a pass reflects. scene, drawn: the pass (its scene and _accumulate()
        output), which covered `pixels` (flat indices; drawn's rows[j] is pixel pixels[j]); target: its
        whole-frame colours (premultiplied), which the mirrors' share of each pixel is changed in. level:
        how many mirrors deep the pass is; weight: how much of the frame's colour it makes up (the
        reflectivities multiplied along the way), so that faint images deep down are skipped.
        """
        inst, fb, buf = scene["inst"], self._fb, self._buffers
        mirrors = np.flatnonzero(self._mirrors(inst))
        if level >= min(int(self.mirror_bounces), MAX_BOUNCES) or not len(mirrors):
            return
        n = len(SAMPLE_PATTERNS[self.samples])
        cover = buf.get(f"mirror{level}_cover", (len(rows),), np.int64)
        for i in mirrors:
            ident, reflectivity = int(inst["ident"][i]), float(inst["shine"][i])
            if weight * reflectivity < MIRROR_MIN_WEIGHT:
                continue
            _mirror_cover(drawn["tris"], scene["tri_inst"], scene["ident"], ident, cover)
            on = np.flatnonzero(cover)
            if not len(on) or (level > 0 and len(on) < MIRROR_MIN_PIXELS):
                continue
            # The camera reflected in the mirror's plane, which faces it; what is behind the plane is cut away.
            plane = self._plane(inst, i)
            if plane is None:
                continue
            normal, d = plane
            eye = scene["eye"]
            if normal @ eye + d < 0.0:
                normal, d = -normal, -d
            mirror = np.eye(4)
            mirror[:3, :3] -= 2.0 * np.outer(normal, normal)
            mirror[:3, 3] = -2.0 * d * normal
            view_proj = scene["view_proj"] @ mirror
            axes = scene["axes"] @ mirror[:3, :3]  # (each pixel's direction, reflected: the matrix is symmetric)
            keep = np.arange(len(inst["mesh"])) != i
            sub = {k: (v[keep] if isinstance(v, np.ndarray) and len(v) == len(keep) else v) for k, v in inst.items()}
            prefix = f"mirror{level}_"
            child = self._view(sub, view_proj, (mirror @ np.r_[eye, 1.0])[:3], scene["near"], np.r_[normal, d],
                               self._pass_shine(sub, level + 1), scene["shading"], prefix, axes)
            shown = pixels[on]
            seen = (buf.get(prefix + "rgb_frame", (fb.width * fb.height, 3), np.float32),  # (as the framebuffer's)
                    buf.get(prefix + "alpha_frame", (fb.width * fb.height,), np.float32),
                    buf.get(prefix + "depth_frame", (fb.width * fb.height,)),
                    buf.get(prefix + "ids_frame", (fb.width * fb.height,), np.int32))
            if child is None:  # (else _gather_pass writes all of each pixel shown)
                seen[0][shown], seen[1][shown] = 0.0, 0.0
            else:
                span = self._spans(shown)
                child["bands"] = self._bins(child["xs"], child["ys"], child["see"], SOLID | CUT, fb.height, prefix,
                                            span)
                pattern = SAMPLE_PATTERNS[self.samples]
                image = self._accumulate(child, pattern, prefix + "pass", shown, keep=self._reflects(sub, level + 1))
                _gather_pass(shown, image["rgb"], image["cover"], buf.get("near_depth", (len(shown),)),
                             buf.get("near_id", (len(shown),), np.int32), n, *seen)
                # As in the frame itself: mirrors in the reflection, then glass in front, then the background.
                self._reflect(child, image, shown, np.arange(len(shown)), seen[0], level + 1, weight * reflectivity)
                if (child["see"] == 1).any():
                    solid = buf.get(prefix + "solid", (fb.width * fb.height, n))
                    solid.fill(np.inf)  # (nothing is drawn outside the mirror)
                    solid[shown] = image["sample_depth"]
                    layers = self._layers(child, pattern, solid, prefix, span)
                    if layers is not None:
                        self._blend_layers(child, layers, n, seen)
            _fill_sky(shown, seen[0], seen[1], fb.width, fb.height, axes, *self._sky)
            extra_rows, extra_tris, extra_rgb = drawn.get("extra") or self._no_extra()
            _mix_mirror(target, shown, rows[on], drawn["tris"], drawn["sample_rgb"], extra_rows, extra_tris, extra_rgb,
                        scene["tri_inst"], scene["ident"], ident, reflectivity, seen[0])

    def _no_extra(self):
        """What _mix_mirror takes for a pass without extra samples at edges: no pixel has any."""
        buf, m = self._buffers, self._fb.width * self._fb.height
        rows = buf.get("no_extra_rows", (m,), np.int64)
        rows.fill(-1)
        return rows, np.zeros((0, 1), np.int32), np.zeros((0, 1, 3))
