"""Z-buffered, vectorized triangle rasterizer writing into a pixel framebuffer.

Pixels are finer than terminal cells: a cell holds PIXELS_PER_CELL pixels stacked
vertically (drawn with half-block characters), so a pixel is roughly square.
"""
import numpy as np

PIXELS_PER_CELL = 2
MAX_FRAGMENTS = 1 << 21  # candidate pixels tested per batch, to bound memory


class FrameBuffer:
    """A grid of pixels.

    shade: surface brightness 0..1, or -1 where nothing was drawn.
    color: colour id of the object that owns the pixel.
    depth: interpolated 1/w (larger is nearer, 0 is empty).
    """

    def __init__(self, width, height):
        self.resize(width, height)

    def resize(self, width, height):
        self.width, self.height = width, height
        self.depth = np.zeros((height, width))
        self.shade = np.full((height, width), -1.0)
        self.color = np.zeros((height, width), dtype=np.int16)

    def clear(self):
        self.depth.fill(0.0)
        self.shade.fill(-1.0)
        self.color.fill(0)

    def copy(self):
        fb = FrameBuffer.__new__(FrameBuffer)
        fb.width, fb.height = self.width, self.height
        fb.depth, fb.shade, fb.color = self.depth.copy(), self.shade.copy(), self.color.copy()
        return fb

    @property
    def drawn(self):
        return self.shade >= 0


def _bboxes(xs, ys, width, height):
    x0 = np.clip(np.floor(xs.min(axis=1)), 0, width).astype(np.int64)
    x1 = np.clip(np.ceil(xs.max(axis=1)), -1, width - 1).astype(np.int64)
    y0 = np.clip(np.floor(ys.min(axis=1)), 0, height).astype(np.int64)
    y1 = np.clip(np.ceil(ys.max(axis=1)), -1, height - 1).astype(np.int64)
    bw, bh = np.maximum(x1 - x0 + 1, 0), np.maximum(y1 - y0 + 1, 0)
    return x0, y0, bw, bh


def _batches(counts):
    """Split triangle indices into runs whose candidate-pixel totals stay under MAX_FRAGMENTS."""
    start, total = 0, 0
    for i, n in enumerate(counts):
        if total and total + n > MAX_FRAGMENTS:
            yield start, i
            start, total = i, 0
        total += n
    if start < len(counts):
        yield start, len(counts)


def rasterize(fb, xs, ys, inv_w):
    """Depth-test every triangle's pixels against fb.depth and keep the nearest.

    xs, ys, inv_w: (T, 3) pixel coordinates and 1/w of each corner.
    Updates fb.depth and returns (pixel flat indices, triangle indices, barycentric
    weights (N, 3)) of the pixels that were won, for the caller to shade.
    """
    out_pix, out_tri, out_bary = [], [], []
    area = (xs[:, 1] - xs[:, 0]) * (ys[:, 2] - ys[:, 0]) - (ys[:, 1] - ys[:, 0]) * (xs[:, 2] - xs[:, 0])
    x0, y0, bw, bh = _bboxes(xs, ys, fb.width, fb.height)
    counts = np.where(np.abs(area) > 1e-9, bw * bh, 0)
    depth = fb.depth.reshape(-1)
    for a, b in _batches(counts):
        n = counts[a:b]
        total = int(n.sum())
        if not total:
            continue
        tri = np.repeat(np.arange(a, b), n)
        local = np.arange(total) - np.repeat(np.cumsum(n) - n, n)
        px = x0[tri] + local % bw[tri]
        py = y0[tri] + local // bw[tri]
        cx, cy = px + 0.5, py + 0.5
        X, Y = xs[tri], ys[tri]

        def edge(i, j):
            return (X[:, j] - X[:, i]) * (cy - Y[:, i]) - (Y[:, j] - Y[:, i]) * (cx - X[:, i])

        bary = np.stack([edge(1, 2), edge(2, 0), edge(0, 1)], axis=1) / area[tri, None]
        inside = (bary >= -1e-4).all(axis=1)  # the slack closes hairline gaps between triangles
        z = np.einsum("ij,ij->i", bary, inv_w[tri])
        pix = py * fb.width + px
        keep = inside & (z > depth[pix])
        if not keep.any():
            continue
        pix, z, tri, bary = pix[keep], z[keep], tri[keep], bary[keep]
        # Several triangles may cover one pixel: the nearest wins.
        order = np.lexsort((-z, pix))
        first = np.r_[True, pix[order][1:] != pix[order][:-1]]
        win = order[first]
        depth[pix[win]] = z[win]
        out_pix.append(pix[win])
        out_tri.append(tri[win])
        out_bary.append(bary[win])
    if not out_pix:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros((0, 3))
    pix, tri, bary = np.concatenate(out_pix), np.concatenate(out_tri), np.concatenate(out_bary)
    if len(out_pix) > 1:  # a later batch may have overdrawn an earlier one
        latest = np.unique(pix[::-1], return_index=True)[1]
        latest = len(pix) - 1 - latest
        pix, tri, bary = pix[latest], tri[latest], bary[latest]
    return pix, tri, bary


def perspective_interpolate(values, tri, bary, inv_w):
    """Perspective-correct interpolation of per-corner values (T, 3, K) at the given fragments."""
    w = bary * inv_w[tri]
    return np.einsum("ij,ijk->ik", w, values[tri]) / w.sum(axis=1, keepdims=True)


def clip_near(corners, near):
    """Clip triangles against the near plane w = near.

    corners: (T, 3, K) per-corner attributes with clip-space w at index 3; every
    attribute is interpolated linearly, which is correct in clip space.
    Returns the surviving triangles (T', 3, K) and, for each, the index of its source triangle.
    """
    w = corners[:, :, 3]
    ahead = w > near
    n_ahead = ahead.sum(axis=1)
    whole = np.flatnonzero(n_ahead == 3)
    parts, sources = [corners[whole]], [whole]
    for t in np.flatnonzero((n_ahead > 0) & (n_ahead < 3)):
        poly = []
        for i in range(3):
            a, b = corners[t, i], corners[t, (i + 1) % 3]
            if ahead[t, i]:
                poly.append(a)
            if ahead[t, i] != ahead[t, (i + 1) % 3]:
                s = (near - a[3]) / (b[3] - a[3])
                poly.append(a + s * (b - a))
        for k in range(1, len(poly) - 1):  # fan-triangulate the clipped polygon
            parts.append(np.array([[poly[0], poly[k], poly[k + 1]]]))
            sources.append(np.array([t]))
    return np.concatenate(parts), np.concatenate(sources)
