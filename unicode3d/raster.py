"""Z-buffered, vectorized triangle rasterizer, and the framebuffer it renders into.

Pixels are finer than terminal cells: each cell covers a small grid of them
(FrameBuffer.cell_pixels), which the glyph set later turns into characters.
"""
import numpy as np

MAX_FRAGMENTS = 1 << 21  # candidate pixels tested per batch, to bound memory


class FrameBuffer:
    """A rendered image, cell_pixels = (columns, rows) of pixels to each terminal cell.

    rgb: linear-light colour premultiplied by coverage (0 where nothing was drawn).
    alpha: how much of the pixel is covered, 0..1.
    depth: 1/w of the nearest surface (larger is nearer, 0 is empty).
    ids: which object owns the pixel, as its index in the render list plus one (0 is empty).
    """

    def __init__(self, width, height, cell_pixels=(1, 2)):
        self.cell_pixels = tuple(cell_pixels)
        self.resize(width, height)

    def resize(self, width, height):
        self.width, self.height = width, height
        self.rgb = np.zeros((height, width, 3))
        self.alpha = np.zeros((height, width))
        self.depth = np.zeros((height, width))
        self.ids = np.zeros((height, width), dtype=np.int32)

    def clear(self):
        self.rgb.fill(0.0)
        self.alpha.fill(0.0)
        self.depth.fill(0.0)
        self.ids.fill(0)

    def copy(self):
        fb = FrameBuffer.__new__(FrameBuffer)
        fb.cell_pixels, fb.width, fb.height = self.cell_pixels, self.width, self.height
        fb.rgb, fb.alpha, fb.depth, fb.ids = self.rgb.copy(), self.alpha.copy(), self.depth.copy(), self.ids.copy()
        return fb

    @property
    def drawn(self):
        return self.alpha > 0

    def colour(self):
        """Each pixel's own linear colour, undoing the premultiplication by coverage."""
        return self.rgb / np.maximum(self.alpha, 1e-9)[..., None]


def _bboxes(xs, ys, width, height, offsets):
    """Per-triangle ranges of pixels that any sample position could put inside the triangle."""
    (oxmin, oymin), (oxmax, oymax) = offsets.min(axis=0), offsets.max(axis=0)
    x0 = np.clip(np.floor(xs.min(axis=1) - oxmax), 0, width).astype(np.int64)
    x1 = np.clip(np.ceil(xs.max(axis=1) - oxmin), -1, width - 1).astype(np.int64)
    y0 = np.clip(np.floor(ys.min(axis=1) - oymax), 0, height).astype(np.int64)
    y1 = np.clip(np.ceil(ys.max(axis=1) - oymin), -1, height - 1).astype(np.int64)
    bw, bh = np.maximum(x1 - x0 + 1, 0), np.maximum(y1 - y0 + 1, 0)
    return x0, y0, bw, bh


def _batches(counts):
    """Split triangle indices into runs whose candidate totals stay under MAX_FRAGMENTS."""
    start, total = 0, 0
    for i, n in enumerate(counts):
        if total and total + n > MAX_FRAGMENTS:
            yield start, i
            start, total = i, 0
        total += n
    if start < len(counts):
        yield start, len(counts)


def rasterize(depth, width, height, xs, ys, inv_w, offsets=((0.5, 0.5),), slots=None):
    """Depth-test every triangle at every sample position and keep the nearest surface.

    depth: (S, M) nearest 1/w so far for each of the S sample positions `offsets`
    ((x, y) within a pixel) in each of M pixels, updated in place. The pixels are
    all width * height of them in row order, or, with `slots` (an int array giving
    each pixel's column in depth, -1 to skip it), just a chosen few.
    xs, ys, inv_w: (T, 3) pixel coordinates and 1/w of each triangle corner.

    Returns (flat indices into depth, triangle indices, barycentric weights (N, 3))
    of the samples that were won, for the caller to shade.
    """
    offsets = np.asarray(offsets, dtype=float)
    n_samples, n_pixels = depth.shape
    out_idx, out_tri, out_bary = [], [], []
    area = (xs[:, 1] - xs[:, 0]) * (ys[:, 2] - ys[:, 0]) - (ys[:, 1] - ys[:, 0]) * (xs[:, 2] - xs[:, 0])
    x0, y0, bw, bh = _bboxes(xs, ys, width, height, offsets)
    counts = np.where(np.abs(area) > 1e-9, bw * bh, 0)
    flat_depth = depth.reshape(-1)
    for a, b in _batches(counts * n_samples):
        n = counts[a:b]
        total = int(n.sum())
        if not total:
            continue
        tri = np.repeat(np.arange(a, b), n)
        local = np.arange(total) - np.repeat(np.cumsum(n) - n, n)
        px = x0[tri] + local % bw[tri]
        py = y0[tri] + local // bw[tri]
        col = py * width + px
        if slots is not None:
            col = slots[col]
            sel = col >= 0
            px, py, col, tri = px[sel], py[sel], col[sel], tri[sel]
        # Every candidate pixel once per sample position.
        sample = np.tile(np.arange(n_samples), len(tri))
        tri, col = np.repeat(tri, n_samples), np.repeat(col, n_samples)
        cx = np.repeat(px, n_samples) + offsets[sample, 0]
        cy = np.repeat(py, n_samples) + offsets[sample, 1]
        X, Y = xs[tri], ys[tri]

        def edge(i, j):
            return (X[:, j] - X[:, i]) * (cy - Y[:, i]) - (Y[:, j] - Y[:, i]) * (cx - X[:, i])

        bary = np.stack([edge(1, 2), edge(2, 0), edge(0, 1)], axis=1) / area[tri, None]
        inside = (bary >= -1e-4).all(axis=1)  # the slack closes hairline gaps between triangles
        idx = sample * n_pixels + col
        z = np.einsum("ij,ij->i", bary, inv_w[tri])
        keep = inside & (z > flat_depth[idx])
        if not keep.any():
            continue
        idx, z, tri, bary = idx[keep], z[keep], tri[keep], bary[keep]
        # Several triangles may cover one sample: the nearest wins.
        order = np.lexsort((-z, idx))
        first = np.r_[True, idx[order][1:] != idx[order][:-1]]
        win = order[first]
        flat_depth[idx[win]] = z[win]
        out_idx.append(idx[win])
        out_tri.append(tri[win])
        out_bary.append(bary[win])
    if not out_idx:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros((0, 3))
    idx, tri, bary = np.concatenate(out_idx), np.concatenate(out_tri), np.concatenate(out_bary)
    if len(out_idx) > 1:  # a later batch may have overdrawn an earlier one
        latest = np.unique(idx[::-1], return_index=True)[1]
        latest = len(idx) - 1 - latest
        idx, tri, bary = idx[latest], tri[latest], bary[latest]
    return idx, tri, bary


def barycentric(xs, ys, tri, cx, cy):
    """Barycentric weights (N, 3) of points (cx, cy) in triangles `tri`, clamped to the triangle.

    Points outside their triangle (a pixel centre just past the edge of a triangle
    that covers some of the pixel's samples) snap onto it, so attributes are never
    extrapolated beyond the corners' values.
    """
    X, Y = xs[tri], ys[tri]

    def edge(i, j):
        return (X[:, j] - X[:, i]) * (cy - Y[:, i]) - (Y[:, j] - Y[:, i]) * (cx - X[:, i])

    area = edge(0, 1) + edge(1, 2) + edge(2, 0)
    bary = np.stack([edge(1, 2), edge(2, 0), edge(0, 1)], axis=1) / np.where(np.abs(area) > 1e-12, area, 1e-12)[:, None]
    bary = np.clip(bary, 0.0, None)
    return bary / np.maximum(bary.sum(axis=1, keepdims=True), 1e-12)


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
