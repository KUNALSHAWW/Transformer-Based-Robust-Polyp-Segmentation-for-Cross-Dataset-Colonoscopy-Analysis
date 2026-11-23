"""Helpers that need only numpy / scipy / cv2 (no torch): image transforms for size experiments and test-time clean-up of masks.
Every parameter below is FIXED BEFORE it was ever run on a test set (see the notes in the evidence folder); sensitivity grids are exploratory only.
python postproc.py   -> self-test"""
import cv2, numpy as np
from scipy import ndimage

CC8 = np.ones((3, 3), int)


# ------------------------------------------------------------------ transforms
def natural_bg(x):
    """A mucosa-coloured background made from the picture itself (flipped, strongly blurred), used instead of a black canvas."""
    return cv2.GaussianBlur(cv2.flip(x, -1), (0, 0), sigmaX=max(3.0, 0.02 * max(x.shape[:2])))


def paste_shrunk(x, m, s, rng, bg="natural"):
    """Shrink picture x and mask m by the factor s (0 < s <= 1) and paste them at a random place on a canvas of the ORIGINAL size.
    The polyp area share becomes (about) s^2 times smaller. bg = 'natural' (blurred copy of the picture) or 'black' (the old canvas)."""
    H, W = x.shape[:2]
    if s >= 0.999:
        return x.copy(), m.copy()
    h, w = max(16, int(round(H * s))), max(16, int(round(W * s)))
    xs = cv2.resize(x, (w, h), interpolation=cv2.INTER_AREA); ms = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    top, left = int(rng.integers(0, H - h + 1)), int(rng.integers(0, W - w + 1))
    cm = np.zeros((H, W), np.uint8); cm[top:top + h, left:left + w] = ms
    if bg == "black":
        cx = np.zeros_like(x); cx[top:top + h, left:left + w] = xs
        return cx, cm
    canvas = natural_bg(x).astype(np.float32); canvas[top:top + h, left:left + w] = xs.astype(np.float32)   # fully opaque picture ...
    out = canvas.copy(); r = max(2, int(0.012 * max(H, W)))                                                   # ... then soften only a thin seam
    a = np.zeros((H, W), np.float32); a[top:top + h, left:left + w] = 1.0
    a = cv2.GaussianBlur(a, (0, 0), sigmaX=r / 2.0)[..., None]
    inner = np.zeros((H, W), np.float32); inner[top + r:top + h - r, left + r:left + w - r] = 1.0
    a = np.maximum(a, inner[..., None])                                                                       # interior stays 100% picture
    out = a * canvas + (1 - a) * natural_bg(x).astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8), cm


def scale_for_area(r, r_lo=0.003, r_hi=0.12, s_min=0.3, rng=None):
    """Scale-matched shrinking: draw a target polyp area share r* log-uniformly in [r_lo, r_hi]; shrink by s = sqrt(r*/r) (never enlarge), at least s_min."""
    r_t = float(np.exp(rng.uniform(np.log(r_lo), np.log(r_hi))))
    return float(np.clip(np.sqrt(r_t / max(r, 1e-6)), s_min, 1.0))


def oracle_crop(img, gt, z):
    """ORACLE zoom for a diagnostic (uses the ground truth to centre the window): crop a window of 1/z of each side around the polyp centre.
    returns crop, gt crop, share of the polyp that was cut off by the window."""
    H, W = gt.shape
    ys, xs = np.nonzero(gt)
    if len(ys) == 0 or z <= 1.0001:
        return img, gt, 0.0
    h, w = int(round(H / z)), int(round(W / z)); cy, cx = int(ys.mean()), int(xs.mean())
    y0, x0 = int(np.clip(cy - h // 2, 0, H - h)), int(np.clip(cx - w // 2, 0, W - w))
    g = gt[y0:y0 + h, x0:x0 + w]
    return img[y0:y0 + h, x0:x0 + w], g, float(1 - g.sum() / max(gt.sum(), 1))


# ------------------------------------------------------------------ mask clean-up (all work on bool masks / probability maps)
def components(mask):
    return ndimage.label(mask, structure=CC8)


def keep_largest(mask):
    lab, n = components(mask)
    if n <= 1:
        return mask
    return lab == 1 + int(np.argmax(ndimage.sum(mask, lab, range(1, n + 1))))


def min_area(mask, frac=2e-4):
    lab, n = components(mask)
    if n == 0:
        return mask
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    return np.isin(lab, [i + 1 for i, s in enumerate(sizes) if s >= frac * mask.size])


def peak_gate(prob, thr, seed=0.65, gamma=0.65, min_frac=2e-4):
    """Components of prob>thr are kept only if their peak is high (>= seed) and not far below the strongest peak of the picture (>= gamma * max)."""
    lab, n = components(prob > thr)
    if n == 0:
        return prob > thr
    pk = ndimage.maximum(prob, lab, range(1, n + 1)); sz = ndimage.sum(prob > thr, lab, range(1, n + 1)); gm = float(prob.max())
    return np.isin(lab, [i + 1 for i, (p, s) in enumerate(zip(pk, sz)) if p >= seed and p >= gamma * gm and s >= min_frac * prob.size])


def hysteresis(prob, lo=0.35, seed=0.55, gamma=0.65, min_frac=2e-4):
    """Grow regions down to prob>=lo, keep a region only if its peak >= seed and >= gamma * strongest peak (changes the outline as well)."""
    lab, n = components(prob >= lo)
    if n == 0:
        return prob >= lo
    pk = ndimage.maximum(prob, lab, range(1, n + 1)); sz = ndimage.sum(prob >= lo, lab, range(1, n + 1)); gm = float(prob.max())
    return np.isin(lab, [i + 1 for i, (p, s) in enumerate(zip(pk, sz)) if p >= seed and p >= gamma * gm and s >= min_frac * prob.size])


def consensus(mask, view_masks, eta=0.25, tau=0.35):
    """Keep a connected component only if at least a share tau of the views (5 folds x 4 flips = 20) cover more than a share eta of it.
    mask: HxW bool at full size. view_masks: [V,h,w] bool on the model grid."""
    lab, n = components(mask)
    if n == 0:
        return mask
    small = cv2.resize(lab.astype(np.int32), (view_masks.shape[2], view_masks.shape[1]), interpolation=cv2.INTER_NEAREST)
    keep = []
    for k in range(1, n + 1):
        c = small == k; a = int(c.sum())
        if a == 0:
            keep.append(k); continue                           # component too small for the model grid: cannot be judged, keep
        cover = (view_masks & c[None]).sum((1, 2)) / a
        if (cover > eta).mean() >= tau:
            keep.append(k)
    return np.isin(lab, keep)


# ------------------------------------------------------------------ zoom cascade pieces
def top_peaks(p, k=2, min_p=0.05, nms=31):
    """up to k local maxima of a probability map (no thresholding); if nothing reaches min_p, the global maximum."""
    pk = (p == ndimage.maximum_filter(p, size=nms)) & (p >= min_p)
    ys, xs = np.nonzero(pk)
    if len(ys) == 0:
        return [tuple(int(v) for v in np.unravel_index(np.argmax(p), p.shape))]
    o = np.argsort(-p[ys, xs])[:k]
    return [(int(ys[i]), int(xs[i])) for i in o]


def box_around(cy, cx, grid_hw, full_hw, frac=0.4):
    """window of frac*H x frac*W (full-size pixels) centred on a point given on the model grid; clamped inside the picture."""
    H, W = full_hw; gh, gw = grid_hw; h, w = int(round(H * frac)), int(round(W * frac))
    y0 = int(np.clip(round(cy / gh * H) - h // 2, 0, H - h)); x0 = int(np.clip(round(cx / gw * W) - w // 2, 0, W - w))
    return y0, x0, h, w


def feather(h, w, ramp=0.15):
    r = max(2, int(ramp * min(h, w)))
    yy = np.minimum(np.arange(h), np.arange(h)[::-1])[:, None]; xx = np.minimum(np.arange(w), np.arange(w)[::-1])[None, :]
    return np.clip(np.minimum(yy, xx) / r, 0, 1).astype(np.float32)


def fuse_zoom(p0, q, box, alpha=0.8):
    """noisy-OR of the first-pass map and the zoomed-in map (placed back into its window, feathered at the edge)."""
    y0, x0, h, w = box; out = p0.copy()
    out[y0:y0 + h, x0:x0 + w] = 1 - (1 - p0[y0:y0 + h, x0:x0 + w]) * (1 - alpha * q * feather(h, w))
    return out


# ------------------------------------------------------------------ self-test
def _selftest():
    rng = np.random.default_rng(0)
    x = (rng.random((120, 160, 3)) * 255).astype(np.uint8); x[:, :, 0] = 200                # reddish noise
    m = np.zeros((120, 160), np.uint8); m[40:70, 60:100] = 1
    for bg in ("natural", "black"):
        cx, cm = paste_shrunk(x, m, 0.5, rng, bg)
        assert cx.shape == x.shape and cm.shape == m.shape
        assert abs(cm.sum() / m.sum() - 0.25) < 0.08, cm.sum() / m.sum()                    # area x s^2
    cx, cm = paste_shrunk(x, m, 0.5, np.random.default_rng(1), "natural"); cb, _ = paste_shrunk(x, m, 0.5, np.random.default_rng(1), "black")
    assert cx.mean() > cb.mean() * 1.3                                                      # natural background is not black
    assert np.array_equal(paste_shrunk(x, m, 1.0, rng)[0], x)
    for _ in range(50):
        s = scale_for_area(0.15, rng=rng); assert 0.3 <= s <= 1.0
    assert scale_for_area(0.001, rng=rng) == 1.0                                            # tiny polyps are never shrunk further
    c, g, cut = oracle_crop(x, m.astype(bool), 2.0); assert c.shape[:2] == (60, 80) and g.shape == (60, 80) and cut == 0.0
    # components
    k = np.zeros((100, 100), bool); k[10:30, 10:30] = True; k[70:73, 70:73] = True
    assert keep_largest(k).sum() == 400 and min_area(k, 0.002).sum() == 400 and min_area(k, 0.0001).sum() == 409
    p = np.zeros((100, 100), np.float32); p[10:30, 10:30] = 0.95; p[70:75, 70:75] = 0.6
    assert peak_gate(p, 0.5).sum() == 400 and (peak_gate(p, 0.5, seed=0.5, gamma=0.5).sum() == 425)
    assert hysteresis(p).sum() == 400
    # consensus: big blob supported by all 20 views, small blob by 2
    full = (p > 0.5); views = np.zeros((20, 50, 50), bool); views[:, 5:15, 5:15] = True; views[:2, 35:38, 35:38] = True
    assert consensus(full, views).sum() == 400
    # zoom pieces
    q = np.zeros((100, 100), np.float32); q[50, 50] = 0.9
    assert top_peaks(q, 2) == [(50, 50)] and top_peaks(np.zeros((9, 9)), 2) == [(0, 0)]
    b = box_around(50, 50, (100, 100), (200, 300), 0.4); assert b == (60, 90, 80, 120), b
    f = fuse_zoom(np.zeros((200, 300), np.float32), np.ones((80, 120), np.float32), b); assert f.max() <= 0.8 and f[100, 150] > 0.7 and f[0, 0] == 0
    print("postproc.py self-test OK")


if __name__ == "__main__":
    _selftest()
