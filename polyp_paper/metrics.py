"""Single source of truth for every number in the paper.

Per-image metrics on BINARY masks, then aggregated over images (Metrics Reloaded:
compute per image first, aggregate after). Never reimplement these in a notebook;
import this file so Dice/IoU mean the same thing everywhere.

Run `python metrics.py` for the self-check.
"""
import numpy as np
from scipy import ndimage, stats


def binarize_gt(mask_gray):
    """Kvasir-SEG masks are JPEGs (intermediate gray values): threshold, never divide by 255."""
    return np.asarray(mask_gray) > 127


def prf(pred, gt):
    """Per-image TP/FP/FN based metrics. pred, gt: bool arrays, same shape."""
    pred, gt = np.asarray(pred, bool), np.asarray(gt, bool)
    assert pred.shape == gt.shape, (pred.shape, gt.shape)
    tp = float((pred & gt).sum())
    fp = float((pred & ~gt).sum())
    fn = float((~pred & gt).sum())
    nan = float("nan")
    # note: empty-GT handling not needed (every Kvasir/CVC image has a polyp); NaN, never silently 0/1
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else nan
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else nan
    prec = tp / (tp + fp) if (tp + fp) else 0.0  # empty prediction -> precision 0 (conservative)
    rec = tp / (tp + fn) if (tp + fn) else nan
    f2 = 5 * prec * rec / (4 * prec + rec) if (4 * prec + rec) else 0.0
    return dict(dice=dice, iou=iou, precision=prec, recall=rec, f2=f2,
                pred_area=float(pred.sum()), gt_area=float(gt.sum()))


def _surface(m):
    return m & ~ndimage.binary_erosion(m)


def hd95(pred, gt):
    """Symmetric 95th-percentile Hausdorff distance in pixels (original resolution). NaN if a mask is empty."""
    pred, gt = np.asarray(pred, bool), np.asarray(gt, bool)
    if not pred.any() or not gt.any():
        return float("nan")
    sp, sg = _surface(pred), _surface(gt)
    d_to_g = ndimage.distance_transform_edt(~sg)[sp]
    d_to_p = ndimage.distance_transform_edt(~sp)[sg]
    return float(np.percentile(np.concatenate([d_to_g, d_to_p]), 95))


def summarize(values):
    v = np.asarray(values, float)
    return dict(mean=float(np.nanmean(v)), std=float(np.nanstd(v, ddof=1)),
                median=float(np.nanmedian(v)), n=int(np.isfinite(v).sum()))


def bootstrap_ci(values, n_boot=10000, alpha=0.05, seed=0):
    """Percentile bootstrap CI of the mean over images.
    resamples IMAGES, so it ignores within-patient/sequence correlation (no IDs available):
    treat as optimistic and say so in the paper."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    means = v[idx].mean(axis=1)
    return float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))


def paired_compare(a, b, n_boot=10000, seed=0):
    """Same images, two methods: mean difference, paired bootstrap CI, Wilcoxon signed-rank p."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    d = (a - b)[ok]
    lo, hi = bootstrap_ci(d, n_boot=n_boot, seed=seed)
    p = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    return dict(mean_diff=float(d.mean()), ci95=(lo, hi), wilcoxon_p=float(p), n=int(ok.sum()))


def _selfcheck():
    rng = np.random.default_rng(0)
    # 1) hand-checkable case: 4 px gt, 4 px pred, 2 overlap -> Dice 0.5, IoU 1/3
    gt = np.zeros((4, 4), bool); gt[0, :] = True
    pr = np.zeros((4, 4), bool); pr[0, :2] = True; pr[1, :2] = True
    m = prf(pr, gt)
    assert abs(m["dice"] - 0.5) < 1e-12 and abs(m["iou"] - 1 / 3) < 1e-12, m
    # 2) per-image identity IoU = D/(2-D) on random masks
    for _ in range(200):
        g = rng.random((32, 32)) > 0.6
        p = rng.random((32, 32)) > 0.6
        if g.any() or p.any():
            m = prf(p, g)
            assert abs(m["iou"] - m["dice"] / (2 - m["dice"])) < 1e-12
    # 3) mean IoU >= D/(2-D) of mean Dice (Jensen): the consistency check used on the reported pairs
    ds, js = [], []
    for _ in range(300):
        g = rng.random((32, 32)) > 0.7
        p = g ^ (rng.random((32, 32)) > rng.uniform(0.7, 0.99))
        m = prf(p, g)
        if np.isfinite(m["dice"]):
            ds.append(m["dice"]); js.append(m["iou"])
    assert np.mean(js) >= np.mean(ds) / (2 - np.mean(ds)) - 1e-12
    # 4) HD95: identical masks -> 0; a square shifted by 5 px -> 5
    sq = np.zeros((64, 64), bool); sq[20:40, 20:40] = True
    sh = np.zeros_like(sq); sh[20:40, 25:45] = True
    assert hd95(sq, sq) == 0.0
    assert abs(hd95(sq, sh) - 5.0) < 1.0, hd95(sq, sh)
    # 5) bootstrap CI brackets the mean; paired test detects a real shift and not a null one
    x = rng.normal(0.85, 0.1, 300)
    lo, hi = bootstrap_ci(x)
    assert lo < x.mean() < hi
    assert paired_compare(x + 0.02 + rng.normal(0, 0.005, 300), x)["wilcoxon_p"] < 1e-6
    assert paired_compare(x + rng.normal(0, 0.005, 300), x)["wilcoxon_p"] > 0.001
    # 6) JPEG-style gray mask is binarized, not rescaled
    assert binarize_gt(np.array([[0, 3, 126, 128, 255]])).tolist() == [[False, False, False, True, True]]
    print("metrics.py self-check OK")


if __name__ == "__main__":
    _selfcheck()
