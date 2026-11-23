"""Extra evaluations of FINISHED models (no training). One script, four experiments (--exp):
  views     per image: single model / + flips / 5-fold ensemble / ensemble + flips, and four ways to clean up the final mask
            (keep-largest, minimum area, peak gate, hysteresis, fold-and-flip consensus)           -> where does the accuracy come from, what does clean-up change
  zoom      soft zoom cascade: where the first-pass map is small or empty, look again at higher magnification around its strongest peaks
  sizeswap  size counterfactuals: (a) shrink the Kvasir test pictures step by step, (b) ORACLE-enlarge ColonDB/ETIS pictures around the polyp
            -> is the drop on small polyps caused by polyp size alone, or by the dataset?
  polypgen  zero-shot test on the PolypGen centres, sequences and polyp-free frames (needs --polypgen_root; research-only data, see polypgen.py)
The threshold is the usual one chosen from out-of-fold predictions of the training folds (or --thr). Every setting of the clean-up and zoom steps is the
FIXED one in CFG below (decided before any test set was looked at for these ideas); change nothing here after seeing results.
python evaluate_extra.py --exp views --data_root data --protocol A --variant base --seed 0 --runs runs --out eval_x [--limit 3]"""
import argparse, csv, glob, json, os, time
import cv2, numpy as np, torch
from splits import get_protocol
from evaluate import load_models, predict_views, oof_threshold
from metrics import prf, hd95, binarize_gt, failure_columns
import postproc as pp

CFG = dict(zoom_frac=0.4, zoom_k=2, zoom_alpha=0.8, zoom_trigger=0.02,          # zoom cascade: window = 40% of each side, 2 peaks, weight 0.8, only if the predicted polyp covers < 2% of the picture
           eta=0.25, tau=0.35,                                                   # consensus: a region stays if >= 35% of the 20 views cover > 25% of it
           seed=0.65, gamma=0.65, min_frac=2e-4,                                 # peak gate / hysteresis: peak >= 0.65 and >= 0.65 x strongest peak, area >= 0.02% of the picture
           hyst_lo=0.35, hyst_seed=0.55,
           shrink=[1.0, 0.7, 0.5, 0.35, 0.25], zoom_factors=[1.0, 1.5, 2.0, 3.0, 4.0], pg_max_side=768)
KEEP = ["dice", "iou", "precision", "recall", "pred_area", "gt_area", "gt_frac", "hd95", "hd95_diag", "empty", "no_overlap", "n_pred_comp"]


def to_native(p, shape):
    return cv2.resize(p, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)


def score(pred, gt, hd=True):
    """per-image numbers. gt None (a polyp-free frame): only what was predicted."""
    pred = np.asarray(pred, bool)
    if gt is None:
        return dict(pred_area=float(pred.sum()), pred_any=float(pred.any()), n_pred_comp=float(pp.components(pred)[1]))
    r = prf(pred, gt); r.update(failure_columns(pred, gt)) if hd else r.update(dict(empty=float(not pred.any()), no_overlap=float(pred.any() and not (pred & gt).any())))
    r["hd95"] = hd95(pred, gt) if hd else float("nan"); r["gt_frac"] = float(gt.mean())
    return {k: r.get(k, float("nan")) for k in KEEP}


def cleanups(pf, V, thr):
    """all mask versions of one picture: pf = final probability at full size, V = per-view maps [M,4,S,S]"""
    raw = pf > thr
    vm = V.reshape(-1, V.shape[2], V.shape[3]) > thr
    return {"raw": raw, "keep_largest": pp.keep_largest(raw), "min_area": pp.min_area(raw, CFG["min_frac"]),
            "peak_gate": pp.peak_gate(pf, thr, CFG["seed"], CFG["gamma"], CFG["min_frac"]),
            "hysteresis": pp.hysteresis(pf, CFG["hyst_lo"], CFG["hyst_seed"], CFG["gamma"], CFG["min_frac"]),
            "consensus": pp.consensus(raw, vm, CFG["eta"], CFG["tau"])}


def zoom_prob(models, img, dev, p_grid, thr):
    """soft zoom cascade. returns (probability map at full size, triggered 0/1)"""
    H, W = img.shape[:2]; p0 = to_native(p_grid, (H, W))
    if float((p0 > thr).mean()) >= CFG["zoom_trigger"]:
        return p0, 0
    p = p0.copy()
    for cy, cx in pp.top_peaks(p_grid, CFG["zoom_k"]):
        box = pp.box_around(cy, cx, p_grid.shape, (H, W), CFG["zoom_frac"]); y0, x0, h, w = box
        q = cv2.resize(predict_views(models, img[y0:y0 + h, x0:x0 + w], dev).mean((0, 1)), (w, h), interpolation=cv2.INTER_LINEAR)
        p = pp.fuse_zoom(p, q, box, CFG["zoom_alpha"])
    return p, 1


def read(it, max_side=0):
    img = cv2.cvtColor(cv2.imread(it["img"]), cv2.COLOR_BGR2RGB)
    gt = None if it.get("mask") is None else binarize_gt(cv2.imread(it["mask"], cv2.IMREAD_GRAYSCALE))
    if max_side and max(img.shape[:2]) > max_side:
        f = max_side / max(img.shape[:2]); sz = (int(round(img.shape[1] * f)), int(round(img.shape[0] * f)))
        img = cv2.resize(img, sz, interpolation=cv2.INTER_AREA)
        if gt is not None:
            gt = cv2.resize(gt.astype(np.uint8), sz, interpolation=cv2.INTER_NEAREST).astype(bool)
    return img, gt


# ------------------------------------------------------------------ experiments (each yields rows)
def exp_views(models, dev, thr, sets, limit, **_):
    for name, items in sets.items():
        for it in items[:limit] if limit else items:
            img, gt = read(it); H, W = gt.shape
            V = predict_views(models, img, dev); S = V.shape[2]
            probs = {"single": V[0, 0], "single_flips": V[0].mean(0), "ensemble": V[:, 0].mean(0), "ensemble_flips": V.mean((0, 1))}
            for c, p in probs.items():
                yield dict(dataset=name, image=it["name"], config=c, **score(to_native(p, (H, W)) > thr, gt))
            pf = to_native(probs["ensemble_flips"], (H, W))
            for c, m in cleanups(pf, V, thr).items():
                if c == "raw": continue                                      # identical to ensemble_flips
                yield dict(dataset=name, image=it["name"], config="ensemble_flips+" + c, **score(m, gt))


def exp_zoom(models, dev, thr, sets, limit, **_):
    for name, items in sets.items():
        for it in items[:limit] if limit else items:
            img, gt = read(it); H, W = gt.shape
            V = predict_views(models, img, dev); pg = V.mean((0, 1))
            pz, trig = zoom_prob(models, img, dev, pg, thr)
            yield dict(dataset=name, image=it["name"], config="raw", triggered=trig, **score(to_native(pg, (H, W)) > thr, gt))
            yield dict(dataset=name, image=it["name"], config="zoom", triggered=trig, **score(pz > thr, gt))


def exp_sizeswap(models, dev, thr, sets, limit, zoom_sets=("ColonDB", "ETIS"), **_):
    for name, items in sets.items():
        mode = "shrink" if name.startswith("Kvasir") else ("zoom" if name in zoom_sets else None)
        if mode is None: continue
        for k, it in enumerate(items[:limit] if limit else items):
            img, gt = read(it)
            for f in (CFG["shrink"] if mode == "shrink" else CFG["zoom_factors"]):
                if mode == "shrink":
                    x, m = pp.paste_shrunk(img, gt.astype(np.uint8), f, np.random.default_rng(1000 + k)); m = m.astype(bool); cut = 0.0
                else:
                    x, m, cut = pp.oracle_crop(img, gt, f)
                p = to_native(predict_views(models, x, dev).mean((0, 1)), m.shape)
                yield dict(dataset=name, image=it["name"], mode=mode, factor=f, rel_area=float(m.mean()), gt_cut=cut, **score(p > thr, m, hd=False))


def exp_polypgen(models, dev, thr, items, limit, **_):
    if limit:
        seen, keep = {}, []
        for it in items:
            seen[it["kind"]] = seen.get(it["kind"], 0) + 1
            if seen[it["kind"]] <= limit: keep.append(it)
        items = keep
    for it in items:
        img, gt = read(it, CFG["pg_max_side"]); H, W = img.shape[:2]
        V = predict_views(models, img, dev); pg = V.mean((0, 1)); pf = to_native(pg, (H, W)); cl = cleanups(pf, V, thr)
        pz, trig = zoom_prob(models, img, dev, pg, thr)
        base = dict(dataset=it["group"], kind=it["kind"], image=it["name"], has_gt=float(gt is not None and gt.any()))
        gt_ = gt if (gt is not None and gt.any()) else None
        for c, m in (("raw", cl["raw"]), ("consensus", cl["consensus"]), ("peak_gate", cl["peak_gate"]), ("zoom", pz > thr), ("zoom+consensus", pp.consensus(pz > thr, V.reshape(-1, V.shape[2], V.shape[3]) > thr, CFG["eta"], CFG["tau"]))):
            yield dict(**base, config=c, triggered=trig, **score(m, gt_, hd=gt_ is not None))


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, choices=["views", "zoom", "sizeswap", "polypgen"])
    ap.add_argument("--data_root", default="data"); ap.add_argument("--protocol", default="A"); ap.add_argument("--variant", default="base"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--runs", default="runs"); ap.add_argument("--out", default="eval_x"); ap.add_argument("--tag", default=""); ap.add_argument("--thr", default="auto")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N images per set"); ap.add_argument("--sets", default="", help="comma list of test-set names (default: all of the protocol)")
    ap.add_argument("--polypgen_root", default=""); ap.add_argument("--zoom_sets", default="ColonDB,ETIS")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dirs = sorted(d for d in glob.glob(os.path.join(a.runs, a.protocol, a.variant, f"seed{a.seed}", "fold*")) if os.path.exists(os.path.join(d, "best.pt")))
    assert run_dirs, f"no trained folds for {a.protocol}/{a.variant}/seed{a.seed} under {a.runs}"
    thr = oof_threshold(run_dirs)[0] if a.thr == "auto" else float(a.thr)
    models = load_models(run_dirs, dev)
    out = os.path.join(a.out, a.exp, a.tag or f"{a.protocol}_{a.variant}_s{a.seed}"); os.makedirs(out, exist_ok=True)
    t0 = time.time(); rows = []
    if a.exp == "polypgen":
        import polypgen
        assert a.polypgen_root, "--polypgen_root needed"; src = polypgen.items(a.polypgen_root); fn = lambda: exp_polypgen(models, dev, thr, src, a.limit)
    else:
        _, tests = get_protocol(a.data_root, a.protocol); sets = {k: v for k, v in tests.items() if not a.sets or k in a.sets.split(",")}
        fn = lambda: {"views": exp_views, "zoom": exp_zoom, "sizeswap": exp_sizeswap}[a.exp](models, dev, thr, sets, a.limit, zoom_sets=tuple(a.zoom_sets.split(",")))
    n = 0
    for r in fn():
        rows.append(dict(protocol=a.protocol, variant=a.variant, seed=a.seed, **r)); n += 1
        if n % 200 == 0: print(f"{a.exp}: {n} rows, {time.time() - t0:.0f}s", flush=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(out, "per_image.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(rows)
    json.dump(dict(thr=thr, n_models=len(models), cfg=CFG, rows=len(rows), seconds=time.time() - t0, exp=a.exp, protocol=a.protocol, variant=a.variant, seed=a.seed), open(os.path.join(out, "meta.json"), "w"), indent=1)
    # short text summary: mean Dice per set and configuration
    import collections
    g = collections.defaultdict(list)
    for r in rows:
        g[(r["dataset"], r.get("config", r.get("mode", "")) + (f" x{r['factor']}" if "factor" in r else ""))].append(r.get("dice", float("nan")))
    for k, v in sorted(g.items()):
        print(f"  {k[0]:12s} {k[1]:26s} n={len(v):5d} Dice={np.nanmean(v) if np.isfinite(v).any() else float('nan'):.4f}")
    print("wrote", os.path.join(out, "per_image.csv"), f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
