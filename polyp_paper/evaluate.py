"""Ensemble inference + per-image metrics on the protocol's TEST sets (original resolution).
python evaluate.py --data_root data --protocol A --variant roy_full --seed 0 --runs runs --out eval [--thr auto|0.5] [--overlay_eval]

The decision threshold is chosen ONLY from the out-of-fold predictions of the training folds (never from a test set).
Output: <out>/<protocol>/<variant>/seed<s>/per_image.csv, summary.json"""
import argparse, csv, glob, json, os, time
import cv2, numpy as np, torch
from splits import get_protocol
from model import PolypNet
from data import SIZE, to_tensor, ui_overlay
from metrics import prf, hd95, binarize_gt


def load_models(run_dirs, dev):
    ms = []
    for d in run_dirs:
        ck = torch.load(os.path.join(d, "best.pt"), map_location=dev)
        m = PolypNet(pretrained=False, **ck["model_kwargs"]); m.load_state_dict(ck["state"]); m.input_size = ck.get("size", SIZE); ms.append(m.to(dev).eval())
    return ms


@torch.no_grad()
def predict_prob(models, img_rgb_u8, dev, tta=True, scales=(1.0,)):
    """ensemble (and flip-TTA, optional multi-scale TTA) probability map on the grid of the first model, plus mean predicted size code (or nan).
    scales: multipliers of each model's own input size (1.0 = the single-scale behaviour of earlier runs)."""
    out_sz = models[0].input_size; acc, th, cnt = 0, [], 0
    for m in models:
        for s_ in scales:
            sz = max(32, int(round(m.input_size * s_ / 32)) * 32)
            x = to_tensor(cv2.resize(img_rgb_u8, (sz, sz), interpolation=cv2.INTER_LINEAR))[None].to(dev)
            views = [(x, ())] + ([(x.flip(-1), (-1,)), (x.flip(-2), (-2,)), (x.flip(-1, -2), (-1, -2))] if tta else [])
            xb = torch.cat([v for v, _ in views])
            with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
                lg, t = m(xb)
            p = torch.sigmoid(lg.float())
            p = torch.stack([pi.flip(dims) if dims else pi for pi, (_, dims) in zip(p, views)]).mean(0)
            if sz != out_sz:
                p = torch.nn.functional.interpolate(p[None], size=(out_sz, out_sz), mode="bilinear", align_corners=False)[0]
            acc = acc + p; cnt += 1
            if t is not None and s_ == scales[0]:
                th.append(float(t[0]))
    return (acc / cnt)[0].cpu().numpy(), (float(np.mean(th)) if th else float("nan"))


def pranet_dice_iou(prob, gt):
    """The metric of the PraNet test toolbox: per-image min-max normalised map, 256 thresholds 1, 1-1/255, ..., 0,
    Dice/IoU at each threshold, averaged over the thresholds (then over images by the caller)."""
    p = ((prob - prob.min()) / (prob.max() - prob.min() + 1e-8)).ravel(); g = gt.ravel().astype(bool)
    ths = 1 - np.arange(256) / 255.0
    sp, sg = np.sort(p), np.sort(p[g]); n_obj = g.sum()
    n_pred = len(sp) - np.searchsorted(sp, ths, side="left"); inter = len(sg) - np.searchsorted(sg, ths, side="left")
    dice = 2.0 * inter / np.maximum(n_obj + n_pred, 1); iou = inter / np.maximum(n_obj + n_pred - inter, 1)
    return float(dice.mean()), float(iou.mean())


def oof_threshold(run_dirs, grid=np.arange(.30, .71, .05)):
    P, G = [], []
    for d in run_dirs:
        z = np.load(os.path.join(d, "oof_probs.npz")); P.append(z["probs"].astype(np.float32)); G.append(z["gts"])
    P, G = np.concatenate(P), np.concatenate(G).astype(bool)
    def score(t):
        b = P > t; inter = (b & G).sum((1, 2)); return float(((2 * inter + 1e-6) / (b.sum((1, 2)) + G.sum((1, 2)) + 1e-6)).mean())
    sc = {round(float(t), 2): score(t) for t in grid}
    return max(sc, key=sc.get), sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data"); ap.add_argument("--protocol", default="A"); ap.add_argument("--variant", default="base")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--runs", default="runs"); ap.add_argument("--out", default="eval")
    ap.add_argument("--thr", default="auto"); ap.add_argument("--no_tta", action="store_true"); ap.add_argument("--overlay_eval", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N images per test set")
    ap.add_argument("--save_pred", action="store_true", help="save the predicted binary masks (original resolution) for figures")
    ap.add_argument("--pranet_metric", action="store_true", help="also report the PraNet-toolbox Dice/IoU (min-max normalised map, mean over 256 thresholds)")
    ap.add_argument("--fallback", type=float, default=0.0, help="if the mask is empty, binarise at this fraction of the map maximum (benchmark-only: every test image holds a polyp); reported as extra columns")
    ap.add_argument("--scales", default="1.0", help="comma list of input-size multipliers for multi-scale TTA, e.g. 1.0,1.25,1.5")
    ap.add_argument("--ens_with", nargs="*", default=[], help="other variants whose fold models join the ensemble (needs --thr number)")
    ap.add_argument("--tag", default="", help="output folder name instead of the variant name")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = os.path.join(a.runs, a.protocol, a.variant, f"seed{a.seed}")
    run_dirs = sorted(d for d in glob.glob(os.path.join(base, "fold*")) if os.path.exists(os.path.join(d, "best.pt")))
    assert run_dirs, f"no trained folds in {base}"
    for v2 in a.ens_with:
        run_dirs += sorted(d for d in glob.glob(os.path.join(a.runs, a.protocol, v2, f"seed{a.seed}", "fold*")) if os.path.exists(os.path.join(d, "best.pt")))
    assert not a.ens_with or a.thr != "auto", "--ens_with needs a fixed --thr (the out-of-fold files of different variants may have different sizes)"
    scales = tuple(float(x) for x in a.scales.split(","))
    thr, sweep = (oof_threshold(run_dirs) if a.thr == "auto" else (float(a.thr), None))
    models = load_models(run_dirs, dev); _, tests = get_protocol(a.data_root, a.protocol)
    out = os.path.join(a.out, a.protocol, a.tag or a.variant, f"seed{a.seed}"); os.makedirs(out, exist_ok=True)
    rows, t0 = [], time.time()
    sets = dict(tests)
    for name, items in sets.items():
        items = items[:a.limit] if a.limit else items
        for k, it in enumerate(items):
            img = cv2.cvtColor(cv2.imread(it["img"]), cv2.COLOR_BGR2RGB); gt = binarize_gt(cv2.imread(it["mask"], cv2.IMREAD_GRAYSCALE))
            variants = [("", img)]
            if a.overlay_eval and name.startswith("Kvasir"):         # overlay-sensitivity probe: same image with synthetic screen items
                variants.append(("+overlay", ui_overlay(img, gt.astype(np.uint8), np.random.default_rng(1000 + k))))
            for suf, im in variants:
                prob, th = predict_prob(models, im, dev, not a.no_tta, scales)
                pf = cv2.resize(prob, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR); pred = pf > thr
                r = prf(pred, gt); r["hd95"] = hd95(pred, gt)
                if a.pranet_metric:
                    r["pn_dice"], r["pn_iou"] = pranet_dice_iou(pf, gt)
                if a.fallback > 0:
                    pb = pred if pred.any() else pf >= a.fallback * pf.max()
                    rb = prf(pb, gt); r["fb_dice"] = rb["dice"]; r["fb_hd95"] = hd95(pb, gt); r["was_empty"] = float(not pred.any())
                if a.save_pred and not suf:
                    os.makedirs(os.path.join(out, "preds", name), exist_ok=True)
                    cv2.imwrite(os.path.join(out, "preds", name, it["name"] + ".png"), pred.astype(np.uint8) * 255)
                rows.append(dict(dataset=name + suf, image=it["name"], seed=a.seed, variant=a.variant, protocol=a.protocol, thr=thr,
                                 t_hat=th, t_gt=float(np.clip((np.log10(max(gt.mean(), 1e-3)) + 3) / 3, 0, 1)), **r))
        print(name, len(items), "done", f"{time.time() - t0:.0f}s", flush=True)
    keys = list(rows[0].keys())
    with open(os.path.join(out, "per_image.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(rows)
    summ = {}
    for d in sorted({r["dataset"] for r in rows}):
        rr = [r for r in rows if r["dataset"] == d]
        summ[d] = {m: float(np.nanmean([r[m] for r in rr])) for m in ("dice", "iou", "precision", "recall", "f2", "hd95", "pn_dice", "pn_iou", "fb_dice", "fb_hd95", "was_empty") if m in rr[0]} | {"n": len(rr)}
    json.dump(dict(thr=thr, oof_sweep=sweep, n_models=len(models), tta=not a.no_tta, summary=summ, seconds=time.time() - t0), open(os.path.join(out, "summary.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
