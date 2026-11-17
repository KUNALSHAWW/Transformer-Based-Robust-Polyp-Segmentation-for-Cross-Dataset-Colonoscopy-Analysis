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
        m = PolypNet(pretrained=False, **ck["model_kwargs"]); m.load_state_dict(ck["state"]); ms.append(m.to(dev).eval())
    return ms


@torch.no_grad()
def predict_prob(models, img_rgb_u8, dev, tta=True):
    """ensemble (and flip-TTA) probability map at SIZE x SIZE, plus mean predicted size code (or nan)."""
    x = to_tensor(cv2.resize(img_rgb_u8, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR))[None].to(dev)
    views = [(x, ())] + ([(x.flip(-1), (-1,)), (x.flip(-2), (-2,)), (x.flip(-1, -2), (-1, -2))] if tta else [])
    xb = torch.cat([v for v, _ in views]); acc, th = 0, []
    for m in models:
        with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
            lg, t = m(xb)
        p = torch.sigmoid(lg.float())
        acc = acc + torch.stack([pi.flip(dims) if dims else pi for pi, (_, dims) in zip(p, views)]).mean(0)
        if t is not None:
            th.append(float(t[0]))
    return (acc / len(models))[0].cpu().numpy(), (float(np.mean(th)) if th else float("nan"))


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
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = os.path.join(a.runs, a.protocol, a.variant, f"seed{a.seed}")
    run_dirs = sorted(d for d in glob.glob(os.path.join(base, "fold*")) if os.path.exists(os.path.join(d, "best.pt")))
    assert run_dirs, f"no trained folds in {base}"
    thr, sweep = (oof_threshold(run_dirs) if a.thr == "auto" else (float(a.thr), None))
    models = load_models(run_dirs, dev); _, tests = get_protocol(a.data_root, a.protocol)
    out = os.path.join(a.out, a.protocol, a.variant, f"seed{a.seed}"); os.makedirs(out, exist_ok=True)
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
                prob, th = predict_prob(models, im, dev, not a.no_tta)
                pred = cv2.resize(prob, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR) > thr
                r = prf(pred, gt); r["hd95"] = hd95(pred, gt)
                rows.append(dict(dataset=name + suf, image=it["name"], seed=a.seed, variant=a.variant, protocol=a.protocol, thr=thr,
                                 t_hat=th, t_gt=float(np.clip((np.log10(max(gt.mean(), 1e-3)) + 3) / 3, 0, 1)), **r))
        print(name, len(items), "done", f"{time.time() - t0:.0f}s", flush=True)
    keys = list(rows[0].keys())
    with open(os.path.join(out, "per_image.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(rows)
    summ = {}
    for d in sorted({r["dataset"] for r in rows}):
        rr = [r for r in rows if r["dataset"] == d]
        summ[d] = {m: float(np.nanmean([r[m] for r in rr])) for m in ("dice", "iou", "precision", "recall", "f2", "hd95")} | {"n": len(rr)}
    json.dump(dict(thr=thr, oof_sweep=sweep, n_models=len(models), tta=not a.no_tta, summary=summ, seconds=time.time() - t0), open(os.path.join(out, "summary.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
