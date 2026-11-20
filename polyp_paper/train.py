"""Train ONE inner fold of one variant.
python train.py --data_root data --protocol A --variant roy_full --fold 0 --seed 0 --out runs

Model selection uses ONLY the inner validation fold (per-image Dice of the EMA model, computed at 384 px). No test set is read here.
Writes: best.pt, history.json, oof_probs.npz (inner-val probabilities for threshold selection), manifest.json."""
import argparse, copy, hashlib, json, os, random, subprocess, sys, time
import numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
import albumentations, segmentation_models_pytorch as smp
from splits import get_protocol, make_folds
from data import PolypDS, worker_init
from model import PolypNet, VARIANTS, param_groups
from losses import SegLoss


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False


class EMA:
    def __init__(self, model, decay=.995):
        self.m, self.d = copy.deepcopy(model).eval(), decay
        for p in self.m.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        for k, v in self.m.state_dict().items():
            v.mul_(self.d).add_(msd[k].detach(), alpha=1 - self.d) if v.dtype.is_floating_point else v.copy_(msd[k])


def dice_per_image(prob, y, thr=.5):
    """per-image Dice of thresholded PROBABILITIES (not logits)."""
    p = (prob > thr).float().flatten(1); t = y.flatten(1)
    return (2 * (p * t).sum(1) + 1e-6) / (p.sum(1) + t.sum(1) + 1e-6)


@torch.no_grad()
def predict(model, loader, dev, amp):
    model.eval(); P, Y, T, TH = [], [], [], []
    for b in loader:
        with torch.autocast(device_type=dev.type, enabled=amp):
            lg, th = model(b["x"].to(dev))
        P.append(torch.sigmoid(lg.float()).cpu()); Y.append(b["mask"]); T.append(b["t"])
        TH.append(th.float().cpu() if th is not None else torch.full_like(b["t"], float("nan")))
    return torch.cat(P), torch.cat(Y), torch.cat(T), torch.cat(TH)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data"); ap.add_argument("--protocol", default="A", choices=["A", "B", "Bclean"])
    ap.add_argument("--variant", default="base", choices=list(VARIANTS)); ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default="runs")
    ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--bs", type=int, default=8); ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--patience", type=int, default=12); ap.add_argument("--min_epochs", type=int, default=10, help="early stopping cannot fire before this epoch"); ap.add_argument("--lr_enc", type=float, default=2e-5); ap.add_argument("--lr_dec", type=float, default=2e-4)
    ap.add_argument("--w_size", type=float, default=0.2); ap.add_argument("--w_cons", type=float, default=1.0); ap.add_argument("--w_kd", type=float, default=1.0)
    ap.add_argument("--no_pretrained", action="store_true"); ap.add_argument("--dry", type=int, default=0, help="tiny CPU smoke test: use N images")
    a = ap.parse_args()
    set_seed(a.seed); t0 = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); amp = dev.type == "cuda"
    mk, fl = VARIANTS[a.variant]
    SZ = fl.get("size", 384)                                  # input size of this variant
    pool, _ = get_protocol(a.data_root, a.protocol)
    tr_i, va_i = make_folds(pool, a.protocol)[a.fold]          # fold assignment is fixed (seed 42), independent of the training seed
    if a.dry:
        tr_i, va_i = tr_i[:a.dry], va_i[:max(2, a.dry // 2)]
    tr, va = [pool[i] for i in tr_i], [pool[i] for i in va_i]
    run = os.path.join(a.out, a.protocol, a.variant, f"seed{a.seed}", f"fold{a.fold}"); os.makedirs(run, exist_ok=True)
    g = torch.Generator(); g.manual_seed(a.seed)
    tl = DataLoader(PolypDS(tr, True, fl.get("overlay", "none"), seed=a.seed, size=SZ, lab=fl.get("lab", False), zoom=fl.get("zoom", False)), batch_size=a.bs, shuffle=True, drop_last=len(tr) > a.bs, num_workers=a.workers,
                    worker_init_fn=worker_init, generator=g, pin_memory=dev.type == "cuda", persistent_workers=a.workers > 0)
    vl = DataLoader(PolypDS(va, size=SZ), batch_size=16, num_workers=a.workers)
    model = PolypNet(pretrained=not (a.no_pretrained or a.dry), **mk).to(dev)
    opt = torch.optim.AdamW(param_groups(model, a.lr_enc, a.lr_dec), weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs, eta_min=1e-7)
    scaler = torch.amp.GradScaler(enabled=amp); ema = EMA(model); crit = SegLoss(size_tversky=fl.get("size_tversky", False))
    teacher = None
    if fl.get("kd"):                       # distillation: teacher = the SAME inner fold of another variant (it never saw this fold's validation images)
        tdir = os.path.join(a.out, a.protocol, fl["kd"], f"seed{a.seed}", f"fold{a.fold}"); waited = 0
        while not os.path.exists(os.path.join(tdir, "manifest.json")):          # wait until the teacher fold has finished
            if waited > 5400: raise SystemExit("teacher fold not finished: " + tdir)
            time.sleep(30); waited += 30
        ck_t = torch.load(os.path.join(tdir, "best.pt"), map_location=dev); assert ck_t.get("size", 384) == SZ, "teacher and student need the same input size"
        teacher = PolypNet(pretrained=False, **ck_t["model_kwargs"]).to(dev).eval(); teacher.load_state_dict(ck_t["state"])
        for p_ in teacher.parameters(): p_.requires_grad_(False)
    best, bad, hist = -1.0, 0, []
    for ep in range(1, a.epochs + 1):
        model.train(); tot = 0.0; te = time.time()
        for b in tl:
            x, y, t = b["x"].to(dev), b["mask"].to(dev), b["t"].to(dev)
            xo = b["x_ovl"].to(dev) if "x_ovl" in b else None
            if fl.get("ms"):                                     # multi-scale training: one random scale per batch
                sz = int(round(SZ * random.choice([0.75, 1.0, 1.25]) / 32) * 32)
                if sz != x.shape[-1]:
                    x = F.interpolate(x, size=(sz, sz), mode="bilinear", align_corners=False); y = F.interpolate(y, size=(sz, sz), mode="nearest")
                    xo = None if xo is None else F.interpolate(xo, size=(sz, sz), mode="bilinear", align_corners=False)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, enabled=amp):
                lg, th = model(x)
                loss = crit(lg.float(), y, None if th is None else th.float())
                if fl.get("size_loss") and th is not None:
                    loss = loss + a.w_size * F.smooth_l1_loss(th.float(), t)
                if teacher is not None:                  # student probabilities should match the teacher's soft probabilities
                    with torch.no_grad():
                        tl_, _ = teacher(x)
                    loss = loss + a.w_kd * F.binary_cross_entropy_with_logits(lg.float(), torch.sigmoid(tl_.float()))
                if xo is not None:          # Idea 1: overlaid copy must give the same answer (and is also supervised, weight 0.5)
                    lo, tho = model(xo)
                    loss = loss + .5 * crit(lo.float(), y, None if tho is None else tho.float()) + a.w_cons * F.mse_loss(torch.sigmoid(lo.float()), torch.sigmoid(lg.float()).detach())
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); ema.update(model); tot += loss.item()
        sched.step()
        P, Y, T, TH = predict(ema.m, vl, dev, amp)
        d = float(dice_per_image(P, Y).mean()); sz = float((TH - T).abs().mean()) if not torch.isnan(TH).all() else None
        hist.append(dict(epoch=ep, train_loss=tot / len(tl), val_dice=d, size_mae_t=sz, sec=time.time() - te))
        print(f"[{a.protocol}/{a.variant}/s{a.seed}/f{a.fold}] ep {ep} loss {tot / len(tl):.4f} val_dice {d:.4f} size_mae {sz} {time.time() - te:.0f}s", flush=True)
        if d > best + 5e-4:
            best, bad, best_ep = d, 0, ep
            torch.save({"state": ema.m.state_dict(), "variant": a.variant, "model_kwargs": mk, "size": SZ, "epoch": ep, "val_dice": d}, os.path.join(run, "best.pt"))
        else:
            bad += 1
            if bad >= a.patience and ep >= a.min_epochs:
                break
    ck = torch.load(os.path.join(run, "best.pt"), map_location=dev); ema.m.load_state_dict(ck["state"])
    P, Y, T, TH = predict(ema.m, vl, dev, amp)
    np.savez_compressed(os.path.join(run, "oof_probs.npz"), probs=P.half().numpy()[:, 0], gts=Y.numpy()[:, 0].astype(np.uint8), t=T.numpy()[:, 0], t_hat=TH.numpy()[:, 0],
                        names=np.array([v["name"] + "|" + v["dataset"] for v in va]))
    json.dump(hist, open(os.path.join(run, "history.json"), "w"))
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip() or None
    except Exception:
        sha = None
    json.dump(dict(args=vars(a), model_kwargs=mk, flags=fl, best_epoch=best_ep, best_val_dice=best, epochs_run=len(hist), seconds=time.time() - t0, git=sha,
                   val_names_sha=hashlib.sha256("\n".join(sorted(v["name"] for v in va)).encode()).hexdigest(), n_train=len(tr), n_val=len(va),
                   versions=dict(torch=torch.__version__, smp=smp.__version__, albumentations=albumentations.__version__, python=sys.version.split()[0]),
                   device=torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu"), open(os.path.join(run, "manifest.json"), "w"), indent=1)
    print("done", run, "best val dice", round(best, 4), "epoch", best_ep)


if __name__ == "__main__":
    main()
