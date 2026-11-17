"""CPU self-check. python selfcheck.py --data_root data     (takes about a minute; run it before any long training)"""
import argparse, copy, time
import numpy as np, torch
from model import PolypNet, VARIANTS
from losses import SegLoss, beta_from_t
from data import PolypDS, area_to_t, ui_overlay
from splits import get_protocol, make_folds

ap = argparse.ArgumentParser(); ap.add_argument("--data_root", default="data"); a = ap.parse_args()
torch.manual_seed(0)

# 1) size target and Tversky beta mapping
assert area_to_t(0.001) == 0.0 and area_to_t(1.0) == 1.0 and abs(area_to_t(0.01) - 1 / 3) < 1e-6
b = beta_from_t(torch.tensor([0.0, (np.log10(0.15) + 3) / 3, 1.0]))
assert abs(float(b[0]) - 0.8) < 0.01 and abs(float(b[1]) - 0.5) < 1e-4 and abs(float(b[2]) - 0.5) < 1e-4, b
print("size mapping OK")

# 2) data: shapes, binary masks, overlay keeps the mask and only edits pixels
pool, tests = get_protocol(a.data_root, "A")
ds = PolypDS(pool[:6], train=True, overlay="pair", seed=1)
s = ds[0]
assert s["x"].shape == (3, 384, 384) and s["mask"].shape == (1, 384, 384) and set(s["mask"].unique().tolist()) <= {0.0, 1.0}
assert not torch.equal(s["x"], s["x_ovl"]), "overlay did nothing"
import cv2
rng = np.random.default_rng(3); changed = 0
for it in pool[:20]:
    im = cv2.cvtColor(cv2.imread(it["img"]), cv2.COLOR_BGR2RGB); im = cv2.resize(im, (384, 384))
    m = (cv2.resize(cv2.imread(it["mask"], 0), (384, 384), interpolation=cv2.INTER_NEAREST) > 127).astype(np.uint8)
    o = ui_overlay(im, m, rng); changed += int((o != im).any())
    diff = (o != im).any(2); assert (diff & (m > 0)).sum() <= 0.05 * max(m.sum(), 1) + 1, "overlay covers the polyp"
assert changed >= 12, changed
print("data + overlay OK (overlay changed", changed, "of 20; never covers >5% of the polyp)")

# 2b) validation Dice must react to the prediction (a double sigmoid once made it constant)
from train import dice_per_image
pr = torch.zeros(2, 1, 8, 8); gt = torch.zeros(2, 1, 8, 8); gt[:, 0, :4] = 1; pr[0, 0, :4] = 0.9
d = dice_per_image(pr, gt)
assert abs(float(d[0]) - 1) < 1e-4 and float(d[1]) < 1e-4, d                      # perfect image -> 1, empty prediction -> 0
assert float(dice_per_image(torch.ones(2, 1, 8, 8), gt).mean()) < 0.7              # "everything is polyp" is NOT a good score
assert float(dice_per_image(torch.full((2, 1, 8, 8), 0.4), gt).mean()) < 1e-4      # probability below threshold -> nothing predicted
print("validation Dice OK")

# 3) every model variant: shapes, finite loss, gradients reach encoder and new heads
x = torch.randn(2, 3, 384, 384); y = (torch.rand(2, 1, 384, 384) > .8).float()
for name, (mk, fl) in VARIANTS.items():
    t0 = time.time(); m = PolypNet(pretrained=False, **mk); m.train()
    lg, th = m(x)
    assert lg.shape == (2, 1, 384, 384), lg.shape
    assert (th is None) == (not mk.get("size_head")), name
    crit = SegLoss(size_tversky=fl.get("size_tversky", False)); loss = crit(lg, y, th)
    if th is not None:
        loss = loss + torch.nn.functional.smooth_l1_loss(th, torch.rand(2, 1))
    assert torch.isfinite(loss), name
    loss.backward()
    enc_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for n_, p in m.named_parameters() if "encoder" in n_)
    assert enc_grad, name + ": no gradient reaches the encoder"
    npar = sum(p.numel() for p in m.parameters()) / 1e6
    print(f"{name:20s} params {npar:6.2f}M  loss {float(loss):.3f}  {time.time() - t0:.0f}s")

# 4) size gate starts as identity: logits equal those of the same weights with the gate removed
m = PolypNet(pretrained=False, decoder="roy", size_head=True, size_gate=True).eval()
with torch.no_grad():
    l1, _ = m(x[:1]); g = m.gate; m.gate = None; l2, _ = m(x[:1]); m.gate = g
assert torch.allclose(l1, l2, atol=1e-5), "gate is not identity at init"
print("size gate identity at init OK")

# 5) splits: counts per protocol
for p, want in (("A", (880, 5)), ("B", (1450, 5)), ("Bclean", (1458, 5))):
    pool, tests = get_protocol(a.data_root, p); folds = make_folds(pool, p)
    assert (len(pool), len(folds)) == want and all(len(set(t) & set(v)) == 0 for t, v in folds), p
print("splits OK")
print("SELF-CHECK PASSED")
