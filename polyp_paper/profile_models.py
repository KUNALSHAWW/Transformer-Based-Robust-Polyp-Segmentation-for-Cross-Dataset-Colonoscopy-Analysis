"""Parameters, FLOPs and speed of every model version at its own input size.
python profile_models.py [--variants base base_b1 ...] [--out profile.csv]
FLOPs are counted for one forward pass of one picture (multiply-adds counted as 2 FLOPs by torch's counter, reported in GFLOPs);
speed = median forward time of one picture (batch 1) after warm-up, in the precision used for inference (fp16 on a GPU)."""
import argparse, csv, time
import torch
from torch.utils.flop_counter import FlopCounterMode
from model import PolypNet, VARIANTS

ap = argparse.ArgumentParser(); ap.add_argument("--variants", nargs="*", default=list(VARIANTS)); ap.add_argument("--out", default="profile.csv"); ap.add_argument("--runs", type=int, default=20)
a = ap.parse_args(); dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rows = []
for v in a.variants:
    mk, fl = VARIANTS[v]; sz = fl.get("size", 384)
    m = PolypNet(pretrained=False, **mk).to(dev).eval(); x = torch.randn(1, 3, sz, sz, device=dev)
    params = sum(p.numel() for p in m.parameters()) / 1e6
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        m(x)
    gflops = fc.get_total_flops() / 1e9
    with torch.no_grad(), torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
        for _ in range(3): m(x)
        if dev.type == "cuda": torch.cuda.synchronize()
        ts = []
        for _ in range(a.runs):
            t0 = time.perf_counter(); m(x)
            if dev.type == "cuda": torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
    ms = sorted(ts)[len(ts) // 2] * 1000
    rows.append(dict(variant=v, input=sz, params_M=round(params, 2), gflops=round(gflops, 1), ms_per_image=round(ms, 1), fps=round(1000 / ms, 1), device=dev.type))
    print(rows[-1], flush=True)
with open(a.out, "w", newline="") as f:
    w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)
print("wrote", a.out)
