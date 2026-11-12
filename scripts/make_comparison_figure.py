"""Regenerates figures/cross_dataset_comparison.png from the numbers reported in the README.

Left panel : same-protocol comparison (all models trained on Kvasir-SEG only, tested on all 612 CVC-ClinicDB frames;
             numbers from notebooks/02_baseline_comparison_and_external_eval.ipynb).
Right panel: published results (PraNet / SANet / HarDNet-MSEG / Polyp-PVT) -- those models were trained on a combined
             Kvasir-SEG + CVC-ClinicDB training set and tested on held-out ClinicDB frames. DIFFERENT protocol,
             shown for context only. Sources: arXiv 2006.11392, 2108.00882, 2101.07172, 2108.06932.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

same = {  # name: (Dice, IoU)
    "UNet (ResNet-34)": (0.8363, 0.7548),
    "UNet++ (ResNet-34)": (0.8390, 0.7584),
    "Attention U-Net": (0.8481, 0.7715),
    "PraNet (re-impl.)": (0.8374, 0.7560),
    "Inf-Net (re-impl.)": (0.8368, 0.7527),
    "UNet + MiT-B3 (ours)": (0.8848, 0.8180),
}
pub = {
    "PraNet": (0.899, 0.849),
    "SANet": (0.916, 0.859),
    "HarDNet-MSEG": (0.932, 0.882),
    "Polyp-PVT": (0.937, 0.889),
    "Ours (zero-shot)": (0.8832, 0.8144),
}

def panel(ax, data, title, highlight):
    names = list(data)
    x = np.arange(len(names)); w = 0.38
    d = [v[0] for v in data.values()]; i = [v[1] for v in data.values()]
    b1 = ax.bar(x - w / 2, d, w, label="Dice", color=["#d62728" if n == highlight else "#4c78a8" for n in names])
    b2 = ax.bar(x + w / 2, i, w, label="IoU", color=["#ff9896" if n == highlight else "#9ecae9" for n in names])
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.004, f"{b.get_height():.3f}", ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    ax.set_ylim(0.6, 1.0); ax.set_title(title, fontsize=10); ax.grid(axis="y", alpha=.3)
    ax.legend(["Dice", "IoU"], loc="upper left", fontsize=8)

fig, axs = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [6, 5]})
panel(axs[0], same, "Same protocol: trained on Kvasir-SEG only, tested on all 612 CVC-ClinicDB frames", "UNet + MiT-B3 (ours)")
panel(axs[1], pub, "Published results: trained WITH ClinicDB images (different protocol, context only)", "Ours (zero-shot)")
fig.tight_layout()
fig.savefig("figures/cross_dataset_comparison.png", dpi=150)
