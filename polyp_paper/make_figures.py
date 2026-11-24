"""Makes every picture and one text file of key numbers from a result folder (Kaggle output or the unpacked bundle).
python make_figures.py --bundle <folder with eval/ eval_x/ eval_extra/> --out <figure folder> [--data_root <data folder for the qualitative picture>]
Each figure is independent: if its input files are missing it is skipped with a message. Nothing here changes any result.
Output: fig_*.png (look at them) and key_numbers.md (the same facts as numbers, short enough to paste into a chat)."""
import argparse, glob, os, traceback
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from metrics import bootstrap_ci

SETS = ["Kvasir120", "ClinicDB612", "ColonDB", "ETIS", "CVC300"]
SC = {"Kvasir120": "#4c72b0", "ClinicDB612": "#55a868", "ColonDB": "#c44e52", "ETIS": "#dd8452", "CVC300": "#8c8c8c"}
BINS = [(0, .01, "<1%"), (.01, .03, "1-3%"), (.03, .06, "3-6%"), (.06, .12, "6-12%"), (.12, 2, ">=12%")]
KN, DONE, SKIP = [], [], []


def kn(s=""):
    KN.append(s)


def ci(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return (np.nan, np.nan) if len(x) < 3 else bootstrap_ci(x, n_boot=2000)


def figure(f):
    def w(*a, **k):
        try:
            r = f(*a, **k)
            (DONE if r is not False else SKIP).append(f.__name__)
        except Exception:
            SKIP.append(f.__name__); print("FAILED", f.__name__); traceback.print_exc()
    return w


def read_eval(bundle, root="eval"):
    fs = sorted(glob.glob(os.path.join(bundle, root, "*", "*", "seed*", "per_image.csv")))
    if not fs: return None
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    d["area_frac"] = 10 ** (3 * d["t_gt"] - 3); d = d[~d.dataset.str.endswith("+overlay")].copy(); d["set"] = d["dataset"]; return d


def avg_seeds(d):
    num = [c for c in ["dice", "iou", "hd95", "hd95_diag", "empty", "no_overlap", "n_pred_comp", "area_frac", "pred_area"] if c in d.columns]
    return d.groupby(["protocol", "variant", "set", "image"], as_index=False)[num].mean()


def binlab(a):
    return next(n for lo, hi, n in BINS if lo <= a < hi)


def xread(bundle, exp):
    fs = sorted(glob.glob(os.path.join(bundle, "eval_x", exp, "*", "per_image.csv")))
    return None if not fs else pd.concat([pd.read_csv(f).assign(run=os.path.basename(os.path.dirname(f))) for f in fs], ignore_index=True)


# ---------------------------------------------------------------- 1 matched size
@figure
def fig_matched_size(E, out):
    d = avg_seeds(E[(E.protocol == "A") & (E.variant == "base")])
    if d.empty: return False
    d["bin"] = d.area_frac.map(binlab); names = [b[2] for b in BINS]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw=dict(width_ratios=[1.6, 1]))
    seen = set()
    kn("## 1. Dice inside matched polyp-size bins (base, Protocol A; mean [95% CI] (n))"); kn("| set | " + " | ".join(names) + " |"); kn("|---|" + "---|" * len(names))
    for k, s in enumerate(SETS):
        x = d[d.set == s]; row = []
        for i, b in enumerate(names):
            v = x[x.bin == b].dice.values
            if len(v) >= 3:
                m = v.mean(); lo, hi = ci(v); ax[0].errorbar(i + (k - 2) * .07, m, yerr=[[m - lo], [hi - m]], fmt="o", color=SC[s], capsize=2, ms=4, label=s if s not in seen else None); seen.add(s); row.append(f"{m:.3f} [{lo:.2f},{hi:.2f}] ({len(v)})")
            else: row.append(f"- ({len(v)})")
        ax[0].plot([i + (k - 2) * .07 for i in range(len(names)) if len(x[x.bin == names[i]]) >= 3], [x[x.bin == b].dice.mean() for b in names if len(x[x.bin == b]) >= 3], color=SC[s], lw=1, alpha=.6)
        kn(f"| {s} | " + " | ".join(row) + " |")
        ax[1].bar(np.arange(len(names)) + (k - 2) * .16, [100 * (x.bin == b).mean() for b in names], .16, color=SC[s])
    ax[0].set_xticks(range(len(names))); ax[0].set_xticklabels(names); ax[0].set_xlabel("polyp area as share of the picture"); ax[0].set_ylabel("mean Dice"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    ax[0].set_title("Same polyp size, different data set: is the gap size or data set?", fontsize=10)
    ax[1].set_xticks(range(len(names))); ax[1].set_xticklabels(names); ax[1].set_ylabel("% of the set's pictures"); ax[1].set_title("How the polyp sizes are distributed", fontsize=10); ax[1].grid(alpha=.3, axis="y")
    plt.tight_layout(); plt.savefig(os.path.join(out, "fig_1_matched_size.png"), dpi=150); plt.close(); kn()


# ---------------------------------------------------------------- 2 size counterfactuals
@figure
def fig_sizeswap(X, E, out):
    d = X["sizeswap"]
    if d is None: return False
    d = d[d.run.str.contains("A_base")] if d.run.str.contains("A_base").any() else d
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2)); kn("## 2. Size counterfactuals (base, Protocol A)")
    kn("| set | mode | factor | n | mean Dice | share empty | median polyp area % |"); kn("|---|---|---|---|---|---|---|")
    for s, mode in (("Kvasir120", "shrink"), ("ColonDB", "zoom"), ("ETIS", "zoom")):
        x = d[(d.dataset == s) & (d["mode"] == mode)]
        if x.empty: continue
        if mode == "zoom": x = x[~x.image.isin(set(d[(d.dataset == s) & (d.gt_cut >= 0.1)].image))]   # same pictures at every zoom (paired): drop pictures whose polyp is cut by the crop at ANY zoom
        g = x.groupby("factor").agg(n=("dice", "size"), dice=("dice", "mean"), empty=("empty", "mean"), area=("rel_area", "median"))
        for f, r in g.iterrows(): kn(f"| {s} | {mode} | {f} | {int(r.n)} | {r.dice:.3f} | {r['empty']:.2f} | {100 * r.area:.2f} |")
        ax[0].plot(g.index, g.dice, "o-", color=SC[s], label=f"{s} ({'shrunk by s' if mode == 'shrink' else 'oracle zoom z'})")
    ax[0].set_xlabel("factor (shrink s <= 1  |  zoom z >= 1)"); ax[0].set_ylabel("mean Dice"); ax[0].grid(alpha=.3); ax[0].legend(fontsize=8); ax[0].set_title("Changing ONLY the polyp size", fontsize=10)
    edges = np.linspace(-3, 0, 9); mid = (edges[:-1] + edges[1:]) / 2
    def curve(x, **kw):
        a = np.log10(np.clip(x.rel_area.values if "rel_area" in x else x.area_frac.values, 1e-3, 1)); b = np.digitize(a, edges) - 1; y = x.dice.values
        v = [y[b == i].mean() if (b == i).sum() >= 5 else np.nan for i in range(len(mid))]; ax[1].plot(10 ** mid, v, **kw)
    if E is not None:
        e = avg_seeds(E[(E.protocol == "A") & (E.variant == "base")])
        for s in ("Kvasir120", "ColonDB", "ETIS"): curve(e[e.set == s], color=SC[s], lw=1, ls="--", label=f"{s} as it is")
    for s, mode in (("Kvasir120", "shrink"), ("ColonDB", "zoom"), ("ETIS", "zoom")):
        x = d[(d.dataset == s) & (d["mode"] == mode)]
        if mode == "zoom": x = x[~x.image.isin(set(d[(d.dataset == s) & (d.gt_cut >= 0.1)].image))]   # same pictures at every zoom (paired): drop pictures whose polyp is cut by the crop at ANY zoom
        if len(x): curve(x, color=SC[s], lw=2.2, label=f"{s} after the size change")
    ax[1].set_xscale("log"); ax[1].set_xlabel("polyp area as share of the picture"); ax[1].set_ylabel("mean Dice"); ax[1].grid(alpha=.3); ax[1].legend(fontsize=7)
    ax[1].set_title("If the curves overlap, size explains the gap; if not, the data set does", fontsize=10)
    plt.tight_layout(); plt.savefig(os.path.join(out, "fig_2_size_counterfactual.png"), dpi=150); plt.close(); kn()


# ---------------------------------------------------------------- 3 failure types
@figure
def fig_failure(E, out):
    d = avg_seeds(E[(E.protocol == "A") & (E.variant == "base")]); d["bin"] = d.area_frac.map(lambda a: "<1%" if a < .01 else ("1-5%" if a < .05 else ">=5%"))
    cats = [("nothing predicted", lambda x: x.pred_area == 0, "#444444"), ("wrong place", lambda x: (x.pred_area > 0) & (x.dice < .01), "#c44e52"), ("poor (Dice .01-.5)", lambda x: (x.dice >= .01) & (x.dice < .5), "#dd8452"),
            ("fair (.5-.8)", lambda x: (x.dice >= .5) & (x.dice < .8), "#ccb974"), ("good (>=.8)", lambda x: x.dice >= .8, "#55a868")]
    fig, ax = plt.subplots(figsize=(11, 4)); xs, labs = [], []; k = 0; kn("## 3. Failure types (base, Protocol A, % of pictures)"); kn("| set | size | n | " + " | ".join(c[0] for c in cats) + " |"); kn("|---|---|---|" + "---|" * len(cats))
    for s in SETS:
        for b in ("<1%", "1-5%", ">=5%"):
            x = d[(d.set == s) & (d.bin == b)]
            if len(x) < 3: continue
            bot = 0; fr = []
            for nm, f, c in cats:
                v = 100 * f(x).mean(); ax.bar(k, v, bottom=bot, color=c, label=nm if k == 0 else None); bot += v; fr.append(v)
            kn(f"| {s} | {b} | {len(x)} | " + " | ".join(f"{v:.0f}" for v in fr) + " |"); labs.append(f"{s[:7]}\n{b}\n(n={len(x)})"); xs.append(k); k += 1
        k += .6
    ax.set_xticks(xs); ax.set_xticklabels(labs, fontsize=6); ax.set_ylabel("% of pictures"); ax.legend(fontsize=7, ncol=5, loc="upper center", bbox_to_anchor=(.5, 1.15)); plt.tight_layout(); plt.savefig(os.path.join(out, "fig_3_failure_types.png"), dpi=150); plt.close(); kn()


# ---------------------------------------------------------------- 4 frontier + clean-up
@figure
def fig_views(X, out):
    d = X["views"]
    if d is None: return False
    d = d[d.run.str.contains("A_base")] if d.run.str.contains("A_base").any() else d
    front = ["single", "single_flips", "ensemble", "ensemble_flips"]; clean = ["ensemble_flips", "ensemble_flips+keep_largest", "ensemble_flips+min_area", "ensemble_flips+peak_gate", "ensemble_flips+hysteresis", "ensemble_flips+consensus"]
    sets = [s for s in ("Kvasir120", "ClinicDB612", "ColonDB", "ETIS") if s in set(d.dataset)]
    fig, ax = plt.subplots(2, 3, figsize=(15, 7)); kn("## 4. Accuracy frontier and mask clean-up (base, Protocol A)"); kn("| config | " + " | ".join(f"{s} Dice / HD95diag% / regions" for s in sets) + " |"); kn("|---|" + "---|" * len(sets))
    for r_, (cfgs, ttl) in enumerate(((front, "where the accuracy comes from"), (clean, "mask clean-up on the final map"))):
        w = .8 / len(cfgs)
        for j, (col, nm) in enumerate((("dice", "Dice"), ("hd95_diag", "failure-penalised HD95 / diagonal (lower is better)"), ("n_pred_comp", "number of predicted regions"))):
            for i, c in enumerate(cfgs):
                v = [d[(d.dataset == s) & (d.config == c)][col].mean() for s in sets]; ax[r_, j].bar(np.arange(len(sets)) + i * w, v, w, label=c.replace("ensemble_flips+", "+") if j == 0 else None)
            ax[r_, j].set_xticks(np.arange(len(sets)) + .4); ax[r_, j].set_xticklabels(sets, fontsize=8); ax[r_, j].set_title(f"{ttl}: {nm}", fontsize=8); ax[r_, j].grid(alpha=.3, axis="y")
            if col == "dice": ax[r_, j].set_ylim(.6, 1)
        ax[r_, 0].legend(fontsize=6)
    for c in dict.fromkeys(front + clean):
        kn(f"| {c} | " + " | ".join(f"{d[(d.dataset == s) & (d.config == c)].dice.mean():.4f} / {100 * d[(d.dataset == s) & (d.config == c)].hd95_diag.mean():.2f} / {d[(d.dataset == s) & (d.config == c)].n_pred_comp.mean():.2f}" for s in sets) + " |")
    # paired difference of every clean-up to the plain final map, pooled ColonDB + ETIS
    p = d[d.dataset.isin(["ColonDB", "ETIS"])]; ref = p[p.config == "ensemble_flips"].set_index(["dataset", "image"])
    kn(); kn("Paired difference to `ensemble_flips` (ColonDB+ETIS pooled): config | dDice [95% CI] | d(HD95/diag, %) | share of pictures with a changed Dice")
    for c in clean[1:]:
        q = p[p.config == c].set_index(["dataset", "image"]); dd = (q.dice - ref.dice).dropna(); lo, hi = ci(dd); kn(f"- {c}: {dd.mean():+.4f} [{lo:+.4f},{hi:+.4f}] | {100 * (q.hd95_diag - ref.hd95_diag).mean():+.2f} | {100 * (dd.abs() > 1e-9).mean():.1f}%")
    plt.tight_layout(); plt.savefig(os.path.join(out, "fig_4_frontier_and_cleanup.png"), dpi=150); plt.close(); kn()


# ---------------------------------------------------------------- 5 zoom cascade
@figure
def fig_zoom(X, out):
    d = X["zoom"]
    if d is None: return False
    d = d[d.run.str.contains("A_base")] if d.run.str.contains("A_base").any() else d
    d = d.assign(bin=d.gt_frac.map(binlab)); sets = [s for s in ("ClinicDB612", "ColonDB", "ETIS", "Kvasir120") if s in set(d.dataset)]; names = [b[2] for b in BINS]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4)); kn("## 5. Zoom cascade vs plain final map (base, Protocol A)"); kn("| set | n | Dice raw | Dice zoom | share triggered | empty raw | empty zoom | Dice <1% raw -> zoom (n) |"); kn("|---|---|---|---|---|---|---|---|")
    for k, s in enumerate(sets):
        x = d[d.dataset == s]; r = x[x.config == "raw"]; z = x[x.config == "zoom"]; t = r[r.bin == "<1%"]; tz = z[z.bin == "<1%"]
        kn(f"| {s} | {len(r)} | {r.dice.mean():.4f} | {z.dice.mean():.4f} | {100 * r.triggered.mean():.0f}% | {100 * r["empty"].mean():.1f}% | {100 * z["empty"].mean():.1f}% | {t.dice.mean():.3f} -> {tz.dice.mean():.3f} ({len(t)}) |")
        for i, b in enumerate(names):
            a, c = r[r.bin == b].dice, z[z.bin == b].dice
            if len(a) >= 3: ax[0].bar(i + k * .2, c.mean() - a.mean(), .2, color=SC[s], label=s if i == 0 else None)
        ax[1].bar(k * 2, 100 * r["empty"].mean(), .8, color="#999999"); ax[1].bar(k * 2 + .8, 100 * z["empty"].mean(), .8, color=SC[s]); ax[2].bar(k, 100 * r.triggered.mean(), .6, color=SC[s])
    ax[0].axhline(0, color="k", lw=.8); ax[0].set_xticks(np.arange(len(names)) + .3); ax[0].set_xticklabels(names); ax[0].set_ylabel("change in mean Dice (zoom - plain)"); ax[0].legend(fontsize=7); ax[0].set_title("Where does zoom help? (by polyp size)", fontsize=10); ax[0].grid(alpha=.3, axis="y")
    ax[1].set_xticks([k * 2 + .4 for k in range(len(sets))]); ax[1].set_xticklabels(sets, fontsize=8); ax[1].set_ylabel("% pictures with nothing predicted"); ax[1].set_title("Empty predictions: plain (grey) vs zoom (colour)", fontsize=10); ax[1].grid(alpha=.3, axis="y")
    ax[2].set_xticks(range(len(sets))); ax[2].set_xticklabels(sets, fontsize=8); ax[2].set_ylabel("% pictures where the zoom step ran"); ax[2].set_title("How often the cascade is triggered", fontsize=10); ax[2].grid(alpha=.3, axis="y")
    plt.tight_layout(); plt.savefig(os.path.join(out, "fig_5_zoom_cascade.png"), dpi=150); plt.close(); kn()


# ---------------------------------------------------------------- 6 seeds
@figure
def fig_seeds(E, out):
    d = E[(E.protocol == "A") & (E.variant == "base")]
    if d.seed.nunique() < 2: return False
    fig, ax = plt.subplots(figsize=(7, 4)); kn("## 6. Seed-to-seed spread (base, Protocol A, mean Dice per seed)"); kn("| set | " + " | ".join(f"seed {s}" for s in sorted(d.seed.unique())) + " | spread (max-min) |"); kn("|---|" + "---|" * (d.seed.nunique() + 1))
    for i, s in enumerate(SETS):
        m = d[d.set == s].groupby("seed").dice.mean(); ax.scatter([i] * len(m), m.values, color=SC[s], s=40); ax.plot([i - .25, i + .25], [m.mean()] * 2, color="k", lw=1)
        kn(f"| {s} | " + " | ".join(f"{v:.4f}" for v in m.values) + f" | {m.max() - m.min():.4f} |")
    ax.set_xticks(range(len(SETS))); ax.set_xticklabels(SETS, fontsize=8); ax.set_ylabel("mean Dice of the seed"); ax.grid(alpha=.3); ax.set_title("Same model, different random seeds (dot = seed, line = mean)", fontsize=10)
    plt.tight_layout(); plt.savefig(os.path.join(out, "fig_6_seed_spread.png"), dpi=150); plt.close(); kn()


# ---------------------------------------------------------------- 7 new training ideas vs base (the pre-registered test)
@figure
def fig_new_variants(E, out):
    new = [v for v in ("base_zoom", "base_zoom_match", "base_zoom_black", "base_lab", "base_lab_zoom", "base_pid", "base_512", "base_ms") if v in set(E.variant)]
    if not new: return False
    d = avg_seeds(E[E.protocol == "A"]); d["tiny"] = d.area_frac < .01; ext = d[d.set.isin(["ColonDB", "ETIS"])]; vs = ["base"] + new
    fig, ax = plt.subplots(1, 3, figsize=(15, 4)); kn("## 7. Training ideas vs base (Protocol A, seeds averaged per image; paired on the same pictures)")
    kn("Decision rule fixed in advance for a training idea: Dice of the <1% polyps (ColonDB+ETIS pooled) +0.04 or more, overall (ColonDB+ETIS) not worse than -0.005, then repeat with 3 seeds.")
    kn("| version | Dice <1% (n) | d vs base [95% CI] | Dice ColonDB+ETIS | d vs base [95% CI] | Kvasir120 | ClinicDB612 | rule |"); kn("|---|---|---|---|---|---|---|---|")
    bt = ext[(ext.variant == "base") & ext.tiny].set_index(["set", "image"]).dice; ba = ext[ext.variant == "base"].set_index(["set", "image"]).dice
    for i, v in enumerate(vs):
        t = ext[(ext.variant == v) & ext.tiny].set_index(["set", "image"]).dice; a = ext[ext.variant == v].set_index(["set", "image"]).dice
        dt = (t - bt).dropna(); da = (a - ba).dropna(); lt, ht = ci(dt); la, ha = ci(da); k_ = d[(d.variant == v) & (d.set == "Kvasir120")].dice.mean(); c_ = d[(d.variant == v) & (d.set == "ClinicDB612")].dice.mean()
        ok = "base" if v == "base" else ("PASS" if (dt.mean() >= .04 and da.mean() >= -.005) else "no")
        kn(f"| {v} | {t.mean():.3f} ({len(t)}) | {dt.mean():+.3f} [{lt:+.3f},{ht:+.3f}] | {a.mean():.4f} | {da.mean():+.4f} [{la:+.4f},{ha:+.4f}] | {k_:.4f} | {c_:.4f} | {ok} |")
        ax[0].errorbar(i, dt.mean() if v != "base" else 0, yerr=None if v == "base" else [[dt.mean() - lt], [ht - dt.mean()]], fmt="o", capsize=3, color="#c44e52"); ax[1].errorbar(i, da.mean() if v != "base" else 0, yerr=None if v == "base" else [[da.mean() - la], [ha - da.mean()]], fmt="o", capsize=3, color="#4c72b0")
        for j, s in enumerate(SETS): ax[2].bar(j + i * .8 / len(vs), d[(d.variant == v) & (d.set == s)].dice.mean(), .8 / len(vs), color=plt.cm.tab20(i), label=v if j == 0 else None)
    for a_, ttl, thr in ((ax[0], "Dice change on <1% polyps (rule: >= +0.04)", .04), (ax[1], "Dice change ColonDB+ETIS (rule: >= -0.005)", -.005)):
        a_.axhline(0, color="k", lw=.8); a_.axhline(thr, color="g", ls="--", lw=.8); a_.set_xticks(range(len(vs))); a_.set_xticklabels(vs, rotation=40, fontsize=7); a_.set_title(ttl, fontsize=9); a_.grid(alpha=.3)
    ax[2].set_xticks(np.arange(len(SETS)) + .35); ax[2].set_xticklabels(SETS, fontsize=7); ax[2].set_ylim(.7, .95); ax[2].legend(fontsize=5, ncol=2); ax[2].set_title("Mean Dice per set", fontsize=9); ax[2].grid(alpha=.3, axis="y")
    plt.tight_layout(); plt.savefig(os.path.join(out, "fig_7_training_ideas.png"), dpi=150); plt.close(); kn()


# ---------------------------------------------------------------- 8 metric definitions
@figure
def fig_metric(bundle, out):
    fs = sorted(glob.glob(os.path.join(bundle, "eval_extra", "A", "base_pranet_fb", "seed*", "per_image.csv")))
    if not fs: return False
    d = pd.read_csv(fs[0]); d = d[~d.dataset.str.endswith("+overlay")]; sets = [s for s in SETS if s in set(d.dataset)]
    fig, ax = plt.subplots(figsize=(8, 4)); kn("## 8. Same maps, different metric definitions (base, Protocol A, mean Dice)"); kn("| set | fixed threshold | PraNet-toolbox (min-max, 256 thresholds) | fixed + never-empty fallback |"); kn("|---|---|---|---|")
    for j, (col, nm, c) in enumerate((("dice", "fixed threshold (ours)", "#4c72b0"), ("pn_dice", "PraNet-toolbox metric", "#dd8452"), ("fb_dice", "fixed + never-empty fallback", "#55a868"))):
        ax.bar(np.arange(len(sets)) + j * .27, [d[d.dataset == s][col].mean() for s in sets], .27, color=c, label=nm)
    for s in sets: kn(f"| {s} | {d[d.dataset == s].dice.mean():.4f} | {d[d.dataset == s].pn_dice.mean():.4f} | {d[d.dataset == s].fb_dice.mean():.4f} |")
    ax.set_xticks(np.arange(len(sets)) + .27); ax.set_xticklabels(sets); ax.set_ylim(.6, 1); ax.set_ylabel("mean Dice"); ax.legend(fontsize=8); ax.grid(alpha=.3, axis="y"); ax.set_title("Published numbers use the middle bar's definition; ours is the left bar", fontsize=10)
    plt.tight_layout(); plt.savefig(os.path.join(out, "fig_8_metric_definitions.png"), dpi=150); plt.close(); kn()


# ---------------------------------------------------------------- 9 PolypGen
@figure
def fig_polypgen(X, out):
    d0 = X["polypgen"]
    if d0 is None: return False
    for pr in sorted(d0.protocol.unique()):          # never pool models trained under different protocols
        _polypgen_one(d0[d0.protocol == pr].copy(), out, pr)


def _polypgen_one(d, out, pr):
    cfgs = [c for c in ("raw", "consensus", "peak_gate", "zoom", "zoom+consensus") if c in set(d.config)]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2)); kn(f"## 9. PolypGen, model trained under Protocol {pr} (zero-shot, research-only data; cite the three PolypGen papers)")
    s = d[d.kind == "single"]; cs = sorted(s.dataset.unique()); w = .8 / len(cfgs)
    kn("| centre (single frames) | n | " + " | ".join(f"Dice {c}" for c in cfgs) + " |"); kn("|---|---|" + "---|" * len(cfgs))
    for i, c in enumerate(cfgs):
        ax[0].bar(np.arange(len(cs)) + i * w, [s[(s.dataset == k) & (s.config == c)].dice.mean() for k in cs], w, label=c)
    for k in cs: kn(f"| {k} | {int((s[(s.dataset == k) & (s.config == cfgs[0])]).shape[0])} | " + " | ".join(f"{s[(s.dataset == k) & (s.config == c)].dice.mean():.3f}" for c in cfgs) + " |")
    kn(f"| all single frames | {int((s.config == cfgs[0]).sum())} | " + " | ".join(f"{s[s.config == c].dice.mean():.3f}" for c in cfgs) + " |")
    ax[0].set_xticks(np.arange(len(cs)) + .4); ax[0].set_xticklabels(cs); ax[0].set_ylabel("mean Dice"); ax[0].legend(fontsize=7); ax[0].set_title("Centres C1-C6, single frames", fontsize=10); ax[0].grid(alpha=.3, axis="y")
    q = d[d.kind == "seq_pos"]; seqs = sorted(q.dataset.unique(), key=lambda t: int("".join(ch for ch in t if ch.isdigit())))
    if len(q): ax[1].bar(range(len(seqs)), [q[(q.dataset == k) & (q.config == "raw")].dice.mean() for k in seqs], color="#4c72b0"); ax[1].set_xticks(range(len(seqs))); ax[1].set_xticklabels([t.replace("seq", "") for t in seqs], fontsize=6); ax[1].set_xlabel("positive sequence"); ax[1].set_ylabel("mean Dice (raw)"); ax[1].set_title("Positive sequences (each bar = one video)", fontsize=10); ax[1].grid(alpha=.3, axis="y")
    kn(); kn("False alarms on frames WITHOUT a polyp (share of frames where anything is predicted; mean predicted pixels)"); kn("| frames | n | " + " | ".join(cfgs) + " |"); kn("|---|---|" + "---|" * len(cfgs))
    for j, (kind, nm) in enumerate((("neg", "polyp-free sequences"), ("seq_empty", "empty frames of positive sequences"))):
        z = d[d.kind == kind]
        if z.empty: continue
        kn(f"| {nm} | {int((z.config == cfgs[0]).sum())} | " + " | ".join(f"{100 * z[z.config == c].pred_any.mean():.1f}% ({z[z.config == c].pred_area.mean():.0f} px)" for c in cfgs) + " |")
        ax[2].bar(np.arange(len(cfgs)) + j * .4, [100 * z[z.config == c].pred_any.mean() for c in cfgs], .4, label=nm)
    ax[2].set_xticks(np.arange(len(cfgs)) + .2); ax[2].set_xticklabels(cfgs, fontsize=7); ax[2].set_ylabel("% of frames with a false alarm"); ax[2].legend(fontsize=7); ax[2].set_title("Frames without a polyp: false alarms", fontsize=10); ax[2].grid(alpha=.3, axis="y")
    plt.tight_layout(); plt.savefig(os.path.join(out, f"fig_9_polypgen_{pr}.png"), dpi=150); plt.close(); kn()


# ---------------------------------------------------------------- 10 qualitative (public data sets only; never PolypGen)
@figure
def fig_qualitative(E, bundle, data_root, out):
    import cv2
    if not data_root: return False
    d = avg_seeds(E[(E.protocol == "A") & (E.variant == "base")]); pdirs = {"ColonDB": "CVC-ColonDB", "ETIS": "ETIS-LaribPolypDB"}
    picks = []
    for s in ("ColonDB", "ETIS"):
        x = d[(d.set == s) & (d.area_frac < .01)].sort_values("dice")
        picks += [(s, r.image, r.dice) for r in pd.concat([x.head(3), x.tail(3)]).itertuples()]
    fig, ax = plt.subplots(3, 4, figsize=(14, 10)); ax = ax.ravel(); n = 0
    for (s, name, dc) in picks[:12]:
        im = glob.glob(os.path.join(data_root, "PraNet_TestDataset", pdirs[s], "images", f"{name}.*")); gm = glob.glob(os.path.join(data_root, "PraNet_TestDataset", pdirs[s], "masks", f"{name}.*"))
        pr = glob.glob(os.path.join(bundle, "eval", "A", "base", "seed0", "preds", s, f"{name}.png"))
        if not (im and gm and pr): continue
        x = cv2.cvtColor(cv2.imread(im[0]), cv2.COLOR_BGR2RGB); g = cv2.imread(gm[0], 0) > 127; p = cv2.imread(pr[0], 0) > 127
        for m, col in ((g, (0, 255, 0)), (p, (255, 0, 0))):
            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(x, cs, -1, col, max(2, x.shape[0] // 200))
        ax[n].imshow(x); ax[n].axis("off"); ax[n].set_title(f"{s} {name}  Dice {dc:.2f}", fontsize=8); n += 1
    for a_ in ax[n:]: a_.axis("off")
    if n == 0: plt.close(); return False
    plt.suptitle("Tiny polyps (<1%), 3 worst and 3 best per set. green = true outline, red = prediction", fontsize=10); plt.tight_layout(); plt.savefig(os.path.join(out, "fig_10_qualitative_tiny_polyps.png"), dpi=130); plt.close()


def summary_table(E):
    d = avg_seeds(E[E.protocol == "A"]); kn("## 0. Mean Dice per version (Protocol A; seeds averaged per image)"); kn("| version | " + " | ".join(SETS) + " | <1% polyps ColonDB+ETIS (n) | empty % (ColonDB+ETIS) | HD95/diag % (ColonDB+ETIS) |"); kn("|---|" + "---|" * (len(SETS) + 3))
    for v in dict.fromkeys(E[E.protocol == "A"].variant):
        x = d[d.variant == v]; e = x[x.set.isin(["ColonDB", "ETIS"])]; t = e[e.area_frac < .01]
        kn(f"| {v} | " + " | ".join(f"{x[x.set == s].dice.mean():.4f}" for s in SETS) + f" | {t.dice.mean():.3f} ({len(t)}) | " + (f"{100 * e["empty"].mean():.1f}" if "empty" in e and e["empty"].notna().any() else "-") + " | " + (f"{100 * e.hd95_diag.mean():.2f}" if "hd95_diag" in e and e.hd95_diag.notna().any() else "-") + " |")
    kn()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--bundle", required=True); ap.add_argument("--out", required=True); ap.add_argument("--data_root", default="")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    E = read_eval(a.bundle); X = {k: xread(a.bundle, k) for k in ("views", "zoom", "sizeswap", "polypgen")}
    kn("# Key numbers (auto-generated by make_figures.py)"); kn(f"bundle: {a.bundle}"); kn()
    if E is not None:
        summary_table(E); fig_matched_size(E, a.out); fig_failure(E, a.out); fig_seeds(E, a.out); fig_new_variants(E, a.out); fig_qualitative(E, a.bundle, a.data_root, a.out)
    fig_sizeswap(X, E, a.out); fig_views(X, a.out); fig_zoom(X, a.out); fig_metric(a.bundle, a.out); fig_polypgen(X, a.out)
    kn("---"); kn("figures made: " + ", ".join(DONE)); kn("skipped (input missing or error): " + (", ".join(SKIP) or "none"))
    open(os.path.join(a.out, "key_numbers.md"), "w", encoding="utf-8").write("\n".join(KN) + "\n")
    print("figures made:", DONE); print("skipped:", SKIP); print("wrote", os.path.join(a.out, "key_numbers.md"))


if __name__ == "__main__":
    main()
