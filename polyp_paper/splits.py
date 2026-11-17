"""Protocol definitions (train pool, inner folds, test sets). Every list comes from files under <data_root>/splits
(verified against the datasets; see the project evidence notes). Nothing here groups, clusters or guesses patients.

Protocols
  A      train pool = official Kvasir-SEG train.txt (880).  Tests: Kvasir-120 (official val.txt) + ClinicDB-612, ColonDB, ETIS, CVC-300 (zero-shot).
  B      PraNet standard: train pool = 900 Kvasir + 550 ClinicDB.  Tests: Kvasir-100, ClinicDB-62, ColonDB, ETIS, CVC-300.
  Bclean as B, but ClinicDB frames are split by the OFFICIAL ClinicDB frame->sequence table: sequences 11, 18, 23 (54 frames)
         are the test set, inner folds keep every sequence on one side.  Kvasir part as in B (no IDs exist).
NEVER mix A and B: 91 of PraNet's 100 Kvasir test images are inside A's training set, and 111 of A's 120 test images are in B's.

data_root layout (made by prepare_data.py):
  Kvasir-SEG/{images,masks}  CVC-ClinicDB/{images,masks} (612, names 1..612.png)  PraNet_TestDataset/<set>/{images,masks}
  PraNet_TrainDataset/TrainDataset/{image,masks}  splits/*.txt  splits/clinicdb_sequences.csv
"""
import argparse, csv, os
import cv2, numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold

EXT = (".jpg", ".jpeg", ".png", ".tif", ".bmp")
HELD_OUT_SEQ = (11, 18, 23)          # test sequences for Bclean (same as the published leakage-free ClinicDB split)


def _file(folder, stem):
    for e in EXT:
        p = os.path.join(folder, stem + e)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(os.path.join(folder, stem))


def _names(folder):
    return sorted(os.path.splitext(f)[0] for f in os.listdir(folder) if f.lower().endswith(EXT))


def _read(path):
    return [x.strip() for x in open(path) if x.strip()]


def _items(img_dir, mask_dir, names, dataset, group_fn=None):
    out = []
    for n in names:
        m = cv2.imread(_file(mask_dir, n), cv2.IMREAD_GRAYSCALE)
        a = float((m > 127).mean())
        out.append(dict(name=n, dataset=dataset, img=_file(img_dir, n), mask=_file(mask_dir, n), area=a,
                        bin=0 if a < .05 else 1 if a < .15 else 2 if a < .35 else 3,
                        group=group_fn(n) if group_fn else None))
    return out


def _seq_table(root):
    return {int(r["frame"]): int(r["sequence"]) for r in csv.DictReader(open(os.path.join(root, "splits", "clinicdb_sequences.csv")))}


def get_protocol(root, protocol):
    """returns (pool: list of item dicts, tests: dict name -> list of items)."""
    sp = lambda f: os.path.join(root, "splits", f)
    kv, clinic = os.path.join(root, "Kvasir-SEG"), os.path.join(root, "CVC-ClinicDB")
    pt = os.path.join(root, "PraNet_TestDataset")
    pr = os.path.join(root, "PraNet_TrainDataset", "TrainDataset")
    ext = {k: _items(os.path.join(pt, d, "images"), os.path.join(pt, d, "masks"), _names(os.path.join(pt, d, "images")), k)
           for k, d in (("ColonDB", "CVC-ColonDB"), ("ETIS", "ETIS-LaribPolypDB"), ("CVC300", "CVC-300"))}
    clinic_all = lambda sel: [i for i in _items(os.path.join(clinic, "images"), os.path.join(clinic, "masks"), sel, "ClinicDB")]
    if protocol == "A":
        pool = _items(os.path.join(kv, "images"), os.path.join(kv, "masks"), _read(sp("kvasir_official_train.txt")), "Kvasir")
        tests = {"Kvasir120": _items(os.path.join(kv, "images"), os.path.join(kv, "masks"), _read(sp("kvasir_official_val.txt")), "Kvasir120"),
                 "ClinicDB612": clinic_all([str(i) for i in range(1, 613)]), **ext}
    elif protocol in ("B", "Bclean"):
        kv_tr = [n for n in _names(os.path.join(pr, "image")) if not n.isdigit()]
        pool = _items(os.path.join(pr, "image"), os.path.join(pr, "masks"), kv_tr, "Kvasir")
        k100 = _items(os.path.join(pt, "Kvasir", "images"), os.path.join(pt, "Kvasir", "masks"), _names(os.path.join(pt, "Kvasir", "images")), "Kvasir100")
        if protocol == "B":
            cl_tr = [n for n in _names(os.path.join(pr, "image")) if n.isdigit()]
            pool += _items(os.path.join(pr, "image"), os.path.join(pr, "masks"), cl_tr, "ClinicDB")
            c62 = _items(os.path.join(pt, "CVC-ClinicDB", "images"), os.path.join(pt, "CVC-ClinicDB", "masks"), _names(os.path.join(pt, "CVC-ClinicDB", "images")), "ClinicDB62")
            tests = {"Kvasir100": k100, "ClinicDB62": c62, **ext}
        else:
            seq = _seq_table(root)
            frames = [str(i) for i in range(1, 613)]
            pool += clinic_all([f for f in frames if seq[int(f)] not in HELD_OUT_SEQ])
            for i in pool:
                if i["dataset"] == "ClinicDB":
                    i["group"] = seq[int(i["name"])]
            held = [f for f in frames if seq[int(f)] in HELD_OUT_SEQ]
            tests = {"Kvasir100": k100, "ClinicDBseq54": [dict(i, dataset="ClinicDBseq54") for i in clinic_all(held)], **ext}
    else:
        raise ValueError(protocol)
    return pool, tests


def make_folds(pool, protocol, k=5, seed=42):
    """List of (train_idx, val_idx). Strata = lesion-area bin x source dataset. Bclean keeps each ClinicDB sequence on one side
    (every Kvasir image is its own group, i.e. NO grouping is applied to Kvasir)."""
    y = np.array([i["bin"] + 4 * (i["dataset"] == "ClinicDB") for i in pool])
    X = np.zeros(len(pool))
    if protocol == "Bclean":
        g = np.array([i["group"] if i["group"] is not None else 10_000 + n for n, i in enumerate(pool)])
        return list(StratifiedGroupKFold(k, shuffle=True, random_state=seed).split(X, y, g))
    return list(StratifiedKFold(k, shuffle=True, random_state=seed).split(X, y))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--data_root", default="data"); ap.add_argument("--protocol", default="A")
    a = ap.parse_args()
    pool, tests = get_protocol(a.data_root, a.protocol)
    print(f"protocol {a.protocol}: pool {len(pool)}", {d: sum(i['dataset'] == d for i in pool) for d in {i['dataset'] for i in pool}})
    print("tests:", {k: len(v) for k, v in tests.items()})
    folds = make_folds(pool, a.protocol)
    print("fold sizes (train/val):", [(len(t), len(v)) for t, v in folds])
    names = {i["name"] + "|" + i["dataset"] for i in pool}
    for k, v in tests.items():       # a test image must never be in the pool
        bad = [i["name"] for i in v if (i["name"] + "|" + i["dataset"].replace("Kvasir120", "Kvasir").replace("Kvasir100", "Kvasir").replace("ClinicDB612", "ClinicDB").replace("ClinicDB62", "ClinicDB").replace("ClinicDBseq54", "ClinicDB")) in names]
        print(f"  test {k}: {len(v)} images, in training pool: {len(bad)}")
    if a.protocol == "Bclean":
        va_groups = [{pool[j]['group'] for j in v if pool[j]['group'] is not None} for _, v in folds]
        tr_groups = [{pool[j]['group'] for j in t if pool[j]['group'] is not None} for t, _ in folds]
        print("  sequences shared between a fold's train and val:", [len(a_ & b_) for a_, b_ in zip(va_groups, tr_groups)])
