"""Are any PolypGen pictures (near-)copies of Kvasir-SEG / ClinicDB / ColonDB / ETIS / CVC-300 pictures? Runs on the CPU, reads the PolypGen zip directly
(nothing is unpacked, nothing is copied). Perceptual difference hash (256 bit); a small Hamming distance means the same or nearly the same picture.
python overlap_check.py --data_root <prepared data folder> --polypgen_zip <PolypGen zip> --out <csv>"""
import argparse, csv, glob, os, zipfile
import cv2, numpy as np

POP = np.array([bin(i).count("1") for i in range(256)], np.uint8)


def dhash(gray):
    g = cv2.resize(gray, (17, 16), interpolation=cv2.INTER_AREA).astype(np.int16)
    return np.packbits((g[:, 1:] > g[:, :-1]).ravel())                      # 16 x 16 = 256 bits -> 32 bytes


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data_root", required=True); ap.add_argument("--polypgen_zip", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0); a = ap.parse_args()
    R = a.data_root
    ours = {"Kvasir": glob.glob(f"{R}/Kvasir-SEG/images/*"), "ClinicDB": glob.glob(f"{R}/CVC-ClinicDB/images/*"),
            "ColonDB": glob.glob(f"{R}/PraNet_TestDataset/CVC-ColonDB/images/*"), "ETIS": glob.glob(f"{R}/PraNet_TestDataset/ETIS-LaribPolypDB/images/*"),
            "CVC300": glob.glob(f"{R}/PraNet_TestDataset/CVC-300/images/*")}
    H, lab, path = [], [], []
    for k, fs in ours.items():
        for f in fs:
            im = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if im is not None: H.append(dhash(im)); lab.append(k); path.append(os.path.basename(f))
    H = np.array(H); lab = np.array(lab); print({k: len(v) for k, v in ours.items()}, "hashed", len(H))
    z = zipfile.ZipFile(a.polypgen_zip)
    names = [n for n in z.namelist() if n.lower().endswith(".jpg") and not n.startswith("__MACOSX") and ("/images_" in n or "/negativeOnly/" in n)]
    names = names[:a.limit] if a.limit else names
    rows = []
    for i, n in enumerate(names):
        im = cv2.imdecode(np.frombuffer(z.read(n), np.uint8), cv2.IMREAD_GRAYSCALE)
        if im is None: continue
        d = POP[np.bitwise_xor(H, dhash(im)[None])].sum(1); j = int(d.argmin())
        rows.append(dict(polypgen=n.split("/", 1)[1], nearest_set=lab[j], nearest_file=path[j], hamming=int(d[j]), **{f"min_{k}": int(d[lab == k].min()) for k in ours}))
        if i % 1000 == 0: print(i, "/", len(names), flush=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    m = np.array([r["hamming"] for r in rows])
    print("PolypGen pictures checked:", len(m))
    for t in (0, 5, 10, 20, 30, 40): print(f"  nearest distance <= {t:2d} of 256 bits: {int((m <= t).sum())} pictures")
    print("  median / 5th percentile / minimum nearest distance:", float(np.median(m)), float(np.percentile(m, 5)), int(m.min()))
    for k in ours: print(f"  {k:9s} closest PolypGen picture, distance:", min(r[f"min_{k}"] for r in rows))


if __name__ == "__main__":
    main()
