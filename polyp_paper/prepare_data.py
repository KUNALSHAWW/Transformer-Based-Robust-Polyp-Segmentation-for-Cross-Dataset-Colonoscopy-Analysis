"""Build the data_root layout used by splits.py, from public sources, and verify counts.

Kaggle:  python prepare_data.py --out /kaggle/working/data --kvasir /kaggle/input/<medico dataset>/.../Kvasir-SEG --download
Local :  python prepare_data.py --out data --kvasir <Kvasir-SEG dir> --pranet_test <dir> --pranet_train <dir> --clinic <dir>   (copies, no download)
Sources: PraNet test/train bundles (Google Drive links in the PraNet README), ClinicDB 612 frames + frame->sequence metadata
(Hugging Face mirror berkaytrhn/cvc-clinicdb, verified against the PraNet copy), official Kvasir-SEG split lists (in ./splits_lists).
"""
import argparse, csv, json, os, shutil, subprocess, sys, urllib.parse, urllib.request, zipfile

DRIVE = "https://drive.usercontent.google.com/download?id={}&export=download&confirm=t"
TEST_ID, TRAIN_ID = "1Y2z7FD5p5y31vkZwQQomXFRB0HutHyao", "1YiGHLw4iTvKdvbT6MgwO9zcCv8zJ_Bnb"
HF = "berkaytrhn/cvc-clinicdb"
EXPECT_TEST = {"CVC-300": 60, "CVC-ClinicDB": 62, "CVC-ColonDB": 380, "ETIS-LaribPolypDB": 196, "Kvasir": 100}


def fetch(url, dst):
    if not os.path.exists(dst):
        print("downloading", url[:70], "->", dst); urllib.request.urlretrieve(url, dst)


def unzip_clean(zpath, dest):
    z = zipfile.ZipFile(zpath); dest = os.path.abspath(dest)
    for i in z.infolist():
        if i.is_dir() or i.filename.startswith("__MACOSX") or i.filename.lower().endswith(".ds_store"):
            continue
        out = os.path.abspath(os.path.join(dest, i.filename)); assert out.startswith(dest + os.sep)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with z.open(i) as s, open(out, "wb") as t:
            t.write(s.read())


def link(src, dst):
    if not os.path.exists(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        (shutil.copytree if os.path.isdir(src) else shutil.copy)(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data"); ap.add_argument("--kvasir", required=True)
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--pranet_test"); ap.add_argument("--pranet_train"); ap.add_argument("--clinic")
    a = ap.parse_args(); out = os.path.abspath(a.out); os.makedirs(out, exist_ok=True)
    here = os.path.dirname(os.path.abspath(__file__))
    link(a.kvasir, os.path.join(out, "Kvasir-SEG"))
    if a.download:
        tmp = os.path.join(out, "_dl"); os.makedirs(tmp, exist_ok=True)
        fetch(DRIVE.format(TEST_ID), os.path.join(tmp, "test.zip")); fetch(DRIVE.format(TRAIN_ID), os.path.join(tmp, "train.zip"))
        unzip_clean(os.path.join(tmp, "test.zip"), tmp + "/t"); unzip_clean(os.path.join(tmp, "train.zip"), tmp + "/r")
        link(tmp + "/t/TestDataset", os.path.join(out, "PraNet_TestDataset")); link(tmp + "/r/TrainDataset", os.path.join(out, "PraNet_TrainDataset", "TrainDataset"))
        info = json.load(urllib.request.urlopen(f"https://huggingface.co/api/datasets/{HF}", timeout=60))
        for f in (s["rfilename"] for s in info["siblings"]):
            if f.lower().endswith(".png") or f == "metadata.csv":
                dst = os.path.join(tmp, "clinic", f); os.makedirs(os.path.dirname(dst), exist_ok=True)
                fetch(f"https://huggingface.co/datasets/{HF}/resolve/main/" + urllib.parse.quote(f), dst)
        clinic = os.path.join(tmp, "clinic", "CVC-ClinicDB")
    else:
        link(a.pranet_test, os.path.join(out, "PraNet_TestDataset")); link(a.pranet_train, os.path.join(out, "PraNet_TrainDataset", "TrainDataset"))
        clinic = a.clinic
    for sub, src in (("images", "Original"), ("masks", "Ground Truth")):
        os.makedirs(os.path.join(out, "CVC-ClinicDB", sub), exist_ok=True)
        for f in os.listdir(os.path.join(clinic, src)):
            link(os.path.join(clinic, src, f), os.path.join(out, "CVC-ClinicDB", sub, f))
    os.makedirs(os.path.join(out, "splits"), exist_ok=True)
    for f in os.listdir(os.path.join(here, "splits_lists")):
        link(os.path.join(here, "splits_lists", f), os.path.join(out, "splits", f))
    # verification
    n = lambda *p: len(os.listdir(os.path.join(out, *p)))
    assert n("Kvasir-SEG", "images") == 1000 and n("Kvasir-SEG", "masks") == 1000
    assert n("CVC-ClinicDB", "images") == 612 and n("CVC-ClinicDB", "masks") == 612
    for d, c in EXPECT_TEST.items():
        assert n("PraNet_TestDataset", d, "images") == c and n("PraNet_TestDataset", d, "masks") == c, d
    assert n("PraNet_TrainDataset", "TrainDataset", "image") == 1450 and n("PraNet_TrainDataset", "TrainDataset", "masks") == 1450
    assert len(csv.DictReader(open(os.path.join(out, "splits", "clinicdb_sequences.csv"))).fieldnames) == 3
    print("data_root ready and counts verified:", out)


if __name__ == "__main__":
    main()
