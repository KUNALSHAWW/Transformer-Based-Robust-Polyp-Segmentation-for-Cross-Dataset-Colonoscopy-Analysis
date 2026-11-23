"""PolypGen loader (external test data; NEVER used for training or for choosing anything).

DATA TERMS (from the dataset owners): research use only; NO redistribution and no new dataset creation; do not put the data on Kaggle as a
dataset, on GitHub or on any web page; download it from Synapse (project syn26376615, data folder syn45200214). Anything that uses it must cite:
 [1] Ali S, Jha D, Ghatwary N et al. A multi-centre polyp detection and segmentation dataset for generalisability assessment. Sci Data 10, 75 (2023). doi 10.1038/s41597-023-01981-y
 [2] Ali S, Ghatwary N, Jha D et al. Assessing generalisability of deep learning-based polyp detection and segmentation methods through a computer vision challenge. Sci Rep 14, 2032 (2024). doi 10.1038/s41598-024-52063-x
 [3] Ali S, Dmitrieva M, Ghatwary N et al. Deep learning for detection and segmentation of artefact and disease instances in gastrointestinal endoscopy. Med Image Anal 2021:102002. doi 10.1016/j.media.2021.102002
Dataset DOI 10.7303/syn26376615.

On Kaggle the data is downloaded INSIDE the running session into a temporary folder that is not part of the saved output; only result tables leave it.
python polypgen.py <root>   -> prints the counts"""
import glob, os, sys, zipfile


def download(token, dest, syn_id="syn45200214"):
    """download the PolypGen entity with a personal access token (never printed) and unpack only the images and masks that are needed."""
    import synapseclient
    syn = synapseclient.Synapse(); syn.login(authToken=token, silent=True)
    os.makedirs(dest, exist_ok=True)
    ent = syn.get(syn_id, downloadFile=False)
    if "Folder" in str(ent.get("concreteType", "")) or "Project" in str(ent.get("concreteType", "")):
        import synapseutils
        synapseutils.syncFromSynapse(syn, ent, path=dest)
    else:
        syn.get(syn_id, downloadLocation=dest)
    for z in glob.glob(os.path.join(dest, "**", "*.zip"), recursive=True):
        extract_needed(z, dest)
        os.remove(z)                                                   # free the disk space
    return find_root(dest)


def extract_needed(zip_path, dest):
    """only .jpg images and masks (no boxes, no duplicate copies, no __MACOSX junk)"""
    with zipfile.ZipFile(zip_path) as z:
        keep = [n for n in z.namelist() if n.lower().endswith(".jpg") and not n.startswith("__MACOSX")
                and ("/images_" in n or "/masks_" in n or "/negativeOnly/" in n)]
        z.extractall(dest, members=keep)
    return len(keep)


def find_root(dest):
    hits = glob.glob(os.path.join(dest, "**", "data_C1"), recursive=True)
    assert hits, "PolypGen folders not found under " + dest
    return os.path.dirname(hits[0])


def items(root):
    """list of dicts: img, mask (path or None = no polyp), group, kind, name.
    kind: single (one frame per patient video, with polyp, centres C1-C6), seq_pos (frame of a positive sequence with a mask),
    seq_empty (frame of a positive sequence that has no polyp mask), neg (frame of a polyp-free sequence)."""
    out = []
    for c in range(1, 7):
        for f in sorted(glob.glob(os.path.join(root, f"data_C{c}", f"images_C{c}", "*.jpg"))):
            stem = os.path.splitext(os.path.basename(f))[0]; m = os.path.join(root, f"data_C{c}", f"masks_C{c}", stem + "_mask.jpg")
            out.append(dict(img=f, mask=m if os.path.exists(m) else None, group=f"C{c}", kind="single" if os.path.exists(m) else "single_nomask", name=f"C{c}/{stem}"))
    for d in sorted(glob.glob(os.path.join(root, "sequenceData", "positive", "seq*")), key=lambda p: int("".join(ch for ch in os.path.basename(p) if ch.isdigit()))):
        n = os.path.basename(d)
        for f in sorted(glob.glob(os.path.join(d, f"images_{n}", "*.jpg"))):
            stem = os.path.splitext(os.path.basename(f))[0]; m = os.path.join(d, f"masks_{n}", stem + "_mask.jpg")
            out.append(dict(img=f, mask=m if os.path.exists(m) else None, group=n, kind="seq_pos" if os.path.exists(m) else "seq_empty", name=f"{n}/{stem}"))
    for d in sorted(glob.glob(os.path.join(root, "sequenceData", "negativeOnly", "seq*_neg")), key=lambda p: int("".join(ch for ch in os.path.basename(p) if ch.isdigit()))):
        n = os.path.basename(d)
        for f in sorted(glob.glob(os.path.join(d, "*.jpg"))):
            out.append(dict(img=f, mask=None, group=n, kind="neg", name=f"{n}/{os.path.splitext(os.path.basename(f))[0]}"))
    return out


if __name__ == "__main__":
    import collections
    it = items(sys.argv[1]); print(len(it), "images"); print(collections.Counter(i["kind"] for i in it)); print(collections.Counter(i["group"] for i in it if i["kind"] == "single"))
