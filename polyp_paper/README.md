# polyp_paper — training and evaluation code (research branch)

Code for a leakage-aware, cross-dataset study of transformer-encoder polyp segmentation (UNet decoder on a MiT-B3 encoder, a decoder built from multi-level attention + depth-wise separable convolutions + ASPP, a size head, and overlay-invariance training).

**Status: work in progress. No result from this branch is final until it is listed in a results table with its run bundle.**

## Protocols (see `splits.py`)
| Name | Train pool | Test sets |
|---|---|---|
| `A` | official Kvasir-SEG `train.txt` (880) | Kvasir-120 (official `val.txt`); ClinicDB-612, ColonDB-380, ETIS-196, CVC-300-60 (zero-shot) |
| `B` | PraNet standard: 900 Kvasir + 550 ClinicDB | Kvasir-100, ClinicDB-62, ColonDB, ETIS, CVC-300 |
| `Bclean` | as `B`, but ClinicDB frames are split by the official frame-to-sequence table (sequences 11, 18, 23 are the test set) | Kvasir-100, ClinicDB-54 (held-out sequences), ColonDB, ETIS, CVC-300 |

Do not mix A and B: 91 of PraNet's 100 Kvasir test images are inside A's training set, and 111 of A's 120 test images are inside B's training set. Kvasir-SEG has no patient or sequence identifiers, so no patient-level split is possible for it; this is a limitation of the data.

## Files
`prepare_data.py` build the data folder · `splits.py` protocols and folds · `data.py` dataset, augmentation, overlay · `model.py` models and variants · `losses.py` losses · `train.py` train one inner fold · `evaluate.py` ensemble inference and per-image metrics · `metrics.py` Dice, IoU, precision, recall, F2, HD95, bootstrap CI, paired test (self-check: `python metrics.py`) · `selfcheck.py` CPU check of data, losses and all variants · `kaggle_train.ipynb` notebook for Kaggle (T4 x2).

## Rules used by the code
Model selection uses only the inner validation fold; the decision threshold comes only from out-of-fold predictions; test sets are read only by `evaluate.py`; masks are thresholded at >127 before any resize; all metrics are per image on binary masks at the original resolution.
