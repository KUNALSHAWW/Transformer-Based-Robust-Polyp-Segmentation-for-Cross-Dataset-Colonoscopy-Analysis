"""Dataset, augmentation and the synthetic on-screen overlay (Idea 1).

Masks are thresholded at >127 BEFORE any resize (Kvasir-SEG masks are JPEG, not binary) and resized with nearest-neighbour.
Overlay modes: 'none' | 'aug' (replace the image by its overlaid version with probability p) | 'pair' (return clean + overlaid).
"""
import math
import albumentations as A
import cv2, numpy as np, torch
from torch.utils.data import Dataset

MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
SIZE = 384
_TEXTS = ["15/03/2012", "12:32:09", "<00:05:02>", "CVP:1", "O:N", "A3", "07/03/2015", "11:14:24", "Sex:", "ID:", "00:42:11", "CVP:6"]


def to_tensor(img_u8):
    return torch.from_numpy(((img_u8.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1).copy())


def build_geo(seed=None, size=SIZE):
    """Same augmentation family as the original recipe (written for albumentations 2.x)."""
    return A.Compose([
        A.Resize(size, size), A.HorizontalFlip(p=.5), A.VerticalFlip(p=.5),
        A.ShiftScaleRotate(shift_limit=.05, scale_limit=.10, rotate_limit=15, border_mode=cv2.BORDER_REFLECT_101, p=.5),
        A.GridDistortion(num_steps=5, distort_limit=.15, border_mode=cv2.BORDER_REFLECT_101, p=.3),
        A.ColorJitter(brightness=.15, contrast=.15, saturation=.15, hue=.05, p=.5),
        A.RandomGamma(gamma_limit=(85, 115), p=.25),
        A.GaussNoise(std_range=(.0088, .0175), p=.20),       # = old var_limit (5, 20) in pixel units
        A.GaussianBlur(blur_limit=(3, 5), p=.15)], seed=seed)


def ui_overlay(img, mask, rng):
    """Paste synthetic endoscope-screen items (white text, teal/black inset boxes) on an RGB uint8 image.
    The mask is NOT changed; a placement that would cover more than 5% of the polyp is retried (3 times), else no overlay."""
    H, W = mask.shape
    for _ in range(3):
        cand, boxes = img.copy(), []
        for _ in range(int(rng.integers(1, 4))):
            kind = rng.choice(["text", "inset", "black"], p=[.5, .25, .25])
            if kind == "text":
                sc = float(rng.uniform(.9, 1.4) * W / 384)
                x, y0 = int(rng.uniform(.01, .12) * W), int(rng.uniform(.15, .5) * H)
                for i in range(int(rng.integers(2, 5))):
                    t, y = _TEXTS[int(rng.integers(len(_TEXTS)))], y0 + int(i * 22 * H / 384)
                    cv2.putText(cand, t, (x, y), cv2.FONT_HERSHEY_PLAIN, sc, (235, 235, 235), 1, cv2.LINE_AA)
                    (tw, th), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_PLAIN, sc, 1)
                    boxes.append((x, y - th, x + tw, y + 3))
            else:
                w, h = int(rng.uniform(.12, .28) * W), int(rng.uniform(.10, .22) * H)
                x = 0 if rng.random() < .6 else W - w
                cand[H - h:H, x:x + w] = (0, 160, 130) if kind == "inset" else (0, 0, 0)
                boxes.append((x, H - h, x + w, H))
        cover = sum(mask[max(0, b[1]):max(0, b[3]), max(0, b[0]):max(0, b[2])].sum() for b in boxes)
        if cover <= .05 * max(float(mask.sum()), 1.0):
            return cand
    return img


def lab_transfer(x, ref):
    """Colour exchange (as in SANet): give image x the per-channel mean/std of a second image in LAB space. Mask unchanged."""
    a = cv2.cvtColor(x, cv2.COLOR_RGB2LAB).astype(np.float32); b = cv2.cvtColor(ref, cv2.COLOR_RGB2LAB).astype(np.float32)
    a = (a - a.mean((0, 1))) / (a.std((0, 1)) + 1e-6) * b.std((0, 1)) + b.mean((0, 1))
    return cv2.cvtColor(np.clip(a, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


def zoom_out(x, m, rng, lo=0.4, hi=1.0):
    """Shrink the whole picture to a random fraction s of its side (polyp area x s^2) and put it on a black canvas at a random place
    (like the black border of a real endoscope frame). The training set (Kvasir) has large polyps, the test sets small ones."""
    n = x.shape[0]; k = max(32, int(round(n * rng.uniform(lo, hi))))
    xs = cv2.resize(x, (k, k), interpolation=cv2.INTER_AREA); ms = cv2.resize(m, (k, k), interpolation=cv2.INTER_NEAREST)
    cx, cm = np.zeros_like(x), np.zeros_like(m); i, j = int(rng.integers(0, n - k + 1)), int(rng.integers(0, n - k + 1))
    cx[i:i + k, j:j + k] = xs; cm[i:i + k, j:j + k] = ms
    return cx, cm


def area_to_t(a):
    """polyp area fraction -> size target t in [0,1]:  t = (log10(a)+3)/3 clipped (a = 0.1% -> 0, a = 100% -> 1)."""
    return float(np.clip((math.log10(max(a, 1e-3)) + 3.0) / 3.0, 0.0, 1.0))


class PolypDS(Dataset):
    def __init__(self, items, train=False, overlay="none", overlay_p=0.5, seed=0, size=SIZE, lab=False, zoom=False, lab_p=0.5, zoom_p=0.5):
        self.items, self.train, self.overlay, self.p, self.seed, self.size = items, train, overlay, overlay_p, seed, size
        self.lab, self.zoom, self.lab_p, self.zoom_p = lab, zoom, lab_p, zoom_p
        self.reseed(seed)

    def reseed(self, seed):
        self.geo = build_geo(seed, self.size) if self.train else A.Compose([A.Resize(self.size, self.size)])
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        img = cv2.cvtColor(cv2.imread(it["img"]), cv2.COLOR_BGR2RGB)
        mask = (cv2.imread(it["mask"], cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        t = area_to_t(float(mask.mean()))                               # size target from the ORIGINAL-resolution mask
        o = self.geo(image=img, mask=mask)
        x, m = o["image"], o["mask"]
        if self.train and self.lab and self.rng.random() < self.lab_p:          # colour exchange with a random other training picture
            j = int(self.rng.integers(0, len(self.items)))
            x = lab_transfer(x, cv2.resize(cv2.cvtColor(cv2.imread(self.items[j]["img"]), cv2.COLOR_BGR2RGB), (x.shape[1], x.shape[0])))
        if self.train and self.zoom and self.rng.random() < self.zoom_p:
            x, m = zoom_out(x, m, self.rng)
        out = {"mask": torch.from_numpy(m.astype(np.float32))[None], "t": torch.tensor([t], dtype=torch.float32), "idx": i}
        if self.train and self.overlay == "pair":
            out["x"], out["x_ovl"] = to_tensor(x), to_tensor(ui_overlay(x, m, self.rng))
        elif self.train and self.overlay == "aug" and self.rng.random() < self.p:
            out["x"] = to_tensor(ui_overlay(x, m, self.rng))
        else:
            out["x"] = to_tensor(x)
        return out


def worker_init(_):
    info = torch.utils.data.get_worker_info()
    info.dataset.reseed((info.dataset.seed * 1000 + info.id * 17 + info.seed) % (2 ** 31))
