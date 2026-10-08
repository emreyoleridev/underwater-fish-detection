"""Generate the figures used in report.typ into report/figures/.

    python report/make_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fishdet import Detector, draw  # noqa: E402
from fishdet.augment import blur, color_cast, low_light, turbidity  # noqa: E402
from fishdet.preprocess import clahe, underwater_wb  # noqa: E402

OUT = ROOT / "report" / "figures"
TEST = ROOT / "data" / "yolo" / "test"
NAMES = ["fish", "jellyfish", "penguin", "puffin", "shark", "starfish", "stingray"]

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"], "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})


def fit(img, h=300):
    return cv2.resize(img, (round(img.shape[1] * h / img.shape[0]), h), interpolation=cv2.INTER_AREA)


def gt_dets(stem, shape):
    h, w = shape[:2]
    rows = np.loadtxt(TEST / "labels" / f"{stem}.txt", ndmin=2)
    c, cx, cy, bw, bh = rows.T
    return np.stack([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h,
                     np.ones_like(c), c], 1)


def augmentation_figure():
    img = cv2.imread(str(ROOT / "data" / "aquarium_coco" / "train" / "IMG_2447_jpeg_jpg.rf.607605da7cec106277768435ac4c97f3.jpg"))
    rng = np.random.default_rng(3)
    panels = [("Original", img), ("Colour cast", color_cast(img, rng)), ("Turbidity", turbidity(img, rng)),
              ("Blur", blur(img, rng)), ("Low light + noise", low_light(img, rng)),
              ("Red comp. + WB + CLAHE", clahe(underwater_wb(img)))]
    fig, axs = plt.subplots(2, 3, figsize=(6.3, 4.45))
    for ax, (t, p) in zip(axs.flat, panels):
        ax.imshow(cv2.cvtColor(p, cv2.COLOR_BGR2RGB))
        ax.set_title(t, fontsize=9, pad=3)
        ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.005, wspace=0.03, hspace=0.1)
    fig.savefig(OUT / "augmentation.jpg", dpi=300, pil_kwargs={"quality": 90})
    plt.close(fig)


def training_figure():
    s1 = pd.read_csv(ROOT / "runs" / "y8s_raw_stage1" / "results.csv")
    s2 = pd.read_csv(ROOT / "runs" / "y8s_raw" / "results.csv")
    s2["epoch"] += len(s1)
    df = pd.concat([s1, s2])
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.3, 2.0))
    for col, lab, ls in [("train/box_loss", "train box", "-"), ("train/cls_loss", "train cls", "-"),
                         ("val/box_loss", "val box", "--"), ("val/cls_loss", "val cls", "--")]:
        a.plot(df.epoch, df[col], ls, marker="o", ms=3, lw=1.2, label=lab,
               color="C0" if "box" in col else "C3")
    a.set_xlabel("Epoch"); a.set_ylabel("Loss"); a.set_ylim(0.9, 6.3); a.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper center")
    b.plot(df.epoch, df["metrics/mAP50(B)"], marker="o", ms=3, lw=1.2, label="mAP50", color="C0")
    b.plot(df.epoch, df["metrics/mAP50-95(B)"], marker="s", ms=3, lw=1.2, label="mAP50-95", color="C2")
    b.set_xlabel("Epoch"); b.set_ylabel("Validation mAP"); b.set_ylim(0, 0.8); b.legend(frameon=False, fontsize=8)
    for ax in (a, b):
        ax.axvline(len(s1) + 0.5, color="0.5", lw=0.8, ls=":")
        ax.set_xticks(range(1, len(df) + 1))
    b.text(len(s1) + 0.7, 0.05, "stage 2", fontsize=8, color="0.35")
    b.text(len(s1) + 0.3, 0.05, "stage 1", fontsize=8, color="0.35", ha="right")
    fig.tight_layout(pad=0.3, w_pad=1.5)
    fig.savefig(OUT / "training_curves.png", dpi=300)
    plt.close(fig)


def qualitative_figure():
    det = Detector(str(ROOT / "models" / "y8s_raw.pt"))
    stems = ["IMG_2496_jpeg_jpg.rf.19dcf7cd965f36e764b5a011dc83fe39", "IMG_2354_jpeg_jpg.rf.0db812e028c3a47698df735becc1a1e4",
             "IMG_2465_jpeg_jpg.rf.f03734958c3da4be2eae500da75d1ed4", "IMG_3136_jpeg_jpg.rf.8a62e3beeefb7c4ea5fef373c8d7c50c"]
    names = dict(enumerate(NAMES))
    gts, preds = [], []
    for s in stems:
        img = cv2.imread(str(TEST / "images" / f"{s}.jpg"))
        d, _ = det(img, conf=0.25, iou=0.5)
        gts.append(fit(draw(img, gt_dets(s, img.shape), names)))
        preds.append(fit(draw(img, d, names)))
    cap = cv2.VideoCapture(str(ROOT / "results" / "genoa_aquarium_detected.mp4"))
    frames = []
    for k in (250, 600):
        cap.set(cv2.CAP_PROP_POS_FRAMES, k)
        frames.append(cap.read()[1])
    cap.release()
    fig = plt.figure(figsize=(6.3, 4.3))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1.15], hspace=0.06, wspace=0.03)
    for i in range(4):
        for r, (row, lab) in enumerate([(gts, "Ground truth"), (preds, "YOLOv8s")]):
            ax = fig.add_subplot(gs[r, i])
            ax.imshow(cv2.cvtColor(row[i], cv2.COLOR_BGR2RGB), aspect="auto")
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_ylabel(lab, fontsize=9)
    for j, f in enumerate(frames):
        ax = fig.add_subplot(gs[2, 2 * j:2 * j + 2])
        ax.imshow(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), aspect="auto")
        ax.set_xticks([]); ax.set_yticks([])
        if j == 0:
            ax.set_ylabel("Video", fontsize=9)
    fig.subplots_adjust(left=0.04, right=0.99, top=0.99, bottom=0.01)
    fig.savefig(OUT / "qualitative.jpg", dpi=300, pil_kwargs={"quality": 88})
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    augmentation_figure()
    training_figure()
    qualitative_figure()
    print("figures written to", OUT)
