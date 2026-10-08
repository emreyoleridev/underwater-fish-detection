"""Convert the Roboflow Aquarium dataset (COCO export) to YOLO format, with optional
preprocessing and offline underwater augmentation of the training split.

Get the data either from Roboflow (needs ROBOFLOW_API_KEY) or from the public RF100 mirror:
    python scripts/prepare_dataset.py --download hf
    python scripts/prepare_dataset.py --download roboflow     # uses brad-dwyer/aquarium-combined
    python scripts/prepare_dataset.py --preprocess clahe --aug-copies 1 --out data/yolo_clahe
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tarfile
import urllib.request
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fishdet import augment, preprocess  # noqa: E402

COCO_DIR = ROOT / "data" / "aquarium_coco"
HF_URL = "https://huggingface.co/datasets/Francesco/aquarium-qlnqy/resolve/main/dataset.tar.gz"
SPLITS = {"train": "train", "valid": "val", "test": "test"}


def download_hf():
    dst = ROOT / "data" / "download"
    dst.mkdir(parents=True, exist_ok=True)
    tgz = dst / "dataset.tar.gz"
    if not tgz.exists():
        print("downloading", HF_URL)
        urllib.request.urlretrieve(HF_URL, tgz)
    with tarfile.open(tgz) as tf:
        members = [m for m in tf.getmembers() if m.isfile() and "/aquarium-qlnqy/" in m.name]
        for m in members:
            rel = m.name.split("/aquarium-qlnqy/", 1)[1]
            out = COCO_DIR / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            with tf.extractfile(m) as f, open(out, "wb") as g:
                shutil.copyfileobj(f, g)


def download_roboflow():
    from roboflow import Roboflow
    rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
    ds = rf.workspace("brad-dwyer").project("aquarium-combined").version(2).download("coco", location=str(COCO_DIR))
    print("downloaded to", ds.location)


def convert(out: Path, method: str, aug_copies: int, seed: int):
    rng = np.random.default_rng(seed)
    if out.exists():
        shutil.rmtree(out)
    names, stats = None, {}
    for src, split in SPLITS.items():
        coco = json.loads((COCO_DIR / src / "_annotations.coco.json").read_text())
        # Category 0 ("aquarium") is Roboflow's supercategory and has no boxes: drop it.
        cats = sorted((c for c in coco["categories"] if c["supercategory"] != "none"), key=lambda c: c["id"])
        cat_map = {c["id"]: i for i, c in enumerate(cats)}
        names = names or [c["name"] for c in cats]
        anns: dict[int, list] = {}
        for a in coco["annotations"]:
            if a["category_id"] in cat_map:
                anns.setdefault(a["image_id"], []).append(a)
        (out / split / "images").mkdir(parents=True)
        (out / split / "labels").mkdir(parents=True)
        counts, n_img = Counter(), 0
        for im in coco["images"]:
            img = cv2.imread(str(COCO_DIR / src / im["file_name"]))
            h, w = img.shape[:2]
            rows = []
            for a in anns.get(im["id"], []):
                x, y, bw, bh = a["bbox"]
                if bw < 1 or bh < 1:
                    continue
                rows.append([cat_map[a["category_id"]], (x + bw / 2) / w, (y + bh / 2) / h, bw / w, bh / h])
            boxes = np.clip(np.array(rows, np.float32).reshape(-1, 5), 0, None)
            counts.update(int(c) for c in boxes[:, 0])
            stem = Path(im["file_name"]).stem
            copies = [(stem, img, boxes)]
            if split == "train":
                copies += [(f"{stem}_aug{k}", *augment(img, boxes, rng)) for k in range(aug_copies)]
            for name, im_, bx in copies:
                cv2.imwrite(str(out / split / "images" / f"{name}.jpg"), preprocess(im_, method), [cv2.IMWRITE_JPEG_QUALITY, 95])
                (out / split / "labels" / f"{name}.txt").write_text(
                    "".join(f"{int(c)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n" for c, cx, cy, bw, bh in bx))
                n_img += 1
        stats[split] = {"images": n_img, "source_images": len(coco["images"]),
                        "boxes": {names[k]: v for k, v in sorted(counts.items())}}
    yaml = f"path: {out.resolve()}\ntrain: train/images\nval: val/images\ntest: test/images\nnames:\n"
    yaml += "".join(f"  {i}: {n}\n" for i, n in enumerate(names))
    (out / "data.yaml").write_text(yaml)
    meta = {"preprocess": method, "aug_copies": aug_copies, "names": names, "splits": stats}
    (out / "stats.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--download", choices=["hf", "roboflow"], help="fetch the COCO export first")
    p.add_argument("--preprocess", default="none", help="one of fishdet.PREPROCESS")
    p.add_argument("--aug-copies", type=int, default=1, help="augmented copies per training image")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "yolo")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    if a.download == "hf":
        download_hf()
    elif a.download == "roboflow":
        download_roboflow()
    convert(a.out, a.preprocess, a.aug_copies, a.seed)
