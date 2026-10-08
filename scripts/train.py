"""Train YOLOv8 on the prepared dataset. Runs on CUDA (Colab), Apple MPS or CPU.

    python scripts/train.py --data data/yolo/data.yaml --model yolov8s.pt --epochs 80 --name y8s_raw
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def pick_device() -> str:
    if torch.cuda.is_available():
        return "0"
    return "mps" if torch.backends.mps.is_available() else "cpu"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(ROOT / "data" / "yolo" / "data.yaml"))
    p.add_argument("--model", default="yolov8s.pt")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--name", default="train")
    p.add_argument("--device", default=None)
    a = p.parse_args()

    model = YOLO(a.model)
    model.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device or pick_device(),
                project=str(ROOT / "runs"), name=a.name, exist_ok=True, patience=25, seed=0,
                cos_lr=True, close_mosaic=10, plots=True, workers=4)
    best = ROOT / "runs" / a.name / "weights" / "best.pt"
    (ROOT / "models").mkdir(exist_ok=True)
    shutil.copy(best, ROOT / "models" / f"{a.name}.pt")
    print("saved", ROOT / "models" / f"{a.name}.pt")
