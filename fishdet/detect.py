"""Thin wrapper around an Ultralytics YOLOv8 model, plus drawing helpers."""
from __future__ import annotations

import time

import cv2
import numpy as np
from ultralytics import YOLO

from .preprocess import preprocess

# Fixed, distinguishable BGR colors per class id.
PALETTE = [(255, 178, 29), (49, 210, 207), (10, 249, 72), (23, 204, 146), (134, 219, 61),
           (52, 147, 26), (187, 212, 0), (168, 153, 44), (255, 194, 0), (147, 69, 52)]


class Detector:
    def __init__(self, weights: str, device: str | None = None):
        self.model = YOLO(weights)
        self.names: dict[int, str] = self.model.names
        self.device = device

    def __call__(self, img: np.ndarray, conf: float = 0.25, iou: float = 0.5, imgsz: int = 640,
                 method: str = "none", classes: list[int] | None = None) -> tuple[np.ndarray, float]:
        """Returns (detections Nx6 [x1, y1, x2, y2, conf, cls], latency in ms incl. preprocessing)."""
        t0 = time.perf_counter()
        x = preprocess(img, method)
        r = self.model.predict(x, conf=conf, iou=iou, imgsz=imgsz, classes=classes,
                               device=self.device, verbose=False)[0]
        dets = r.boxes.data.cpu().numpy() if len(r.boxes) else np.zeros((0, 6), np.float32)
        return dets, (time.perf_counter() - t0) * 1000


def draw(img: np.ndarray, dets: np.ndarray, names: dict[int, str], thickness: int = 2) -> np.ndarray:
    out = img.copy()
    scale = max(out.shape[:2]) / 1000
    for x1, y1, x2, y2, conf, cls in dets:
        c = PALETTE[int(cls) % len(PALETTE)]
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(out, p1, p2, c, thickness)
        label = f"{names[int(cls)]} {conf:.2f}"
        fs = max(0.45, 0.6 * scale)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        y = max(p1[1], th + 6)
        cv2.rectangle(out, (p1[0], y - th - 6), (p1[0] + tw + 6, y), c, -1)
        cv2.putText(out, label, (p1[0] + 3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def hud(img: np.ndarray, text: str) -> np.ndarray:
    cv2.putText(img, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return img
