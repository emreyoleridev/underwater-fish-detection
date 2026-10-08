"""Offline, box-aware data augmentation with OpenCV + NumPy.

Boxes are YOLO-normalized arrays of shape (N, 5): class, cx, cy, w, h.
Ultralytics adds its own online augmentation (mosaic, HSV, flips) on top of this during training;
these transforms target underwater-specific variation: color cast, turbidity, blur and low light.
"""
from __future__ import annotations

import cv2
import numpy as np


def hflip(img, boxes):
    boxes = boxes.copy()
    boxes[:, 1] = 1.0 - boxes[:, 1]
    return img[:, ::-1].copy(), boxes


def color_cast(img, rng):
    """Shift toward blue/green, as different depths and water types do."""
    gains = np.array([rng.uniform(1.0, 1.25), rng.uniform(0.95, 1.2), rng.uniform(0.6, 0.95)], np.float32)
    return np.clip(img.astype(np.float32) * gains, 0, 255).astype(np.uint8)


def turbidity(img, rng):
    """Blend toward a hazy veil color: a cheap stand-in for the scattering model I = J·t + A·(1−t)."""
    t = rng.uniform(0.6, 0.9)
    veil = np.array([rng.uniform(120, 200), rng.uniform(120, 190), rng.uniform(40, 100)], np.float32)
    return np.clip(img.astype(np.float32) * t + veil * (1 - t), 0, 255).astype(np.uint8)


def blur(img, rng):
    k = int(rng.choice([3, 5]))
    return cv2.GaussianBlur(img, (k, k), 0) if rng.random() < 0.5 else _motion_blur(img, k + 2)


def _motion_blur(img, k):
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2] = 1.0 / k
    return cv2.filter2D(img, -1, kernel)


def low_light(img, rng):
    g = rng.uniform(1.4, 2.0)
    lut = (np.linspace(0, 1, 256) ** g * 255).astype(np.uint8)
    out = cv2.LUT(img, lut)
    noise = rng.normal(0, rng.uniform(3, 8), img.shape)
    return np.clip(out + noise, 0, 255).astype(np.uint8)


PHOTOMETRIC = [color_cast, turbidity, blur, low_light]


def augment(img: np.ndarray, boxes: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """One random augmented copy: optional flip + one or two underwater photometric transforms."""
    if rng.random() < 0.5:
        img, boxes = hflip(img, boxes)
    for i in rng.choice(len(PHOTOMETRIC), size=rng.integers(1, 3), replace=False):
        img = PHOTOMETRIC[i](img, rng)
    return img, boxes
