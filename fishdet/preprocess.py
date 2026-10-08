"""Image preprocessing for underwater frames. All functions take and return uint8 BGR images."""
from __future__ import annotations

import cv2
import numpy as np


def clahe(img: np.ndarray, clip_limit: float = 2.0, tile: int = 8) -> np.ndarray:
    """CLAHE on the L channel of CIELAB: local contrast boost without shifting colors."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def underwater_wb(img: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Red-channel compensation (Ancuti et al., 2018) followed by Gray-World white balance.
    Water absorbs red first, so red is rebuilt from the better-preserved green channel."""
    f = img.astype(np.float32) / 255.0
    b, g, r = cv2.split(f)
    r = r + alpha * (g.mean() - r.mean()) * (1.0 - r) * g
    f = np.clip(cv2.merge([b, g, r]), 0, 1)
    means = f.reshape(-1, 3).mean(axis=0)
    f *= means.mean() / np.maximum(means, 1e-6)
    return (np.clip(f, 0, 1) * 255).astype(np.uint8)


def denoise(img: np.ndarray) -> np.ndarray:
    """Edge-preserving bilateral filter to suppress backscatter / sensor noise."""
    return cv2.bilateralFilter(img, d=5, sigmaColor=40, sigmaSpace=40)


def wb_clahe(img: np.ndarray) -> np.ndarray:
    return clahe(underwater_wb(img))


PREPROCESS = {
    "none": lambda img: img,
    "clahe": clahe,
    "white_balance": underwater_wb,
    "wb+clahe": wb_clahe,
    "denoise+clahe": lambda img: clahe(denoise(img)),
}


def preprocess(img: np.ndarray, method: str = "none") -> np.ndarray:
    return PREPROCESS[method](img)


def letterbox(img: np.ndarray, size: int = 640, color: int = 114) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize keeping aspect ratio and pad to a square. Returns (image, scale, (pad_x, pad_y))."""
    h, w = img.shape[:2]
    s = size / max(h, w)
    nh, nw = round(h * s), round(w * s)
    out = np.full((size, size, 3), color, np.uint8)
    px, py = (size - nw) // 2, (size - nh) // 2
    out[py:py + nh, px:px + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return out, s, (px, py)
