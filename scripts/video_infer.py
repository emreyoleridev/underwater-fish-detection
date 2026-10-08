"""Run the detector on a video file or webcam and write an annotated MP4 + per-frame stats.

    python scripts/video_infer.py --source data/videos/demo.mp4 --weights models/y8s_raw.pt
    python scripts/video_infer.py --source 0          # webcam, shows a live window
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fishdet import Detector, draw  # noqa: E402
from fishdet.detect import hud  # noqa: E402


def h264_writer(path, size, fps):
    """Browser-playable H.264 MP4 writer (OpenCV's own mp4v output does not play in browsers)."""
    w = imageio_ffmpeg.write_frames(str(path), size, fps=fps, codec="libx264", pix_fmt_out="yuv420p",
                                    macro_block_size=2, quality=7, ffmpeg_log_level="error")
    w.send(None)
    return w


def run(source, weights, out=None, conf=0.25, iou=0.5, imgsz=640, method="none", show=False,
        max_frames=None, on_frame=None) -> dict:
    det = Detector(str(weights))
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else str(source))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = h264_writer(out, (w, h), fps_in) if out else None
    lat, counts, per_frame, n = [], Counter(), [], 0
    t_start = time.perf_counter()
    while max_frames is None or n < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        dets, ms = det(frame, conf=conf, iou=iou, imgsz=imgsz, method=method)
        lat.append(ms)
        frame_counts = Counter(det.names[int(c)] for c in dets[:, 5])
        counts.update(frame_counts)
        per_frame.append(sum(frame_counts.values()))
        vis = hud(draw(frame, dets, det.names), f"{1000 / np.mean(lat[-30:]):.1f} FPS | {len(dets)} objects")
        if writer:
            writer.send(np.ascontiguousarray(vis[..., ::-1]))
        if on_frame:
            on_frame(vis, n, dets, ms)
        if show:
            cv2.imshow("fish detection", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        n += 1
    wall = time.perf_counter() - t_start
    cap.release()
    if writer:
        writer.close()
    if show:
        cv2.destroyAllWindows()
    return {"frames": n, "resolution": [w, h], "mean_latency_ms": float(np.mean(lat)) if lat else 0,
            "p95_latency_ms": float(np.percentile(lat, 95)) if lat else 0,
            "model_fps": 1000 / float(np.mean(lat)) if lat else 0, "end_to_end_fps": n / wall if wall else 0,
            "detections_per_class": dict(counts), "mean_objects_per_frame": float(np.mean(per_frame)) if per_frame else 0}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--weights", default=str(ROOT / "models" / "y8s_raw.pt"))
    p.add_argument("--out", default=None)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--preprocess", default="none")
    p.add_argument("--show", action="store_true")
    a = p.parse_args()
    out = a.out or (None if str(a.source).isdigit() else str(ROOT / "results" / f"{Path(a.source).stem}_detected.mp4"))
    stats = run(a.source, a.weights, out, a.conf, method=a.preprocess, show=a.show)
    print(json.dumps(stats, indent=2))
    if out:
        Path(out).with_suffix(".json").write_text(json.dumps(stats, indent=2))
