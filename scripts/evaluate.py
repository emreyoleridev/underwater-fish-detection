"""Evaluate trained models on the held-out test split: precision, recall, mAP50, mAP50-95
(overall and per class) and inference speed. Writes results/metrics.json.

    python scripts/evaluate.py y8s_raw:data/yolo/data.yaml y8s_clahe:data/yolo_clahe/data.yaml
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train import pick_device  # noqa: E402


def evaluate(name: str, data: str, split: str = "test") -> dict:
    model = YOLO(ROOT / "models" / f"{name}.pt")
    m = model.val(data=data, split=split, imgsz=640, batch=16, conf=0.001, iou=0.6, device=pick_device(),
                  project=str(ROOT / "runs" / "eval"), name=f"{name}_{split}", exist_ok=True, plots=True, verbose=False)
    names = m.names
    per_class = []
    for i, c in enumerate(m.box.ap_class_index):
        p, r, ap50, ap = m.box.class_result(i)
        per_class.append({"class": names[int(c)], "precision": p, "recall": r, "mAP50": ap50, "mAP50-95": ap})
    p, r, map50, map_ = m.box.mean_results()
    return {"model": name, "split": split, "precision": p, "recall": r, "mAP50": map50, "mAP50-95": map_,
            "f1": 2 * p * r / max(p + r, 1e-9), "speed_ms": m.speed, "per_class": per_class}


if __name__ == "__main__":
    out = ROOT / "results" / "metrics.json"
    results = json.loads(out.read_text()) if out.exists() else {}
    for arg in sys.argv[1:]:
        name, data = arg.split(":", 1)
        for split in ("val", "test"):
            results[f"{name}/{split}"] = evaluate(name, data, split)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=float))
    rows = [{k: v for k, v in r.items() if k not in ("per_class", "speed_ms")} for r in results.values()]
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    for key, r in results.items():
        print(f"\n{key}\n", pd.DataFrame(r["per_class"]).round(3).to_string(index=False))
