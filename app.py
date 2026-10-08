"""Streamlit app: underwater fish detection with YOLOv8.

Run:  streamlit run app.py
"""
import json
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from fishdet import PREPROCESS, Detector, draw, preprocess
from scripts.video_infer import run as run_video

ROOT = Path(__file__).parent
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
RUNS = ROOT / "runs"
TEST_IMAGES = ROOT / "data" / "yolo" / "test" / "images"
VIDEOS = ROOT / "data" / "videos"
MAX_SIDE = 1280

st.set_page_config(page_title="Underwater Fish Detection", page_icon="🐟", layout="wide")


@st.cache_resource(show_spinner="Loading model…")
def load(weights: str) -> Detector:
    return Detector(weights)


def decode(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    s = MAX_SIDE / max(img.shape[:2])
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img


def rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def ground_truth(img_path: Path, img: np.ndarray, names: dict) -> np.ndarray:
    """YOLO label file → Nx6 [x1, y1, x2, y2, 1, cls] in pixels, for drawing."""
    lbl = img_path.parent.parent / "labels" / f"{img_path.stem}.txt"
    if not lbl.exists():
        return np.zeros((0, 6))
    h, w = img.shape[:2]
    rows = []
    for line in lbl.read_text().splitlines():
        c, cx, cy, bw, bh = map(float, line.split())
        rows.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h, 1.0, c])
    return np.array(rows).reshape(-1, 6)


def det_table(dets: np.ndarray, names: dict) -> pd.DataFrame:
    return pd.DataFrame([{"class": names[int(c)], "confidence": round(float(p), 3),
                          "box (x1, y1, x2, y2)": f"{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}"}
                         for x1, y1, x2, y2, p, c in dets])


# --- Sidebar: model + inference settings --------------------------------------------------
st.sidebar.title("🐟 Fish Detection")
weights = sorted(MODELS.glob("*.pt"))
if not weights:
    st.sidebar.error("No trained model in `models/`. Run `python scripts/train.py` first.")
    st.stop()
model_name = st.sidebar.selectbox("Model", [w.stem for w in weights])
det = load(str(MODELS / f"{model_name}.pt"))
names = det.names

conf = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
iou = st.sidebar.slider("NMS IoU threshold", 0.1, 0.9, 0.5, 0.05)
imgsz = st.sidebar.select_slider("Inference size", [320, 416, 512, 640, 800, 960], value=640)
default_pre = "clahe" if "clahe" in model_name else "none"
method = st.sidebar.selectbox("Preprocessing", list(PREPROCESS), index=list(PREPROCESS).index(default_pre),
                              help="Applied before the model. Use the same one the model was trained with.")
picked = st.sidebar.multiselect("Classes", list(names.values()), default=list(names.values()))
classes = [k for k, v in names.items() if v in picked] or None
kw = dict(conf=conf, iou=iou, imgsz=imgsz, method=method, classes=classes)

# --- Main ---------------------------------------------------------------------------------
st.title("Underwater Fish Detection")
st.caption("YOLOv8 trained on the Roboflow Aquarium dataset (7 classes) — detection on images, "
           "videos and webcam, plus the evaluation report (precision, recall, mAP).")

tab_img, tab_vid, tab_cam, tab_eval, tab_about = st.tabs(
    ["🖼️ Image", "🎞️ Video", "📷 Webcam", "📊 Evaluation", "ℹ️ How it works"])

with tab_img:
    src = st.radio("Image source", ["Test set sample", "Upload"], horizontal=True)
    img, path = None, None
    if src == "Upload":
        up = st.file_uploader("Underwater image", type=["png", "jpg", "jpeg", "bmp", "webp"])
        if up:
            img = decode(up.getvalue())
    else:
        samples = sorted(TEST_IMAGES.glob("*.jpg")) if TEST_IMAGES.exists() else []
        if not samples:
            st.warning("Test split not found. Run `python scripts/prepare_dataset.py --download hf`.")
        else:
            i = st.slider("Sample #", 0, len(samples) - 1, 0)
            path = samples[i]
            img = decode(path.read_bytes())

    if img is not None:
        dets, ms = det(img, **kw)
        cols = st.columns(3 if path else 2)
        cols[0].image(rgb(preprocess(img, method)), caption=f"Input ({method})", width="stretch")
        cols[1].image(rgb(draw(img, dets, names)), caption=f"Prediction — {len(dets)} objects, {ms:.0f} ms",
                      width="stretch")
        if path:
            gt = ground_truth(path, img, names)
            cols[2].image(rgb(draw(img, gt, names)), caption=f"Ground truth — {len(gt)} objects", width="stretch")
        if len(dets):
            c1, c2 = st.columns([1, 2])
            c1.dataframe(det_table(dets, names)["class"].value_counts().rename("count"), width="stretch")
            c2.dataframe(det_table(dets, names), width="stretch", hide_index=True)
            st.download_button("⬇️ Download annotated image", cv2.imencode(".png", draw(img, dets, names))[1].tobytes(),
                               "detections.png", "image/png")

with tab_vid:
    vids = sorted(p for p in VIDEOS.glob("*") if p.suffix.lower() in {".mp4", ".mov", ".avi", ".ogv", ".webm", ".mkv"}) \
        if VIDEOS.exists() else []
    vsrc = st.radio("Video source", (["Demo video"] if vids else []) + ["Upload"], horizontal=True)
    vpath = None
    if vsrc == "Upload":
        upv = st.file_uploader("Underwater video", type=["mp4", "mov", "avi", "mkv", "webm"])
        if upv:
            tmp = Path(tempfile.mkdtemp()) / upv.name
            tmp.write_bytes(upv.getvalue())
            vpath = tmp
    else:
        vpath = VIDEOS / st.selectbox("Demo video", [v.name for v in vids])
    max_frames = st.number_input("Max frames to process (0 = all)", 0, 100000, 150, 50)

    if vpath and st.button("▶️ Run detection", type="primary"):
        frame_box, stat_box = st.empty(), st.empty()
        out = Path(tempfile.mkdtemp()) / f"{vpath.stem}_detected.mp4"
        last = [0.0]

        def on_frame(vis, n, dets, ms):
            if time.perf_counter() - last[0] > 0.1:  # throttle UI updates, not inference
                frame_box.image(rgb(vis), caption=f"frame {n}", width="stretch")
                stat_box.caption(f"frame {n} · {len(dets)} objects · {ms:.0f} ms/frame")
                last[0] = time.perf_counter()

        with st.spinner("Detecting…"):
            stats = run_video(vpath, MODELS / f"{model_name}.pt", out, conf=conf, iou=iou, imgsz=imgsz,
                              method=method, max_frames=max_frames or None, on_frame=on_frame)
        frame_box.empty(); stat_box.empty()
        m = st.columns(4)
        m[0].metric("Frames", stats["frames"])
        m[1].metric("Model FPS", f"{stats['model_fps']:.1f}", help="Preprocessing + inference + NMS")
        m[2].metric("End-to-end FPS", f"{stats['end_to_end_fps']:.1f}", help="Incl. decode, drawing, encoding, UI")
        m[3].metric("Objects / frame", f"{stats['mean_objects_per_frame']:.1f}")
        st.video(out.read_bytes())
        if stats["detections_per_class"]:
            st.bar_chart(pd.Series(stats["detections_per_class"], name="detections"), horizontal=True)
        st.download_button("⬇️ Download annotated video", out.read_bytes(), out.name, "video/mp4")

with tab_cam:
    st.caption("Point the camera at a fish photo or an aquarium. For a continuous live stream use "
               "`python scripts/video_infer.py --source 0 --show`.")
    shot = st.camera_input("Take a photo")
    if shot:
        img = decode(shot.getvalue())
        dets, ms = det(img, **kw)
        st.image(rgb(draw(img, dets, names)), caption=f"{len(dets)} objects, {ms:.0f} ms", width="stretch")

with tab_eval:
    mfile = RESULTS / "metrics.json"
    if not mfile.exists():
        st.info("Run `python scripts/evaluate.py` to produce the evaluation report.")
    else:
        res = json.loads(mfile.read_text())
        split = st.radio("Split", ["test", "val"], horizontal=True)
        rows = {k.split("/")[0]: v for k, v in res.items() if k.endswith(f"/{split}")}
        overall = pd.DataFrame({k: {m: v[m] for m in ("precision", "recall", "f1", "mAP50", "mAP50-95")}
                                | {"inference ms/img": v["speed_ms"]["inference"]} for k, v in rows.items()}).T
        st.subheader("Overall")
        st.dataframe(overall.style.format("{:.3f}").highlight_max(axis=0, subset=["precision", "recall", "f1", "mAP50",
                                                                                   "mAP50-95"], color="#2a78d633"),
                     width="stretch")
        st.subheader("Per class")
        pick = st.selectbox("Model", list(rows), index=list(rows).index(model_name) if model_name in rows else 0)
        pc = pd.DataFrame(rows[pick]["per_class"]).set_index("class")
        c1, c2 = st.columns([3, 2])
        c1.bar_chart(pc[["mAP50", "mAP50-95"]], stack=False, horizontal=True)
        c2.dataframe(pc.style.format("{:.3f}"), width="stretch")

        curves = {k: pd.read_csv(RUNS / k / "results.csv") for k in rows if (RUNS / k / "results.csv").exists()}
        if curves:
            st.subheader("Training curves (validation)")
            metric = st.selectbox("Metric", ["metrics/mAP50(B)", "metrics/mAP50-95(B)", "metrics/precision(B)",
                                             "metrics/recall(B)", "val/box_loss", "val/cls_loss"])
            st.line_chart(pd.DataFrame({k: df.set_index("epoch")[metric] for k, df in curves.items()}))

        ev = RUNS / "eval" / f"{pick}_{split}"
        plots = [p for p in ["confusion_matrix_normalized.png", "BoxPR_curve.png", "BoxF1_curve.png"]
                 if (ev / p).exists()]
        if plots:
            st.subheader("Diagnostics")
            for col, p in zip(st.columns(len(plots)), plots):
                col.image(str(ev / p), caption=p.removesuffix(".png").replace("_", " "), width="stretch")

with tab_about:
    stats = ROOT / "data" / "yolo" / "stats.json"
    st.markdown("""
**Pipeline.** Roboflow Aquarium (COCO) → YOLO labels → OpenCV/NumPy preprocessing and offline
underwater augmentation → YOLOv8s fine-tuned from COCO weights → evaluation on the held-out test split
→ frame-by-frame inference on video.

| Step | What it does |
|---|---|
| **Preprocessing** | `clahe`: CLAHE on the L channel of CIELAB. `white_balance`: red-channel compensation + Gray-World. Others combine these with denoising. |
| **Offline augmentation** | One extra copy per training image: horizontal flip + 1–2 of color cast, turbidity veil, Gaussian/motion blur, low light + sensor noise. |
| **Online augmentation** | Ultralytics defaults: mosaic, HSV jitter, scale/translate, flips (mosaic off for the last 10 epochs). |
| **Metrics** | Precision/recall at the F1-optimal confidence, mAP@0.5 and mAP@0.5:0.95 (COCO-style). |
""")
    if stats.exists():
        s = json.loads(stats.read_text())
        st.dataframe(pd.DataFrame({k: v["boxes"] | {"images": v["images"]} for k, v in s["splits"].items()}).T,
                     width="stretch")
