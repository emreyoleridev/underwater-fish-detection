# Underwater Fish Detection

YOLOv8 object detection for underwater footage, trained on the Roboflow **Aquarium** dataset
(7 classes: fish, jellyfish, penguin, puffin, shark, starfish, stingray), with OpenCV/NumPy
preprocessing and underwater-specific augmentation, evaluation with precision / recall / mAP,
real-time video inference and a Streamlit control panel.

**Live demo:** [https://emreyoleridev-underwater-fish-detection-app-q9aaiy.streamlit.app/](https://emreyoleridev-underwater-fish-detection-app-q9aaiy.streamlit.app/)

## Results

YOLOv8s fine-tuned from COCO weights (short run: 4 + 6 epochs on an Apple M3 Pro), held-out test split (63 images, 582 boxes):

| Precision | Recall | F1 | mAP50 | mAP50-95 | Video speed (1280×720, M3 Pro) |
|---|---|---|---|---|---|
| 0.752 | 0.654 | 0.699 | 0.727 | 0.414 | ~15 FPS |

| Class | fish | jellyfish | penguin | puffin | shark | starfish | stingray |
|---|---|---|---|---|---|---|---|
| mAP50 | 0.744 | 0.847 | 0.718 | 0.524 | 0.815 | 0.922 | 0.520 |

The model had not converged (validation mAP was still rising), so a full 50-epoch run should do better.

## Layout

```
fishdet/            preprocessing (CLAHE, underwater white balance), augmentation, detector wrapper
scripts/
  prepare_dataset.py   download (HF mirror or Roboflow) + COCO → YOLO + preprocessing + offline augmentation
  train.py             YOLOv8 training (CUDA / Apple MPS / CPU)
  evaluate.py          precision, recall, mAP50, mAP50-95, per class → results/metrics.json
  video_infer.py       annotated MP4 + FPS stats from a video file or webcam
notebooks/colab_train.ipynb   the same pipeline on a Colab GPU
app.py              Streamlit app
models/             trained weights (best.pt of each run)
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/prepare_dataset.py --download hf                         # data/yolo
python scripts/prepare_dataset.py --preprocess clahe --out data/yolo_clahe

python scripts/train.py --data data/yolo/data.yaml --name y8s_raw --epochs 50
python scripts/train.py --data data/yolo_clahe/data.yaml --name y8s_clahe --epochs 50
python scripts/evaluate.py y8s_raw:data/yolo/data.yaml y8s_clahe:data/yolo_clahe/data.yaml

python scripts/video_infer.py --source data/videos/genoa_aquarium_15s.mp4   # or --source 0 --show for webcam
streamlit run app.py
```

With a Roboflow API key you can pull the dataset straight from Roboflow:
`ROBOFLOW_API_KEY=... python scripts/prepare_dataset.py --download roboflow`.

## Streamlit app

- **Image** — test-set samples (prediction next to ground truth) or your own upload
- **Video** — demo clip or upload; live preview while processing, FPS, per-class counts, annotated MP4 download
- **Webcam** — snapshot detection
- **Evaluation** — overall and per-class metrics, training curves, confusion matrix, PR / F1 curves
- Sidebar: model, confidence and NMS thresholds, input size, preprocessing, class filter

Data: Roboflow Aquarium via the RF100 mirror on Hugging Face (`Francesco/aquarium-qlnqy`).
Demo video: see `data/videos/CREDITS.md`.
