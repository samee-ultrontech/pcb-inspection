# YOLOv8 Automated PCB Post-Wave Inspection System

## Project Summary

This project implements an automated visual inspection system for printed circuit boards (PCBs) following the wave soldering stage of manufacturing. Using a YOLOv8 object detection model and OpenCV for image preprocessing, the system analyzes PCB images to identify common soldering defects — such as bridges, cold joints, and missing components — and outputs a final **PASS**, **FAIL**, or **FLAG** verdict for each board. The pipeline is modular and extensible, allowing new defect classes and camera configurations to be integrated with minimal changes.

---

## Tech Stack

| Technology | Version |
|---|---|
| Python | 3.10+ |
| OpenCV | 4.8+ |
| YOLOv8 (Ultralytics) | Latest |
| PyTorch | 2.0+ |
| scikit-image | Latest |
| pandas | Latest |

---

## Project Structure

```
pcb-inspection/
├── data/
│   ├── data.yaml           # YOLOv8 dataset config (classes, train/val paths)
│   ├── reference.jpg       # Known-good PCB image (required at runtime, not committed)
│   ├── raw/                # Place annotated images here before running organise_dataset.py
│   ├── images/
│   │   ├── train/          # Created by organise_dataset.py
│   │   └── val/
│   └── labels/
│       ├── train/          # YOLO .txt annotation files
│       └── val/
├── models/
│   └── pcb_yolov8.pt       # Trained weights (created by train.py, not committed)
├── notebooks/              # Exploratory Jupyter notebooks
├── output/                 # Annotated images and inspection_log.csv (not committed)
├── scripts/
│   ├── config.py           # All thresholds and paths — change behaviour here
│   ├── load_image.py       # Load + validate a single image
│   ├── preprocess.py       # 7-step OpenCV alignment pipeline (SSIM scoring)
│   ├── organise_dataset.py # Split annotated images into train/val folders
│   ├── train.py            # Fine-tune YOLOv8 on the PCB defect dataset
│   ├── detect.py           # Run trained model, return defect bounding boxes
│   ├── verdict.py          # Combine SSIM + YOLO detections → PASS/FAIL/FLAG
│   ├── inspect.py          # End-to-end runner with CSV logging
│   └── verify_steps.py     # Smoke-test every pipeline step on a synthetic image
├── tests/
│   ├── test_preprocess.py  # 10 tests for the preprocessing pipeline
│   ├── test_verdict.py     # 18 tests for verdict logic
│   └── test_detect.py      # 11 tests for defect detection (mocked YOLO)
├── CLAUDE.md
├── PROGRESS.md
├── README.md
└── requirements.txt
```

---

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/samee-ultrontech/pcb-inspection.git
cd pcb-inspection

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt
```

---

## How to Run

### Full inspection (end-to-end)

```bash
python scripts/inspect.py --image data/query.jpg --save-annotated
```

Requires `data/reference.jpg` (known-good board) and `models/pcb_yolov8.pt` (trained weights).  
Outputs a verdict to the console, saves an annotated image to `output/`, and appends a row to `output/inspection_log.csv`.

---

### Individual pipeline steps

```bash
# Load and validate an image
python scripts/load_image.py data/tiger.jpg

# Run the preprocessing pipeline only (alignment + SSIM)
python scripts/preprocess.py --image data/query.jpg --reference data/reference.jpg

# Verify all 7 preprocessing steps on a synthetic image
python scripts/verify_steps.py

# Organise annotated images into the YOLOv8 folder layout
python scripts/organise_dataset.py --raw-dir data/raw

# Train the YOLOv8 model
python scripts/train.py

# Run defect detection on an already-aligned image
python scripts/detect.py --image output/aligned.jpg
```

---

### Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test files
pytest tests/test_verdict.py -v
pytest tests/test_detect.py -v
```

**Current test status:** 39 tests — 37 passing, 2 skipped (require `data/reference.jpg`).

---

## Pipeline Overview

```
data/query.jpg  +  data/reference.jpg
        │
        ▼
  preprocess.py   — grayscale → blur → Otsu → Canny → ORB → homography → SSIM
        │
        ├── ssim_score  ──────────────────────────────────────────────┐
        └── bgr_aligned                                               │
                │                                                     │
                ▼                                                     │
          detect.py  — YOLOv8 inference → [{class, conf, bbox}, ...]  │
                │                                                     │
                └──────────────────────┐                             │
                                       ▼                             ▼
                                   verdict.py  — PASS / FAIL / FLAG
                                       │
                                       ▼
                              inspect.py  — annotated image + CSV log
```

### Verdict logic

| SSIM score | Defects found | Verdict |
|---|---|---|
| < 0.80 | any | FAIL |
| 0.80 – 0.85 | any | FLAG (manual review) |
| ≥ 0.85 | yes | FAIL |
| ≥ 0.85 | no | PASS |

### Defect classes

| ID | Class | Description |
|---|---|---|
| 0 | solder_bridge | Two pads shorted by excess solder |
| 1 | missing_solder | Pad has no solder at all |
| 2 | cold_joint | Dull, grainy joint — poor electrical contact |
| 3 | lifted_lead | Component pin not touching the pad |
| 4 | excess_solder | Solder blob — not a bridge but excessive |

---

## Project Status

| Phase | Description | Status |
|---|---|---|
| 1 | Project setup, folder structure, image loading | ✅ Complete |
| 2 | OpenCV preprocessing pipeline (7 steps, SSIM scoring) | ✅ Complete |
| 3 | Dataset collection and annotation | ⏳ Awaiting data |
| 4 | YOLOv8 model training | ⏳ Awaiting data |
| 5 | Defect detection inference pipeline | ✅ Code complete |
| 6 | PASS/FAIL verdict logic and CSV reporting | ✅ Code complete |
