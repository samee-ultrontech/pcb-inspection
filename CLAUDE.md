# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

YOLOv8-based automated visual inspection pipeline for PCBs (printed circuit boards) after wave soldering. The system preprocesses images, aligns a query board against a known-good reference using homography, computes SSIM, and ultimately produces a PASS/FAIL verdict. YOLOv8 inference for defect detection is planned for Phase 4–5 (not yet implemented).

## Common Commands

### Setup
```bash
python -m venv venv
venv/Scripts/activate        # Windows
pip install -r requirements.txt
```

### Run scripts
```bash
# Load and inspect an image
python scripts/load_image.py data/tiger.jpg

# Run the 7-step preprocessing pipeline
python scripts/preprocess.py --image data/query.jpg --reference data/reference.jpg --blur-k 5 --ssim-min 0.80 --output output/aligned.jpg

# Verify all preprocessing steps with a synthetic PCB image (outputs debug images to output/verify_*.jpg)
python scripts/verify_steps.py
```

### Tests
```bash
# Run all tests
pytest tests/ -v

# Run a single test
pytest tests/test_preprocess.py::test_even_blur_k_raises_valueerror -v
```

## Architecture

### Pipeline flow
```
Image input (data/)
  → scripts/load_image.py        # Load + validate numpy array
  → scripts/preprocess.py        # 7-step OpenCV pipeline
      1. BGR → Grayscale
      2. GaussianBlur (kernel must be odd)
      3. Otsu threshold → binary (query + reference)
      4. Canny edge detection on binary images (query + reference)
      5. ORB keypoint detection on edge maps + Lowe's ratio test matching
      6. findHomography + warpPerspective → bgr_aligned, gray_aligned
      7. compute_ssim → ssim_score + verdict_hint
  → verdict_hint: PASS_CANDIDATE (≥0.85) / FLAG_CANDIDATE (0.80–0.85) / FAIL_CANDIDATE (<0.80)
  → [Phase 4–5] YOLOv8 inference on aligned image → defect bounding boxes
  → [Phase 6] Final PASS/FAIL verdict + CSV log
```

### Key files
- **scripts/config.py** — All thresholds and paths in one place (`SSIM_PASS_THRESHOLD`, `SSIM_FLAG_THRESHOLD`, `IMAGE_SIZE`, `CONFIDENCE_THRESHOLD`, `OUTPUT_DIR`, `REFERENCE_IMAGE`). Change behaviour here first.
- **scripts/preprocess.py** — Core function `preprocess_frame(bgr_query, bgr_reference, blur_k, ssim_min)` returns a dict: `{bgr_aligned, gray_aligned, ssim_score, verdict_hint}`. Accepts numpy arrays only, not file paths.
- **scripts/load_image.py** — `load_image(path)` wraps `cv2.imread`, validates the result, and copies to `output/`.
- **tests/test_preprocess.py** — 10 passing tests (input validation + Step 2 GaussianBlur behaviour + UT-PP-02); 2 skipped awaiting real PCB images (`data/reference.jpg`).

### Data & model directories
- `data/` — Raw input images. `data/reference.jpg` (known-good board) is required at runtime but not committed.
- `models/` — YOLOv8 `.pt` weight files (not committed, excluded by .gitignore).
- `output/` — All generated images and `inspection_log.csv` (excluded by .gitignore except `.gitkeep`).

## Development Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Project setup, image loading | Complete |
| 2 | OpenCV preprocessing pipeline | Complete |
| 3 | Dataset collection & YOLOv8 annotation | Pending |
| 4 | YOLOv8 model training | Pending |
| 5 | Defect detection inference | Pending |
| 6 | PASS/FAIL verdict logic & CSV reporting | Pending |

All 7 Phase 2 steps are complete: grayscale conversion, GaussianBlur, Otsu threshold, Canny edge detection, ORB keypoint matching (Lowe's ratio test), findHomography + warpPerspective, and SSIM scoring with verdict_hint.

The 2 remaining skipped pytest tests will be unblocked once `data/reference.jpg` (a real known-good PCB image) is available.
