# Progress Log

## Week 1-2 — Phase 1: Environment Setup

### Completed
- Initialised Git repository and pushed to GitHub
- Created project folder structure
- Set up Python virtual environment
- Installed all dependencies and saved requirements.txt
- Written and tested load_image.py script

### Blockers
None

---

## Week 3-4 — Phase 2: OpenCV Preprocessing Pipeline

### Completed
All 7 pipeline steps implemented in `scripts/preprocess.py`:

1. BGR → Grayscale conversion
2. GaussianBlur noise suppression (odd kernel enforced)
3. Otsu threshold → binary image
4. Canny edge detection on binary images
5. ORB keypoint detection + Lowe's ratio test matching
6. findHomography + warpPerspective → aligned image
7. SSIM scoring → verdict hint (PASS_CANDIDATE / FLAG_CANDIDATE / FAIL_CANDIDATE)

- `scripts/verify_steps.py` — smoke-tests all 7 steps on a synthetic image
- `tests/test_preprocess.py` — 10 tests (input validation + GaussianBlur behaviour)
- 2 tests skipped pending `data/reference.jpg`

### Blockers
None (code complete — real PCB images needed for skipped tests)

---

## Week 5 — Phases 3–6: Dataset Tooling, Training, Inference, Verdict

### Completed
All remaining pipeline code written:

**Phase 3**
- `scripts/organise_dataset.py` — splits annotated images 80/20 into YOLOv8 folder layout
- `data/data.yaml` — Ultralytics dataset config (5 defect classes)

**Phase 4**
- `scripts/train.py` — fine-tunes YOLOv8n on PCB dataset, saves best weights to `models/`

**Phase 5**
- `scripts/detect.py` — runs trained model on aligned image, returns typed detection dicts

**Phase 6**
- `scripts/verdict.py` — combines SSIM score + YOLO detections → PASS / FAIL / FLAG
- `scripts/inspect.py` — end-to-end runner: load → preprocess → detect → verdict → CSV log

**Tests**
- `tests/test_verdict.py` — 18 unit tests covering all verdict rules and boundary values
- `tests/test_detect.py` — 11 tests using `unittest.mock` (no GPU or model file required)

**Total: 37 passing, 2 skipped**

### Blockers
- `data/reference.jpg` (known-good PCB photo) — unblocks 2 skipped tests and runtime use
- Annotated defect images in `data/raw/` — required before training can begin

### Next Steps
1. Photograph a known-good board → `data/reference.jpg`
2. Photograph defective boards → annotate in LabelImg → save to `data/raw/`
3. `python scripts/organise_dataset.py --raw-dir data/raw`
4. `python scripts/train.py`
5. `python scripts/inspect.py --image data/query.jpg --save-annotated`
