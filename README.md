# YOLOv8 Automated PCB Post-Wave Inspection System

## Project Summary

This project implements an automated visual inspection system for printed circuit boards (PCBs) following the wave soldering stage of manufacturing. Using a YOLOv8 object detection model and OpenCV for image preprocessing, the system analyzes PCB images to identify common soldering defects — such as bridges, cold joints, and missing components — and outputs a final **PASS** or **FAIL** verdict for each board. The pipeline is designed to be modular and extensible, allowing new defect classes and camera configurations to be integrated with minimal changes.

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
| scikit-learn | Latest |

---

## Project Structure

```
pcb-inspection/
├── data/               # Raw and processed PCB images
├── models/             # Trained YOLOv8 model weights (.pt files)
├── notebooks/          # Exploratory Jupyter notebooks
├── output/             # Generated inspection results and annotated images
├── scripts/            # Python scripts for each pipeline stage
│   └── load_image.py   # Image loading and validation utility
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Setup Instructions

```bash
# 1. Clone the repository
git clone https://github.com/your-username/pcb-inspection.git
cd pcb-inspection

# 2. Create a virtual environment
python -m venv venv

# 3. Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

---

## How to Run

### Load and inspect an image

```bash
python scripts/load_image.py data/your-image.jpg
```

**Example output:**

```
Resolution : 1920 x 1080
Channels   : 3
Saved copy  : C:\...\pcb-inspection\output\your-image.jpg
```

This will print the image resolution and channel count, and save a copy of the image to the `output/` folder.

---

## Project Status

| Phase | Description | Status |
|---|---|---|
| 1 | Project setup, folder structure, and image loading | ✅ Complete |
| 2 | Image preprocessing (grayscale, thresholding, noise reduction) | 🔲 Pending |
| 3 | Dataset collection and annotation for YOLOv8 training | 🔲 Pending |
| 4 | YOLOv8 model training and validation | 🔲 Pending |
| 5 | Defect detection inference pipeline | 🔲 Pending |
| 6 | PASS/FAIL verdict logic and results reporting | 🔲 Pending |
