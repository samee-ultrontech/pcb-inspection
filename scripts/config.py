"""Project-wide configuration constants for the YOLOv8 PCB Inspection System.

This module centralises all tunable parameters and file paths used across the
inspection pipeline. Import constants from here rather than hardcoding values
in individual scripts, so that changes only need to be made in one place.
"""

# Input image size for YOLOv8 inference
IMAGE_SIZE = 640

# Minimum confidence score for a detection to be considered valid
CONFIDENCE_THRESHOLD = 0.5

# SSIM score required for a board to receive a PASS verdict
SSIM_PASS_THRESHOLD = 0.85

# SSIM score below which a board is flagged for manual review
SSIM_FLAG_THRESHOLD = 0.80

# Folder where annotated output images are saved
OUTPUT_DIR = "output"

# Path to the CSV file used to log inspection results
LOG_FILE = "output/inspection_log.csv"

# Path to the reference (known-good) PCB image used for SSIM comparison
REFERENCE_IMAGE = "data/reference.jpg"

# Path to the trained YOLOv8 weights file produced by train.py
MODEL_PATH = "models/pcb_yolov8.pt"

# Path to the dataset YAML consumed by YOLOv8 during training
DATASET_YAML = "data/data.yaml"

# Train/validation split ratio (80 % train, 20 % val)
TRAIN_SPLIT = 0.8

# Number of training epochs
EPOCHS = 50

# Defect class names (must match the order in data/data.yaml)
DEFECT_CLASSES = [
    "solder_bridge",
    "missing_solder",
    "cold_joint",
    "lifted_lead",
    "excess_solder",
]
