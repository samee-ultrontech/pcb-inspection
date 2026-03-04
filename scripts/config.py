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
