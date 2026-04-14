"""Phase 5: run YOLOv8 inference on a pre-processed (aligned) PCB image.

Accepts a BGR numpy array — the 'bgr_aligned' output from preprocess_frame —
and returns a clean list of detection dicts.  Each dict contains the class
name, confidence score, and pixel-space bounding box so that verdict.py can
make a PASS / FAIL decision without knowing anything about Ultralytics internals.

Usage (standalone)
------------------
    python scripts/detect.py --image output/aligned.jpg

Typical programmatic use
------------------------
    from scripts.detect import detect_defects

    detections = detect_defects(result["bgr_aligned"])
    # [{'class_id': 0, 'class_name': 'solder_bridge', 'confidence': 0.91,
    #   'bbox': [x1, y1, x2, y2]}, ...]
"""

import argparse
import os
import sys

import cv2
import numpy as np

from scripts.config import CONFIDENCE_THRESHOLD, MODEL_PATH


def detect_defects(
    bgr_image: np.ndarray,
    model_path: str = MODEL_PATH,
    conf: float = CONFIDENCE_THRESHOLD,
) -> list[dict]:
    """Run YOLOv8 inference and return normalised detection results.

    Parameters
    ----------
    bgr_image  : aligned BGR frame from preprocess_frame (numpy array)
    model_path : path to the trained .pt weights file
    conf       : minimum confidence threshold (default from config.py: 0.5)

    Returns
    -------
    List of dicts, one per detection above the confidence threshold:
        class_id    : int   — numeric class index (matches data.yaml)
        class_name  : str   — human-readable defect name
        confidence  : float — model confidence (0.0–1.0)
        bbox        : list  — [x1, y1, x2, y2] pixel coordinates

    Raises
    ------
    TypeError        : if bgr_image is not a numpy array
    FileNotFoundError: if the model weights file does not exist
    """
    if not isinstance(bgr_image, np.ndarray):
        raise TypeError(
            "bgr_image must be a numpy array. "
            "Pass the 'bgr_aligned' output from preprocess_frame."
        )

    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Model weights not found: {model_path}\n"
            "Train the model first with: python scripts/train.py"
        )

    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("ultralytics not installed. Run: pip install ultralytics")

    model = YOLO(model_path)

    # Run inference.
    # verbose=False suppresses the per-frame Ultralytics console log.
    results = model(bgr_image, conf=conf, verbose=False)

    detections = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            class_id   = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append({
                "class_id":   class_id,
                "class_name": class_name,
                "confidence": round(confidence, 4),
                "bbox":       [int(x1), int(y1), int(x2), int(y2)],
            })

    return detections


def _draw_detections(bgr_image: np.ndarray, detections: list[dict]) -> np.ndarray:
    """Draw bounding boxes and labels onto a copy of the image."""
    annotated = bgr_image.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = f"{det['class_name']} {det['confidence']:.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            annotated, label,
            (x1, max(y1 - 8, 0)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
        )
    return annotated


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLOv8 defect detection on a PCB image.")
    parser.add_argument("--image",  required=True, help="Path to the aligned image.")
    parser.add_argument("--model",  default=MODEL_PATH, help="Path to .pt weights.")
    parser.add_argument("--conf",   type=float, default=CONFIDENCE_THRESHOLD)
    parser.add_argument("--output", default="output/detections.jpg")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        print(f"ERROR: Could not load image: {args.image}")
        sys.exit(1)

    try:
        dets = detect_defects(img, model_path=args.model, conf=args.conf)
    except (FileNotFoundError, ImportError) as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    print(f"Detections: {len(dets)}")
    for d in dets:
        print(f"  {d['class_name']:20s}  conf={d['confidence']:.2f}  bbox={d['bbox']}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    annotated = _draw_detections(img, dets)
    cv2.imwrite(args.output, annotated)
    print(f"Annotated image saved to: {args.output}")
