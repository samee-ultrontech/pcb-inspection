"""Phase 6b: end-to-end PCB inspection runner.

Chains the full pipeline:
    load → preprocess (align + SSIM) → detect (YOLOv8) → verdict → CSV log

Every inspection appends one row to output/inspection_log.csv so you have a
permanent record of every board that passed through the system.

Usage
-----
    python scripts/inspect.py --image data/query.jpg
    python scripts/inspect.py --image data/query.jpg --reference data/reference.jpg
    python scripts/inspect.py --image data/query.jpg --save-annotated

CSV columns
-----------
    timestamp, image, ssim_score, verdict, num_defects, defects
"""

import argparse
import csv
import os
import sys
from datetime import datetime

import cv2

from scripts.config import (
    CONFIDENCE_THRESHOLD,
    LOG_FILE,
    MODEL_PATH,
    OUTPUT_DIR,
    REFERENCE_IMAGE,
    SSIM_FLAG_THRESHOLD,
)
from scripts.detect import detect_defects
from scripts.preprocess import preprocess_frame
from scripts.verdict import compute_verdict

CSV_COLUMNS = ["timestamp", "image", "ssim_score", "verdict", "num_defects", "defects"]


def _append_csv(log_path: str, row: dict) -> None:
    """Append one result row to the CSV log, creating the file if needed."""
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    file_exists = os.path.isfile(log_path)
    with open(log_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def inspect(
    image_path: str,
    reference_path: str = REFERENCE_IMAGE,
    model_path: str = MODEL_PATH,
    conf: float = CONFIDENCE_THRESHOLD,
    blur_k: int = 5,
    ssim_min: float = SSIM_FLAG_THRESHOLD,
    save_annotated: bool = False,
    log_path: str = LOG_FILE,
) -> dict:
    """Run the full inspection pipeline on a single PCB image.

    Parameters
    ----------
    image_path      : path to the query (board under test) image
    reference_path  : path to the known-good reference image
    model_path      : path to trained YOLOv8 .pt weights
    conf            : YOLO confidence threshold
    blur_k          : GaussianBlur kernel size (must be odd)
    ssim_min        : lower SSIM boundary for FAIL/FLAG split
    save_annotated  : if True, save the annotated aligned image to OUTPUT_DIR
    log_path        : path to the CSV log file

    Returns
    -------
    dict with keys: verdict, reason, ssim_score, num_defects, defects, image_path
    """
    # ── Load images ───────────────────────────────────────────────────────────
    query = cv2.imread(image_path)
    if query is None:
        raise FileNotFoundError(f"Could not load query image: {image_path}")

    reference = cv2.imread(reference_path)
    if reference is None:
        raise FileNotFoundError(
            f"Could not load reference image: {reference_path}\n"
            "Place a known-good PCB photo at data/reference.jpg"
        )

    # ── Step 1–7: align + SSIM ────────────────────────────────────────────────
    print(f"[1/3] Preprocessing: aligning {os.path.basename(image_path)} ...")
    try:
        pre = preprocess_frame(query, reference, blur_k=blur_k, ssim_min=ssim_min)
    except ValueError as exc:
        raise ValueError(f"Preprocessing failed: {exc}") from exc

    print(f"      SSIM = {pre['ssim_score']:.4f}  hint = {pre['verdict_hint']}")

    # ── Step 5: detect defects ────────────────────────────────────────────────
    print("[2/3] Detecting defects ...")
    try:
        detections = detect_defects(pre["bgr_aligned"], model_path=model_path, conf=conf)
    except FileNotFoundError:
        print("      WARNING: model weights not found — skipping YOLO step.")
        print(f"      Train first with: python scripts/train.py")
        detections = []

    print(f"      Found {len(detections)} defect(s)")

    # ── Step 6: verdict ───────────────────────────────────────────────────────
    print("[3/3] Computing verdict ...")
    result = compute_verdict(
        ssim_score=pre["ssim_score"],
        detections=detections,
    )

    # ── Optional: save annotated output image ─────────────────────────────────
    if save_annotated:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        stem = os.path.splitext(os.path.basename(image_path))[0]
        out_path = os.path.join(OUTPUT_DIR, f"{stem}_inspected.jpg")

        annotated = pre["bgr_aligned"].copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = f"{det['class_name']} {det['confidence']:.2f}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                annotated, label,
                (x1, max(y1 - 8, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
            )
        # Stamp the verdict in the top-left corner
        colour = (0, 200, 0) if result["verdict"] == "PASS" else (0, 0, 255)
        cv2.putText(
            annotated, result["verdict"],
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, colour, 2,
        )
        cv2.imwrite(out_path, annotated)
        print(f"      Annotated image saved: {out_path}")

    # ── Log to CSV ────────────────────────────────────────────────────────────
    csv_row = {
        "timestamp":   datetime.now().isoformat(timespec="seconds"),
        "image":       image_path,
        "ssim_score":  f"{pre['ssim_score']:.4f}",
        "verdict":     result["verdict"],
        "num_defects": result["num_defects"],
        "defects":     " ".join(result["defects"]),
    }
    _append_csv(log_path, csv_row)
    print(f"      Logged to: {log_path}")

    # ── Print summary ─────────────────────────────────────────────────────────
    print()
    print(f"  VERDICT : {result['verdict']}")
    print(f"  REASON  : {result['reason']}")

    result["image_path"] = image_path
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-end PCB inspection pipeline.")
    parser.add_argument("--image",      required=True, help="Query PCB image path.")
    parser.add_argument("--reference",  default=REFERENCE_IMAGE, help="Reference image path.")
    parser.add_argument("--model",      default=MODEL_PATH)
    parser.add_argument("--conf",       type=float, default=CONFIDENCE_THRESHOLD)
    parser.add_argument("--blur-k",     type=int,   default=5)
    parser.add_argument("--ssim-min",   type=float, default=SSIM_FLAG_THRESHOLD)
    parser.add_argument(
        "--save-annotated",
        action="store_true",
        help="Save the aligned + annotated image to output/.",
    )
    args = parser.parse_args()

    try:
        inspect(
            image_path=args.image,
            reference_path=args.reference,
            model_path=args.model,
            conf=args.conf,
            blur_k=args.blur_k,
            ssim_min=args.ssim_min,
            save_annotated=args.save_annotated,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
