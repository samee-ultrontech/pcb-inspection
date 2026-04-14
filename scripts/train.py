"""Phase 4: fine-tune a YOLOv8 model on the PCB defect dataset.

Starts from Ultralytics pre-trained weights (yolov8n.pt — the 'nano' variant)
so you get the benefit of ImageNet-level features without training from scratch.
The trained weights are saved to models/pcb_yolov8.pt when training finishes.

Usage
-----
    python scripts/train.py
    python scripts/train.py --base-model yolov8s.pt --epochs 100 --device cpu

Models (smallest → largest, faster ↔ more accurate)
-----------------------------------------------------
    yolov8n.pt   nano   ~3 MB    fast, lower accuracy
    yolov8s.pt   small  ~11 MB   good balance for PCB inspection
    yolov8m.pt   medium ~26 MB   better accuracy, slower inference

Run this from the project root so that data/data.yaml resolves correctly.
"""

import argparse
import os
import shutil
import sys

from scripts.config import CONFIDENCE_THRESHOLD, DATASET_YAML, EPOCHS, IMAGE_SIZE, MODEL_PATH


def train(
    base_model: str = "yolov8n.pt",
    data_yaml: str = DATASET_YAML,
    epochs: int = EPOCHS,
    imgsz: int = IMAGE_SIZE,
    device: str = "0",
) -> str:
    """Fine-tune a YOLOv8 model and save the best weights to MODEL_PATH.

    Parameters
    ----------
    base_model : Ultralytics model name or path to existing .pt weights.
                 Use a name (e.g. 'yolov8n.pt') to download pretrained weights.
    data_yaml  : path to the dataset YAML (default: data/data.yaml)
    epochs     : number of training epochs (default: 50)
    imgsz      : image size in pixels — both sides (default: 640)
    device     : '0' for first GPU, 'cpu' for CPU-only training

    Returns
    -------
    str — path to the saved model weights file
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    if not os.path.isfile(data_yaml):
        print(f"ERROR: Dataset YAML not found: {data_yaml}")
        print("       Run scripts/organise_dataset.py first.")
        sys.exit(1)

    print(f"Base model : {base_model}")
    print(f"Dataset    : {data_yaml}")
    print(f"Epochs     : {epochs}")
    print(f"Image size : {imgsz}px")
    print(f"Device     : {device}")
    print()

    # Load the base model (downloads automatically if it is a standard name)
    model = YOLO(base_model)

    # Launch training.
    # results are saved to runs/detect/train/ by default.
    # 'patience=10' stops early if val loss does not improve for 10 epochs.
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        device=device,
        patience=10,
        batch=-1,         # auto-detect largest batch that fits in VRAM
        augment=True,     # random flips, crops, colour jitter — improves generalisation
        verbose=True,
    )

    # Copy the best checkpoint to the project's models/ folder
    best_weights = os.path.join("runs", "detect", "train", "weights", "best.pt")
    if not os.path.isfile(best_weights):
        print(f"WARNING: best.pt not found at {best_weights}. Check the Ultralytics output.")
        return ""

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    shutil.copy2(best_weights, MODEL_PATH)
    print(f"\nBest weights saved to: {MODEL_PATH}")
    return MODEL_PATH


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 on the PCB defect dataset.")
    parser.add_argument(
        "--base-model",
        default="yolov8n.pt",
        help="Starting weights (default: yolov8n.pt — downloads automatically).",
    )
    parser.add_argument(
        "--data",
        default=DATASET_YAML,
        help=f"Path to dataset YAML (default: {DATASET_YAML}).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help=f"Training epochs (default: {EPOCHS}).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=IMAGE_SIZE,
        help=f"Image size (default: {IMAGE_SIZE}).",
    )
    parser.add_argument(
        "--device",
        default="0",
        help="Device: '0' for GPU 0, 'cpu' for CPU (default: '0').",
    )
    args = parser.parse_args()

    train(
        base_model=args.base_model,
        data_yaml=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=args.device,
    )
