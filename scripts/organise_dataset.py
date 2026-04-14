"""Phase 3: organise raw annotated images into the YOLOv8 folder structure.

After annotating images in LabelImg / Roboflow you will have a flat folder of
image files (.jpg / .png) and matching YOLO-format label files (.txt).  Run
this script once to split them into train / val sets and copy them into the
directory layout that data/data.yaml expects.

Usage
-----
    python scripts/organise_dataset.py --raw-dir data/raw --output-dir data

Output layout
-------------
    data/
    ├── images/train/   ← 80 % of images
    ├── images/val/     ← 20 % of images
    ├── labels/train/   ← matching .txt files
    └── labels/val/

Notes
-----
* Images without a matching .txt file are skipped with a warning.
* Existing files in the output directories are NOT deleted — re-run is safe
  as long as filenames are unique.
* Set --seed for a reproducible split (useful when comparing training runs).
"""

import argparse
import os
import random
import shutil
import sys
from pathlib import Path

from scripts.config import TRAIN_SPLIT

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _gather_pairs(raw_dir: Path) -> list[tuple[Path, Path]]:
    """Return (image_path, label_path) pairs from raw_dir.

    Only images that have a matching .txt label file are included.
    Images without a label file are skipped with a warning.
    """
    pairs = []
    for img_path in sorted(raw_dir.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label_path = img_path.with_suffix(".txt")
        if not label_path.exists():
            print(f"WARNING: no label file for {img_path.name} — skipped")
            continue
        pairs.append((img_path, label_path))
    return pairs


def organise_dataset(
    raw_dir: str,
    output_dir: str,
    train_split: float = TRAIN_SPLIT,
    seed: int = 42,
) -> dict:
    """Split annotated images into train / val and copy into the YOLOv8 layout.

    Parameters
    ----------
    raw_dir     : folder containing image files and matching .txt label files
    output_dir  : dataset root (will contain images/ and labels/ sub-folders)
    train_split : fraction of data used for training (default 0.8)
    seed        : random seed for the split (default 42)

    Returns
    -------
    dict with keys:
        train_count : int — number of images placed in the train split
        val_count   : int — number of images placed in the val split
    """
    raw_path = Path(raw_dir)
    out_path = Path(output_dir)

    if not raw_path.is_dir():
        raise FileNotFoundError(f"raw_dir not found: {raw_path.resolve()}")
    if not 0 < train_split < 1:
        raise ValueError(f"train_split must be between 0 and 1, got {train_split}")

    pairs = _gather_pairs(raw_path)
    if not pairs:
        raise ValueError(f"No image+label pairs found in {raw_path.resolve()}")

    # Shuffle with a fixed seed so the split is reproducible
    random.seed(seed)
    random.shuffle(pairs)

    split_idx = int(len(pairs) * train_split)
    train_pairs = pairs[:split_idx]
    val_pairs   = pairs[split_idx:]

    # Create output directories
    for subset in ("train", "val"):
        (out_path / "images" / subset).mkdir(parents=True, exist_ok=True)
        (out_path / "labels" / subset).mkdir(parents=True, exist_ok=True)

    def _copy_pairs(pair_list: list, subset: str) -> None:
        for img_path, label_path in pair_list:
            shutil.copy2(img_path,   out_path / "images" / subset / img_path.name)
            shutil.copy2(label_path, out_path / "labels" / subset / label_path.name)

    _copy_pairs(train_pairs, "train")
    _copy_pairs(val_pairs,   "val")

    print(f"Train : {len(train_pairs)} images → {out_path / 'images' / 'train'}")
    print(f"Val   : {len(val_pairs)}   images → {out_path / 'images' / 'val'}")

    return {"train_count": len(train_pairs), "val_count": len(val_pairs)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Organise raw annotated PCB images into the YOLOv8 folder layout."
    )
    parser.add_argument(
        "--raw-dir",
        required=True,
        help="Folder containing images and matching .txt label files.",
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Dataset root directory (default: data/).",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=TRAIN_SPLIT,
        help=f"Fraction used for training (default: {TRAIN_SPLIT}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible split (default: 42).",
    )
    args = parser.parse_args()

    try:
        organise_dataset(
            raw_dir=args.raw_dir,
            output_dir=args.output_dir,
            train_split=args.train_split,
            seed=args.seed,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
