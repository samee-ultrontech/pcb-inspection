"""Script to load an image, display its properties, and save a copy to the output folder."""

import argparse
import os
import shutil
import sys

import cv2


def load_image(image_path: str) -> cv2.Mat:
    """Load an image from disk, print its properties, and save a copy to the output folder.

    Accepts a file path to an image, validates that the file exists, loads it using
    OpenCV, prints the resolution and channel count, and saves a copy of the image
    to the ``output/`` folder located relative to the project root.

    Args:
        image_path: Path to the image file to load. Can be relative (e.g.
            ``data/your-image.jpg``) or absolute.

    Returns:
        The loaded image as a NumPy array (BGR format, as returned by cv2.imread).

    Raises:
        FileNotFoundError: If no file exists at the provided ``image_path``.
        ValueError: If OpenCV fails to decode the file as a valid image.
    """
    # Resolve to an absolute path so error messages are unambiguous.
    abs_path = os.path.abspath(image_path)

    if not os.path.isfile(abs_path):
        raise FileNotFoundError(f"Image file not found: '{abs_path}'")

    image = cv2.imread(abs_path)
    if image is None:
        raise ValueError(
            f"OpenCV could not decode the file as an image: '{abs_path}'"
        )

    height, width = image.shape[:2]
    channels = image.shape[2] if image.ndim == 3 else 1

    print(f"Resolution : {width} x {height}")
    print(f"Channels   : {channels}")

    # Determine the output folder relative to this script's parent directory
    # (i.e. the project root, one level above scripts/).
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, "output")
    os.makedirs(output_dir, exist_ok=True)

    filename = os.path.basename(abs_path)
    output_path = os.path.join(output_dir, filename)
    shutil.copy2(abs_path, output_path)
    print(f"Saved copy  : {output_path}")

    return image


def main() -> None:
    """Parse command-line arguments and invoke load_image."""
    parser = argparse.ArgumentParser(
        description="Load an image, print its properties, and save a copy to output/."
    )
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to the image file (e.g. data/your-image.jpg).",
    )
    args = parser.parse_args()

    try:
        load_image(args.image_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
