# scripts/preprocess.py
"""
Phase 2 OpenCV preprocessing pipeline for PCB inspection.

Implements spec §7.2 (image normalisation) and §7.8 (alignment pipeline).

Inputs
------
bgr_query     : BGR numpy array — raw frame captured from the inspection camera.
bgr_reference : BGR numpy array — pre-loaded image of a known-good board.

Outputs
-------
A dict with:
    bgr_aligned  : query frame warped to the reference coordinate space (BGR).
    gray_aligned : grayscale version of bgr_aligned, used for SSIM comparison.
    ssim_score   : structural similarity index vs the reference (0.0–1.0).
    verdict_hint : PASS_CANDIDATE / FLAG_CANDIDATE / FAIL_CANDIDATE.

Pipeline steps
--------------
1. Colour conversion  — BGR → grayscale for both images.
2. GaussianBlur       — noise suppression before thresholding.
3. Otsu threshold     — global binarisation of both blurred images.
4. Canny              — edge detection on both binary images.
5. ORB + matching     — keypoint detection on edge maps; Lowe's ratio test.
6. Homography + warp  — perspective alignment of query onto reference.
7. SSIM               — structural similarity score and verdict (TODO).
"""

import cv2
import numpy as np
import logging
from skimage.metrics import structural_similarity as compute_ssim

logger = logging.getLogger(__name__)


def preprocess_frame(
    bgr_query: np.ndarray,
    bgr_reference: np.ndarray,
    blur_k: int = 5,
    ssim_min: float = 0.80,
) -> dict:
    """Pre-process a single PCB frame against the reference board.

    Parameters
    ----------
    bgr_query     : raw BGR frame from camera — numpy array, NOT a file path
    bgr_reference : pre-loaded reference board — numpy array, NOT a file path
    blur_k        : GaussianBlur kernel size (must be odd, default 5)
    ssim_min      : lower SSIM threshold for FAIL/FLAG boundary (default 0.80)

    Returns
    -------
    dict with keys:
        bgr_aligned   : np.ndarray  — 3-channel BGR, homography-warped
        gray_aligned  : np.ndarray  — 1-channel grayscale of bgr_aligned
        ssim_score    : float       — SSIM vs reference (0.0 to 1.0)
        verdict_hint  : str         — PASS_CANDIDATE / FLAG_CANDIDATE / FAIL_CANDIDATE

    Raises
    ------
    TypeError  : if bgr_reference is not a numpy array
    ValueError : if bgr_query is None, or SSIM returns NaN / negative
    """

    # ── Input validation ─────────────────────────────────────────────────────
    if not isinstance(bgr_reference, np.ndarray):
        raise TypeError(
            "bgr_reference must be a numpy array, not a file path. "
            "Load the image once in inspect.py and pass the array here."
        )
    if bgr_query is None:
        raise ValueError(
            "bgr_query is None — image could not be loaded upstream."
        )
    if not isinstance(bgr_query, np.ndarray):
        raise TypeError("bgr_query must be a numpy array.")

    # ── Step 1: Colour conversion ─────────────────────────────────────────────
    # Convert both images to grayscale.
    # bgr_query is kept UNCHANGED — needed for warpPerspective in Step 6.
    gray_query     = cv2.cvtColor(bgr_query,     cv2.COLOR_BGR2GRAY)
    gray_reference = cv2.cvtColor(bgr_reference, cv2.COLOR_BGR2GRAY)

    # ── Step 2: Gaussian blur ─────────────────────────────────────────────────
    # Smooths noise before thresholding and edge detection.
    # blur_k must be odd (3, 5, 7...). Default 5 works for most webcam images.
    if blur_k % 2 == 0:
        raise ValueError(f"blur_k must be an odd number, got {blur_k}")

    blurred_query     = cv2.GaussianBlur(gray_query,     (blur_k, blur_k), 0)
    blurred_reference = cv2.GaussianBlur(gray_reference, (blur_k, blur_k), 0)

    # ── Step 3: Adaptive threshold ────────────────────────────────────────────
    # Converts to black/white binary image.
    # Adaptive handles uneven lighting across the board surface.
    _, binary_query = cv2.threshold(
        blurred_query, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    _, binary_reference = cv2.threshold(
        blurred_reference, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # ── Step 4: Canny edge detection ────────────────────────────────────────
    # Detect edges in both binary images. These edge maps are passed to the
    # ORB keypoint detector in Step 5 to find alignment feature points.
    # Thresholds 50/150 are the standard values for PCB imagery.
    edges_query     = cv2.Canny(binary_query,     50, 150)
    edges_reference = cv2.Canny(binary_reference, 50, 150)
    # ────────────────────────────────────────────────────────────────────────

    # ── Step 5: ORB keypoint detection and matching ──────────────────────────
    # Detect distinctive feature points in both edge maps and match them.
    # These matches are used in Step 6 to compute the homography matrix
    # that aligns the query image to the reference.
    orb = cv2.ORB_create(nfeatures=1000)

    kp_query, des_query         = orb.detectAndCompute(edges_query,     None)
    kp_reference, des_reference = orb.detectAndCompute(edges_reference, None)

    if des_query is None or des_reference is None or len(kp_query) < 4 or len(kp_reference) < 4:
        raise ValueError("ORB found too few keypoints to compute alignment.")

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw_matches = matcher.knnMatch(des_query, des_reference, k=2)

    # Lowe's ratio test — keep only unambiguous matches
    good_matches = [m for m, n in raw_matches if m.distance < 0.75 * n.distance]

    if len(good_matches) < 4:
        raise ValueError(
            f"Too few good matches ({len(good_matches)}) to compute homography. "
            "Check image quality or lighting."
        )
    # ────────────────────────────────────────────────────────────────────────

    # ── Step 6: findHomography + warpPerspective (alignment) ────────────────
    # Build (query → reference) point arrays from the good matches.
    src_pts = np.float32(
        [kp_query[m.queryIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)
    dst_pts = np.float32(
        [kp_reference[m.trainIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if H is None:
        raise ValueError(
            "findHomography returned None — could not compute a valid "
            "transformation. Check image quality or keypoint matches."
        )

    h, w = bgr_reference.shape[:2]

    # Warp the full-colour query frame to align with the reference.
    bgr_aligned  = cv2.warpPerspective(bgr_query,  H, (w, h))

    # Warp the grayscale query too — this is what SSIM compares in Step 7.
    gray_aligned = cv2.warpPerspective(gray_query, H, (w, h))
    # ────────────────────────────────────────────────────────────────────────

    raise NotImplementedError("Step 7 not yet implemented.")


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse, sys, os

    parser = argparse.ArgumentParser(description="PCB pre-processing pipeline")
    parser.add_argument("--image",     required=True,  help="Path to query image")
    parser.add_argument("--reference", required=True,  help="Path to reference image")
    parser.add_argument("--blur-k",    type=int,   default=5)
    parser.add_argument("--ssim-min",  type=float, default=0.80)
    parser.add_argument("--output",    default="output/aligned.jpg")
    args = parser.parse_args()

    query = cv2.imread(args.image)
    ref   = cv2.imread(args.reference)

    if query is None:
        print(f"ERROR: Could not load query image: {args.image}")
        sys.exit(1)
    if ref is None:
        print(f"ERROR: Could not load reference image: {args.reference}")
        sys.exit(1)

    print(f"Query loaded:     {query.shape}")
    print(f"Reference loaded: {ref.shape}")

    try:
        result = preprocess_frame(query, ref, blur_k=args.blur_k)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    cv2.imwrite(args.output, result["bgr_aligned"])

    print(f"SSIM score:       {result['ssim_score']:.4f}")
    print(f"Verdict hint:     {result['verdict_hint']}")
    print(f"Aligned image:    {args.output}")