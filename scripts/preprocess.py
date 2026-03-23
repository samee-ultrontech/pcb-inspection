# scripts/preprocess.py
# Phase 2 — OpenCV pre-processing pipeline
# Spec reference: Section 7.8

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
    binary_query = cv2.adaptiveThreshold(
        blurred_query, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,
        C=2
    )

    # ── Step 4: Canny edge detection ──────────────────────────────────────────
    # Finds sharp brightness transitions — solder joint boundaries,
    # component edges, pad outlines.
    edges_query = cv2.Canny(blurred_query, threshold1=50, threshold2=150)

    # ── Step 5: Contour detection ─────────────────────────────────────────────
    # Finds blobs in the binary image — coarse check for gross anomalies.
    contours, _ = cv2.findContours(
        binary_query,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    significant_contours = [c for c in contours if cv2.contourArea(c) > 100]
    logger.debug("Found %d significant contours", len(significant_contours))

    # ── Step 6: Homography alignment ─────────────────────────────────────────
    # Aligns the query board to the reference board using ORB keypoints.
    # Warps BOTH bgr_query and gray_query using the same matrix H.
    orb = cv2.ORB_create()
    kp_query, des_query = orb.detectAndCompute(gray_query,     None)
    kp_ref,   des_ref   = orb.detectAndCompute(gray_reference, None)

    bf      = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_query, des_ref)
    matches = sorted(matches, key=lambda x: x.distance)[:50]

    if len(matches) < 4:
        raise ValueError(
            f"Not enough keypoint matches for homography: {len(matches)}. "
            "Ensure the query and reference images show the same board type."
        )

    src_pts = np.float32(
        [kp_query[m.queryIdx].pt for m in matches]
    ).reshape(-1, 1, 2)
    dst_pts = np.float32(
        [kp_ref[m.trainIdx].pt for m in matches]
    ).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if H is None:
        raise ValueError(
            "Homography could not be computed — findHomography returned None. "
            "Common causes: low-texture image, too few keypoints, "
            "or very different board orientations."
        )

    h, w         = gray_reference.shape
    bgr_aligned  = cv2.warpPerspective(bgr_query,  H, (w, h))
    gray_aligned = cv2.warpPerspective(gray_query, H, (w, h))

    # ── Step 7: SSIM computation ──────────────────────────────────────────────
    # Compares aligned grayscale query against grayscale reference.
    score, _ = compute_ssim(gray_aligned, gray_reference, full=True)

    if score is None or score != score:
        raise ValueError(
            "SSIM returned NaN — likely cause: image dimensions mismatch "
            "or all-black image."
        )
    if score < 0:
        raise ValueError(
            f"SSIM returned a negative value ({score:.4f})."
        )

    ssim_score = float(score)

    if ssim_score >= 0.85:
        verdict_hint = "PASS_CANDIDATE"
    elif ssim_score >= 0.80:
        verdict_hint = "FLAG_CANDIDATE"
    else:
        verdict_hint = "FAIL_CANDIDATE"

    return {
        "bgr_aligned":  bgr_aligned,
        "gray_aligned": gray_aligned,
        "ssim_score":   ssim_score,
        "verdict_hint": verdict_hint,
    }


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