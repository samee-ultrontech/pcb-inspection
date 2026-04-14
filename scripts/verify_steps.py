# scripts/verify_steps.py
# Verifies each of the 7 preprocessing steps individually.
# Run with: python scripts/verify_steps.py
# No real PCB image needed — generates a synthetic test image.

import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as compute_ssim
from scripts.config import SSIM_PASS_THRESHOLD, SSIM_FLAG_THRESHOLD

os.makedirs("output", exist_ok=True)

PASS = "PASS"
FAIL = "FAIL"
results = []


def log(step, name, status, detail=""):
    icon = "✓" if status == PASS else "✗"
    print(f"  {icon}  Step {step} — {name}: {status}  {detail}")
    results.append((step, name, status))


def make_synthetic_image():
    """Creates a synthetic PCB-like image for testing."""
    img = np.ones((640, 640, 3), dtype=np.uint8) * 45
    cv2.rectangle(img, (20, 20), (620, 620), (34, 100, 34), -1)
    for x in range(80, 580, 80):
        for y in range(80, 580, 80):
            cv2.rectangle(img, (x, y), (x+40, y+20), (180, 130, 10), -1)
            cv2.circle(img, (x+20, y+10), 6, (200, 200, 190), -1)
    for i in range(80, 580, 80):
        cv2.line(img, (i+20, 20),  (i+20, 620), (34, 80, 34), 2)
        cv2.line(img, (20,  i+10), (620,  i+10), (34, 80, 34), 2)
    return img


print("\n========================================")
print("  Phase 2 — Step-by-step verification")
print("========================================\n")

# Create synthetic test image
bgr = make_synthetic_image()
cv2.imwrite("output/verify_0_synthetic.jpg", bgr)
print(f"Synthetic image created: {bgr.shape} {bgr.dtype}\n")

# ── Step 1: cvtColor ─────────────────────────────────────────────────────────
print("Step 1 — cv2.cvtColor")
try:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("output/verify_1_gray.jpg", gray)
    assert gray.ndim == 2,          "Expected 2D array (H, W)"
    assert gray.dtype == np.uint8,  "Expected uint8"
    assert gray.shape == bgr.shape[:2], "Shape mismatch"
    log(1, "cvtColor", PASS, f"shape={gray.shape} dtype={gray.dtype}")
except Exception as e:
    log(1, "cvtColor", FAIL, str(e))
    gray = None

# ── Step 2: GaussianBlur ──────────────────────────────────────────────────────
print("Step 2 — cv2.GaussianBlur")
try:
    assert gray is not None, "Step 1 must pass first"
    blur_k = 5
    blurred = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
    cv2.imwrite("output/verify_2_blurred.jpg", blurred)
    assert blurred.shape == gray.shape, "Shape changed after blur"
    assert blurred.dtype == np.uint8,   "dtype changed after blur"
    # Blurred image should be smoother — std dev should be lower
    assert blurred.std() <= gray.std(), "Blur did not reduce variance"
    log(2, "GaussianBlur", PASS, f"shape={blurred.shape} blur_k={blur_k}")
except Exception as e:
    log(2, "GaussianBlur", FAIL, str(e))
    blurred = gray

# ── Step 3: Otsu threshold ────────────────────────────────────────────────────
print("Step 3 — cv2.threshold (Otsu)")
try:
    assert blurred is not None, "Step 2 must pass first"
    _, binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    cv2.imwrite("output/verify_3_binary.jpg", binary)
    assert binary.shape == blurred.shape, "Shape mismatch"
    unique_vals = np.unique(binary)
    assert set(unique_vals).issubset({0, 255}), \
        f"Binary image contains non-binary values: {unique_vals}"
    log(3, "threshold (Otsu)", PASS,
        f"shape={binary.shape} unique_values={unique_vals}")
except Exception as e:
    log(3, "threshold (Otsu)", FAIL, str(e))
    binary = None

# ── Step 4: Canny ─────────────────────────────────────────────────────────────
print("Step 4 — cv2.Canny")
edges = None
try:
    assert binary is not None, "Step 3 must pass first"
    edges = cv2.Canny(binary, threshold1=50, threshold2=150)
    cv2.imwrite("output/verify_4_edges.jpg", edges)
    assert edges.shape == binary.shape, "Shape mismatch"
    assert edges.dtype == np.uint8,     "Wrong dtype"
    edge_pixels = np.count_nonzero(edges)
    assert edge_pixels > 0, "No edges detected — image may be blank"
    log(4, "Canny", PASS, f"edge_pixels={edge_pixels}")
except Exception as e:
    log(4, "Canny", FAIL, str(e))

# ── Step 5: ORB keypoint detection and matching ───────────────────────────────
print("Step 5 — ORB + Lowe's ratio test")
try:
    assert edges is not None, "Step 4 must pass first"
    orb = cv2.ORB_create(nfeatures=1000)
    kp1, des1 = orb.detectAndCompute(edges, None)
    # Use a slightly shifted edge map as a stand-in for a second image
    M_shift = np.float32([[1, 0, 5], [0, 1, 5]])
    edges_shifted = cv2.warpAffine(edges, M_shift, (edges.shape[1], edges.shape[0]))
    kp2, des2 = orb.detectAndCompute(edges_shifted, None)

    assert des1 is not None and des2 is not None, "ORB found no descriptors"
    assert len(kp1) >= 4 and len(kp2) >= 4, \
        f"Too few keypoints: {len(kp1)} / {len(kp2)}"

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw_matches = matcher.knnMatch(des1, des2, k=2)
    good_matches = [m for m, n in raw_matches if m.distance < 0.75 * n.distance]

    assert len(good_matches) >= 4, \
        f"Too few good matches after ratio test: {len(good_matches)}"

    # Draw matches for visual inspection
    match_img = cv2.drawMatches(edges, kp1, edges_shifted, kp2,
                                good_matches[:20], None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite("output/verify_5_matches.jpg", match_img)

    log(5, "ORB + Lowe's ratio test", PASS,
        f"keypoints={len(kp1)} good_matches={len(good_matches)}")
except Exception as e:
    log(5, "ORB + Lowe's ratio test", FAIL, str(e))

# ── Step 6: Homography + warpPerspective ──────────────────────────────────────
print("Step 6 — cv2.warpPerspective (homography)")
try:
    assert gray is not None, "Step 1 must pass first"
    # Use a slightly shifted version as the query to test alignment
    M_shift = np.float32([[1, 0, 10], [0, 1, 10]])
    shifted = cv2.warpAffine(bgr, M_shift, (bgr.shape[1], bgr.shape[0]))
    gray_shifted = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(gray_shifted, None)
    kp2, des2 = orb.detectAndCompute(gray,         None)

    bf      = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des1, des2), key=lambda x: x.distance)[:50]

    assert len(matches) >= 4, \
        f"Not enough matches: {len(matches)} (need at least 4)"

    src_pts = np.float32(
        [kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32(
        [kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    assert H is not None, "findHomography returned None"

    h, w         = gray.shape
    bgr_aligned  = cv2.warpPerspective(shifted, H, (w, h))
    gray_aligned = cv2.warpPerspective(gray_shifted, H, (w, h))

    cv2.imwrite("output/verify_6_aligned_bgr.jpg",  bgr_aligned)
    cv2.imwrite("output/verify_6_aligned_gray.jpg", gray_aligned)

    assert bgr_aligned.shape  == (h, w, 3), "bgr_aligned wrong shape"
    assert gray_aligned.shape == (h, w),    "gray_aligned wrong shape"
    assert bgr_aligned.shape[:2] == gray_aligned.shape, \
        "bgr and gray aligned shapes don't match"

    log(6, "warpPerspective", PASS,
        f"matches={len(matches)} bgr={bgr_aligned.shape} "
        f"gray={gray_aligned.shape}")
except Exception as e:
    log(6, "warpPerspective", FAIL, str(e))
    bgr_aligned  = bgr
    gray_aligned = gray

# ── Step 7: SSIM ─────────────────────────────────────────────────────────────
print("Step 7 — skimage SSIM")
try:
    assert gray_aligned is not None, "Step 6 must pass first"
    ref_gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    score = compute_ssim(gray_aligned, ref_gray, data_range=255)

    assert score == score,      "SSIM returned NaN"
    assert 0.0 <= score <= 1.0, f"SSIM out of range: {score}"

    if score >= SSIM_PASS_THRESHOLD:
        hint = "PASS_CANDIDATE"
    elif score >= SSIM_FLAG_THRESHOLD:
        hint = "FLAG_CANDIDATE"
    else:
        hint = "FAIL_CANDIDATE"

    # Also test: reference vs itself should return 1.0
    score_self = compute_ssim(ref_gray, ref_gray, data_range=255)
    assert abs(score_self - 1.0) < 1e-6, \
        f"SSIM of ref vs itself should be 1.0, got {score_self}"

    log(7, "SSIM", PASS,
        f"score={score:.4f} hint={hint} self_check={score_self:.4f}")
except Exception as e:
    log(7, "SSIM", FAIL, str(e))

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n========================================")
print("  Summary")
print("========================================")
passed = sum(1 for _, _, s in results if s == PASS)
failed = sum(1 for _, _, s in results if s == FAIL)
for step, name, status in results:
    icon = "✓" if status == PASS else "✗"
    print(f"  {icon}  Step {step} — {name}: {status}")
print(f"\n  {passed}/7 steps passed  |  {failed} failed")
print("\nDebug images saved to output/:")
for f in sorted(os.listdir("output")):
    if f.startswith("verify_"):
        print(f"  output/{f}")
print()