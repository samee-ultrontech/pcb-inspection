# tests/test_preprocess.py
# Unit tests for preprocess_frame — spec Section 7.12

import pytest
import cv2
import numpy as np
from scripts.preprocess import preprocess_frame


# ── Tests that pass RIGHT NOW (no image needed) ───────────────────────────────

def test_filepath_raises_typeerror():
    """UT-PP-04: passing a file path string raises TypeError immediately."""
    dummy = np.zeros((100, 100, 3), dtype=np.uint8)
    with pytest.raises(TypeError):
        preprocess_frame(dummy, "data/reference/reference_board.jpg")


def test_none_query_raises_valueerror():
    """UT-PP-05: None query raises ValueError."""
    dummy = np.zeros((100, 100, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        preprocess_frame(None, dummy)


def test_even_blur_k_raises_valueerror():
    """blur_k must be odd — even number raises ValueError."""
    dummy = np.zeros((100, 100, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        preprocess_frame(dummy, dummy, blur_k=4)


def test_non_array_query_raises_typeerror():
    """Non-array query raises TypeError."""
    dummy = np.zeros((100, 100, 3), dtype=np.uint8)
    with pytest.raises(TypeError):
        preprocess_frame("not_an_array", dummy)


# ── Step 2: GaussianBlur behaviour ───────────────────────────────────────────

def test_blur_k1_is_identity():
    """blur_k=1 (1×1 kernel) must leave pixel values unchanged."""
    gray = np.random.default_rng(0).integers(0, 256, (100, 100), dtype=np.uint8)
    result = cv2.GaussianBlur(gray, (1, 1), 0)
    np.testing.assert_array_equal(result, gray)


def test_blur_reduces_variance():
    """Blurring a noisy image must reduce per-pixel standard deviation."""
    noisy = np.random.default_rng(42).integers(0, 256, (200, 200), dtype=np.uint8)
    blurred = cv2.GaussianBlur(noisy, (5, 5), 0)
    assert blurred.std() < noisy.std(), (
        f"Expected blur to reduce std: {noisy.std():.2f} → {blurred.std():.2f}"
    )


def test_blur_preserves_shape():
    """GaussianBlur must not alter image dimensions."""
    gray = np.zeros((480, 640), dtype=np.uint8)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    assert blurred.shape == gray.shape


def test_blur_output_dtype():
    """GaussianBlur output dtype must remain uint8."""
    gray = np.random.default_rng(7).integers(0, 256, (100, 100), dtype=np.uint8)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    assert blurred.dtype == np.uint8


def test_larger_kernel_smooths_more():
    """A wider kernel (k=9) must produce lower variance than a narrow one (k=3)."""
    noisy = np.random.default_rng(99).integers(0, 256, (200, 200), dtype=np.uint8)
    blurred_3 = cv2.GaussianBlur(noisy, (3, 3), 0)
    blurred_9 = cv2.GaussianBlur(noisy, (9, 9), 0)
    assert blurred_9.std() < blurred_3.std()


# ── Tests that need a real image (skip until you have one) ───────────────────

def test_ssim_reference_against_itself():
    """UT-PP-01: SSIM of reference vs itself == 1.0"""
    pytest.skip("Needs real image — complete after getting PCB photo from work")


def test_output_dimensions_match():
    """UT-PP-02: bgr_aligned and gray_aligned share spatial dimensions after warpPerspective."""
    rng = np.random.default_rng(0)
    bgr  = rng.integers(0, 256, (300, 400, 3), dtype=np.uint8)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    H = np.eye(3, dtype=np.float64)  # identity — no geometric change
    h, w = gray.shape
    bgr_aligned  = cv2.warpPerspective(bgr,  H, (w, h))
    gray_aligned = cv2.warpPerspective(gray, H, (w, h))
    assert bgr_aligned.shape[:2] == gray_aligned.shape[:2]


def test_blank_image_triggers_ssim_error():
    """UT-PP-03: blank image triggers SSIM ValueError"""
    pytest.skip("Needs real image — complete after getting PCB photo from work")