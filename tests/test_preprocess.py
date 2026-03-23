# tests/test_preprocess.py
# Unit tests for preprocess_frame — spec Section 7.12

import pytest
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


# ── Tests that need a real image (skip until you have one) ───────────────────

def test_ssim_reference_against_itself():
    """UT-PP-01: SSIM of reference vs itself == 1.0"""
    pytest.skip("Needs real image — complete after getting PCB photo from work")


def test_output_dimensions_match():
    """UT-PP-02: bgr_aligned and gray_aligned share spatial dimensions"""
    pytest.skip("Needs real image — complete after getting PCB photo from work")


def test_blank_image_triggers_ssim_error():
    """UT-PP-03: blank image triggers SSIM ValueError"""
    pytest.skip("Needs real image — complete after getting PCB photo from work")