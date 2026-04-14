# tests/test_verdict.py
# Unit tests for compute_verdict — no images or model required.
#
# All four decision rules are covered:
#   Rule 1 — SSIM < ssim_flag                  → FAIL
#   Rule 2 — SSIM ≥ ssim_pass, no detections   → PASS
#   Rule 3 — SSIM ≥ ssim_pass, defects found   → FAIL
#   Rule 4 — ssim_flag ≤ SSIM < ssim_pass      → FLAG

import pytest
from scripts.verdict import compute_verdict

# Thresholds used across tests — mirrors config.py defaults
PASS_T = 0.85
FLAG_T = 0.80


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_detection(class_name="solder_bridge", confidence=0.92):
    return {"class_name": class_name, "confidence": confidence, "bbox": [10, 10, 50, 50]}


# ── Rule 1: SSIM too low → FAIL regardless of detections ─────────────────────

def test_low_ssim_no_detections_is_fail():
    result = compute_verdict(0.70, [])
    assert result["verdict"] == "FAIL"


def test_low_ssim_with_detections_is_still_fail():
    result = compute_verdict(0.60, [make_detection()])
    assert result["verdict"] == "FAIL"


def test_ssim_exactly_at_flag_threshold_is_flag_not_fail():
    """The boundary value ssim_flag=0.80 is included in FLAG, not FAIL."""
    result = compute_verdict(FLAG_T, [])
    assert result["verdict"] == "FLAG"


# ── Rule 2: high SSIM, no defects → PASS ─────────────────────────────────────

def test_high_ssim_no_defects_is_pass():
    result = compute_verdict(0.92, [])
    assert result["verdict"] == "PASS"


def test_ssim_exactly_at_pass_threshold_no_defects_is_pass():
    result = compute_verdict(PASS_T, [])
    assert result["verdict"] == "PASS"


def test_perfect_ssim_no_defects_is_pass():
    result = compute_verdict(1.0, [])
    assert result["verdict"] == "PASS"


# ── Rule 3: high SSIM but defects present → FAIL ─────────────────────────────

def test_high_ssim_with_one_defect_is_fail():
    result = compute_verdict(0.90, [make_detection()])
    assert result["verdict"] == "FAIL"


def test_high_ssim_with_multiple_defects_is_fail():
    dets = [make_detection("solder_bridge"), make_detection("cold_joint", 0.78)]
    result = compute_verdict(0.95, dets)
    assert result["verdict"] == "FAIL"


def test_fail_result_lists_all_defects():
    dets = [make_detection("solder_bridge", 0.92), make_detection("cold_joint", 0.78)]
    result = compute_verdict(0.90, dets)
    assert result["num_defects"] == 2
    assert any("solder_bridge" in tag for tag in result["defects"])
    assert any("cold_joint"    in tag for tag in result["defects"])


# ── Rule 4: borderline SSIM → FLAG ───────────────────────────────────────────

def test_medium_ssim_no_defects_is_flag():
    result = compute_verdict(0.82, [])
    assert result["verdict"] == "FLAG"


def test_medium_ssim_with_defects_is_still_flag():
    """FLAG is for manual review — YOLO detections are noted but don't change it to FAIL."""
    result = compute_verdict(0.83, [make_detection()])
    assert result["verdict"] == "FLAG"


def test_flag_reason_mentions_defects_when_present():
    result = compute_verdict(0.83, [make_detection("lifted_lead", 0.88)])
    assert "lifted_lead" in result["reason"]


# ── Return structure ──────────────────────────────────────────────────────────

def test_result_has_all_required_keys():
    result = compute_verdict(0.90, [])
    for key in ("verdict", "reason", "ssim_score", "num_defects", "defects"):
        assert key in result, f"Missing key: {key}"


def test_num_defects_matches_detection_count():
    dets = [make_detection() for _ in range(3)]
    result = compute_verdict(0.90, dets)
    assert result["num_defects"] == 3


def test_defects_list_is_empty_for_pass():
    result = compute_verdict(0.90, [])
    assert result["defects"] == []


def test_ssim_score_preserved_in_result():
    result = compute_verdict(0.8765, [])
    assert result["ssim_score"] == pytest.approx(0.8765)


# ── Input validation ──────────────────────────────────────────────────────────

def test_non_numeric_ssim_raises_typeerror():
    with pytest.raises(TypeError):
        compute_verdict("high", [])


def test_non_list_detections_raises_typeerror():
    with pytest.raises(TypeError):
        compute_verdict(0.90, "no detections")
