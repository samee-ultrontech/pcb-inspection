# tests/test_detect.py
# Unit tests for detect_defects.
#
# Tests are split into two groups:
#   1. Input-validation tests — run without any model or real image.
#   2. Integration tests      — require the trained model file; skipped if absent.
#      These use unittest.mock to swap out YOLO so we can test the parsing
#      logic (dict keys, rounding, bbox types) without GPU or real weights.

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from scripts.config import MODEL_PATH
from scripts.detect import detect_defects


# ── Input validation (no model needed) ───────────────────────────────────────

def test_non_array_raises_typeerror():
    with pytest.raises(TypeError):
        detect_defects("not_an_array")


def test_none_raises_typeerror():
    with pytest.raises(TypeError):
        detect_defects(None)


def test_missing_model_raises_filenotfounderror():
    dummy = np.zeros((100, 100, 3), dtype=np.uint8)
    with pytest.raises(FileNotFoundError):
        detect_defects(dummy, model_path="models/does_not_exist.pt")


# ── Mocked YOLO tests (no real model, no GPU) ─────────────────────────────────
#
# We patch 'scripts.detect.YOLO' so it never touches the filesystem.
# The mock returns a synthetic Results object shaped like Ultralytics output.

def _make_mock_results(class_id: int, class_name: str, conf: float, xyxy: list):
    """Build a minimal Ultralytics Results mock with one detection."""
    box = MagicMock()
    box.cls  = [class_id]
    box.conf = [conf]
    box.xyxy = [xyxy]

    result = MagicMock()
    result.boxes = [box]

    mock_model = MagicMock()
    mock_model.names = {class_id: class_name}
    mock_model.return_value = [result]
    return mock_model


@pytest.fixture
def fake_model_file(tmp_path):
    """Create a temporary .pt file so the FileNotFoundError guard passes."""
    pt = tmp_path / "fake.pt"
    pt.write_bytes(b"fake weights")
    return str(pt)


def test_returns_list(fake_model_file):
    mock_model = _make_mock_results(0, "solder_bridge", 0.92, [10.0, 20.0, 50.0, 60.0])
    with patch("scripts.detect.YOLO", return_value=mock_model):
        result = detect_defects(
            np.zeros((100, 100, 3), dtype=np.uint8),
            model_path=fake_model_file,
        )
    assert isinstance(result, list)


def test_detection_dict_has_required_keys(fake_model_file):
    mock_model = _make_mock_results(0, "solder_bridge", 0.92, [10.0, 20.0, 50.0, 60.0])
    with patch("scripts.detect.YOLO", return_value=mock_model):
        dets = detect_defects(
            np.zeros((100, 100, 3), dtype=np.uint8),
            model_path=fake_model_file,
        )
    assert len(dets) == 1
    for key in ("class_id", "class_name", "confidence", "bbox"):
        assert key in dets[0], f"Missing key: {key}"


def test_confidence_is_rounded_to_4dp(fake_model_file):
    mock_model = _make_mock_results(0, "solder_bridge", 0.912345678, [10.0, 20.0, 50.0, 60.0])
    with patch("scripts.detect.YOLO", return_value=mock_model):
        dets = detect_defects(
            np.zeros((100, 100, 3), dtype=np.uint8),
            model_path=fake_model_file,
        )
    assert dets[0]["confidence"] == pytest.approx(0.9123, abs=1e-4)


def test_bbox_values_are_integers(fake_model_file):
    mock_model = _make_mock_results(0, "solder_bridge", 0.92, [10.7, 20.3, 50.9, 60.1])
    with patch("scripts.detect.YOLO", return_value=mock_model):
        dets = detect_defects(
            np.zeros((100, 100, 3), dtype=np.uint8),
            model_path=fake_model_file,
        )
    assert all(isinstance(v, int) for v in dets[0]["bbox"])


def test_bbox_has_four_values(fake_model_file):
    mock_model = _make_mock_results(0, "solder_bridge", 0.92, [10.0, 20.0, 50.0, 60.0])
    with patch("scripts.detect.YOLO", return_value=mock_model):
        dets = detect_defects(
            np.zeros((100, 100, 3), dtype=np.uint8),
            model_path=fake_model_file,
        )
    assert len(dets[0]["bbox"]) == 4


def test_class_name_matches_model_names(fake_model_file):
    mock_model = _make_mock_results(2, "cold_joint", 0.85, [5.0, 5.0, 30.0, 30.0])
    with patch("scripts.detect.YOLO", return_value=mock_model):
        dets = detect_defects(
            np.zeros((100, 100, 3), dtype=np.uint8),
            model_path=fake_model_file,
        )
    assert dets[0]["class_name"] == "cold_joint"
    assert dets[0]["class_id"]   == 2


def test_no_boxes_returns_empty_list(fake_model_file):
    result_mock = MagicMock()
    result_mock.boxes = None

    mock_model = MagicMock()
    mock_model.return_value = [result_mock]

    with patch("scripts.detect.YOLO", return_value=mock_model):
        dets = detect_defects(
            np.zeros((100, 100, 3), dtype=np.uint8),
            model_path=fake_model_file,
        )
    assert dets == []
