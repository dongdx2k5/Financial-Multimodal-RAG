from pathlib import Path

import numpy as np
import pymupdf
import pytest

from financial_rag.ocr_backends import (
    OCREngineUnavailable,
    OCRLine,
    OCRResult,
    VietOCRBackend,
    pixel_box_to_pdf,
    vietocr_inference_config,
)


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_pixel_boxes_are_unrotated_pdf_coordinates(rotation):
    with pymupdf.open() as doc:
        page = doc.new_page(width=200, height=300)
        page.set_rotation(rotation)
        expected = pymupdf.Rect(20, 30, 80, 50)
        rendered = expected * page.rotation_matrix
        scale = 300 / 72
        pixels = [value * scale for value in rendered]
        assert pixel_box_to_pdf(pixels, page, scale) == pytest.approx(list(expected))
        assert page.rotation == rotation


def test_cropped_pixmap_origin_is_included():
    with pymupdf.open() as doc:
        page = doc.new_page(width=200, height=300)
        result = pixel_box_to_pdf([0, 0, 40, 20], page, 2, origin=(100, 120))
        assert result == pytest.approx([50, 60, 70, 70])


def test_cropbox_coordinates_are_relative_to_visible_page():
    with pymupdf.open() as doc:
        page = doc.new_page(width=200, height=300)
        page.set_cropbox(pymupdf.Rect(20, 30, 180, 260))
        assert pixel_box_to_pdf([0, 0, 20, 20], page, 2) == pytest.approx([0, 0, 10, 10])


def test_boxes_are_clipped_to_page():
    with pymupdf.open() as doc:
        page = doc.new_page(width=200, height=300)
        assert pixel_box_to_pdf([-10, -20, 450, 650], page, 2) == [0, 0, 200, 300]


@pytest.mark.parametrize("box", [[1, 2, 3], [float("nan"), 0, 4, 5], [0, 0, 0, 0]])
def test_invalid_boxes_are_rejected(box):
    with pymupdf.open() as doc, pytest.raises(ValueError):
        pixel_box_to_pdf(box, doc.new_page(), 2)


def test_compatibility_words_do_not_invent_token_boxes():
    result = OCRResult([OCRLine("Doanh thu 1.250.000", [1, 2, 90, 12], 0.9)])
    assert result.text == "Doanh thu 1.250.000"
    assert result.words == [[1, 2, 90, 12, "Doanh thu 1.250.000", 0, 0, 0]]


def test_vietocr_config_has_vietnamese_and_no_download():
    cfg = vietocr_inference_config(Path("local.pth"))
    assert set("ếạợỸ0123456789().,-") <= set(cfg["vocab"])
    assert cfg["cnn"]["pretrained"] is False
    assert cfg["weights"] == "local.pth"
    assert cfg["device"] == "cpu"
    assert cfg["seq_modeling"] == "seq2seq"


def test_missing_models_fail_before_optional_imports(tmp_path):
    backend = VietOCRBackend(weights_path=tmp_path / "missing.pth")
    with pytest.raises(OCREngineUnavailable, match="VIETOCR_WEIGHTS"):
        backend._load_predictor()
    with pytest.raises(OCREngineUnavailable, match="OCR_DETECTOR_PATH"):
        backend._load_detector()


def test_empty_crop_list_does_not_initialize_models():
    assert VietOCRBackend().recognize_images([]) == []


def test_batch_recognition_preserves_numbers_order_and_missing_confidence():
    class Predictor:
        def predict_batch(self, images, return_prob):
            assert return_prob is True
            return images, [float("nan") if value == "?" else 0.95 for value in images]

    backend = VietOCRBackend(batch_size=2)
    backend._predictor = Predictor()
    assert backend.recognize_images(["(1.250)", "0", "?"]) == [
        ("(1.250)", 0.95),
        ("0", 0.95),
        ("?", None),
    ]


def test_batch_output_length_mismatch_fails_loudly():
    class Predictor:
        def predict_batch(self, images, return_prob):
            return ["only one"], [0.9]

    backend = VietOCRBackend()
    backend._predictor = Predictor()
    with pytest.raises(RuntimeError, match="không khớp"):
        backend.recognize_images(["one", "two"])


def test_invalid_settings():
    with pytest.raises(ValueError):
        VietOCRBackend(min_confidence=2)
    with pytest.raises(ValueError):
        VietOCRBackend(batch_size=0)


def test_extract_rejects_oversized_page_before_loading_optional_dependencies():
    with pymupdf.open() as doc:
        page = doc.new_page(width=10000, height=10000)
        with pytest.raises(RuntimeError, match="quá lớn"):
            VietOCRBackend().extract(page)


def test_quad_coordinates():
    with pymupdf.open() as doc:
        page = doc.new_page(width=100, height=100)
        box = np.array([[10, 10], [40, 12], [38, 30], [8, 28]])
        assert pixel_box_to_pdf(box, page, 2) == [4, 5, 20, 15]
