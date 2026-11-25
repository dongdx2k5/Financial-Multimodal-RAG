import json
from pathlib import Path

import pytest

from financial_rag.layout_detector import LayoutDetectionError, YOLOLayoutDetector
from scripts.create_sample_pdf import generate_sample_pdf
from scripts.label_studio_to_yolo import convert_label_studio_to_yolo
from scripts.pdf_to_annotations import process_pdf
from scripts.xanylabeling_to_yolo import convert_xanylabeling_to_yolo


def test_layout_annotation_pipeline(tmp_path: Path):
    # 1. Create a sample synthetic PDF with table and chart
    pdf_path = tmp_path / "test_report.pdf"
    generate_sample_pdf(pdf_path)
    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 0

    # 2. Run PDF to annotation extraction
    out_dir = tmp_path / "annotation_workspace"
    process_pdf(pdf_path, out_dir, dpi=100)

    # Check images generated
    images_dir = out_dir / "images"
    assert images_dir.exists()
    images = list(images_dir.glob("*.png"))
    assert len(images) == 1

    # Check Label Studio tasks JSON
    tasks_file = out_dir / "label_studio_tasks.json"
    assert tasks_file.exists()
    tasks_data = json.loads(tasks_file.read_text(encoding="utf-8"))
    assert len(tasks_data) == 1

    task = tasks_data[0]
    assert "data" in task
    assert "predictions" in task
    assert len(task["predictions"]) == 1
    pred_results = task["predictions"][0]["result"]
    assert len(pred_results) >= 1

    for res in pred_results:
        assert res["type"] == "rectanglelabels"
        val = res["value"]
        assert 0 <= val["x"] <= 100
        assert 0 <= val["y"] <= 100
        assert 0 < val["width"] <= 100
        assert 0 < val["height"] <= 100
        assert any(label in ["table", "chart"] for label in val["rectanglelabels"])

    # Check YOLO dataset
    yolo_dir = out_dir / "yolo"
    assert (yolo_dir / "data.yaml").exists()
    yolo_labels = list((yolo_dir / "labels").glob("*.txt"))
    assert len(yolo_labels) == 1

    lines = yolo_labels[0].read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 1
    for line in lines:
        parts = line.split()
        assert len(parts) == 5
        class_id, cx, cy, w, h = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        assert class_id in (0, 1)
        assert 0 <= cx <= 1.0
        assert 0 <= cy <= 1.0
        assert 0 < w <= 1.0
        assert 0 < h <= 1.0

    # 3. Test Label Studio export converter
    yolo_final = tmp_path / "yolo_final"
    convert_label_studio_to_yolo(tasks_file, images_dir, yolo_final, val_split=0.0)

    assert (yolo_final / "data.yaml").exists()
    assert len(list((yolo_final / "images" / "train").glob("*.png"))) == 1
    assert len(list((yolo_final / "labels" / "train").glob("*.txt"))) == 1

    # 4. Test X-AnyLabeling JSON export & conversion
    xany_dir = out_dir / "xanylabeling"
    assert len(list(xany_dir.glob("*.json"))) == 1
    yolo_xany = tmp_path / "yolo_xany"
    summary = convert_xanylabeling_to_yolo(xany_dir, images_dir, yolo_xany, val_split=0.0)
    assert summary["images"] == 1
    assert (yolo_xany / "data.yaml").exists()


def test_yolo_detector_requires_explicit_model(tmp_path: Path):
    detector = YOLOLayoutDetector(tmp_path / "missing.onnx")
    with pytest.raises(LayoutDetectionError, match="LAYOUT_MODEL_PATH"):
        detector._load_session()
