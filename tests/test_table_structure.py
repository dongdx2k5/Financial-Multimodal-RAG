from types import SimpleNamespace

import pymupdf
import pytest

from financial_rag.table_structure import (
    SLANetTableBackend,
    TableStructureError,
    _statement_region,
    assign_text_to_cells,
    cells_from_prediction,
)


def cell(box, row=0, col=0):
    return {
        "row": row,
        "col": col,
        "row_span": 1,
        "col_span": 1,
        "bbox": box,
        "text": "",
        "confidence": None,
    }


def test_cells_scale_origin_and_merged_span():
    cells = cells_from_prediction(
        [[0, 0, 80, 40], [80, 0, 120, 40]],
        [[0, 1, 0, 0], [0, 0, 1, 1]],
        scale=2,
        pixel_origin=(20, 40),
        region=[10, 20, 70, 40],
    )
    assert cells[0]["bbox"] == [10, 20, 50, 40]
    assert cells[0]["row_span"] == 2
    assert cells[1]["col"] == 1


@pytest.mark.parametrize(
    "logic",
    [
        [[0, 0, 0, 0], [0, 0, 0, 0]],
        [[-1, 0, 0, 0], [0, 0, 1, 1]],
        [[0, 0, 0, 100000], [0, 0, 1, 1]],
        [[0.5, 1, 0, 0], [0, 0, 1, 1]],
    ],
)
def test_rejects_invalid_logic(logic):
    with pytest.raises(TableStructureError):
        cells_from_prediction(
            [[0, 0, 20, 20], [20, 0, 40, 20]],
            logic,
            scale=1,
            pixel_origin=(0, 0),
            region=[0, 0, 40, 20],
        )


def test_ocr_keeps_raw_numbers_and_does_not_duplicate_across_cells():
    cells, warnings = assign_text_to_cells(
        [cell([0, 0, 100, 20]), cell([100, 0, 200, 20], col=1)],
        [
            SimpleNamespace(text="(1.250.000)", bbox=[10, 2, 90, 18], confidence=0.91),
            {"text": "5.000 6.000", "bbox": [90, 2, 160, 18], "confidence": 0.9},
        ],
    )
    assert cells[0]["text"] == "(1.250.000)"
    # The second region has >60% in cell 2 but its >20% overlap check below
    # uses a deliberately symmetric crossing region instead.
    assert cells[0]["confidence"] == 0.91
    symmetric, warnings = assign_text_to_cells(
        [cell([0, 0, 100, 20]), cell([100, 0, 200, 20], col=1)],
        [{"text": "5.000 6.000", "bbox": [50, 2, 150, 18]}],
    )
    assert all(not c["text"] for c in symmetric)
    assert warnings


def test_reject_ambiguous_despite_majority_overlap():
    cells, warnings = assign_text_to_cells(
        [cell([0, 0, 100, 20]), cell([100, 0, 200, 20], col=1)],
        [{"text": "1.000 2.000", "bbox": [70, 2, 170, 18]}],
    )
    assert all(not c["text"] for c in cells)
    assert any("giao nhiều ô" in warning for warning in warnings)


def test_no_region_does_not_load_optional_model():
    backend = SLANetTableBackend()
    doc = pymupdf.open()
    page = doc.new_page()
    backend._load_engine = lambda: pytest.fail("Must not load model without a table region")
    assert backend.extract_tables(page, []) == []
    doc.close()


def test_explicit_region_render_mapping_and_rotation_restore():
    doc = pymupdf.open()
    page = doc.new_page(width=300, height=400)
    page.set_rotation(90)
    backend = SLANetTableBackend(dpi=144)
    backend._engine = lambda image: SimpleNamespace(
        cell_bboxes=[[[0, 0, image.shape[1], image.shape[0]]]],
        logic_points=[[[0, 0, 0, 0]]],
    )
    tables = backend.extract_tables(
        page, [[30, 40, 90, 50, "Doanh thu", 0, 0, 0]], regions=[[20, 30, 120, 60]]
    )
    assert page.rotation == 90
    assert tables[0]["cells"][0]["bbox"] == [20, 30, 120, 60]
    assert tables[0]["cells"][0]["text"] == "Doanh thu"
    assert tables[0]["region_source"] == "explicit"
    doc.close()


def test_rotation_restored_on_failure():
    doc = pymupdf.open()
    page = doc.new_page()
    page.set_rotation(270)
    backend = SLANetTableBackend()

    def fail():
        raise TableStructureError("missing weights")

    backend._load_engine = fail
    with pytest.raises(TableStructureError):
        backend.extract_tables(page, [], regions=[[10, 10, 100, 100]])
    assert page.rotation == 270
    doc.close()


def test_statement_heuristic_requires_title_and_repeated_numeric_columns():
    rows = [{"text": "BÁO CÁO TÌNH HÌNH TÀI CHÍNH", "bbox": [10, 10, 200, 30]}]
    for y in (120, 140, 160, 180):
        rows.extend(
            [
                {"text": "Tiền", "bbox": [30, y, 80, y + 10]},
                {"text": "1.250.000", "bbox": [300, y, 380, y + 10]},
                {"text": "1.000.000", "bbox": [450, y, 530, y + 10]},
            ]
        )
    region = _statement_region(rows, pymupdf.Rect(0, 0, 600, 800))
    assert region is not None
    assert region[1] > 30
    assert _statement_region(rows[1:], pymupdf.Rect(0, 0, 600, 800)) is None


def test_local_model_required_without_download(tmp_path):
    backend = SLANetTableBackend(model_path=tmp_path / "absent.onnx")
    # Missing optional package OR local weights both fail with a clear domain error.
    with pytest.raises(TableStructureError):
        backend._load_engine()
