"""Regressions for experimental OCR boundaries, without model downloads."""

from types import SimpleNamespace

import pymupdf
import pytest

from financial_rag.extraction import FinancialFactExtractor
from financial_rag.ingestion import Chunker, DocumentLoader
from financial_rag.models import Chunk, Document, PageContent
from financial_rag.ocr_backends import OCREngineUnavailable, OCRLine, OCRResult, VietOCRBackend
from financial_rag.table_structure import SLANetTableBackend, TableStructureError


def test_vietocr_line_spanning_two_numeric_columns_is_evidence_only():
    # The legacy word parser selects a column by the box center and strips
    # spaces. Feeding this whole line to it could turn two values into 10002000.
    page = PageContent(
        page=1,
        text="BẢNG CÂN ĐỐI KẾ TOÁN\nVND\n31/03/2026 31/12/2025\nTiền 110 1000 2000",
        width=600,
        height=800,
        extraction_method="ocr",
        ocr_engine="vietocr",
        words=[
            [360, 80, 400, 100, "31/03/2026", 0, 0, 0],
            [460, 80, 500, 100, "31/12/2025", 0, 0, 1],
            [30, 200, 100, 210, "Tiền", 0, 1, 0],
            [270, 200, 290, 210, "110", 0, 1, 1],
            [350, 200, 500, 210, "1000 2000", 0, 1, 2],
        ],
    )
    document = Document(id="mixed-columns", source="report.pdf", pages=[page])

    chunks = Chunker().split(document)

    assert chunks
    assert any("1000 2000" in chunk.text for chunk in chunks)
    assert all(chunk.ocr_engine == "vietocr" for chunk in chunks)
    assert not any(chunk.financial_rows for chunk in chunks)
    assert not [fact for chunk in chunks for fact in FinancialFactExtractor().extract(chunk)]


@pytest.mark.parametrize(
    "fields",
    [
        {"text": "Doanh thu năm 2026 đạt 125 tỷ đồng"},
        {
            "text": "Doanh thu | 2026 | 125",
            "table": [["Metric", "2026"], ["Revenue", "125"]],
        },
        {
            "text": "Tiền 2026-03-31: 10002000 VND",
            "financial_rows": [
                {
                    "metric": "Tiền",
                    "value": 10002000,
                    "raw_value": "1000 2000",
                    "period": 2026,
                    "period_label": "2026-03-31",
                    "unit": "VND",
                }
            ],
        },
    ],
    ids=["regex", "table", "legacy-financial-rows"],
)
def test_vietocr_provenance_blocks_alternative_fact_extraction_paths(fields):
    chunk = Chunk(
        id="experimental",
        document_id="doc",
        source="report.pdf",
        page=1,
        extraction_method="ocr",
        ocr_engine="vietocr",
        **fields,
    )
    extractor = FinancialFactExtractor()
    # Verify that each fixture would reach a real extraction path without the
    # experimental provenance; otherwise an empty result could hide a weak test.
    assert extractor.extract(chunk.model_copy(update={"ocr_engine": None}))
    assert extractor.extract(chunk) == []


def test_strict_vietocr_fails_mixed_pdf_instead_of_returning_partial_document(tmp_path):
    path = tmp_path / "mixed.pdf"
    with pymupdf.open() as doc:
        native = doc.new_page()
        native.insert_text((40, 40), "Native financial report with enough readable text.")
        scan_image = native.get_pixmap().tobytes("png")
        scanned = doc.new_page()
        scanned.insert_image(scanned.rect, stream=scan_image)
        doc.save(path)

    calls = []
    failure = OCREngineUnavailable("model unavailable")

    def fail_ocr(page, **kwargs):
        calls.append(page.number + 1)
        raise failure

    loader = DocumentLoader(
        ocr_enabled=True, ocr_backend="vietocr", fallback_to_tesseract=False
    )
    loader._vietocr = SimpleNamespace(extract=fail_ocr)
    loader._tesseract = lambda page: pytest.fail("Strict mode must not use Tesseract")
    progress = []

    with pytest.raises(RuntimeError, match="Trang 2") as caught:
        loader.load(path, progress=lambda current, total: progress.append((current, total)))

    assert caught.value.__cause__ is failure
    assert calls == [2]
    assert progress == [(1, 2)]


def test_table_native_runtime_error_is_wrapped_and_rotation_restored():
    class NativeRuntimeFailure(Exception):
        """Emulate ONNX errors that do not inherit Python RuntimeError."""

    failure = NativeRuntimeFailure("inference failed")

    def fail_inference(image):
        raise failure

    backend = SLANetTableBackend(dpi=72)
    backend._engine = fail_inference
    with pymupdf.open() as doc:
        page = doc.new_page(width=200, height=300)
        page.set_rotation(90)
        with pytest.raises(TableStructureError, match="inference failed") as caught:
            backend.extract_tables(page, [], regions=[[10, 10, 100, 100]])
        assert caught.value.__cause__ is failure
        assert page.rotation == 90


@pytest.mark.parametrize("rotation", [90, 180, 270])
@pytest.mark.parametrize("fail", [False, True], ids=["success", "failure"])
def test_vietocr_receives_upright_page_and_restores_pdf_rotation(rotation, fail):
    backend = VietOCRBackend()
    expected = OCRResult([OCRLine("1.250.000", [10, 20, 90, 30], 0.9)])
    failure = OCREngineUnavailable("recognizer failed")

    def recognize_upright(page, dpi):
        assert page.rotation == 0
        assert list(page.rect) == [0, 0, 200, 300]
        assert dpi == 144
        if fail:
            raise failure
        return expected

    backend._extract_upright = recognize_upright
    with pymupdf.open() as doc:
        page = doc.new_page(width=200, height=300)
        page.set_rotation(rotation)
        if fail:
            with pytest.raises(OCREngineUnavailable) as caught:
                backend.extract(page, dpi=144)
            assert caught.value is failure
        else:
            assert backend.extract(page, dpi=144) is expected
        assert page.rotation == rotation
