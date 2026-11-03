from types import SimpleNamespace

import pymupdf
import pytest

from financial_rag.extraction import FinancialFactExtractor
from financial_rag.ingestion import Chunker, DocumentLoader
from financial_rag.models import Document, PageContent, StructuredTable
from financial_rag.ocr_backends import OCREngineUnavailable, OCRLine, OCRResult


def test_default_does_not_load_optional_ocr():
    loader = DocumentLoader()
    assert loader.ocr_backend == "tesseract"
    assert loader.table_backend == "none"
    assert loader._vietocr is None


def test_native_pdf_text_skips_vietocr(monkeypatch):
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((40, 40), "Native financial report with enough readable text.")
        loader = DocumentLoader(ocr_enabled=True, ocr_backend="vietocr")
        loader._vietocr = SimpleNamespace(
            extract=lambda *a, **k: (_ for _ in ()).throw(AssertionError())
        )
        content, cacheable = loader._extract_page(page)
        assert content.extraction_method == "text"
        assert content.ocr_engine is None
        assert cacheable


def test_vietocr_keeps_line_boxes_and_provenance():
    with pymupdf.open() as pdf:
        loader = DocumentLoader(ocr_enabled=True, ocr_backend="vietocr")
        loader._vietocr = SimpleNamespace(
            extract=lambda *a, **k: OCRResult(
                [
                    OCRLine("Tiền và tương đương tiền", [10, 20, 200, 30], 0.9),
                    OCRLine("(1.250.000)", [250, 20, 310, 30], 0.8),
                ]
            )
        )
        content, cacheable = loader._extract_page(pdf.new_page())
        assert content.ocr_engine == "vietocr"
        assert content.extraction_method == "ocr"
        assert len(content.words) == 2
        assert content.words[1][:5] == [250, 20, 310, 30, "(1.250.000)"]
        assert content.ocr_lines[0].confidence == 0.9
        assert cacheable


def test_missing_vietocr_has_explicit_fallback_and_is_not_cached():
    with pymupdf.open() as pdf:
        loader = DocumentLoader(ocr_enabled=True, ocr_backend="vietocr")

        def unavailable(*args, **kwargs):
            raise OCREngineUnavailable("missing model")

        loader._vietocr = SimpleNamespace(extract=unavailable)
        loader._tesseract = lambda page: ("Fallback text", [[10, 20, 100, 30, "Fallback", 0, 0, 0]])
        content, cacheable = loader._extract_page(pdf.new_page())
        assert content.ocr_engine == "tesseract"
        assert any("dự phòng" in warning for warning in content.warnings)
        assert not cacheable


def test_strict_mode_does_not_silently_run_tesseract():
    with pymupdf.open() as pdf:
        loader = DocumentLoader(
            ocr_enabled=True, ocr_backend="vietocr", fallback_to_tesseract=False
        )
        with pytest.raises(RuntimeError, match="vietocr thất bại"):
            loader._extract_page(pdf.new_page())


def test_cache_is_separate_for_backend_and_model_changes(tmp_path):
    model = tmp_path / "test.pth"
    model.write_bytes(b"old")
    default = DocumentLoader()
    experimental = DocumentLoader(ocr_backend="vietocr", vietocr_weights=model)
    old_signature = experimental._cache_signature()
    assert old_signature != default._cache_signature()
    model.write_bytes(b"new larger model")
    assert old_signature != experimental._cache_signature()
    assert (
        default._cache_signature() != DocumentLoader(table_backend="slanet_plus")._cache_signature()
    )


def test_structure_chunks_have_coordinates_and_cannot_become_financial_facts():
    table = StructuredTable(
        bbox=[10, 20, 300, 80],
        cells=[
            {
                "row": 0,
                "col": 0,
                "text": "Doanh thu năm 2024 đạt 125 tỷ đồng",
                "bbox": [10, 20, 300, 40],
            },
        ],
    )
    document = Document(
        id="doc",
        source="report.pdf",
        pages=[
            PageContent(
                page=2,
                text="",
                extraction_method="ocr",
                structured_tables=[table],
            )
        ],
    )
    chunks = Chunker().split(document)
    assert len(chunks) == 1
    assert chunks[0].bbox == table.bbox
    assert chunks[0].table_structure_engine == "slanet_plus"
    assert chunks[0].page == 2
    assert FinancialFactExtractor().extract(chunks[0]) == []


def test_native_table_chunks_keep_existing_extraction_and_now_have_bbox():
    document = Document(
        id="doc",
        source="report.pdf",
        pages=[
            PageContent(
                page=1,
                text="",
                tables=[[["Metric", "2024"], ["Revenue", "123"]]],
                table_bboxes=[[10, 20, 200, 100]],
            )
        ],
    )
    chunk = Chunker().split(document)[0]
    assert chunk.bbox == [10, 20, 200, 100]
    assert FinancialFactExtractor().extract(chunk)[0].value == 123
