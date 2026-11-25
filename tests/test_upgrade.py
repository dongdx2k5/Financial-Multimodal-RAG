from pathlib import Path

import pymupdf
import pytest
from fastapi.testclient import TestClient

from financial_rag import api
from financial_rag.calculator import calculate
from financial_rag.config import Settings
from financial_rag.extraction import FinancialFactExtractor
from financial_rag.models import Chunk
from financial_rag.service import FinancialRAGService


def test_pdf_upload_persists_and_repeat_is_idempotent(tmp_path, monkeypatch):
    service = FinancialRAGService(Settings(data_dir=tmp_path))
    monkeypatch.setattr(api, "service", service)
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((40, 40), "Financial report 2024. Revenue: 1234 million. " * 3)
    content = pdf.tobytes()
    pdf.close()
    with TestClient(api.app) as client:
        result = client.post("/ingest", files={"file": ("report.pdf", content, "application/pdf")})
        assert result.status_code == 200
        report = result.json()
        count = service.graph.graph.number_of_edges()
        again = client.post("/ingest", files={"file": ("report.pdf", content, "application/pdf")})
        assert again.json()["document_id"] == report["document_id"]
        assert service.graph.graph.number_of_edges() == count
        assert client.get(report["source_url"]).content == content
        answer = client.post(
            "/ask", json={"question": "Revenue", "document_id": report["document_id"]}
        )
        assert answer.json()["evidence"]
        assert client.get("/").status_code == 200
        assert client.get("/documents/unknown/file").status_code == 404
    reloaded = FinancialRAGService(Settings(data_dir=tmp_path))
    assert reloaded.ask("Revenue").evidence
    assert Path(reloaded.retriever.chunks[0].source).exists()


def test_scan_fails_explicitly_without_ocr(tmp_path):
    pdf = pymupdf.open()
    pdf.new_page()
    path = tmp_path / "blank.pdf"
    pdf.save(path)
    pdf.close()
    service = FinancialRAGService(Settings(data_dir=tmp_path / "data", ocr_enabled=False))
    with pytest.raises(ValueError, match="OCR"):
        service.ingest(path)


def test_scanned_pdf_with_digital_signature_only_fails_without_ocr(tmp_path):
    pdf = pymupdf.open()
    sample_page = pdf.new_page()
    sample_page.insert_text((40, 40), "Sample scan")
    img_bytes = sample_page.get_pixmap().tobytes("png")
    pdf.close()

    scanned_pdf = pymupdf.open()
    for i in range(3):
        p = scanned_pdf.new_page()
        p.insert_image(p.rect, stream=img_bytes)
        if i == 0:
            p.insert_text((40, 40), "Digitally signed by DEMO")
    path = tmp_path / "signed_scan.pdf"
    scanned_pdf.save(path)
    scanned_pdf.close()

    service = FinancialRAGService(Settings(data_dir=tmp_path / "data", ocr_enabled=False))
    with pytest.raises(ValueError, match="OCR_ENABLED=true"):
        service.ingest(path)


def test_table_alignment_negative_values_and_calculator():
    chunk = Chunk(
        id="table",
        document_id="doc",
        source="report.pdf",
        page=2,
        text="Revenue",
        table=[["Metric", "2024", "2023"], ["Revenue", "1.250", "1.000"], ["Loss", "(20)", "30"]],
        table_unit="triệu",
        ticker="ABC",
    )
    facts = FinancialFactExtractor().extract(chunk)
    assert [(f.period, f.value) for f in facts[:2]] == [(2024, 1250), (2023, 1000)]
    assert facts[2].value == -20
    assert calculate("growth", facts[0], facts[1])["value"] == "25.00"
    with pytest.raises(ValueError, match="Đơn vị"):
        calculate("growth", facts[0].model_copy(update={"unit": None}), facts[1])
    with pytest.raises(ValueError, match="chia cho 0"):
        calculate("growth", facts[0], facts[1].model_copy(update={"value": 0}))


def test_document_filter_and_unconfigured_colab(tmp_path):
    service = FinancialRAGService(Settings(data_dir=tmp_path / "data", colab_gpu_url=None))
    for ticker, value in [("AAA", 100), ("BBB", 200)]:
        path = tmp_path / f"{ticker}.txt"
        path.write_text(f"Doanh thu năm 2024 đạt {value} tỷ đồng.", encoding="utf-8")
        report = service.ingest(path, ticker=ticker, year=2024)
    response = service.ask("Doanh thu", document_id=report["document_id"], use_llm=True)
    assert response.mode == "evidence" and response.warnings
    assert all("200" in e.text for e in response.evidence)
    # Regex facts must carry document_id so /analysis includes them.
    facts = [
        attrs
        for _, attrs in service.graph.graph.nodes(data=True)
        if attrs.get("kind") == "Fact" and attrs.get("document_id") == report["document_id"]
    ]
    assert facts and facts[0]["value"] == 200.0


def test_pdf_table_is_extracted(tmp_path):
    pdf = pymupdf.open()
    page = pdf.new_page()
    for y in [100, 130, 160]:
        page.draw_line((40, y), (400, y))
    for x in [40, 200, 300, 400]:
        page.draw_line((x, 100), (x, 160))
    for x, text in [(45, "Metric"), (205, "2024"), (305, "2023")]:
        page.insert_text((x, 120), text)
    for x, text in [(45, "Revenue"), (205, "1250"), (305, "1000")]:
        page.insert_text((x, 150), text)
    path = tmp_path / "table.pdf"
    pdf.save(path)
    pdf.close()
    service = FinancialRAGService(Settings(data_dir=tmp_path / "data"))
    result = service.ingest(path)
    assert result["tables"] == 1
    table_chunks = [c for c in service.retriever.chunks if c.table]
    facts = service.graph.facts_for_chunk(table_chunks[0].id)
    assert [(f.period, f.value) for f in facts] == [(2024, 1250), (2023, 1000)]


def test_cross_unit_currency_calculation():
    from financial_rag.models import FinancialFact

    left = FinancialFact(
        id="l",
        metric="Doanh thu",
        value=2.0,
        raw_value="2",
        unit="tỷ",
        period=2025,
        chunk_id="c1",
        source="r.pdf",
        page=1,
        confidence=0.9,
        ticker="ABC",
    )
    right = left.model_copy(update={"id": "r", "value": 1000.0, "raw_value": "1000", "unit": "triệu", "period": 2024})
    assert calculate("growth", left, right)["value"] == "100"
    assert calculate("difference", left, right)["value"] == "1.0"
