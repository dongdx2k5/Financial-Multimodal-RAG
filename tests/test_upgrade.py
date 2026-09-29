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
    service = FinancialRAGService(Settings(data_dir=tmp_path, openai_api_key=None))
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
    reloaded = FinancialRAGService(Settings(data_dir=tmp_path, openai_api_key=None))
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


def test_document_filter_and_missing_llm_key(tmp_path):
    service = FinancialRAGService(
        Settings(data_dir=tmp_path / "data", openai_api_key=None, llm_provider="openai")
    )
    for ticker, value in [("AAA", 100), ("BBB", 200)]:
        path = tmp_path / f"{ticker}.txt"
        path.write_text(f"Doanh thu năm 2024 đạt {value} tỷ đồng.", encoding="utf-8")
        report = service.ingest(path, ticker=ticker, year=2024)
    response = service.ask("Doanh thu", document_id=report["document_id"], use_llm=True)
    assert response.mode == "evidence" and response.warnings
    assert all("200" in e.text for e in response.evidence)


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


def test_llm_adapter_and_invalid_citation_fallback(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import openai

    calls = []

    class Client:
        def __init__(self, **kwargs):
            self.responses = self

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(output_text="Doanh thu 100 tỷ [1].")

    monkeypatch.setattr(openai, "OpenAI", Client)
    service = FinancialRAGService(
        Settings(data_dir=tmp_path / "data", openai_api_key="test-only", llm_provider="openai")
    )
    path = tmp_path / "report.txt"
    path.write_text("Doanh thu năm 2024 đạt 100 tỷ đồng.", encoding="utf-8")
    service.ingest(path)
    assert service.ask("Doanh thu", use_llm=True).mode == "llm"
    assert calls[0]["store"] is False
    monkeypatch.setattr(
        Client, "create", lambda self, **kwargs: SimpleNamespace(output_text="Sai [99]")
    )
    response = service.ask("Doanh thu", use_llm=True)
    assert response.mode == "evidence" and response.warnings
