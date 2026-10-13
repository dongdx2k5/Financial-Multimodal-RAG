from pathlib import Path

from financial_rag.config import Settings
from financial_rag.extraction import FinancialFactExtractor, parse_vietnamese_number
from financial_rag.ingestion import Chunker, DocumentLoader
from financial_rag.service import FinancialRAGService


def test_vietnamese_number_parser():
    assert parse_vietnamese_number("1.234,5") == 1234.5
    assert parse_vietnamese_number("125,5") == 125.5


def test_load_chunk_and_extract(tmp_path: Path):
    report = tmp_path / "report.txt"
    report.write_text(
        "===== PAGE 1 =====\nDoanh thu thuần năm 2024 đạt 125,5 tỷ đồng.", encoding="utf-8"
    )
    document = DocumentLoader().load(report, ticker="ABC", year=2024)
    chunks = Chunker(200, 20).split(document)
    facts = FinancialFactExtractor().extract(chunks[0])
    assert document.pages[0].page == 1
    assert len(chunks) == 1
    assert facts[0].value == 125.5
    assert facts[0].unit == "tỷ"
    assert facts[0].metric == "Doanh thu thuần"


def test_end_to_end_retrieval(tmp_path: Path):
    data_dir = tmp_path / "data"
    report = tmp_path / "report.txt"
    report.write_text("Doanh thu thuần năm 2024 đạt 125,5 tỷ đồng.", encoding="utf-8")
    service = FinancialRAGService(Settings(data_dir=data_dir))
    stats = service.ingest(report, ticker="ABC", year=2024)
    response = service.ask("Doanh thu thuần 2024", top_k=1)
    assert stats["facts"] >= 1
    assert response.evidence
    assert "125,5" in response.answer
