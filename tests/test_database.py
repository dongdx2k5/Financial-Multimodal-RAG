from pathlib import Path

from financial_rag.config import Settings
from financial_rag.extraction import FinancialFactExtractor, parse_number
from financial_rag.ingestion import Chunker, DocumentLoader
from financial_rag.service import FinancialRAGService


def test_english_number_and_fact_extraction(tmp_path: Path):
    assert parse_number("1,234.5") == 1234.5
    assert parse_number("250.8") == 250.8
    assert parse_number("$45.2") == 45.2

    report = tmp_path / "apple_report.txt"
    report.write_text(
        "===== PAGE 1 =====\n"
        "Total Revenue in 2024 was $383.3 billion.\n"
        "Net Income in 2024 was $93.7 billion.",
        encoding="utf-8",
    )
    doc = DocumentLoader().load(report, ticker="AAPL", year=2024)
    chunks = Chunker(300, 30).split(doc)
    facts = FinancialFactExtractor().extract(chunks[0])

    assert len(facts) >= 2
    metrics = {f.metric: f.value for f in facts}
    assert "Total Revenue" in metrics or "Revenue" in metrics
    assert any(f.value == 383.3 for f in facts)
    assert any(f.value == 93.7 for f in facts)


def test_database_persistence_and_queries(tmp_path: Path):
    data_dir = tmp_path / "data"
    report = tmp_path / "report_en.txt"
    report.write_text(
        "===== PAGE 1 =====\n"
        "Microsoft Corporation (MSFT)\n"
        "Total Revenue in 2024 was $245.1 billion. Total Assets stood at 512.2 billion.\n",
        encoding="utf-8",
    )

    settings = Settings(data_dir=data_dir)
    service = FinancialRAGService(settings)

    # 1. Ingest
    res = service.ingest(report, ticker="MSFT", year=2024)
    assert res["chunks"] > 0
    assert res["facts"] >= 1

    # 2. List documents
    docs = service.list_documents()
    assert len(docs) == 1
    assert docs[0]["ticker"] == "MSFT"
    assert docs[0]["year"] == 2024

    # 3. Query facts directly from SQL database
    facts = service.get_facts(ticker="MSFT", year=2024)
    assert len(facts) >= 1
    assert any(f.ticker == "MSFT" for f in facts)

    # 4. Check stats
    stats = service.get_stats()
    assert stats["documents"] == 1
    assert stats["chunks"] >= 1
    assert stats["facts"] >= 1

    # 5. Hybrid search / Ask
    ask_res = service.ask("What was Total Revenue in 2024?")
    assert ask_res.evidence
    assert "245.1" in ask_res.answer

    # 6. Delete document
    deleted = service.delete_document(res["document_id"])
    assert deleted is True
    assert len(service.list_documents()) == 0
    assert len(service.get_facts(ticker="MSFT")) == 0
