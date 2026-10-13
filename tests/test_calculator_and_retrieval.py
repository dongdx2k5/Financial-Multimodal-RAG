from pathlib import Path

import pytest

from financial_rag.calculator import SafeCalculator, process_calculations
from financial_rag.config import Settings
from financial_rag.retrieval import extract_query_metadata
from financial_rag.service import FinancialRAGService


def test_safe_calculator_basic_and_financial():
    # Simple arithmetic
    assert SafeCalculator.evaluate("100 + 25.5") == 125.5
    assert SafeCalculator.evaluate("200 - 50.5") == 149.5
    assert SafeCalculator.evaluate("10 * 12.5") == 125.0
    assert SafeCalculator.evaluate("100 / 4") == 25.0

    # Financial growth rate calculation
    growth = SafeCalculator.evaluate("(125.5 - 110.0) / 110.0 * 100")
    assert round(growth, 2) == 14.09

    # Financial margin calculation
    margin = SafeCalculator.evaluate("18.2 / 125.5 * 100")
    assert round(margin, 2) == 14.50

    # Division by zero handling
    with pytest.raises(ZeroDivisionError):
        SafeCalculator.evaluate("100 / 0")

    # Security: blocks function calls and code execution
    with pytest.raises(ValueError):
        SafeCalculator.evaluate("__import__('os').system('ls')")

    with pytest.raises(ValueError):
        SafeCalculator.evaluate("eval('2 + 2')")


def test_process_calculations_tag():
    text = (
        "Doanh thu tăng trưởng <<<CALC: (125.5 - 110.0) / 110.0 * 100>>>% "
        "và biên lợi nhuận ròng là <<<CALC: 18.2 / 125.5 * 100>>>%."
    )
    processed, calcs = process_calculations(text)

    assert len(calcs) == 2
    assert "14.09" in processed
    assert "14.50" in processed
    assert calcs[0]["expression"] == "(125.5 - 110.0) / 110.0 * 100"
    assert round(calcs[0]["result"], 2) == 14.09


def test_query_metadata_extraction():
    known = {"DEMO", "AAPL", "MSFT"}

    ticker, year, _ = extract_query_metadata("Doanh thu thuần của DEMO năm 2024 là bao nhiêu?", known)
    assert ticker == "DEMO"
    assert year == 2024

    ticker_en, year_en, _ = extract_query_metadata("What was AAPL Total Revenue in 2023?", known)
    assert ticker_en == "AAPL"
    assert year_en == 2023


def test_metadata_filtered_retrieval(tmp_path: Path):
    data_dir = tmp_path / "data"
    demo_file = tmp_path / "demo.txt"
    demo_file.write_text(
        "===== PAGE 1 =====\nCông ty DEMO: Doanh thu thuần năm 2024 là 125,5 tỷ đồng.",
        encoding="utf-8",
    )
    aapl_file = tmp_path / "aapl.txt"
    aapl_file.write_text(
        "===== PAGE 1 =====\nApple Inc. (AAPL): Total Revenue in 2024 was $383.3 billion.",
        encoding="utf-8",
    )

    service = FinancialRAGService(Settings(data_dir=data_dir))
    service.ingest(demo_file, ticker="DEMO", year=2024)
    service.ingest(aapl_file, ticker="AAPL", year=2024)

    # Ask specifically about DEMO
    res_demo = service.ask("Doanh thu của DEMO", top_k=1)
    assert len(res_demo.evidence) == 1
    assert "DEMO" in res_demo.evidence[0].text

    # Ask specifically about AAPL
    res_aapl = service.ask("Total Revenue of AAPL", top_k=1)
    assert len(res_aapl.evidence) == 1
    assert "AAPL" in res_aapl.evidence[0].text
