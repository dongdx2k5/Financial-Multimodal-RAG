from pathlib import Path

from financial_rag.extraction import FinancialFactExtractor, parse_number
from financial_rag.ingestion import Chunker, DocumentLoader
from financial_rag.models import Chunk


def test_accounting_negative_number_parser():
    # Vietnamese formats
    assert parse_number("125,5") == 125.5
    assert parse_number("(125,5)") == -125.5
    assert parse_number("( 1.234,5 )") == -1234.5
    assert parse_number("-18,2") == -18.2

    # English / International formats
    assert parse_number("1,234.5") == 1234.5
    assert parse_number("(1,234.5)") == -1234.5
    assert parse_number("( 45.2 )") == -45.2
    assert parse_number("($45.2)") == -45.2
    assert parse_number("-45.2") == -45.2


def test_extract_negative_facts_from_text():
    extractor = FinancialFactExtractor()

    # Vietnamese negative fact
    vn_chunk = Chunk(
        id="c1",
        document_id="d1",
        source="doc.txt",
        page=1,
        text="Lợi nhuận sau thuế năm 2024 đạt (18,2) tỷ đồng.",
    )
    vn_facts = extractor.extract(vn_chunk)
    assert len(vn_facts) >= 1
    assert any(f.metric == "Lợi nhuận sau thuế" and f.value == -18.2 for f in vn_facts)

    # English negative fact
    en_chunk = Chunk(
        id="c2",
        document_id="d2",
        source="doc_en.txt",
        page=1,
        text="Net Income in 2024 was ($4.5) billion.",
    )
    en_facts = extractor.extract(en_chunk)
    assert len(en_facts) >= 1
    assert any("Net Income" in f.metric and f.value == -4.5 for f in en_facts)


def test_extract_facts_from_markdown_table():
    extractor = FinancialFactExtractor()

    table_text = (
        "===== PAGE 1 =====\n"
        "BÁO CÁO KẾT QUẢ HOẠT ĐỘNG KINH DOANH\n\n"
        "| Chỉ tiêu | Năm 2024 | Năm 2023 |\n"
        "| --- | --- | --- |\n"
        "| Doanh thu thuần | 125,5 tỷ | 110,0 tỷ |\n"
        "| Lợi nhuận sau thuế | (18,2) tỷ | 15,0 tỷ |\n"
        "| Tổng tài sản | 450 tỷ | 410 tỷ |\n"
    )
    chunk = Chunk(
        id="c3",
        document_id="d3",
        source="report.txt",
        page=1,
        text=table_text,
        ticker="DEMO",
    )
    facts = extractor.extract(chunk)
    assert len(facts) >= 6

    # Verify 2024 facts
    facts_2024 = {f.metric: f.value for f in facts if f.period == 2024}
    assert facts_2024.get("Doanh thu thuần") == 125.5
    assert facts_2024.get("Lợi nhuận sau thuế") == -18.2
    assert facts_2024.get("Tổng tài sản") == 450.0

    # Verify 2023 facts
    facts_2023 = {f.metric: f.value for f in facts if f.period == 2023}
    assert facts_2023.get("Doanh thu thuần") == 110.0
    assert facts_2023.get("Lợi nhuận sau thuế") == 15.0
    assert facts_2023.get("Tổng tài sản") == 410.0


def test_html_table_ingestion_and_extraction(tmp_path: Path):
    html_file = tmp_path / "financial_table.html"
    html_content = """
    <html>
      <body>
        <h1>Báo cáo tài chính</h1>
        <table>
          <tr><th>Chỉ tiêu</th><th>2024</th></tr>
          <tr><td>Doanh thu thuần</td><td>125,5 tỷ</td></tr>
          <tr><td>Lợi nhuận ròng</td><td>(10,5) tỷ</td></tr>
        </table>
      </body>
    </html>
    """
    html_file.write_text(html_content, encoding="utf-8")

    doc = DocumentLoader().load(html_file, ticker="ABC", year=2024)
    assert len(doc.pages) == 1
    # Check table was converted to markdown
    assert "| Chỉ tiêu | 2024 |" in doc.pages[0].text

    chunks = Chunker(size=500, overlap=50).split(doc)
    assert len(chunks) >= 1

    facts = FinancialFactExtractor().extract(chunks[0])
    assert len(facts) >= 2
    metrics = {f.metric: f.value for f in facts}
    assert metrics.get("Doanh thu thuần") == 125.5
    assert metrics.get("Lợi nhuận ròng") == -10.5
