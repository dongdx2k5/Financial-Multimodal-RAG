from financial_rag.ingestion import Chunker
from financial_rag.models import Document, PageContent
from financial_rag.quality import check_facts


def create_page_1():
    # Width = 600, Height = 800
    # Date headers at x > 330, y < 240
    words = [
        # Date header 1: 31/12/2024
        [360, 100, 420, 115, "31/12/2024", 0, 0, 0],
        # Date header 2: 01/01/2024
        [460, 100, 520, 115, "01/01/2024", 0, 1, 0],
        # Row 1: Code 100 (Tài sản ngắn hạn)
        [50, 180, 200, 195, "Tài sản ngắn hạn", 0, 2, 0],
        [270, 180, 290, 195, "100", 0, 3, 0],  # x around 280 (between 0.43*600=258 and 0.50*600=300)
        [360, 180, 420, 195, "100.000.000", 0, 4, 0],
        [460, 180, 520, 195, "90.000.000", 0, 5, 0],
        # Row 2: Code 280 (Tổng cộng tài sản)
        [50, 240, 200, 255, "Tổng cộng tài sản", 0, 6, 0],
        [270, 240, 290, 255, "280", 0, 7, 0],
        [360, 240, 420, 255, "200.000.000", 0, 8, 0],
        [460, 240, 520, 255, "180.000.000", 0, 9, 0],
    ]
    text = (
        "BÁO CÁO TÌNH HÌNH TÀI CHÍNH (BẢNG CÂN ĐỐI KẾ TOÁN)\n"
        "Đơn vị tính: VND\n"
        "31/12/2024  01/01/2024\n"
        "Tài sản ngắn hạn 100 100.000.000 90.000.000\n"
        "Tổng cộng tài sản 280 200.000.000 180.000.000\n"
    )
    return PageContent(page=1, text=text, words=words, width=600.0, height=800.0)


def create_page_2():
    # Continuation page (Page 2): NO date headers in header area, only rows
    words = [
        # Row 3: Code 300 (Nợ phải trả)
        [50, 100, 200, 115, "Nợ phải trả", 0, 0, 0],
        [270, 100, 290, 115, "300", 0, 1, 0],
        [360, 100, 420, 115, "50.000.000", 0, 2, 0],
        [460, 100, 520, 115, "40.000.000", 0, 3, 0],
        # Row 4: Code 400 (Vốn chủ sở hữu)
        [50, 160, 200, 175, "Vốn chủ sở hữu", 0, 4, 0],
        [270, 160, 290, 175, "400", 0, 5, 0],
        [360, 160, 420, 175, "150.000.000", 0, 6, 0],
        [460, 160, 520, 175, "140.000.000", 0, 7, 0],
        # Row 5: Code 440 (Tổng cộng nguồn vốn)
        [50, 220, 200, 235, "Tổng cộng nguồn vốn", 0, 8, 0],
        [270, 220, 290, 235, "440", 0, 9, 0],
        [360, 220, 420, 235, "200.000.000", 0, 10, 0],
        [460, 220, 520, 235, "180.000.000", 0, 11, 0],
    ]
    text = (
        "BÁO CÁO TÌNH HÌNH TÀI CHÍNH (TIẾP THEO)\n"
        "Nợ phải trả 300 50.000.000 40.000.000\n"
        "Vốn chủ sở hữu 400 150.000.000 140.000.000\n"
        "Tổng cộng nguồn vốn 440 200.000.000 180.000.000\n"
    )
    return PageContent(page=2, text=text, words=words, width=600.0, height=800.0)


def test_multi_page_table_stitching():
    doc = Document(id="doc_test_split", source="report.pdf", pages=[create_page_1(), create_page_2()], scope="separate")

    chunker = Chunker(size=900, overlap=120)
    chunks = chunker.split(doc)

    statement_chunks = [c for c in chunks if c.financial_rows]
    # Page 1 has 2 rows (100, 280), Page 2 has 3 rows (300, 400, 440) -> Total 5 rows!
    assert len(statement_chunks) == 5

    page1_chunks = [c for c in statement_chunks if c.page == 1]
    page2_chunks = [c for c in statement_chunks if c.page == 2]

    assert len(page1_chunks) == 2
    assert len(page2_chunks) == 3

    # Check metric codes extracted from continuation page
    page2_codes = [c.financial_rows[0]["metric_code"] for c in page2_chunks]
    assert page2_codes == ["300", "400", "440"]

    # Check warning on continuation page
    assert any("nối tiếp bảng" in w for w in doc.pages[1].warnings)

    # Verify accounting balance: 280 (200.000.000) == 440 (200.000.000)
    from financial_rag.extraction import FinancialFactExtractor
    extractor = FinancialFactExtractor()
    facts = [f for c in statement_chunks for f in extractor.extract(c)]
    checks = check_facts(facts)

    balance_checks = [c for c in checks if c["statement"] == "balance" and c.get("target_code") == "280"]
    # Check 280 = 440 should pass
    passed = [c for c in balance_checks if c.get("status") == "pass" and "440" in c.get("formula", "")]
    assert len(passed) >= 1


def test_cashflow_does_not_clear_context_at_code_60():
    p1 = PageContent(
        page=1,
        text="BÁO CÁO LƯU CHUYỂN TIỀN TỆ\nBa tháng\nĐơn vị tính: VND",
        width=600.0,
        height=800.0,
        words=[
            [360, 100, 420, 115, "31/03/2026", 0, 0, 0],
            [460, 100, 520, 115, "31/03/2025", 0, 1, 0],
            [50, 200, 200, 215, "Tiền đầu kỳ", 0, 2, 0],
            [270, 200, 290, 215, "60", 0, 3, 0],
            [360, 200, 420, 215, "100.000", 0, 4, 0],
            [460, 200, 520, 215, "80.000", 0, 5, 0],
        ],
    )
    p2 = PageContent(
        page=2,
        text="Tiền cuối kỳ 70 120.000 90.000",
        width=600.0,
        height=800.0,
        words=[
            [50, 30, 200, 45, "Tiền cuối kỳ", 0, 0, 0],
            [270, 30, 290, 45, "70", 0, 1, 0],
            [360, 30, 420, 45, "120.000", 0, 2, 0],
            [460, 30, 520, 45, "90.000", 0, 3, 0],
        ],
    )
    doc = Document(id="cf-split", source="cf.pdf", pages=[p1, p2], scope="separate")
    chunks = [c for c in Chunker(900, 120).split(doc) if c.financial_rows]
    assert [c.financial_rows[0]["metric_code"] for c in chunks] == ["60", "70"]
    assert chunks[1].page == 2
