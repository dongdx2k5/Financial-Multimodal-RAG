import pytest

from financial_rag.financial_tables import extract_statement, money
from financial_rag.models import Document, FinancialFact, PageContent
from financial_rag.quality import check_facts


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("(1.234)", -1234),
        ("-1.234", -1234),
        ("1.234", 1234),
        ("0", 0),
        ("-", None),
        ("(1.234", None),
        ("1.234)", None),
        ("(-1.234)", None),
        ("1.23", None),
    ],
)
def test_money_strict(raw, expected):
    assert money(raw) == expected


def fact(code, value, statement="cashflow", period="2026-Q1", document="a"):
    return FinancialFact(
        id=f"{document}:{statement}:{code}:{period}",
        metric=code,
        value=value,
        raw_value=str(value),
        unit="VND",
        period=2026,
        chunk_id=code,
        source=document + ".pdf",
        page=1,
        confidence=0.7,
        document_id=document,
        scope="separate",
        statement=statement,
        metric_code=code,
        period_label=period,
        period_kind="instant" if statement == "balance" else "quarter",
    )


def test_cash_reconciliation_and_cross_statement():
    facts = [
        fact("20", -10),
        fact("30", -20),
        fact("40", 50),
        fact("50", 20),
        fact("60", 100),
        fact("61", -1),
        fact("70", 119),
        fact("110", 100, "balance", "2026-01-01"),
        fact("110", 119, "balance", "2026-03-31"),
    ]
    checks = check_facts(facts)
    cash_checks = [c for c in checks if c["statement"] in {"cashflow", "cross_statement"}]
    assert len(cash_checks) == 4
    assert all(c["status"] == "pass" for c in cash_checks)
    facts[-1].value = 120
    assert any(c["status"] == "fail" for c in check_facts(facts))
    assert facts[-1].quality == "needs_review"


def test_no_mix_between_reports_or_duplicates():
    facts = [fact("50", 20), fact("60", 100), fact("61", -1, document="b"), fact("70", 119)]
    assert not any(c["status"] == "pass" for c in check_facts(facts))
    facts = [fact("50", 20), fact("60", 100), fact("61", -1), fact("70", 119), fact("70", 120)]
    checks = check_facts(facts)
    assert any(c.get("ambiguous_codes") == ["70"] for c in checks)
    assert facts[-1].quality == "needs_review"


def test_recovered_code_near_bottom_retains_provenance():
    page = PageContent(
        page=2,
        text="Báo cáo lưu chuyển tiền tệ riêng ba tháng VND",
        width=600,
        height=840,
        extraction_method="ocr",
        words=[
            [365, 160, 415, 170, "31/3/2026"],
            [465, 160, 515, 170, "31/3/2025"],
            [80, 680, 200, 690, "Đầu tư"],
            [279, 680, 290, 690, "??"],
            [350, 680, 430, 690, "(1.234)"],
            [450, 680, 530, 690, "2.345"],
        ],
        recovered_codes=[{"y": 685, "code": "30"}],
    )
    document = Document(id="a", source="a.pdf", pages=[page], scope="separate")
    chunks, warnings = extract_statement(page, document)
    assert len(chunks) == 1 and chunks[0].financial_rows[0]["value"] == -1234
    assert chunks[0].financial_rows[0]["recognition_note"]
    assert warnings and chunks[0].bbox[1] < 685 < chunks[0].bbox[3]
