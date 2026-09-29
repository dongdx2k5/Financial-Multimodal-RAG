from types import SimpleNamespace

import networkx as nx
import pytest

from financial_rag.answers import factual_answer
from financial_rag.models import Chunk, FinancialFact


def fake_service():
    graph = nx.MultiDiGraph()
    chunks = []
    for date, value in [("2026-01-01", 100), ("2026-03-31", 120)]:
        chunk = Chunk(
            id=date,
            document_id="report",
            source="report.pdf",
            page=8,
            text="Tổng tài sản",
            financial_rows=[{"value": value}],
        )
        fact = FinancialFact(
            id="fact" + date,
            metric="Tổng tài sản",
            value=value,
            raw_value=str(value),
            unit="VND",
            period=2026,
            chunk_id=date,
            source="report.pdf",
            page=8,
            confidence=0.7,
            document_id="report",
            period_label=date,
            period_kind="instant",
            scope="separate",
            metric_code="280",
            statement="balance",
        )
        graph.add_node(fact.id, kind="Fact", **fact.model_dump(exclude={"id"}))
        chunks.append(chunk)
    return SimpleNamespace(
        graph=SimpleNamespace(graph=graph), retriever=SimpleNamespace(chunks=chunks)
    )


@pytest.mark.parametrize(
    "question, expected",
    [
        ("Tổng tài sản đầu năm 2026", "100 VND"),
        ("Tổng tài sản cuối kỳ 2026", "120 VND"),
        ("Tổng tài sản ngày 1/1/2026", "100 VND"),
    ],
)
def test_exact_dates(question, expected):
    result = factual_answer(fake_service(), question, "report")
    assert result.mode == "structured"
    assert expected in result.answer
    assert result.evidence[0].page == 8


@pytest.mark.parametrize(
    "question",
    [
        "Tổng tài sản 2024",
        "Tổng tài sản hợp nhất 2026",
        "Tổng tài sản ngày 30/6/2026",
    ],
)
def test_wrong_period_or_scope(question):
    result = factual_answer(fake_service(), question, "report")
    assert result.mode == "evidence"
    assert not result.evidence


def test_review_blocks_numeric_answer():
    service = fake_service()
    service.graph.graph.nodes["fact2026-03-31"]["quality"] = "needs_review"
    assert factual_answer(service, "Tổng tài sản", "report").mode == "evidence"


def test_multi_metric_does_not_silently_drop_question():
    assert factual_answer(fake_service(), "Tổng tài sản và nợ phải trả", "report") is None


def test_policy_question_routes_to_narrative():
    assert (
        factual_answer(fake_service(), "Tóm tắt chính sách ghi nhận tổng tài sản", "report") is None
    )
