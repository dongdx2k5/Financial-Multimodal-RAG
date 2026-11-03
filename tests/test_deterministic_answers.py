import re

import pytest
from fastapi.testclient import TestClient

from financial_rag import api
from financial_rag.config import Settings
from financial_rag.models import Chunk, Document, FinancialFact
from financial_rag.service import FinancialRAGService


@pytest.fixture
def service(tmp_path):
    service = FinancialRAGService(Settings(data_dir=tmp_path / "data", _env_file=None))
    chunks, facts = [], []
    for year, revenue in [(2026, 120), (2025, 100), (2024, 80)]:
        for code, metric, value in [
            ("10", "Doanh thu thuần", revenue),
            ("20", "Lợi nhuận gộp", revenue / 2),
            ("60", "Lợi nhuận sau thuế", revenue / 4),
        ]:
            key = f"{year}-{code}"
            chunk = Chunk(
                id=key,
                document_id="report",
                source="report.pdf",
                page=10,
                text=f"{metric}: {year}-Q1: {value} VND",
                financial_rows=[{"value": value}],
                bbox=[10, 20, 400, 40],
                scope="separate",
            )
            chunks.append(chunk)
            facts.append(
                FinancialFact(
                    id="fact-" + key,
                    metric=metric,
                    value=value,
                    raw_value=str(value),
                    unit="VND",
                    period=year,
                    chunk_id=key,
                    source=chunk.source,
                    page=10,
                    confidence=0.7,
                    document_id="report",
                    period_label=f"{year}-Q1",
                    period_kind="quarter",
                    scope="separate",
                    metric_code=code,
                    statement="income",
                    bbox=chunk.bbox,
                )
            )
    service.graph.add_document(
        Document(id="report", source="report.pdf", pages=[], scope="separate"), chunks, facts
    )
    service.retriever.upsert(chunks)
    return service


def test_supported_answer_and_citations(service):
    response = service.ask("Doanh thu thuần quý 1/2026 là bao nhiêu?", document_id="report")
    assert response.answer_status == "answered"
    assert "120 VND" in response.answer
    assert response.steps
    for citation in re.findall(r"\[(\d+)\]", response.answer):
        source = response.evidence[int(citation) - 1]
        assert source.document_id == "report"
        assert source.source_url == "/documents/report/file#page=10"
        assert source.bbox == [10, 20, 400, 40]
        assert "120" in source.text


def test_comparison_uses_requested_pair_not_latest_pair(service):
    response = service.ask(
        "Doanh thu thuần quý 1/2025 tăng bao nhiêu phần trăm so với cùng kỳ 2024?",
        document_id="report",
    )
    assert response.answer_status == "answered"
    assert response.mode == "calculation"
    assert "25,00 %" in response.answer
    assert {e.graph_facts[0].period for e in response.evidence} == {2025, 2024}
    assert "[1] [2]" in response.answer


@pytest.mark.parametrize(
    "question,reason",
    [
        ("Doanh thu thuần cả năm 2026", "quarter_required"),
        ("Doanh thu thuần năm 2026", "quarter_required"),
        ("Doanh thu thuần quý 1/2026 và quý 1/2025", "multiple_periods"),
        ("Doanh thu thuần quý 1/2026 so với quý 2/2025", "multiple_quarters"),
        ("Biên lợi nhuận gộp quý 1/2026 tăng so với 2025", "margin_comparison_unsupported"),
        ("Doanh thu thuần quý 1/2025 so với 2026", "reverse_comparison"),
        ("Doanh thu thuần riêng và hợp nhất quý 1/2026", "multiple_scopes"),
        ("Doanh thu thuần cả năm quý 1/2026", "mixed_periods"),
    ],
)
def test_ambiguous_or_unsupported_periods_are_not_silently_changed(service, question, reason):
    response = service.ask(question, document_id="report")
    assert response.answer_status == "needs_clarification"
    assert response.reason_code == reason
    assert response.mode == "evidence"


@pytest.mark.parametrize(
    "question",
    [
        "Doanh thu thuần quý 1/2026 và cổ tức là bao nhiêu?",
        "Doanh thu thuần xuất khẩu quý 1/2026 là bao nhiêu?",
        "Doanh thu thuần bình quân quý 1/2026 là bao nhiêu?",
        "Doanh thu thuần quý 1/2026 có vượt kế hoạch không?",
        "Tại sao doanh thu thuần quý 1/2026 tăng?",
        "Biên lợi nhuận trước thuế quý 1/2026 là bao nhiêu?",
        "Biên lợi nhuận gộp và doanh thu thuần quý 1/2026 là bao nhiêu?",
    ],
)
def test_unknown_qualifier_does_not_produce_partial_answer(service, question):
    response = service.ask(question, document_id="report")
    assert response.answer_status == "evidence_only"
    assert response.reason_code == "unsupported_question"
    assert "không phải kết luận" in response.answer


@pytest.mark.parametrize(
    "field,value",
    [
        ("unit", None),
        ("scope", "unknown"),
        ("period_kind", None),
        ("period_label", None),
        ("quality", "needs_review"),
    ],
)
def test_incomplete_or_conflicting_facts_are_not_answers(service, field, value):
    service.graph.graph.nodes["fact-2026-10"][field] = value
    response = service.ask("Doanh thu thuần", document_id="report")
    assert response.answer_status == "insufficient_evidence"
    assert response.mode == "evidence"


def test_missing_source_is_refused_without_crash(service):
    service.graph.graph.nodes["fact-2026-10"]["chunk_id"] = "does-not-exist"
    response = service.ask("Doanh thu thuần", document_id="report")
    assert response.answer_status == "insufficient_evidence"
    assert response.reason_code == "invalid_provenance"


def test_multiple_reports_require_selection(service):
    service.retriever.chunks.append(
        Chunk(id="other", document_id="other", source="other.pdf", page=1, text="Doanh thu 999")
    )
    response = service.ask("Doanh thu thuần")
    assert response.answer_status == "needs_clarification"
    assert response.reason_code == "document_required"
    assert not response.evidence


def test_exact_question_never_calls_model(service, monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("No network model call is allowed")

    monkeypatch.setattr("financial_rag.ollama_client.urlopen", fail)
    response = service.ask("LNST quý 1/2026", document_id="report", use_llm=True)
    assert response.answer_status == "answered"
    assert response.llm_model is None
    assert any("bỏ qua use_llm" in w for w in response.warnings)


def test_public_api_exposes_policy_and_decision(service, monkeypatch):
    monkeypatch.setattr(api, "service", service)
    with TestClient(api.app) as client:
        health = client.get("/health").json()
        assert health["llm_enabled"] is False
        assert health["answer_policy"] == "deterministic"
        response = client.post(
            "/ask",
            json={
                "question": "LNST quý 1/2026",
                "document_id": "report",
                "use_llm": True,
            },
        ).json()
        assert response["answer_status"] == "answered"
        assert response["steps"] and response["evidence"][0]["source_url"]
        page = client.get("/").text
        assert 'id="llm"' not in page
        assert "Cách tạo kết quả" in page


def test_percent_written_in_words_is_not_absolute_difference(service):
    response = service.ask(
        "Chênh lệch doanh thu thuần quý 1/2026 so với 2025 là bao nhiêu phần trăm?",
        document_id="report",
    )
    assert response.mode == "calculation"
    assert "20,00 %" in response.answer


def test_decimal_fact_is_not_rounded_to_integer(service):
    service.graph.graph.nodes["fact-2026-10"]["value"] = 120.25
    service.graph.graph.nodes["fact-2026-10"]["raw_value"] = "120,25"
    next(
        c for c in service.retriever.chunks if c.id == "2026-10"
    ).text = "Doanh thu thuần: 2026-Q1: 120,25 VND"
    response = service.ask("Doanh thu thuần quý 1/2026", document_id="report")
    assert "120,25 VND" in response.answer


def test_fact_value_must_agree_with_saved_source(service):
    service.graph.graph.nodes["fact-2026-10"]["value"] = 999
    response = service.ask("Doanh thu thuần quý 1/2026", document_id="report")
    assert response.answer_status == "insufficient_evidence"
    assert response.reason_code == "source_value_mismatch"
