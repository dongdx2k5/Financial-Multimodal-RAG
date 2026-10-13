"""Phase 1 tests – Graph Traversal (1.1) & Financial Taxonomy (1.2).

These tests run fully offline – no LLM / API key required.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from financial_rag.extraction import (
    FINANCIAL_TAXONOMY,
    LLMFactExtractor,
    extract_metric_from_query,
    standardize_metric,
)
from financial_rag.graph import KnowledgeGraph
from financial_rag.models import Chunk, FinancialFact

# ============================================================
# Bước 1.2 – Financial Taxonomy & standardize_metric
# ============================================================


class TestFinancialTaxonomy:
    def test_taxonomy_has_required_codes(self):
        required = {
            "REVENUE_NET", "NET_PROFIT", "TOTAL_ASSETS",
            "EQUITY", "COGS", "OPERATING_PROFIT",
        }
        assert required.issubset(set(FINANCIAL_TAXONOMY.keys()))

    def test_standardize_metric_exact_vietnamese(self):
        assert standardize_metric("lợi nhuận sau thuế") == "NET_PROFIT"

    def test_standardize_metric_exact_english(self):
        assert standardize_metric("net revenue") == "REVENUE_NET"

    def test_standardize_metric_case_insensitive(self):
        assert standardize_metric("Doanh Thu Thuần") == "REVENUE_NET"

    def test_standardize_metric_substring_match(self):
        result = standardize_metric("Doanh thu thuần về bán hàng và cung cấp dịch vụ (VAS)")
        assert result == "REVENUE_NET"

    def test_standardize_metric_unknown_returns_none(self):
        assert standardize_metric("phí môi giới") is None

    def test_extract_metric_from_query_vn(self):
        alias, code = extract_metric_from_query(
            "Doanh thu thuần của HPG năm 2023 là bao nhiêu?"
        )
        assert alias is not None
        assert code == "REVENUE_NET"

    def test_extract_metric_from_query_en(self):
        _alias, code = extract_metric_from_query("What is the net profit of VNM in 2022?")
        assert code == "NET_PROFIT"

    def test_extract_metric_from_query_no_match(self):
        alias, code = extract_metric_from_query("Who is the CEO of the company?")
        assert alias is None
        assert code is None

    def test_llm_extractor_disabled_without_client(self):
        chunk = Chunk(
            id="c1", document_id="d1", source="test.pdf",
            page=1, text="Revenue: 100 billion VND 2023",
        )
        extractor = LLMFactExtractor(api_key="dummy")
        extractor._client = None
        assert extractor.extract(chunk) == []


# ============================================================
# Bước 1.1 – Graph Traversal (find_historical_facts)
# ============================================================


def _make_fact(idx: int, ticker: str, std_metric: str, period: int) -> FinancialFact:
    return FinancialFact(
        id=f"fact_{idx}",
        metric="doanh thu thuần",
        standardized_metric=std_metric,
        value=float(1000 + idx * 100),
        raw_value=str(1000 + idx * 100),
        unit="tỷ",
        period=period,
        ticker=ticker,
        chunk_id=f"chunk_{idx}",
        source="test.pdf",
        page=1,
        confidence=0.9,
    )


class TestGraphTraversal:
    @pytest.fixture()
    def graph(self, tmp_path: Path) -> KnowledgeGraph:
        g = KnowledgeGraph(tmp_path / "graph.json")
        for i, period in enumerate([2021, 2022, 2023, 2024]):
            fact = _make_fact(i, "HPG", "REVENUE_NET", period)
            g.graph.add_node(
                fact.id,
                kind="Fact",
                metric=fact.metric,
                standardized_metric=fact.standardized_metric,
                value=fact.value,
                raw_value=fact.raw_value,
                unit=fact.unit,
                period=fact.period,
                ticker=fact.ticker,
                chunk_id=fact.chunk_id,
                source=fact.source,
                page=fact.page,
                confidence=fact.confidence,
            )
        other = _make_fact(99, "VNM", "NET_PROFIT", 2023)
        g.graph.add_node(
            other.id, kind="Fact",
            metric=other.metric,
            standardized_metric=other.standardized_metric,
            value=other.value, raw_value=other.raw_value,
            unit=other.unit, period=other.period,
            ticker=other.ticker, chunk_id=other.chunk_id,
            source=other.source, page=other.page, confidence=other.confidence,
        )
        return g

    def test_find_historical_facts_returns_correct_ticker(self, graph: KnowledgeGraph):
        facts = graph.find_historical_facts("HPG", "REVENUE_NET", limit_years=4)
        assert all(f.ticker == "HPG" for f in facts)

    def test_find_historical_facts_correct_metric(self, graph: KnowledgeGraph):
        facts = graph.find_historical_facts("HPG", "REVENUE_NET", limit_years=4)
        assert all(f.standardized_metric == "REVENUE_NET" for f in facts)

    def test_find_historical_facts_sorted_chronologically(self, graph: KnowledgeGraph):
        facts = graph.find_historical_facts("HPG", "REVENUE_NET", limit_years=4)
        periods = [f.period for f in facts]
        assert periods == sorted(periods)

    def test_find_historical_facts_limit_respected(self, graph: KnowledgeGraph):
        facts = graph.find_historical_facts("HPG", "REVENUE_NET", limit_years=2)
        assert len(facts) == 2
        assert facts[0].period == 2023
        assert facts[1].period == 2024

    def test_find_historical_facts_excludes_other_tickers(self, graph: KnowledgeGraph):
        facts = graph.find_historical_facts("HPG", "REVENUE_NET", limit_years=10)
        assert not any(f.ticker == "VNM" for f in facts)

    def test_find_historical_facts_empty_when_no_match(self, graph: KnowledgeGraph):
        facts = graph.find_historical_facts("HPG", "TOTAL_ASSETS", limit_years=3)
        assert facts == []

    def test_find_facts_by_standard_metric_no_ticker_filter(self, graph: KnowledgeGraph):
        facts = graph.find_facts_by_standard_metric("REVENUE_NET")
        assert len(facts) == 4

    def test_find_facts_by_standard_metric_with_ticker(self, graph: KnowledgeGraph):
        facts = graph.find_facts_by_standard_metric("NET_PROFIT", ticker="VNM")
        assert len(facts) == 1
        assert facts[0].ticker == "VNM"
