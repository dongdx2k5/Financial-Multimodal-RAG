"""Tests for the context re-ranker module."""

from financial_rag.retrieval.reranker import (
    extract_financial_entities,
    extract_time_anchors,
    rerank_contexts,
)


# ---------------------------------------------------------------------------
# extract_time_anchors
# ---------------------------------------------------------------------------

class TestExtractTimeAnchors:
    def test_year_extraction(self) -> None:
        anchors = extract_time_anchors("Revenue increased from 2018 to 2019")
        assert "2018" in anchors
        assert "2019" in anchors

    def test_quarter_extraction(self) -> None:
        anchors = extract_time_anchors("Q3 results were strong")
        assert "Q3" in anchors

    def test_fiscal_year(self) -> None:
        anchors = extract_time_anchors("In fiscal year 2021, the company grew")
        assert any("FY" in a or "2021" in a for a in anchors)

    def test_fy_abbreviation(self) -> None:
        anchors = extract_time_anchors("FY2020 was exceptional")
        assert any("FY2020" in a or "2020" in a for a in anchors)

    def test_no_time_anchors(self) -> None:
        anchors = extract_time_anchors("The company's strategy is diversification")
        assert len(anchors) == 0

    def test_empty_text(self) -> None:
        assert extract_time_anchors("") == []

    def test_vietnamese_year(self) -> None:
        anchors = extract_time_anchors("Doanh thu năm 2021 tăng mạnh")
        assert "2021" in anchors

    def test_multiple_quarters(self) -> None:
        anchors = extract_time_anchors("Q1 and Q4 performance comparison")
        assert "Q1" in anchors
        assert "Q4" in anchors


# ---------------------------------------------------------------------------
# extract_financial_entities
# ---------------------------------------------------------------------------

class TestExtractFinancialEntities:
    def test_revenue(self) -> None:
        entities = extract_financial_entities("Total revenue grew 10%")
        assert "total revenue" in entities or "revenue" in entities

    def test_net_income(self) -> None:
        entities = extract_financial_entities("Net income was $500M")
        assert "net income" in entities

    def test_ebitda(self) -> None:
        entities = extract_financial_entities("EBITDA margin expanded")
        assert "ebitda" in entities

    def test_operating_income(self) -> None:
        entities = extract_financial_entities("Operating income declined by 5%")
        assert "operating income" in entities

    def test_no_entities(self) -> None:
        entities = extract_financial_entities("The sky is blue")
        assert len(entities) == 0

    def test_vietnamese(self) -> None:
        entities = extract_financial_entities("Doanh thu và lợi nhuận tăng")
        assert any(e in entities for e in ["doanh thu", "lợi nhuận"])

    def test_multiple_entities(self) -> None:
        entities = extract_financial_entities(
            "Revenue was strong, but operating expenses grew faster than gross profit"
        )
        assert len(entities) >= 2


# ---------------------------------------------------------------------------
# rerank_contexts
# ---------------------------------------------------------------------------

class TestRerankContexts:
    def _make_context(self, content: str, modality: str = "text", distance: float = 0.5) -> dict:
        return {
            "doc_id": f"doc_{hash(content) % 1000}",
            "content": content,
            "raw_content": content,
            "modality": modality,
            "metadata": {"source": "test.pdf", "page": 1},
            "distance": distance,
        }

    def test_time_anchored_context_ranked_higher(self) -> None:
        question = "What was the revenue in 2019?"
        contexts = [
            self._make_context("Revenue in 2017 was $3B", distance=0.3),
            self._make_context("Revenue in 2019 reached $5B", distance=0.5),
            self._make_context("Company overview and history", distance=0.2),
        ]
        result = rerank_contexts(question, contexts, top_k=3)
        # The 2019 context should be ranked first despite higher vector distance
        assert "2019" in result[0]["content"]

    def test_entity_matched_context_ranked_higher(self) -> None:
        question = "What is the EBITDA margin?"
        contexts = [
            self._make_context("The company has 500 employees", distance=0.3),
            self._make_context("EBITDA margin was 25% in 2020", distance=0.5),
        ]
        result = rerank_contexts(question, contexts, top_k=2)
        assert "EBITDA" in result[0]["content"]

    def test_top_k_limits(self) -> None:
        contexts = [self._make_context(f"Context {i}") for i in range(10)]
        result = rerank_contexts("test question", contexts, top_k=3)
        assert len(result) == 3

    def test_empty_contexts(self) -> None:
        result = rerank_contexts("test", [], top_k=5)
        assert result == []

    def test_rerank_score_in_metadata(self) -> None:
        contexts = [self._make_context("Revenue was $1B")]
        result = rerank_contexts("What is the revenue?", contexts, top_k=1)
        assert "rerank_score" in result[0]["metadata"]

    def test_preserves_all_fields(self) -> None:
        ctx = self._make_context("Some content")
        ctx["extra_field"] = "should_survive"
        result = rerank_contexts("test", [ctx], top_k=1)
        assert result[0]["extra_field"] == "should_survive"
        assert result[0]["doc_id"] == ctx["doc_id"]
