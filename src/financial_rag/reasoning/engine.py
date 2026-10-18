"""End-to-end Financial RAG Engine executing the complete online answering pipeline."""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from financial_rag.config import settings
from financial_rag.db.repository import FinancialRepository
from financial_rag.extractor.slot_extractor import SlotExtractor, ExtractedSlots
from financial_rag.llm.model import LLMBackend
from financial_rag.llm.prompt import (
    SLOT_EXTRACTION_PROMPT,
    VERBALIZATION_PROMPT,
    TEXT_EXPLANATION_PROMPT,
)
from financial_rag.parser.number_normalizer import NumberNormalizer
from financial_rag.reasoning.calculator import FinancialCalculator, CalculationResult
from financial_rag.reasoning.router import QueryRouter, ExecutionRoute
from financial_rag.reasoning.verifier import NumericalVerifier, VerificationOutcome
from financial_rag.retrieval.hybrid import HybridRetriever, RetrievedChunk


logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    query: str
    answer: str
    question_type: str
    citations: List[str] = field(default_factory=list)
    facts_used: List[Dict[str, Any]] = field(default_factory=list)
    retrieved_chunks: List[Dict[str, Any]] = field(default_factory=list)
    slots: Dict[str, Any] = field(default_factory=dict)
    calculation: Optional[Dict[str, Any]] = None
    verification: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "question_type": self.question_type,
            "citations": self.citations,
            "facts_used": self.facts_used,
            "retrieved_chunks": self.retrieved_chunks,
            "slots": self.slots,
            "calculation": self.calculation,
            "verification": self.verification,
        }


class FinancialRAGEngine:
    """Master orchestrator connecting DB, retrieval, financial math, LLM, and verifier."""

    def __init__(
        self,
        repo: FinancialRepository,
        llm: Optional[LLMBackend] = None,
        retriever: Optional[HybridRetriever] = None,
    ):
        self.repo = repo
        self.llm = llm or LLMBackend()
        self.retriever = retriever or HybridRetriever(repo=repo)
        self.slot_extractor = SlotExtractor(repo=repo)
        self.calculator = FinancialCalculator()

    def answer_question(self, query: str) -> RAGResponse:
        """Execute online answering flow conforming strictly to design specification."""
        clean_query = query.strip()
        if not clean_query:
            return RAGResponse(query="", answer="Vui lòng nhập câu hỏi.", question_type="empty")

        # Step 1: Slot Extraction
        # Try extracting slots via LLM or rule-based fallback
        llm_slot_raw = None
        if self.llm.backend_type == "qwen_local":
            try:
                slot_prompt = SLOT_EXTRACTION_PROMPT.format(query=clean_query)
                llm_slot_raw = self.llm.generate(slot_prompt, max_new_tokens=150)
            except Exception as e:
                logger.warning(f"LLM slot extraction error: {e}")

        slots = self.slot_extractor.extract(clean_query, llm_response_json=llm_slot_raw)
        route = QueryRouter.determine_route(slots)

        # Step 2: Route Handling

        # 2A: TRAP QUESTION / OUT OF SCOPE
        if route == ExecutionRoute.TRAP:
            refusal_text = (
                "Hệ thống từ chối trả lời: Câu hỏi này nằm ngoài phạm vi tài liệu BCTC đã công bố. "
                "Hệ thống không đưa ra dự báo giá, khuyến nghị mua/bán chứng khoán hoặc thông tin không có trong tài liệu."
            )
            if any(w in clean_query.lower() for w in ["what", "how", "should", "predict", "forecast", "stock"]):
                refusal_text = (
                    "Information not available: This question falls outside the scope of the published financial statements. "
                    "The system does not provide investment advice, stock price predictions, or ungrounded data."
                )
            return RAGResponse(
                query=clean_query,
                answer=refusal_text,
                question_type="trap",
                slots=slots.to_dict(),
            )

        # 2B: STRUCTURED FACT LOOKUP
        if route == ExecutionRoute.LOOKUP:
            response = self._handle_lookup(clean_query, slots)
            if response:
                return response
            # If not found in structured facts, fallback to hybrid text search
            route = ExecutionRoute.TEXT_EXPLANATION

        # 2C: PYTHON CALCULATION (Margins, D/E, YoY Growth)
        if route == ExecutionRoute.CALCULATION:
            response = self._handle_calculation(clean_query, slots)
            if response:
                return response
            route = ExecutionRoute.TEXT_EXPLANATION

        # 2D: CROSS-COMPANY COMPARISON
        if route == ExecutionRoute.COMPARISON:
            response = self._handle_comparison(clean_query, slots)
            if response:
                return response
            route = ExecutionRoute.TEXT_EXPLANATION

        # 2E: TEXT EXPLANATION / HYBRID RETRIEVAL
        return self._handle_text_explanation(clean_query, slots)

    def _handle_lookup(self, query: str, slots: ExtractedSlots) -> Optional[RAGResponse]:
        """Look up facts directly in structured SQLite fact table."""
        if not slots.ticker:
            return None

        facts = self.repo.query_facts(
            ticker=slots.ticker,
            fiscal_year=slots.fiscal_year,
            period_type=slots.period_type,
            metric_code=slots.metric_code,
            line_item=slots.line_item,
            column_role="current",
        )

        if not facts:
            # Try without period_type restriction
            facts = self.repo.query_facts(
                ticker=slots.ticker,
                fiscal_year=slots.fiscal_year,
                metric_code=slots.metric_code,
                column_role="current",
            )

        if not facts:
            return None

        valid_facts = [f for f in facts if f.get("value") is not None and abs(f["value"]) > 0]
        primary_fact = valid_facts[0] if valid_facts else facts[0]
        val = primary_fact["value"]
        currency = primary_fact.get("currency", "VND")
        metric_name = primary_fact.get("metric_name_vi") or primary_fact.get("line_item")
        ticker = primary_fact["ticker"]
        fy = primary_fact["fiscal_year"]
        pt = primary_fact["period_type"]
        page = primary_fact.get("page", 1)

        formatted_val = NumberNormalizer.format_financial(val, currency=currency)
        citation = f"[{ticker} | {pt}/{fy} | Trang {page} | {primary_fact.get('table_ref', 'BCTC')}]"

        grounding = f"{metric_name} của {ticker} ({pt}/{fy}): {formatted_val} (Giá trị gốc: {primary_fact.get('value_raw', '')})."
        
        # Verbalize via LLM
        prompt = VERBALIZATION_PROMPT.format(
            query=query,
            grounding=grounding,
            citations=citation,
        )
        answer = self.llm.generate(prompt)

        # Verification
        v_res = NumericalVerifier.verify(answer, [primary_fact])

        return RAGResponse(
            query=query,
            answer=answer,
            question_type="lookup",
            citations=[citation],
            facts_used=[primary_fact],
            slots=slots.to_dict(),
            verification={
                "is_verified": v_res.is_verified,
                "warning": v_res.warning_message,
            },
        )

    def _handle_calculation(self, query: str, slots: ExtractedSlots) -> Optional[RAGResponse]:
        """Compute margins, D/E, or YoY growth using Python calculator."""
        calc_res: Optional[CalculationResult] = None
        calc_type = slots.calc_type

        # Case 1: Margins (Gross / Net)
        if calc_type == "margin" or "biên" in query.lower() or "margin" in query.lower():
            is_gross = "gộp" in query.lower() or "gross" in query.lower()
            profit_code = "gross_profit" if is_gross else "net_profit"
            profit_facts = [f for f in self.repo.query_facts(ticker=slots.ticker, fiscal_year=slots.fiscal_year, metric_code=profit_code) if f.get("value") and f["value"] > 0]
            rev_facts = [f for f in self.repo.query_facts(ticker=slots.ticker, fiscal_year=slots.fiscal_year, metric_code="net_revenue") if f.get("value") and f["value"] > 0]

            if profit_facts and rev_facts:
                calc_res = self.calculator.calculate_margin(
                    profit_fact=profit_facts[0],
                    revenue_fact=rev_facts[0],
                    margin_type="gross" if is_gross else "net",
                )

        # Case 2: Debt-to-Equity (D/E)
        elif calc_type == "de" or "nợ" in query.lower() or "d/e" in query.lower():
            liab_facts = [f for f in self.repo.query_facts(ticker=slots.ticker, fiscal_year=slots.fiscal_year, metric_code="total_liabilities") if f.get("value") and f["value"] > 0]
            eq_facts = [f for f in self.repo.query_facts(ticker=slots.ticker, fiscal_year=slots.fiscal_year, metric_code="equity") if f.get("value") and f["value"] > 0]

            if liab_facts and eq_facts:
                calc_res = self.calculator.calculate_debt_to_equity(
                    liabilities_fact=liab_facts[0],
                    equity_fact=eq_facts[0],
                )

        # Case 3: YoY Growth
        elif calc_type == "yoy" or "tăng trưởng" in query.lower() or "cùng kỳ" in query.lower() or "growth" in query.lower():
            target_metric = slots.metric_code or "net_revenue"
            cur_facts = [f for f in self.repo.query_facts(ticker=slots.ticker, fiscal_year=slots.fiscal_year, metric_code=target_metric, column_role="current") if f.get("value") and f["value"] > 0]
            prior_facts = [f for f in self.repo.query_facts(ticker=slots.ticker, fiscal_year=slots.fiscal_year, metric_code=target_metric, column_role="prior_year_same_period") if f.get("value") and f["value"] > 0]

            if cur_facts and prior_facts:
                calc_res = self.calculator.calculate_yoy_growth(
                    current_fact=cur_facts[0],
                    prior_fact=prior_facts[0],
                )

        if not calc_res:
            return None

        citations = [
            f"[{f.get('ticker')} | {f.get('period_type')}/{f.get('fiscal_year')} | Trang {f.get('page', 1)}]"
            for f in calc_res.input_facts
        ]

        prompt = VERBALIZATION_PROMPT.format(
            query=query,
            grounding=calc_res.explanation,
            citations="; ".join(citations),
        )
        answer = self.llm.generate(prompt)
        v_res = NumericalVerifier.verify(answer, calc_res.input_facts, calc_result=calc_res)

        return RAGResponse(
            query=query,
            answer=answer,
            question_type="calc",
            citations=citations,
            facts_used=calc_res.input_facts,
            slots=slots.to_dict(),
            calculation={
                "formula": calc_res.formula_str,
                "result": calc_res.formatted_result,
            },
            verification={
                "is_verified": v_res.is_verified,
                "warning": v_res.warning_message,
            },
        )

    def _handle_comparison(self, query: str, slots: ExtractedSlots) -> Optional[RAGResponse]:
        """Compare two companies on common metrics with safety checks on currency and industry."""
        ticker_a = slots.ticker
        ticker_b = slots.secondary_ticker

        if not ticker_a or not ticker_b:
            return None

        metric_code = slots.metric_code or "net_revenue"
        facts_a = self.repo.query_facts(ticker=ticker_a, metric_code=metric_code)
        facts_b = self.repo.query_facts(ticker=ticker_b, metric_code=metric_code)

        if not facts_a or not facts_b:
            return None

        fact_a = facts_a[0]
        fact_b = facts_b[0]

        comp_res = self.calculator.compare_companies(fact_a, fact_b)
        citations = [
            f"[{fact_a.get('ticker')} | {fact_a.get('period_type')}/{fact_a.get('fiscal_year')} | Trang {fact_a.get('page', 1)}]",
            f"[{fact_b.get('ticker')} | {fact_b.get('period_type')}/{fact_b.get('fiscal_year')} | Trang {fact_b.get('page', 1)}]",
        ]

        prompt = VERBALIZATION_PROMPT.format(
            query=query,
            grounding=comp_res.explanation,
            citations="; ".join(citations),
        )
        answer = self.llm.generate(prompt)
        v_res = NumericalVerifier.verify(answer, [fact_a, fact_b], calc_result=comp_res)

        return RAGResponse(
            query=query,
            answer=answer,
            question_type="compare",
            citations=citations,
            facts_used=[fact_a, fact_b],
            slots=slots.to_dict(),
            calculation={"result": comp_res.formatted_result},
            verification={
                "is_verified": v_res.is_verified,
                "warning": v_res.warning_message,
            },
        )

    def _handle_text_explanation(self, query: str, slots: ExtractedSlots) -> RAGResponse:
        """Retrieve relevant text/table chunks and verbalize response with citations."""
        chunks = self.retriever.retrieve(
            query=query,
            ticker=slots.ticker,
            fiscal_year=slots.fiscal_year,
            period_type=slots.period_type,
            top_k=settings.top_k_chunks,
        )

        # Check relevance threshold
        max_score = max([c.hybrid_score for c in chunks], default=0.0)
        if not chunks or max_score < self.retriever.relevance_threshold:
            # Low relevance -> polite refusal
            refusal_msg = (
                "Không tìm thấy thông tin phù hợp trong các báo cáo tài chính được nạp."
            )
            return RAGResponse(
                query=query,
                answer=refusal_msg,
                question_type="text_refusal",
                slots=slots.to_dict(),
            )

        # Build context from top chunks
        context_parts = []
        citations = []
        for ch in chunks:
            cit = ch.to_citation()
            citations.append(cit)
            context_parts.append(f"Source: {cit}\n{ch.content}\n")

        prompt = TEXT_EXPLANATION_PROMPT.format(
            query=query,
            context="\n---\n".join(context_parts),
        )
        answer = self.llm.generate(prompt)

        return RAGResponse(
            query=query,
            answer=answer,
            question_type="text",
            citations=citations,
            retrieved_chunks=[{
                "chunk_id": c.chunk_id,
                "ticker": c.ticker,
                "page": c.page_start,
                "score": c.hybrid_score,
                "prefix": c.context_prefix,
            } for c in chunks],
            slots=slots.to_dict(),
        )
