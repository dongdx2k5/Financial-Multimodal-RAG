"""Core service layer orchestrating ingestion, retrieval, and answer generation.

When a ``GOOGLE_API_KEY`` is configured the service enables:
* Dense multilingual embeddings (cross-lingual Vietnamese ↔ English retrieval)
* LLM-powered answer generation via Gemini

Without an API key the service falls back to TF-IDF retrieval with evidence-first
answers, keeping the MVP fully functional offline.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .calculator import process_calculations
from .config import Settings
from .database import FinancialDatabase
from .extraction import FinancialFactExtractor
from .graph import KnowledgeGraph
from .ingestion import Chunker, DocumentLoader
from .models import AskResponse, CalculationResult, Chunk, Document, FinancialFact
from .retrieval import HybridRetriever

logger = logging.getLogger(__name__)


class FinancialRAGService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.db = FinancialDatabase(settings.database_path)
        self.graph = KnowledgeGraph(settings.index_dir / "graph.json")

        # ------------------------------------------------------------------
        # Optional Gemini components (free tier)
        # ------------------------------------------------------------------
        self.embedder = None
        self.llm = None

        if settings.google_api_key:
            try:
                from .embeddings import GeminiEmbedder

                self.embedder = GeminiEmbedder(
                    settings.google_api_key, settings.embedding_model
                )
                logger.info("Gemini dense embedder enabled (%s)", settings.embedding_model)
            except Exception as exc:
                logger.warning(
                    "Failed to initialize Gemini dense embedder (%s): %s. "
                    "Dense embeddings disabled.",
                    settings.embedding_model,
                    exc,
                )

            try:
                from .llm import GeminiLLM

                self.llm = GeminiLLM(settings.google_api_key, settings.gemini_model)
                logger.info("Gemini LLM enabled (%s)", settings.gemini_model)
            except Exception as exc:
                logger.warning(
                    "Failed to initialize Gemini LLM (%s): %s. "
                    "LLM generation disabled.",
                    settings.gemini_model,
                    exc,
                )

        from .retrieval import CrossEncoderReranker

        self.reranker = CrossEncoderReranker()
        self.retriever = HybridRetriever(
            settings.index_dir / "chunks.json",
            self.graph,
            embedder=self.embedder,
            db=self.db,
            reranker=self.reranker,
        )
        self.loader = DocumentLoader(vision_llm=self.llm)
        self.chunker = Chunker(settings.chunk_size, settings.chunk_overlap)
        self.extractor = FinancialFactExtractor()

        # Sync retriever from database if index file is empty but DB has chunks
        if not self.retriever.chunks:
            db_chunks = self.db.get_all_chunks()
            if db_chunks:
                self.retriever.sync(db_chunks)

    def ingest(self, path, ticker: str | None = None, year: int | None = None) -> dict:
        document = self.loader.load(path, ticker=ticker, year=year)
        chunks = self.chunker.split(document)
        facts = [fact for chunk in chunks for fact in self.extractor.extract(chunk)]

        # Extract structured facts from detected financial charts (Phase 2.2)
        chart_facts = self._extract_chart_facts(document, chunks)
        fact_ids = {f.id for f in facts}
        for cf in chart_facts:
            if cf.id not in fact_ids:
                facts.append(cf)
                fact_ids.add(cf.id)

        # 1. Save to SQLite database (primary persistent store)
        self.db.save_document(document, chunks, facts)

        # 2. Update Knowledge Graph
        self.graph.add_document(document, chunks, facts)

        # 3. Update Hybrid Retriever index (TF-IDF + optional dense)
        self.retriever.upsert(chunks)

        return {
            "document_id": document.id,
            "filename": Path(document.source).name,
            "ticker": document.ticker,
            "year": document.year,
            "pages": len(document.pages),
            "chunks": len(chunks),
            "facts": len(facts),
            "charts": len(document.charts),
        }

    def _extract_chart_facts(
        self, document: Document, chunks: list[Chunk]
    ) -> list[FinancialFact]:
        from .extraction import standardize_metric
        from .ingestion import _stable_id

        chart_facts: list[FinancialFact] = []
        for chart in document.charts:
            page_chunk = next(
                (c for c in chunks if c.page == chart.page),
                chunks[0] if chunks else None,
            )
            chunk_id = page_chunk.id if page_chunk else f"chart_{document.id}_{chart.page}"

            for dp in chart.data_points:
                std = standardize_metric(dp.metric)
                fact_id = _stable_id(
                    "chart",
                    document.id,
                    chart.page,
                    dp.metric,
                    dp.period or document.year,
                    dp.value,
                )
                chart_facts.append(
                    FinancialFact(
                        id=fact_id,
                        metric=dp.metric,
                        standardized_metric=std,
                        value=dp.value,
                        raw_value=dp.raw_value,
                        unit=dp.unit,
                        period=dp.period or document.year,
                        ticker=document.ticker,
                        chunk_id=chunk_id,
                        source=document.source,
                        page=chart.page,
                        confidence=0.95,
                        is_chart=True,
                    )
                )
        return chart_facts

    def list_documents(self) -> list[dict]:
        return self.db.list_documents()

    def get_document(self, document_id: str) -> dict | None:
        return self.db.get_document(document_id)

    def delete_document(self, document_id: str) -> bool:
        deleted = self.db.delete_document(document_id)
        if deleted:
            self.graph.remove_document(document_id)
            self.retriever.sync(self.db.get_all_chunks())
        return deleted

    def get_facts(
        self,
        ticker: str | None = None,
        year: int | None = None,
        metric: str | None = None,
        is_chart: bool | None = None,
        limit: int = 100,
    ) -> list[FinancialFact]:
        return self.db.get_facts(
            ticker=ticker,
            year=year,
            metric=metric,
            is_chart=is_chart,
            limit=limit,
        )

    def find_historical_facts(
        self,
        ticker: str,
        standardized_metric: str,
        limit_years: int = 3,
    ) -> list[FinancialFact]:
        """Phase 1.1 – Public API for graph-based time-series retrieval.

        Returns fact objects for a given ticker/metric sorted oldest to newest.
        """
        return self.graph.find_historical_facts(
            ticker=ticker,
            standardized_metric=standardized_metric,
            limit_years=limit_years,
        )

    def get_stats(self) -> dict:
        db_stats = self.db.get_stats()
        return {
            **db_stats,
            "graph_nodes": self.graph.graph.number_of_nodes(),
            "graph_edges": self.graph.graph.number_of_edges(),
            "dense_embeddings": self.embedder is not None,
            "llm_enabled": self.llm is not None,
            "sqlite_vec_enabled": self.db.has_sqlite_vec,
            "reranker_enabled": self.reranker is not None and self.reranker.is_available,
        }

    # ------------------------------------------------------------------
    # Question answering
    # ------------------------------------------------------------------

    def ask(
        self,
        question: str,
        top_k: int | None = None,
        ticker: str | None = None,
        year: int | None = None,
        history_years: int = 3,
    ) -> AskResponse:
        """Answer a financial question using hybrid retrieval + optional LLM.

        Phase 1.1 enhancement: after normal retrieval, the detected metric taxonomy
        code is used to fetch multi-year historical facts from the Knowledge Graph.
        These are injected into the LLM context so it can compute YoY growth rates
        and time-series comparisons accurately via SafeCalculator.
        """
        evidence, detected_metric = self.retriever.search(
            question,
            top_k=top_k or self.settings.retrieval_top_k,
            ticker=ticker,
            year=year,
        )
        if not evidence:
            return AskResponse(
                answer="Chưa tìm thấy bằng chứng phù hợp trong chỉ mục. Hãy nạp báo cáo trước.\n"
                       "No matching evidence found in the index. Please ingest a report first.",
                evidence=[],
                mode="evidence",
            )

        # --- LLM mode: generate a natural-language answer via Gemini ---
        if self.llm:
            evidence_texts = [
                f"[{item.source}, trang/page {item.page}]: {item.text}"
                for item in evidence
            ]
            # Collect facts from chunk-level graph lookup
            all_facts = [f for item in evidence for f in (item.graph_facts or [])]

            # --- Phase 1.1: Time-series Graph Traversal ---
            # Use detected metric code to fetch historical facts from graph
            historical_facts: list[FinancialFact] = []
            if detected_metric and evidence:
                # Infer ticker from first matched evidence chunk
                inferred_ticker = ticker
                if not inferred_ticker:
                    for ev in evidence:
                        for fact in (ev.graph_facts or []):
                            if fact.ticker:
                                inferred_ticker = fact.ticker
                                break
                        if inferred_ticker:
                            break
                if inferred_ticker:
                    historical_facts = self.graph.find_historical_facts(
                        ticker=inferred_ticker,
                        standardized_metric=detected_metric,
                        limit_years=history_years,
                    )
                    logger.info(
                        "Graph traversal: found %d historical facts for %s/%s",
                        len(historical_facts),
                        inferred_ticker,
                        detected_metric,
                    )

            # Build facts text: chunk-level facts + historical time-series
            facts_text = ""
            combined_facts = all_facts + [
                f for f in historical_facts
                if not any(f.id == af.id for af in all_facts)  # deduplicate
            ]
            if combined_facts:
                # Group historical facts into a time-series section
                ts_facts = [f for f in combined_facts if f in historical_facts
                            and f not in all_facts]
                regular_facts = [f for f in combined_facts if f not in ts_facts]

                lines = []
                if regular_facts:
                    lines.extend(
                        f"• {f.metric}: {f.value} {f.unit or ''} ({f.period or 'N/A'})"
                        for f in regular_facts
                    )
                if ts_facts:
                    lines.append(f"\n[Chuỗi thời gian / Time-series: {detected_metric}]")
                    lines.extend(
                        f"  {f.period}: {f.value} {f.unit or ''}"
                        for f in sorted(ts_facts, key=lambda x: x.period or 0)
                    )
                facts_text = "\n".join(lines)
            elif historical_facts:
                facts_text = f"[Chuỗi thời gian / Time-series: {detected_metric}]\n" + "\n".join(
                    f"  {f.period}: {f.value} {f.unit or ''}"
                    for f in historical_facts
                )

            try:
                raw_answer = self.llm.generate_answer(question, evidence_texts, facts_text)
                answer, calculations = process_calculations(raw_answer)
                return AskResponse(
                    answer=answer,
                    evidence=evidence,
                    mode="llm",
                    calculations=[CalculationResult(**c) for c in calculations],
                )
            except Exception as exc:
                logger.warning("LLM generation failed, falling back to evidence: %s", exc)

        # --- Evidence-first fallback ---
        raw_answer = "Các bằng chứng liên quan nhất:\n" + "\n".join(
            f"- {item.text[:300]} [nguồn: {item.source}, trang {item.page}]" for item in evidence
        )
        answer, calculations = process_calculations(raw_answer)
        return AskResponse(
            answer=answer,
            evidence=evidence,
            mode="evidence",
            calculations=[CalculationResult(**c) for c in calculations],
        )
