"""GraphRAG-enhanced Financial QA Engine.

This module provides two QA engines:

FinancialQAEngine (legacy, vector-only)
    Original ChromaDB vector search → re-rank → LLM generate.
    Kept for backward compatibility.

GraphRAGQAEngine (new, default)
    Hybrid retrieval: ChromaDB vector search  +  Knowledge Graph subgraph
    traversal → fused context → PoT (if needed) → LLM generate.

The hybrid context combines:
  1. Top-K vector results (summaries + raw content from ChromaDB/DocStore)
  2. Subgraph context (entity nodes + typed triples from the Knowledge Graph)

Both are formatted into a unified prompt for the LLM.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from financial_rag.config.settings import Settings
from financial_rag.reasoning.llm_client import UnifiedLLMClient
from financial_rag.reasoning.pot_reasoning import (
    ProgramOfThoughtEngine,
    is_quantitative_question,
)
from financial_rag.retrieval.reranker import rerank_contexts
from financial_rag.retrieval.vector_store import FinancialVectorStore


# ---------------------------------------------------------------------------
# System prompt (shared by both engines)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an advanced Financial Analyst AI with access to a hybrid
Knowledge Graph + Vector retrieval system (GraphRAG).

Your context includes:
  1. VECTOR EVIDENCE  — retrieved document chunks (text narratives, tables)
  2. GRAPH EVIDENCE   — structured Knowledge Graph triples (entities + typed relations)

Guidelines:
1. Ground your answers strictly on the supplied context. Do NOT invent figures.
2. Cite specific metrics, numbers, dates, and trends from tables and graph triples.
3. If the Knowledge Graph provides structured triples, use them for precise lookups.
4. If a Program-of-Thought computed answer is provided, verify it against the evidence.
5. Structure your answer:
   - **Executive Summary**: Direct answer.
   - **Supporting Evidence**: Key figures, table breakdown, graph-derived facts.
   - **Sources & References**: Dataset IDs, page numbers, or graph triple origins.
"""

SYSTEM_PROMPT_LEGACY = """You are an advanced Multimodal Financial Analyst AI.
You have access to retrieved contextual evidence including textual narratives,
structured HTML tables, and visual charts/figures.

Guidelines:
1. Ground your answers strictly on the supplied context.
2. Cite specific metrics, numbers, percentages, dates, and trends from tables.
3. If evidence is insufficient, state what is missing. Do not invent figures.
4. Structure your answer:
   - **Executive Summary / Conclusion**: Direct answer.
   - **Supporting Evidence & Details**: Key figures, table breakdown.
   - **Sources & References**: Page numbers and file sources.
"""


# ---------------------------------------------------------------------------
# GraphRAGQAEngine  (NEW — default engine)
# ---------------------------------------------------------------------------

class GraphRAGQAEngine:
    """Hybrid Knowledge Graph + Vector RAG question-answering engine.

    Parameters
    ----------
    settings : Settings
    vector_store : FinancialVectorStore
    graph_builder : KnowledgeGraphBuilder | None
        If None, graph retrieval is skipped and the engine falls back to
        vector-only mode.
    """

    def __init__(
        self,
        settings: Settings,
        vector_store: FinancialVectorStore,
        graph_builder: Any | None = None,
    ) -> None:
        self.settings     = settings
        self.vector_store = vector_store
        self.llm_client   = UnifiedLLMClient(settings)
        self.pot_engine   = ProgramOfThoughtEngine(settings, llm_client=self.llm_client)
        self._graph_builder = graph_builder

        # Lazily import graph retriever to avoid circular imports
        self._graph_retriever: Any = None

    # ------------------------------------------------------------------ #
    # Graph retriever (lazy init)
    # ------------------------------------------------------------------ #

    def _get_graph_retriever(self) -> Any | None:
        if self._graph_builder is None:
            return None
        if self._graph_retriever is None:
            from financial_rag.graph.graph_retriever import GraphRetriever
            self._graph_retriever = GraphRetriever(
                self._graph_builder,
                max_hops=self.settings.graph_max_hops,
            )
        return self._graph_retriever

    def set_graph_builder(self, builder: Any) -> None:
        """Attach or replace the KnowledgeGraphBuilder at runtime."""
        self._graph_builder  = builder
        self._graph_retriever = None   # reset cached retriever

    # ------------------------------------------------------------------ #
    # Main answer method
    # ------------------------------------------------------------------ #

    def answer(
        self,
        question: str,
        limit: int = 6,
        use_pot: bool = False,
    ) -> dict[str, Any]:
        """Answer a financial question using Graph + Vector hybrid retrieval.

        Parameters
        ----------
        question : str
        limit : int      — Number of vector context chunks to use.
        use_pot : bool   — Force Program-of-Thought reasoning.
        """
        # ── 1. Vector retrieval + re-rank ─────────────────────────────
        raw_contexts = self.vector_store.search(question, limit=limit * 2)
        contexts     = rerank_contexts(question, raw_contexts, top_k=limit)

        # ── 2. Graph retrieval ────────────────────────────────────────
        graph_result: dict[str, Any] = {}
        graph_context_text: str      = ""
        retriever = self._get_graph_retriever()
        if retriever is not None and self.settings.use_graph_rag:
            try:
                graph_result      = retriever.retrieve(question)
                graph_context_text = graph_result.get("context_text", "")
            except Exception as exc:
                graph_context_text = f"[Graph retrieval error: {exc}]"

        # ── 3. Separate modalities from vector results ────────────────
        text_table_contexts: list[str]     = []
        retrieved_images: list[dict]       = []
        retrieved_tables: list[dict]       = []

        for idx, item in enumerate(contexts, start=1):
            modality    = str(item.get("modality", "text")).lower()
            source      = item.get("metadata", {}).get("source", "unknown")
            page        = item.get("metadata", {}).get("page", 0)
            raw_content = item.get("raw_content", item.get("content", ""))

            if modality == "image":
                mime = item.get("metadata", {}).get("meta_mime_type", "image/png")
                retrieved_images.append({
                    "doc_id":       item.get("doc_id"),
                    "source":       source,
                    "page":         page,
                    "summary":      item.get("content"),
                    "image_base64": raw_content,
                    "mime_type":    mime,
                    "metadata":     item.get("metadata", {}),
                })
            elif modality == "table":
                retrieved_tables.append({
                    "doc_id":    item.get("doc_id"),
                    "source":    source,
                    "page":      page,
                    "table_html": raw_content,
                    "summary":   item.get("content"),
                    "metadata":  item.get("metadata", {}),
                })
                text_table_contexts.append(
                    f"[Context #{idx} – Table | Source: {source}, Page: {page}]\n{raw_content}"
                )
            else:
                text_table_contexts.append(
                    f"[Context #{idx} – Text | Source: {source}, Page: {page}]\n{raw_content}"
                )

        combined_vector_context = (
            "\n\n---\n\n".join(text_table_contexts)
            if text_table_contexts
            else "No vector context retrieved."
        )

        # ── 4. Fuse vector + graph contexts ───────────────────────────
        if graph_context_text:
            combined_context = (
                f"### VECTOR EVIDENCE (ChromaDB)\n\n{combined_vector_context}\n\n"
                f"### KNOWLEDGE GRAPH EVIDENCE\n\n{graph_context_text}"
            )
        else:
            combined_context = combined_vector_context

        # ── 5. Program-of-Thought (quantitative questions) ────────────
        should_use_pot = use_pot or is_quantitative_question(question)
        pot_result: dict[str, Any] | None = None
        if should_use_pot:
            pot_result = self.pot_engine.solve(question, combined_context)

        # ── 6. Build final prompt ─────────────────────────────────────
        prompt_text = (
            f"### RETRIEVED FINANCIAL CONTEXT (Vector + Knowledge Graph):\n"
            f"{combined_context}\n\n"
            f"### USER QUESTION:\n{question}\n\n"
        )

        if pot_result and pot_result.get("success"):
            prompt_text += (
                f"### COMPUTED ANSWER (via Python execution):\n"
                f"The following Python program was generated and executed:\n"
                f"```python\n{pot_result['generated_code']}\n```\n"
                f"**Computed result: {pot_result['answer']}**\n\n"
                f"Please verify this result against the context and provide a comprehensive "
                f"answer with reasoning and source citations:\n"
            )
        else:
            prompt_text += (
                "Please synthesise all retrieved evidence (vector + graph) "
                "and answer thoroughly:\n"
            )

        # ── 7. Generate answer ────────────────────────────────────────
        answer_text = self.llm_client.generate_multimodal(
            prompt=prompt_text,
            images=retrieved_images,
            system_prompt=SYSTEM_PROMPT,
            temperature=0.2,
        )

        result: dict[str, Any] = {
            "answer":           answer_text,
            "contexts":         contexts,
            "retrieved_images": retrieved_images,
            "retrieved_tables": retrieved_tables,
            "graph_retrieval":  graph_result,
            "retrieval_mode":   "graphrag" if graph_context_text else "vector_only",
        }

        if pot_result is not None:
            result["generated_code"]   = pot_result.get("generated_code", "")
            result["computed_answer"]  = pot_result.get("answer")
            result["pot_success"]      = pot_result.get("success", False)

        return result


# ---------------------------------------------------------------------------
# FinancialQAEngine  (legacy — vector-only, kept for backward compat)
# ---------------------------------------------------------------------------

class FinancialQAEngine:
    """Vector-only QA engine (original implementation).

    Use ``GraphRAGQAEngine`` for the full GraphRAG pipeline.
    This class is retained so existing CLI commands and tests continue to work.
    """

    def __init__(self, settings: Settings, vector_store: FinancialVectorStore):
        self.settings     = settings
        self.vector_store = vector_store
        self.llm_client   = UnifiedLLMClient(settings)
        self.pot_engine   = ProgramOfThoughtEngine(settings, llm_client=self.llm_client)

    @property
    def genai_client(self) -> Any:
        return self.llm_client.genai_client

    @genai_client.setter
    def genai_client(self, value: Any) -> None:
        self.llm_client.genai_client = value

    def answer(self, question: str, limit: int = 6, use_pot: bool = False) -> dict[str, Any]:
        raw_contexts = self.vector_store.search(question, limit=limit * 2)
        contexts     = rerank_contexts(question, raw_contexts, top_k=limit)

        text_table_contexts: list[str] = []
        retrieved_images: list[dict]   = []
        retrieved_tables: list[dict]   = []

        for idx, item in enumerate(contexts, start=1):
            modality    = str(item.get("modality", "text")).lower()
            source      = item.get("metadata", {}).get("source", "unknown")
            page        = item.get("metadata", {}).get("page", 0)
            raw_content = item.get("raw_content", item.get("content", ""))

            if modality == "image":
                mime = item.get("metadata", {}).get("meta_mime_type", "image/png")
                retrieved_images.append({
                    "doc_id": item.get("doc_id"), "source": source, "page": page,
                    "summary": item.get("content"), "image_base64": raw_content,
                    "mime_type": mime, "metadata": item.get("metadata", {}),
                })
            elif modality == "table":
                retrieved_tables.append({
                    "doc_id": item.get("doc_id"), "source": source, "page": page,
                    "table_html": raw_content, "summary": item.get("content"),
                    "metadata": item.get("metadata", {}),
                })
                text_table_contexts.append(
                    f"[Context #{idx} - Structured Table | Source: {source}, Page: {page}]\n{raw_content}"
                )
            else:
                text_table_contexts.append(
                    f"[Context #{idx} - Text Narrative | Source: {source}, Page: {page}]\n{raw_content}"
                )

        combined_text_context = (
            "\n\n" + ("\n\n---\n\n".join(text_table_contexts))
            if text_table_contexts
            else "No textual context retrieved."
        )

        should_use_pot = use_pot or is_quantitative_question(question)
        pot_result: dict[str, Any] | None = None
        if should_use_pot:
            pot_result = self.pot_engine.solve(question, combined_text_context)

        prompt_text = (
            f"### RETRIEVED FINANCIAL CONTEXT (Text & Tables):\n{combined_text_context}\n\n"
            f"### USER QUESTION:\n{question}\n\n"
        )

        if pot_result and pot_result.get("success"):
            prompt_text += (
                f"### COMPUTED ANSWER (via Python execution):\n"
                f"```python\n{pot_result['generated_code']}\n```\n"
                f"**Computed result: {pot_result['answer']}**\n\n"
                f"Please verify and explain:\n"
            )
        else:
            prompt_text += "Please synthesise all retrieved evidence and answer thoroughly:\n"

        answer_text = self.llm_client.generate_multimodal(
            prompt=prompt_text,
            images=retrieved_images,
            system_prompt=SYSTEM_PROMPT_LEGACY,
            temperature=0.2,
        )

        result: dict[str, Any] = {
            "answer":           answer_text,
            "contexts":         contexts,
            "retrieved_images": retrieved_images,
            "retrieved_tables": retrieved_tables,
        }
        if pot_result is not None:
            result["generated_code"]  = pot_result.get("generated_code", "")
            result["computed_answer"] = pot_result.get("answer")
            result["pot_success"]     = pot_result.get("success", False)

        return result
