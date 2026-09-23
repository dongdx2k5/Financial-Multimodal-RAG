"""LLM-based Entity & Relation Extractor for GraphRAG.

Given a raw text passage (from FinQA pre/post text or ViFinTab captions),
this module asks the LLM to output structured (Subject, Relation, Object)
triples in JSON format.

Schema
------
Entities  : COMPANY | FINANCIAL_METRIC | YEAR | MONETARY_VALUE
Relations : HAS_REVENUE | HAS_PROFIT | HAS_NET_INCOME | HAS_VALUE |
            INCREASED_BY | DECREASED_BY | REPORT_PERIOD | SAME_AS

Result is injected into a KnowledgeGraphBuilder instance.
"""

from __future__ import annotations

import json
import re
from typing import Any

from financial_rag.config.settings import Settings
from financial_rag.graph.knowledge_graph import KnowledgeGraphBuilder, NodeType, RelationType


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_EXTRACTION_PROMPT = """\
You are a financial knowledge graph extraction engine.

Given the financial text below, extract ALL factual (Subject, Relation, Object) triples.

Allowed entity types:
  COMPANY          – organization name (e.g. "Apple Inc", "Vinamilk")
  FINANCIAL_METRIC – financial metric name (e.g. "Net Income", "Revenue", "Lợi nhuận ròng")
  YEAR             – fiscal year or period (e.g. "2020", "Q1 2021", "FY2022")
  MONETARY_VALUE   – numeric financial value with optional unit (e.g. "$57.4B", "9,000 tỷ VND")

Allowed relations:
  HAS_REVENUE      – company has revenue value
  HAS_PROFIT       – company has profit/gross profit
  HAS_NET_INCOME   – company has net income / lợi nhuận ròng
  HAS_VALUE        – generic metric→value relation
  INCREASED_BY     – metric increased by a percentage or absolute value
  DECREASED_BY     – metric decreased by a percentage or absolute value
  REPORT_PERIOD    – entity relates to a time period/year

Output ONLY a valid JSON array of triples. Each triple is:
{{"subject": "...", "subject_type": "...", "relation": "...", "object": "...", "object_type": "...", "year": "..."}}

If "year" is unknown, set it to null.

### Financial Text:
{text}

### JSON Triples:
"""

_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def _strip_fences(raw: str) -> str:
    m = _FENCE_RE.search(raw)
    if m:
        return m.group(1).strip()
    return raw.strip()


# ---------------------------------------------------------------------------
# LLMEntityExtractor
# ---------------------------------------------------------------------------

class LLMEntityExtractor:
    """Extracts (Subject, Relation, Object) triples from text using an LLM.

    Parameters
    ----------
    settings : Settings
        Application settings (LLM provider + API key).
    llm_client : optional
        Pre-built UnifiedLLMClient. If None, creates one from settings.
    cache : bool
        If True, skips texts that have already been processed.
    """

    def __init__(
        self,
        settings: Settings,
        llm_client: Any = None,
        cache: bool = True,
    ) -> None:
        self.settings = settings
        self._cache: set[int] = set()
        self._use_cache = cache

        if llm_client is None:
            from financial_rag.reasoning.llm_client import UnifiedLLMClient
            self._llm = UnifiedLLMClient(settings)
        else:
            self._llm = llm_client

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def extract_and_add(
        self,
        text: str,
        builder: KnowledgeGraphBuilder,
        source: str = "llm_extracted",
        max_chars: int = 3000,
    ) -> int:
        """Extract triples from *text* and insert them into *builder*.

        Returns the number of triples successfully added.
        """
        if not text or not text.strip():
            return 0

        text = text[:max_chars]
        text_hash = hash(text)

        if self._use_cache and text_hash in self._cache:
            return 0

        triples = self._call_llm(text)
        count = 0
        for t in triples:
            try:
                self._add_triple_to_graph(t, builder, source)
                count += 1
            except Exception:
                continue

        if self._use_cache:
            self._cache.add(text_hash)

        return count

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _call_llm(self, text: str) -> list[dict[str, Any]]:
        """Call the LLM and parse the JSON response."""
        prompt = _EXTRACTION_PROMPT.format(text=text)
        try:
            raw = self._llm.generate_text(prompt=prompt, temperature=0.0)
        except Exception:
            return []

        raw = _strip_fences(raw)

        # Find first "[" and last "]" to handle extra LLM preamble
        start = raw.find("[")
        end   = raw.rfind("]")
        if start == -1 or end == -1:
            return []
        raw = raw[start : end + 1]

        try:
            triples = json.loads(raw)
            if not isinstance(triples, list):
                return []
            return triples
        except json.JSONDecodeError:
            return []

    def _add_triple_to_graph(
        self,
        triple: dict[str, Any],
        builder: KnowledgeGraphBuilder,
        source: str,
    ) -> None:
        """Validate and add one triple to the builder."""
        subject       = str(triple.get("subject", "")).strip()
        subject_type  = str(triple.get("subject_type", NodeType.COMPANY)).strip()
        relation      = str(triple.get("relation", RelationType.HAS_VALUE)).strip()
        obj           = str(triple.get("object", "")).strip()
        object_type   = str(triple.get("object_type", NodeType.VALUE)).strip()
        year          = triple.get("year")
        year_str      = str(year).strip() if year else None

        if not subject or not obj:
            return

        # Validate types (fall back to generic if unknown)
        valid_node_types = {
            NodeType.COMPANY, NodeType.FINANCIAL_METRIC,
            NodeType.YEAR, NodeType.VALUE,
        }
        if subject_type not in valid_node_types:
            subject_type = NodeType.COMPANY
        if object_type not in valid_node_types:
            object_type = NodeType.VALUE

        builder.add_triple(
            subject=subject,
            relation=relation,
            obj=obj,
            subject_type=subject_type,
            object_type=object_type,
            year=year_str,
            source=source,
        )
