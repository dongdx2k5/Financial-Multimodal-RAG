"""Context re-ranking with time anchoring and financial entity matching.

After initial vector search (ChromaDB), this module re-scores and filters
retrieved contexts to prioritize the ones truly relevant to the question,
using keyword overlap, temporal alignment, and financial entity matching.
"""

from __future__ import annotations

import re
from typing import Any


# ---------------------------------------------------------------------------
# Time anchor extraction
# ---------------------------------------------------------------------------

_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_QUARTER_RE = re.compile(r"\b[Qq][1-4]\b")
_FISCAL_YEAR_RE = re.compile(r"\b(?:FY|fiscal\s+year)\s*(19|20)?\d{2,4}\b", re.IGNORECASE)
_HALF_RE = re.compile(r"\b[Hh][12]\b")
_MONTH_RE = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December"
    r"|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b",
    re.IGNORECASE,
)
# Vietnamese time anchors
_VN_YEAR_RE = re.compile(r"\bnăm\s+(19|20)\d{2}\b", re.IGNORECASE)
_VN_QUARTER_RE = re.compile(r"\bquý\s+[1-4IViv]+\b", re.IGNORECASE)


def extract_time_anchors(text: str) -> list[str]:
    """Extract temporal references from text.

    Returns a list of normalized time strings, e.g. ["2019", "Q3", "FY2021"].
    """
    if not text:
        return []

    anchors: list[str] = []

    # Fiscal year patterns (must come before generic year to capture "FY2021")
    for m in _FISCAL_YEAR_RE.finditer(text):
        anchors.append(m.group(0).strip().upper().replace(" ", ""))

    # Generic 4-digit years
    for m in _YEAR_RE.finditer(text):
        year = m.group(0)
        if year not in anchors:
            anchors.append(year)

    # Quarters
    for m in _QUARTER_RE.finditer(text):
        anchors.append(m.group(0).upper())

    # Halves
    for m in _HALF_RE.finditer(text):
        anchors.append(m.group(0).upper())

    # Vietnamese year
    for m in _VN_YEAR_RE.finditer(text):
        year = _YEAR_RE.search(m.group(0))
        if year and year.group(0) not in anchors:
            anchors.append(year.group(0))

    # Vietnamese quarter
    for m in _VN_QUARTER_RE.finditer(text):
        anchors.append(m.group(0).strip())

    return anchors


# ---------------------------------------------------------------------------
# Financial entity extraction
# ---------------------------------------------------------------------------

_FINANCIAL_ENTITIES = [
    # English
    "revenue", "net revenue", "total revenue",
    "net income", "net profit", "net loss",
    "gross profit", "gross margin",
    "operating income", "operating profit", "operating expenses", "operating margin",
    "ebitda", "ebit",
    "earnings per share", "eps",
    "total assets", "total liabilities", "total equity",
    "stockholders equity", "shareholders equity",
    "cash flow", "free cash flow", "operating cash flow",
    "cost of revenue", "cost of goods sold", "cogs",
    "interest expense", "interest income",
    "income tax", "tax expense",
    "depreciation", "amortization",
    "dividends", "dividend per share",
    "return on equity", "roe", "return on assets", "roa",
    "debt to equity", "current ratio",
    "accounts receivable", "accounts payable",
    "inventory", "goodwill",
    "capital expenditure", "capex",
    # Vietnamese
    "doanh thu", "lợi nhuận", "lợi nhuận ròng", "lợi nhuận gộp",
    "chi phí", "chi phí hoạt động", "tổng tài sản", "vốn chủ sở hữu",
    "nợ phải trả", "dòng tiền", "thu nhập trên mỗi cổ phiếu",
    "biên lợi nhuận", "tỷ suất",
]

# Pre-compile patterns for fast matching
_ENTITY_PATTERNS = [
    re.compile(r"\b" + re.escape(e) + r"\b", re.IGNORECASE)
    for e in _FINANCIAL_ENTITIES
]


def extract_financial_entities(text: str) -> list[str]:
    """Extract recognized financial metric names from text."""
    if not text:
        return []
    found: list[str] = []
    for pattern, entity_name in zip(_ENTITY_PATTERNS, _FINANCIAL_ENTITIES):
        if pattern.search(text):
            found.append(entity_name)
    return found


# ---------------------------------------------------------------------------
# Tokenization helper
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> set[str]:
    """Simple word-level tokenization, lowercased."""
    return {m.group(0).lower() for m in _TOKEN_RE.finditer(text)}


# ---------------------------------------------------------------------------
# Re-ranking
# ---------------------------------------------------------------------------

def rerank_contexts(
    question: str,
    contexts: list[dict[str, Any]],
    top_k: int = 6,
    *,
    time_weight: float = 3.0,
    entity_weight: float = 2.0,
    keyword_weight: float = 1.0,
) -> list[dict[str, Any]]:
    """Re-rank retrieved contexts by relevance to the question.

    Scoring factors:
    1. **Keyword overlap** — proportion of question tokens found in context.
    2. **Time anchor match** — bonus for contexts containing the same
       temporal references as the question.
    3. **Financial entity match** — bonus for contexts mentioning the same
       financial metrics as the question.

    Parameters
    ----------
    question : str
        The user's financial question.
    contexts : list of dicts
        Retrieved contexts from vector search, each containing at least
        ``content`` and optionally ``raw_content``.
    top_k : int
        Number of top results to return.
    time_weight, entity_weight, keyword_weight : float
        Relative weights for each scoring component.

    Returns
    -------
    Re-ranked list of context dicts, trimmed to ``top_k``, with
    ``rerank_score`` added to each item's metadata.
    """
    if not contexts:
        return []

    q_tokens = _tokenize(question)
    q_time_anchors = set(extract_time_anchors(question))
    q_entities = set(extract_financial_entities(question))

    scored: list[tuple[float, int, dict[str, Any]]] = []

    for idx, ctx in enumerate(contexts):
        # Combine content + raw_content for matching
        ctx_text = " ".join(filter(None, [
            str(ctx.get("content", "")),
            str(ctx.get("raw_content", "")),
        ]))

        # 1. Keyword overlap
        ctx_tokens = _tokenize(ctx_text)
        overlap = len(q_tokens & ctx_tokens)
        keyword_score = overlap / max(len(q_tokens), 1)

        # 2. Time anchor match
        ctx_time = set(extract_time_anchors(ctx_text))
        time_matches = len(q_time_anchors & ctx_time)
        time_score = time_matches / max(len(q_time_anchors), 1) if q_time_anchors else 0.0

        # 3. Financial entity match
        ctx_entities = set(extract_financial_entities(ctx_text))
        entity_matches = len(q_entities & ctx_entities)
        entity_score = entity_matches / max(len(q_entities), 1) if q_entities else 0.0

        # Composite score
        total = (
            keyword_weight * keyword_score
            + time_weight * time_score
            + entity_weight * entity_score
        )

        # Use negative vector distance as tie-breaker (lower distance = better)
        distance = ctx.get("distance", 1.0)
        total -= 0.01 * distance  # small bonus for closer vectors

        scored.append((total, idx, ctx))

    # Sort by score descending, then by original order for ties
    scored.sort(key=lambda x: (-x[0], x[1]))

    results: list[dict[str, Any]] = []
    for score, _idx, ctx in scored[:top_k]:
        ctx_copy = dict(ctx)
        meta = dict(ctx_copy.get("metadata", {}))
        meta["rerank_score"] = round(score, 4)
        ctx_copy["metadata"] = meta
        results.append(ctx_copy)

    return results
