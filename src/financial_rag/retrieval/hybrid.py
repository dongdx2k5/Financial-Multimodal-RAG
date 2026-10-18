"""Hybrid retrieval combining dense vector search and SQLite FTS5 lexical search."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from financial_rag.config import settings
from financial_rag.db.repository import FinancialRepository
from financial_rag.retrieval.dense import DenseRetriever
from financial_rag.retrieval.lexical import LexicalRetriever
from financial_rag.retrieval.query_expansion import QueryExpander


@dataclass
class RetrievedChunk:
    chunk_id: int
    report_id: int
    ticker: str
    fiscal_year: int
    period_type: str
    section: Optional[str]
    chunk_type: str
    page_start: int
    page_end: int
    context_prefix: str
    content: str
    dense_score: float = 0.0
    lexical_score: float = 0.0
    hybrid_score: float = 0.0

    def to_citation(self) -> str:
        """Format citation string for provenance verification."""
        page_str = f"Trang {self.page_start}" if self.page_start == self.page_end else f"Trang {self.page_start}-{self.page_end}"
        return f"[{self.ticker} | {self.period_type}/{self.fiscal_year} | {page_str} | {self.section or 'Nội dung'}]"


class HybridRetriever:
    """Orchestrates query expansion, dense FAISS retrieval, FTS5 search, and rank fusion."""

    def __init__(
        self,
        repo: FinancialRepository,
        dense_weight: float = 0.6,
        fts_weight: float = 0.4,
        relevance_threshold: float = 0.35,
    ):
        self.repo = repo
        self.dense_retriever = DenseRetriever(repo=repo)
        self.lexical_retriever = LexicalRetriever(repo=repo)
        self.expander = QueryExpander(repo=repo)
        self.dense_weight = dense_weight
        self.fts_weight = fts_weight
        self.relevance_threshold = relevance_threshold

    def retrieve(
        self,
        query: str,
        ticker: Optional[str] = None,
        fiscal_year: Optional[int] = None,
        period_type: Optional[str] = None,
        top_k: int = 4,
    ) -> List[RetrievedChunk]:
        """
        Execute hybrid search:
        1. Expand query with bilingual aliases.
        2. Retrieve dense vector matches.
        3. Retrieve lexical FTS5 matches with metadata filtering.
        4. Fuse scores via normalized linear combination.
        5. Return top-k chunks.
        """
        expanded_query = self.expander.expand(query)

        # 1. Lexical FTS5 Retrieval
        fts_results = self.lexical_retriever.search(
            query=expanded_query,
            ticker=ticker,
            fiscal_year=fiscal_year,
            period_type=period_type,
            limit=top_k * 3,
        )

        # 2. Dense Vector Retrieval
        dense_results = self.dense_retriever.search(
            query=expanded_query,
            limit=top_k * 3,
        )

        # Score map: chunk_id -> dict
        merged: Dict[int, Dict[str, Any]] = {}

        # Process FTS
        max_lex = max([r["lexical_score"] for r in fts_results], default=1.0) or 1.0
        for r in fts_results:
            cid = r["chunk_id"]
            merged[cid] = {
                "chunk": r,
                "lex_score": r["lexical_score"] / max_lex,
                "dense_score": 0.0,
            }

        # Process Dense
        max_dense = max([s for _, s in dense_results], default=1.0) or 1.0
        for cid, score in dense_results:
            chunk = self.repo.get_chunk_by_id(cid)
            if not chunk:
                continue

            # Apply metadata filters if specified
            if ticker and chunk.get("ticker", "").upper() != ticker.upper():
                continue
            if fiscal_year and chunk.get("fiscal_year") != fiscal_year:
                continue
            if period_type and str(chunk.get("period_type", "")).upper() != period_type.upper():
                continue

            norm_dense = score / max_dense if max_dense > 0 else 0.0
            if cid in merged:
                merged[cid]["dense_score"] = norm_dense
            else:
                merged[cid] = {
                    "chunk": chunk,
                    "lex_score": 0.0,
                    "dense_score": norm_dense,
                }

        # Compute fused score
        ranked: List[RetrievedChunk] = []
        for cid, data in merged.items():
            ch = data["chunk"]
            d_s = data["dense_score"]
            l_s = data["lex_score"]
            hybrid_score = (self.dense_weight * d_s) + (self.fts_weight * l_s)

            ranked.append(RetrievedChunk(
                chunk_id=cid,
                report_id=ch["report_id"],
                ticker=ch["ticker"],
                fiscal_year=ch["fiscal_year"],
                period_type=ch["period_type"],
                section=ch.get("section"),
                chunk_type=ch.get("chunk_type", "text"),
                page_start=ch.get("page_start", 1),
                page_end=ch.get("page_end", 1),
                context_prefix=ch.get("context_prefix", ""),
                content=ch["content"],
                dense_score=d_s,
                lexical_score=l_s,
                hybrid_score=hybrid_score,
            ))

        ranked.sort(key=lambda x: x.hybrid_score, reverse=True)
        return ranked[:top_k]
