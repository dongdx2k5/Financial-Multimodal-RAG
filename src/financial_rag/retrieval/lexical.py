"""Lexical retrieval using SQLite FTS5 with BM25 ranking."""

from typing import Any, Dict, List, Optional
from financial_rag.db.repository import FinancialRepository


class LexicalRetriever:
    """Retrieves document chunks using SQLite FTS5 full-text search with metadata filtering."""

    def __init__(self, repo: FinancialRepository):
        self.repo = repo

    def search(
        self,
        query: str,
        ticker: Optional[str] = None,
        fiscal_year: Optional[int] = None,
        period_type: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Execute FTS5 search on chunk content.
        Returns chunks ordered by BM25 relevance score.
        """
        results = self.repo.search_chunks_fts(
            query_text=query,
            ticker=ticker,
            fiscal_year=fiscal_year,
            period_type=period_type,
            limit=limit,
        )

        # Normalize bm25_rank to 0..1 scale (BM25 in SQLite is negative where lower is better)
        formatted: List[Dict[str, Any]] = []
        for r in results:
            bm25 = r.get("bm25_rank", 0.0)
            # Convert negative bm25 score to positive score: -bm25
            score = max(0.0, -float(bm25))
            r_copy = dict(r)
            r_copy["lexical_score"] = score
            formatted.append(r_copy)

        return formatted
