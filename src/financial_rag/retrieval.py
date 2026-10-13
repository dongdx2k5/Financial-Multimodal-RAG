"""Hybrid retriever combining TF-IDF (lexical) and dense embeddings (semantic).

When no embedder is configured the retriever falls back to TF-IDF only,
preserving full backward compatibility with the original MVP pipeline.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .extraction import extract_metric_from_query
from .graph import KnowledgeGraph
from .models import Chunk, Evidence

logger = logging.getLogger(__name__)

STOP_WORDS_UPPER = {
    "AND", "FOR", "THE", "WHAT", "FROM", "WITH", "THIS", "THAT",
    "NĂM", "VND", "USD", "BÁO", "CÁO", "TÀI", "CHÍNH", "QUÝ",
}


def extract_query_metadata(
    query: str, known_tickers: set[str] | None = None
) -> tuple[str | None, int | None, str | None]:
    """Extract stock ticker, 4-digit fiscal year, and metric taxonomy code from query.

    Returns
    -------
    (ticker, year, standardized_metric_code)
    """
    matched_ticker: str | None = None
    matched_year: int | None = None

    # Detect 4-digit year (1900-2099)
    year_match = re.search(r"\b(20\d{2}|19\d{2})\b", query)
    if year_match:
        matched_year = int(year_match.group(1))

    # Match known tickers from indexed chunks first
    if known_tickers:
        for ticker in sorted(known_tickers, key=len, reverse=True):
            if re.search(rf"\b{re.escape(ticker)}\b", query, re.IGNORECASE):
                matched_ticker = ticker.upper()
                break

    # Fallback to uppercase 3-5 letter words
    if not matched_ticker:
        for word in re.findall(r"\b[A-Za-z]{3,5}\b", query):
            upper = word.upper()
            if upper not in STOP_WORDS_UPPER and (word.isupper() or len(word) <= 4):
                matched_ticker = upper
                break

    # Detect financial metric term in query (Phase 1.1)
    _, matched_metric_code = extract_metric_from_query(query)

    return matched_ticker, matched_year, matched_metric_code


class CrossEncoderReranker:
    """Two-Stage Retrieval: Stage 2 Cross-Encoder Reranker using FlashRank (Phase 3.1)."""

    def __init__(
        self,
        model_name: str = "ms-marco-TinyBERT-L-2-v2",
        cache_dir: str | None = None,
        max_length: int = 512,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self._ranker = None
        try:
            import tempfile

            from flashrank import Ranker

            cache = cache_dir or str(Path(tempfile.gettempdir()) / "flashrank")
            self._ranker = Ranker(model_name=model_name, cache_dir=cache, max_length=max_length)
            logger.info("CrossEncoderReranker initialized (%s)", model_name)
        except Exception as exc:
            logger.warning("FlashRank initialization notice: %s. Reranking disabled.", exc)

    @property
    def is_available(self) -> bool:
        return self._ranker is not None

    def rerank(
        self,
        query: str,
        candidates: list[Evidence],
        top_k: int = 5,
    ) -> list[Evidence]:
        """Rerank candidate evidences using cross-encoder relevance scoring."""
        if not candidates or len(candidates) <= 1:
            return candidates[:top_k]
        if self._ranker is None:
            return candidates[:top_k]

        try:
            from flashrank import RerankRequest

            passages = [
                {"id": i, "text": cand.text}
                for i, cand in enumerate(candidates)
            ]
            req = RerankRequest(query=query, passages=passages)
            results = self._ranker.rerank(req)
            reranked: list[Evidence] = []
            for item in results[:top_k]:
                idx = int(item["id"])
                rerank_score = float(item["score"])
                original = candidates[idx]
                reranked.append(
                    Evidence(
                        chunk_id=original.chunk_id,
                        source=original.source,
                        page=original.page,
                        text=original.text,
                        score=rerank_score,
                        graph_facts=original.graph_facts,
                    )
                )
            return reranked
        except Exception as exc:
            logger.warning("Reranking failed: %s. Returning top candidates.", exc)
            return candidates[:top_k]


class HybridRetriever:
    def __init__(
        self,
        chunks_path: Path,
        graph: KnowledgeGraph,
        embedder: object | None = None,
        db: object | None = None,
        reranker: CrossEncoderReranker | None = None,
        enable_reranker: bool = True,
    ):
        self.chunks_path = chunks_path
        self.graph = graph
        self.embedder = embedder  # Optional GeminiEmbedder (or any embedder with the same API)
        self.db = db  # Optional FinancialDatabase for SQLite-vec persistent vector storage
        self.reranker = reranker  # Optional CrossEncoderReranker
        self.enable_reranker = enable_reranker
        self.chunks: list[Chunk] = []
        self.vectorizer: TfidfVectorizer | None = None
        self.matrix = None
        self.dense_matrix: np.ndarray | None = None
        self.load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load(self) -> None:
        if self.chunks_path.exists():
            raw = json.loads(self.chunks_path.read_text(encoding="utf-8"))
            self.chunks = [Chunk.model_validate(item) for item in raw]
        self._fit()
        self._load_dense()

    def _save_chunks(self) -> None:
        self.chunks_path.parent.mkdir(parents=True, exist_ok=True)
        self.chunks_path.write_text(
            json.dumps([chunk.model_dump() for chunk in self.chunks], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # TF-IDF (sparse)
    # ------------------------------------------------------------------

    def _fit(self) -> None:
        if not self.chunks:
            self.vectorizer, self.matrix = None, None
            return
        self.vectorizer = TfidfVectorizer(
            lowercase=True, ngram_range=(1, 2), max_features=50_000, token_pattern=r"(?u)\b\w+\b"
        )
        self.matrix = self.vectorizer.fit_transform([chunk.text for chunk in self.chunks])

    # ------------------------------------------------------------------
    # Dense embeddings (Phase 3.2: SQLite-vec unified storage)
    # ------------------------------------------------------------------

    @property
    def _dense_path(self) -> Path:
        return self.chunks_path.with_suffix(".dense.npy")

    def _load_dense(self) -> None:
        """Load precomputed dense embeddings from SQLite database (Phase 3.2).

        Migrates legacy .dense.npy to SQLite if found, and removes the flat npy file.
        """
        # 1. Try loading from SQLite database first
        if self.db is not None and self.chunks and hasattr(self.db, "get_chunk_embeddings"):
            chunk_ids = [c.id for c in self.chunks]
            saved = self.db.get_chunk_embeddings(chunk_ids)
            if len(saved) == len(self.chunks) and all(cid in saved for cid in chunk_ids):
                self.dense_matrix = np.array([saved[cid] for cid in chunk_ids], dtype=np.float32)
                # Remove legacy file if it exists
                if self._dense_path.exists():
                    try:
                        self._dense_path.unlink()
                    except OSError as exc:
                        logger.debug("Legacy dense unlink notice: %s", exc)
                return

        # 2. Migration from legacy .dense.npy if it exists
        if self._dense_path.exists():
            try:
                legacy = np.load(self._dense_path)
                if len(legacy) == len(self.chunks):
                    self.dense_matrix = legacy
                    if self.db is not None and hasattr(self.db, "save_chunk_embeddings"):
                        self.db.save_chunk_embeddings([c.id for c in self.chunks], self.dense_matrix)
                        self._dense_path.unlink(missing_ok=True)
                    return
            except Exception as exc:
                logger.warning("Legacy dense embedding load failed: %s", exc)

        self.dense_matrix = None

    def _save_dense(self) -> None:
        """Persist dense embeddings directly to SQLite database (Phase 3.2)."""
        if (
            self.dense_matrix is not None
            and self.db is not None
            and self.chunks
            and hasattr(self.db, "save_chunk_embeddings")
        ):
            chunk_ids = [chunk.id for chunk in self.chunks]
            self.db.save_chunk_embeddings(chunk_ids, self.dense_matrix)
        if self._dense_path.exists():
            try:
                self._dense_path.unlink(missing_ok=True)
            except OSError as exc:
                logger.debug("Legacy dense unlink notice: %s", exc)

    def _compute_dense(self) -> None:
        """Compute and persist dense embeddings into SQLite database (Phase 3.2)."""
        if self.embedder is None or not self.chunks:
            return
        texts = [chunk.text for chunk in self.chunks]
        chunk_ids = [chunk.id for chunk in self.chunks]
        try:
            self.dense_matrix = self.embedder.embed_documents(texts)  # type: ignore[union-attr]
            if (
                self.db is not None
                and self.dense_matrix is not None
                and hasattr(self.db, "save_chunk_embeddings")
            ):
                self.db.save_chunk_embeddings(chunk_ids, self.dense_matrix)
            if self._dense_path.exists():
                try:
                    self._dense_path.unlink(missing_ok=True)
                except OSError as exc:
                    logger.debug("Legacy dense unlink notice: %s", exc)
        except Exception as exc:
            logger.warning(
                "Dense embedding computation failed: %s. Continuing with sparse search.", exc
            )
            self.dense_matrix = None

    # ------------------------------------------------------------------
    # Index mutation
    # ------------------------------------------------------------------

    def upsert(self, chunks: list[Chunk]) -> None:
        current = {chunk.id: chunk for chunk in self.chunks}
        current.update({chunk.id: chunk for chunk in chunks})
        self.chunks = list(current.values())
        self._save_chunks()
        self._fit()
        self._compute_dense()

    def sync(self, chunks: list[Chunk]) -> None:
        self.chunks = list(chunks)
        self._save_chunks()
        self._fit()
        self._compute_dense()

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    @property
    def known_tickers(self) -> set[str]:
        return {c.ticker.upper() for c in self.chunks if c.ticker}

    def search(
        self,
        query: str,
        top_k: int = 5,
        ticker: str | None = None,
        year: int | None = None,
        use_rrf: bool = True,
    ) -> tuple[list[Evidence], str | None]:
        """Hybrid search returning (evidence_list, detected_metric_code).

        The second return value is the standardized metric taxonomy code detected
        from the query (e.g. 'REVENUE_NET'), or None. The service layer uses this
        to pull historical time-series facts from the Knowledge Graph (Phase 1.1).
        """
        if not self.chunks:
            return [], None

        # 1. Automatic query metadata detection if not explicitly passed
        extracted_ticker, extracted_year, detected_metric = extract_query_metadata(query, self.known_tickers)
        target_ticker = ticker or extracted_ticker
        target_year = year or extracted_year

        n = len(self.chunks)

        # 2. TF-IDF sparse scores (always available)
        sparse_scores = np.zeros(n)
        if self.vectorizer is not None and self.matrix is not None:
            query_vector = self.vectorizer.transform([query])
            sparse_scores = (self.matrix @ query_vector.T).toarray().ravel()

        # 3. Dense embedding scores (only when embedder + vectors are ready)
        dense_scores = np.zeros(n)
        has_dense = self.dense_matrix is not None and len(self.dense_matrix) == n
        if has_dense and self.embedder is not None:
            try:
                query_embedding = self.embedder.embed_query(query)  # type: ignore[union-attr]
                norms = np.linalg.norm(self.dense_matrix, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1, norms)
                normalized = self.dense_matrix / norms
                q_norm = np.linalg.norm(query_embedding) or 1.0
                dense_scores = (normalized @ (query_embedding / q_norm)).ravel()
            except Exception as exc:
                logger.warning(
                    "Dense query embedding failed: %s. Using sparse only.", exc
                )
                has_dense = False

        # 4. Hybrid Ranking: Reciprocal Rank Fusion (RRF) vs linear
        if use_rrf and has_dense:
            k_rrf = 60
            sparse_ranks = np.argsort(np.argsort(-sparse_scores))
            dense_ranks = np.argsort(np.argsort(-dense_scores))
            base_scores = (1.0 / (k_rrf + sparse_ranks)) + (1.0 / (k_rrf + dense_ranks))
        elif has_dense:
            alpha = 0.35  # sparse weight
            base_scores = alpha * sparse_scores + (1 - alpha) * dense_scores
        else:
            base_scores = sparse_scores

        # 5. Metadata Boost: prioritize chunks matching ticker or year
        metadata_boost = np.zeros(n)
        for i, chunk in enumerate(self.chunks):
            boost = 0.0
            if target_ticker and chunk.ticker and chunk.ticker.upper() == target_ticker.upper():
                boost += 0.05
            if target_year and chunk.year == target_year:
                boost += 0.05
            metadata_boost[i] = boost

        final_scores = base_scores + metadata_boost

        # 6. Candidate Filtering: prioritize chunks matching target ticker/year
        matching_indices = [
            i
            for i, chunk in enumerate(self.chunks)
            if (not target_ticker or (chunk.ticker and chunk.ticker.upper() == target_ticker.upper()))
            and (not target_year or chunk.year == target_year)
        ]
        if matching_indices and any(final_scores[i] > 0 for i in matching_indices):
            candidate_mask = np.zeros(n, dtype=bool)
            candidate_mask[matching_indices] = True
            final_scores = np.where(candidate_mask, final_scores, final_scores * 0.1)

        # Stage 1: Candidate selection (Two-Stage Retrieval - Phase 3.1)
        has_reranker = (
            self.enable_reranker
            and self.reranker is not None
            and self.reranker.is_available
        )
        k_candidates = max(20, top_k * 4) if has_reranker else top_k

        indices = np.argsort(final_scores)[::-1][:k_candidates]
        candidates = [
            Evidence(
                chunk_id=self.chunks[index].id,
                source=self.chunks[index].source,
                page=self.chunks[index].page,
                text=self.chunks[index].text,
                score=float(final_scores[index]),
                graph_facts=self.graph.facts_for_chunk(self.chunks[index].id),
            )
            for index in indices
            if final_scores[index] > 0
        ]

        # Stage 2: Cross-Encoder Reranking
        if has_reranker and len(candidates) > top_k:
            evidence = self.reranker.rerank(query, candidates, top_k=top_k)
        else:
            evidence = candidates[:top_k]

        return evidence, detected_metric
