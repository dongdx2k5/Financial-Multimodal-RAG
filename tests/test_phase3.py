"""Phase 3 tests – Cross-Encoder Reranker (3.1) & SQLite-vec storage (3.2).

Tests Two-Stage Retrieval with FlashRank and vector storage/search in SQLite.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from financial_rag.config import Settings
from financial_rag.database import FinancialDatabase
from financial_rag.graph import KnowledgeGraph
from financial_rag.models import Chunk, Document, Evidence, PageContent
from financial_rag.retrieval import CrossEncoderReranker, HybridRetriever
from financial_rag.service import FinancialRAGService

# =====================================================================
# Bước 3.1 – Cross-Encoder Reranker
# =====================================================================

class TestCrossEncoderReranker:
    def test_reranker_initialization(self):
        reranker = CrossEncoderReranker()
        # FlashRank should be available in the environment
        assert reranker.is_available is True

    def test_reranker_reorders_by_relevance(self):
        reranker = CrossEncoderReranker()
        query = "Doanh thu thuần năm 2024 của HPG là bao nhiêu?"

        irrelevant_chunk = Chunk(
            id="c1",
            document_id="d1",
            source="doc1.pdf",
            page=1,
            text="Thời tiết hôm nay tại thủ đô Hà Nội nhiều mây và có mưa rào rải rác.",
        )
        relevant_chunk = Chunk(
            id="c2",
            document_id="d1",
            source="doc1.pdf",
            page=2,
            text="Doanh thu thuần về bán hàng của tập đoàn HPG trong năm 2024 đạt mốc 125.500 tỷ VND.",
        )

        candidates = [
            # Irrelevant placed first with artificially high initial score
            Evidence(
                chunk_id=irrelevant_chunk.id,
                source=irrelevant_chunk.source,
                page=irrelevant_chunk.page,
                text=irrelevant_chunk.text,
                score=0.99,
            ),
            Evidence(
                chunk_id=relevant_chunk.id,
                source=relevant_chunk.source,
                page=relevant_chunk.page,
                text=relevant_chunk.text,
                score=0.10,
            ),
        ]

        reranked = reranker.rerank(query, candidates, top_k=2)
        assert len(reranked) == 2
        # The relevant financial chunk must be promoted to rank 1
        assert reranked[0].chunk_id == relevant_chunk.id
        assert "125.500" in reranked[0].text
        assert reranked[0].score > reranked[1].score

    def test_reranker_fallback_when_unavailable(self):
        reranker = CrossEncoderReranker()
        reranker._ranker = None  # Force disable
        assert reranker.is_available is False

        candidates = [
            Evidence(chunk_id="c1", source="s", page=1, text="Text 1", score=0.8),
            Evidence(chunk_id="c2", source="s", page=2, text="Text 2", score=0.5),
        ]
        res = reranker.rerank("Query", candidates, top_k=1)
        assert len(res) == 1
        assert res[0].chunk_id == "c1"

    def test_hybrid_retriever_two_stage_flow(self, tmp_path: Path):
        chunks_file = tmp_path / "chunks.json"
        graph = KnowledgeGraph(tmp_path / "graph.json")
        retriever = HybridRetriever(
            chunks_file,
            graph,
            reranker=CrossEncoderReranker(),
            enable_reranker=True,
        )

        c1 = Chunk(id="c1", document_id="d1", source="s", page=1, text="Tin tức thể thao bóng đá quốc tế", ticker="DEMO")
        c2 = Chunk(id="c2", document_id="d1", source="s", page=2, text="Lợi nhuận sau thuế năm 2024 đạt 35 tỷ đồng", ticker="DEMO")
        retriever.upsert([c1, c2])

        evidence, _ = retriever.search("Lợi nhuận sau thuế năm 2024 là bao nhiêu?", top_k=1)
        assert len(evidence) == 1
        assert evidence[0].chunk_id == "c2"


# =====================================================================
# Bước 3.2 – SQLite-vec Vector Storage
# =====================================================================

class TestSQLiteVecStorage:
    def test_sqlite_vec_extension_loaded(self, tmp_path: Path):
        db = FinancialDatabase(tmp_path / "test.db")
        assert db.has_sqlite_vec is True

    def test_save_and_get_chunk_embeddings(self, tmp_path: Path):
        db = FinancialDatabase(tmp_path / "test.db")
        chunk_ids = ["c1", "c2"]
        vectors = np.array(
            [[0.1, 0.2, 0.3, 0.4], [0.9, 0.8, 0.7, 0.6]],
            dtype=np.float32,
        )

        db.save_chunk_embeddings(chunk_ids, vectors)
        loaded = db.get_chunk_embeddings(chunk_ids)

        assert len(loaded) == 2
        assert np.allclose(loaded["c1"], vectors[0])
        assert np.allclose(loaded["c2"], vectors[1])

    def test_knn_search_vector(self, tmp_path: Path):
        db = FinancialDatabase(tmp_path / "test.db")
        chunk_ids = ["apple", "banana"]
        vectors = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        )
        db.save_chunk_embeddings(chunk_ids, vectors)

        # Query vector close to 'apple'
        query_vec = np.array([0.9, 0.1, 0.0], dtype=np.float32)
        results = db.search_vector(query_vec, limit=2)

        assert len(results) >= 1
        top_id, top_score = results[0]
        assert top_id == "apple"
        assert top_score > 0.8

    def test_dense_embeddings_stored_in_sqlite_without_npy(self, tmp_path: Path):
        db = FinancialDatabase(tmp_path / "test.db")
        chunks_file = tmp_path / "index" / "chunks.json"
        graph = KnowledgeGraph(tmp_path / "index" / "graph.json")

        mock_embedder = MagicMock()
        mock_embedder.embed_documents.return_value = np.array(
            [[0.2, 0.4, 0.6], [0.8, 0.6, 0.4]],
            dtype=np.float32,
        )

        retriever = HybridRetriever(
            chunks_file,
            graph,
            embedder=mock_embedder,
            db=db,
        )

        c1 = Chunk(id="c1", document_id="d1", source="doc.pdf", page=1, text="Text A")
        c2 = Chunk(id="c2", document_id="d1", source="doc.pdf", page=2, text="Text B")
        retriever.upsert([c1, c2])

        # Verify embeddings were persisted in SQLite
        saved_embs = db.get_chunk_embeddings(["c1", "c2"])
        assert len(saved_embs) == 2

        # Verify .dense.npy file does NOT exist
        npy_path = chunks_file.with_suffix(".dense.npy")
        assert not npy_path.exists()

    def test_delete_document_cleans_up_vectors(self, tmp_path: Path):
        db = FinancialDatabase(tmp_path / "test.db")
        doc = Document(
            id="doc1",
            source="test.pdf",
            pages=[PageContent(page=1, text="Sample")],
        )
        chunks = [Chunk(id="c1", document_id="doc1", source="test.pdf", page=1, text="Sample")]
        db.save_document(doc, chunks, [])
        db.save_chunk_embeddings(["c1"], np.array([[0.5, 0.5]], dtype=np.float32))

        assert len(db.get_chunk_embeddings(["c1"])) == 1

        db.delete_document("doc1")
        assert len(db.get_chunk_embeddings(["c1"])) == 0


# =====================================================================
# End-to-End Service Integration
# =====================================================================

class TestPhase3ServiceIntegration:
    def test_service_stats_shows_sqlite_vec_and_reranker(self, tmp_path: Path):
        settings = Settings(data_dir=tmp_path / "data")
        svc = FinancialRAGService(settings)
        stats = svc.get_stats()

        assert "sqlite_vec_enabled" in stats
        assert stats["sqlite_vec_enabled"] is True
        assert "reranker_enabled" in stats
        assert stats["reranker_enabled"] is True
