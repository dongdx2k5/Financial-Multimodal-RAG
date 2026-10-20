"""Dense vector retrieval using BGE-M3 embeddings and FAISS index with NumPy fallback."""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from financial_rag.config import settings
from financial_rag.db.repository import FinancialRepository


logger = logging.getLogger(__name__)


class DenseRetriever:
    """Encodes chunks into dense embeddings and performs vector similarity search."""

    def __init__(
        self,
        repo: FinancialRepository,
        model_name: Optional[str] = None,
        index_path: Optional[Union[str, Path]] = None,
        chunk_ids_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
    ):
        self.repo = repo
        self.model_name = model_name or settings.embedding_model
        self.device = device or settings.embedding_device
        self.index_path = Path(index_path) if index_path else settings.resolve_path(settings.faiss_index_path)
        self.chunk_ids_path = Path(chunk_ids_path) if chunk_ids_path else settings.resolve_path(settings.chunk_ids_path)

        self.model = None
        self.index = None
        self.chunk_ids: List[int] = []
        self.embeddings: Optional[np.ndarray] = None  # Fallback matrix if faiss not available

    def _load_model(self):
        if self.model is None and SentenceTransformer is not None:
            try:
                self.model = SentenceTransformer(self.model_name, device=self.device)
            except Exception as e:
                logger.warning(f"Could not load SentenceTransformer {self.model_name}: {e}. Fallback to TF-IDF.")
                self.model = None

    def build_index(self, chunks: Optional[List[Dict[str, Any]]] = None) -> int:
        """Encode chunks and build vector index."""
        if chunks is None:
            chunks = self.repo.get_all_chunks()

        if not chunks:
            logger.warning("No chunks available to index.")
            return 0

        self.chunk_ids = [c["chunk_id"] for c in chunks]
        texts = [f"{c.get('context_prefix', '')}\n{c['content']}" for c in chunks]

        self._load_model()
        if self.model is not None:
            raw_embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            self.embeddings = raw_embeddings.astype(np.float32)
        else:
            # Fallback lightweight TF-IDF dense embeddings
            from sklearn.feature_extraction.text import TfidfVectorizer
            vectorizer = TfidfVectorizer(max_features=512)
            dense_mat = vectorizer.fit_transform(texts).toarray()
            # Normalize rows
            norms = np.linalg.norm(dense_mat, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self.embeddings = (dense_mat / norms).astype(np.float32)

        dim = self.embeddings.shape[1]

        # Build FAISS index if library is present
        if faiss is not None:
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(self.embeddings)
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self.index, str(self.index_path))

        # Save chunk IDs and numpy embeddings
        self.chunk_ids_path.parent.mkdir(parents=True, exist_ok=True)
        self.chunk_ids_path.write_text(json.dumps(self.chunk_ids), encoding="utf-8")
        npy_path = self.index_path.with_suffix(".npy")
        np.save(str(npy_path), self.embeddings)

        logger.info(f"Built dense index for {len(self.chunk_ids)} chunks.")
        return len(self.chunk_ids)

    def load_index(self) -> bool:
        """Load index and chunk mapping from disk."""
        if not self.chunk_ids_path.exists():
            return False

        self.chunk_ids = json.loads(self.chunk_ids_path.read_text(encoding="utf-8"))

        if faiss is not None and self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            return True

        npy_path = self.index_path.with_suffix(".npy")
        if npy_path.exists():
            self.embeddings = np.load(str(npy_path))
            return True

        return False

    def search(
        self,
        query: str,
        limit: int = 10,
    ) -> List[Tuple[int, float]]:
        """Search top-k chunks returning (chunk_id, score) pairs."""
        if not self.chunk_ids:
            if not self.load_index():
                self.build_index()

        if not self.chunk_ids:
            return []

        self._load_model()
        if self.model is not None:
            q_emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        elif self.embeddings is not None:
            # Fallback text overlap vector
            from sklearn.feature_extraction.text import TfidfVectorizer
            all_chunks = self.repo.get_all_chunks()
            texts = [f"{c.get('context_prefix', '')}\n{c['content']}" for c in all_chunks]
            vec = TfidfVectorizer(max_features=self.embeddings.shape[1])
            vec.fit(texts)
            q_emb = vec.transform([query]).toarray()
            q_norm = np.linalg.norm(q_emb)
            if q_norm > 0:
                q_emb = (q_emb / q_norm).astype(np.float32)
            else:
                q_emb = np.zeros((1, self.embeddings.shape[1]), dtype=np.float32)
        else:
            return []

        # FAISS search
        if self.index is not None:
            scores, indices = self.index.search(q_emb, min(limit, len(self.chunk_ids)))
            results = []
            for idx, score in zip(indices[0], scores[0]):
                if 0 <= idx < len(self.chunk_ids):
                    results.append((self.chunk_ids[idx], float(score)))
            return results

        # NumPy cosine fallback
        if self.embeddings is not None:
            sims = np.dot(self.embeddings, q_emb.T).flatten()
            top_indices = np.argsort(sims)[::-1][:limit]
            return [(self.chunk_ids[i], float(sims[i])) for i in top_indices]

        return []
