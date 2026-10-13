"""Dense embedding client for multilingual retrieval (Vietnamese + English).

Uses the official Google GenAI SDK (``google-genai``) with fallback to ``google.generativeai``.
Embeddings are persisted as a ``.dense.npy`` file next to ``chunks.json``
so they only need to be recomputed when new chunks are ingested.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class GeminiEmbedder:
    """Wrapper around Google GenAI embedding models."""

    def __init__(self, api_key: str, model_name: str = "gemini-embedding-001"):
        self.api_key = api_key
        self.clean_model_name = model_name.removeprefix("models/")
        self._client = None
        self._legacy_genai = None

        try:
            from google import genai

            self._client = genai.Client(api_key=api_key)
        except ImportError:
            try:
                import google.generativeai as genai_legacy

                genai_legacy.configure(api_key=api_key)
                self._legacy_genai = genai_legacy
            except ImportError as exc:
                raise ImportError(
                    "Either 'google-genai' or 'google-generativeai' is required. "
                    "Install with: pip install google-genai"
                ) from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of document texts.

        Batches of up to 100 texts are processed.
        """
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        all_embeddings: list[list[float]] = []
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            if self._client is not None:
                response = self._client.models.embed_content(
                    model=self.clean_model_name,
                    contents=batch,
                )
                all_embeddings.extend([list(emb.values) for emb in response.embeddings])
            elif self._legacy_genai is not None:
                response = self._legacy_genai.embed_content(
                    model=f"models/{self.clean_model_name}",
                    content=batch,
                    task_type="retrieval_document",
                )
                all_embeddings.extend(response["embedding"])
        return np.array(all_embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query text for retrieval."""
        if self._client is not None:
            response = self._client.models.embed_content(
                model=self.clean_model_name,
                contents=query,
            )
            return np.array(response.embeddings[0].values, dtype=np.float32)
        elif self._legacy_genai is not None:
            response = self._legacy_genai.embed_content(
                model=f"models/{self.clean_model_name}",
                content=query,
                task_type="retrieval_query",
            )
            return np.array(response["embedding"], dtype=np.float32)
        return np.empty((0,), dtype=np.float32)

