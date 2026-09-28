import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .graph import KnowledgeGraph
from .models import Chunk, Evidence


class HybridRetriever:
    def __init__(self, chunks_path: Path, graph: KnowledgeGraph):
        self.chunks_path = chunks_path
        self.graph = graph
        self.chunks: list[Chunk] = []
        self.vectorizer: TfidfVectorizer | None = None
        self.matrix = None
        self.load()

    def load(self) -> None:
        if self.chunks_path.exists():
            raw = json.loads(self.chunks_path.read_text(encoding="utf-8"))
            self.chunks = [Chunk.model_validate(item) for item in raw]
        self._fit()

    def _fit(self) -> None:
        if not self.chunks:
            self.vectorizer, self.matrix = None, None
            return
        self.vectorizer = TfidfVectorizer(
            lowercase=True, ngram_range=(1, 2), max_features=50_000, token_pattern=r"(?u)\b\w+\b"
        )
        self.matrix = self.vectorizer.fit_transform([chunk.text for chunk in self.chunks])

    def upsert(self, chunks: list[Chunk]) -> None:
        current = {chunk.id: chunk for chunk in self.chunks}
        current.update({chunk.id: chunk for chunk in chunks})
        self.chunks = list(current.values())
        self.chunks_path.parent.mkdir(parents=True, exist_ok=True)
        self.chunks_path.write_text(
            json.dumps([chunk.model_dump() for chunk in self.chunks], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._fit()

    def search(self, query: str, top_k: int = 5) -> list[Evidence]:
        if not self.chunks or self.vectorizer is None or self.matrix is None:
            return []
        query_vector = self.vectorizer.transform([query])
        scores = (self.matrix @ query_vector.T).toarray().ravel()
        indices = np.argsort(scores)[::-1][:top_k]
        return [
            Evidence(
                chunk_id=self.chunks[index].id,
                source=self.chunks[index].source,
                page=self.chunks[index].page,
                text=self.chunks[index].text,
                score=float(scores[index]),
                graph_facts=self.graph.facts_for_chunk(self.chunks[index].id),
            )
            for index in indices
            if scores[index] > 0
        ]
