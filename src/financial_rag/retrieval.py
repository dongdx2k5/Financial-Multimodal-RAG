import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .financial_tables import fold
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
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            max_features=50_000,
            token_pattern=r"(?u)\b\w+\b",
        )
        self.matrix = self.vectorizer.fit_transform([chunk.text for chunk in self.chunks])

    def upsert(self, chunks: list[Chunk]) -> None:
        reports = {chunk.document_id for chunk in chunks}
        current = {chunk.id: chunk for chunk in self.chunks if chunk.document_id not in reports}
        current.update({chunk.id: chunk for chunk in chunks})
        self.chunks = list(current.values())
        self.chunks_path.parent.mkdir(parents=True, exist_ok=True)
        self.chunks_path.write_text(
            json.dumps([chunk.model_dump() for chunk in self.chunks], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._fit()

    def search(self, query: str, top_k: int = 5, document_id: str | None = None) -> list[Evidence]:
        if not self.chunks or self.vectorizer is None or self.matrix is None:
            return []
        from .answers import expand_query

        query = expand_query(query)
        for alias, full in {
            "dong tien kinh doanh": "luu chuyen tien thuan tu hoat dong kinh doanh",
            "dong tien dau tu": "luu chuyen tien thuan tu hoat dong dau tu",
            "dong tien tai chinh": "luu chuyen tien thuan tu hoat dong tai chinh",
        }.items():
            query = query.replace(alias, full)
        query_vector = self.vectorizer.transform([query])
        scores = (self.matrix @ query_vector.T).toarray().ravel()
        policy_query = any(term in query for term in ["ghi nhan", "chinh sach", "dieu kien"])
        # Traverse matching Metric <- MEASURES - Fact -> SUPPORTED_BY -> Chunk edges.
        graph = self.graph.graph
        positions = {chunk.id: i for i, chunk in enumerate(self.chunks)}
        query_words = set(fold(query).split())
        for node, attrs in graph.nodes(data=True):
            if policy_query:
                break  # Policies need explanatory text, not metric-node boosts.
            if attrs.get("kind") != "Metric":
                continue
            words = set(fold(attrs["name"]).split())
            match = len(words & query_words) / max(len(words), 1)
            if match < 0.6:
                continue
            for fact in graph.predecessors(node):
                for target in graph.successors(fact):
                    if target in positions:
                        scores[positions[target]] += 0.35 * match
        if document_id:
            for i, chunk in enumerate(self.chunks):
                if chunk.document_id != document_id:
                    scores[i] = 0
        for i, chunk in enumerate(self.chunks):
            if policy_query and chunk.financial_rows:
                scores[i] *= 0.1
            compact = fold(chunk.text).replace(" ", "")
            if "noidungtrang" in compact and "thuyetminh" in compact:
                scores[i] = 0  # A table of contents is not evidence of financial performance.
        indices = np.argsort(scores)[::-1][:top_k]
        return [
            Evidence(
                chunk_id=self.chunks[index].id,
                source=self.chunks[index].source,
                page=self.chunks[index].page,
                text=self.chunks[index].text,
                score=float(scores[index]),
                graph_facts=self.graph.facts_for_chunk(self.chunks[index].id),
                document_id=self.chunks[index].document_id,
                source_url=f"/documents/{self.chunks[index].document_id}/file#page={self.chunks[index].page}",
                bbox=self.chunks[index].bbox,
                extraction_method=self.chunks[index].extraction_method,
            )
            for index in indices
            if scores[index] > 0
        ]
