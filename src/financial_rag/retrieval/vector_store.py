from typing import Any

import chromadb

from financial_rag.config.settings import Settings
from financial_rag.ingestion.models import Chunk
from financial_rag.reasoning.llm_client import UnifiedLLMClient
from financial_rag.retrieval.doc_store import DocStore


class FinancialVectorStore:
    """Multi-Vector Store: Indexes summaries in ChromaDB and links to raw elements in DocStore."""

    def __init__(
        self,
        settings: Settings,
        doc_store: DocStore | None = None,
        llm_client: UnifiedLLMClient | None = None,
    ):
        self.settings = settings
        self.client = chromadb.PersistentClient(path=str(settings.chroma_dir))
        self.collection = self.client.get_or_create_collection(settings.collection_name)
        self.doc_store = doc_store or DocStore(storage_path=settings.data_dir / "processed" / "docstore.json")
        try:
            self.llm_client = llm_client or UnifiedLLMClient(settings)
        except Exception:
            self.llm_client = None

    @property
    def genai_client(self) -> Any:
        return self.llm_client.genai_client if self.llm_client else None

    @genai_client.setter
    def genai_client(self, value: Any) -> None:
        if self.llm_client:
            self.llm_client.genai_client = value

    def add_chunks(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0

        # Filter chunks with non-empty content
        valid_chunks = [c for c in chunks if c.content and c.content.strip()]
        if not valid_chunks:
            return 0

        # 1. Embed summaries
        embeddings = self._embed([c.content for c in valid_chunks])

        # 2. Build Chroma metadata and DocStore items
        chroma_metadatas = []
        doc_store_items = {}

        for chunk in valid_chunks:
            meta = {
                "doc_id": chunk.chunk_id,
                "source": str(chunk.source_path),
                "page": chunk.page or 0,
                "modality": chunk.modality,
            }
            for key, value in chunk.metadata.items():
                clean_key = f"meta_{key}"
                if value is None:
                    meta[clean_key] = ""
                elif isinstance(value, (str, int, float, bool)):
                    meta[clean_key] = value
                else:
                    meta[clean_key] = str(value)
            chroma_metadatas.append(meta)

            # Store full raw content in DocStore
            doc_store_items[chunk.chunk_id] = {
                "doc_id": chunk.chunk_id,
                "raw_content": chunk.raw_content or chunk.content,
                "summary": chunk.content,
                "modality": chunk.modality,
                "source": str(chunk.source_path),
                "page": chunk.page,
                "metadata": chunk.metadata,
            }

        # Save to DocStore
        self.doc_store.set_batch(doc_store_items)

        # Upsert to Chroma
        self.collection.upsert(
            ids=[c.chunk_id for c in valid_chunks],
            documents=[c.content for c in valid_chunks],
            embeddings=embeddings,
            metadatas=chroma_metadatas,
        )
        return len(valid_chunks)

    def search(self, query: str, limit: int = 6) -> list[dict[str, Any]]:
        query_embedding = self._embed([query])[0]
        result = self.collection.query(query_embeddings=[query_embedding], n_results=limit)

        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        ids = result.get("ids", [[]])[0]

        retrieved = []
        for doc_id, summary_text, meta, dist in zip(ids, documents, metadatas, distances):
            raw_element = self.doc_store.get(doc_id)
            raw_content = raw_element.get("raw_content") if raw_element else summary_text
            modality = meta.get("modality", "text") if meta else "text"

            retrieved.append({
                "doc_id": doc_id,
                "content": summary_text,  # Summary for reference
                "raw_content": raw_content,  # Original raw HTML / text / base64 image
                "modality": modality,
                "metadata": meta or {},
                "distance": dist,
            })
        return retrieved

    def _embed(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        if not texts:
            return []
        if self.llm_client:
            return self.llm_client.embed_texts(texts, batch_size=batch_size)
        raise ValueError(
            "Chưa cấu hình API key để tạo embeddings! Vui lòng điền GEMINI_API_KEY hoặc LLAMA_EMBEDDING_MODEL vào .env."
        )
