import time
from typing import Any

import chromadb
from google import genai

from financial_rag.config.settings import Settings
from financial_rag.ingestion.models import Chunk
from financial_rag.retrieval.doc_store import DocStore


class FinancialVectorStore:
    """Multi-Vector Store: Indexes summaries in ChromaDB and links to raw elements in DocStore."""

    def __init__(self, settings: Settings, doc_store: DocStore | None = None):
        self.settings = settings
        self.client = chromadb.PersistentClient(path=str(settings.chroma_dir))
        self.collection = self.client.get_or_create_collection(settings.collection_name)
        self.doc_store = doc_store or DocStore(storage_path=settings.data_dir / "processed" / "docstore.json")

        api_key = settings.api_key
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY chưa được cấu hình! Vui lòng điền GEMINI_API_KEY vào file '.env'."
            )
        self.genai_client = genai.Client(api_key=api_key)

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
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            success = False
            for attempt in range(7):
                try:
                    response = self.genai_client.models.embed_content(
                        model=self.settings.embedding_model,
                        contents=batch,
                    )
                    all_embeddings.extend([item.values for item in response.embeddings])
                    success = True
                    break
                except Exception as e:
                    err_msg = str(e)
                    if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                        time.sleep(6 * (attempt + 1))
                    else:
                        raise e
            if not success:
                raise RuntimeError(f"Failed to embed batch starting at index {i} after 7 retries.")
            time.sleep(0.3)
        return all_embeddings
