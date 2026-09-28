from .config import Settings
from .extraction import FinancialFactExtractor
from .graph import KnowledgeGraph
from .ingestion import Chunker, DocumentLoader
from .models import AskResponse
from .retrieval import HybridRetriever


class FinancialRAGService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.graph = KnowledgeGraph(settings.index_dir / "graph.json")
        self.retriever = HybridRetriever(settings.index_dir / "chunks.json", self.graph)
        self.loader = DocumentLoader()
        self.chunker = Chunker(settings.chunk_size, settings.chunk_overlap)
        self.extractor = FinancialFactExtractor()

    def ingest(self, path, ticker: str | None = None, year: int | None = None) -> dict:
        document = self.loader.load(path, ticker=ticker, year=year)
        chunks = self.chunker.split(document)
        facts = [fact for chunk in chunks for fact in self.extractor.extract(chunk)]
        self.graph.add_document(document, chunks, facts)
        self.retriever.upsert(chunks)
        return {
            "document_id": document.id,
            "pages": len(document.pages),
            "chunks": len(chunks),
            "facts": len(facts),
        }

    def ask(self, question: str, top_k: int | None = None) -> AskResponse:
        evidence = self.retriever.search(question, top_k or self.settings.retrieval_top_k)
        if not evidence:
            return AskResponse(
                answer="Chưa tìm thấy bằng chứng phù hợp trong chỉ mục. Hãy nạp báo cáo trước.",
                evidence=[],
                mode="evidence",
            )
        answer = "Các bằng chứng liên quan nhất:\n" + "\n".join(
            f"- {item.text[:300]} [nguồn: {item.source}, trang {item.page}]" for item in evidence
        )
        return AskResponse(answer=answer, evidence=evidence, mode="evidence")
