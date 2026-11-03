import hashlib
import shutil
from pathlib import Path
from threading import RLock

from .config import Settings
from .extraction import FinancialFactExtractor
from .graph import KnowledgeGraph
from .ingestion import Chunker, DocumentLoader
from .models import AskResponse
from .retrieval import HybridRetriever
from .storage import atomic_json


class FinancialRAGService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.lock = RLock()
        self.graph = KnowledgeGraph(settings.index_dir / "graph.json")
        self.retriever = HybridRetriever(settings.index_dir / "chunks.json", self.graph)
        self.loader = DocumentLoader(
            settings.ocr_enabled,
            settings.ocr_language,
            settings.tessdata_prefix,
            settings.data_dir / "processed" / "ocr",
            settings.tessdata_best_prefix,
            ocr_backend=settings.ocr_backend,
            fallback_to_tesseract=settings.ocr_fallback_to_tesseract,
            vietocr_weights=settings.vietocr_weights,
            detector_path=settings.ocr_detector_path,
            table_backend=settings.table_backend,
            table_model_path=settings.table_model_path,
        )
        self.chunker = Chunker(settings.chunk_size, settings.chunk_overlap)
        self.extractor = FinancialFactExtractor()

    def ingest(
        self, path, ticker: str | None = None, year: int | None = None, name=None, progress=None
    ) -> dict:
        with self.lock:
            return self._ingest(Path(path), ticker, year, name, progress)

    def _ingest(self, path, ticker, year, name=None, progress=None):
        original_name = name or path.name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        destination = self.settings.data_dir / "uploads" / (digest + path.suffix.lower())
        destination.parent.mkdir(parents=True, exist_ok=True)
        if path.resolve() != destination.resolve():
            shutil.copyfile(path, destination)
        path = destination
        document = self.loader.load(path, ticker=ticker, year=year, progress=progress)
        document.name = original_name
        chunks = self.chunker.split(document)
        warnings = [w for page in document.pages for w in page.warnings]
        if not chunks:
            raise ValueError("Không đọc được văn bản. PDF scan cần OCR. " + " ".join(warnings))
        facts = [fact for chunk in chunks for fact in self.extractor.extract(chunk)]
        from .quality import check_facts

        checks = check_facts(facts)
        metadata = {
            **document.model_dump(exclude={"pages"}),
            "checks": checks,
            "warnings": warnings,
            "ocr_pages": sum(p.extraction_method == "ocr" for p in document.pages),
            "pages": len(document.pages),
            "statement_rows": sum(bool(c.financial_rows) for c in chunks),
            "ocr_engines": sorted({p.ocr_engine for p in document.pages if p.ocr_engine}),
            "structured_tables": sum(len(p.structured_tables) for p in document.pages),
        }
        meta_path = self.settings.data_dir / "processed" / "reports" / f"{document.id}.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(meta_path, metadata)
        self.graph.add_document(document, chunks, facts)
        self.retriever.upsert(chunks)
        return {
            "document_id": document.id,
            "pages": len(document.pages),
            "chunks": len(chunks),
            "facts": len(facts),
            "tables": sum(len(page.tables) for page in document.pages),
            "warnings": warnings,
            "source_url": f"/documents/{document.id}/file",
            "ocr_pages": metadata["ocr_pages"],
            "statement_rows": metadata["statement_rows"],
            "checks": checks,
            "scope": document.scope,
            "ocr_engines": metadata["ocr_engines"],
            "structured_tables": metadata["structured_tables"],
        }

    def ask(
        self,
        question: str,
        top_k: int | None = None,
        document_id: str | None = None,
        use_llm: bool = False,
    ) -> AskResponse:
        warnings = (
            ["Đã bỏ qua use_llm: hỏi đáp chỉ dùng quy tắc, phép tính và trích dẫn; không gọi LLM."]
            if use_llm
            else []
        )
        with self.lock:
            from .answers import factual_answer

            reports = {chunk.document_id for chunk in self.retriever.chunks}
            if not document_id and len(reports) == 1:
                document_id = next(iter(reports))
            if not document_id and len(reports) > 1:
                return AskResponse(
                    answer="Hãy chọn một báo cáo để tránh trộn dữ liệu từ nhiều tài liệu.",
                    evidence=[],
                    mode="evidence",
                    answer_status="needs_clarification",
                    reason_code="document_required",
                    warnings=warnings,
                )
            if document_id and document_id not in reports:
                return AskResponse(
                    answer="Không có nội dung đã lập chỉ mục cho báo cáo được chọn.",
                    evidence=[],
                    mode="evidence",
                    answer_status="insufficient_evidence",
                    reason_code="document_not_indexed",
                    warnings=warnings,
                )
            exact = factual_answer(self, question, document_id)
            if exact:
                exact.warnings.extend(warnings)
                if exact.mode in {"structured", "calculation"}:
                    exact.answer_status = "answered"
                    exact.reason_code = (
                        "supported_calculation" if exact.mode == "calculation" else "supported_fact"
                    )
                elif exact.answer_status == "evidence_only":
                    exact.answer_status = "insufficient_evidence"
                    exact.reason_code = exact.reason_code or "fact_validation_failed"
                exact.steps = [
                    "Giới hạn dữ kiện trong báo cáo: " + str(document_id),
                    "Nhận diện chỉ tiêu và yêu cầu bằng bộ luật trong answers.py.",
                    "Kiểm tra kỳ, phạm vi, đơn vị, dữ kiện trùng và chất lượng trích xuất.",
                    "Tính bằng calculator.py và điền mẫu có nguồn."
                    if exact.mode == "calculation"
                    else "Điền mẫu từ dữ kiện có nguồn."
                    if exact.mode == "structured"
                    else "Dừng trả lời vì chưa vượt qua điều kiện kiểm tra.",
                ]
                return exact
            evidence = self.retriever.search(
                question, top_k or self.settings.retrieval_top_k, document_id
            )
        if not evidence:
            return AskResponse(
                answer="Chưa tìm thấy bằng chứng phù hợp trong chỉ mục. Hãy nạp báo cáo trước.",
                evidence=[],
                mode="evidence",
                answer_status="insufficient_evidence",
                reason_code="no_matching_evidence",
                warnings=warnings,
            )
        answer = (
            "Chưa có quy tắc và dữ kiện đủ để trả lời chính xác toàn bộ câu hỏi này. "
            "Các đoạn trích dưới đây chỉ là kết quả tìm kiếm liên quan, "
            "không phải kết luận đã được xác minh:\n\n"
        ) + "\n\n".join(
            f"[{i}] Trang {item.page}: {item.text}" for i, item in enumerate(evidence, 1)
        )
        if any(item.extraction_method == "ocr" for item in evidence):
            warnings.append("Đoạn trích lấy từ OCR; hãy đối chiếu chữ và số với trang PDF gốc.")
        return AskResponse(
            answer=answer,
            evidence=evidence,
            mode="evidence",
            warnings=warnings,
            answer_status="evidence_only",
            reason_code="unsupported_question",
            steps=[
                "Chỉ tìm trong báo cáo: " + str(document_id),
                "Xếp hạng bằng TF-IDF và quy tắc cộng điểm từ graph.",
                "Trích lại nội dung chunk; không tự suy luận hoặc sinh thêm câu trả lời.",
                "Điểm tìm kiếm không phải xác suất câu trả lời đúng.",
            ],
        )
