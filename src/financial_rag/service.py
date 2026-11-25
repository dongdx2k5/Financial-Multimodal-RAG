import json
import shutil
from pathlib import Path
from threading import RLock

from .calculator import calculate
from .config import Settings
from .extraction import FinancialFactExtractor
from .graph import KnowledgeGraph
from .ingestion import Chunker, DocumentLoader, _sha256_file
from .models import AskResponse, FinancialFact
from .retrieval import HybridRetriever
from .sqlite_store import FinancialSQLiteStore
from .storage import atomic_json


class FinancialRAGService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.lock = RLock()
        self._ingest_lock = RLock()
        self.store = FinancialSQLiteStore(settings.index_dir / "financial.db")
        self.graph = KnowledgeGraph(settings.index_dir / "graph.json")
        self.retriever = HybridRetriever(
            settings.index_dir / "chunks.json",
            self.graph,
            store=self.store,
            colab_settings=settings,
        )
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
            layout_backend=settings.layout_backend,
            layout_model_path=settings.layout_model_path,
            chart_extraction_enabled=settings.chart_extraction_enabled,
            colab_settings=settings,
        )
        self.chunker = Chunker(settings.chunk_size, settings.chunk_overlap)
        self.extractor = FinancialFactExtractor()

    def ingest(
        self, path, ticker: str | None = None, year: int | None = None, name=None, progress=None
    ) -> dict:
        with self._ingest_lock:
            return self._ingest(Path(path), ticker, year, name, progress)

    def _ingest(self, path: Path, ticker, year, name=None, progress=None):
        original_name = name or path.name
        digest = _sha256_file(path)
        destination = self.settings.data_dir / "uploads" / (digest + path.suffix.lower())
        destination.parent.mkdir(parents=True, exist_ok=True)
        if path.resolve() != destination.resolve():
            shutil.copyfile(path, destination)
        path = destination

        # Heavy PDF/OCR/layout extraction runs without holding self.lock so API reads stay responsive.
        document = self.loader.load(path, ticker=ticker, year=year, progress=progress)
        document.name = original_name
        chunks = self.chunker.split(document)
        warnings = [w for page in document.pages for w in page.warnings]
        if not chunks:
            raise ValueError("Không đọc được văn bản. PDF scan cần OCR. " + " ".join(warnings))
        facts = [fact for chunk in chunks for fact in self.extractor.extract(chunk)]
        from .quality import check_facts

        checks = check_facts(facts)
        embedding_rows: list[tuple[str, str, list[float]]] = []
        if self.settings.colab_embeddings_enabled and self.settings.colab_gpu_url and chunks:
            from .colab_client import ColabGPUError, embed_texts

            try:
                for start in range(0, len(chunks), 32):
                    batch = chunks[start : start + 32]
                    vectors = embed_texts(self.settings, [c.text for c in batch])
                    for chunk, vec in zip(batch, vectors, strict=True):
                        embedding_rows.append((chunk.id, document.id, vec))
            except ColabGPUError as exc:
                warnings.append(f"Chưa tạo được embedding trên Colab GPU ({exc}); dùng chỉ mục cục bộ.")

        metadata = {
            **document.model_dump(exclude={"pages"}),
            "checks": checks,
            "warnings": warnings,
            "ocr_pages": sum(p.extraction_method == "ocr" for p in document.pages),
            "pages": len(document.pages),
            "statement_rows": sum(bool(c.financial_rows) for c in chunks),
            "ocr_engines": sorted({p.ocr_engine for p in document.pages if p.ocr_engine}),
            "structured_tables": sum(len(p.structured_tables) for p in document.pages),
            "charts": sum(len(p.chart_results) for p in document.pages),
            "embeddings": len(embedding_rows),
        }
        meta_path = self.settings.data_dir / "processed" / "reports" / f"{document.id}.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)

        with self.lock:
            atomic_json(meta_path, metadata)
            self.store.save_document(document, metadata=metadata)
            self.graph.add_document(document, chunks, facts)
            self.retriever.upsert(chunks)
            self.store.save_facts(facts)
            if embedding_rows:
                self.retriever.upsert_embeddings(embedding_rows)

        return {
            "document_id": document.id,
            "pages": len(document.pages),
            "chunks": len(chunks),
            "facts": len(facts),
            "tables": sum(len(page.tables) for page in document.pages),
            "charts": metadata["charts"],
            "warnings": warnings,
            "source_url": f"/documents/{document.id}/file",
            "ocr_pages": metadata["ocr_pages"],
            "statement_rows": metadata["statement_rows"],
            "checks": checks,
            "scope": document.scope,
            "ocr_engines": metadata["ocr_engines"],
            "structured_tables": metadata["structured_tables"],
        }

    def delete_document(self, document_id: str) -> bool:
        with self.lock:
            attrs = self.graph.graph.nodes.get(document_id, {})
            if attrs.get("kind") != "Report" and not self.store.get_document_metadata(document_id):
                return False
            deleted = self.store.delete_document(document_id)
            source_path = Path(deleted["source"]) if deleted else (
                Path(attrs["source"]) if attrs.get("source") else None
            )
            self.graph.remove_document(document_id)
            self.retriever.remove_document(document_id)
            meta_path = self.settings.data_dir / "processed" / "reports" / f"{document_id}.json"
            meta_path.unlink(missing_ok=True)
            if (
                source_path
                and (not deleted or not deleted.get("source_shared"))
                and source_path.is_file()
            ):
                uploads_dir = (self.settings.data_dir / "uploads").resolve()
                if source_path.resolve().is_relative_to(uploads_dir):
                    source_path.unlink(missing_ok=True)
            return True

    def verify_fact(
        self,
        fact_id: str,
        *,
        value: float | None = None,
        raw_value: str | None = None,
        quality: str | None = None,
        recognition_note: str | None = None,
    ) -> dict | None:
        with self.lock:
            effective_raw = raw_value if raw_value is not None else (
                str(value) if value is not None else None
            )
            updated = self.store.update_fact(
                fact_id,
                value=value,
                raw_value=effective_raw,
                quality=quality,
                recognition_note=recognition_note,
            )
            if fact_id in self.graph.graph.nodes:
                attrs = self.graph.graph.nodes[fact_id]
                if value is not None:
                    attrs["value"] = float(value)
                if effective_raw is not None:
                    attrs["raw_value"] = effective_raw
                if quality is not None:
                    attrs["quality"] = quality
                if recognition_note is not None:
                    attrs["recognition_note"] = recognition_note
                if updated is None:
                    updated = FinancialFact(
                        id=fact_id, **{k: v for k, v in attrs.items() if k != "kind"}
                    )
                self.graph.save()
            if updated is None:
                return None
            doc_id = updated.document_id
            checks = []
            if doc_id:
                from .quality import check_facts

                doc_facts = self.store.load_facts(doc_id)
                if not doc_facts:
                    doc_facts = [
                        FinancialFact(id=n, **{k: v for k, v in a.items() if k != "kind"})
                        for n, a in self.graph.graph.nodes(data=True)
                        if a.get("kind") == "Fact" and a.get("document_id") == doc_id
                    ]
                prior_verified = {f.id for f in doc_facts if f.quality == "verified"}
                checks = check_facts(doc_facts)
                for item in doc_facts:
                    if item.id in prior_verified and item.quality != "needs_review":
                        item.quality = "verified"
                    if item.id == updated.id:
                        updated.quality = item.quality
                    if item.id in self.graph.graph.nodes:
                        self.graph.graph.nodes[item.id]["quality"] = item.quality
                    self.store.update_fact(item.id, quality=item.quality)
                self.graph.save()
                self.store.update_document_checks(doc_id, checks)
                meta_path = self.settings.data_dir / "processed" / "reports" / f"{doc_id}.json"
                if meta_path.exists():
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    meta["checks"] = checks
                    atomic_json(meta_path, meta)
            return {"fact": updated.model_dump(), "checks": checks}

    def compare_documents(
        self,
        left_document_id: str,
        right_document_id: str,
        statement: str,
        metric_code: str,
        operation: str = "difference",
        period_label: str | None = None,
    ) -> dict:
        with self.lock:
            def pick_fact(doc_id: str) -> FinancialFact:
                facts = self.store.get_facts_by_code(doc_id, statement, metric_code)
                if not facts:
                    facts = [
                        FinancialFact(id=n, **{k: v for k, v in a.items() if k != "kind"})
                        for n, a in self.graph.graph.nodes(data=True)
                        if a.get("kind") == "Fact"
                        and a.get("document_id") == doc_id
                        and a.get("statement") == statement
                        and a.get("metric_code") == metric_code
                    ]
                if period_label:
                    facts = [f for f in facts if f.period_label == period_label]
                if not facts:
                    raise ValueError(f"Thiếu chỉ tiêu {statement}.{metric_code} ở báo cáo {doc_id}.")
                facts.sort(key=lambda f: f.period_label or "", reverse=True)
                target_period = facts[0].period_label
                same_period = [f for f in facts if f.period_label == target_period]
                if len(same_period) != 1:
                    raise ValueError(f"Có nhiều số liệu trùng kỳ tại báo cáo {doc_id}.")
                return same_period[0]

            left = pick_fact(left_document_id)
            right = pick_fact(right_document_id)
            if operation in {"growth", "difference"} and left.period_label == right.period_label:
                # Cross-report comparison at the same period computes difference safely via ratio/diff.
                from decimal import Decimal

                a, b = Decimal(str(left.value)), Decimal(str(right.value))
                if not left.unit or left.unit != right.unit:
                    raise ValueError("Hai báo cáo khác đơn vị; cần đối chiếu đơn vị trước.")
                if "needs_review" in {left.quality, right.quality}:
                    raise ValueError("Có số liệu chưa khớp kiểm tra kế toán.")
                if operation == "difference":
                    return {
                        "operation": "difference",
                        "value": str(a - b),
                        "formula": f"{a} - {b}",
                        "unit": left.unit,
                        "facts": [left.model_dump(), right.model_dump()],
                    }
                if b == 0:
                    raise ValueError("Không thể chia cho 0.")
                if b < 0:
                    raise ValueError("Nền âm: không tự diễn giải phần trăm chênh lệch.")
                return {
                    "operation": "growth",
                    "value": str((a - b) / b * 100),
                    "formula": f"({a} - {b}) / {b} * 100",
                    "unit": "%",
                    "facts": [left.model_dump(), right.model_dump()],
                }
            return calculate(operation, left, right, allow_cross_document=True)

    def ask(
        self,
        question: str,
        top_k: int | None = None,
        document_id: str | None = None,
        use_llm: bool = False,
    ) -> AskResponse:
        warnings: list[str] = []
        with self.lock:
            from .answers import INTENTS, answer_from_intent_payload, factual_answer
            from .financial_tables import fold

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
                if use_llm:
                    exact.warnings.append(
                        "Đã bỏ qua use_llm: câu hỏi số liệu được giải bằng quy tắc, phép tính và trích dẫn cục bộ."
                    )
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
            if use_llm and not self.settings.colab_gpu_url:
                warnings.append(
                    "Đã bỏ qua use_llm: chưa cấu hình COLAB_GPU_URL, chỉ trả kết quả cục bộ."
                )
            return AskResponse(
                answer="Chưa tìm thấy bằng chứng phù hợp trong chỉ mục. Hãy nạp báo cáo trước.",
                evidence=[],
                mode="evidence",
                answer_status="insufficient_evidence",
                reason_code="no_matching_evidence",
                warnings=warnings,
            )

        if use_llm and self.settings.colab_gpu_url:
            from . import colab_client

            qualitative = any(
                term in fold(question)
                for term in [
                    "tai sao",
                    "vi sao",
                    "nguyen nhan",
                    "rui ro",
                    "trien vong",
                    "phan tich",
                    "chinh sach",
                    "ghi nhan",
                    "khi nao",
                    "dieu kien",
                    "tom tat",
                    "quy dinh",
                ]
            )
            if not qualitative and document_id:
                try:
                    catalog = [
                        {"phrase": p, "statement": s, "metric_code": c} for p, s, c in INTENTS
                    ]
                    parsed_intent = colab_client.parse_intent(self.settings, question, catalog)
                    with self.lock:
                        resolved = answer_from_intent_payload(self, document_id, parsed_intent)
                    if resolved and resolved.mode in {"structured", "calculation"}:
                        resolved.answer_status = "answered"
                        resolved.reason_code = "colab_intent_deterministic_calc"
                        resolved.steps = [
                            "Phân tích ý định câu hỏi qua Colab GPU API (JSON có cấu trúc).",
                            "Đối chiếu dữ kiện gốc và tính toán tất định bằng Decimal cục bộ.",
                        ]
                        return resolved
                except colab_client.ColabGPUError as exc:
                    warnings.append(f"Colab intent parser không khả dụng: {exc}")

            evidence_text = "\n\n".join(
                f"[{i}] Trang {item.page}: {item.text}" for i, item in enumerate(evidence, 1)
            )
            try:
                synth_answer, model_name, elapsed = colab_client.synthesize(
                    self.settings, question, evidence_text
                )
                validation_err = colab_client.validate_output(synth_answer, evidence, question)
                if validation_err is None:
                    return AskResponse(
                        answer=synth_answer,
                        evidence=evidence,
                        mode="llm",
                        warnings=warnings,
                        llm_model=model_name,
                        llm_elapsed_seconds=elapsed,
                        answer_status="answered",
                        reason_code="colab_grounded_synthesis",
                        steps=[
                            "Truy xuất bằng chứng bằng TF-IDF + SQLite FTS5 BM25 + Knowledge Graph.",
                            "Tổng hợp giải thích qua Colab GPU API kèm trích dẫn [1]..[n].",
                            "Kiểm tra hậu kiểm số liệu và trích dẫn cục bộ (validate_output).",
                        ],
                    )
                warnings.append(validation_err)
            except colab_client.ColabGPUError as exc:
                warnings.append(f"Colab GPU không khả dụng ({exc}); trả về bằng chứng gốc.")
        elif use_llm:
            warnings.append(
                "Đã bỏ qua use_llm: chưa cấu hình COLAB_GPU_URL, hệ thống chỉ dùng quy tắc và bằng chứng cục bộ."
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
                "Xếp hạng bằng TF-IDF, SQLite FTS5 BM25 và quy tắc cộng điểm từ graph.",
                "Trích lại nội dung chunk; không tự suy luận hoặc sinh thêm câu trả lời.",
                "Điểm tìm kiếm không phải xác suất câu trả lời đúng.",
            ],
        )
