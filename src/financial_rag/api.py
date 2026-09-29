import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated
from uuid import uuid4

import pymupdf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response

from .calculator import CalculationRequest, calculate
from .config import settings
from .models import AskRequest, AskResponse, GraphExport
from .service import FinancialRAGService

app = FastAPI(title="Multimodal RAG Financial", version="0.1.0")
service = FinancialRAGService(settings)
executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pdf-ingest")
jobs = {}


def run_ingestion(job_id, path, ticker, year, name):
    jobs[job_id] = {"status": "running", "completed_pages": 0, "total_pages": None}

    def progress(done, total):
        jobs[job_id] = {"status": "running", "completed_pages": done, "total_pages": total}

    try:
        result = service.ingest(path, ticker=ticker, year=year, name=name, progress=progress)
        jobs[job_id] = {"status": "complete", "result": result}
    except Exception as exc:
        logging.getLogger(__name__).exception("Document ingestion failed")
        jobs[job_id] = {"status": "failed", "error": str(exc)}
    finally:
        path.unlink(missing_ok=True)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "llm_provider": service.settings.llm_provider,
        "llm_model": service.settings.ollama_model
        if service.settings.llm_provider == "ollama"
        else service.settings.openai_model,
        "chunks": len(service.retriever.chunks),
        "nodes": service.graph.graph.number_of_nodes(),
        "edges": service.graph.graph.number_of_edges(),
    }


@app.post("/ingest")
def ingest(
    file: Annotated[UploadFile, File()],
    ticker: Annotated[str | None, Form()] = None,
    year: Annotated[int | None, Form()] = None,
    background: Annotated[bool, Form()] = False,
) -> dict:
    suffix = Path(file.filename or "upload.txt").suffix
    if suffix.lower() not in {".pdf", ".txt", ".html", ".htm", ".md"}:
        raise HTTPException(400, "Chỉ hỗ trợ PDF, TXT, HTML, MD")
    try:
        with NamedTemporaryFile(delete=False, suffix=suffix) as target:
            temp_path = Path(target.name)
            size = 0
            while block := file.file.read(1024 * 1024):
                size += len(block)
                if size > 50 * 1024 * 1024:
                    raise HTTPException(413, "Giới hạn tài liệu là 50 MB")
                target.write(block)
        if background:
            job_id = uuid4().hex
            jobs[job_id] = {"status": "queued"}
            executor.submit(run_ingestion, job_id, temp_path, ticker, year, file.filename)
            return {"job_id": job_id}
        return service.ingest(temp_path, ticker=ticker, year=year, name=file.filename)
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if "temp_path" in locals() and not (background and "job_id" in locals()):
            temp_path.unlink(missing_ok=True)


@app.get("/jobs/{job_id}")
def ingestion_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(
            404, "Không tìm thấy tác vụ; nếu server vừa khởi động lại hãy nạp lại tài liệu."
        )
    return jobs[job_id]


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    return service.ask(request.question, request.top_k, request.document_id, request.use_llm)


@app.get("/", response_class=HTMLResponse)
def home():
    return (Path(__file__).parent / "app.html").read_text(encoding="utf-8")


@app.get("/documents")
def documents():
    return [
        {
            "id": node,
            "name": attrs.get("name") or Path(attrs["source"]).name,
            "ticker": attrs.get("ticker"),
            "year": attrs.get("year"),
            "scope": attrs.get("scope", "unknown"),
        }
        for node, attrs in service.graph.graph.nodes(data=True)
        if attrs.get("kind") == "Report"
    ]


@app.get("/documents/{document_id}/analysis")
def report_analysis(document_id: str):
    if service.graph.graph.nodes.get(document_id, {}).get("kind") != "Report":
        raise HTTPException(404, "Không tìm thấy báo cáo")
    path = service.settings.data_dir / "processed" / "reports" / f"{document_id}.json"
    metadata = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    facts = [
        {"id": node, **attrs}
        for node, attrs in service.graph.graph.nodes(data=True)
        if attrs.get("kind") == "Fact" and attrs.get("document_id") == document_id
    ]
    return {"metadata": metadata, "facts": facts}


@app.get("/documents/{document_id}/pages/{page_number}")
def source_page(document_id: str, page_number: int, chunk_id: str | None = None):
    attrs = service.graph.graph.nodes.get(document_id, {})
    if attrs.get("kind") != "Report" or Path(attrs["source"]).suffix.lower() != ".pdf":
        raise HTTPException(404, "Không có trang PDF")
    with pymupdf.open(attrs["source"]) as pdf:
        if not 1 <= page_number <= len(pdf):
            raise HTTPException(404, "Số trang không hợp lệ")
        page = pdf[page_number - 1]
        if chunk_id:
            chunk = service.graph.graph.nodes.get(chunk_id, {})
            if chunk.get("document_id") != document_id or chunk.get("page") != page_number:
                raise HTTPException(400, "Bằng chứng không thuộc trang này")
            if chunk.get("bbox"):
                page.draw_rect(pymupdf.Rect(chunk["bbox"]), color=(1, 0.45, 0), width=1.8)
        return Response(
            page.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5)).tobytes("png"), media_type="image/png"
        )


@app.get("/documents/{document_id}/file")
def document_file(document_id: str):
    attrs = service.graph.graph.nodes.get(document_id, {})
    if attrs.get("kind") != "Report" or not Path(attrs["source"]).is_file():
        raise HTTPException(404, "Không tìm thấy tài liệu")
    return FileResponse(
        attrs["source"],
        media_type="application/pdf" if Path(attrs["source"]).suffix == ".pdf" else "text/plain",
    )


@app.post("/calculate")
def calculation(request: CalculationRequest):
    from .models import FinancialFact

    try:
        facts = []
        for key in (request.left_fact_id, request.right_fact_id):
            attrs = service.graph.graph.nodes.get(key, {})
            if attrs.get("kind") != "Fact":
                raise ValueError("Fact không tồn tại")
            facts.append(FinancialFact(id=key, **{k: v for k, v in attrs.items() if k != "kind"}))
        return calculate(request.operation, *facts)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get("/graph", response_model=GraphExport)
def graph(limit: int = 500) -> GraphExport:
    return service.graph.export(min(max(limit, 1), 5000))
