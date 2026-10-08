import json
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated

import pymupdf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response

from .analytics import summarize
from .calculator import CalculationRequest, calculate
from .config import settings
from .jobs import JobStore
from .models import AskRequest, AskResponse, GraphExport
from .service import FinancialRAGService

app = FastAPI(title="Multimodal RAG Financial", version="0.1.0")
service = FinancialRAGService(settings)
executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pdf-ingest")
jobs = JobStore(settings.data_dir / "processed" / "jobs.json")


def locked(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with service.lock:
            return function(*args, **kwargs)

    return wrapped


def run_ingestion(job_id, path, ticker, year, name):
    jobs.update(job_id, status="running", completed_pages=0, total_pages=None)

    def progress(done, total):
        jobs.update(job_id, status="running", completed_pages=done, total_pages=total)

    try:
        result = service.ingest(path, ticker=ticker, year=year, name=name, progress=progress)
        jobs.update(job_id, status="complete", result=result)
    except Exception as exc:
        logging.getLogger(__name__).exception("Document ingestion failed")
        jobs.update(job_id, status="failed", error=str(exc))
    finally:
        path.unlink(missing_ok=True)


@app.get("/health")
@locked
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
    submitted = False
    try:
        with NamedTemporaryFile(delete=False, suffix=suffix) as target:
            temp_path = Path(target.name)
            size = 0
            while block := file.file.read(1024 * 1024):
                size += len(block)
                if size > 50 * 1024 * 1024:
                    raise HTTPException(413, "Giới hạn tài liệu là 50 MB")
                target.write(block)
        if not size:
            raise HTTPException(400, "Tài liệu rỗng.")
        if background:
            job_id = jobs.create(Path(file.filename or "upload").name)
            try:
                executor.submit(run_ingestion, job_id, temp_path, ticker, year, file.filename)
                submitted = True
            except RuntimeError:
                jobs.update(job_id, status="failed", error="Không khởi tạo được tác vụ.")
                raise
            return {"job_id": job_id}
        return service.ingest(temp_path, ticker=ticker, year=year, name=file.filename)
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if "temp_path" in locals() and not submitted:
            temp_path.unlink(missing_ok=True)


@app.get("/jobs/{job_id}")
def ingestion_job(job_id: str):
    result = jobs.get(job_id)
    if result is None:
        raise HTTPException(
            404, "Không tìm thấy tác vụ; nếu server vừa khởi động lại hãy nạp lại tài liệu."
        )
    return result


@app.get("/jobs")
def recent_jobs():
    return jobs.recent()


@app.get("/health/ollama")
def ollama_health():
    from .ollama_client import probe

    return probe(service.settings)


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    if (
        request.document_id
        and service.graph.graph.nodes.get(request.document_id, {}).get("kind") != "Report"
    ):
        raise HTTPException(404, "Không tìm thấy báo cáo")
    return service.ask(request.question, request.top_k, request.document_id, request.use_llm)


@app.get("/", response_class=HTMLResponse)
def home():
    return (Path(__file__).parent / "app.html").read_text(encoding="utf-8")


@app.get("/assets/dashboard.js")
def dashboard_asset():
    return FileResponse(Path(__file__).parent / "dashboard.js", media_type="text/javascript")


@app.get("/documents")
@locked
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
@locked
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
    return {
        "metadata": metadata,
        "facts": facts,
        "summary": summarize(facts, metadata.get("checks", [])),
    }


@app.get("/documents/{document_id}/export")
def export_report(document_id: str):
    with service.lock:
        result = report_analysis(document_id)

    # Downloads contain provenance, not machine-specific absolute filesystem paths.
    def clean(value):
        if isinstance(value, dict):
            return {
                key: (Path(item).name if key == "source" and isinstance(item, str) else clean(item))
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [clean(item) for item in value]
        return value

    return Response(
        json.dumps(clean(result), ensure_ascii=False, indent=2),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="analysis-{document_id}.json"'},
    )


@app.get("/documents/{document_id}/graph")
def report_graph(document_id: str, limit: int = 250):
    with service.lock:
        graph = service.graph.graph
        if graph.nodes.get(document_id, {}).get("kind") != "Report":
            raise HTTPException(404, "Không tìm thấy báo cáo")
        owned = [
            node
            for node, attrs in graph.nodes(data=True)
            if attrs.get("document_id") == document_id
            or node == document_id
            or (attrs.get("kind") in {"Fact", "Chunk"} and graph.has_edge(document_id, node))
        ]
        related = set(owned)
        for node in owned:
            for neighbor in list(graph.successors(node)) + list(graph.predecessors(node)):
                if graph.nodes[neighbor].get("kind") in {"Metric", "Company", "Period"}:
                    related.add(neighbor)
        ordered = sorted(
            related, key=lambda node: (node != document_id, graph.nodes[node].get("kind", ""), node)
        )
        selected = set(ordered[: min(max(limit, 1), 1000)])
        nodes = [
            {"id": node, **{k: v for k, v in graph.nodes[node].items() if k != "source"}}
            for node in ordered
            if node in selected
        ]
        edges = [
            {"source": a, "target": b, **attrs}
            for a, b, attrs in graph.edges(data=True)
            if a in selected and b in selected
        ]
        return {
            "nodes": nodes,
            "edges": edges,
            "total_nodes": len(related),
            "truncated": len(selected) < len(related),
        }


@app.get("/documents/{document_id}/pages/{page_number}")
@locked
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
@locked
def document_file(document_id: str):
    attrs = service.graph.graph.nodes.get(document_id, {})
    if attrs.get("kind") != "Report" or not Path(attrs["source"]).is_file():
        raise HTTPException(404, "Không tìm thấy tài liệu")
    return FileResponse(
        attrs["source"],
        media_type="application/pdf" if Path(attrs["source"]).suffix == ".pdf" else "text/plain",
    )


@app.post("/calculate")
@locked
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
@locked
def graph(limit: int = 500) -> GraphExport:
    return service.graph.export(min(max(limit, 1), 5000))
