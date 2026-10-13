import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from .config import settings
from .models import AskRequest, AskResponse, DocumentSummary, FinancialFact, GraphExport
from .service import FinancialRAGService

app = FastAPI(title="Multimodal RAG Financial", version="0.1.0")
service = FinancialRAGService(settings)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", **service.get_stats()}


@app.get("/documents", response_model=list[DocumentSummary])
def list_documents() -> list[DocumentSummary]:
    """List all stored financial reports in the database."""
    return [DocumentSummary.model_validate(doc) for doc in service.list_documents()]


@app.get("/documents/{document_id}")
def get_document(document_id: str) -> dict:
    """Get details of a stored financial report."""
    doc = service.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@app.delete("/documents/{document_id}")
def delete_document(document_id: str) -> dict:
    """Delete a financial report and its chunks/facts from the database and graph."""
    success = service.delete_document(document_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"message": "Document deleted successfully", "document_id": document_id}


@app.get("/facts", response_model=list[FinancialFact])
def get_facts(
    ticker: str | None = None,
    year: int | None = None,
    metric: str | None = None,
    is_chart: bool | None = None,
    limit: int = 100,
) -> list[FinancialFact]:
    """Query structured financial facts from the database."""
    return service.get_facts(
        ticker=ticker, year=year, metric=metric, is_chart=is_chart, limit=limit
    )


@app.post("/ingest")
def ingest(
    file: Annotated[UploadFile, File()],
    ticker: Annotated[str | None, Form()] = None,
    year: Annotated[int | None, Form()] = None,
) -> dict:
    suffix = Path(file.filename or "upload.txt").suffix
    try:
        with NamedTemporaryFile(delete=False, suffix=suffix) as target:
            shutil.copyfileobj(file.file, target)
            temp_path = Path(target.name)
        result = service.ingest(temp_path, ticker=ticker, year=year)
        if file.filename:
            result["filename"] = file.filename
        return result
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if "temp_path" in locals():
            temp_path.unlink(missing_ok=True)


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    return service.ask(
        request.question,
        top_k=request.top_k,
        ticker=request.ticker,
        year=request.year,
    )


@app.get("/graph", response_model=GraphExport)
def graph(limit: int = 500) -> GraphExport:
    return service.graph.export(min(max(limit, 1), 5000))

