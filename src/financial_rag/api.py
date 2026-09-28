import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from .config import settings
from .models import AskRequest, AskResponse, GraphExport
from .service import FinancialRAGService

app = FastAPI(title="Multimodal RAG Financial", version="0.1.0")
service = FinancialRAGService(settings)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "chunks": len(service.retriever.chunks),
        "nodes": service.graph.graph.number_of_nodes(),
        "edges": service.graph.graph.number_of_edges(),
    }


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
        return service.ingest(temp_path, ticker=ticker, year=year)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if "temp_path" in locals():
            temp_path.unlink(missing_ok=True)


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    return service.ask(request.question, request.top_k)


@app.get("/graph", response_model=GraphExport)
def graph(limit: int = 500) -> GraphExport:
    return service.graph.export(min(max(limit, 1), 5000))
