from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, UploadFile
from pydantic import BaseModel

from financial_rag.config.settings import get_settings
from financial_rag.pipeline import ingest, query

app = FastAPI(title="Financial Multimodal RAG API", version="0.2.0")


class AskRequest(BaseModel):
    question: str
    limit: int = 6


class AskResponse(BaseModel):
    answer: str
    contexts: list[dict[str, Any]]
    retrieved_images: list[dict[str, Any]]
    retrieved_tables: list[dict[str, Any]]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/upload")
async def upload_report(
    file: UploadFile = File(...),
    summarize: bool = Form(True),
) -> dict:
    settings = get_settings()
    raw_dir = settings.data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    safe_filename = Path(file.filename or "upload.pdf").name
    target = raw_dir / safe_filename
    target.write_bytes(await file.read())
    indexed = ingest(target, summarize_multimodal=summarize)
    return {"filename": safe_filename, "indexed_chunks": indexed}


@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest) -> dict[str, Any]:
    return query(request.question, limit=request.limit)


@app.post("/ingest")
def ingest_existing(path: str = "data/raw", summarize: bool = True) -> dict:
    indexed = ingest(Path(path), summarize_multimodal=summarize)
    return {"indexed_chunks": indexed}
