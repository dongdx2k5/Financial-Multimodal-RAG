"""
FastAPI Server cho Unified Financial Multimodal RAG Pipeline
Chạy bằng lệnh: uvicorn src.financial_rag.api_server:app --host 0.0.0.0 --port 8000
"""
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from .pipeline import UnifiedFinancialRAGPipeline

app = FastAPI(title="Unified Financial Multimodal RAG API", version="2.0.0")
pipeline = UnifiedFinancialRAGPipeline(db_path="financial_rag_unified.sqlite")


class QueryRequest(BaseModel):
    query: str
    company: str = "FPT"
    period: str = "FY2024"
    comparison_period: Optional[str] = None


@app.get("/health")
def health_check():
    return {"status": "ok", "backbone": pipeline.vlm_backbone.model_id}


@app.post("/api/v1/ask")
def ask_financial_question(req: QueryRequest):
    return pipeline.answer_question(
        query=req.query,
        company=req.company,
        period=req.period,
        comparison_period=req.comparison_period,
    )
