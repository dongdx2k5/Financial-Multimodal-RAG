from typing import Any, Literal

from pydantic import BaseModel, Field


class PageContent(BaseModel):
    page: int
    text: str


class Document(BaseModel):
    id: str
    source: str
    ticker: str | None = None
    year: int | None = None
    pages: list[PageContent]


class Chunk(BaseModel):
    id: str
    document_id: str
    source: str
    page: int
    text: str
    ticker: str | None = None
    year: int | None = None


class FinancialFact(BaseModel):
    id: str
    metric: str
    value: float
    raw_value: str
    unit: str | None = None
    period: int | None = None
    ticker: str | None = None
    chunk_id: str
    source: str
    page: int
    confidence: float = Field(ge=0, le=1)


class Evidence(BaseModel):
    chunk_id: str
    source: str
    page: int
    text: str
    score: float
    graph_facts: list[FinancialFact] = Field(default_factory=list)


class AskRequest(BaseModel):
    question: str = Field(min_length=3)
    top_k: int = Field(default=5, ge=1, le=20)


class AskResponse(BaseModel):
    answer: str
    evidence: list[Evidence]
    mode: Literal["evidence", "llm"]


class GraphExport(BaseModel):
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
