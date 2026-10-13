from typing import Any, Literal

from pydantic import BaseModel, Field


class PageContent(BaseModel):
    page: int
    text: str
    is_scanned: bool = False
    images_count: int = 0


class ChartDataPoint(BaseModel):
    metric: str
    value: float
    raw_value: str
    unit: str | None = None
    period: int | None = None


class ExtractedChart(BaseModel):
    page: int
    chart_type: str = "chart"
    title: str = ""
    summary: str = ""
    markdown_table: str = ""
    data_points: list[ChartDataPoint] = Field(default_factory=list)


class Document(BaseModel):
    id: str
    source: str
    ticker: str | None = None
    year: int | None = None
    pages: list[PageContent]
    charts: list[ExtractedChart] = Field(default_factory=list)


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
    standardized_metric: str | None = None
    value: float
    raw_value: str
    unit: str | None = None
    period: int | None = None
    ticker: str | None = None
    chunk_id: str
    source: str
    page: int
    confidence: float = Field(ge=0, le=1)
    is_chart: bool = False


class Evidence(BaseModel):
    chunk_id: str
    source: str
    page: int
    text: str
    score: float
    graph_facts: list[FinancialFact] = Field(default_factory=list)


class CalculationResult(BaseModel):
    expression: str
    result: float
    formatted: str


class AskRequest(BaseModel):
    question: str = Field(min_length=3)
    top_k: int = Field(default=5, ge=1, le=20)
    ticker: str | None = None
    year: int | None = None


class AskResponse(BaseModel):
    answer: str
    evidence: list[Evidence]
    mode: Literal["evidence", "llm"]
    calculations: list[CalculationResult] = Field(default_factory=list)


class GraphExport(BaseModel):
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]


class DocumentSummary(BaseModel):
    id: str
    source: str
    filename: str
    ticker: str | None = None
    year: int | None = None
    pages_count: int = 0
    chunks_count: int = 0
    facts_count: int = 0
    created_at: str | None = None


class DatabaseStats(BaseModel):
    db_path: str
    documents: int
    chunks: int
    facts: int
    size_kb: float

