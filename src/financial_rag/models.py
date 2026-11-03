from typing import Any, Literal

from pydantic import BaseModel, Field


class OCRSpan(BaseModel):
    """A detected line, not an invented per-word bounding box."""

    text: str
    bbox: list[float] = Field(min_length=4, max_length=4)
    confidence: float | None = Field(default=None, ge=0, le=1)


class TableCell(BaseModel):
    row: int = Field(ge=0)
    col: int = Field(ge=0)
    row_span: int = Field(default=1, ge=1)
    col_span: int = Field(default=1, ge=1)
    text: str
    bbox: list[float] = Field(min_length=4, max_length=4)
    confidence: float | None = Field(default=None, ge=0, le=1)


class StructuredTable(BaseModel):
    bbox: list[float] = Field(min_length=4, max_length=4)
    cells: list[TableCell] = Field(default_factory=list)
    engine: str = "slanet_plus"
    region_source: str = "unknown"
    warnings: list[str] = Field(default_factory=list)


class PageContent(BaseModel):
    page: int
    text: str
    tables: list[list[list[str | None]]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    words: list[list[Any]] = Field(default_factory=list)
    width: float = 0
    height: float = 0
    extraction_method: str = "text"
    secondary_words: list[list[Any]] = Field(default_factory=list)
    recovered_codes: list[dict[str, Any]] = Field(default_factory=list)
    code_recovery_version: int = 0
    ocr_engine: str | None = None
    ocr_lines: list[OCRSpan] = Field(default_factory=list)
    structured_tables: list[StructuredTable] = Field(default_factory=list)
    table_bboxes: list[list[float]] = Field(default_factory=list)


class Document(BaseModel):
    id: str
    source: str
    ticker: str | None = None
    year: int | None = None
    pages: list[PageContent]
    name: str = ""
    scope: str = "unknown"
    reporting_period: str | None = None


class Chunk(BaseModel):
    id: str
    document_id: str
    source: str
    page: int
    text: str
    ticker: str | None = None
    year: int | None = None
    table: list[list[str | None]] | None = None
    table_unit: str | None = None
    scope: str = "unknown"
    bbox: list[float] | None = None
    extraction_method: str = "text"
    financial_rows: list[dict[str, Any]] = Field(default_factory=list)
    table_structure_engine: str | None = None
    table_cells: list[TableCell] = Field(default_factory=list)
    ocr_engine: str | None = None


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
    document_id: str | None = None
    period_label: str | None = None
    period_kind: str | None = None
    scope: str = "unknown"
    metric_code: str | None = None
    statement: str | None = None
    bbox: list[float] | None = None
    extraction_method: str = "text"
    quality: str = "unverified"
    recognition_note: str | None = None


class Evidence(BaseModel):
    chunk_id: str
    source: str
    page: int
    text: str
    score: float
    graph_facts: list[FinancialFact] = Field(default_factory=list)
    document_id: str | None = None
    source_url: str | None = None
    bbox: list[float] | None = None
    extraction_method: str = "text"


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    document_id: str | None = None
    use_llm: bool = False


class AskResponse(BaseModel):
    answer: str
    evidence: list[Evidence]
    mode: Literal["evidence", "llm", "calculation", "structured"]
    warnings: list[str] = Field(default_factory=list)
    llm_model: str | None = None
    llm_elapsed_seconds: float | None = None
    answer_status: Literal[
        "answered", "evidence_only", "needs_clarification", "insufficient_evidence"
    ] = "evidence_only"
    reason_code: str = ""
    steps: list[str] = Field(default_factory=list)


class GraphExport(BaseModel):
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
