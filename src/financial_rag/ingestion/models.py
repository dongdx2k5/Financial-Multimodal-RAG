from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class Modality(str, Enum):
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"


@dataclass(frozen=True)
class DocumentElement:
    """Represents an extracted element from a document (text, table, or image)."""
    content: str  # Text body, HTML table, or base64 image string / image description
    modality: str | Modality
    source_path: Path
    page: int | None = None
    raw_content: str | None = None  # Original raw text, raw HTML table, or base64 data
    summary: str | None = None  # LLM-generated summary for vector search
    element_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    """Represents a chunk prepared for multi-vector indexing and retrieval."""
    content: str  # The search text (summary or text content)
    source_path: Path
    chunk_id: str
    page: int | None
    modality: str
    raw_content: str | None = None  # The full raw content (HTML table, base64 image, raw text)
    metadata: dict[str, Any] = field(default_factory=dict)
