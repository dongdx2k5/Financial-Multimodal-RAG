import hashlib
import re
from pathlib import Path

import pymupdf
from bs4 import BeautifulSoup

from .models import Chunk, Document, PageContent

PAGE_MARKER = re.compile(r"^=====\s*PAGE\s+(\d+)\s*=====\s*$", re.MULTILINE | re.IGNORECASE)


def _stable_id(*parts: object) -> str:
    return hashlib.sha1("|".join(map(str, parts)).encode("utf-8")).hexdigest()[:16]


class DocumentLoader:
    def load(self, path: Path, ticker: str | None = None, year: int | None = None) -> Document:
        path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            pages = self._load_pdf(path)
        elif suffix in {".html", ".htm"}:
            pages = [
                PageContent(
                    page=1,
                    text=BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser").get_text(
                        " ", strip=True
                    ),
                )
            ]
        elif suffix in {".txt", ".md"}:
            pages = self._load_text(path)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
        return Document(
            id=_stable_id(path, path.stat().st_size),
            source=str(path),
            ticker=ticker,
            year=year,
            pages=pages,
        )

    @staticmethod
    def _load_pdf(path: Path) -> list[PageContent]:
        with pymupdf.open(path) as pdf:
            return [
                PageContent(page=i + 1, text=page.get_text("text")) for i, page in enumerate(pdf)
            ]

    @staticmethod
    def _load_text(path: Path) -> list[PageContent]:
        raw = path.read_text(encoding="utf-8", errors="replace")
        matches = list(PAGE_MARKER.finditer(raw))
        if not matches:
            return [
                PageContent(
                    page=1, text=BeautifulSoup(raw, "html.parser").get_text(" ", strip=True)
                )
            ]
        pages: list[PageContent] = []
        for index, match in enumerate(matches):
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(raw)
            text = BeautifulSoup(raw[start:end], "html.parser").get_text(" ", strip=True)
            pages.append(PageContent(page=int(match.group(1)), text=text))
        return pages


class Chunker:
    def __init__(self, size: int = 900, overlap: int = 120):
        if overlap >= size:
            raise ValueError("overlap must be smaller than size")
        self.size = size
        self.overlap = overlap

    def split(self, document: Document) -> list[Chunk]:
        chunks: list[Chunk] = []
        for page in document.pages:
            clean = re.sub(r"\s+", " ", page.text).strip()
            start = 0
            while start < len(clean):
                end = min(start + self.size, len(clean))
                if end < len(clean):
                    boundary = clean.rfind(" ", start, end)
                    if boundary > start + self.size // 2:
                        end = boundary
                text = clean[start:end].strip()
                if text:
                    chunk_id = _stable_id(document.id, page.page, start, text)
                    chunks.append(
                        Chunk(
                            id=chunk_id,
                            document_id=document.id,
                            source=document.source,
                            page=page.page,
                            text=text,
                            ticker=document.ticker,
                            year=document.year,
                        )
                    )
                if end >= len(clean):
                    break
                start = max(end - self.overlap, start + 1)
        return chunks
