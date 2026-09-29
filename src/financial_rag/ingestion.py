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
    def __init__(
        self,
        ocr_enabled=False,
        ocr_language="vie+eng",
        tessdata=None,
        cache_dir=None,
        best_tessdata=None,
    ):
        self.ocr_enabled = ocr_enabled
        self.ocr_language = ocr_language
        self.tessdata = tessdata
        self.cache_dir = cache_dir
        self.best_tessdata = best_tessdata

    def load(
        self, path: Path, ticker: str | None = None, year: int | None = None, progress=None
    ) -> Document:
        path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            pages = self._load_pdf(path, progress)
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
        from .financial_tables import fold

        title = fold(" ".join(p.text[:450] for p in pages[:12])).replace(" ", "")
        scope = (
            "separate"
            if "rieng" in title
            else ("consolidated" if "hopnhat" in title else "unknown")
        )
        return Document(
            id=_stable_id("v4", hashlib.sha256(path.read_bytes()).hexdigest(), ticker, year),
            source=str(path),
            ticker=ticker,
            year=year,
            pages=pages,
            name=path.name,
            scope=scope,
        )

    def _load_pdf(self, path: Path, progress=None) -> list[PageContent]:
        cache_root = None
        if self.cache_dir:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            cache_root = (
                Path(self.cache_dir)
                / f"{digest}-{self.ocr_language}-300-v4-{self.ocr_enabled}-{bool(self.best_tessdata)}"
            )
            cache_root.mkdir(parents=True, exist_ok=True)
        with pymupdf.open(path) as pdf:
            if pdf.needs_pass:
                raise ValueError("PDF có mật khẩu. Hãy cung cấp bản đã mở khóa.")
            pages = []
            for i, page in enumerate(pdf):
                cache_file = cache_root / f"{i + 1}.json" if cache_root else None
                if cache_file and cache_file.exists():
                    content = PageContent.model_validate_json(
                        cache_file.read_text(encoding="utf-8")
                    )
                    from .ocr_recovery import recover_codes

                    recover_codes(page, content, self.tessdata)
                    cache_file.write_text(content.model_dump_json(), encoding="utf-8")
                    pages.append(content)
                    if progress:
                        progress(i + 1, len(pdf))
                    continue
                text = page.get_text("text", sort=True)
                words = page.get_text("words")
                method = "text"
                warnings = []
                image_area = max(
                    (pymupdf.Rect(item["bbox"]).get_area() for item in page.get_image_info()),
                    default=0,
                )
                if len(text.strip()) < 30 or (
                    len(text.strip()) < 500 and image_area > page.rect.get_area() * 0.65
                ):
                    if self.ocr_enabled:
                        try:
                            tp = page.get_textpage_ocr(
                                language=self.ocr_language,
                                dpi=300,
                                full=True,
                                tessdata=self.tessdata,
                            )
                            text = page.get_text("text", textpage=tp, sort=True)
                            words = page.get_text("words", textpage=tp)
                            method = "ocr"
                        except (RuntimeError, ValueError, OSError):
                            warnings.append(
                                f"Trang {i + 1}: OCR thất bại; kiểm tra tessdata vie/eng."
                            )
                    else:
                        warnings.append(
                            f"Trang {i + 1}: ít văn bản; cần bật OCR nếu đây là trang scan."
                        )
                try:
                    tables = [table.extract() for table in page.find_tables().tables]
                except (RuntimeError, ValueError, TypeError):
                    tables = []
                    warnings.append(f"Trang {i + 1}: không trích xuất được bảng.")
                # Remove huge spacing artifacts emitted by OCR while retaining lines.
                text = re.sub(r"[^\S\n]{2,}", " ", text)
                content = PageContent(
                    page=i + 1,
                    text=text,
                    tables=tables,
                    warnings=warnings,
                    words=words,
                    width=page.rect.width,
                    height=page.rect.height,
                    extraction_method=method,
                )
                from .financial_tables import statement_kind

                if method == "ocr" and self.best_tessdata and statement_kind(text):
                    try:
                        secondary = page.get_textpage_ocr(
                            language=self.ocr_language,
                            dpi=300,
                            full=True,
                            tessdata=self.best_tessdata,
                        )
                        content.secondary_words = [
                            list(word) for word in page.get_text("words", textpage=secondary)
                        ]
                    except (RuntimeError, ValueError, OSError):
                        content.warnings.append(f"Trang {i + 1}: OCR bổ sung không thành công.")
                from .ocr_recovery import recover_codes

                recover_codes(page, content, self.tessdata)
                pages.append(content)
                if cache_file:
                    cache_file.write_text(content.model_dump_json(), encoding="utf-8")
                if progress:
                    progress(i + 1, len(pdf))
            return pages

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
        from .financial_tables import extract_statement

        chunks: list[Chunk] = []
        for page in document.pages:
            rows, warnings = extract_statement(page, document)
            chunks.extend(rows)
            page.warnings.extend(warnings)
            units = set(
                re.findall(
                    r"(?:đơn vị tính|đơn vị|unit)\s*:\s*(nghìn tỷ|tỷ|triệu|nghìn|VND|đồng)",
                    page.text,
                    re.IGNORECASE,
                )
            )
            table_unit = next(iter(units)).casefold() if len(units) == 1 else None
            for index, table in enumerate(page.tables):
                chunks.append(
                    Chunk(
                        id=_stable_id(document.id, page.page, "table", index),
                        document_id=document.id,
                        source=document.source,
                        page=page.page,
                        text="\n".join(" | ".join(cell or "" for cell in row) for row in table),
                        ticker=document.ticker,
                        year=document.year,
                        table=table,
                        table_unit=table_unit,
                    )
                )
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
                            scope=document.scope,
                            extraction_method=page.extraction_method,
                        )
                    )
                if end >= len(clean):
                    break
                start = max(end - self.overlap, start + 1)
        return chunks
