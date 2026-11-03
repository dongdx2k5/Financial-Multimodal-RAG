import hashlib
import json
import re
from pathlib import Path

import pymupdf
from bs4 import BeautifulSoup

from .models import Chunk, Document, OCRSpan, PageContent, StructuredTable

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
        *,
        ocr_backend="tesseract",
        fallback_to_tesseract=True,
        vietocr_weights=None,
        detector_path=None,
        table_backend="none",
        table_model_path=None,
    ):
        self.ocr_enabled = ocr_enabled
        self.ocr_language = ocr_language
        self.tessdata = tessdata
        self.cache_dir = cache_dir
        self.best_tessdata = best_tessdata
        if ocr_backend not in {"tesseract", "vietocr"}:
            raise ValueError("Unsupported OCR backend")
        if table_backend not in {"none", "slanet_plus"}:
            raise ValueError("Unsupported table backend")
        self.ocr_backend = ocr_backend
        self.fallback_to_tesseract = fallback_to_tesseract
        self.vietocr_weights = vietocr_weights
        self.detector_path = detector_path
        self.table_backend = table_backend
        self.table_model_path = table_model_path
        self._vietocr = None
        self._table_model = None

    def _cache_signature(self):
        def identity(path):
            if not path:
                return None
            p = Path(path).resolve()
            if not p.exists():
                return [str(p), "missing"]
            stat = p.stat()
            return [str(p), stat.st_size, stat.st_mtime_ns]

        config = [
            "v5-ocr-tables",
            self.ocr_enabled,
            self.ocr_language,
            self.ocr_backend,
            self.fallback_to_tesseract,
            identity(self.tessdata),
            identity(self.best_tessdata),
            identity(self.vietocr_weights),
            identity(self.detector_path),
            self.table_backend,
            identity(self.table_model_path),
        ]
        return hashlib.sha256(json.dumps(config).encode()).hexdigest()[:20]

    def _tesseract(self, page):
        tp = page.get_textpage_ocr(
            language=self.ocr_language, dpi=300, full=True, tessdata=self.tessdata
        )
        return (
            page.get_text("text", textpage=tp, sort=True),
            page.get_text("words", textpage=tp),
        )

    def _extract_page(self, page) -> tuple[PageContent, bool]:
        """Return content and whether it is safe to cache (no transient fallback)."""
        text = page.get_text("text", sort=True)
        words = page.get_text("words")
        method, engine = "text", None
        spans, warnings = [], []
        cacheable = True
        image_area = max(
            (pymupdf.Rect(item["bbox"]).get_area() for item in page.get_image_info()),
            default=0,
        )
        needs_ocr = len(text.strip()) < 30 or (
            len(text.strip()) < 500 and image_area > page.rect.get_area() * 0.65
        )
        if needs_ocr and self.ocr_enabled:
            try:
                if self.ocr_backend == "vietocr":
                    from .ocr_backends import VietOCRBackend

                    if self._vietocr is None:
                        self._vietocr = VietOCRBackend(
                            weights_path=self.vietocr_weights,
                            detector_path=self.detector_path,
                            device="cpu",
                        )
                    result = self._vietocr.extract(page, dpi=300)
                    spans = [
                        OCRSpan(text=s.text, bbox=list(s.bbox), confidence=s.confidence)
                        for s in result.lines
                    ]
                    if not any(s.text.strip() for s in spans):
                        raise RuntimeError("VietOCR không nhận được dòng chữ nào")
                    text = "\n".join(s.text for s in spans)
                    # Preserve whole-line boxes; never invent positions of individual words.
                    words = [[*s.bbox, s.text, 0, n, 0] for n, s in enumerate(spans)]
                    warnings.extend(result.warnings)
                    engine = "vietocr"
                else:
                    text, words = self._tesseract(page)
                    engine = "tesseract"
                method = "ocr"
            except (ImportError, RuntimeError, ValueError, OSError) as exc:
                cacheable = False
                warnings.append(f"Trang {page.number + 1}: {self.ocr_backend} thất bại: {exc}")
                if self.ocr_backend == "vietocr" and not self.fallback_to_tesseract:
                    raise RuntimeError(warnings[-1]) from exc
                if self.ocr_backend == "vietocr" and self.fallback_to_tesseract:
                    try:
                        text, words = self._tesseract(page)
                        method, engine = "ocr", "tesseract"
                        spans = []
                        warnings.append(f"Trang {page.number + 1}: đã dùng Tesseract dự phòng.")
                    except (RuntimeError, ValueError, OSError) as fallback_exc:
                        warnings.append(f"Tesseract dự phòng thất bại: {fallback_exc}")
        elif needs_ocr:
            warnings.append(
                f"Trang {page.number + 1}: ít văn bản; cần bật OCR nếu đây là trang scan."
            )
        tables, table_bboxes = [], []
        try:
            for table in page.find_tables().tables:
                tables.append(table.extract())
                table_bboxes.append(list(table.bbox))
        except (RuntimeError, ValueError, TypeError):
            warnings.append(f"Trang {page.number + 1}: không trích xuất được bảng gốc.")
        content = PageContent(
            page=page.number + 1,
            text=re.sub(r"[^\S\n]{2,}", " ", text),
            tables=tables,
            table_bboxes=table_bboxes,
            warnings=warnings,
            words=words,
            width=page.cropbox.width,
            height=page.cropbox.height,
            extraction_method=method,
            ocr_engine=engine,
            ocr_lines=spans,
        )
        from .financial_tables import statement_kind

        if engine == "tesseract" and self.best_tessdata and statement_kind(text):
            try:
                secondary = page.get_textpage_ocr(
                    language=self.ocr_language, dpi=300, full=True, tessdata=self.best_tessdata
                )
                content.secondary_words = [
                    list(word) for word in page.get_text("words", textpage=secondary)
                ]
            except (RuntimeError, ValueError, OSError):
                cacheable = False
                content.warnings.append(f"Trang {content.page}: OCR bổ sung không thành công.")
        if engine == "tesseract":
            from .ocr_recovery import recover_codes

            recover_codes(page, content, self.tessdata)
        if self.table_backend == "slanet_plus":
            try:
                from .table_structure import SLANetTableBackend

                if self._table_model is None:
                    self._table_model = SLANetTableBackend(model_path=self.table_model_path)
                lines = spans or [OCRSpan(text=w[4], bbox=list(w[:4])) for w in words]
                found = self._table_model.extract_tables(page, lines, regions=table_bboxes or None)
                content.structured_tables = [StructuredTable.model_validate(t) for t in found]
                for table in content.structured_tables:
                    content.warnings.extend(table.warnings)
            except (ImportError, RuntimeError, ValueError, OSError) as exc:
                cacheable = False
                content.warnings.append(f"Trang {content.page}: nhận cấu trúc bảng thất bại: {exc}")
        return content, cacheable

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
            cache_root = Path(self.cache_dir) / f"{digest}-{self._cache_signature()}"
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
                    pages.append(content)
                    if progress:
                        progress(i + 1, len(pdf))
                    continue
                content, cacheable = self._extract_page(page)
                pages.append(content)
                if cache_file and cacheable:
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
            if page.ocr_engine == "vietocr":
                # Legacy column parsing expects word boxes. A detected line can
                # span multiple numeric columns; never concatenate those values.
                rows, warnings = [], [
                    "VietOCR thử nghiệm: chỉ truy xuất bằng chứng; chưa tạo dữ kiện để tính toán."
                ]
            else:
                rows, warnings = extract_statement(page, document)
            for row in rows:
                row.ocr_engine = page.ocr_engine
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
                        bbox=page.table_bboxes[index] if index < len(page.table_bboxes) else None,
                        scope=document.scope,
                        extraction_method=page.extraction_method,
                        ocr_engine=page.ocr_engine,
                    )
                )
            for index, table in enumerate(page.structured_tables):
                # Keep row/column/span provenance explicit, without guessing financial semantics.
                text = "\n".join(
                    f"[hàng {c.row + 1}, cột {c.col + 1}; ô gộp {c.row_span}x{c.col_span}] {c.text}"
                    for c in table.cells
                    if c.text.strip()
                )
                if text:
                    chunks.append(
                        Chunk(
                            id=_stable_id(document.id, page.page, "structure", index),
                            document_id=document.id,
                            source=document.source,
                            page=page.page,
                            text=text,
                            ticker=document.ticker,
                            year=document.year,
                            scope=document.scope,
                            bbox=table.bbox,
                            extraction_method=page.extraction_method,
                            ocr_engine=page.ocr_engine,
                            table_structure_engine=table.engine,
                            table_cells=table.cells,
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
                            ocr_engine=page.ocr_engine,
                        )
                    )
                if end >= len(clean):
                    break
                start = max(end - self.overlap, start + 1)
        return chunks
