import hashlib
import logging
import re
from pathlib import Path
from typing import Any

import pymupdf
from bs4 import BeautifulSoup

from .models import ChartDataPoint, Chunk, Document, ExtractedChart, PageContent

logger = logging.getLogger(__name__)
PAGE_MARKER = re.compile(r"^=====\s*PAGE\s+(\d+)\s*=====\s*$", re.MULTILINE | re.IGNORECASE)


def _stable_id(*parts: object) -> str:
    return hashlib.sha1("|".join(map(str, parts)).encode("utf-8")).hexdigest()[:16]


class DocumentLoader:
    def __init__(
        self,
        vision_llm: Any | None = None,
        enable_charts: bool = True,
        min_chart_width: int = 50,
        min_chart_height: int = 50,
        min_chart_bytes: int = 300,
    ):
        self.vision_llm = vision_llm
        self.enable_charts = enable_charts
        self.min_chart_width = min_chart_width
        self.min_chart_height = min_chart_height
        self.min_chart_bytes = min_chart_bytes

    def load(self, path: Path, ticker: str | None = None, year: int | None = None) -> Document:
        path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        suffix = path.suffix.lower()
        charts: list[ExtractedChart] = []
        if suffix == ".pdf":
            pages, charts = self._load_pdf(path, ticker=ticker, year=year)
        elif suffix in {".html", ".htm"}:
            pages = self._load_html(path)
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
            charts=charts,
        )

    def _load_pdf(
        self,
        path: Path,
        ticker: str | None = None,
        year: int | None = None,
    ) -> tuple[list[PageContent], list[ExtractedChart]]:
        pages: list[PageContent] = []
        charts: list[ExtractedChart] = []
        with pymupdf.open(path) as pdf:
            for i, page in enumerate(pdf):
                page_text = page.get_text("text").strip()
                is_scanned = len(page_text) < 50
                ocr_success = False

                # Phase 2.1: Gemini Vision OCR for scanned PDF pages
                if is_scanned and self.vision_llm:
                    try:
                        pix = page.get_pixmap(dpi=150)
                        img_bytes = pix.tobytes("png")
                        if hasattr(self.vision_llm, "ocr_image"):
                            ocr_text = self.vision_llm.ocr_image(img_bytes)
                            if ocr_text and ocr_text.strip():
                                page_text = ocr_text.strip()
                                ocr_success = True
                    except Exception as exc:
                        logger.warning("Vision OCR failed on page %d: %s", i + 1, exc)

                table_md_list: list[str] = []

                # 1. Primary table extraction with PyMuPDF find_tables (high speed C-engine)
                try:
                    tabs = page.find_tables()
                    if tabs and tabs.tables:
                        for tab in tabs.tables:
                            md = tab.to_markdown()
                            if md and md.strip():
                                table_md_list.append(md.strip())
                except Exception as exc:
                    logger.debug("PyMuPDF table extraction notice: %s", exc)

                # 2. Fallback table extraction with pdfplumber if PyMuPDF found no tables (and not OCR'd)
                if not table_md_list and not ocr_success:
                    try:
                        import pdfplumber

                        with pdfplumber.open(path) as pl_pdf:
                            if i < len(pl_pdf.pages):
                                pl_page = pl_pdf.pages[i]
                                pl_tables = pl_page.extract_tables()
                                for tbl in pl_tables:
                                    if tbl and len(tbl) >= 2:
                                        headers = [
                                            c.replace("\n", " ").strip() if c else "" for c in tbl[0]
                                        ]
                                        if any(headers):
                                            hdr_line = "| " + " | ".join(headers) + " |"
                                            sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
                                            data_lines = [
                                                "| "
                                                + " | ".join(
                                                    c.replace("\n", " ").strip() if c else ""
                                                    for c in row
                                                )
                                                + " |"
                                                for row in tbl[1:]
                                                if any(row)
                                            ]
                                            table_md_list.append(
                                                "\n".join([hdr_line, sep_line] + data_lines)
                                            )
                    except Exception as exc:
                        logger.debug("pdfplumber table extraction notice: %s", exc)

                if table_md_list:
                    full_text = (
                        page_text
                        + "\n\n[BẢNG TÀI CHÍNH / FINANCIAL TABLES]:\n"
                        + "\n\n".join(table_md_list)
                    ).strip()
                else:
                    full_text = page_text

                # Phase 2.2: Phân tích biểu đồ (Chart Understanding)
                page_images = []
                try:
                    page_images = page.get_images(full=True)
                except Exception as exc:
                    logger.debug("PyMuPDF get_images notice: %s", exc)

                chart_md_blocks: list[str] = []
                if self.enable_charts and self.vision_llm and page_images:
                    for img_info in page_images:
                        xref = img_info[0]
                        try:
                            base_image = pdf.extract_image(xref)
                        except Exception as exc:
                            logger.debug("extract_image notice: %s", exc)
                            continue
                        img_bytes = base_image.get("image")
                        width = base_image.get("width", 0)
                        height = base_image.get("height", 0)
                        ext = base_image.get("ext", "png")

                        if (
                            not img_bytes
                            or width < self.min_chart_width
                            or height < self.min_chart_height
                            or len(img_bytes) < self.min_chart_bytes
                        ):
                            continue

                        try:
                            if hasattr(self.vision_llm, "analyze_chart"):
                                chart_data = self.vision_llm.analyze_chart(
                                    img_bytes, mime_type=f"image/{ext}"
                                )
                                if chart_data and chart_data.get("is_financial_chart"):
                                    title = chart_data.get("title") or f"Biểu đồ trang {i + 1}"
                                    summary = chart_data.get("summary") or ""
                                    table_md = chart_data.get("markdown_table") or ""
                                    chart_type = chart_data.get("chart_type") or "chart"

                                    dps: list[ChartDataPoint] = []
                                    for raw_dp in chart_data.get("data_points", []):
                                        try:
                                            dps.append(
                                                ChartDataPoint(
                                                    metric=str(raw_dp.get("metric", "")),
                                                    value=float(raw_dp.get("value", 0.0)),
                                                    raw_value=str(
                                                        raw_dp.get("raw_value", raw_dp.get("value", ""))
                                                    ),
                                                    unit=raw_dp.get("unit"),
                                                    period=(
                                                        int(raw_dp["period"])
                                                        if raw_dp.get("period")
                                                        else year
                                                    ),
                                                )
                                            )
                                        except Exception as e:
                                            logger.debug("Failed parsing chart point: %s", e)

                                    extracted_chart = ExtractedChart(
                                        page=i + 1,
                                        chart_type=chart_type,
                                        title=title,
                                        summary=summary,
                                        markdown_table=table_md,
                                        data_points=dps,
                                    )
                                    charts.append(extracted_chart)

                                    block = (
                                        f"[BIỂU ĐỒ TÀI CHÍNH / FINANCIAL CHART: {title}]:\n"
                                        f"{summary}\n\n"
                                        f"{table_md}"
                                    ).strip()
                                    chart_md_blocks.append(block)
                        except Exception as exc:
                            logger.warning("Chart analysis failed on page %d: %s", i + 1, exc)

                if chart_md_blocks:
                    full_text = (full_text + "\n\n" + "\n\n".join(chart_md_blocks)).strip()

                pages.append(
                    PageContent(
                        page=i + 1,
                        text=full_text,
                        is_scanned=is_scanned,
                        images_count=len(page_images),
                    )
                )
        return pages, charts

    @staticmethod
    def _load_html(path: Path) -> list[PageContent]:
        raw_html = path.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(raw_html, "html.parser")

        # Convert HTML <table> structures into Markdown tables
        for table in soup.find_all("table"):
            rows = table.find_all("tr")
            if not rows:
                continue
            table_md_rows: list[str] = []
            for r_idx, row in enumerate(rows):
                cells = [c.get_text(" ", strip=True) for c in row.find_all(["th", "td"])]
                if not cells or not any(cells):
                    continue
                table_md_rows.append("| " + " | ".join(cells) + " |")
                if r_idx == 0:
                    table_md_rows.append("| " + " | ".join(["---"] * len(cells)) + " |")
            if table_md_rows:
                replacement = soup.new_tag("div")
                replacement.string = "\n\n" + "\n".join(table_md_rows) + "\n\n"
                table.replace_with(replacement)

        text = soup.get_text("\n", strip=True)
        return [PageContent(page=1, text=text)]

    @staticmethod
    def _load_text(path: Path) -> list[PageContent]:
        raw = path.read_text(encoding="utf-8", errors="replace")
        matches = list(PAGE_MARKER.finditer(raw))
        if not matches:
            return [
                PageContent(
                    page=1, text=BeautifulSoup(raw, "html.parser").get_text("\n", strip=True)
                )
            ]
        pages: list[PageContent] = []
        for index, match in enumerate(matches):
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(raw)
            text = BeautifulSoup(raw[start:end], "html.parser").get_text("\n", strip=True)
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
            # Preserve line breaks for tables while normalizing horizontal spacing
            lines = [re.sub(r"[ \t]+", " ", line).strip() for line in page.text.splitlines()]
            clean = "\n".join(line for line in lines if line)
            start = 0
            while start < len(clean):
                end = min(start + self.size, len(clean))
                if end < len(clean):
                    # Prefer splitting along paragraph / table row breaks
                    newline_pos = clean.rfind("\n", start, end)
                    if newline_pos > start + self.size // 2:
                        end = newline_pos
                    else:
                        space_pos = clean.rfind(" ", start, end)
                        if space_pos > start + self.size // 2:
                            end = space_pos
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

