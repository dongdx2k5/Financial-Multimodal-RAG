"""Chart data extraction engine: reads chart areas, extracts data points, and produces structured facts.

Supports:
- Remote VLM parsing on Google Colab GPU (``colab_client.parse_chart_remote``) for raster/multi-series charts.
- Local PyMuPDF vector-text heuristic fallback for vector PDF charts.

All extracted chart facts are explicitly marked ``quality="unverified"`` because chart labels are
not verified against Circular 200 accounting identities.
"""

from __future__ import annotations

import hashlib
import re

import pymupdf
from pydantic import BaseModel, Field

from .extraction import parse_vietnamese_number
from .models import Chunk, Document, FinancialFact


class ChartDataPoint(BaseModel):
    series: str
    value: float
    raw_value: str
    unit: str | None = None
    period: int | None = None
    period_label: str | None = None


class ChartExtractionResult(BaseModel):
    title: str
    chart_type: str = "bar_or_line"
    page: int
    bbox: list[float] = Field(min_length=4, max_length=4)
    data_points: list[ChartDataPoint] = Field(default_factory=list)
    facts: list[FinancialFact] = Field(default_factory=list)
    chunk: Chunk | None = None


class ChartReader:
    """Extracts structured financial facts from chart regions with coordinate bounds."""

    def __init__(self, dpi: int = 150, colab_settings=None):
        self.dpi = dpi
        self.colab_settings = colab_settings

    def extract_from_page(
        self,
        doc_page: pymupdf.Page,
        page_num: int,
        chart_boxes: list[list[float]],
        document: Document,
    ) -> list[ChartExtractionResult]:
        results: list[ChartExtractionResult] = []
        page_rect = doc_page.rect

        for chart_idx, bbox in enumerate(chart_boxes):
            x0, y0, x1, y1 = bbox
            clip_rect = pymupdf.Rect(x0, y0, x1, y1) & page_rect
            if clip_rect.is_empty or clip_rect.width < 50 or clip_rect.height < 50:
                continue

            title_rect = pymupdf.Rect(x0, max(0.0, y0 - 45), x1, y0 + 10) & page_rect
            title_text = doc_page.get_text("text", clip=title_rect).strip()
            lines = [line.strip() for line in title_text.splitlines() if line.strip()]
            title = lines[0] if lines else f"Biểu đồ trang {page_num}"

            full_chart_rect = pymupdf.Rect(x0, max(0.0, y0 - 45), x1, y1) & page_rect
            raw_text = doc_page.get_text("text", clip=full_chart_rect)

            unit = self._detect_unit(raw_text)
            chunk_id = hashlib.sha1(
                f"{document.id}|chart|{page_num}|{chart_idx}".encode()
            ).hexdigest()[:16]

            data_points: list[ChartDataPoint] = []
            chart_type = "bar_or_line"

            # 1. Optional remote Colab VLM chart parser when COLAB_GPU_URL is configured
            if self.colab_settings and getattr(self.colab_settings, "colab_gpu_url", None):
                try:
                    from .colab_client import ColabGPUError, parse_chart_remote

                    scale = self.dpi / 72.0
                    pix = doc_page.get_pixmap(
                        matrix=pymupdf.Matrix(scale, scale),
                        clip=full_chart_rect,
                        colorspace=pymupdf.csRGB,
                        alpha=False,
                    )
                    parsed = parse_chart_remote(self.colab_settings, pix.tobytes("png"), title=title)
                    chart_type = parsed.get("chart_type") or chart_type
                    unit = parsed.get("unit") or unit
                    title = parsed.get("title") or title
                    for series in parsed.get("series", []):
                        series_name = str(series.get("name") or title).strip()
                        for pt in series.get("points", []):
                            raw_val = str(pt.get("raw_value") or pt.get("value") or "").strip()
                            if not raw_val:
                                continue
                            val = float(pt["value"]) if pt.get("value") is not None else parse_vietnamese_number(raw_val)
                            label = str(pt.get("label") or "").strip() or None
                            year_match = re.search(r"(20\d{2})", label or "")
                            period = int(year_match.group(1)) if year_match else document.year
                            data_points.append(
                                ChartDataPoint(
                                    series=series_name,
                                    value=val,
                                    raw_value=raw_val,
                                    unit=unit,
                                    period=period,
                                    period_label=label,
                                )
                            )
                except (ColabGPUError, KeyError, ValueError, TypeError):
                    data_points = []

            # 2. Local vector-text pairing fallback
            if not data_points:
                data_points = self._extract_vector_points(doc_page, clip_rect, title, unit)

            facts: list[FinancialFact] = []
            for pt_idx, dp in enumerate(data_points):
                fact_id = hashlib.sha1(
                    f"{chunk_id}|{dp.series}|{dp.period_label}|{dp.raw_value}|{pt_idx}".encode()
                ).hexdigest()[:16]
                facts.append(
                    FinancialFact(
                        id=fact_id,
                        document_id=document.id,
                        chunk_id=chunk_id,
                        source=document.source,
                        ticker=document.ticker,
                        page=page_num,
                        metric=dp.series,
                        value=dp.value,
                        raw_value=dp.raw_value,
                        unit=dp.unit,
                        period=dp.period,
                        period_label=dp.period_label,
                        period_kind="year" if dp.period_label and dp.period_label.isdigit() else None,
                        scope=document.scope,
                        statement="chart",
                        bbox=bbox,
                        confidence=0.70,
                        extraction_method="chart",
                        quality="unverified",
                    )
                )

            summary_parts = [
                f"{dp.series} ({dp.period_label or '?'}): {dp.raw_value} {dp.unit or ''}".strip()
                for dp in data_points
            ]
            chunk_text = f"Biểu đồ: {title}."
            if summary_parts:
                chunk_text += " " + " | ".join(summary_parts)

            chart_chunk = Chunk(
                id=chunk_id,
                document_id=document.id,
                source=document.source,
                page=page_num,
                text=chunk_text,
                ticker=document.ticker,
                year=document.year,
                scope=document.scope,
                bbox=bbox,
                extraction_method="chart",
            )

            results.append(
                ChartExtractionResult(
                    title=title,
                    chart_type=chart_type,
                    page=page_num,
                    bbox=bbox,
                    data_points=data_points,
                    facts=facts,
                    chunk=chart_chunk,
                )
            )

        return results

    @staticmethod
    def _detect_unit(raw_text: str) -> str | None:
        lower = raw_text.lower()
        if "nghìn tỷ" in lower:
            return "nghìn tỷ VND"
        if "tỷ" in lower:
            return "tỷ VND"
        if "triệu" in lower:
            return "triệu VND"
        if "%" in raw_text:
            return "%"
        if "vnd" in lower or "đồng" in lower:
            return "VND"
        return None

    @staticmethod
    def _extract_vector_points(
        doc_page: pymupdf.Page,
        clip_rect: pymupdf.Rect,
        title: str,
        unit: str | None,
    ) -> list[ChartDataPoint]:
        words = doc_page.get_text("words", clip=clip_rect)
        period_tokens: list[tuple[int | None, str, float, float]] = []
        number_tokens: list[tuple[float, str, float, float]] = []

        for w in words:
            token = w[4].strip()
            cx = (w[0] + w[2]) / 2
            cy = (w[1] + w[3]) / 2
            if re.fullmatch(r"20\d{2}", token):
                period_tokens.append((int(token), token, cx, cy))
            elif re.fullmatch(r"Q[1-4]/20\d{2}", token, re.IGNORECASE):
                year = int(token.split("/")[1])
                period_tokens.append((year, token.upper(), cx, cy))
            elif re.fullmatch(r"[-+]?(?:\d{1,3}(?:[.\s]\d{3})+(?:,\d+)?|\d+(?:[.,]\d+)?)", token):
                try:
                    val = parse_vietnamese_number(token)
                    number_tokens.append((val, token, cx, cy))
                except ValueError:
                    pass

        period_tokens.sort(key=lambda item: item[2])
        if not period_tokens:
            return []
        x_window = (clip_rect.width / max(len(period_tokens), 1)) * 0.45
        used: set[int] = set()
        points: list[ChartDataPoint] = []
        for year, label, px, py in period_tokens:
            candidates = []
            for idx, (val, raw_str, nx, ny) in enumerate(number_tokens):
                if idx in used:
                    continue
                # Skip tokens sitting directly on the x-axis line whose text equals the axis year.
                if abs(ny - py) <= 5 and str(int(val)) == label:
                    continue
                if abs(nx - px) <= x_window:
                    candidates.append((idx, val, raw_str, abs(nx - px), ny))
            candidates.sort(key=lambda item: (item[3], item[4]))
            if candidates:
                idx, val, raw_str, _, _ = candidates[0]
                used.add(idx)
                points.append(
                    ChartDataPoint(
                        series=title,
                        value=val,
                        raw_value=raw_str,
                        unit=unit,
                        period=year,
                        period_label=label,
                    )
                )
        return points
