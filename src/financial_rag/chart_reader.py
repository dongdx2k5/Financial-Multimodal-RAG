"""Chart data extraction engine: reads chart areas, extracts data points, and produces structured facts.

Operates on detected chart bounding boxes:
- Crops the visual chart region
- Parses chart titles, series labels, and numeric data points
- Produces FinancialFact objects with provenance and bounding boxes
"""

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

    def __init__(self, dpi: int = 150):
        self.dpi = dpi

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

            # Look for chart title in a region slightly above the chart box
            title_rect = pymupdf.Rect(x0, max(0.0, y0 - 45), x1, y0 + 10)
            title_text = doc_page.get_text("text", clip=title_rect).strip()
            # Clean title
            lines = [line.strip() for line in title_text.splitlines() if line.strip()]
            title = lines[0] if lines else f"Biểu đồ trang {page_num}"

            # Detect unit from title or chart area
            full_chart_rect = pymupdf.Rect(x0, max(0.0, y0 - 45), x1, y1)
            raw_text = doc_page.get_text("text", clip=full_chart_rect)

            unit = None
            if "tỷ" in raw_text.lower() or "tỷ vnd" in raw_text.lower():
                unit = "tỷ VND"
            elif "triệu" in raw_text.lower():
                unit = "triệu VND"
            elif "%" in raw_text:
                unit = "%"
            elif "vnd" in raw_text.lower() or "đồng" in raw_text.lower():
                unit = "VND"

            # Extract words inside the chart box
            words = doc_page.get_text("words", clip=clip_rect)
            # Find years (e.g. 2022, 2023, 2024)
            year_tokens = []
            number_tokens = []

            for w in words:
                token = w[4].strip()
                if re.fullmatch(r"20\d{2}", token):
                    year_tokens.append((int(token), (w[0] + w[2]) / 2, w[1]))
                elif re.fullmatch(r"[-+]?(?:\d{1,3}(?:[.\s]\d{3})+(?:,\d+)?|\d+(?:[.,]\d+)?)", token):
                    try:
                        val = parse_vietnamese_number(token)
                        number_tokens.append((val, token, (w[0] + w[2]) / 2, w[1]))
                    except ValueError:
                        pass

            year_tokens.sort(key=lambda item: item[1])  # Sort left-to-right by x coordinate

            data_points: list[ChartDataPoint] = []
            facts: list[FinancialFact] = []

            chunk_id = hashlib.sha1(f"{document.id}|chart|{page_num}|{chart_idx}".encode()).hexdigest()[:16]

            # Pair numbers with nearest years by horizontal (x) proximity
            for year, yx, _ in year_tokens:
                # Find numbers close in x
                nearby_numbers = [
                    (num_val, num_str, abs(nx - yx))
                    for num_val, num_str, nx, ny in number_tokens
                    if abs(nx - yx) < (clip_rect.width / max(len(year_tokens), 1)) * 0.8
                    and int(num_val) != year
                ]
                if nearby_numbers:
                    nearby_numbers.sort(key=lambda item: item[2])
                    val, raw_val, _ = nearby_numbers[0]
                    dp = ChartDataPoint(
                        series=title,
                        value=val,
                        raw_value=raw_val,
                        unit=unit,
                        period=year,
                        period_label=str(year),
                    )
                    data_points.append(dp)

                    fact_id = hashlib.sha1(f"{chunk_id}|{year}|{raw_val}".encode()).hexdigest()[:16]
                    facts.append(
                        FinancialFact(
                            id=fact_id,
                            document_id=document.id,
                            chunk_id=chunk_id,
                            source=document.source,
                            page=page_num,
                            metric=title,
                            value=val,
                            raw_value=raw_val,
                            unit=unit,
                            period=year,
                            period_label=str(year),
                            period_kind="annual",
                            scope=document.scope,
                            statement="chart",
                            bbox=bbox,
                            confidence=0.75,
                            extraction_method="chart",
                            quality="consistent" if len(data_points) >= 2 else "unverified",
                        )
                    )

            chart_chunk = Chunk(
                id=chunk_id,
                document_id=document.id,
                source=document.source,
                page=page_num,
                text=f"Biểu đồ: {title}. " + " | ".join(f"Năm {dp.period}: {dp.raw_value} {dp.unit or ''}" for dp in data_points),
                ticker=document.ticker,
                year=document.year,
                scope=document.scope,
                bbox=bbox,
                extraction_method="chart",
            )

            results.append(
                ChartExtractionResult(
                    title=title,
                    chart_type="bar_or_line",
                    page=page_num,
                    bbox=bbox,
                    data_points=data_points,
                    facts=facts,
                    chunk=chart_chunk,
                )
            )

        return results
