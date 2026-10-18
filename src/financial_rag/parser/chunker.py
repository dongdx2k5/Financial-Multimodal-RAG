"""Structural chunker adhering to financial report section and table boundaries."""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from financial_rag.parser.pdf_parser import DocumentParseResult, ParsedTable


@dataclass
class ChunkItem:
    report_id: int
    section: Optional[str]
    chunk_type: str  # text | table
    page_start: int
    page_end: int
    context_prefix: str
    content: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "section": self.section,
            "chunk_type": self.chunk_type,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "context_prefix": self.context_prefix,
            "content": self.content,
        }


class FinancialChunker:
    """Chunks financial documents while preserving table headers and semantic context."""

    def __init__(self, target_chunk_tokens: int = 450, max_table_rows_per_chunk: int = 12):
        self.target_tokens = target_chunk_tokens
        self.max_table_rows = max_table_rows_per_chunk

    def build_context_prefix(
        self,
        ticker: str,
        period_type: str,
        fiscal_year: int,
        scope: str = "Hợp nhất",
        section: str = "BCTC",
        unit_str: str = "triệu đồng",
    ) -> str:
        """Construct standard prefix, e.g. [AAA | FY/2025 | Riêng lẻ | Bảng KQKD | Đơn vị: VND]."""
        scope_str = "Hợp nhất" if scope in ("consolidated", "Hợp nhất") else "Riêng lẻ"
        return f"[{ticker.upper()} | {period_type.upper()}/{fiscal_year} | {scope_str} | {section} | Đơn vị: {unit_str}]"

    def chunk_document(
        self,
        parse_result: DocumentParseResult,
        report_id: int,
        ticker: str,
        period_type: str,
        fiscal_year: int,
        scope: str = "consolidated",
        currency: str = "VND",
        unit_scale: int = 1,
    ) -> List[ChunkItem]:
        chunks: List[ChunkItem] = []

        # Unit string representation
        if currency.upper() == "USD":
            unit_str = "triệu USD" if unit_scale == 1_000_000 else "USD"
        else:
            if unit_scale == 1_000_000_000:
                unit_str = "tỷ đồng"
            elif unit_scale == 1_000_000:
                unit_str = "triệu đồng"
            elif unit_scale == 1_000:
                unit_str = "nghìn đồng"
            else:
                unit_str = "đồng"

        for page in parse_result.pages:
            p_num = page.page_number

            # 1. Chunk tables
            for table in page.tables:
                section_name = table.title or f"Bảng số liệu Trang {p_num}"
                table_chunks = self._chunk_table(
                    table=table,
                    report_id=report_id,
                    ticker=ticker,
                    period_type=period_type,
                    fiscal_year=fiscal_year,
                    scope=scope,
                    section=section_name,
                    unit_str=unit_str,
                )
                chunks.extend(table_chunks)

            # 2. Chunk text narrative
            if page.text.strip():
                text_chunks = self._chunk_narrative_text(
                    text=page.text,
                    page_num=p_num,
                    report_id=report_id,
                    ticker=ticker,
                    period_type=period_type,
                    fiscal_year=fiscal_year,
                    scope=scope,
                    unit_str=unit_str,
                )
                chunks.extend(text_chunks)

        return chunks

    def _chunk_table(
        self,
        table: ParsedTable,
        report_id: int,
        ticker: str,
        period_type: str,
        fiscal_year: int,
        scope: str,
        section: str,
        unit_str: str,
    ) -> List[ChunkItem]:
        chunks: List[ChunkItem] = []
        headers = table.headers
        rows = table.rows

        prefix = self.build_context_prefix(
            ticker=ticker,
            period_type=period_type,
            fiscal_year=fiscal_year,
            scope=scope,
            section=section,
            unit_str=unit_str,
        )

        if not rows:
            return []

        # If table fits within threshold, single chunk
        if len(rows) <= self.max_table_rows:
            content = f"{prefix}\n{table.to_markdown()}"
            chunks.append(ChunkItem(
                report_id=report_id,
                section=section,
                chunk_type="table",
                page_start=table.page,
                page_end=table.page,
                context_prefix=prefix,
                content=content,
            ))
            return chunks

        # Split large table into windows repeating headers
        for start_idx in range(0, len(rows), self.max_table_rows):
            sub_rows = rows[start_idx : start_idx + self.max_table_rows]
            sub_table = ParsedTable(
                page=table.page,
                table_id=f"{table.table_id}_part_{start_idx // self.max_table_rows + 1}",
                title=f"{table.title or 'Bảng'} (Phần {start_idx // self.max_table_rows + 1})",
                headers=headers,
                rows=sub_rows,
            )
            content = f"{prefix}\n{sub_table.to_markdown()}"
            chunks.append(ChunkItem(
                report_id=report_id,
                section=section,
                chunk_type="table",
                page_start=table.page,
                page_end=table.page,
                context_prefix=prefix,
                content=content,
            ))

        return chunks

    def _chunk_narrative_text(
        self,
        text: str,
        page_num: int,
        report_id: int,
        ticker: str,
        period_type: str,
        fiscal_year: int,
        scope: str,
        unit_str: str,
    ) -> List[ChunkItem]:
        chunks: List[ChunkItem] = []
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        current_tokens = 0
        current_paras: List[str] = []
        section_name = f"Thuyết minh / Nội dung Trang {page_num}"

        prefix = self.build_context_prefix(
            ticker=ticker,
            period_type=period_type,
            fiscal_year=fiscal_year,
            scope=scope,
            section=section_name,
            unit_str=unit_str,
        )

        for para in paragraphs:
            para_tokens = len(para.split())
            if current_tokens + para_tokens > self.target_tokens and current_paras:
                # Flush chunk
                content = f"{prefix}\n" + "\n\n".join(current_paras)
                chunks.append(ChunkItem(
                    report_id=report_id,
                    section=section_name,
                    chunk_type="text",
                    page_start=page_num,
                    page_end=page_num,
                    context_prefix=prefix,
                    content=content,
                ))
                current_paras = [para]
                current_tokens = para_tokens
            else:
                current_paras.append(para)
                current_tokens += para_tokens

        if current_paras:
            content = f"{prefix}\n" + "\n\n".join(current_paras)
            chunks.append(ChunkItem(
                report_id=report_id,
                section=section_name,
                chunk_type="text",
                page_start=page_num,
                page_end=page_num,
                context_prefix=prefix,
                content=content,
            ))

        return chunks
