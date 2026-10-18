"""PDF and document parser for Vietnamese and English financial reports."""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import pymupdf as fitz
except ImportError:
    try:
        import fitz
    except ImportError:
        fitz = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None


logger = logging.getLogger(__name__)


@dataclass
class ParsedTable:
    page: int
    table_id: str
    title: Optional[str] = None
    headers: List[str] = field(default_factory=list)
    rows: List[List[str]] = field(default_factory=list)

    def to_markdown(self) -> str:
        """Convert parsed table to Markdown format."""
        if not self.headers and not self.rows:
            return ""
        lines = []
        if self.title:
            lines.append(f"**{self.title}**\n")

        headers = self.headers if self.headers else [f"Col {i+1}" for i in range(len(self.rows[0]))]
        clean_headers = [str(h).replace("\n", " ").strip() for h in headers]
        lines.append("| " + " | ".join(clean_headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(clean_headers)) + " |")

        for row in self.rows:
            clean_row = [str(c).replace("\n", " ").strip() for c in row]
            # Pad row if fewer columns than header
            if len(clean_row) < len(clean_headers):
                clean_row.extend([""] * (len(clean_headers) - len(clean_row)))
            lines.append("| " + " | ".join(clean_row[:len(clean_headers)]) + " |")

        return "\n".join(lines)


@dataclass
class ParsedPage:
    page_number: int
    text: str
    tables: List[ParsedTable] = field(default_factory=list)


@dataclass
class DocumentParseResult:
    file_path: str
    pages: List[ParsedPage]
    is_scanned: bool = False
    log_messages: List[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        """Generate unified markdown document with page markers."""
        output = []
        for p in self.pages:
            output.append(f"\n<!-- PAGE {p.page_number} -->\n")
            if p.text.strip():
                output.append(p.text.strip() + "\n")
            for t in p.tables:
                output.append("\n" + t.to_markdown() + "\n")
        return "\n".join(output)

    def to_tables_dict(self) -> List[Dict[str, Any]]:
        tables = []
        for p in self.pages:
            for t in p.tables:
                tables.append({
                    "page": t.page,
                    "table_id": t.table_id,
                    "title": t.title,
                    "headers": t.headers,
                    "rows": t.rows,
                })
        return tables


class FinancialPDFParser:
    """Parser extracting text and tabular structures from PDF or pre-extracted text files."""

    def parse(self, file_path: Union[str, Path]) -> DocumentParseResult:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        ext = path.suffix.lower()
        if ext in (".txt", ".text", ".md"):
            return self._parse_text_file(path)
        elif ext == ".pdf":
            return self._parse_pdf_file(path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def _parse_text_file(self, path: Path) -> DocumentParseResult:
        """Parse pre-extracted text files with page markers (e.g., ===== PAGE X =====)."""
        content = path.read_text(encoding="utf-8", errors="replace")
        logs = [f"Parsing text file: {path.name}"]

        # Split by page patterns: "===== PAGE X =====" or "<!-- PAGE X -->"
        page_pattern = re.compile(r"(?:={3,}\s*PAGE\s*(\d+)\s*={3,}|<!--\s*PAGE\s*(\d+)\s*-->)", re.IGNORECASE)
        splits = page_pattern.split(content)

        pages: List[ParsedPage] = []
        if len(splits) > 1:
            # First element is pre-page content
            i = 1
            while i < len(splits):
                p_num_str = splits[i] or splits[i+1]
                p_num = int(p_num_str) if p_num_str and p_num_str.isdigit() else len(pages) + 1
                page_text = splits[i+2] if (i+2) < len(splits) else ""
                
                # Check for markdown tables inside page text
                extracted_tables, remaining_text = self._extract_markdown_tables(page_text, p_num)
                pages.append(ParsedPage(page_number=p_num, text=remaining_text, tables=extracted_tables))
                i += 3
        else:
            # Single page or no page markers
            extracted_tables, remaining_text = self._extract_markdown_tables(content, 1)
            pages.append(ParsedPage(page_number=1, text=remaining_text, tables=extracted_tables))

        logs.append(f"Successfully extracted {len(pages)} pages.")
        return DocumentParseResult(file_path=str(path), pages=pages, log_messages=logs)

    def _parse_pdf_file(self, path: Path) -> DocumentParseResult:
        """Extract text and tables from PDF using pdfplumber and/or PyMuPDF."""
        logs = [f"Parsing PDF document: {path.name}"]
        pages: List[ParsedPage] = []
        is_scanned = False

        if pdfplumber is not None:
            try:
                with pdfplumber.open(path) as pdf:
                    logs.append(f"pdfplumber opened {len(pdf.pages)} pages.")
                    total_chars = 0
                    for idx, page in enumerate(pdf.pages, start=1):
                        text = page.extract_text() or ""
                        total_chars += len(text.strip())

                        # Extract tables
                        tables_data = page.extract_tables() or []
                        parsed_tables: List[ParsedTable] = []
                        for t_idx, raw_table in enumerate(tables_data):
                            if not raw_table or len(raw_table) < 2:
                                continue
                            # First non-empty row as header
                            headers = [str(c or "").strip() for c in raw_table[0]]
                            rows = [[str(c or "").strip() for c in r] for r in raw_table[1:]]
                            parsed_tables.append(ParsedTable(
                                page=idx,
                                table_id=f"page_{idx}_table_{t_idx+1}",
                                headers=headers,
                                rows=rows,
                            ))

                        pages.append(ParsedPage(page_number=idx, text=text, tables=parsed_tables))

                    if total_chars < 100 * len(pdf.pages):
                        is_scanned = True
                        logs.append("Warning: Low text volume detected, document may be scanned.")
                    return DocumentParseResult(file_path=str(path), pages=pages, is_scanned=is_scanned, log_messages=logs)
            except Exception as e:
                logs.append(f"pdfplumber parsing encountered error: {e}. Falling back to PyMuPDF.")

        # Fallback to PyMuPDF
        if fitz is not None:
            try:
                doc = fitz.open(path)
                logs.append(f"PyMuPDF opened {len(doc)} pages.")
                for idx, page in enumerate(doc, start=1):
                    text = page.get_text() or ""
                    pages.append(ParsedPage(page_number=idx, text=text, tables=[]))
                return DocumentParseResult(file_path=str(path), pages=pages, is_scanned=is_scanned, log_messages=logs)
            except Exception as e:
                logs.append(f"PyMuPDF failed: {e}")

        raise RuntimeError("No suitable PDF extraction library (pdfplumber / fitz) could parse the file.")

    def _extract_markdown_tables(self, text: str, page_num: int) -> Tuple[List[ParsedTable], str]:
        """Detect and extract HTML and markdown tables from text, returning tables and remaining text."""
        tables: List[ParsedTable] = []

        # 1. Check for HTML <table> tags (e.g. from OCR/ViFinTab extracts)
        if "<table" in text.lower():
            table_regex = re.compile(r"<table.*?>(.*?)</table>", re.DOTALL | re.IGNORECASE)
            for m in table_regex.finditer(text):
                table_html = m.group(0)
                row_matches = re.findall(r"<tr.*?>(.*?)</tr>", table_html, re.DOTALL | re.IGNORECASE)
                parsed_rows = []
                for tr in row_matches:
                    cols = re.findall(r"<t[dh].*?>(.*?)</t[dh]>", tr, re.DOTALL | re.IGNORECASE)
                    clean_cols = [re.sub(r"<.*?>", "", c).strip() for c in cols]
                    if any(clean_cols):
                        parsed_rows.append(clean_cols)
                if len(parsed_rows) >= 2:
                    headers = parsed_rows[0]
                    data_rows = parsed_rows[1:]
                    tables.append(ParsedTable(
                        page=page_num,
                        table_id=f"page_{page_num}_table_{len(tables)+1}",
                        headers=headers,
                        rows=data_rows,
                    ))
            text = table_regex.sub("", text)

        # 2. Check for Markdown | ... | tables
        lines = text.splitlines()
        remaining_lines: List[str] = []
        table_lines: List[str] = []
        in_table = False

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 2:
                in_table = True
                table_lines.append(stripped)
            else:
                if in_table and table_lines:
                    # Process completed table
                    parsed = self._parse_single_md_table(table_lines, page_num, len(tables) + 1)
                    if parsed:
                        tables.append(parsed)
                    table_lines = []
                    in_table = False
                remaining_lines.append(line)

        if in_table and table_lines:
            parsed = self._parse_single_md_table(table_lines, page_num, len(tables) + 1)
            if parsed:
                tables.append(parsed)

        return tables, "\n".join(remaining_lines)

    def _parse_single_md_table(self, lines: List[str], page_num: int, table_num: int) -> Optional[ParsedTable]:
        if len(lines) < 2:
            return None
        # Header
        headers = [c.strip() for c in lines[0].strip("|").split("|")]
        # Skip separator line (|---|---|)
        data_rows = []
        for line in lines[2:]:
            cols = [c.strip() for c in line.strip("|").split("|")]
            data_rows.append(cols)

        return ParsedTable(
            page=page_num,
            table_id=f"page_{page_num}_table_{table_num}",
            headers=headers,
            rows=data_rows,
        )

    def save_parsed_artifacts(self, result: DocumentParseResult, output_dir: Union[str, Path]) -> Tuple[Path, Path, Path]:
        """Save document.md, tables.json, and parse_log.txt into output_dir."""
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)

        md_path = out_p / "document.md"
        tables_path = out_p / "tables.json"
        log_path = out_p / "parse_log.txt"

        md_path.write_text(result.to_markdown(), encoding="utf-8")
        tables_path.write_text(json.dumps(result.to_tables_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        log_path.write_text("\n".join(result.log_messages), encoding="utf-8")

        return md_path, tables_path, log_path
