"""Financial value normalization and table header flattening.

Handles the quirks of financial data:
- Negative numbers in parentheses: (125.4) -> -125.4
- Currency symbols and thousand separators: $1,200.50 -> 1200.50
- Scale factor detection from headers: "in millions" -> 1_000_000
- Flattened row headers to disambiguate metrics like Operating Income vs Net Income
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import Any


# ---------------------------------------------------------------------------
# 1. Single-value normalization
# ---------------------------------------------------------------------------

# Regex for parenthesised negative: (1,234.56) or ($1,234.56)
_PAREN_NEG_RE = re.compile(
    r"^\s*\(?\s*[$€£¥₫]?\s*\(([^)]+)\)\s*[$€£¥₫%]?\s*\)?\s*$"
)

# Currency / percent symbols to strip before numeric parsing
_CURRENCY_SYMBOLS = re.compile(r"[$€£¥₫]")

# Thousand-separator commas (only those followed by exactly 3 digits)
_THOUSAND_SEP = re.compile(r"(?<=\d),(?=\d{3}(?:\D|$))")

# Whitespace / non-breaking spaces
_WHITESPACE = re.compile(r"[\s\u00a0]+")


def normalize_financial_value(raw: str) -> str:
    """Normalize a single financial cell value to a clean numeric string.

    Transformations applied (in order):
    1. Parenthesised negatives:  ``(125.4)`` → ``-125.4``
    2. Currency symbols removed: ``$``, ``€``, ``£``, ``¥``, ``₫``
    3. Percent symbol removed (value kept as-is, caller handles semantics).
    4. Thousand-separator commas: ``1,200`` → ``1200``
    5. Leading/trailing whitespace stripped.

    Non-numeric strings (labels, dates, headers) are returned unchanged.
    """
    if not raw or not raw.strip():
        return raw

    text = _WHITESPACE.sub(" ", raw).strip()

    # --- Parenthesised negative ---
    paren_match = _PAREN_NEG_RE.match(text)
    if not paren_match:
        # Simpler pattern: just look for (digits) without extra symbols outside
        inner_match = re.match(r"^\(([^)]+)\)$", text)
        if inner_match:
            inner = inner_match.group(1).strip()
            inner_clean = _CURRENCY_SYMBOLS.sub("", inner)
            inner_clean = _THOUSAND_SEP.sub("", inner_clean).strip()
            # Only treat as negative if the inner content is actually numeric
            try:
                float(inner_clean)
                text = f"-{inner_clean}"
                return text
            except ValueError:
                pass
    else:
        inner = paren_match.group(1).strip()
        inner_clean = _CURRENCY_SYMBOLS.sub("", inner)
        inner_clean = _THOUSAND_SEP.sub("", inner_clean).strip()
        try:
            float(inner_clean)
            text = f"-{inner_clean}"
            return text
        except ValueError:
            pass

    # --- Remove currency symbols ---
    text = _CURRENCY_SYMBOLS.sub("", text)

    # --- Remove percent symbol (keep the number) ---
    text = text.replace("%", "").strip()

    # --- Remove thousand-separator commas ---
    text = _THOUSAND_SEP.sub("", text)

    text = text.strip()
    # If the result looks numeric after cleanup, return it; otherwise return
    # the *original* stripped text so labels stay readable.
    try:
        float(text.replace(" ", ""))
        return text
    except ValueError:
        return raw.strip()


# ---------------------------------------------------------------------------
# 2. Scale-factor detection
# ---------------------------------------------------------------------------

_SCALE_PATTERNS: list[tuple[re.Pattern[str], int]] = [
    (re.compile(r"\bin\s+billions?\b", re.IGNORECASE), 1_000_000_000),
    (re.compile(r"\bin\s+millions?\b", re.IGNORECASE), 1_000_000),
    (re.compile(r"\bin\s+thousands?\b", re.IGNORECASE), 1_000),
    (re.compile(r"\bin\s+hundreds?\b", re.IGNORECASE), 100),
    # Vietnamese equivalents
    (re.compile(r"\btỷ\s+đồng\b", re.IGNORECASE), 1_000_000_000),
    (re.compile(r"\btriệu\s+đồng\b", re.IGNORECASE), 1_000_000),
    (re.compile(r"\bnghìn\s+đồng\b", re.IGNORECASE), 1_000),
    (re.compile(r"\bngàn\s+đồng\b", re.IGNORECASE), 1_000),
]


def detect_scale_factor(header_text: str) -> int:
    """Detect the numeric scale from a table header / caption.

    Returns the multiplier (e.g. 1_000_000 for "in millions").
    Returns 1 when no scale indicator is found.
    """
    if not header_text:
        return 1
    for pattern, factor in _SCALE_PATTERNS:
        if pattern.search(header_text):
            return factor
    return 1


# ---------------------------------------------------------------------------
# 3. Normalize all values in a 2D table
# ---------------------------------------------------------------------------

def normalize_table_2d(
    table_2d: list[list[str | None]],
    skip_first_row: bool = True,
    skip_first_col: bool = True,
) -> list[list[str]]:
    """Normalize numeric values in a 2D table grid.

    Parameters
    ----------
    table_2d : list of lists
        The raw 2D table (rows × cols).
    skip_first_row : bool
        If True, treat row 0 as header and leave it unchanged.
    skip_first_col : bool
        If True, treat column 0 as row labels and leave it unchanged.

    Returns
    -------
    Normalized copy of the table.
    """
    if not table_2d:
        return []

    result: list[list[str]] = []
    for row_idx, row in enumerate(table_2d):
        new_row: list[str] = []
        for col_idx, cell in enumerate(row):
            cell_str = str(cell) if cell is not None else ""
            if (skip_first_row and row_idx == 0) or (skip_first_col and col_idx == 0):
                new_row.append(cell_str)
            else:
                new_row.append(normalize_financial_value(cell_str))
            
        result.append(new_row)
    return result


# ---------------------------------------------------------------------------
# 4. HTML table normalization
# ---------------------------------------------------------------------------

class _SimpleTableParser(HTMLParser):
    """Minimal HTML parser that extracts table cell values."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._current_row: list[str] = []
        self._current_cell: list[str] = []
        self._in_cell = False
        self._header_text: str = ""  # text from <thead>
        self._in_thead = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag == "tr":
            self._current_row = []
        elif tag in ("td", "th"):
            self._in_cell = True
            self._current_cell = []
        elif tag == "thead":
            self._in_thead = True

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in ("td", "th"):
            self._in_cell = False
            self._current_row.append("".join(self._current_cell).strip())
        elif tag == "tr":
            if self._current_row:
                self.rows.append(self._current_row)
        elif tag == "thead":
            self._in_thead = False

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._current_cell.append(data)
            if self._in_thead:
                self._header_text += " " + data


def _rebuild_html(rows: list[list[str]], has_header: bool = True) -> str:
    """Rebuild a clean HTML table from a 2D string grid."""
    parts = ["<table>"]
    for row_idx, row in enumerate(rows):
        if has_header and row_idx == 0:
            header_cells = "".join(f"<th>{c}</th>" for c in row)
            parts.append(f"<thead><tr>{header_cells}</tr></thead><tbody>")
        else:
            data_cells = "".join(f"<td>{c}</td>" for c in row)
            parts.append(f"<tr>{data_cells}</tr>")
    if has_header and len(rows) > 0:
        parts.append("</tbody>")
    parts.append("</table>")
    return "".join(parts)


def normalize_table_values(table_html: str) -> tuple[str, int]:
    """Parse an HTML table, normalize financial values, and rebuild.

    Returns
    -------
    tuple of (normalized_html, scale_factor)
    """
    if not table_html or not table_html.strip():
        return table_html, 1

    parser = _SimpleTableParser()
    parser.feed(table_html)

    if not parser.rows:
        return table_html, 1

    scale = detect_scale_factor(parser._header_text)

    normalized = normalize_table_2d(
        parser.rows, skip_first_row=True, skip_first_col=True
    )
    has_header = "<thead>" in table_html.lower()
    return _rebuild_html(normalized, has_header=has_header), scale


# ---------------------------------------------------------------------------
# 5. Flatten row headers (disambiguate similar metrics)
# ---------------------------------------------------------------------------

def _is_group_header_row(row: list[str], total_cols: int) -> bool:
    """Heuristic: a row is a group header if only the first cell has content
    and the rest are empty/whitespace, and the table has more than 2 columns."""
    if total_cols <= 2:
        return False
    if not row or not row[0].strip():
        return False
    non_empty = sum(1 for c in row[1:] if c.strip())
    return non_empty == 0


def flatten_row_headers_2d(
    table_2d: list[list[str | None]],
    separator: str = " > ",
) -> list[list[str]]:
    """Add parent group prefixes to row labels for disambiguation.

    Detects group-header rows (rows where only the first column has text)
    and prepends their label to subsequent data rows.

    Example::

        ["Operating Expenses", "", "", ""]     ← group header
        ["Research",        "100", "120", ""]   → "Operating Expenses > Research"
        ["Marketing",      "80",  "90",  ""]   → "Operating Expenses > Marketing"
        ["Net Income",     "200", "250", ""]    ← standalone row (unchanged)
    """
    if not table_2d:
        return []

    total_cols = max(len(r) for r in table_2d) if table_2d else 0
    result: list[list[str]] = []
    current_group: str = ""

    for row in table_2d:
        str_row = [str(c) if c is not None else "" for c in row]
        # Pad to uniform width
        while len(str_row) < total_cols:
            str_row.append("")

        if _is_group_header_row(str_row, total_cols):
            current_group = str_row[0].strip()
            # Keep the group header row itself in the output
            result.append(str_row)
        else:
            has_data_values = any(c.strip() for c in str_row[1:])
            if current_group and str_row[0].strip() and not has_data_values:
                # Child row under a group header (no data values = sub-header/label)
                new_row = list(str_row)
                new_row[0] = f"{current_group}{separator}{str_row[0].strip()}"
                result.append(new_row)
            elif current_group and str_row[0].strip() and has_data_values:
                # Row with data that belongs to the group
                non_empty_data = sum(1 for c in str_row[1:] if c.strip())
                # If most data columns are filled, treat as a child row
                # If it looks like a standalone metric (all data filled), check
                # whether the label is indented or clearly a sub-item
                new_row = list(str_row)
                new_row[0] = f"{current_group}{separator}{str_row[0].strip()}"
                result.append(new_row)
            else:
                # No active group or empty first cell → reset group if this row has data
                if str_row[0].strip() and has_data_values:
                    current_group = ""
                result.append(str_row)

    return result


def flatten_row_headers(table_html: str, separator: str = " > ") -> str:
    """Apply flatten_row_headers logic to an HTML table string."""
    if not table_html or not table_html.strip():
        return table_html

    parser = _SimpleTableParser()
    parser.feed(table_html)

    if not parser.rows:
        return table_html

    # Don't flatten the header row (row 0 in a <thead>)
    has_header = "<thead>" in table_html.lower()
    if has_header and len(parser.rows) > 1:
        header = [parser.rows[0]]
        body = flatten_row_headers_2d(parser.rows[1:], separator=separator)
        all_rows = header + body
    else:
        all_rows = flatten_row_headers_2d(parser.rows, separator=separator)

    return _rebuild_html(all_rows, has_header=has_header)
