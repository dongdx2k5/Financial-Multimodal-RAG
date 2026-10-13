import hashlib
import json
import logging
import re

from .models import Chunk, FinancialFact

logger = logging.getLogger(__name__)

# ============================================================
# Financial Taxonomy (Vietnamese VAS/TT200 + English IFRS)
# ============================================================

FINANCIAL_TAXONOMY: dict[str, list[str]] = {
    "REVENUE_NET": [
        "doanh thu thuần về bán hàng và cung cấp dịch vụ",
        "doanh thu thuần về bán hàng",
        "doanh thu thuần bán hàng",
        "doanh thu thuần",
        "total net revenue", "net revenue", "net sales",
        "total revenue", "revenue",
    ],
    "REVENUE_GROSS": [
        "doanh thu bán hàng và cung cấp dịch vụ",
        "doanh thu bán hàng",
        "tổng doanh thu",
        "gross revenue", "gross sales", "total sales",
    ],
    "COGS": [
        "giá vốn hàng bán",
        "giá vốn",
        "cost of goods sold", "cogs",
        "cost of revenue", "cost of sales",
    ],
    "GROSS_PROFIT": [
        "lợi nhuận gộp về bán hàng và cung cấp dịch vụ",
        "lợi nhuận gộp",
        "lãi gộp",
        "gross profit",
    ],
    "OPERATING_PROFIT": [
        "lợi nhuận thuần từ hoạt động kinh doanh",
        "lợi nhuận từ hoạt động kinh doanh",
        "lợi nhuận hoạt động",
        "operating income", "operating profit", "ebit",
    ],
    "NET_PROFIT": [
        "lợi nhuận sau thuế thu nhập doanh nghiệp",
        "lợi nhuận sau thuế",
        "lợi nhuận ròng",
        "lãi ròng",
        "net income", "net profit", "net earnings",
    ],
    "EBITDA": [
        "ebitda",
        "lợi nhuận trước lãi vay thuế và khấu hao",
    ],
    "TOTAL_ASSETS": [
        "tổng cộng tài sản",
        "tổng tài sản",
        "total assets",
    ],
    "CURRENT_ASSETS": [
        "tài sản ngắn hạn",
        "current assets",
    ],
    "NON_CURRENT_ASSETS": [
        "tài sản dài hạn",
        "non-current assets", "non current assets",
    ],
    "TOTAL_LIABILITIES": [
        "tổng nợ phải trả",
        "nợ phải trả",
        "total liabilities",
    ],
    "CURRENT_LIABILITIES": [
        "nợ ngắn hạn",
        "current liabilities",
    ],
    "NON_CURRENT_LIABILITIES": [
        "nợ dài hạn",
        "non-current liabilities", "long-term liabilities",
    ],
    "EQUITY": [
        "tổng vốn chủ sở hữu",
        "nguồn vốn chủ sở hữu",
        "vốn chủ sở hữu",
        "total equity", "total stockholders' equity",
        "stockholders' equity", "shareholders' equity",
    ],
    "CASH": [
        "tiền và các khoản tương đương tiền",
        "tiền và tương đương tiền",
        "tiền mặt",
        "cash and cash equivalents", "cash",
    ],
    "OPERATING_CASH_FLOW": [
        "lưu chuyển tiền thuần từ hoạt động kinh doanh",
        "dòng tiền hoạt động",
        "operating cash flow",
    ],
    "FREE_CASH_FLOW": [
        "dòng tiền tự do",
        "free cash flow",
    ],
    "EPS": [
        "lãi cơ bản trên cổ phiếu",
        "lãi trên cổ phiếu",
        "thu nhập trên mỗi cổ phiếu",
        "eps", "earnings per share",
    ],
    "RETAINED_EARNINGS": [
        "lợi nhuận sau thuế chưa phân phối",
        "lợi nhuận chưa phân phối",
        "retained earnings",
    ],
    "DEPRECIATION": [
        "khấu hao tài sản cố định",
        "chi phí khấu hao",
        "khấu hao",
        "depreciation", "depreciation and amortization",
    ],
    "INTEREST_EXPENSE": [
        "chi phí lãi vay",
        "chi phí tài chính",
        "interest expense", "finance costs",
    ],
    "TAX_EXPENSE": [
        "chi phí thuế thu nhập doanh nghiệp",
        "chi phí thuế tndn",
        "thuế thu nhập doanh nghiệp",
        "income tax expense", "tax expense",
    ],
    "SELLING_EXPENSE": [
        "chi phí bán hàng",
        "selling expenses",
    ],
    "ADMIN_EXPENSE": [
        "chi phí quản lý doanh nghiệp",
        "chi phí quản lý",
        "general and administrative expenses", "admin expenses",
    ],
}

# Build reverse lookup: alias (casefold) -> standardized code
_METRIC_LOOKUP: dict[str, str] = {}
for _code, _aliases in FINANCIAL_TAXONOMY.items():
    for _alias in _aliases:
        _METRIC_LOOKUP[_alias.casefold()] = _code


def standardize_metric(raw_metric: str) -> str | None:
    """Map a raw metric name to a standardized financial taxonomy code.

    Uses exact match first, then longest-substring match to handle
    variations like 'Doanh thu thuần về bán hàng và CCDV'.
    Returns None if no match found.
    """
    normalized = re.sub(r"\s+", " ", raw_metric).strip().casefold()
    # 1. Exact match
    if normalized in _METRIC_LOOKUP:
        return _METRIC_LOOKUP[normalized]
    # 2. Longest substring match (alias found inside the raw metric name)
    best_code: str | None = None
    best_length = 0
    for alias, code in _METRIC_LOOKUP.items():
        if alias in normalized and len(alias) > best_length:
            best_code = code
            best_length = len(alias)
    return best_code


def extract_metric_from_query(query: str) -> tuple[str | None, str | None]:
    """Detect financial metric terms in a user query.

    Returns (matched_alias, standardized_code) or (None, None).
    """
    query_lower = query.casefold()
    best_alias: str | None = None
    best_code: str | None = None
    best_length = 0
    for alias, code in _METRIC_LOOKUP.items():
        if alias in query_lower and len(alias) > best_length:
            best_alias = alias
            best_code = code
            best_length = len(alias)
    return best_alias, best_code


# ============================================================
# Regex patterns for deterministic extraction
# ============================================================

CURRENCY = r"(?:\$|USD|€|£|₫)"
RAW_NUMBER = r"(?:[-+]?\d{1,3}(?:[.,\s]\d{3})*(?:[.,]\d+)?|[-+]?\d+(?:[.,]\d+)?)"
NUMBER = rf"(?:\(\s*{CURRENCY}?\s*{RAW_NUMBER}\s*\)|{CURRENCY}?\s*{RAW_NUMBER})"

VN_FACT_PATTERN = re.compile(
    rf"(?P<metric>[A-Za-zÀ-ỹĐđ][A-Za-zÀ-ỹĐđ\s,/()%-]{{3,80}}?)"
    rf"(?:\s+năm\s+(?P<fact_year>20\d{{2}}|19\d{{2}}))?\s*(?:là|đạt|:)\s*"
    rf"(?P<value>{NUMBER})\s*(?P<unit>nghìn tỷ|tỷ|triệu|nghìn|%|VND|đồng)?",
    re.IGNORECASE,
)

EN_METRICS = (
    r"Total Revenue|Net Revenue|Revenue|Total Assets|Total Liabilities|Stockholders' Equity|"
    r"Total Equity|Net Income|Net Profit|Operating Income|Operating Profit|Gross Profit|"
    r"EBITDA|EBIT|Operating Cash Flow|Free Cash Flow|Cash and cash equivalents|EPS|Earnings Per Share"
)

EN_FACT_PATTERN = re.compile(
    rf"(?P<metric>{EN_METRICS}|[A-Za-z][A-Za-z\s,/()%-]{{3,60}}?)"
    rf"(?:\s+(?:for|in|of)\s+(?:the\s+year\s+)?(?P<fact_year>20\d{{2}}|19\d{{2}}))?\s*"
    rf"(?:was|reached|amounted to|stood at|is|:|=)\s*"
    rf"(?:\$|USD\s*)?(?P<value>{NUMBER})\s*"
    rf"(?P<unit>billion|million|thousand|trillion|%|USD|\$|dollars)?",
    re.IGNORECASE,
)

FACT_PATTERN = VN_FACT_PATTERN
YEAR_PATTERN = re.compile(r"\b(20\d{2}|19\d{2})\b")
TABLE_ROW_PATTERN = re.compile(r"^\s*\|(.+)\|\s*$")
UNIT_PATTERN = re.compile(
    r"\b(nghìn tỷ|tỷ|triệu|nghìn|billion|million|thousand|trillion|VND|đồng|USD|\$|%)\b",
    re.IGNORECASE,
)


def parse_number(raw: str) -> float:
    """Parse numbers formatted in Vietnamese (1.234,5) or English/International (1,234.5),
    including accounting negative notation: (1,234.5) -> -1234.5.
    """
    clean = raw.strip()
    is_negative = False
    if clean.startswith("(") and clean.endswith(")"):
        is_negative = True
        clean = clean[1:-1].strip()
    elif clean.startswith(("-", "–", "—")):
        is_negative = True
        clean = clean[1:].strip()

    value = (
        clean.replace(" ", "")
        .replace("$", "")
        .replace("€", "")
        .replace("£", "")
        .replace("₫", "")
    )
    if "." in value and "," in value:
        if value.rfind(",") > value.rfind("."):
            # Vietnamese: dot is thousands, comma is decimal (1.234,5)
            value = value.replace(".", "").replace(",", ".")
        else:
            # English: comma is thousands, dot is decimal (1,234.5)
            value = value.replace(",", "")
    elif "," in value:
        parts = value.split(",")
        if len(parts) == 2 and len(parts[1]) != 3:
            value = value.replace(",", ".")
        elif len(parts) > 1 and all(len(p) == 3 for p in parts[1:]):
            value = value.replace(",", "")
        else:
            value = value.replace(",", ".")
    elif "." in value:
        parts = value.split(".")
        if len(parts) > 2 or (len(parts) == 2 and len(parts[1]) == 3 and not value.startswith("0.")):
            value = value.replace(".", "")
    result = float(value)
    return -result if is_negative else result


parse_vietnamese_number = parse_number


def normalize_metric(raw: str) -> str:
    """Remove common report/header residue from an extracted metric name."""
    metric = re.sub(r"\s+", " ", raw).strip(" .,-:")
    if ")" in metric:
        tail = metric.rsplit(")", 1)[-1].strip()
        if tail:
            metric = tail
    unit_join = re.search(r"(?:đồng|VND|USD|\$)\s+và\s+(.+)$", metric, re.IGNORECASE)
    if unit_join:
        metric = unit_join.group(1).strip()
    return metric


def _extract_unit(text: str) -> str | None:
    match = UNIT_PATTERN.search(text)
    return match.group(1) if match else None


def extract_facts_from_tables(chunk: Chunk, default_period: int | None) -> list[FinancialFact]:
    """Extract financial facts directly from structured Markdown table rows."""
    facts: list[FinancialFact] = []
    lines = chunk.text.splitlines()

    def process_table(rows: list[list[str]]):
        if len(rows) < 2:
            return
        header = rows[0]
        start_row = 1
        # Skip markdown separator line (e.g. |---|---|)
        if len(rows) > 1 and all(
            set(c.strip()).issubset({"-", ":", " "}) for c in rows[1] if c.strip()
        ):
            start_row = 2

        col_periods: dict[int, int] = {}
        col_units: dict[int, str] = {}
        metric_col_idx = 0

        for idx, col_name in enumerate(header):
            year_match = YEAR_PATTERN.search(col_name)
            if year_match:
                col_periods[idx] = int(year_match.group(1))
            col_u = _extract_unit(col_name)
            if col_u:
                col_units[idx] = col_u
            if any(
                term in col_name.lower()
                for term in ("chỉ tiêu", "khoản mục", "nội dung", "metric", "item", "account")
            ):
                metric_col_idx = idx

        for row in rows[start_row:]:
            if not row or len(row) <= metric_col_idx:
                continue
            metric_raw = row[metric_col_idx].strip()
            metric = normalize_metric(metric_raw)
            if not metric or len(metric) < 3 or metric.replace(".", "").isdigit():
                continue

            for c_idx, cell in enumerate(row):
                if c_idx == metric_col_idx:
                    continue
                cell_val = cell.strip()
                if not cell_val or cell_val in {"-", "—", "N/A", "na"}:
                    continue

                num_match = re.search(NUMBER, cell_val)
                if not num_match:
                    continue

                raw_value = num_match.group(0).strip()
                try:
                    val = parse_number(raw_value)
                except ValueError:
                    continue

                period = col_periods.get(c_idx) or default_period
                unit = _extract_unit(cell_val) or col_units.get(c_idx)

                std = standardize_metric(metric)
                fact_id = hashlib.sha1(
                    f"{chunk.id}|{metric}|{raw_value}|{period}".encode()
                ).hexdigest()[:16]
                facts.append(
                    FinancialFact(
                        id=fact_id,
                        metric=metric,
                        standardized_metric=std,
                        value=val,
                        raw_value=raw_value,
                        unit=unit,
                        period=period,
                        ticker=chunk.ticker,
                        chunk_id=chunk.id,
                        source=chunk.source,
                        page=chunk.page,
                        confidence=0.85,
                    )
                )

    current_table: list[list[str]] = []
    for line in lines:
        match = TABLE_ROW_PATTERN.match(line)
        if match:
            cells = [c.strip() for c in match.group(1).split("|")]
            current_table.append(cells)
        else:
            if current_table:
                process_table(current_table)
                current_table = []
    if current_table:
        process_table(current_table)

    return facts


class FinancialFactExtractor:
    """Deterministic extractor supporting Vietnamese and English reports, text, and tables."""

    def extract(self, chunk: Chunk) -> list[FinancialFact]:
        years = [int(value) for value in YEAR_PATTERN.findall(chunk.text)]
        default_period = chunk.year or (years[-1] if years else None)
        facts: list[FinancialFact] = []
        seen_keys: set[str] = set()

        # 1. Extract from structured Markdown tables
        for table_fact in extract_facts_from_tables(chunk, default_period):
            dedup_key = f"{table_fact.metric.lower()}|{table_fact.raw_value}|{table_fact.period}"
            if dedup_key not in seen_keys:
                seen_keys.add(dedup_key)
                facts.append(table_fact)

        # 2. Extract from narrative text patterns
        for pattern in (VN_FACT_PATTERN, EN_FACT_PATTERN):
            for match in pattern.finditer(chunk.text):
                metric = normalize_metric(match.group("metric"))
                raw_value = match.group("value")
                try:
                    value = parse_number(raw_value)
                except ValueError:
                    continue

                period = (
                    int(match.group("fact_year")) if match.group("fact_year") else default_period
                )
                dedup_key = f"{metric.lower()}|{raw_value}|{period}"
                if dedup_key in seen_keys:
                    continue
                seen_keys.add(dedup_key)

                std = standardize_metric(metric)
                fact_id = hashlib.sha1(f"{chunk.id}|{metric}|{raw_value}".encode()).hexdigest()[:16]
                facts.append(
                    FinancialFact(
                        id=fact_id,
                        metric=metric,
                        standardized_metric=std,
                        value=value,
                        raw_value=raw_value,
                        unit=match.group("unit"),
                        period=period,
                        ticker=chunk.ticker,
                        chunk_id=chunk.id,
                        source=chunk.source,
                        page=chunk.page,
                        confidence=0.75,
                    )
                )
        return facts


# ============================================================
# LLM Structured Extraction (optional fallback for complex tables)
# ============================================================


class LLMFactExtractor:
    """Gemini-based structured extraction fallback for complex financial tables.

    Uses Gemini's generative capability to extract facts from text that the
    regex-based extractor cannot handle (e.g. merged cells, nested tables,
    non-standard formatting).
    """

    def __init__(self, api_key: str, model_name: str = "gemini-3.8-flash"):
        self.api_key = api_key
        self.model_name = model_name.removeprefix("models/")
        self._client = None
        try:
            from google import genai

            self._client = genai.Client(api_key=api_key)
        except ImportError:
            logger.debug("google-genai not available; LLM fact extraction disabled.")

    def extract(self, chunk: Chunk) -> list[FinancialFact]:
        """Extract financial facts using Gemini structured output."""
        if not self._client:
            return []

        prompt = (
            "Bạn là chuyên gia phân tích tài chính. Trích xuất TẤT CẢ các dữ kiện tài chính "
            "từ đoạn văn/bảng biểu sau. Mỗi dữ kiện gồm:\n"
            '- "metric": Tên chỉ tiêu tài chính (giữ nguyên ngôn ngữ gốc)\n'
            '- "value": Giá trị số (float, số âm nếu là lỗ)\n'
            '- "raw_value": Giá trị gốc dạng chuỗi\n'
            '- "unit": Đơn vị (tỷ, triệu, billion, million, %, VND, USD,...) hoặc null\n'
            '- "period": Năm tài chính (4 chữ số) hoặc null\n\n'
            f"Đoạn văn:\n{chunk.text}\n\n"
            "Trả về ĐÚNG một JSON array. Nếu không có dữ kiện, trả về []."
        )

        try:
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            text = response.text or ""
            # Strip markdown code fences if present
            text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`")
            items = json.loads(text)
            if not isinstance(items, list):
                return []
        except Exception as exc:
            logger.debug("LLM fact extraction failed: %s", exc)
            return []

        facts: list[FinancialFact] = []
        for item in items:
            try:
                metric = str(item.get("metric", "")).strip()
                if not metric or len(metric) < 2:
                    continue
                raw_value = str(item.get("raw_value", item.get("value", "")))
                value = float(item["value"])
                unit = item.get("unit")
                period = int(item["period"]) if item.get("period") else chunk.year
                std = standardize_metric(metric)

                fact_id = hashlib.sha1(
                    f"llm|{chunk.id}|{metric}|{raw_value}|{period}".encode()
                ).hexdigest()[:16]

                facts.append(
                    FinancialFact(
                        id=fact_id,
                        metric=metric,
                        standardized_metric=std,
                        value=value,
                        raw_value=raw_value,
                        unit=unit,
                        period=period,
                        ticker=chunk.ticker,
                        chunk_id=chunk.id,
                        source=chunk.source,
                        page=chunk.page,
                        confidence=0.90,
                    )
                )
            except (ValueError, KeyError, TypeError):
                continue

        return facts

