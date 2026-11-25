import hashlib
import re

from .models import Chunk, FinancialFact

NUMBER = r"[-+]?(?:\d{1,3}(?:[.\s]\d{3})+(?:,\d+)?|\d+(?:[.,]\d+)?)"
FACT_PATTERN = re.compile(
    rf"(?P<metric>[A-Za-zÀ-ỹĐđ][A-Za-zÀ-ỹĐđ\s,/()%-]{{3,80}}?)"
    rf"(?:\s+năm\s+(?P<fact_year>20\d{{2}}|19\d{{2}}))?\s*(?:là|đạt|:)\s*"
    rf"(?P<value>{NUMBER})\s*(?P<unit>nghìn tỷ|tỷ|triệu|nghìn|%|VND|đồng)?",
    re.IGNORECASE,
)
YEAR_PATTERN = re.compile(r"\b(20\d{2}|19\d{2})\b")


def parse_vietnamese_number(raw: str) -> float:
    value = raw.replace(" ", "")
    if "." in value and "," in value:
        value = value.replace(".", "").replace(",", ".")
    elif value.count(".") > 1 or ("." in value and len(value.rsplit(".", 1)[1]) == 3):
        value = value.replace(".", "")
    else:
        value = value.replace(",", ".")
    return float(value)


def normalize_metric(raw: str) -> str:
    """Remove common report/header residue from a regex-extracted metric name."""
    metric = re.sub(r"\s+", " ", raw).strip(" .,-")
    if ")" in metric:
        tail = metric.rsplit(")", 1)[-1].strip()
        if tail:
            metric = tail
    unit_join = re.search(r"(?:đồng|VND)\s+và\s+(.+)$", metric, re.IGNORECASE)
    if unit_join:
        metric = unit_join.group(1).strip()
    return metric


class FinancialFactExtractor:
    """Deterministic extraction with source provenance; no language-model calls."""

    def extract(self, chunk: Chunk) -> list[FinancialFact]:
        # Experimental structure recognition is evidence, not an approved financial mapping.
        # Merged headers / period semantics must be validated before arithmetic is allowed.
        if chunk.table_structure_engine or chunk.ocr_engine in {"vietocr", "colab_gpu"}:
            return []
        if chunk.extraction_method == "chart" and chunk.chart_points:
            chart_facts = []
            for idx, pt in enumerate(chunk.chart_points):
                raw_val = str(pt.get("raw_value") or pt.get("value") or "")
                val = pt.get("value")
                if val is None or not raw_val:
                    continue
                series = str(pt.get("series") or "Biểu đồ")
                period_label = pt.get("period_label")
                period = pt.get("period")
                fact_id = hashlib.sha1(
                    f"{chunk.id}|{series}|{period_label}|{raw_val}|{idx}".encode()
                ).hexdigest()[:16]
                chart_facts.append(
                    FinancialFact(
                        id=fact_id,
                        document_id=chunk.document_id,
                        chunk_id=chunk.id,
                        source=chunk.source,
                        ticker=chunk.ticker,
                        page=chunk.page,
                        metric=series,
                        value=float(val),
                        raw_value=raw_val,
                        unit=pt.get("unit"),
                        period=period,
                        period_label=period_label,
                        period_kind="year" if period_label and str(period_label).isdigit() else None,
                        scope=chunk.scope,
                        statement="chart",
                        bbox=chunk.bbox,
                        extraction_method="chart",
                        confidence=0.70,
                        quality="unverified",
                    )
                )
            return chart_facts
        if chunk.financial_rows:
            return [
                FinancialFact(
                    id=hashlib.sha1(f"{chunk.id}|{row['period_label']}".encode()).hexdigest()[:16],
                    **row,
                    document_id=chunk.document_id,
                    ticker=chunk.ticker,
                    scope=chunk.scope,
                    chunk_id=chunk.id,
                    source=chunk.source,
                    page=chunk.page,
                    bbox=chunk.bbox,
                    extraction_method=chunk.extraction_method,
                    confidence=0.7,
                    quality="ocr_unverified" if chunk.extraction_method == "ocr" else "unverified",
                )
                for row in chunk.financial_rows
            ]
        if chunk.table:
            return self.extract_table(chunk)
        years = [int(value) for value in YEAR_PATTERN.findall(chunk.text)]
        default_period = chunk.year or (years[-1] if years else None)
        facts: list[FinancialFact] = []
        for match in FACT_PATTERN.finditer(chunk.text):
            metric = normalize_metric(match.group("metric"))
            raw_value = match.group("value")
            try:
                value = parse_vietnamese_number(raw_value)
            except ValueError:
                continue
            period = int(match.group("fact_year")) if match.group("fact_year") else default_period
            fact_id = hashlib.sha1(
                f"{chunk.id}|{metric}|{period}|{raw_value}|{match.start()}".encode()
            ).hexdigest()[:16]
            facts.append(
                FinancialFact(
                    id=fact_id,
                    metric=metric,
                    value=value,
                    raw_value=raw_value,
                    unit=match.group("unit"),
                    period=period,
                    document_id=chunk.document_id,
                    ticker=chunk.ticker,
                    scope=chunk.scope,
                    chunk_id=chunk.id,
                    source=chunk.source,
                    page=chunk.page,
                    bbox=chunk.bbox,
                    extraction_method=chunk.extraction_method,
                    confidence=0.65,
                )
            )
        return facts

    def extract_table(self, chunk: Chunk) -> list[FinancialFact]:
        # Only explicit year headers are accepted; merged/ambiguous headers are not guessed.
        rows = chunk.table or []
        facts = []
        headers = {}
        header_index = -1
        for row in rows[:5]:
            candidate = {}
            for col, cell in enumerate(row):
                years = YEAR_PATTERN.findall(cell or "")
                if len(years) == 1:
                    candidate[col] = int(years[0])
            if candidate:
                headers = candidate
                header_index = rows.index(row)
                break
        for row_index, row in enumerate(rows):
            if row_index <= header_index:
                continue
            metric = (row[0] or "").strip() if row else ""
            if not metric or YEAR_PATTERN.search(metric):
                continue
            for col, year in headers.items():
                if col == 0 or col >= len(row):
                    continue
                raw = (row[col] or "").strip()
                if not re.fullmatch(r"\(?[-+]?\d[\d.,\s]*\)?", raw):
                    continue
                try:
                    value = parse_vietnamese_number(raw.strip("()"))
                except ValueError:
                    continue
                if raw.startswith("("):
                    value = -value
                facts.append(
                    FinancialFact(
                        id=hashlib.sha1(f"{chunk.id}|{row_index}|{col}".encode()).hexdigest()[:16],
                        metric=metric,
                        value=value,
                        raw_value=raw,
                        period=year,
                        unit=chunk.table_unit,
                        document_id=chunk.document_id,
                        scope=chunk.scope,
                        chunk_id=chunk.id,
                        source=chunk.source,
                        page=chunk.page,
                        bbox=chunk.bbox,
                        extraction_method=chunk.extraction_method,
                        ticker=chunk.ticker,
                        confidence=0.7,
                    )
                )
        return facts
