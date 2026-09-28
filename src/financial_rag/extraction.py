import hashlib
import re

from .models import Chunk, FinancialFact

NUMBER = r"[-+]?\d{1,3}(?:[.\s]\d{3})*(?:,\d+)?|[-+]?\d+(?:[.,]\d+)?"
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
    """Deterministic baseline. Replace with a structured-output LLM/VLM adapter later."""

    def extract(self, chunk: Chunk) -> list[FinancialFact]:
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
            fact_id = hashlib.sha1(f"{chunk.id}|{metric}|{raw_value}".encode()).hexdigest()[:16]
            period = int(match.group("fact_year")) if match.group("fact_year") else default_period
            facts.append(
                FinancialFact(
                    id=fact_id,
                    metric=metric,
                    value=value,
                    raw_value=raw_value,
                    unit=match.group("unit"),
                    period=period,
                    ticker=chunk.ticker,
                    chunk_id=chunk.id,
                    source=chunk.source,
                    page=chunk.page,
                    confidence=0.65,
                )
            )
        return facts
