"""Slot extractor for converting natural language queries into structured query parameters."""

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from financial_rag.db.repository import FinancialRepository


@dataclass
class ExtractedSlots:
    ticker: Optional[str] = None
    fiscal_year: Optional[int] = None
    period_type: Optional[str] = None
    metric_code: Optional[str] = None
    line_item: Optional[str] = None
    question_type: str = "text"  # lookup | text | calc | compare | trap
    calc_type: Optional[str] = None  # margin | de | yoy | growth
    secondary_ticker: Optional[str] = None
    raw_query: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "fiscal_year": self.fiscal_year,
            "period_type": self.period_type,
            "metric_code": self.metric_code,
            "line_item": self.line_item,
            "question_type": self.question_type,
            "calc_type": self.calc_type,
            "secondary_ticker": self.secondary_ticker,
            "raw_query": self.raw_query,
        }


class SlotExtractor:
    """Extracts entity, time period, metric, and intent slots from financial questions."""

    def __init__(self, repo: FinancialRepository):
        self.repo = repo
        self._load_metadata()

    def _load_metadata(self) -> None:
        self.companies = self.repo.get_all_companies()
        self.metrics = self.repo.get_all_metrics()
        self.known_tickers = {c["ticker"].upper(): c for c in self.companies}

    def extract(self, query: str, llm_response_json: Optional[str] = None) -> ExtractedSlots:
        """
        Extract slots using LLM output if provided; otherwise applies high-precision rule parser.
        """
        if llm_response_json:
            try:
                # Find JSON block
                json_match = re.search(r"\{.*?\}", llm_response_json, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group(0))
                    return ExtractedSlots(
                        ticker=data.get("ticker"),
                        fiscal_year=data.get("fiscal_year"),
                        period_type=data.get("period_type"),
                        metric_code=data.get("metric_code"),
                        line_item=data.get("line_item"),
                        question_type=data.get("question_type", "text"),
                        calc_type=data.get("calc_type"),
                        secondary_ticker=data.get("secondary_ticker"),
                        raw_query=query,
                    )
            except Exception:
                pass

        return self.rule_based_extract(query)

    def rule_based_extract(self, query: str) -> ExtractedSlots:
        q_upper = query.upper()
        q_lower = query.lower()

        # 1. Detect Tickers
        self._load_metadata()
        found_tickers = []
        # 1. Match against known database tickers
        for ticker in self.known_tickers:
            if re.search(rf"\b{re.escape(ticker)}\b", query, re.IGNORECASE):
                if ticker not in found_tickers:
                    found_tickers.append(ticker)

        # 2. Match company full names and brand keywords
        for c in self.companies:
            t = c["ticker"].upper()
            if t in found_tickers:
                continue
            names = [c.get("name_vi") or "", c.get("name_en") or ""]
            for name in names:
                if not name:
                    continue
                if name.lower() in q_lower:
                    found_tickers.append(t)
                    break
                # Strip corporate prefixes/suffixes: công ty, cổ phần, tập đoàn, jsc, corporation, corp, inc, ltd
                cleaned_name = re.sub(
                    r"\b(công ty|cổ phần|tập đoàn|jsc|corporation|corp|inc|ltd|limited|holdings)\b",
                    "",
                    name,
                    flags=re.IGNORECASE,
                ).strip()
                # Clean punctuation like parentheses
                clean_tokens = [tok.strip("()") for tok in cleaned_name.split() if len(tok.strip("()")) >= 3]
                if any(tok.lower() in q_lower for tok in clean_tokens):
                    found_tickers.append(t)
                    break

        # 3. Check for external/unknown tickers explicitly typed in all-caps (e.g., TSLA, AAPL, NVL)
        candidate_stopwords = {
            "BCTC", "VND", "USD", "ROA", "ROE", "GROSS", "NET", "TOTAL",
            "DOANH", "THU", "LNST", "KQKD", "CDKT", "VCSH", "TSNH", "TSDH",
            "WHAT", "WHEN", "WHERE", "HOW", "WHY"
        }
        # Only inspect uppercase tokens from the raw query
        extra_caps = re.findall(r"\b[A-Z]{3,5}\b", query)
        for t in extra_caps:
            if t not in found_tickers and t not in candidate_stopwords:
                found_tickers.append(t)

        primary_ticker = found_tickers[0] if len(found_tickers) >= 1 else None
        secondary_ticker = found_tickers[1] if len(found_tickers) >= 2 else None

        # 2. Fiscal Year (e.g., 2024, 2025, 2026, 2027)
        year_match = re.search(r"\b(202[0-9]|201[5-9])\b", query)
        fiscal_year = int(year_match.group(1)) if year_match else None

        # 3. Period Type: H1, 6 tháng, 9M, 9 tháng, Q1, Q2, Q3, Q4, FY, cả năm
        period_type = None
        if re.search(r"\b(h1|6\s*tháng|bán\s*niên)\b", q_lower):
            period_type = "H1"
        elif re.search(r"\b(9m|9\s*tháng)\b", q_lower):
            period_type = "9M"
        elif re.search(r"\b(q1|quý\s*1)\b", q_lower):
            period_type = "Q1"
        elif re.search(r"\b(q2|quý\s*2)\b", q_lower):
            period_type = "Q2"
        elif re.search(r"\b(q3|quý\s*3)\b", q_lower):
            period_type = "Q3"
        elif re.search(r"\b(q4|quý\s*4)\b", q_lower):
            period_type = "Q4"
        elif re.search(r"\b(fy|cả\s*năm|năm\s*tài\s*chính|full\s*year)\b", q_lower) or fiscal_year:
            period_type = "FY"

        # 4. Metric matching
        matched_metric = self.repo.match_metric_alias(query)
        metric_code = matched_metric["code"] if matched_metric else None

        # 5. Question Type & Intent Classification
        question_type = "text"
        calc_type = None

        # Check for Trap / Out-of-scope questions
        trap_keywords = [
            "dự báo giá", "mua hay bán", "có nên mua", "khuyến nghị",
            "giá cổ phiếu", "stock price", "should i buy", "price prediction",
            "forecast", "kế hoạch 2027", "2027", "2028", "tokyo", "văn phòng đại diện"
        ]
        if any(kw in q_lower for kw in trap_keywords):
            question_type = "trap"

        # Check for Comparison
        elif secondary_ticker or any(kw in q_lower for kw in ["so sánh", "compare", "giữa", "lớn hơn", "nhỏ hơn", "cao hơn", "thấp hơn"]):
            question_type = "compare"

        # Check for Calculation
        elif any(kw in q_lower for kw in ["biên lợi nhuận", "margin", "tỷ số", "tỷ lệ nợ", "d/e", "tăng trưởng", "growth", "yoy", "phần trăm"]):
            question_type = "calc"
            if "biên lợi nhuận" in q_lower or "margin" in q_lower:
                calc_type = "margin"
            elif "nợ trên vốn" in q_lower or "d/e" in q_lower or "nợ/vốn" in q_lower:
                calc_type = "de"
            elif "tăng trưởng" in q_lower or "growth" in q_lower or "yoy" in q_lower or "cùng kỳ" in q_lower:
                calc_type = "yoy"

        # Check for Direct Metric Lookup
        elif metric_code or any(kw in q_lower for kw in ["bao nhiêu", "how much", "what was", "giá trị", "đạt bao nhiêu"]):
            if any(kw in q_lower for kw in ["tại sao", "giải thích", "nguyên nhân", "chính sách", "ý kiến", "rủi ro", "why", "explain", "risk", "opinion"]):
                question_type = "text"
            else:
                question_type = "lookup"

        return ExtractedSlots(
            ticker=primary_ticker,
            fiscal_year=fiscal_year,
            period_type=period_type,
            metric_code=metric_code,
            question_type=question_type,
            calc_type=calc_type,
            secondary_ticker=secondary_ticker,
            raw_query=query,
        )
