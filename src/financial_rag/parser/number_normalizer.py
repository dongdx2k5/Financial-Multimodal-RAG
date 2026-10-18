"""Number and financial unit normalization module."""

import re
from typing import Optional, Tuple


class NumberNormalizer:
    """Robust parser for Vietnamese and English financial numbers, units and currency."""

    # Unit scaling mapping
    UNIT_SCALE_MAP = {
        "dong": 1,
        "đồng": 1,
        "vnd": 1,
        "nghìn": 1_000,
        "ngàn": 1_000,
        "thousand": 1_000,
        "triệu": 1_000_000,
        "million": 1_000_000,
        "m": 1_000_000,
        "tỷ": 1_000_000_000,
        "ti": 1_000_000_000,
        "billion": 1_000_000_000,
        "b": 1_000_000_000,
        "usd": 1,
        "$": 1,
    }

    @classmethod
    def parse_number(cls, raw: str) -> Optional[float]:
        """
        Parse raw financial string into float.
        Handles:
          - Vietnamese: 1.234.567,89
          - English: 1,234,567.89
          - Negative in parens: (1.234) or (1,234.5)
          - Negative with dash/en-dash: -1.234 or –1.234
          - Percentage: 12.5% -> 12.5
          - Blank or dash placeholders: '-', '–', '' -> None or 0
        """
        if raw is None:
            return None
        text = str(raw).strip()
        if not text or text in ("-", "–", "—", "N/A", "n/a", "None"):
            return 0.0

        is_negative = False

        # Check negative parentheses: (123,456) or ( 123.456 )
        if (text.startswith("(") and text.endswith(")")) or (text.startswith("（") and text.endswith("）")):
            is_negative = True
            text = text[1:-1].strip()

        # Check negative sign
        if text.startswith("-") or text.startswith("–") or text.startswith("—"):
            is_negative = True
            text = text.lstrip("-–— ").strip()

        # Remove currency symbols and percentage
        text = re.sub(r"[đ₫\$€£¥%]", "", text).strip()

        # Distinguish Vietnamese (1.234.567,89) vs English (1,234,567.89)
        # Case 1: Has both dot and comma
        if "." in text and "," in text:
            first_dot = text.find(".")
            first_comma = text.find(",")
            if first_dot < first_comma:
                # English thousands separator is comma, decimal is dot: 1,234.56 (comma before dot)
                # Vietnamese: dot before comma: 1.234,56
                # Dot first: Vietnamese
                clean = text.replace(".", "").replace(",", ".")
            else:
                # Comma first: English (1,234.56)
                clean = text.replace(",", "")
        # Case 2: Only comma
        elif "," in text:
            comma_parts = text.split(",")
            # If after comma there are exactly 3 digits repeated (e.g. 1,234,567), it's English thousands separator
            if all(len(p) == 3 for p in comma_parts[1:]):
                clean = text.replace(",", "")
            elif len(comma_parts) == 2 and len(comma_parts[1]) <= 2:
                # Decimal comma: Vietnamese 123,45
                clean = f"{comma_parts[0]}.{comma_parts[1]}"
            else:
                clean = text.replace(",", "")
        # Case 3: Only dot
        elif "." in text:
            dot_parts = text.split(".")
            # If after dot there are exactly 3 digits (e.g. 1.234.567), it's Vietnamese thousands separator
            if all(len(p) == 3 for p in dot_parts[1:]):
                clean = text.replace(".", "")
            else:
                # Standard decimal dot: 123.45
                clean = text
        else:
            clean = text

        # Remove any remaining stray spaces
        clean = re.sub(r"\s+", "", clean)

        try:
            val = float(clean)
            return -val if is_negative else val
        except ValueError:
            return None

    @classmethod
    def normalize_with_scale(
        cls, raw_value: str, unit_scale: int = 1
    ) -> Tuple[Optional[float], str]:
        """Return (scaled_value, raw_value_string)."""
        val = cls.parse_number(raw_value)
        if val is None:
            return None, str(raw_value)
        scaled = val * unit_scale
        return scaled, str(raw_value)

    @classmethod
    def extract_unit_scale_from_text(cls, text: str) -> int:
        """Infer unit scale from text headers such as 'Đơn vị tính: triệu đồng'."""
        text_lower = text.lower()
        if "tỷ đồng" in text_lower or "billion" in text_lower:
            return 1_000_000_000
        if "triệu đồng" in text_lower or "million" in text_lower:
            return 1_000_000
        if "nghìn đồng" in text_lower or "ngàn đồng" in text_lower or "thousand" in text_lower:
            return 1_000
        return 1

    @classmethod
    def format_financial(cls, value: float, currency: str = "VND", unit: str = "auto") -> str:
        """Pretty format number for natural language presentation."""
        if value is None:
            return "N/A"
        abs_v = abs(value)
        sign = "-" if value < 0 else ""

        if currency.upper() == "USD":
            if abs_v >= 1e9:
                return f"{sign}${abs_v / 1e9:,.2f} billion"
            if abs_v >= 1e6:
                return f"{sign}${abs_v / 1e6:,.2f} million"
            return f"{sign}${abs_v:,.2f}"

        # Vietnamese VND
        if unit == "auto":
            if abs_v >= 1e9:
                return f"{sign}{abs_v / 1e9:,.2f} tỷ VND"
            if abs_v >= 1e6:
                return f"{sign}{abs_v / 1e6:,.2f} triệu VND"
            return f"{sign}{abs_v:,.0f} VND"
        return f"{sign}{value:,.2f} {currency}"
