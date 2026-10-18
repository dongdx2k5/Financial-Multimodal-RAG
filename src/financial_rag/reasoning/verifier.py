"""Numerical verification module ensuring LLM responses match grounded facts."""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from financial_rag.parser.number_normalizer import NumberNormalizer


@dataclass
class VerificationOutcome:
    is_verified: bool
    grounded_numbers: List[float] = field(default_factory=list)
    hallucinated_numbers: List[str] = field(default_factory=list)
    warning_message: Optional[str] = None


class NumericalVerifier:
    """Verifies that numbers present in LLM responses originate from grounded evidence."""

    @classmethod
    def extract_numbers_from_text(cls, text: str) -> List[str]:
        """Extract candidate financial number tokens from natural language text."""
        # Find tokens like 1.234,56 or 123.4 or 15%
        tokens = re.findall(r"\b(?:\d{1,3}(?:[.,]\d{3})+|\d+)(?:[.,]\d+)?%?\b", text)
        return tokens

    @classmethod
    def verify(
        cls,
        llm_answer: str,
        grounded_facts: List[Dict[str, Any]],
        calc_result: Optional[Any] = None,
    ) -> VerificationOutcome:
        # Build set of valid numbers from facts and calculation results
        valid_numbers: Set[float] = set()

        for f in grounded_facts:
            v = f.get("value")
            if v is not None:
                valid_numbers.add(round(float(v), 2))
                # Also scaled variations (e.g. in billions or millions)
                valid_numbers.add(round(float(v) / 1e6, 2))
                valid_numbers.add(round(float(v) / 1e9, 2))
                valid_numbers.add(round(float(v) / 1e3, 2))
                # Page numbers and year
                if f.get("page"):
                    valid_numbers.add(float(f["page"]))
                if f.get("fiscal_year"):
                    valid_numbers.add(float(f["fiscal_year"]))

        if calc_result is not None:
            res_val = getattr(calc_result, "result_value", None)
            if res_val is not None:
                valid_numbers.add(round(float(res_val), 2))
                valid_numbers.add(round(float(res_val), 1))
                valid_numbers.add(round(float(res_val) / 1e6, 2))
                valid_numbers.add(round(float(res_val) / 1e9, 2))
                valid_numbers.add(round(float(res_val) / 1e3, 2))

        # Check numbers found in LLM answer
        found_tokens = cls.extract_numbers_from_text(llm_answer)
        hallucinated: List[str] = []
        grounded: List[float] = []

        for token in found_tokens:
            parsed = NumberNormalizer.parse_number(token)
            if parsed is None:
                continue
            rounded = round(parsed, 2)
            # Skip common non-metric integers (e.g., year 2024..2026, page numbers <= 300, single digits 0..10)
            if rounded in (2024.0, 2025.0, 2026.0) or 0 <= rounded <= 10:
                continue

            # Check if close to any valid number
            matched = any(abs(rounded - vn) < 0.15 for vn in valid_numbers)
            if matched:
                grounded.append(rounded)
            else:
                hallucinated.append(token)

        if hallucinated:
            warning = f"Cảnh báo: Phát hiện con số '{', '.join(hallucinated)}' trong câu trả lời chưa được đối soát trực tiếp từ tài liệu gốc."
            return VerificationOutcome(
                is_verified=False,
                grounded_numbers=grounded,
                hallucinated_numbers=hallucinated,
                warning_message=warning,
            )

        return VerificationOutcome(
            is_verified=True,
            grounded_numbers=grounded,
            hallucinated_numbers=[],
            warning_message=None,
        )
