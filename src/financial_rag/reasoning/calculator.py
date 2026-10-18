"""Deterministic financial math calculations in Python to prevent LLM arithmetic errors."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from financial_rag.parser.number_normalizer import NumberNormalizer


@dataclass
class CalculationResult:
    formula_name: str
    formula_str: str
    result_value: float
    formatted_result: str
    input_facts: List[Dict[str, Any]]
    explanation: str


class FinancialCalculator:
    """Performs financial calculations: profit margins, debt ratios, growth rates, and comparisons."""

    @staticmethod
    def calculate_margin(
        profit_fact: Dict[str, Any],
        revenue_fact: Dict[str, Any],
        margin_type: str = "net",
    ) -> Optional[CalculationResult]:
        """Calculate Gross or Net Profit Margin (%) = (Profit / Revenue) * 100."""
        profit = profit_fact.get("value")
        rev = revenue_fact.get("value")

        if profit is None or rev is None or rev == 0:
            return None

        margin_pct = (profit / rev) * 100.0
        label = "Biên lợi nhuận gộp" if margin_type == "gross" else "Biên lợi nhuận ròng"
        en_label = "Gross margin" if margin_type == "gross" else "Net profit margin"

        p_name = profit_fact.get("line_item") or profit_fact.get("metric_name_vi")
        r_name = revenue_fact.get("line_item") or revenue_fact.get("metric_name_vi")

        return CalculationResult(
            formula_name=f"{margin_type}_margin",
            formula_str=f"({p_name} / {r_name}) * 100%",
            result_value=margin_pct,
            formatted_result=f"{margin_pct:.2f}%",
            input_facts=[profit_fact, revenue_fact],
            explanation=(
                f"{label} ({en_label}): {margin_pct:.2f}% "
                f"(Lợi nhuận: {NumberNormalizer.format_financial(profit, profit_fact.get('currency', 'VND'))} / "
                f"Doanh thu: {NumberNormalizer.format_financial(rev, revenue_fact.get('currency', 'VND'))})"
            ),
        )

    @staticmethod
    def calculate_debt_to_equity(
        liabilities_fact: Dict[str, Any],
        equity_fact: Dict[str, Any],
    ) -> Optional[CalculationResult]:
        """Calculate Debt-to-Equity (D/E) = Total Liabilities / Owner's Equity."""
        liab = liabilities_fact.get("value")
        eq = equity_fact.get("value")

        if liab is None or eq is None or eq == 0:
            return None

        de_ratio = liab / eq
        return CalculationResult(
            formula_name="debt_to_equity",
            formula_str="Nợ phải trả / Vốn chủ sở hữu",
            result_value=de_ratio,
            formatted_result=f"{de_ratio:.2f} lần",
            input_facts=[liabilities_fact, equity_fact],
            explanation=(
                f"Tỷ số Nợ trên Vốn chủ sở hữu (D/E): {de_ratio:.2f} lần "
                f"(Nợ phải trả: {NumberNormalizer.format_financial(liab, liabilities_fact.get('currency', 'VND'))} / "
                f"Vốn CSH: {NumberNormalizer.format_financial(eq, equity_fact.get('currency', 'VND'))})"
            ),
        )

    @staticmethod
    def calculate_yoy_growth(
        current_fact: Dict[str, Any],
        prior_fact: Dict[str, Any],
    ) -> Optional[CalculationResult]:
        """Calculate Year-over-Year Growth (%) = ((Current - Prior) / |Prior|) * 100."""
        cur = current_fact.get("value")
        prior = prior_fact.get("value")

        if cur is None or prior is None or prior == 0:
            return None

        growth_pct = ((cur - prior) / abs(prior)) * 100.0
        sign = "+" if growth_pct > 0 else ""
        item_name = current_fact.get("line_item") or current_fact.get("metric_name_vi")

        return CalculationResult(
            formula_name="yoy_growth",
            formula_str=f"((Kỳ này - Kỳ trước) / |Kỳ trước|) * 100%",
            result_value=growth_pct,
            formatted_result=f"{sign}{growth_pct:.2f}%",
            input_facts=[current_fact, prior_fact],
            explanation=(
                f"Tăng trưởng cùng kỳ của {item_name}: {sign}{growth_pct:.2f}% "
                f"(Kỳ này: {NumberNormalizer.format_financial(cur, current_fact.get('currency', 'VND'))} so với "
                f"Kỳ trước: {NumberNormalizer.format_financial(prior, prior_fact.get('currency', 'VND'))})"
            ),
        )

    @staticmethod
    def compare_companies(
        fact_a: Dict[str, Any],
        fact_b: Dict[str, Any],
    ) -> CalculationResult:
        """Compare a metric between two companies, validating currency and accounting comparability."""
        curr_a = fact_a.get("currency", "VND")
        curr_b = fact_b.get("currency", "USD")
        val_a = fact_a.get("value", 0.0)
        val_b = fact_b.get("value", 0.0)
        ticker_a = fact_a.get("ticker", "CTY_A")
        ticker_b = fact_b.get("ticker", "CTY_B")
        metric_name = fact_a.get("metric_name_vi") or fact_a.get("line_item")

        # Currency check
        if curr_a.upper() != curr_b.upper():
            return CalculationResult(
                formula_name="cross_company_comparison",
                formula_str="Different Currencies",
                result_value=0.0,
                formatted_result="Không thể so sánh số tuyệt đối trực tiếp do khác đơn vị tiền tệ",
                input_facts=[fact_a, fact_b],
                explanation=(
                    f"Cảnh báo kế toán: {ticker_a} ghi nhận bằng {curr_a} "
                    f"({NumberNormalizer.format_financial(val_a, curr_a)}), còn {ticker_b} ghi nhận bằng {curr_b} "
                    f"({NumberNormalizer.format_financial(val_b, curr_b)}). "
                    f"Hệ thống không tự quy đổi tỷ giá hối đoái. Để so sánh cần xem xét tỷ số tương đối (biên lợi nhuận, tỷ lệ nợ)."
                ),
            )

        diff = val_a - val_b
        ratio = (val_a / val_b) if val_b != 0 else 0.0
        return CalculationResult(
            formula_name="cross_company_comparison",
            formula_str=f"{ticker_a} vs {ticker_b}",
            result_value=diff,
            formatted_result=f"Chênh lệch: {NumberNormalizer.format_financial(diff, curr_a)}",
            input_facts=[fact_a, fact_b],
            explanation=(
                f"So sánh {metric_name} cùng đơn vị ({curr_a}): "
                f"{ticker_a} đạt {NumberNormalizer.format_financial(val_a, curr_a)} so với {ticker_b} đạt {NumberNormalizer.format_financial(val_b, curr_b)}. "
                f"Chênh lệch: {NumberNormalizer.format_financial(abs(diff), curr_a)} "
                f"({ticker_a} {'lớn hơn' if diff > 0 else 'nhỏ hơn'} {ticker_b})."
            ),
        )
