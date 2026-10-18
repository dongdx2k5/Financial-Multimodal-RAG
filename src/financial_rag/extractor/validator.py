"""Ingestion data validator implementing quality checks from section 5.4."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from financial_rag.db.repository import FinancialRepository


@dataclass
class ValidationReport:
    report_id: int
    is_valid: bool = True
    needs_manual_check: bool = False
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    checked_rules: List[str] = field(default_factory=list)


class FinancialDataValidator:
    """Validates accounting equations and core metrics completeness for ingested financial reports."""

    def __init__(self, repo: FinancialRepository):
        self.repo = repo

    def validate_report(self, report_id: int, tolerance_pct: float = 0.02) -> ValidationReport:
        result = ValidationReport(report_id=report_id)

        # Retrieve all facts for current period of this report
        facts = self.repo.query_facts()
        report_facts = [f for f in facts if f["report_id"] == report_id and f["column_role"] == "current"]
        metric_map = {f["metric_code"]: f["value"] for f in report_facts if f.get("metric_code") and f.get("value") is not None}

        # Rule 1: Balance Sheet Fundamental Equation: Total Assets = Total Liabilities + Equity
        result.checked_rules.append("Assets = Liabilities + Equity")
        total_assets = metric_map.get("total_assets")
        liabilities = metric_map.get("total_liabilities")
        equity = metric_map.get("equity")

        if total_assets is not None and liabilities is not None and equity is not None:
            calc_assets = liabilities + equity
            diff = abs(total_assets - calc_assets)
            allowed_diff = max(abs(total_assets), abs(calc_assets)) * tolerance_pct
            if diff > allowed_diff:
                result.is_valid = False
                result.needs_manual_check = True
                result.errors.append(
                    f"Balance sheet mismatch: Total Assets ({total_assets:,.0f}) != Liabilities + Equity ({calc_assets:,.0f}), diff: {diff:,.0f}"
                )

        # Rule 2: Sub-items sum: Current Assets + Non-current Assets = Total Assets
        result.checked_rules.append("Current Assets + Non-current Assets = Total Assets")
        cur_assets = metric_map.get("current_assets")
        non_cur_assets = metric_map.get("non_current_assets")

        if total_assets is not None and cur_assets is not None and non_cur_assets is not None:
            sum_assets = cur_assets + non_cur_assets
            diff = abs(total_assets - sum_assets)
            allowed_diff = abs(total_assets) * tolerance_pct
            if diff > allowed_diff:
                result.warnings.append(
                    f"Asset sub-items mismatch: Total Assets ({total_assets:,.0f}) vs Sum ({sum_assets:,.0f}), diff: {diff:,.0f}"
                )

        # Rule 3: Common Core Metrics Presence Check
        result.checked_rules.append("Common Core Metrics Completeness")
        core_expected = ["net_revenue", "net_profit", "total_assets", "equity"]
        missing = [m for m in core_expected if m not in metric_map]
        if missing:
            result.warnings.append(f"Missing core standard metrics: {', '.join(missing)}")
            if len(missing) >= 3:
                result.needs_manual_check = True

        return result
