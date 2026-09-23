"""Bilingual Entity Linker — Vietnamese ↔ English financial terms.

When building the Knowledge Graph from both ViFinTab (Vietnamese) and
FinQA (English), the same financial concept appears under different names.
This module creates SAME_AS edges between equivalent nodes so that a
query about "net income" can also surface nodes tagged "lợi nhuận ròng".

The linker works in two passes:
1. **Dictionary-based**: hardcoded bilingual dictionary of ~40 common
   financial terms maps Vietnamese surface forms → canonical English key.
2. **Node-merge scan**: after both datasets are ingested, scan all
   FINANCIAL_METRIC nodes and connect pairs with matching canonical keys.
"""

from __future__ import annotations

from typing import Any

from financial_rag.graph.knowledge_graph import KnowledgeGraphBuilder, RelationType


# ---------------------------------------------------------------------------
# Bilingual dictionary  (Vietnamese → canonical English key)
# ---------------------------------------------------------------------------

BILINGUAL_MAP: dict[str, str] = {
    # Revenue
    "doanh thu":                    "revenue",
    "doanh thu thuần":              "net_revenue",
    "tổng doanh thu":               "total_revenue",
    "doanh thu bán hàng":           "sales_revenue",

    # Profit
    "lợi nhuận ròng":               "net_income",
    "lợi nhuận sau thuế":           "net_income",
    "lợi nhuận thuần":              "net_profit",
    "lợi nhuận gộp":                "gross_profit",
    "lợi nhuận từ hoạt động kinh doanh": "operating_profit",
    "lợi nhuận trước thuế":         "pretax_income",
    "thu nhập ròng":                "net_income",

    # Costs / Expenses
    "giá vốn hàng bán":             "cost_of_goods_sold",
    "chi phí bán hàng":             "selling_expenses",
    "chi phí quản lý":              "general_administrative_expenses",
    "chi phí hoạt động":            "operating_expenses",
    "chi phí lãi vay":              "interest_expense",
    "chi phí khấu hao":             "depreciation",
    "chi phí thuế thu nhập doanh nghiệp": "income_tax_expense",

    # Balance sheet
    "tổng tài sản":                 "total_assets",
    "tài sản ngắn hạn":             "current_assets",
    "tài sản dài hạn":              "non_current_assets",
    "nợ phải trả":                  "total_liabilities",
    "nợ ngắn hạn":                  "current_liabilities",
    "nợ dài hạn":                   "long_term_liabilities",
    "vốn chủ sở hữu":               "shareholders_equity",
    "vốn điều lệ":                  "charter_capital",
    "tiền và tương đương tiền":      "cash_and_equivalents",
    "phải thu ngắn hạn":            "accounts_receivable",
    "hàng tồn kho":                 "inventory",

    # Cash flow
    "dòng tiền":                    "cash_flow",
    "dòng tiền từ hoạt động kinh doanh": "operating_cash_flow",
    "dòng tiền từ hoạt động đầu tư": "investing_cash_flow",
    "dòng tiền từ hoạt động tài chính": "financing_cash_flow",

    # Ratios
    "tỷ suất lợi nhuận":            "profit_margin",
    "tỷ suất sinh lời trên vốn":    "return_on_equity",
    "tỷ suất sinh lời trên tài sản": "return_on_assets",
    "tỷ lệ nợ":                     "debt_ratio",
    "thu nhập trên mỗi cổ phiếu":   "earnings_per_share",
    "cổ tức":                       "dividends",
}

# Reverse: canonical English key → Vietnamese equivalents (for lookup)
_CANONICAL_TO_VN: dict[str, list[str]] = {}
for _vn, _en in BILINGUAL_MAP.items():
    _CANONICAL_TO_VN.setdefault(_en, []).append(_vn)

# English surface form → canonical key
EN_SURFACE_MAP: dict[str, str] = {
    "revenue":                      "revenue",
    "net revenue":                  "net_revenue",
    "total revenue":                "total_revenue",
    "net income":                   "net_income",
    "net profit":                   "net_income",
    "gross profit":                 "gross_profit",
    "operating profit":             "operating_profit",
    "operating income":             "operating_profit",
    "pretax income":                "pretax_income",
    "income before taxes":          "pretax_income",
    "cost of goods sold":           "cost_of_goods_sold",
    "cogs":                         "cost_of_goods_sold",
    "cost of revenue":              "cost_of_goods_sold",
    "selling expenses":             "selling_expenses",
    "operating expenses":           "operating_expenses",
    "interest expense":             "interest_expense",
    "depreciation":                 "depreciation",
    "income tax":                   "income_tax_expense",
    "total assets":                 "total_assets",
    "current assets":               "current_assets",
    "non current assets":           "non_current_assets",
    "total liabilities":            "total_liabilities",
    "current liabilities":          "current_liabilities",
    "long term liabilities":        "long_term_liabilities",
    "shareholders equity":          "shareholders_equity",
    "stockholders equity":          "shareholders_equity",
    "cash and equivalents":         "cash_and_equivalents",
    "accounts receivable":          "accounts_receivable",
    "inventory":                    "inventory",
    "operating cash flow":          "operating_cash_flow",
    "investing cash flow":          "investing_cash_flow",
    "financing cash flow":          "financing_cash_flow",
    "profit margin":                "profit_margin",
    "net margin":                   "profit_margin",
    "return on equity":             "return_on_equity",
    "roe":                          "return_on_equity",
    "return on assets":             "return_on_assets",
    "roa":                          "return_on_assets",
    "earnings per share":           "earnings_per_share",
    "eps":                          "earnings_per_share",
    "dividends":                    "dividends",
    "ebitda":                       "ebitda",
    "ebit":                         "ebit",
}


def get_canonical_key(label: str) -> str | None:
    """Return the canonical (language-neutral) key for a financial term label."""
    normalized = label.lower().strip()
    if normalized in BILINGUAL_MAP:
        return BILINGUAL_MAP[normalized]
    if normalized in EN_SURFACE_MAP:
        return EN_SURFACE_MAP[normalized]
    return None


# ---------------------------------------------------------------------------
# BilingualEntityLinker
# ---------------------------------------------------------------------------

class BilingualEntityLinker:
    """Scans all FINANCIAL_METRIC nodes in the graph and adds SAME_AS edges
    between Vietnamese and English nodes that share the same canonical concept.

    Usage
    -----
    >>> linker = BilingualEntityLinker()
    >>> linker.link(builder)   # adds SAME_AS edges in-place
    """

    def link(self, builder: KnowledgeGraphBuilder) -> int:
        """Add SAME_AS edges between bilingual-equivalent metric nodes.

        Returns the number of new SAME_AS edges added.
        """
        from financial_rag.graph.knowledge_graph import NodeType

        # Group FINANCIAL_METRIC nodes by their canonical key
        canonical_groups: dict[str, list[str]] = {}

        for node_id, attrs in builder.graph.nodes(data=True):
            if attrs.get("type") != NodeType.FINANCIAL_METRIC:
                continue
            label: str = attrs.get("label", node_id)
            key = get_canonical_key(label)
            if key:
                canonical_groups.setdefault(key, []).append(node_id)

        added = 0
        for canonical, node_ids in canonical_groups.items():
            # Connect all pairs within the same canonical group
            for i in range(len(node_ids)):
                for j in range(i + 1, len(node_ids)):
                    src, dst = node_ids[i], node_ids[j]
                    if not builder.graph.has_edge(src, dst):
                        builder.add_edge(
                            src, dst,
                            relation=RelationType.SAME_AS,
                            canonical=canonical,
                        )
                        builder.add_edge(
                            dst, src,
                            relation=RelationType.SAME_AS,
                            canonical=canonical,
                        )
                        added += 2
        return added
