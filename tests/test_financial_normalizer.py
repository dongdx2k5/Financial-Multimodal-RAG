"""Tests for financial_normalizer module."""

from financial_rag.ingestion.financial_normalizer import (
    detect_scale_factor,
    flatten_row_headers,
    flatten_row_headers_2d,
    normalize_financial_value,
    normalize_table_2d,
    normalize_table_values,
)


# ---------------------------------------------------------------------------
# normalize_financial_value
# ---------------------------------------------------------------------------

class TestNormalizeFinancialValue:
    def test_parenthesised_negative(self) -> None:
        assert normalize_financial_value("(125.4)") == "-125.4"

    def test_parenthesised_negative_with_currency(self) -> None:
        assert normalize_financial_value("($1,200.50)") == "-1200.50"

    def test_positive_with_dollar(self) -> None:
        assert normalize_financial_value("$5,234.00") == "5234.00"

    def test_euro_symbol(self) -> None:
        assert normalize_financial_value("€1,000") == "1000"

    def test_percent_stripped(self) -> None:
        assert normalize_financial_value("8.82%") == "8.82"

    def test_thousand_comma_removed(self) -> None:
        assert normalize_financial_value("1,200,000") == "1200000"

    def test_plain_number_unchanged(self) -> None:
        assert normalize_financial_value("42.5") == "42.5"

    def test_negative_number(self) -> None:
        assert normalize_financial_value("-300") == "-300"

    def test_label_unchanged(self) -> None:
        assert normalize_financial_value("Revenue") == "Revenue"

    def test_empty_string(self) -> None:
        assert normalize_financial_value("") == ""

    def test_whitespace_only(self) -> None:
        assert normalize_financial_value("   ") == "   "

    def test_date_unchanged(self) -> None:
        assert normalize_financial_value("2019-12-31") == "2019-12-31"


# ---------------------------------------------------------------------------
# detect_scale_factor
# ---------------------------------------------------------------------------

class TestDetectScaleFactor:
    def test_millions(self) -> None:
        assert detect_scale_factor("All amounts in millions of USD") == 1_000_000

    def test_billions(self) -> None:
        assert detect_scale_factor("Revenue (in billions)") == 1_000_000_000

    def test_thousands(self) -> None:
        assert detect_scale_factor("Values are in thousands") == 1_000

    def test_no_scale(self) -> None:
        assert detect_scale_factor("Revenue by segment") == 1

    def test_empty(self) -> None:
        assert detect_scale_factor("") == 1

    def test_vietnamese_ty(self) -> None:
        assert detect_scale_factor("Đơn vị: tỷ đồng") == 1_000_000_000

    def test_vietnamese_trieu(self) -> None:
        assert detect_scale_factor("triệu đồng") == 1_000_000


# ---------------------------------------------------------------------------
# normalize_table_2d
# ---------------------------------------------------------------------------

class TestNormalizeTable2D:
    def test_basic_normalization(self) -> None:
        table = [
            ["Metric", "2018", "2019"],
            ["Revenue", "$1,200", "$1,500"],
            ["Loss", "(100)", "(50)"],
        ]
        result = normalize_table_2d(table)
        assert result[0] == ["Metric", "2018", "2019"]  # header unchanged
        assert result[1] == ["Revenue", "1200", "1500"]
        assert result[2] == ["Loss", "-100", "-50"]

    def test_skip_first_col(self) -> None:
        table = [
            ["Metric", "2018"],
            ["Revenue", "$500"],
        ]
        result = normalize_table_2d(table, skip_first_col=True)
        assert result[1][0] == "Revenue"  # label kept
        assert result[1][1] == "500"

    def test_empty_table(self) -> None:
        assert normalize_table_2d([]) == []


# ---------------------------------------------------------------------------
# normalize_table_values (HTML)
# ---------------------------------------------------------------------------

class TestNormalizeTableValues:
    def test_html_table_normalization(self) -> None:
        html = (
            "<table>"
            "<thead><tr><th>Metric</th><th>Value (in millions)</th></tr></thead>"
            "<tbody><tr><td>Revenue</td><td>$1,234</td></tr>"
            "<tr><td>Loss</td><td>(500)</td></tr></tbody>"
            "</table>"
        )
        result_html, scale = normalize_table_values(html)
        assert scale == 1_000_000
        assert "1234" in result_html
        assert "-500" in result_html

    def test_empty_html(self) -> None:
        result, scale = normalize_table_values("")
        assert result == ""
        assert scale == 1


# ---------------------------------------------------------------------------
# flatten_row_headers_2d
# ---------------------------------------------------------------------------

class TestFlattenRowHeaders2D:
    def test_basic_flattening(self) -> None:
        table = [
            ["Operating Expenses", "", "", ""],
            ["Research", "100", "120", ""],
            ["Marketing", "80", "90", ""],
            ["Net Income", "200", "250", "300"],
        ]
        result = flatten_row_headers_2d(table)
        # Group header row preserved
        assert result[0][0] == "Operating Expenses"
        # Child rows get prefix
        assert result[1][0] == "Operating Expenses > Research"
        assert result[2][0] == "Operating Expenses > Marketing"
        # Row with data still gets prefix (under same group section)
        assert result[3][0] == "Operating Expenses > Net Income"

    def test_no_group_headers(self) -> None:
        table = [
            ["Revenue", "100", "200"],
            ["Expense", "50", "60"],
        ]
        result = flatten_row_headers_2d(table)
        assert result[0][0] == "Revenue"
        assert result[1][0] == "Expense"

    def test_empty_table(self) -> None:
        assert flatten_row_headers_2d([]) == []

    def test_multiple_groups(self) -> None:
        """When a new group header appears, it resets the prefix."""
        table = [
            ["Revenue", "", "", ""],
            ["Product Revenue", "100", "200", ""],
            ["Service Revenue", "50", "60", ""],
            ["Expenses", "", "", ""],
            ["COGS", "30", "40", ""],
        ]
        result = flatten_row_headers_2d(table)
        assert result[1][0] == "Revenue > Product Revenue"
        assert result[2][0] == "Revenue > Service Revenue"
        assert result[3][0] == "Expenses"  # new group header
        assert result[4][0] == "Expenses > COGS"


# ---------------------------------------------------------------------------
# flatten_row_headers (HTML)
# ---------------------------------------------------------------------------

class TestFlattenRowHeadersHTML:
    def test_html_flattening(self) -> None:
        html = (
            "<table><thead><tr><th>Metric</th><th>2019</th><th>2020</th></tr></thead>"
            "<tbody>"
            "<tr><td>Operating Costs</td><td></td><td></td></tr>"
            "<tr><td>R&amp;D</td><td>100</td><td>120</td></tr>"
            "</tbody></table>"
        )
        result = flatten_row_headers(html)
        assert "Operating Costs > R&D" in result
