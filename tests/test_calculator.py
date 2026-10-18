"""Tests for Python Financial Calculator formulas."""

from financial_rag.reasoning.calculator import FinancialCalculator


def test_gross_and_net_margins():
    rev_fact = {"value": 1000.0, "line_item": "Doanh thu thuần", "currency": "VND"}
    gross_fact = {"value": 250.0, "line_item": "Lợi nhuận gộp", "currency": "VND"}
    net_fact = {"value": 120.0, "line_item": "Lợi nhuận sau thuế", "currency": "VND"}

    gross_res = FinancialCalculator.calculate_margin(gross_fact, rev_fact, "gross")
    assert gross_res is not None
    assert gross_res.result_value == 25.0
    assert gross_res.formatted_result == "25.00%"

    net_res = FinancialCalculator.calculate_margin(net_fact, rev_fact, "net")
    assert net_res is not None
    assert net_res.result_value == 12.0
    assert net_res.formatted_result == "12.00%"


def test_debt_to_equity():
    liab_fact = {"value": 400.0, "currency": "VND"}
    eq_fact = {"value": 500.0, "currency": "VND"}

    de_res = FinancialCalculator.calculate_debt_to_equity(liab_fact, eq_fact)
    assert de_res is not None
    assert de_res.result_value == 0.8
    assert "0.80 lần" in de_res.formatted_result


def test_yoy_growth():
    cur_fact = {"value": 120.0, "line_item": "LNST", "currency": "VND"}
    prior_fact = {"value": 100.0, "line_item": "LNST", "currency": "VND"}

    yoy_res = FinancialCalculator.calculate_yoy_growth(cur_fact, prior_fact)
    assert yoy_res is not None
    assert yoy_res.result_value == 20.0
    assert "+20.00%" in yoy_res.formatted_result


def test_cross_company_different_currencies():
    fact_vnd = {"ticker": "AAA", "value": 4000000000000.0, "currency": "VND"}
    fact_usd = {"ticker": "NVDA", "value": 130000000000.0, "currency": "USD"}

    res = FinancialCalculator.compare_companies(fact_vnd, fact_usd)
    assert "khác đơn vị tiền tệ" in res.formatted_result or "Cảnh báo" in res.explanation
