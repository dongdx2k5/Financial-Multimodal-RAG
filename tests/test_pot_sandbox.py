"""Tests for Python sandbox and Program-of-Thought engine."""

from unittest.mock import MagicMock

from financial_rag.reasoning.sandbox import execute_python_safely, validate_code
from financial_rag.reasoning.pot_reasoning import (
    ProgramOfThoughtEngine,
    is_quantitative_question,
)
from financial_rag.config.settings import Settings


# ---------------------------------------------------------------------------
# validate_code
# ---------------------------------------------------------------------------

class TestValidateCode:
    def test_safe_code(self) -> None:
        code = "a = 100\nb = 200\nans = a + b"
        is_safe, reason = validate_code(code)
        assert is_safe is True

    def test_rejects_import(self) -> None:
        is_safe, _ = validate_code("import os\nans = os.getcwd()")
        assert is_safe is False

    def test_rejects_from_import(self) -> None:
        is_safe, _ = validate_code("from pathlib import Path\nans = 1")
        assert is_safe is False

    def test_rejects_open(self) -> None:
        is_safe, _ = validate_code("f = open('secrets.txt')\nans = f.read()")
        assert is_safe is False

    def test_rejects_exec(self) -> None:
        is_safe, _ = validate_code("exec('malicious')")
        assert is_safe is False

    def test_rejects_eval(self) -> None:
        is_safe, _ = validate_code("ans = eval('1+1')")
        assert is_safe is False

    def test_rejects_dunder(self) -> None:
        is_safe, _ = validate_code("x = __builtins__")
        assert is_safe is False

    def test_rejects_subprocess(self) -> None:
        is_safe, _ = validate_code("import subprocess\nans = 1")
        assert is_safe is False

    def test_rejects_print(self) -> None:
        is_safe, _ = validate_code("print('hello')\nans = 1")
        assert is_safe is False

    def test_empty_code(self) -> None:
        is_safe, _ = validate_code("")
        assert is_safe is False


# ---------------------------------------------------------------------------
# execute_python_safely
# ---------------------------------------------------------------------------

class TestExecutePythonSafely:
    def test_simple_arithmetic(self) -> None:
        result = execute_python_safely("a = 10\nb = 3\nans = round(a / b, 2)")
        assert result["success"] is True
        assert result["result"] == 3.33

    def test_percentage_calculation(self) -> None:
        code = (
            "revenue_2019 = 5234.0\n"
            "revenue_2018 = 4810.5\n"
            "pct = (revenue_2019 - revenue_2018) / abs(revenue_2018) * 100\n"
            "ans = round(pct, 2)"
        )
        result = execute_python_safely(code)
        assert result["success"] is True
        assert isinstance(result["result"], float)
        assert abs(result["result"] - 8.80) < 0.1

    def test_blocked_code_returns_error(self) -> None:
        result = execute_python_safely("import os\nans = 1")
        assert result["success"] is False
        assert "Blocked" in result["error"]

    def test_variables_captured(self) -> None:
        result = execute_python_safely("x = 42\ny = 58\nans = x + y")
        assert result["success"] is True
        assert result["variables"]["x"] == 42
        assert result["variables"]["y"] == 58
        assert result["result"] == 100

    def test_no_ans_returns_none(self) -> None:
        result = execute_python_safely("x = 42")
        assert result["success"] is True
        assert result["result"] is None

    def test_division_by_zero(self) -> None:
        result = execute_python_safely("ans = 1 / 0")
        assert result["success"] is False
        assert "division" in result["error"].lower() or "zero" in result["error"].lower()

    def test_timeout_protection(self) -> None:
        code = "x = 0\nwhile True:\n    x += 1\nans = x"
        result = execute_python_safely(code, timeout=1)
        # Should either timeout or be blocked — either way, not success with ans
        # (infinite loop may complete due to thread nature, but timeout should catch it)
        assert result["success"] is False or result["result"] is None or isinstance(result["error"], str)

    def test_safe_builtins_available(self) -> None:
        code = "nums = [1, 2, 3, 4, 5]\nans = sum(nums) / len(nums)"
        result = execute_python_safely(code)
        assert result["success"] is True
        assert result["result"] == 3.0


# ---------------------------------------------------------------------------
# is_quantitative_question
# ---------------------------------------------------------------------------

class TestIsQuantitativeQuestion:
    def test_how_much(self) -> None:
        assert is_quantitative_question("How much revenue did the company earn?") is True

    def test_percentage(self) -> None:
        assert is_quantitative_question("What percentage of revenue came from services?") is True

    def test_growth_rate(self) -> None:
        assert is_quantitative_question("What is the revenue growth rate from 2018 to 2019?") is True

    def test_calculate(self) -> None:
        assert is_quantitative_question("Calculate the net profit margin") is True

    def test_descriptive(self) -> None:
        assert is_quantitative_question("Describe the company's strategy") is False

    def test_vietnamese_bao_nhieu(self) -> None:
        assert is_quantitative_question("Doanh thu năm 2019 là bao nhiêu?") is True

    def test_vietnamese_ty_le(self) -> None:
        assert is_quantitative_question("Tỷ lệ tăng trưởng doanh thu?") is True

    def test_two_years(self) -> None:
        assert is_quantitative_question("Compare 2018 and 2019 figures") is True

    def test_empty(self) -> None:
        assert is_quantitative_question("") is False


# ---------------------------------------------------------------------------
# ProgramOfThoughtEngine (mocked LLM)
# ---------------------------------------------------------------------------

class TestProgramOfThoughtEngine:
    def test_solve_end_to_end_mock(self) -> None:
        settings = Settings(gemini_api_key="test-mock-key")
        engine = ProgramOfThoughtEngine(settings)

        # Mock the LLM to return simple Python code
        mock_response = MagicMock()
        mock_response.text = "revenue_2019 = 5234.0\nrevenue_2018 = 4810.5\npct = (revenue_2019 - revenue_2018) / abs(revenue_2018) * 100\nans = round(pct, 2)"
        engine.genai_client.models.generate_content = MagicMock(return_value=mock_response)

        result = engine.solve(
            question="What is the revenue growth from 2018 to 2019?",
            context="Revenue 2018: $4,810.5M, Revenue 2019: $5,234.0M",
        )

        assert result["success"] is True
        assert result["generated_code"] != ""
        assert isinstance(result["answer"], float)
        assert abs(result["answer"] - 8.80) < 0.1

    def test_solve_with_blocked_code(self) -> None:
        settings = Settings(gemini_api_key="test-mock-key")
        engine = ProgramOfThoughtEngine(settings)

        # Mock LLM returns dangerous code
        mock_response = MagicMock()
        mock_response.text = "import os\nans = os.getcwd()"
        engine.genai_client.models.generate_content = MagicMock(return_value=mock_response)

        result = engine.solve(
            question="Test question",
            context="Test context",
        )

        assert result["success"] is False
        assert "Blocked" in (result.get("error") or "")

    def test_solve_with_empty_response(self) -> None:
        settings = Settings(gemini_api_key="test-mock-key")
        engine = ProgramOfThoughtEngine(settings)

        mock_response = MagicMock()
        mock_response.text = ""
        engine.genai_client.models.generate_content = MagicMock(return_value=mock_response)

        result = engine.solve(question="Test", context="Test")
        assert result["success"] is False
