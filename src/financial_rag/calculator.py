"""Safe mathematical evaluator for financial calculations.

Evaluates arithmetic expressions (e.g. profit margins, YoY growth rates)
using Python's AST without allowing dangerous built-ins or arbitrary code execution.
"""

from __future__ import annotations

import ast
import operator
import re
from typing import Any

OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

CALC_TAG_PATTERN = re.compile(r"<<<CALC:\s*(.+?)\s*>>>", re.IGNORECASE)


class SafeCalculator:
    """AST-based arithmetic evaluator that only allows safe math operations."""

    @classmethod
    def evaluate(cls, expression: str) -> float:
        """Evaluate a mathematical expression safely.

        Only allows numeric constants, unary and binary operators (+, -, *, /, **, %).
        """
        clean_expr = expression.strip()
        # Remove commas in numbers like 1,234.50 -> 1234.50
        clean_expr = re.sub(r"(\d),(\d)", r"\1\2", clean_expr)

        try:
            tree = ast.parse(clean_expr, mode="eval")
        except SyntaxError as exc:
            raise ValueError(f"Invalid math syntax: '{expression}'") from exc

        return cls._eval_node(tree.body)

    @classmethod
    def _eval_node(cls, node: ast.AST) -> float:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError(f"Unsupported constant type: {type(node.value)}")

        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type in OPERATORS:
                return float(OPERATORS[op_type](cls._eval_node(node.operand)))
            raise ValueError(f"Unsupported unary operator: {op_type}")

        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type in OPERATORS:
                left = cls._eval_node(node.left)
                right = cls._eval_node(node.right)
                if op_type is ast.Div and right == 0:
                    raise ZeroDivisionError("Division by zero in calculation")
                return float(OPERATORS[op_type](left, right))
            raise ValueError(f"Unsupported binary operator: {op_type}")

        raise ValueError(f"Unsupported expression element: {type(node).__name__}")


def process_calculations(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Find all <<<CALC: expr>>> tags in text, compute results, and replace them."""
    calculations: list[dict[str, Any]] = []

    def _replace(match: re.Match) -> str:
        expr = match.group(1).strip()
        try:
            val = SafeCalculator.evaluate(expr)
            formatted = f"{val:,.2f}" if abs(val) >= 0.01 else f"{val:.4f}"
            calculations.append(
                {
                    "expression": expr,
                    "result": val,
                    "formatted": formatted,
                }
            )
            return f"{formatted} (tính từ: `{expr}`)"
        except Exception as exc:
            return f"[Lỗi tính toán `{expr}`: {exc}]"

    processed_text = CALC_TAG_PATTERN.sub(_replace, text)
    return processed_text, calculations
