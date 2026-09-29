from decimal import Decimal
from typing import Literal

from pydantic import BaseModel

from .models import FinancialFact


class CalculationRequest(BaseModel):
    operation: Literal["growth", "difference", "ratio"]
    left_fact_id: str
    right_fact_id: str


def calculate(operation: str, left: FinancialFact, right: FinancialFact) -> dict:
    if "needs_review" in {left.quality, right.quality}:
        raise ValueError("Số liệu không khớp kiểm tra kế toán; cần đối chiếu nguồn trước khi tính.")
    if left.scope != right.scope or left.period_kind != right.period_kind:
        raise ValueError("Không trộn phạm vi báo cáo hoặc loại kỳ kế toán.")
    if left.source != right.source or left.ticker != right.ticker:
        raise ValueError("Chọn hai fact trong cùng báo cáo và doanh nghiệp.")
    if not left.unit or left.unit != right.unit:
        raise ValueError("Đơn vị thiếu hoặc khác nhau; cần xác minh trước khi tính.")
    if left.period is None or right.period is None:
        raise ValueError("Thiếu kỳ kế toán.")
    if operation in {"growth", "difference"}:
        left_period = left.period_label or str(left.period)
        right_period = right.period_label or str(right.period)
        if left.period_kind == "quarter" and left_period[-2:] != right_period[-2:]:
            raise ValueError("Tăng trưởng cùng kỳ cần cùng quý.")
        if left.metric.casefold() != right.metric.casefold() or left_period <= right_period:
            raise ValueError("Cần cùng chỉ tiêu; fact bên trái thuộc năm sau fact bên phải.")
    elif (left.period_label or str(left.period)) != (right.period_label or str(right.period)):
        raise ValueError("Tỷ số cần hai chỉ tiêu cùng kỳ.")
    a, b = Decimal(str(left.value)), Decimal(str(right.value))
    if operation != "difference" and b == 0:
        raise ValueError("Không thể chia cho 0.")
    if operation == "growth" and b < 0:
        raise ValueError("Nền âm: không tự diễn giải tỷ lệ tăng trưởng.")
    result = (
        a - b
        if operation == "difference"
        else ((a - b) / b * 100 if operation == "growth" else a / b * 100)
    )
    return {
        "operation": operation,
        "value": str(result),
        "formula": f"{a} - {b}"
        if operation == "difference"
        else (f"({a} - {b}) / {b} * 100" if operation == "growth" else f"{a} / {b} * 100"),
        "unit": left.unit if operation == "difference" else "%",
        "facts": [left.model_dump(), right.model_dump()],
    }
