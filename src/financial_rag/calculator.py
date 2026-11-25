from decimal import Decimal
from typing import Literal

from pydantic import BaseModel

from .models import FinancialFact

CURRENCY_SCALES = {
    "vnd": Decimal(1),
    "đồng": Decimal(1),
    "dong": Decimal(1),
    "nghìn": Decimal(1000),
    "nghin": Decimal(1000),
    "nghìn đồng": Decimal(1000),
    "triệu": Decimal(1000000),
    "trieu": Decimal(1000000),
    "triệu đồng": Decimal(1000000),
    "tỷ": Decimal(1000000000),
    "ty": Decimal(1000000000),
    "tỷ đồng": Decimal(1000000000),
    "tỷ vnd": Decimal(1000000000),
    "nghìn tỷ": Decimal(1000000000000),
    "nghin ty": Decimal(1000000000000),
    "nghìn tỷ đồng": Decimal(1000000000000),
}


class CalculationRequest(BaseModel):
    operation: Literal["growth", "difference", "ratio"]
    left_fact_id: str
    right_fact_id: str


def _unit_scale(unit: str | None) -> Decimal | None:
    if not unit:
        return None
    return CURRENCY_SCALES.get(" ".join(unit.casefold().split()))


def calculate(
    operation: str,
    left: FinancialFact,
    right: FinancialFact,
    *,
    allow_cross_document: bool = False,
) -> dict:
    if operation not in {"growth", "difference", "ratio"}:
        raise ValueError("Phép tính không được hỗ trợ.")
    if "needs_review" in {left.quality, right.quality}:
        raise ValueError("Số liệu không khớp kiểm tra kế toán; cần đối chiếu nguồn trước khi tính.")
    if left.scope != right.scope or left.period_kind != right.period_kind:
        raise ValueError("Không trộn phạm vi báo cáo hoặc loại kỳ kế toán.")
    if not allow_cross_document and (left.source != right.source or left.ticker != right.ticker):
        raise ValueError("Chọn hai fact trong cùng báo cáo và doanh nghiệp.")
    if not left.unit or not right.unit:
        raise ValueError("Đơn vị thiếu hoặc khác nhau; cần xác minh trước khi tính.")
    left_scale = _unit_scale(left.unit)
    right_scale = _unit_scale(right.unit)
    if left.unit == right.unit:
        scale_left = Decimal(1)
        scale_right = Decimal(1)
        diff_unit = left.unit
    elif left_scale is not None and right_scale is not None:
        scale_left = left_scale
        scale_right = right_scale
        diff_unit = left.unit
    else:
        raise ValueError("Đơn vị thiếu hoặc khác nhau; cần xác minh trước khi tính.")
    if left.period is None or right.period is None:
        raise ValueError("Thiếu kỳ kế toán.")
    if operation in {"growth", "difference"}:
        left_period = left.period_label or str(left.period)
        right_period = right.period_label or str(right.period)
        if left.period_kind in {"quarter", "half_year", "nine_months"} and (
            left_period.split("-", 1)[-1] != right_period.split("-", 1)[-1]
        ):
            raise ValueError("Tăng trưởng cùng kỳ cần cùng quý.")
        if left.metric.casefold() != right.metric.casefold() or left_period <= right_period:
            raise ValueError("Cần cùng chỉ tiêu; fact bên trái thuộc năm sau fact bên phải.")
    elif (left.period_label or str(left.period)) != (right.period_label or str(right.period)):
        raise ValueError("Tỷ số cần hai chỉ tiêu cùng kỳ.")
    raw_a, raw_b = Decimal(str(left.value)), Decimal(str(right.value))
    if not raw_a.is_finite() or not raw_b.is_finite():
        raise ValueError("Số liệu không hữu hạn; cần đối chiếu nguồn.")
    a, b = raw_a * scale_left, raw_b * scale_right
    if operation != "difference" and b == 0:
        raise ValueError("Không thể chia cho 0.")
    if operation == "growth" and b < 0:
        raise ValueError("Nền âm: không tự diễn giải tỷ lệ tăng trưởng.")
    if operation == "difference":
        result = (a - b) / scale_left
        formula = (
            f"{raw_a} - {raw_b}"
            if scale_left == scale_right
            else f"({a} - {b}) / {scale_left}"
        )
    elif operation == "growth":
        result = (a - b) / b * 100
        formula = f"({a} - {b}) / {b} * 100"
    else:
        result = a / b * 100
        formula = f"{a} / {b} * 100"
    return {
        "operation": operation,
        "value": str(result),
        "formula": formula,
        "unit": diff_unit if operation == "difference" else "%",
        "facts": [left.model_dump(), right.model_dump()],
    }
