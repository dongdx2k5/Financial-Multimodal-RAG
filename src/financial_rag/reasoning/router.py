"""Query routing module directing execution to structured lookups, Python math, or hybrid text search."""

from enum import Enum
from financial_rag.extractor.slot_extractor import ExtractedSlots


class ExecutionRoute(str, Enum):
    TRAP = "trap"
    LOOKUP = "lookup"
    CALCULATION = "calculation"
    COMPARISON = "comparison"
    TEXT_EXPLANATION = "text_explanation"


class QueryRouter:
    """Routes financial query based on extracted slots and semantic classification."""

    @staticmethod
    def determine_route(slots: ExtractedSlots) -> ExecutionRoute:
        if slots.question_type == "trap":
            return ExecutionRoute.TRAP

        if slots.question_type == "compare" or slots.secondary_ticker:
            return ExecutionRoute.COMPARISON

        if slots.question_type == "calc" or slots.calc_type:
            return ExecutionRoute.CALCULATION

        if slots.question_type == "lookup" and (slots.metric_code or slots.ticker):
            return ExecutionRoute.LOOKUP

        return ExecutionRoute.TEXT_EXPLANATION
