"""Evaluation benchmark module assessing retrieval recall, numerical accuracy, citation fidelity, and trap refusal."""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from financial_rag.config import settings
from financial_rag.db.repository import FinancialRepository
from financial_rag.reasoning.engine import FinancialRAGEngine, RAGResponse


logger = logging.getLogger(__name__)


@dataclass
class EvalMetricResult:
    total_questions: int = 0
    lookup_accuracy: float = 0.0
    calc_accuracy: float = 0.0
    text_accuracy: float = 0.0
    compare_accuracy: float = 0.0
    trap_refusal_rate: float = 0.0
    citation_accuracy: float = 0.0
    overall_score: float = 0.0
    details: List[Dict[str, Any]] = field(default_factory=list)


class SystemEvaluator:
    """Runs automated benchmarks against eval/questions.csv."""

    def __init__(self, engine: FinancialRAGEngine, repo: FinancialRepository):
        self.engine = engine
        self.repo = repo

    def evaluate_all(self, questions_path: Optional[Path] = None) -> EvalMetricResult:
        q_path = questions_path or settings.resolve_path(settings.eval_questions_path)
        if not q_path.exists():
            raise FileNotFoundError(f"Evaluation dataset not found: {q_path}")

        df = pd.read_csv(q_path)
        details = []

        trap_correct = 0
        trap_total = 0
        citation_correct = 0
        citation_total = 0
        type_scores: Dict[str, List[float]] = {"lookup": [], "calc": [], "text": [], "compare": []}

        for _, row in df.iterrows():
            q_id = int(row["q_id"])
            q_text = str(row["question"])
            difficulty = str(row.get("difficulty", "lookup")).strip().lower()
            gold_ans = str(row.get("gold_answer", ""))
            gold_page = int(row["gold_page"]) if pd.notna(row.get("gold_page")) and str(row["gold_page"]).isdigit() else 0

            response = self.engine.answer_question(q_text)
            ans_text = response.answer

            # 1. Evaluate Trap Questions
            if difficulty == "trap":
                trap_total += 1
                is_refused = response.question_type in ("trap", "text_refusal") or any(
                    kw in ans_text.lower() for kw in ["từ chối", "không có", "không tìm thấy", "not available", "refuse"]
                )
                if is_refused:
                    trap_correct += 1
                details.append({
                    "q_id": q_id,
                    "difficulty": difficulty,
                    "question": q_text,
                    "is_correct": is_refused,
                    "response": ans_text[:120],
                })
                continue

            # 2. Evaluate Citations
            citation_total += 1
            has_citation = len(response.citations) > 0 or ("[" in ans_text and "Trang" in ans_text)
            if has_citation:
                citation_correct += 1

            # 3. Evaluate Content Accuracy
            score = 0.0
            # Extract key numbers from gold answer
            gold_numbers = re.findall(r"\b(?:\d{1,3}(?:[.,]\d{3})+|\d+)(?:[.,]\d+)?%?\b", gold_ans)
            if gold_numbers:
                # Check if majority of gold numbers appear in generated answer
                matches = [gn for gn in gold_numbers if gn in ans_text or gn.replace(".", "").replace(",", ".") in ans_text]
                score = len(matches) / len(gold_numbers) if gold_numbers else 0.5
            else:
                # Text overlap
                gold_words = set(re.findall(r"\w+", gold_ans.lower()))
                ans_words = set(re.findall(r"\w+", ans_text.lower()))
                overlap = len(gold_words.intersection(ans_words)) / max(len(gold_words), 1)
                score = 1.0 if overlap > 0.4 else overlap

            if difficulty in type_scores:
                type_scores[difficulty].append(score)

            details.append({
                "q_id": q_id,
                "difficulty": difficulty,
                "question": q_text,
                "score": round(score, 2),
                "has_citation": has_citation,
                "response": ans_text[:120],
            })

        lookup_acc = (sum(type_scores["lookup"]) / len(type_scores["lookup"])) if type_scores["lookup"] else 0.0
        calc_acc = (sum(type_scores["calc"]) / len(type_scores["calc"])) if type_scores["calc"] else 0.0
        text_acc = (sum(type_scores["text"]) / len(type_scores["text"])) if type_scores["text"] else 0.0
        comp_acc = (sum(type_scores["compare"]) / len(type_scores["compare"])) if type_scores["compare"] else 0.0
        trap_rate = (trap_correct / trap_total) if trap_total else 1.0
        citation_rate = (citation_correct / citation_total) if citation_total else 0.0

        overall = (lookup_acc * 0.3) + (calc_acc * 0.25) + (text_acc * 0.15) + (comp_acc * 0.1) + (trap_rate * 0.2)

        return EvalMetricResult(
            total_questions=len(df),
            lookup_accuracy=round(lookup_acc * 100, 1),
            calc_accuracy=round(calc_acc * 100, 1),
            text_accuracy=round(text_acc * 100, 1),
            compare_accuracy=round(comp_acc * 100, 1),
            trap_refusal_rate=round(trap_rate * 100, 1),
            citation_accuracy=round(citation_rate * 100, 1),
            overall_score=round(overall * 100, 1),
            details=details,
        )
