"""Run real local Ollama calls; save outcomes, including safe fallback, for inspection."""

import argparse
import json
from pathlib import Path

from financial_rag.config import settings
from financial_rag.service import FinancialRAGService


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--document-id", required=True)
    parser.add_argument("--output", default="work/ollama-smoke.json")
    parser.add_argument("--question", help="Run one custom narrative question")
    args = parser.parse_args()
    if settings.llm_provider != "ollama":
        raise SystemExit("Set LLM_PROVIDER=ollama; this script does not call cloud providers.")
    service = FinancialRAGService(settings)
    results = []
    for question in (
        [args.question]
        if args.question
        else [
            "Phân tích ngắn gọn doanh thu thuần trong báo cáo, chỉ nêu số liệu có nguồn.",
            "Tại sao dòng tiền kinh doanh âm? Nếu báo cáo chưa giải thích nguyên nhân thì nói rõ.",
            "Báo cáo quy định khi nào ghi nhận doanh thu bán hàng? Tóm tắt điều kiện, không nêu số.",
        ]
    ):
        print("Running: " + question, flush=True)
        result = service.ask(question, top_k=3, document_id=args.document_id, use_llm=True)
        results.append({"question": question, **result.model_dump()})
        print(
            json.dumps(
                {
                    "mode": result.mode,
                    "seconds": result.llm_elapsed_seconds,
                    "answer": result.answer,
                    "warnings": result.warnings,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    if not any(r["mode"] == "llm" for r in results):
        raise SystemExit("No LLM answer accepted; inspect saved fallback reasons.")


if __name__ == "__main__":
    main()
