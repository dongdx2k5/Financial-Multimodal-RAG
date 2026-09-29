"""Optional local regression using the user-provided VNM report, never downloaded."""

import argparse
import json

from financial_rag.config import settings
from financial_rag.service import FinancialRAGService


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--document-id", required=True)
    args = parser.parse_args()
    service = FinancialRAGService(settings)
    cases = [
        ("Dòng tiền kinh doanh quý 1 năm 2026", "-86.395.780.249", "structured", 11),
        ("Dòng tiền đầu tư quý 1 năm 2026", "-637.219.470.794", "structured", 11),
        ("Dòng tiền tài chính quý 1 năm 2026", "1.038.925.762.460", "structured", 12),
        ("Tiền đầu kỳ quý 1 năm 2026", "1.047.628.845.195", "structured", 12),
        ("Tiền cuối kỳ quý 1 năm 2026", "1.362.241.151.028", "structured", 12),
        ("Dòng tiền kinh doanh tăng bao nhiêu so với cùng kỳ", "Nền âm", "evidence", 11),
        ("Doanh thu hoạt động tài chính quý 1 năm 2026", "337.478.602.888", "structured", 10),
        ("Doanh thu bán hàng quý 1 năm 2026", "13.240.394.611.726", "structured", 10),
        ("Doanh thu thuần quý 1 năm 2026", "13.217.639.635.987", "structured", 10),
        ("LNST quý 1 năm 2026", "2.049.247.209.782", "structured", 10),
        ("Doanh thu thuần tăng bao nhiêu so với cùng kỳ?", "26,79", "calculation", 10),
        ("Biên lợi nhuận gộp quý 1 năm 2026", "44,93", "calculation", 10),
        ("Tổng tài sản cuối kỳ 2026", "47.775.826.926.383", "structured", 8),
        ("Tổng tài sản đầu năm 2026", "45.952.496.972.636", "structured", 8),
        ("Vốn chủ sở hữu ngày 31/3/2026", "31.109.254.777.023", "structured", 9),
        ("LNST quý 2 năm 2026", "Không tìm thấy", "evidence", None),
        ("Doanh thu thuần năm 2024", "Không tìm thấy", "evidence", None),
        ("LNST hợp nhất quý 1 năm 2026", "không khớp", "evidence", None),
    ]
    for question, expected, mode, page in cases:
        answer = service.ask(question, document_id=args.document_id)
        assert expected in answer.answer, (question, answer.answer)
        assert answer.mode == mode, question
        if page:
            assert answer.evidence and answer.evidence[0].page == page
            assert answer.evidence[0].bbox and answer.evidence[0].source_url
    print(json.dumps({"passed": len(cases), "document_id": args.document_id}))


if __name__ == "__main__":
    main()
