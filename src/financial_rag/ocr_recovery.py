"""Read missing cash-flow codes from narrow image bands, without inventing values."""

import pymupdf

from .financial_tables import METRICS, money, spatial_lines, statement_kind


def recover_codes(page, content, tessdata):
    if content.code_recovery_version == 1:
        return
    if content.extraction_method != "ocr" or statement_kind(content.text) != "cashflow":
        content.code_recovery_version = 1
        return
    width = content.width
    codes = METRICS["cashflow"]
    for y, words in spatial_lines(content.words):
        if not content.height * 0.25 < y < content.height * 0.88:
            continue
        numeric = [
            w for w in words if w[0] > width * 0.55 and len(w[4]) > 6 and money(w[4]) is not None
        ]
        if len(numeric) != 2:
            continue
        existing = [
            w
            for w in content.words + content.secondary_words
            if abs((w[1] + w[3]) / 2 - y) <= 4
            and width * 0.43 < (w[0] + w[2]) / 2 < width * 0.5
            and w[4].strip(".") in codes
        ]
        if existing:
            continue
        clip = pymupdf.Rect(width * 0.42, y - 9, width * 0.94, y + 9)
        try:
            pix = page.get_pixmap(matrix=pymupdf.Matrix(4, 4), clip=clip)
            data = pix.pdfocr_tobytes(language="eng", tessdata=tessdata)
            with pymupdf.open(stream=data, filetype="pdf") as cropped:
                candidates = []
                for word in cropped[0].get_text("words"):
                    x = clip.x0 + ((word[0] + word[2]) / 2) / cropped[0].rect.width * clip.width
                    if width * 0.43 < x < width * 0.5 and word[4] in codes:
                        candidates.append(word[4])
                if len(candidates) == 1:
                    content.recovered_codes.append({"y": y, "code": candidates[0]})
        except (RuntimeError, ValueError, OSError):
            content.warnings.append(
                f"Trang {content.page}: không đọc lại được mã dòng tại y={y:.1f}."
            )
    content.code_recovery_version = 1
