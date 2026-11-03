"""Compare OCR without touching the application's index, uploads, or configuration.

Examples (run from the repository):
    python scripts/benchmark_ocr.py report.pdf --pages 8,9 --engines tesseract
    python scripts/benchmark_ocr.py report.pdf --pages 8,9 --engines tesseract,vietocr \
        --weights work/models/vgg_seq2seq.pth --detector work/models/det.onnx \
        --reference work/ocr-reference.json --output work/ocr-benchmark --render

Reference JSON contains manually transcribed SINGLE-LINE crops, in unrotated PDF
points: {"samples": [{"id": "p8-number", "page": 8, "bbox": [x0,y0,x1,y1],
"text": "(1.250.000)", "kind": "number"}]}.  Crop recognition is scored separately
from whole-page OCR: it does not measure table alignment or document understanding.
Without a reference, differences are disagreements, NOT accuracy measurements.
"""

from __future__ import annotations

# A benchmark must record unexpected third-party failures as errors, not discard the run.
# JSON schema validation deliberately exposes one ValueError contract to the CLI.
# ruff: noqa: BLE001, TRY004
import argparse
import hashlib
import itertools
import json
import math
import re
import sys
import time
import unicodedata
from collections import Counter
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Preserve punctuation and signs. A different comma, dot, minus or bracket is a
# disagreement, not an automatically corrected financial value.
NUMBER_TOKEN = re.compile(r"(?<!\w)(?:\([-+−]?\d[\d.,]*\)|[-+−]?\d[\d.,]*\d|[-+−]?\d)(?!\w)")


def normalize_text(text: str) -> str:
    """Only normalize Unicode and whitespace; preserve case, accents and signs."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def runtime_manifest(model_paths: dict[str, Path | None]) -> dict:
    packages = {}
    for package in ("pymupdf", "torch", "torchvision", "vietocr", "rapidocr", "rapid-table", "onnxruntime"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    models = {}
    for name, path in model_paths.items():
        if path and path.is_file():
            models[name] = {
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
    return {"packages": packages, "models": models}


def numeric_tokens(text: str) -> list[str]:
    return NUMBER_TOKEN.findall(unicodedata.normalize("NFC", text))


def edit_distance(expected: str, actual: str) -> int:
    """Levenshtein distance with linear auxiliary memory."""
    if len(expected) < len(actual):
        expected, actual = actual, expected
    previous = list(range(len(actual) + 1))
    for i, a in enumerate(expected, 1):
        current = [i]
        for j, b in enumerate(actual, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def score_reference(expected: str, actual: str) -> dict:
    expected_normalized, actual_normalized = normalize_text(expected), normalize_text(actual)
    errors = edit_distance(expected_normalized, actual_normalized)
    expected_numbers = numeric_tokens(expected_normalized)
    actual_numbers = numeric_tokens(actual_normalized)
    return {
        "expected": expected,
        "actual": actual,
        "exact_text": expected_normalized == actual_normalized,
        "character_errors": errors,
        "reference_characters": len(expected_normalized),
        "cer": errors / len(expected_normalized) if expected_normalized else None,
        "expected_numeric_tokens": expected_numbers,
        "actual_numeric_tokens": actual_numbers,
        "numeric_sequence_exact": expected_numbers == actual_numbers,
    }


def compare_without_reference(left_text: str, right_text: str) -> dict:
    left, right = Counter(numeric_tokens(left_text)), Counter(numeric_tokens(right_text))
    return {
        "interpretation": "disagreement_only_not_accuracy",
        "left_only_numeric_tokens": dict(sorted((left - right).items())),
        "right_only_numeric_tokens": dict(sorted((right - left).items())),
        "shared_numeric_token_count": sum((left & right).values()),
        "same_normalized_text": normalize_text(left_text) == normalize_text(right_text),
        "warning": "Agreement does not prove correctness; both OCR engines may make the same error.",
    }


def parse_pages(value: str, page_count: int) -> list[int]:
    selected = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            raise ValueError("--pages không được có mục trống")
        if "-" in part:
            bounds = part.split("-")
            if len(bounds) != 2:
                raise ValueError(f"Khoảng trang không hợp lệ: {part}")
            start, end = (int(item) for item in bounds)
            if start > end:
                raise ValueError(f"Khoảng trang đảo ngược: {part}")
            values = range(start, end + 1)
        else:
            values = [int(part)]
        for page in values:
            if not 1 <= page <= page_count:
                raise ValueError(f"Trang {page} ngoài phạm vi 1..{page_count}")
            if page not in selected:
                selected.append(page)
    return selected


def validate_references(raw: object, page_count: int) -> list[dict]:
    if not isinstance(raw, dict) or not isinstance(raw.get("samples"), list):
        raise ValueError("Reference cần object có danh sách samples")
    result, identifiers = [], set()
    for sample in raw["samples"]:
        if not isinstance(sample, dict):
            raise ValueError("Mỗi reference sample phải là object")
        identifier, page = sample.get("id"), sample.get("page")
        if not isinstance(identifier, str) or not identifier or identifier in identifiers:
            raise ValueError("Mỗi reference sample cần id duy nhất, không rỗng")
        if isinstance(page, bool) or not isinstance(page, int) or not 1 <= page <= page_count:
            raise ValueError(f"Reference {identifier}: page không hợp lệ")
        bbox = sample.get("bbox")
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                for v in bbox
            )
            or bbox[0] < 0
            or bbox[1] < 0
            or bbox[2] <= bbox[0]
            or bbox[3] <= bbox[1]
        ):
            raise ValueError(f"Reference {identifier}: bbox không hợp lệ")
        text = sample.get("text")
        if not isinstance(text, str) or not normalize_text(text):
            raise ValueError(f"Reference {identifier}: text nhãn thật không được rỗng")
        if "\n" in text or "\r" in text:
            raise ValueError(f"Reference {identifier}: chỉ chấp nhận crop một dòng")
        kind = sample.get("kind", "text")
        if kind not in {"number", "text"}:
            raise ValueError(f"Reference {identifier}: kind phải là text hoặc number")
        if kind == "number" and not numeric_tokens(text):
            raise ValueError(f"Reference {identifier}: nhãn number cần có số")
        identifiers.add(identifier)
        result.append({"id": identifier, "page": page, "bbox": bbox, "text": text, "kind": kind})
    return result


def tesseract_page(page: pymupdf.Page, dpi: int, language: str, tessdata: str | None) -> dict:
    started = time.perf_counter()
    textpage = page.get_textpage_ocr(language=language, dpi=dpi, full=True, tessdata=tessdata)
    lines = []
    for block in page.get_text("dict", textpage=textpage, sort=True)["blocks"]:
        for line in block.get("lines", []):
            text = "".join(span["text"] for span in line.get("spans", []))
            if text.strip():
                lines.append({"text": text, "bbox": list(line["bbox"]), "confidence": None})
    return {
        "status": "ok",
        "engine": "tesseract",
        "elapsed_seconds": time.perf_counter() - started,
        "text": page.get_text("text", textpage=textpage, sort=True),
        "lines": lines,
        "warnings": [],
    }


def vietocr_page(page: pymupdf.Page, dpi: int, backend) -> dict:
    started = time.perf_counter()
    result = backend.extract(page, dpi=dpi)
    lines = [
        {"text": item.text, "bbox": list(item.bbox), "confidence": item.confidence}
        for item in result.lines
    ]
    return {
        "status": "ok",
        "engine": result.engine,
        "elapsed_seconds": time.perf_counter() - started,
        "text": "\n".join(line["text"] for line in lines),
        "lines": lines,
        "warnings": list(result.warnings),
    }


def reference_image(page: pymupdf.Page, bbox: list[float], dpi: int):
    """Return an upright, padded one-line RGB crop; never save or alter the source PDF."""
    from PIL import Image, ImageOps

    rotation = page.rotation
    try:
        page.set_rotation(0)
        clip = pymupdf.Rect(bbox)
        if not page.rect.contains(clip):
            raise ValueError("Reference bbox nằm ngoài trang PDF")
        pix = page.get_pixmap(dpi=dpi, clip=clip, colorspace=pymupdf.csRGB, alpha=False)
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return ImageOps.expand(image, border=8, fill="white")
    finally:
        page.set_rotation(rotation)


def tesseract_image(image, dpi: int, language: str, tessdata: str | None) -> str:
    pix = pymupdf.Pixmap(pymupdf.csRGB, image.width, image.height, image.tobytes(), False)
    pix.set_dpi(dpi, dpi)
    # An OCR PDF exists only in memory to access PyMuPDF's bundled Tesseract.
    data = pix.pdfocr_tobytes(language=language, tessdata=tessdata)
    with pymupdf.open(stream=data, filetype="pdf") as recognized:
        return recognized[0].get_text("text", sort=True).strip()


def run_reference(sample: dict, page: pymupdf.Page, args, backend) -> dict:
    result = {**sample, "evaluation": "same_single_line_crop", "engines": {}}
    image = reference_image(page, sample["bbox"], args.dpi)
    for engine in args.engine_names:
        started = time.perf_counter()
        try:
            if engine == "tesseract":
                text = tesseract_image(image, args.dpi, args.language, args.tessdata)
            else:
                text, _confidence = backend.recognize_images([image])[0]
            result["engines"][engine] = {
                "status": "ok",
                "elapsed_seconds": time.perf_counter() - started,
                **score_reference(sample["text"], text),
            }
        except Exception as exc:
            result["engines"][engine] = {
                "status": "error",
                "elapsed_seconds": time.perf_counter() - started,
                "error": f"{type(exc).__name__}: {exc}",
            }
    return result


def summarize_references(samples: list[dict], engine_names: list[str]) -> dict:
    summaries = {}
    for engine in engine_names:
        scores = [s["engines"][engine] for s in samples if s["engines"][engine]["status"] == "ok"]
        number_scores = [
            s["engines"][engine]
            for s in samples
            if s["kind"] == "number" and s["engines"][engine]["status"] == "ok"
        ]
        characters = sum(score["reference_characters"] for score in scores)
        summaries[engine] = {
            "reference_samples": len(samples),
            "evaluated_samples": len(scores),
            "failed_samples": len(samples) - len(scores),
            "exact_text_count": sum(score["exact_text"] for score in scores),
            "character_error_rate": (
                sum(score["character_errors"] for score in scores) / characters
                if characters
                else None
            ),
            "numeric_samples_evaluated": len(number_scores),
            "exact_numeric_text_count": sum(score["exact_text"] for score in number_scores),
        }
    return summaries


def write_markdown(report: dict, path: Path) -> None:
    rows = [
        "# Thử nghiệm OCR",
        "",
        f"PDF: `{report['pdf']}`",
        "",
        "Không sửa PDF nguồn, cấu hình hay chỉ mục ứng dụng. Các kết quả là thử nghiệm.",
        "",
        "## Nhận dạng toàn trang",
        "",
        "Thời gian gồm khởi tạo model khi được nạp lần đầu; không dùng để so tốc độ thuần model.",
        "",
        "| Trang | Engine | Trạng thái | Giây | Dòng |",
        "|---|---|---|---:|---:|",
    ]
    for page in report["pages"]:
        for name, engine in page["engines"].items():
            rows.append(
                f"| {page['page']} | {name} | {engine['status']} | "
                f"{engine.get('elapsed_seconds', 0):.3f} | {len(engine.get('lines', []))} |"
            )
            if engine["status"] == "error":
                rows.extend(["", f"Lỗi {name}, trang {page['page']}: `{engine['error']}`", ""])
    rows.extend(
        [
            "",
            "## Đối chiếu số giữa hai engine",
            "",
            (
                "Khác biệt chỉ là bất đồng, KHÔNG cho biết engine nào đúng. "
                "Trùng nhau cũng không chứng minh chính xác. Token gồm cả số trang/ngày tháng."
            ),
            "",
        ]
    )
    for page in report["pages"]:
        for pair in page["comparisons"]:
            rows.append(f"### Trang {page['page']}: {pair['left']} / {pair['right']}")
            for key, label in [
                ("left_only_numeric_tokens", "Chỉ phía trái"),
                ("right_only_numeric_tokens", "Chỉ phía phải"),
            ]:
                rows.append(f"\n{label}: `{json.dumps(pair[key], ensure_ascii=False)}`")
            rows.append("")
    rows.extend(["## Nhãn kiểm chứng thủ công", ""])
    if not report["references"]:
        rows.append("Chưa cung cấp nhãn thật: không tính độ chính xác hoặc kết luận model tốt hơn.")
    else:
        rows.extend(
            [
                (
                    "Các engine đọc cùng ảnh crop một dòng. CER chỉ chuẩn hóa Unicode/khoảng trắng; "
                    "không xóa dấu tiếng Việt, dấu âm hoặc dấu phân cách. "
                    "Đây KHÔNG phải độ chính xác của cấu trúc bảng hoặc pipeline toàn trang."
                ),
                "",
                "| Engine | Đã chấm / Tổng | Đúng cả dòng | CER | Đúng ô số / Đã chấm |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for name, stats in report["reference_summary"].items():
            cer = stats["character_error_rate"]
            cer_text = "N/A" if cer is None else f"{cer:.4f}"
            rows.append(
                f"| {name} | {stats['evaluated_samples']} / {stats['reference_samples']} | "
                f"{stats['exact_text_count']} | {cer_text} | "
                f"{stats['exact_numeric_text_count']} / {stats['numeric_samples_evaluated']} |"
            )
        rows.extend(
            [
                "",
                (
                    "Các mẫu lỗi được ghi riêng, không tính là đã chấm thành công. "
                    "Xem report.json để đối chiếu từng ảnh crop, nhãn và kết quả."
                ),
            ]
        )
    rows.extend(["", "## Cấu trúc bảng (thử nghiệm)", ""])
    table_runs = 0
    for page in report["pages"]:
        for name, engine in page["engines"].items():
            if "table_error" in engine:
                table_runs += 1
                rows.append(f"Trang {page['page']}, {name}: lỗi `{engine['table_error']}`.")
            elif "tables" in engine:
                table_runs += 1
                tables = engine["tables"]
                cells = [cell for table in tables for cell in table["cells"]]
                filled = sum(bool(cell["text"].strip()) for cell in cells)
                rows.append(
                    f"Trang {page['page']}, {name}: {len(tables)} bảng, {len(cells)} ô, "
                    f"{filled} ô ghép được chữ; {engine['table_elapsed_seconds']:.3f} giây."
                )
                for table in tables:
                    rows.extend(f"- {warning}" for warning in table["warnings"])
    if not table_runs:
        rows.append("Chưa chạy SLANet_plus. Dùng --tables và --table-model để thử.")
    else:
        rows.append("\nSố ô ghép được chữ không phải độ chính xác cấu trúc. Xem JSON và PNG để kiểm tra.")
    rows.append("")
    path.write_text("\n".join(rows), encoding="utf-8")


def render_overlay(page: pymupdf.Page, result: dict, target: Path, dpi: int = 120) -> None:
    from PIL import Image, ImageDraw

    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB, alpha=False)
    image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    draw = ImageDraw.Draw(image)
    sx, sy = pix.width / page.rect.width, pix.height / page.rect.height
    for line in result.get("lines", []):
        rect = pymupdf.Rect(line["bbox"]) * page.rotation_matrix
        draw.rectangle(
            (rect.x0 * sx, rect.y0 * sy, rect.x1 * sx, rect.y1 * sy), outline="#e66613", width=2
        )
    image.save(target)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--pages", default="1", help="Trang 1-based, ví dụ 8,9,11-12")
    parser.add_argument("--engines", default="tesseract,vietocr")
    parser.add_argument("--output", type=Path, default=ROOT / "work" / "ocr-benchmark")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--weights", type=Path, help="VietOCR vgg_seq2seq.pth cục bộ")
    parser.add_argument("--detector", type=Path, help="PP-OCR detector ONNX cục bộ")
    parser.add_argument("--tessdata", help="Thư mục vie.traineddata và eng.traineddata")
    parser.add_argument("--language", default="vie+eng")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--tables", action="store_true", help="Thử SLANet_plus trên vùng bảng")
    parser.add_argument("--table-model", type=Path, help="Đường dẫn slanet-plus.onnx cục bộ")
    parser.add_argument("--table-region", help="Một vùng x0,y0,x1,y1; chỉ dùng với một trang")
    parser.add_argument(
        "--render", action="store_true", help="Lưu PNG có bbox để kiểm tra trực quan"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Cho phép ghi lại report trong output"
    )
    args = parser.parse_args(argv)
    args.engine_names = list(dict.fromkeys(name.strip() for name in args.engines.split(",")))
    if not args.engine_names or any(
        name not in {"tesseract", "vietocr"} for name in args.engine_names
    ):
        parser.error("--engines chỉ nhận tesseract,vietocr")
    if not 72 <= args.dpi <= 600:
        parser.error("--dpi phải trong khoảng 72..600")
    if not 1 <= args.cpu_threads <= 32:
        parser.error("--cpu-threads phải trong khoảng 1..32")
    if "vietocr" in args.engine_names and (not args.weights or not args.detector):
        parser.error("VietOCR cần --weights và --detector cục bộ; không tự tải model")
    args.output = args.output.resolve()
    if args.output == ROOT or args.output.is_relative_to(ROOT / "data"):
        parser.error("Hãy ghi benchmark trong work/, không ghi vào dữ liệu/chỉ mục ứng dụng")
    return args


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args(argv)
    if not args.pdf.is_file():
        raise FileNotFoundError(args.pdf)
    for name in ("report.json", "report.md"):
        if (args.output / name).exists() and not args.overwrite:
            raise FileExistsError(f"Đã có {args.output / name}; dùng output khác hoặc --overwrite")
    backend = None
    if "vietocr" in args.engine_names:
        from financial_rag.ocr_backends import VietOCRBackend

        backend = VietOCRBackend(
            weights_path=args.weights,
            detector_path=args.detector,
            device="cpu",
            cpu_threads=args.cpu_threads,
        )
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "pdf": str(args.pdf.resolve()),
        "pdf_sha256": hashlib.sha256(args.pdf.read_bytes()).hexdigest(),
        "dpi": args.dpi,
        "coordinate_space": "unrotated_pdf_points",
        "runtime": runtime_manifest(
            {"vietocr": args.weights, "detector": args.detector, "table": args.table_model}
        ),
        "pages": [],
        "references": [],
        "reference_summary": {},
        "notes": [
            "No reference means no accuracy claim.",
            "Timing includes lazy model startup.",
            "This benchmark does not establish table row/column or period correctness.",
            "Tesseract confidence is not exposed by this PyMuPDF path; null is not zero.",
        ],
    }
    errors = False
    with pymupdf.open(args.pdf) as pdf:
        if pdf.needs_pass:
            raise ValueError("PDF có mật khẩu; hãy dùng bản đã mở khóa")
        selected = parse_pages(args.pages, len(pdf))
        table_regions = None
        if args.table_region:
            if not args.tables or len(selected) != 1:
                raise ValueError("--table-region cần --tables và đúng một trang")
            region = [float(value) for value in args.table_region.split(",")]
            if len(region) != 4 or not all(math.isfinite(value) for value in region):
                raise ValueError("--table-region cần bốn tọa độ hữu hạn")
            table_regions = [region]
        table_engine = None
        if args.tables:
            from financial_rag.table_structure import SLANetTableBackend

            table_engine = SLANetTableBackend(model_path=args.table_model)
        references = []
        if args.reference:
            references = validate_references(
                json.loads(args.reference.read_text(encoding="utf-8")), len(pdf)
            )
            if any(sample["page"] not in selected for sample in references):
                raise ValueError("Mọi trang trong --reference phải nằm trong --pages")
        args.output.mkdir(parents=True, exist_ok=True)
        for number in selected:
            page = pdf[number - 1]
            page_result = {"page": number, "engines": {}, "comparisons": []}
            for engine in args.engine_names:
                print(f"Trang {number}/{len(pdf)} - {engine}...", flush=True)
                started = time.perf_counter()
                try:
                    result = (
                        tesseract_page(page, args.dpi, args.language, args.tessdata)
                        if engine == "tesseract"
                        else vietocr_page(page, args.dpi, backend)
                    )
                except Exception as exc:
                    errors = True
                    result = {
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                page_result["engines"][engine] = result
                if table_engine is not None and result["status"] == "ok":
                    table_started = time.perf_counter()
                    try:
                        result["tables"] = table_engine.extract_tables(
                            page, result["lines"], regions=table_regions
                        )
                        result["table_elapsed_seconds"] = time.perf_counter() - table_started
                        if args.render:
                            for index, table in enumerate(result["tables"]):
                                render_overlay(
                                    page,
                                    {"lines": table["cells"]},
                                    args.output / f"page-{number}-{engine}-table-{index}.png",
                                )
                    except Exception as exc:
                        errors = True
                        result["table_error"] = f"{type(exc).__name__}: {exc}"
                        result["warnings"].append(f"Cấu trúc bảng thất bại: {exc}")
                if args.render and result["status"] == "ok":
                    try:
                        render_overlay(page, result, args.output / f"page-{number}-{engine}.png")
                    except Exception as exc:
                        result["warnings"].append(
                            f"Không render được PNG: {type(exc).__name__}: {exc}"
                        )
            for left, right in itertools.combinations(args.engine_names, 2):
                lresult, rresult = page_result["engines"][left], page_result["engines"][right]
                if lresult["status"] == rresult["status"] == "ok":
                    page_result["comparisons"].append(
                        {
                            "left": left,
                            "right": right,
                            **compare_without_reference(lresult["text"], rresult["text"]),
                        }
                    )
            report["pages"].append(page_result)
            for sample in references:
                if sample["page"] == number:
                    try:
                        result = run_reference(sample, page, args, backend)
                    except Exception as exc:
                        result = {
                            **sample,
                            "evaluation": "same_single_line_crop",
                            "engines": {
                                engine: {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
                                for engine in args.engine_names
                            },
                        }
                    errors = errors or any(r["status"] != "ok" for r in result["engines"].values())
                    report["references"].append(result)
            # Save partial results after each page, including explicit model errors.
            report["reference_summary"] = summarize_references(
                report["references"], args.engine_names
            )
            (args.output / "report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            write_markdown(report, args.output / "report.md")
    print(f"Đã ghi kết quả: {args.output / 'report.md'}", flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
