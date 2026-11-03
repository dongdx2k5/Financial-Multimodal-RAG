"""Pure scoring/CLI tests; no model downloads, inference, or application data writes."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_ocr.py"
SPEC = importlib.util.spec_from_file_location("ocr_benchmark", SCRIPT)
assert SPEC and SPEC.loader
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def test_scoring_preserves_vietnamese_diacritics_and_numeric_punctuation():
    assert benchmark.score_reference("Lợi nhuận", "Loi nhuan")["cer"] > 0
    for actual in ["1.250.000", "(1,250,000)", "(1.250.800)"]:
        score = benchmark.score_reference("(1.250.000)", actual)
        assert not score["exact_text"]
        assert not score["numeric_sequence_exact"]
    assert benchmark.score_reference("Lợi  nhuận", "Lợi nhuận")["exact_text"]


def test_unicode_normalization_does_not_remove_accents():
    assert benchmark.score_reference("ạ", "a\u0323")["exact_text"]
    assert not benchmark.score_reference("ạ", "a")["exact_text"]


def test_empty_reference_has_no_cer_and_distance_can_exceed_one():
    assert benchmark.score_reference("", "123")["cer"] is None
    assert benchmark.score_reference("a", "abcdef")["cer"] == 5


def test_disagreement_preserves_repeated_numbers_and_has_no_accuracy():
    result = benchmark.compare_without_reference("100 100 (200)", "100 200")
    assert result["left_only_numeric_tokens"] == {"(200)": 1, "100": 1}
    assert result["right_only_numeric_tokens"] == {"200": 1}
    assert result["interpretation"] == "disagreement_only_not_accuracy"
    assert "accuracy" not in result


def test_numeric_sequences_keep_order():
    result = benchmark.score_reference("2026 100 2025 200", "2025 200 2026 100")
    assert not result["numeric_sequence_exact"]


def test_parse_pages_deduplicates_and_supports_ranges():
    assert benchmark.parse_pages("8,9,11-12,8", 20) == [8, 9, 11, 12]


@pytest.mark.parametrize("value", ["0", "21", "3-1", "1,,2", "", "x", "1-2-3"])
def test_invalid_pages_are_rejected(value):
    with pytest.raises(ValueError):
        benchmark.parse_pages(value, 20)


def test_validate_references_rejects_ambiguous_or_invalid_labels():
    valid = {"id": "a", "page": 1, "bbox": [10, 10, 80, 30], "text": "(1.000)", "kind": "number"}
    assert benchmark.validate_references({"samples": [valid]}, 2) == [valid]
    for replacement in [
        {"bbox": [10, 10, 9, 20]},
        {"bbox": [0, 0, float("nan"), 10]},
        {"page": 0},
        {"page": True},
        {"text": ""},
        {"text": "two\nlines"},
        {"text": "abc", "kind": "number"},
    ]:
        with pytest.raises(ValueError):
            benchmark.validate_references({"samples": [{**valid, **replacement}]}, 2)
    with pytest.raises(ValueError):
        benchmark.validate_references({"samples": [valid, valid]}, 2)


def test_reference_summary_discloses_failed_denominator():
    samples = [
        {
            "kind": "number",
            "engines": {"tesseract": {"status": "ok", **benchmark.score_reference("100", "100")}},
        },
        {"kind": "number", "engines": {"tesseract": {"status": "error", "error": "missing"}}},
    ]
    result = benchmark.summarize_references(samples, ["tesseract"])["tesseract"]
    assert result["reference_samples"] == 2
    assert result["evaluated_samples"] == 1
    assert result["failed_samples"] == 1
    assert result["exact_numeric_text_count"] == 1


def test_cli_rejects_vietocr_without_local_paths():
    with pytest.raises(SystemExit):
        benchmark.parse_args(["report.pdf"])


def test_cli_rejects_app_data_as_output():
    with pytest.raises(SystemExit):
        benchmark.parse_args(
            [
                "report.pdf",
                "--engines",
                "tesseract",
                "--output",
                str(benchmark.ROOT / "data" / "index"),
            ]
        )


def test_markdown_never_claims_accuracy_without_gold(tmp_path):
    path = tmp_path / "report.md"
    benchmark.write_markdown({"pdf": "sample.pdf", "pages": [], "references": []}, path)
    assert "không tính độ chính xác" in path.read_text(encoding="utf-8")
