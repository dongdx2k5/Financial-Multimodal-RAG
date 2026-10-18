"""Tests for NumberNormalizer covering Vietnamese and English number formats."""

from financial_rag.parser.number_normalizer import NumberNormalizer


def test_parse_vietnamese_numbers():
    assert NumberNormalizer.parse_number("1.234.567,89") == 1234567.89
    assert NumberNormalizer.parse_number("4.225.864.218.423") == 4225864218423.0
    assert NumberNormalizer.parse_number("123,45") == 123.45


def test_parse_english_numbers():
    assert NumberNormalizer.parse_number("1,234,567.89") == 1234567.89
    assert NumberNormalizer.parse_number("130,497") == 130497.0
    assert NumberNormalizer.parse_number("72.88") == 72.88


def test_parse_negative_numbers():
    assert NumberNormalizer.parse_number("(1.234.567)") == -1234567.0
    assert NumberNormalizer.parse_number("(1,234.5)") == -1234.5
    assert NumberNormalizer.parse_number("-500.000") == -500000.0
    assert NumberNormalizer.parse_number("– 250") == -250.0


def test_parse_blank_or_dash():
    assert NumberNormalizer.parse_number("-") == 0.0
    assert NumberNormalizer.parse_number("–") == 0.0
    assert NumberNormalizer.parse_number("") == 0.0
    assert NumberNormalizer.parse_number(None) is None


def test_normalize_with_scale():
    val, raw = NumberNormalizer.normalize_with_scale("4.225,5", unit_scale=1_000_000)
    assert val == 4225500000.0
    assert raw == "4.225,5"


def test_extract_unit_scale_from_text():
    assert NumberNormalizer.extract_unit_scale_from_text("Đơn vị tính: triệu đồng") == 1_000_000
    assert NumberNormalizer.extract_unit_scale_from_text("Unit: In millions of USD") == 1_000_000
    assert NumberNormalizer.extract_unit_scale_from_text("Đơn vị: tỷ đồng") == 1_000_000_000
    assert NumberNormalizer.extract_unit_scale_from_text("Đơn vị: đồng") == 1
