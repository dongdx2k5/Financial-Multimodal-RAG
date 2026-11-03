"""Kiểm tra bài thực hành TF-IDF; chỉ dùng dữ liệu giả lập và tmp_path."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from financial_rag.graph import KnowledgeGraph
from financial_rag.models import Chunk
from financial_rag.retrieval import HybridRetriever

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "explain_tfidf.py"
SPEC = importlib.util.spec_from_file_location("tfidf_lesson", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
lesson = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lesson)


@pytest.mark.parametrize("mode", ["simplified", "project"])
def test_manual_math_matches_sklearn(mode):
    result = lesson.explain_tfidf(mode=mode)
    lesson.verify_explanation(result)
    assert result["scores"][0] > 0
    np.testing.assert_allclose(result["scores"][1:], [0, 0])
    assert result["sparse_matrix"].shape[0] == 3


def test_simplified_document_frequency_is_not_total_occurrences():
    result = lesson.explain_tfidf()
    columns = {word: index for index, word in enumerate(result["features"])}
    assert result["sparse_matrix"].shape == (3, 7)
    assert result["counts"][0, columns["doanh"]] == 2
    assert result["df"][columns["doanh"]] == 1
    assert result["df"][columns["tang"]] == 3
    assert result["idf"][columns["doanh"]] == pytest.approx(np.log(4 / 2) + 1)
    assert result["idf"][columns["tang"]] == pytest.approx(1)


def test_project_mode_counts_repeated_bigram():
    result = lesson.explain_tfidf(mode="project")
    column = list(result["features"]).index("doanh thu")
    assert result["counts"][0, column] == 2
    assert result["df"][column] == 1
    assert "thu doanh" in result["features"]


@pytest.mark.parametrize("mode", ["simplified", "project"])
def test_oov_query_is_zero_and_does_not_extend_vocabulary(mode):
    result = lesson.explain_tfidf(query="zxqv", mode=mode)
    lesson.verify_explanation(result)
    assert result["sparse_query"].nnz == 0
    assert result["oov_tokens"] == ["zxqv"]
    assert "zxqv" not in result["vectorizer"].vocabulary_
    np.testing.assert_allclose(result["scores"], [0, 0, 0])


def test_query_transform_keeps_corpus_idf():
    first = lesson.explain_tfidf(query="doanh thu")
    second = lesson.explain_tfidf(query="doanh thu doanh thu tang")
    np.testing.assert_array_equal(first["idf"], second["idf"])
    np.testing.assert_array_equal(first["features"], second["features"])
    assert not np.allclose(first["normalized_query"], second["normalized_query"])


def test_project_configuration_matches_real_retriever(tmp_path):
    # Không gọi upsert nên không ghi chunks.json; mọi đường dẫn đều ở tmp_path.
    retriever = HybridRetriever(tmp_path / "chunks.json", KnowledgeGraph(tmp_path / "graph.json"))
    retriever.chunks = [
        Chunk(id=f"c{i}", document_id="demo", source="gia_lap.txt", page=i + 1, text=text)
        for i, text in enumerate(lesson.TOY_CHUNKS)
    ]
    retriever._fit()
    result = lesson.explain_tfidf(mode="project")
    assert retriever.vectorizer is not None
    assert retriever.vectorizer.get_params() == result["vectorizer"].get_params()
    np.testing.assert_array_equal(
        retriever.vectorizer.get_feature_names_out(), result["features"]
    )
    np.testing.assert_allclose(retriever.matrix.toarray(), result["normalized"])
    assert not (tmp_path / "chunks.json").exists()


def test_invalid_mode_has_clear_error():
    with pytest.raises(ValueError, match="mode"):
        lesson.build_vectorizer("unknown")


@pytest.mark.parametrize("mode", ["simplified", "project"])
def test_cli_runs_with_vietnamese_output(mode):
    completed = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--mode", mode, "--verify"],
        capture_output=True,
        encoding="utf-8",
        timeout=60,
        check=True,
    )
    assert "DỮ LIỆU GIẢ LẬP" in completed.stdout
    assert "KIỂM TRA THÀNH CÔNG" in completed.stdout
    assert completed.stderr == ""
