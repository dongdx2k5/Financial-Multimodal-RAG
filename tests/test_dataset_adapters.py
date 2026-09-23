from pathlib import Path

from financial_rag.ingestion.dataset_adapters import (
    FinQADatasetLoader,
    SECEdgarDatasetLoader,
    ViFinTabDatasetLoader,
)
from financial_rag.ingestion.models import DocumentElement
from financial_rag.retrieval.chunking import chunk_elements


import pytest


def test_finqa_loader_sample() -> None:
    loader = FinQADatasetLoader()
    if not (loader.data_dir / "train.parquet").exists():
        pytest.skip("FinQA raw dataset not downloaded.")
    elements = loader.load(split="train", sample_size=3)
    assert len(elements) == 3
    assert elements[0].metadata["dataset"] == "finqa"
    assert "Financial Context ID:" in elements[0].content


def test_vifintab_loader_sample() -> None:
    loader = ViFinTabDatasetLoader()
    if not (loader.data_dir / "train.parquet").exists():
        pytest.skip("ViFinTab raw dataset not downloaded.")
    elements = loader.load(split="train", sample_size=3)
    assert len(elements) == 3
    assert elements[0].metadata["dataset"] == "vifintab"
    assert "Vietnamese Financial Table Document:" in elements[0].content


def test_sec_edgar_loader_sample() -> None:
    loader = SECEdgarDatasetLoader()
    if not loader.data_dir.exists():
        pytest.skip("SEC EDGAR raw dataset not downloaded.")
    elements = loader.load(sample_size=3)
    assert isinstance(elements, list)
    if elements:
        assert elements[0].metadata["dataset"] == "sec_edgar"


def test_table_chunking_preserves_newlines() -> None:
    table_content = "Header 1 | Header 2\nValue A | Value B"
    element = DocumentElement(content=table_content, modality="table", source_path=Path("table.pdf"))
    chunks = chunk_elements([element])
    assert len(chunks) == 1
    assert "\n" in chunks[0].content
