from dataclasses import replace
from pathlib import Path
from typing import Any

from financial_rag.config.settings import get_settings
from financial_rag.ingestion.dataset_adapters import (
    FinQADatasetLoader,
    SECEdgarDatasetLoader,
    ViFinTabDatasetLoader,
)
from financial_rag.ingestion.document_loader import DocumentLoader
from financial_rag.ingestion.models import DocumentElement
from financial_rag.reasoning.financial_qa import FinancialQAEngine
from financial_rag.reasoning.multimodal_summarizer import MultimodalSummarizer
from financial_rag.retrieval.chunking import chunk_elements
from financial_rag.retrieval.vector_store import FinancialVectorStore


def ingest(path: Path, summarize_multimodal: bool = True) -> int:
    """Ingests documents/PDFs using the Multi-Vector multimodal pipeline."""
    settings = get_settings()
    elements = DocumentLoader(image_output_dir=settings.data_dir / "processed" / "images").load_path(path)
    
    if not elements:
        return 0

    summarized_elements: list[DocumentElement] = []
    if summarize_multimodal and settings.api_key:
        summarizer = MultimodalSummarizer(settings)
        for el in elements:
            summary = summarizer.summarize_element(el)
            summarized_elements.append(replace(el, summary=summary))
    else:
        summarized_elements = elements

    chunks = chunk_elements(summarized_elements)
    vector_store = FinancialVectorStore(settings)
    return vector_store.add_chunks(chunks)


def ingest_dataset(name: str, split: str = "train", sample_size: int | None = None) -> int:
    """Ingests standard financial benchmark datasets."""
    settings = get_settings()
    name_clean = name.lower().replace("-", "_")

    if name_clean == "finqa":
        elements = FinQADatasetLoader().load(split=split, sample_size=sample_size)
    elif name_clean in {"vifintab", "vi_fin_tab"}:
        elements = ViFinTabDatasetLoader().load(split=split, sample_size=sample_size)
    elif name_clean in {"sec_edgar", "secedgar"}:
        elements = SECEdgarDatasetLoader().load(sample_size=sample_size)
    else:
        raise ValueError(f"Unknown dataset: {name}. Choose from: finqa, vifintab, sec_edgar")

    chunks = chunk_elements(elements)
    vector_store = FinancialVectorStore(settings)
    return vector_store.add_chunks(chunks)


def query(question: str, limit: int = 6) -> dict[str, Any]:
    """Queries the multimodal RAG engine and returns synthesized answer with visual & tabular evidence."""
    settings = get_settings()
    vector_store = FinancialVectorStore(settings)
    qa_engine = FinancialQAEngine(settings, vector_store)
    return qa_engine.answer(question, limit=limit)
