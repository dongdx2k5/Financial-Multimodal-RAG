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
from rich.progress import track

from financial_rag.ingestion.models import DocumentElement, Modality
from financial_rag.reasoning.financial_qa import FinancialQAEngine
from financial_rag.reasoning.multimodal_summarizer import MultimodalSummarizer
from financial_rag.retrieval.chunking import chunk_elements
from financial_rag.retrieval.vector_store import FinancialVectorStore


def ingest(path: Path, summarize_multimodal: bool = True, max_pages: int | None = None) -> int:
    """Ingests documents/PDFs using the Multi-Vector multimodal pipeline."""
    settings = get_settings()
    elements = DocumentLoader(
        image_output_dir=settings.data_dir / "processed" / "images",
        max_pages=max_pages,
    ).load_path(path)
    
    if not elements:
        return 0

    summarized_elements: list[DocumentElement] = []
    if summarize_multimodal and settings.api_key:
        summarizer = MultimodalSummarizer(settings)
        # Summarize tables & images with Gemini, keep natural text directly to save API quota and time
        for el in track(elements, description="[bold cyan]Bóc tách & Tóm tắt đa phương thức (Gemini)...[/bold cyan]"):
            modality_val = el.modality.value if hasattr(el.modality, "value") else str(el.modality).lower()
            if modality_val == "text":
                summarized_elements.append(replace(el, summary=el.content))
            else:
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
