"""Financial GraphRAG Pipeline — public API.

This module is the main entry point for all pipeline operations:

  ingest(path, ...)
      Ingest a PDF or directory of documents (multimodal RAG pipeline).

  ingest_dataset(name, ...)
      Load a benchmark dataset (FinQA / ViFinTab) into vector store + graph.

  build_graph(dataset, ...)
      Build (or update) the Knowledge Graph from a dataset without updating
      the vector store.  Useful for iterating on graph construction separately.

  query(question, ...)
      Answer a question using the GraphRAG hybrid engine.

  graph_stats()
      Return statistics about the current persisted Knowledge Graph.

All graph operations use NetworkX in-memory and persist to
``data/graph/knowledge_graph.json`` (+ .gexf) via GraphStore.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from rich.progress import track

from financial_rag.config.settings import get_settings
from financial_rag.ingestion.dataset_adapters import (
    FinQADatasetLoader,
    SECEdgarDatasetLoader,
    ViFinTabDatasetLoader,
)
from financial_rag.ingestion.document_loader import DocumentLoader
from financial_rag.ingestion.financial_normalizer import (
    flatten_row_headers,
    normalize_table_values,
)
from financial_rag.ingestion.models import DocumentElement, Modality
from financial_rag.reasoning.financial_qa import FinancialQAEngine, GraphRAGQAEngine
from financial_rag.reasoning.multimodal_summarizer import MultimodalSummarizer
from financial_rag.retrieval.chunking import chunk_elements
from financial_rag.retrieval.vector_store import FinancialVectorStore


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalize_table_element(element: DocumentElement) -> DocumentElement:
    """Apply financial normalisation and row-header flattening to table elements."""
    modality_val = (
        element.modality.value
        if hasattr(element.modality, "value")
        else str(element.modality).lower()
    )
    if modality_val != "table":
        return element

    raw = element.raw_content or element.content
    if not raw:
        return element

    normalized_html, scale = normalize_table_values(raw)
    normalized_html = flatten_row_headers(normalized_html)

    new_meta = dict(element.metadata)
    new_meta["scale_factor"] = scale

    return replace(
        element,
        content=normalized_html,
        raw_content=normalized_html,
        metadata=new_meta,
    )


def _get_graph_builder():
    """Return a KnowledgeGraphBuilder loaded from disk (if exists) or fresh."""
    from financial_rag.graph.knowledge_graph import KnowledgeGraphBuilder
    from financial_rag.graph.graph_store import GraphStore

    settings = get_settings()
    builder  = KnowledgeGraphBuilder()
    store    = GraphStore(settings.graph_dir)

    if store.exists():
        store.load(builder)

    return builder, store


# ---------------------------------------------------------------------------
# PDF / Document ingestion (unchanged behaviour)
# ---------------------------------------------------------------------------

def ingest(
    path: Path,
    summarize_multimodal: bool = True,
    max_pages: int | None = None,
) -> int:
    """Ingest PDF documents using the Multi-Vector multimodal pipeline.

    Returns the number of chunks added to the vector store.
    """
    settings = get_settings()
    elements = DocumentLoader(
        image_output_dir=settings.data_dir / "processed" / "images",
        max_pages=max_pages,
    ).load_path(path)

    if not elements:
        return 0

    elements = [_normalize_table_element(el) for el in elements]

    summarized_elements: list[DocumentElement] = []
    if summarize_multimodal and settings.api_key:
        summarizer = MultimodalSummarizer(settings)
        for el in track(elements, description="[bold cyan]Tóm tắt đa phương thức (Gemini)...[/bold cyan]"):
            modality_val = (
                el.modality.value
                if hasattr(el.modality, "value")
                else str(el.modality).lower()
            )
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


# ---------------------------------------------------------------------------
# Dataset ingestion — vector store + Knowledge Graph
# ---------------------------------------------------------------------------

def ingest_dataset(
    name: str,
    split: str = "train",
    sample_size: int | None = None,
    build_kg: bool = True,
    use_llm_extraction: bool = False,
) -> dict[str, int]:
    """Ingest a benchmark dataset into the vector store and Knowledge Graph.

    Parameters
    ----------
    name : str
        Dataset name: ``"finqa"``, ``"vifintab"``, or ``"sec_edgar"``.
    split : str
        Dataset split (``"train"``, ``"validation"``, ``"test"``).
    sample_size : int | None
        Limit the number of records (useful for development).
    build_kg : bool
        If True (default), also build / update the Knowledge Graph.
    use_llm_extraction : bool
        If True, run LLM-based entity extraction on FinQA text passages
        (requires API key; slow but produces richer graph triples).

    Returns
    -------
    dict with keys ``"vector_chunks"`` and ``"graph_nodes"`` / ``"graph_edges"``.
    """
    settings  = get_settings()
    name_clean = name.lower().replace("-", "_")

    if name_clean == "finqa":
        loader   = FinQADatasetLoader()
        elements = loader.load(split=split, sample_size=sample_size)
    elif name_clean in {"vifintab", "vi_fin_tab"}:
        loader   = ViFinTabDatasetLoader()
        elements = loader.load(split=split, sample_size=sample_size)
    elif name_clean in {"sec_edgar", "secedgar"}:
        elements = SECEdgarDatasetLoader().load(sample_size=sample_size)
    else:
        raise ValueError(f"Unknown dataset: {name}. Choose from: finqa, vifintab, sec_edgar")

    # ── Vector store ingestion ────────────────────────────────────────
    chunks       = chunk_elements(elements)
    vector_store = FinancialVectorStore(settings)
    n_chunks     = vector_store.add_chunks(chunks)

    result: dict[str, int] = {"vector_chunks": n_chunks}

    # ── Knowledge Graph construction ──────────────────────────────────
    if build_kg:
        kg_stats = _build_graph_from_elements(
            elements=elements,
            dataset_name=name_clean,
            use_llm_extraction=use_llm_extraction,
            settings=settings,
        )
        result.update(kg_stats)

    return result


# ---------------------------------------------------------------------------
# Graph-only build (no vector store update)
# ---------------------------------------------------------------------------

def build_graph(
    dataset: str = "finqa",
    split: str = "train",
    sample_size: int | None = None,
    use_llm_extraction: bool = False,
    link_bilingual: bool = True,
) -> dict[str, Any]:
    """Build (or rebuild) the Knowledge Graph from a dataset.

    This does NOT update the ChromaDB vector store.

    Parameters
    ----------
    dataset : str
        ``"finqa"``, ``"vifintab"``, or ``"both"``.
    split : str
        Dataset split.
    sample_size : int | None
        Max records per dataset.
    use_llm_extraction : bool
        Run LLM entity extraction on text passages (expensive).
    link_bilingual : bool
        Run BilingualEntityLinker after graph construction.

    Returns
    -------
    dict with graph statistics.
    """
    from financial_rag.graph.knowledge_graph import KnowledgeGraphBuilder
    from financial_rag.graph.graph_store import GraphStore
    from financial_rag.graph.bilingual_linker import BilingualEntityLinker

    settings = get_settings()
    builder  = KnowledgeGraphBuilder()
    store    = GraphStore(settings.graph_dir)

    datasets_to_load: list[str] = []
    name = dataset.lower().replace("-", "_")
    if name == "both":
        datasets_to_load = ["finqa", "vifintab"]
    else:
        datasets_to_load = [name]

    for ds_name in datasets_to_load:
        if ds_name == "finqa":
            elements = FinQADatasetLoader().load(split=split, sample_size=sample_size)
        elif ds_name in {"vifintab", "vi_fin_tab"}:
            elements = ViFinTabDatasetLoader().load(split=split, sample_size=sample_size)
        else:
            continue

        _build_graph_from_elements(
            elements=elements,
            dataset_name=ds_name,
            use_llm_extraction=use_llm_extraction,
            settings=settings,
            builder=builder,
            save=False,         # save once at the end
        )

    # Bilingual linking
    same_as_edges = 0
    if link_bilingual:
        linker = BilingualEntityLinker()
        same_as_edges = linker.link(builder)

    # Persist
    store.save(builder)

    stats = store.stats(builder)
    stats["same_as_edges_added"] = same_as_edges
    return stats


# ---------------------------------------------------------------------------
# Internal graph build helper
# ---------------------------------------------------------------------------

def _build_graph_from_elements(
    elements: list[DocumentElement],
    dataset_name: str,
    use_llm_extraction: bool,
    settings: Any,
    builder: Any = None,
    save: bool = True,
) -> dict[str, int]:
    """Internal: ingest a list of DocumentElements into the KG.

    If ``builder`` is None, loads (or creates) the graph from disk first.
    """
    from financial_rag.graph.knowledge_graph import KnowledgeGraphBuilder
    from financial_rag.graph.graph_store import GraphStore
    from financial_rag.graph.bilingual_linker import BilingualEntityLinker

    if builder is None:
        builder = KnowledgeGraphBuilder()
        store   = GraphStore(settings.graph_dir)
        if store.exists():
            store.load(builder)
    else:
        store = GraphStore(settings.graph_dir)

    # ── Optional LLM entity extraction ───────────────────────────────
    extractor = None
    if use_llm_extraction and settings.api_key:
        from financial_rag.graph.entity_extractor import LLMEntityExtractor
        extractor = LLMEntityExtractor(settings)

    limit = settings.graph_entity_extract_limit
    llm_count = 0

    for idx, element in enumerate(
        track(elements, description=f"[bold green]Building Knowledge Graph ({dataset_name})...[/bold green]")
    ):
        meta    = element.metadata
        dataset = meta.get("dataset", dataset_name)

        if dataset == "finqa":
            # Reconstruct sample dict from metadata + raw content
            table_2d_json = meta.get("table_2d", "")
            table_2d: list[list[str]] = []
            if table_2d_json:
                try:
                    table_2d = json.loads(table_2d_json)
                except Exception:
                    table_2d = []

            sample = {
                "id":        meta.get("id", f"finqa_{idx}"),
                "question":  meta.get("question", ""),
                "answer":    meta.get("answer", ""),
                "pre_text":  [meta.get("pre_text", "")],
                "post_text": [meta.get("post_text", "")],
                "table":     table_2d,
            }
            builder.add_finqa_sample(sample, source="finqa")

            # Optional LLM extraction on pre_text
            if extractor and limit and llm_count < limit:
                pre_text = meta.get("pre_text", "")
                if pre_text:
                    llm_count += extractor.extract_and_add(pre_text, builder, source="finqa_llm")

        elif dataset == "vifintab":
            objects_json = meta.get("objects_json", "{}")
            try:
                objects = json.loads(objects_json)
            except Exception:
                objects = {}

            sample = {
                "image_path": meta.get("image_path", ""),
                "objects":    objects,
            }
            builder.add_vifintab_sample(sample, source="vifintab")

    # Bilingual linking (always run on each incremental build)
    linker = BilingualEntityLinker()
    same_as = linker.link(builder)

    if save:
        store.save(builder)

    stats = builder.stats()
    return {
        "graph_nodes":       stats["total_nodes"],
        "graph_edges":       stats["total_edges"],
        "same_as_edges":     same_as,
        "llm_extractions":   llm_count,
    }


# ---------------------------------------------------------------------------
# Query — GraphRAG hybrid engine
# ---------------------------------------------------------------------------

def query(
    question: str,
    limit: int = 6,
    use_pot: bool = False,
    use_graph: bool | None = None,
) -> dict[str, Any]:
    """Query the GraphRAG engine.

    Parameters
    ----------
    question : str
        The financial question to answer.
    limit : int
        Number of vector context chunks.
    use_pot : bool
        Force Program-of-Thought arithmetic reasoning.
    use_graph : bool | None
        Override ``settings.use_graph_rag``. If None, use the setting value.
    """
    settings     = get_settings()
    vector_store = FinancialVectorStore(settings)

    effective_use_graph = settings.use_graph_rag if use_graph is None else use_graph

    if effective_use_graph:
        builder, _ = _get_graph_builder()
        engine = GraphRAGQAEngine(settings, vector_store, graph_builder=builder)
    else:
        engine = FinancialQAEngine(settings, vector_store)  # type: ignore[assignment]

    return engine.answer(question, limit=limit, use_pot=use_pot)


# ---------------------------------------------------------------------------
# Graph stats utility
# ---------------------------------------------------------------------------

def graph_stats() -> dict[str, Any]:
    """Return statistics about the persisted Knowledge Graph."""
    from financial_rag.graph.graph_store import GraphStore
    from financial_rag.graph.knowledge_graph import KnowledgeGraphBuilder

    settings = get_settings()
    builder  = KnowledgeGraphBuilder()
    store    = GraphStore(settings.graph_dir)

    if not store.exists():
        return {"error": "No graph found. Run build_graph() or ingest_dataset() first."}

    store.load(builder)
    return store.stats(builder)
