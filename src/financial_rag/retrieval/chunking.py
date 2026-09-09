from financial_rag.ingestion.models import Chunk, DocumentElement, Modality


def _get_modality_str(modality: str | Modality) -> str:
    if hasattr(modality, "value"):
        return str(modality.value).lower()
    return str(modality).lower()


def chunk_elements(
    elements: list[DocumentElement],
    chunk_size: int = 1200,
    overlap: int = 180,
) -> list[Chunk]:
    chunks: list[Chunk] = []

    for element_index, element in enumerate(elements):
        modality_str = _get_modality_str(element.modality)

        # Image Elements: Atomic chunks with image summary as search content and base64 as raw content
        if modality_str == "image":
            chunk_id = element.element_id or f"{element.source_path.name}:{element.page or 0}:img:{element_index}"
            chunks.append(
                Chunk(
                    content=element.summary or element.content,
                    source_path=element.source_path,
                    chunk_id=chunk_id,
                    page=element.page,
                    modality="image",
                    raw_content=element.raw_content or element.content,
                    metadata=element.metadata,
                )
            )
            continue

        # Table Elements: Preserve structured table (HTML or markdown)
        if modality_str == "table":
            table_raw = element.raw_content or element.content
            table_summary = element.summary or table_raw
            chunk_id = element.element_id or f"{element.source_path.name}:{element.page or 0}:tbl:{element_index}"
            chunks.append(
                Chunk(
                    content=table_summary,
                    source_path=element.source_path,
                    chunk_id=chunk_id,
                    page=element.page,
                    modality="table",
                    raw_content=table_raw,
                    metadata=element.metadata,
                )
            )
            continue

        # Text Elements: Standard sliding window chunking if long
        text_content = element.raw_content or element.content
        cleaned_text = " ".join(text_content.split())
        if not cleaned_text:
            continue

        if len(cleaned_text) <= chunk_size:
            chunk_id = element.element_id or f"{element.source_path.name}:{element.page or 0}:{element_index}:0"
            chunks.append(
                Chunk(
                    content=element.summary or cleaned_text,
                    source_path=element.source_path,
                    chunk_id=chunk_id,
                    page=element.page,
                    modality="text",
                    raw_content=cleaned_text,
                    metadata=element.metadata,
                )
            )
        else:
            step = max(1, chunk_size - overlap)
            for start in range(0, len(cleaned_text), step):
                chunk_slice = cleaned_text[start : start + chunk_size]
                if not chunk_slice:
                    continue
                chunk_id = f"{element.source_path.name}:{element.page or 0}:{element_index}:{start}"
                chunks.append(
                    Chunk(
                        content=chunk_slice,
                        source_path=element.source_path,
                        chunk_id=chunk_id,
                        page=element.page,
                        modality="text",
                        raw_content=chunk_slice,
                        metadata=element.metadata,
                    )
                )

    return chunks
