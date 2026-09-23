"""Dataset adapters for FinQA and ViFinTab.

Loads pre-downloaded Parquet files (cached locally via hf_downloader.py)
and converts each record into DocumentElement objects for the RAG pipeline
and KnowledgeGraphBuilder for the GraphRAG pipeline.

FinQA  (dreamerdeo/finqa)
  Fields : id, pre_text, post_text, table, question, answer
  Purpose: English financial Q&A with structured table + narrative text.

ViFinTab  (ngnquan/ViFinTab)
  Fields : image_path, image_base64 (optional), objects (YOLO OBB bboxes)
  Purpose: Vietnamese table-detection dataset — images of financial reports
           with bounding-box annotations for 3 table types:
             0 = Income Statement (Báo cáo kết quả kinh doanh)
             1 = Balance Sheet (Bảng cân đối kế toán)
             2 = Cash Flow Statement (Báo cáo lưu chuyển tiền tệ)
"""

import json
from pathlib import Path
from typing import Any

import pandas as pd

from financial_rag.ingestion.financial_normalizer import (
    normalize_financial_value,
    normalize_table_2d,
)
from financial_rag.ingestion.models import DocumentElement


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_text(val: Any) -> str:
    if val is None:
        return ""
    if hasattr(val, "tolist"):
        val = val.tolist()
    if isinstance(val, (list, tuple)):
        return "\n".join(str(item) for item in val if item is not None)
    return str(val)


def _format_table(table_obj: Any) -> str:
    if table_obj is None:
        return ""
    if hasattr(table_obj, "tolist"):
        table_obj = table_obj.tolist()
    if isinstance(table_obj, (list, tuple)):
        rows = []
        for r in table_obj:
            if hasattr(r, "tolist"):
                r = r.tolist()
            if isinstance(r, (list, tuple)):
                rows.append(" | ".join(str(c) for c in r if c is not None))
            else:
                rows.append(str(r))
        return "\n".join(rows)
    return str(table_obj)


def _parse_table_2d(raw_table: Any) -> list[list[str]]:
    """Convert any table representation into a list-of-lists of str."""
    if raw_table is None:
        return []
    if hasattr(raw_table, "tolist"):
        raw_table = raw_table.tolist()
    if not isinstance(raw_table, (list, tuple)):
        return []
    rows: list[list[str]] = []
    for r in raw_table:
        if hasattr(r, "tolist"):
            r = r.tolist()
        if isinstance(r, (list, tuple)):
            rows.append([str(c) if c is not None else "" for c in r])
        else:
            rows.append([str(r)])
    return rows


# ---------------------------------------------------------------------------
# ViFinTab table-class labels
# ---------------------------------------------------------------------------

_VIFINTAB_CLASS_NAMES: dict[int, str] = {
    0: "Báo cáo kết quả kinh doanh (Income Statement)",
    1: "Bảng cân đối kế toán (Balance Sheet)",
    2: "Báo cáo lưu chuyển tiền tệ (Cash Flow Statement)",
}


# ---------------------------------------------------------------------------
# FinQA Dataset Loader
# ---------------------------------------------------------------------------

class FinQADatasetLoader:
    """Load FinQA dataset from local Parquet cache.

    Parameters
    ----------
    data_dir : Path | None
        Directory containing ``{split}.parquet`` files.
        Defaults to ``data/raw/datasets/huggingface/finqa_parquet``.
    """

    def __init__(self, data_dir: Path | None = None) -> None:
        self.data_dir = data_dir or Path("data/raw/datasets/huggingface/finqa_parquet")

    # ------------------------------------------------------------------ #
    # Load from local Parquet
    # ------------------------------------------------------------------ #

    def load(
        self,
        split: str = "train",
        sample_size: int | None = None,
    ) -> list[DocumentElement]:
        """Load a split from local Parquet and return DocumentElement list."""
        file_path = self.data_dir / f"{split}.parquet"
        if not file_path.exists():
            raise FileNotFoundError(
                f"FinQA dataset file not found: {file_path}\n"
                f"Run: python -m financial_rag.ingestion.hf_downloader --dataset finqa"
            )

        df = pd.read_parquet(file_path)
        if sample_size:
            df = df.head(sample_size)

        return self._convert_dataframe(df, split, file_path)

    # ------------------------------------------------------------------ #
    # Load directly from HuggingFace Hub
    # ------------------------------------------------------------------ #

    def load_from_hf(
        self,
        split: str = "train",
        sample_size: int | None = None,
    ) -> list[DocumentElement]:
        """Download split from HuggingFace Hub on-demand (no local cache required)."""
        try:
            from datasets import load_dataset  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError("Install 'datasets': pip install datasets")

        ds = load_dataset("dreamerdeo/finqa", split=split)
        if sample_size:
            ds = ds.select(range(min(sample_size, len(ds))))
        df = ds.to_pandas()
        return self._convert_dataframe(df, split, Path(f"hf://dreamerdeo/finqa/{split}"))

    # ------------------------------------------------------------------ #
    # Conversion logic
    # ------------------------------------------------------------------ #

    def _convert_dataframe(
        self,
        df: pd.DataFrame,
        split: str,
        source_path: Path,
    ) -> list[DocumentElement]:
        """Convert a FinQA DataFrame to DocumentElement list."""
        elements: list[DocumentElement] = []

        for index, row in df.iterrows():
            item_id  = str(row.get("id", f"finqa_{index}"))
            question = str(row.get("question", ""))
            answer   = str(row.get("answer", ""))

            pre_text  = _to_text(row.get("pre_text"))
            post_text = _to_text(row.get("post_text"))

            # ── Parse and normalise table ─────────────────────────────
            raw_table = row.get("table")
            table_2d  = _parse_table_2d(raw_table)
            if table_2d:
                normalized = normalize_table_2d(table_2d)
                table_str  = "\n".join(" | ".join(c for c in r) for r in normalized)
            else:
                table_str = _format_table(raw_table)

            # ── Additional metadata fields ────────────────────────────
            program  = _to_text(row.get("program"))
            exe_ans  = row.get("exe_ans")
            if exe_ans is not None:
                if hasattr(exe_ans, "item"):
                    exe_ans = exe_ans.item()
                exe_ans = str(exe_ans)

            # ── Build Hybrid Fusion text block ────────────────────────
            content_parts = [
                f"Financial Context ID: {item_id}",
                f"Pre-Text:\n{pre_text}" if pre_text else "",
                f"Financial Table:\n{table_str}" if table_str else "",
                f"Post-Text:\n{post_text}" if post_text else "",
                f"Question: {question}",
                f"Answer / Formula: {answer}",
            ]
            full_content = "\n\n".join(p for p in content_parts if p)

            metadata: dict[str, Any] = {
                "dataset":   "finqa",
                "split":     split,
                "id":        item_id,
                "question":  question,
                "answer":    answer,
                # Store raw table for graph builder
                "table_2d":  json.dumps(table_2d) if table_2d else "",
                "pre_text":  pre_text,
                "post_text": post_text,
            }
            if program:
                metadata["program"] = program
            if exe_ans is not None:
                metadata["exe_ans"] = exe_ans

            elements.append(
                DocumentElement(
                    content=full_content,
                    modality="finqa_qa",
                    source_path=source_path,
                    metadata=metadata,
                )
            )

        return elements


# ---------------------------------------------------------------------------
# ViFinTab Dataset Loader
# ---------------------------------------------------------------------------

class ViFinTabDatasetLoader:
    """Load ViFinTab (table-detection) dataset from local Parquet cache.

    ViFinTab is a Vietnamese financial table *detection* dataset.
    Each row contains:
      - ``image_path``   : str  — original file path in the source corpus
      - ``image_base64`` : str  — base64-encoded PNG/JPEG (may not be present
                                   if downloaded without --include-images)
      - ``objects``      : dict — {"class_id": [int, ...], "coordinates": [[float, ...], ...]}

    The loader creates a DocumentElement whose content describes the
    detected table regions (type + bounding-box count) so that the
    graph builder and vector store can index it meaningfully.

    Parameters
    ----------
    data_dir : Path | None
        Directory containing ``{split}.parquet`` files.
        Defaults to ``data/raw/datasets/huggingface/ViFinTab_parquet``.
    """

    def __init__(self, data_dir: Path | None = None) -> None:
        self.data_dir = data_dir or Path("data/raw/datasets/huggingface/ViFinTab_parquet")

    # ------------------------------------------------------------------ #
    # Load from local Parquet
    # ------------------------------------------------------------------ #

    def load(
        self,
        split: str = "train",
        sample_size: int | None = None,
    ) -> list[DocumentElement]:
        """Load a split from local Parquet and return DocumentElement list."""
        file_path = self.data_dir / f"{split}.parquet"
        if not file_path.exists():
            raise FileNotFoundError(
                f"ViFinTab dataset file not found: {file_path}\n"
                f"Run: python -m financial_rag.ingestion.hf_downloader --dataset vifintab"
            )

        df = pd.read_parquet(file_path)
        if sample_size:
            df = df.head(sample_size)

        return self._convert_dataframe(df, split, file_path)

    # ------------------------------------------------------------------ #
    # Load directly from HuggingFace Hub
    # ------------------------------------------------------------------ #

    def load_from_hf(
        self,
        split: str = "train",
        sample_size: int | None = None,
    ) -> list[DocumentElement]:
        """Download split from HuggingFace Hub on-demand."""
        try:
            from datasets import load_dataset  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError("Install 'datasets': pip install datasets")

        ds = load_dataset("ngnquan/ViFinTab", split=split)
        if sample_size:
            ds = ds.select(range(min(sample_size, len(ds))))
        df = ds.to_pandas()
        # Drop heavy image column if present
        if "image_base64" in df.columns:
            df = df.drop(columns=["image_base64"])
        return self._convert_dataframe(df, split, Path(f"hf://ngnquan/ViFinTab/{split}"))

    # ------------------------------------------------------------------ #
    # Conversion logic
    # ------------------------------------------------------------------ #

    def _convert_dataframe(
        self,
        df: pd.DataFrame,
        split: str,
        source_path: Path,
    ) -> list[DocumentElement]:
        """Convert a ViFinTab DataFrame to DocumentElement list."""
        elements: list[DocumentElement] = []

        for index, row in df.iterrows():
            img_path: str = str(row.get("image_path", f"vifintab_{index}.png"))
            doc_name: str = img_path.split("/")[-1].split("\\")[-1]

            objects = row.get("objects")
            if objects is None:
                objects = {}

            # ── Parse YOLO OBB annotations ───────────────────────────
            class_ids: list[int]         = []
            coordinates: list[list[float]] = []

            if isinstance(objects, dict):
                raw_cls   = objects.get("class_id", [])
                raw_coords = objects.get("coordinates", [])
                if hasattr(raw_cls, "tolist"):
                    raw_cls = raw_cls.tolist()
                if hasattr(raw_coords, "tolist"):
                    raw_coords = raw_coords.tolist()
                class_ids   = [int(c) for c in raw_cls]
                coordinates = [
                    (c.tolist() if hasattr(c, "tolist") else list(c))
                    for c in raw_coords
                ]

            # ── Build human-readable description ─────────────────────
            region_descriptions: list[str] = []
            for i, cls_id in enumerate(class_ids):
                table_type = _VIFINTAB_CLASS_NAMES.get(cls_id, f"Unknown Table Type ({cls_id})")
                coords_str = ""
                if i < len(coordinates):
                    coords_str = f" [bbox: {coordinates[i]}]"
                region_descriptions.append(f"  - Region {i}: {table_type}{coords_str}")

            table_count = len(class_ids)
            regions_text = "\n".join(region_descriptions) if region_descriptions else "  No tables detected."

            content = (
                f"Vietnamese Financial Report Document: {doc_name}\n"
                f"Image Path: {img_path}\n"
                f"Detected Table Regions: {table_count}\n"
                f"{regions_text}"
            )

            # Store objects as JSON string for graph builder access
            objects_json = json.dumps({
                "class_id":    class_ids,
                "coordinates": coordinates,
            }, ensure_ascii=False)

            elements.append(
                DocumentElement(
                    content=content,
                    modality="vifintab_table",
                    source_path=source_path,
                    metadata={
                        "dataset":      "vifintab",
                        "split":        split,
                        "image_path":   img_path,
                        "table_count":  table_count,
                        "objects_json": objects_json,
                        "language":     "vi",
                    },
                )
            )

        return elements


# ---------------------------------------------------------------------------
# SEC EDGAR Loader (unchanged from original)
# ---------------------------------------------------------------------------

class SECEdgarDatasetLoader:
    def __init__(self, data_dir: Path | None = None):
        cache_path = Path.home() / ".cache/kagglehub/datasets/jamesglang/sec-edgar-company-facts-september2023/versions/6"
        local_path = Path("data/raw/datasets/kaggle/sec-edgar-company-facts-september2023")
        self.data_dir = data_dir or (cache_path if cache_path.exists() else local_path)

    def load(self, sample_size: int | None = 20) -> list[DocumentElement]:
        if not self.data_dir.exists():
            raise FileNotFoundError(f"SEC EDGAR dataset directory not found: {self.data_dir}")

        csv_file = self.data_dir / "companyfacts.csv"
        if csv_file.exists():
            df = pd.read_csv(csv_file, nrows=sample_size or 50)
            elements: list[DocumentElement] = []
            for index, row in df.iterrows():
                cik     = str(row.get("cik", ""))
                company = str(row.get("entityName", ""))
                fact    = str(row.get("companyFact", ""))
                val     = str(row.get("val", ""))
                units   = str(row.get("units", "USD"))
                form    = str(row.get("form", ""))
                fy      = str(row.get("fy", ""))
                fp      = str(row.get("fp", ""))
                end_date = str(row.get("end", ""))

                content = (
                    f"SEC EDGAR Company Fact: {company} (CIK: {cik})\n"
                    f"Metric / Fact: {fact}\n"
                    f"Value: {val} {units}\n"
                    f"Period: FY{fy} {fp} (Ending: {end_date})\n"
                    f"Filing Form: {form}"
                )
                elements.append(
                    DocumentElement(
                        content=content,
                        modality="sec_edgar",
                        source_path=csv_file,
                        metadata={
                            "dataset":      "sec_edgar",
                            "cik":          cik,
                            "company_name": company,
                            "fact":         fact,
                            "val":          val,
                            "form":         form,
                        },
                    )
                )
            return elements

        json_files = list(self.data_dir.rglob("*.json"))
        if not json_files:
            return []
        if sample_size:
            json_files = json_files[:sample_size]

        elements = []
        for file in json_files:
            try:
                data = json.loads(file.read_text(encoding="utf-8"))
                cik         = data.get("cik", "")
                entity_name = data.get("entityName", file.stem)
                facts       = data.get("facts", {})

                facts_summary: list[str] = []
                for taxonomy, tax_facts in facts.items():
                    for concept, details in list(tax_facts.items())[:20]:
                        label       = details.get("label", concept)
                        description = details.get("description", "")
                        facts_summary.append(
                            f"- Concept: {label} ({concept})\n  Description: {description}"
                        )

                content = (
                    "SEC EDGAR Company Fact Sheet\n"
                    f"Company: {entity_name} (CIK: {cik})\n"
                    "Financial Facts Taxonomy:\n" + "\n".join(facts_summary[:30])
                )
                elements.append(
                    DocumentElement(
                        content=content,
                        modality="sec_edgar",
                        source_path=file,
                        metadata={
                            "dataset":      "sec_edgar",
                            "cik":          cik,
                            "company_name": entity_name,
                        },
                    )
                )
            except Exception:
                continue
        return elements
