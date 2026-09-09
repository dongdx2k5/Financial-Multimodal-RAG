import json
from pathlib import Path
from typing import Any

import pandas as pd

from financial_rag.ingestion.models import DocumentElement


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


class FinQADatasetLoader:
    def __init__(self, data_dir: Path | None = None):
        self.data_dir = data_dir or Path("data/raw/datasets/huggingface/finqa_parquet")

    def load(self, split: str = "train", sample_size: int | None = None) -> list[DocumentElement]:
        file_path = self.data_dir / f"{split}.parquet"
        if not file_path.exists():
            raise FileNotFoundError(f"FinQA dataset file not found: {file_path}")

        df = pd.read_parquet(file_path)
        if sample_size:
            df = df.head(sample_size)

        elements: list[DocumentElement] = []
        for index, row in df.iterrows():
            item_id = str(row.get("id", f"finqa_{index}"))
            question = str(row.get("question", ""))
            answer = str(row.get("answer", ""))

            pre_text = _to_text(row.get("pre_text"))
            post_text = _to_text(row.get("post_text"))
            table_str = _format_table(row.get("table"))

            content_parts = [
                f"Financial Context ID: {item_id}",
                f"Pre-Text:\n{pre_text}" if pre_text else "",
                f"Financial Table:\n{table_str}" if table_str else "",
                f"Post-Text:\n{post_text}" if post_text else "",
                f"Question: {question}",
                f"Answer / Formula: {answer}",
            ]
            full_content = "\n\n".join(p for p in content_parts if p)

            elements.append(
                DocumentElement(
                    content=full_content,
                    modality="finqa_qa",
                    source_path=file_path,
                    metadata={
                        "dataset": "finqa",
                        "split": split,
                        "id": item_id,
                        "question": question,
                        "answer": answer,
                    },
                )
            )
        return elements


class ViFinTabDatasetLoader:
    def __init__(self, data_dir: Path | None = None):
        self.data_dir = data_dir or Path("data/raw/datasets/huggingface/ViFinTab_parquet")

    def load(self, split: str = "train", sample_size: int | None = None) -> list[DocumentElement]:
        file_path = self.data_dir / f"{split}.parquet"
        if not file_path.exists():
            raise FileNotFoundError(f"ViFinTab dataset file not found: {file_path}")

        df = pd.read_parquet(file_path)
        if sample_size:
            df = df.head(sample_size)

        elements: list[DocumentElement] = []
        for index, row in df.iterrows():
            img_path = str(row.get("image_path", f"vifintab_{index}.png"))
            objects = row.get("objects")
            
            obj_desc = _to_text(objects)
            if isinstance(objects, dict):
                obj_desc = json.dumps(objects, ensure_ascii=False)

            content = f"Vietnamese Financial Table Document: {img_path}\nStructure & Elements:\n{obj_desc}"
            elements.append(
                DocumentElement(
                    content=content,
                    modality="vifintab_table",
                    source_path=file_path,
                    metadata={
                        "dataset": "vifintab",
                        "split": split,
                        "image_path": img_path,
                    },
                )
            )
        return elements


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
                cik = str(row.get("cik", ""))
                company = str(row.get("entityName", ""))
                fact = str(row.get("companyFact", ""))
                val = str(row.get("val", ""))
                units = str(row.get("units", "USD"))
                form = str(row.get("form", ""))
                fy = str(row.get("fy", ""))
                fp = str(row.get("fp", ""))
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
                            "dataset": "sec_edgar",
                            "cik": cik,
                            "company_name": company,
                            "fact": fact,
                            "val": val,
                            "form": form,
                        },
                    )
                )
            return elements

        json_files = list(self.data_dir.rglob("*.json"))
        if not json_files:
            return []

        if sample_size:
            json_files = json_files[:sample_size]

        elements: list[DocumentElement] = []
        for file in json_files:
            try:
                data = json.loads(file.read_text(encoding="utf-8"))
                cik = data.get("cik", "")
                entity_name = data.get("entityName", file.stem)
                facts = data.get("facts", {})

                facts_summary = []
                for taxonomy, tax_facts in facts.items():
                    for concept, details in list(tax_facts.items())[:20]:
                        label = details.get("label", concept)
                        description = details.get("description", "")
                        facts_summary.append(f"- Concept: {label} ({concept})\n  Description: {description}")

                content = (
                    f"SEC EDGAR Company Fact Sheet\n"
                    f"Company: {entity_name} (CIK: {cik})\n"
                    f"Financial Facts Taxonomy:\n" + "\n".join(facts_summary[:30])
                )
                elements.append(
                    DocumentElement(
                        content=content,
                        modality="sec_edgar",
                        source_path=file,
                        metadata={
                            "dataset": "sec_edgar",
                            "cik": cik,
                            "company_name": entity_name,
                        },
                    )
                )
            except Exception:
                continue
        return elements
