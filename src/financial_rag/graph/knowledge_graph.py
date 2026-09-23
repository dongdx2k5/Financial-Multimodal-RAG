"""Knowledge Graph Builder for Financial GraphRAG.

Builds a directed property graph G = (V, E) where:
  - Nodes  represent financial entities: COMPANY, FINANCIAL_METRIC, YEAR, VALUE,
                                         REPORT_IMAGE, TABLE_REGION
  - Edges  represent typed relationships: HAS_REVENUE, HAS_PROFIT, HAS_NET_INCOME,
                                          INCREASED_BY, DECREASED_BY, REPORT_PERIOD,
                                          HAS_TABLE_REGION, SAME_AS

The graph is built from two data sources:
  - FinQA    (dreamerdeo/finqa)   : English financial Q&A with text + 2-D table
  - ViFinTab (ngnquan/ViFinTab)   : Vietnamese table-detection images with YOLO OBB
"""

from __future__ import annotations

import re
import uuid
from typing import Any

import networkx as nx


# ---------------------------------------------------------------------------
# Node type constants
# ---------------------------------------------------------------------------

class NodeType:
    COMPANY          = "COMPANY"
    FINANCIAL_METRIC = "FINANCIAL_METRIC"
    YEAR             = "YEAR"
    VALUE            = "VALUE"
    REPORT_IMAGE     = "REPORT_IMAGE"
    TABLE_REGION     = "TABLE_REGION"
    QUESTION         = "QUESTION"
    ANSWER           = "ANSWER"


# ---------------------------------------------------------------------------
# Edge (relation) type constants
# ---------------------------------------------------------------------------

class RelationType:
    HAS_REVENUE      = "HAS_REVENUE"
    HAS_PROFIT       = "HAS_PROFIT"
    HAS_NET_INCOME   = "HAS_NET_INCOME"
    HAS_VALUE        = "HAS_VALUE"
    INCREASED_BY     = "INCREASED_BY"
    DECREASED_BY     = "DECREASED_BY"
    REPORT_PERIOD    = "REPORT_PERIOD"
    HAS_TABLE_REGION = "HAS_TABLE_REGION"
    SAME_AS          = "SAME_AS"          # cross-language equivalence
    ANSWERS          = "ANSWERS"
    RELATES_TO       = "RELATES_TO"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_MONEY_RE = re.compile(
    r"[\$€£¥₫]?\s*[\d,]+(?:\.\d+)?\s*(?:billion|million|thousand|B|M|K|tỷ|triệu|nghìn)?",
    re.IGNORECASE,
)

# Map common financial column header keywords → relation type
_HEADER_TO_RELATION: dict[str, str] = {
    "revenue":         RelationType.HAS_REVENUE,
    "net revenue":     RelationType.HAS_REVENUE,
    "total revenue":   RelationType.HAS_REVENUE,
    "sales":           RelationType.HAS_REVENUE,
    "doanh thu":       RelationType.HAS_REVENUE,
    "net income":      RelationType.HAS_NET_INCOME,
    "net profit":      RelationType.HAS_NET_INCOME,
    "lợi nhuận ròng":  RelationType.HAS_NET_INCOME,
    "lợi nhuận sau thuế": RelationType.HAS_NET_INCOME,
    "gross profit":    RelationType.HAS_PROFIT,
    "lợi nhuận gộp":   RelationType.HAS_PROFIT,
    "operating profit":RelationType.HAS_PROFIT,
}


def _sanitize_id(raw: str) -> str:
    """Create a graph-safe node ID from a raw string."""
    return re.sub(r"\s+", "_", raw.strip().lower())[:120]


def _unique_id(prefix: str = "node") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# KnowledgeGraphBuilder
# ---------------------------------------------------------------------------

class KnowledgeGraphBuilder:
    """Builds and manages the financial Knowledge Graph.

    Usage
    -----
    >>> builder = KnowledgeGraphBuilder()
    >>> builder.add_finqa_sample(sample)           # from FinQA dataset
    >>> builder.add_vifintab_sample(sample)        # from ViFinTab dataset
    >>> builder.add_triple("Apple", "HAS_REVENUE", "265B", year="2021")
    >>> G = builder.graph
    """

    def __init__(self) -> None:
        self.graph: nx.DiGraph = nx.DiGraph()
        self._node_index: dict[str, str] = {}   # canonical_key → node_id

    # ------------------------------------------------------------------ #
    # Core graph primitives
    # ------------------------------------------------------------------ #

    def add_node(self, node_id: str, node_type: str, **attrs: Any) -> str:
        """Add or update a node. Returns node_id."""
        self.graph.add_node(node_id, type=node_type, **attrs)
        return node_id

    def add_edge(self, src: str, dst: str, relation: str, **attrs: Any) -> None:
        """Add a directed edge src → dst with a relation label."""
        self.graph.add_edge(src, dst, relation=relation, **attrs)

    def get_or_create_node(
        self,
        canonical_key: str,
        node_type: str,
        **attrs: Any,
    ) -> str:
        """Return existing node ID or create new one for canonical_key."""
        if canonical_key in self._node_index:
            node_id = self._node_index[canonical_key]
            # Update attrs
            self.graph.nodes[node_id].update(attrs)
            return node_id
        node_id = _sanitize_id(canonical_key) or _unique_id(node_type.lower())
        if node_id in self.graph:
            node_id = f"{node_id}_{uuid.uuid4().hex[:4]}"
        self._node_index[canonical_key] = node_id
        # Ensure 'label' is in attrs (do NOT pass it again as a separate kwarg)
        if "label" not in attrs:
            attrs["label"] = canonical_key
        self.add_node(node_id, node_type, **attrs)
        return node_id

    # ------------------------------------------------------------------ #
    # High-level triple builder
    # ------------------------------------------------------------------ #

    def add_triple(
        self,
        subject: str,
        relation: str,
        obj: str,
        subject_type: str = NodeType.COMPANY,
        object_type: str = NodeType.VALUE,
        year: str | None = None,
        source: str = "unknown",
        **edge_attrs: Any,
    ) -> None:
        """Add an (S, R, O) triple as two nodes + one edge.

        Optionally links the subject to a YEAR node via REPORT_PERIOD.
        """
        src_id = self.get_or_create_node(subject, subject_type, source=source)
        dst_id = self.get_or_create_node(obj, object_type, source=source)
        self.add_edge(src_id, dst_id, relation=relation, year=year or "", source=source, **edge_attrs)

        if year:
            year_id = self.get_or_create_node(year, NodeType.YEAR, label=year)
            # subject → REPORT_PERIOD → year
            if not self.graph.has_edge(src_id, year_id):
                self.add_edge(src_id, year_id, relation=RelationType.REPORT_PERIOD)

    # ------------------------------------------------------------------ #
    # FinQA dataset integration
    # ------------------------------------------------------------------ #

    def add_finqa_sample(
        self,
        sample: dict[str, Any],
        source: str = "finqa",
    ) -> None:
        """Ingest one FinQA record into the knowledge graph.

        The FinQA record structure:
          id, pre_text (list[str]), post_text (list[str]),
          table (list[list[str]]), question (str), answer (str)

        Strategy
        --------
        1. Parse the 2-D table: first row = headers, first column = row labels.
           Each (row_label, col_header) pair with a value → triple.
        2. Extract any company name from pre_text / id.
        3. Add Question and Answer nodes linked to the extracted triples.
        """
        item_id: str = str(sample.get("id", _unique_id("finqa")))
        question: str = str(sample.get("question", ""))
        answer: str   = str(sample.get("answer", ""))
        pre_text_parts = sample.get("pre_text", []) or []
        post_text_parts = sample.get("post_text", []) or []
        table = sample.get("table") or []

        pre_text = " ".join(str(p) for p in pre_text_parts if p)

        # ── Extract company name from id or pre_text ──────────────────
        company_name = self._extract_company(item_id, pre_text)

        # ── Parse table into triples ──────────────────────────────────
        if table and isinstance(table, (list, tuple)) and len(table) > 1:
            self._ingest_finqa_table(
                table=table,
                company_name=company_name,
                item_id=item_id,
                source=source,
            )

        # ── Add Question / Answer nodes ───────────────────────────────
        if question:
            q_id = f"q_{_sanitize_id(item_id)}"
            self.add_node(q_id, NodeType.QUESTION, label=question, source=source)
            if company_name:
                co_id = self.get_or_create_node(company_name, NodeType.COMPANY, source=source)
                if not self.graph.has_edge(q_id, co_id):
                    self.add_edge(q_id, co_id, relation=RelationType.RELATES_TO)
            if answer:
                a_id = f"a_{_sanitize_id(item_id)}"
                self.add_node(a_id, NodeType.ANSWER, label=answer, source=source)
                self.add_edge(q_id, a_id, relation=RelationType.ANSWERS)

    def _ingest_finqa_table(
        self,
        table: list[Any],
        company_name: str,
        item_id: str,
        source: str,
    ) -> None:
        """Parse FinQA 2-D table and add triples to the graph.

        Table format: [[header1, header2, ...], [row_label, val1, val2, ...], ...]
        """
        # Normalise rows
        rows: list[list[str]] = []
        for r in table:
            if hasattr(r, "tolist"):
                r = r.tolist()
            if isinstance(r, (list, tuple)):
                rows.append([str(c) if c is not None else "" for c in r])
            else:
                rows.append([str(r)])

        if not rows:
            return

        headers = rows[0]  # first row = column headers

        for data_row in rows[1:]:
            if not data_row:
                continue
            row_label = data_row[0].strip()   # first column = metric name / row label

            # Extract any year from the row label or adjacent headers
            year_in_label = _YEAR_RE.search(row_label)
            row_year = year_in_label.group(0) if year_in_label else None

            for col_idx, cell_val in enumerate(data_row[1:], start=1):
                if col_idx >= len(headers):
                    break
                col_header = headers[col_idx].strip()
                cell_val   = cell_val.strip()
                if not cell_val or cell_val in {"-", "—", "N/A", "n/a", ""}:
                    continue

                # Determine year from column header
                year_in_header = _YEAR_RE.search(col_header)
                year = (
                    year_in_header.group(0) if year_in_header
                    else row_year
                )

                # Determine relation type
                combined_label = f"{row_label} {col_header}".lower()
                relation = RelationType.HAS_VALUE
                for kw, rel in _HEADER_TO_RELATION.items():
                    if kw in combined_label:
                        relation = rel
                        break

                # Metric node (e.g. "net income | 2021")
                metric_key = f"{row_label} | {col_header}"
                metric_id  = self.get_or_create_node(
                    metric_key, NodeType.FINANCIAL_METRIC, source=source
                )

                # Value node
                value_key = f"{metric_key} = {cell_val}"
                value_id  = self.get_or_create_node(
                    value_key, NodeType.VALUE,
                    raw_value=cell_val,
                    source=source,
                )

                if company_name:
                    co_id = self.get_or_create_node(
                        company_name, NodeType.COMPANY, source=source
                    )
                    self.add_edge(co_id, metric_id, relation=relation, year=year or "")
                    self.add_edge(
                        metric_id, value_id, relation=RelationType.HAS_VALUE,
                        year=year or "", source=source,
                    )
                    if year:
                        yr_id = self.get_or_create_node(year, NodeType.YEAR, label=year)
                        if not self.graph.has_edge(co_id, yr_id):
                            self.add_edge(co_id, yr_id, relation=RelationType.REPORT_PERIOD)

    # ------------------------------------------------------------------ #
    # ViFinTab dataset integration
    # ------------------------------------------------------------------ #

    def add_vifintab_sample(
        self,
        sample: dict[str, Any],
        source: str = "vifintab",
    ) -> None:
        """Ingest one ViFinTab record into the knowledge graph.

        ViFinTab is a table-detection dataset. Each record contains:
          image_path   : str — file path of the financial report image
          image_base64 : str — base64-encoded image (can be large)
          objects      : dict — {"class_id": [...], "coordinates": [...]}
                         class_id: 0=income statement, 1=balance sheet, 2=cash flow
                         coordinates: YOLO OBB bounding boxes

        Strategy
        --------
        - Create a REPORT_IMAGE node for the image.
        - For each detected table region (bounding box), create a TABLE_REGION node.
        - Link image → table_region via HAS_TABLE_REGION.
        """
        img_path: str = str(sample.get("image_path", _unique_id("vifintab")))

        # Determine a short document name from path
        doc_name = img_path.split("/")[-1].split("\\")[-1]

        img_node_id = self.get_or_create_node(
            img_path,
            NodeType.REPORT_IMAGE,
            label=doc_name,
            source=source,
            language="vi",
        )

        # Parse objects field
        objects = sample.get("objects")
        if objects is None:
            return

        # HuggingFace stores objects as dict of lists:
        # {"class_id": [0, 1, ...], "coordinates": [[x,y,...], ...]}
        class_ids   = objects.get("class_id", []) if isinstance(objects, dict) else []
        coordinates = objects.get("coordinates", []) if isinstance(objects, dict) else []

        _CLASS_NAMES = {
            0: "Báo cáo kết quả kinh doanh (Income Statement)",
            1: "Bảng cân đối kế toán (Balance Sheet)",
            2: "Báo cáo lưu chuyển tiền tệ (Cash Flow Statement)",
        }

        for i, (cls_id, coords) in enumerate(zip(class_ids, coordinates)):
            table_type  = _CLASS_NAMES.get(int(cls_id), f"Table Type {cls_id}")
            region_key  = f"{img_path}::region_{i}::{table_type}"
            region_id   = self.get_or_create_node(
                region_key,
                NodeType.TABLE_REGION,
                label=table_type,
                class_id=int(cls_id),
                source=source,
                language="vi",
            )
            self.add_edge(
                img_node_id, region_id,
                relation=RelationType.HAS_TABLE_REGION,
                region_index=i,
            )

    # ------------------------------------------------------------------ #
    # Utility
    # ------------------------------------------------------------------ #

    def _extract_company(self, item_id: str, text: str) -> str:
        """Heuristically extract a company name from FinQA item_id or pre_text."""
        # FinQA IDs are often like "AAPL/2020/annual_report/2"
        parts = re.split(r"[/_\-]", item_id)
        ticker = parts[0].upper() if parts else ""

        # Known ticker→company mapping
        _TICKER_MAP: dict[str, str] = {
            "AAPL": "Apple",
            "MSFT": "Microsoft",
            "GOOG": "Google",
            "AMZN": "Amazon",
            "META": "Meta",
            "TSLA": "Tesla",
            "NVDA": "NVIDIA",
            "JPM":  "JPMorgan Chase",
            "BAC":  "Bank of America",
            "XOM":  "ExxonMobil",
        }
        if ticker in _TICKER_MAP:
            return _TICKER_MAP[ticker]

        # Try to find "Company Name" pattern in first sentence of pre_text
        match = re.search(
            r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\s+(?:Inc\.?|Corp\.?|Ltd\.?|LLC|Co\.?|Group|Holdings?)\b",
            text,
        )
        if match:
            return match.group(0)

        return ticker or "Unknown Company"

    def stats(self) -> dict[str, int]:
        """Return basic graph statistics."""
        type_counts: dict[str, int] = {}
        for _, attrs in self.graph.nodes(data=True):
            t = attrs.get("type", "UNKNOWN")
            type_counts[t] = type_counts.get(t, 0) + 1

        rel_counts: dict[str, int] = {}
        for _, _, attrs in self.graph.edges(data=True):
            r = attrs.get("relation", "UNKNOWN")
            rel_counts[r] = rel_counts.get(r, 0) + 1

        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "node_types":  type_counts,
            "edge_types":  rel_counts,
        }
