import json
import logging
from pathlib import Path

import networkx as nx

from .models import Chunk, Document, FinancialFact, GraphExport

logger = logging.getLogger(__name__)

# Maximum years to look back in historical traversal
_MAX_HISTORY_YEARS = 10


class KnowledgeGraph:
    def __init__(self, path: Path):
        self.path = path
        self.graph = nx.MultiDiGraph()
        self.load()

    def load(self) -> None:
        if self.path.exists():
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            self.graph = nx.node_link_graph(payload, edges="edges")

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = nx.node_link_data(self.graph, edges="edges")
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def add_document(
        self, document: Document, chunks: list[Chunk], facts: list[FinancialFact]
    ) -> None:
        self.graph.add_node(
            document.id,
            kind="Report",
            source=document.source,
            ticker=document.ticker,
            year=document.year,
        )
        if document.ticker:
            company_id = f"company:{document.ticker.upper()}"
            self.graph.add_node(company_id, kind="Company", ticker=document.ticker.upper())
            self.graph.add_edge(company_id, document.id, relation="HAS_REPORT")
        if document.year:
            period_id = f"period:{document.year}"
            self.graph.add_node(period_id, kind="Period", year=document.year)
            self.graph.add_edge(document.id, period_id, relation="FOR_PERIOD")
        previous_id = None
        for chunk in chunks:
            self.graph.add_node(chunk.id, kind="Chunk", **chunk.model_dump(exclude={"id"}))
            self.graph.add_edge(document.id, chunk.id, relation="MENTIONS")
            if previous_id:
                self.graph.add_edge(previous_id, chunk.id, relation="NEXT")
            previous_id = chunk.id
        for fact in facts:
            metric_id = f"metric:{fact.metric.casefold()}"
            self.graph.add_node(metric_id, kind="Metric", name=fact.metric)
            self.graph.add_node(fact.id, kind="Fact", **fact.model_dump(exclude={"id"}))
            self.graph.add_edge(document.id, fact.id, relation="HAS_FACT")
            self.graph.add_edge(fact.id, metric_id, relation="MEASURES")
            self.graph.add_edge(fact.id, fact.chunk_id, relation="SUPPORTED_BY")
        self.save()

    def facts_for_chunk(self, chunk_id: str) -> list[FinancialFact]:
        result = []
        for node_id, attrs in self.graph.nodes(data=True):
            if attrs.get("kind") == "Fact" and attrs.get("chunk_id") == chunk_id:
                result.append(
                    FinancialFact(id=node_id, **{k: v for k, v in attrs.items() if k != "kind"})
                )
        return result

    def find_historical_facts(
        self,
        ticker: str,
        standardized_metric: str,
        limit_years: int = 3,
    ) -> list[FinancialFact]:
        """Phase 1.1 – Multi-hop graph traversal for time-series facts.

        Traversal path:
          Fact(t) -[MEASURES]-> Metric <-[MEASURES]- Fact(t-1) -[FOR_PERIOD]-> Period(t-1)

        Collects all Fact nodes that:
          1. Belong to the given ticker (or any ticker if None).
          2. Have standardized_metric matching the requested taxonomy code.
          3. Are sorted descending by period, limited to `limit_years` most recent.

        Returns facts sorted oldest -> newest (chronological) for LLM context.
        """
        matched: list[FinancialFact] = []
        ticker_upper = ticker.upper() if ticker else None

        for node_id, attrs in self.graph.nodes(data=True):
            if attrs.get("kind") != "Fact":
                continue
            # Filter by ticker
            if ticker_upper and attrs.get("ticker") and attrs["ticker"].upper() != ticker_upper:
                continue
            # Filter by standardized metric code
            if attrs.get("standardized_metric") != standardized_metric:
                continue
            try:
                matched.append(
                    FinancialFact(id=node_id, **{k: v for k, v in attrs.items() if k != "kind"})
                )
            except Exception as exc:
                logger.debug("Failed building FinancialFact from node %s: %s", node_id, exc)
                continue

        # Deduplicate by period: keep highest confidence fact per period
        by_period: dict[int, FinancialFact] = {}
        no_period: list[FinancialFact] = []
        for f in matched:
            if f.period is not None:
                if f.period not in by_period or f.confidence > by_period[f.period].confidence:
                    by_period[f.period] = f
            else:
                no_period.append(f)

        deduped = list(by_period.values()) + no_period
        # Sort by period descending, take the `limit_years` most recent, then reverse to oldest-first
        deduped.sort(key=lambda f: f.period or 0, reverse=True)
        return list(reversed(deduped[:limit_years]))

    def find_facts_by_standard_metric(
        self,
        standardized_metric: str,
        ticker: str | None = None,
        limit: int = 20,
    ) -> list[FinancialFact]:
        """Return all facts matching a taxonomy code, optionally filtered by ticker.

        Useful for cross-document aggregation and comparison queries.
        """
        results: list[FinancialFact] = []
        ticker_upper = ticker.upper() if ticker else None
        for node_id, attrs in self.graph.nodes(data=True):
            if attrs.get("kind") != "Fact":
                continue
            if attrs.get("standardized_metric") != standardized_metric:
                continue
            if ticker_upper and attrs.get("ticker") and attrs["ticker"].upper() != ticker_upper:
                continue
            try:
                results.append(
                    FinancialFact(id=node_id, **{k: v for k, v in attrs.items() if k != "kind"})
                )
            except Exception as exc:
                logger.debug("Failed building FinancialFact from node %s: %s", node_id, exc)
                continue
            if len(results) >= limit:
                break
        results.sort(key=lambda f: (f.ticker or "", f.period or 0))
        return results

    def remove_document(self, document_id: str) -> None:
        if not self.graph.has_node(document_id):
            return
        chunk_ids = {
            target
            for _, target, data in self.graph.out_edges(document_id, data=True)
            if data.get("relation") == "MENTIONS"
        }
        fact_ids = {
            target
            for _, target, data in self.graph.out_edges(document_id, data=True)
            if data.get("relation") == "HAS_FACT"
        }
        for node in fact_ids | chunk_ids | {document_id}:
            if self.graph.has_node(node):
                self.graph.remove_node(node)
        self.save()

    def export(self, limit: int = 500) -> GraphExport:
        nodes = [
            {"id": node_id, **attrs} for node_id, attrs in list(self.graph.nodes(data=True))[:limit]
        ]
        allowed = {node["id"] for node in nodes}
        edges = [
            {"source": source, "target": target, **attrs}
            for source, target, attrs in self.graph.edges(data=True)
            if source in allowed and target in allowed
        ]
        return GraphExport(nodes=nodes, edges=edges[:limit])

