import json
from pathlib import Path

import networkx as nx

from .models import Chunk, Document, FinancialFact, GraphExport


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
        if document.id in self.graph:
            # Replace only this report's derived nodes after parser improvements.
            owned = [
                n
                for n, a in self.graph.nodes(data=True)
                if a.get("kind") in {"Chunk", "Fact"}
                and (a.get("document_id") == document.id or self.graph.has_edge(document.id, n))
            ]
            self.graph.remove_nodes_from(owned + [document.id])
        self.graph.add_node(
            document.id,
            kind="Report",
            source=document.source,
            ticker=document.ticker,
            year=document.year,
            name=document.name,
            scope=document.scope,
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
