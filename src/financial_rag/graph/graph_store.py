"""Graph persistence — save and load the Knowledge Graph to/from disk.

Two formats are supported:
  - **JSON**  (``knowledge_graph.json``) — human-readable, always written.
  - **GEXF**  (``knowledge_graph.gexf``) — standard graph-exchange format,
    readable by Gephi, NetworkX, and other tools.

The JSON format uses node-link data (networkx.node_link_data) which is
compact and round-trips perfectly through networkx.node_link_graph.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx

from financial_rag.graph.knowledge_graph import KnowledgeGraphBuilder


# ---------------------------------------------------------------------------
# GraphStore
# ---------------------------------------------------------------------------

class GraphStore:
    """Persist and restore a KnowledgeGraphBuilder's DiGraph.

    Parameters
    ----------
    graph_dir : Path
        Directory where graph files are stored.  Created automatically.
    """

    JSON_FILENAME = "knowledge_graph.json"
    GEXF_FILENAME = "knowledge_graph.gexf"

    def __init__(self, graph_dir: Path) -> None:
        self.graph_dir = Path(graph_dir)
        self.graph_dir.mkdir(parents=True, exist_ok=True)

    @property
    def json_path(self) -> Path:
        return self.graph_dir / self.JSON_FILENAME

    @property
    def gexf_path(self) -> Path:
        return self.graph_dir / self.GEXF_FILENAME

    # ------------------------------------------------------------------ #
    # Save
    # ------------------------------------------------------------------ #

    def save(self, builder: KnowledgeGraphBuilder) -> None:
        """Persist the graph to JSON and GEXF."""
        G = builder.graph

        # ── JSON (node-link format) ───────────────────────────────────
        data: dict[str, Any] = nx.node_link_data(G)
        with self.json_path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2, default=str)

        # ── GEXF ─────────────────────────────────────────────────────
        try:
            nx.write_gexf(G, str(self.gexf_path))
        except Exception:
            # GEXF write may fail for complex attribute types; JSON is the primary format
            pass

    # ------------------------------------------------------------------ #
    # Load
    # ------------------------------------------------------------------ #

    def load(self, builder: KnowledgeGraphBuilder) -> bool:
        """Load JSON graph into *builder*.  Returns True if successful."""
        if not self.json_path.exists():
            return False

        try:
            with self.json_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            G: nx.DiGraph = nx.node_link_graph(data, directed=True, multigraph=False)
            builder.graph = G
            # Rebuild node index from loaded graph
            builder._node_index = {
                attrs.get("label", nid): nid
                for nid, attrs in G.nodes(data=True)
                if attrs.get("label")
            }
            return True
        except Exception:
            return False

    def exists(self) -> bool:
        """Return True if a persisted graph file exists."""
        return self.json_path.exists()

    # ------------------------------------------------------------------ #
    # Statistics
    # ------------------------------------------------------------------ #

    def stats(self, builder: KnowledgeGraphBuilder) -> dict[str, Any]:
        """Return a stats dict about the current in-memory graph."""
        G = builder.graph
        return {
            "nodes":              G.number_of_nodes(),
            "edges":              G.number_of_edges(),
            "json_path":          str(self.json_path),
            "json_exists":        self.json_path.exists(),
            "gexf_exists":        self.gexf_path.exists(),
            "node_type_counts":   builder.stats().get("node_types", {}),
            "edge_type_counts":   builder.stats().get("edge_types", {}),
        }
