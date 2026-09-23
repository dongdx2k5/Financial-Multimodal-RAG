"""Subgraph-based context retriever for GraphRAG.

Given a natural-language question, this retriever:
1. Identifies seed nodes (entities mentioned in the question).
2. Expands to a neighbourhood subgraph (up to ``max_hops`` hops).
3. Filters edges by temporal anchors (years/quarters in the question).
4. Serialises the subgraph as a structured text block for the LLM prompt.

The resulting text is combined with ChromaDB vector search results in the
``GraphRAGQAEngine`` to form the hybrid context.
"""

from __future__ import annotations

import re
from typing import Any

import networkx as nx

from financial_rag.graph.knowledge_graph import KnowledgeGraphBuilder
from financial_rag.retrieval.reranker import extract_time_anchors, extract_financial_entities


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TOKEN_RE  = re.compile(r"\w+", re.UNICODE)
_YEAR_RE   = re.compile(r"\b(19|20)\d{2}\b")


def _tokenize(text: str) -> set[str]:
    return {m.group(0).lower() for m in _TOKEN_RE.finditer(text)}


def _node_matches_query(node_attrs: dict[str, Any], q_tokens: set[str]) -> bool:
    """Return True if the node label / type overlaps with query tokens."""
    label = str(node_attrs.get("label", "")).lower()
    node_type = str(node_attrs.get("type", "")).lower()
    node_tokens = _tokenize(label) | _tokenize(node_type)
    return bool(q_tokens & node_tokens)


# ---------------------------------------------------------------------------
# GraphRetriever
# ---------------------------------------------------------------------------

class GraphRetriever:
    """Retrieves a relevant subgraph from the Knowledge Graph for a given question.

    Parameters
    ----------
    builder : KnowledgeGraphBuilder
        The in-memory knowledge graph.
    max_hops : int
        How many hops to expand from seed nodes. Default is 2.
    max_nodes : int
        Cap on subgraph size to prevent context overflow.
    """

    def __init__(
        self,
        builder: KnowledgeGraphBuilder,
        max_hops: int = 2,
        max_nodes: int = 60,
    ) -> None:
        self.builder   = builder
        self.graph     = builder.graph
        self.max_hops  = max_hops
        self.max_nodes = max_nodes

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def retrieve(self, question: str) -> dict[str, Any]:
        """Return subgraph context for the given question.

        Returns a dict with:
          ``subgraph``        : nx.DiGraph — the extracted subgraph
          ``context_text``    : str        — text representation for the LLM
          ``seed_nodes``      : list[str]  — node IDs used as starting points
          ``time_anchors``    : list[str]  — years/periods detected in question
          ``entity_anchors``  : list[str]  — financial entities detected
          ``node_count``      : int
          ``edge_count``      : int
        """
        q_tokens       = _tokenize(question)
        time_anchors   = extract_time_anchors(question)
        entity_anchors = extract_financial_entities(question)

        # ── 1. Find seed nodes ────────────────────────────────────────
        seed_nodes = self._find_seed_nodes(q_tokens, time_anchors, entity_anchors)

        # ── 2. Expand subgraph ────────────────────────────────────────
        subgraph = self._expand_subgraph(seed_nodes)

        # ── 3. Time-filter edges (optional boost) ────────────────────
        if time_anchors:
            subgraph = self._filter_by_time(subgraph, time_anchors)

        # ── 4. Format as text ─────────────────────────────────────────
        context_text = self.format_subgraph_as_context(subgraph, question)

        return {
            "subgraph":       subgraph,
            "context_text":   context_text,
            "seed_nodes":     seed_nodes,
            "time_anchors":   time_anchors,
            "entity_anchors": entity_anchors,
            "node_count":     subgraph.number_of_nodes(),
            "edge_count":     subgraph.number_of_edges(),
        }

    # ------------------------------------------------------------------ #
    # Seed node discovery
    # ------------------------------------------------------------------ #

    def _find_seed_nodes(
        self,
        q_tokens: set[str],
        time_anchors: list[str],
        entity_anchors: list[str],
    ) -> list[str]:
        """Return node IDs whose labels/types overlap with the question."""
        seed_set: dict[str, float] = {}   # node_id → relevance score

        for node_id, attrs in self.graph.nodes(data=True):
            label      = str(attrs.get("label", "")).lower()
            node_type  = str(attrs.get("type", "")).lower()
            node_toks  = _tokenize(label)

            score = 0.0

            # Token overlap
            overlap = len(q_tokens & node_toks)
            score  += overlap * 1.0

            # Time-anchor exact match
            for anchor in time_anchors:
                if anchor.lower() in label or anchor.lower() in node_type:
                    score += 3.0

            # Financial entity match
            for ent in entity_anchors:
                if ent.lower() in label:
                    score += 2.0

            if score > 0:
                seed_set[node_id] = score

        # Sort by score, return top seeds (cap at 15)
        sorted_seeds = sorted(seed_set, key=lambda n: -seed_set[n])
        return sorted_seeds[:15]

    # ------------------------------------------------------------------ #
    # Subgraph expansion
    # ------------------------------------------------------------------ #

    def _expand_subgraph(self, seed_nodes: list[str]) -> nx.DiGraph:
        """BFS expansion up to max_hops from all seed nodes."""
        if not seed_nodes:
            return nx.DiGraph()

        visited: set[str] = set()
        frontier: set[str] = set(seed_nodes) & set(self.graph.nodes)
        visited.update(frontier)

        for _ in range(self.max_hops):
            if len(visited) >= self.max_nodes:
                break
            next_frontier: set[str] = set()
            for node in frontier:
                # Successors (outgoing edges)
                for nbr in self.graph.successors(node):
                    if nbr not in visited:
                        next_frontier.add(nbr)
                # Predecessors (incoming edges)
                for nbr in self.graph.predecessors(node):
                    if nbr not in visited:
                        next_frontier.add(nbr)
            next_frontier = set(list(next_frontier)[: self.max_nodes - len(visited)])
            visited.update(next_frontier)
            frontier = next_frontier

        return self.graph.subgraph(visited).copy()

    # ------------------------------------------------------------------ #
    # Time filtering
    # ------------------------------------------------------------------ #

    def _filter_by_time(
        self,
        subgraph: nx.DiGraph,
        time_anchors: list[str],
    ) -> nx.DiGraph:
        """Prefer (boost) edges whose 'year' attribute matches a time anchor.

        Rather than removing non-matching edges (which would lose context),
        we add a 'time_match' attribute = True to matching edges.
        """
        anchor_set = set(time_anchors)
        for src, dst, attrs in subgraph.edges(data=True):
            edge_year = str(attrs.get("year", ""))
            if any(a in edge_year or edge_year in a for a in anchor_set if a):
                subgraph[src][dst]["time_match"] = True
        return subgraph

    # ------------------------------------------------------------------ #
    # Text serialisation
    # ------------------------------------------------------------------ #

    def format_subgraph_as_context(
        self,
        subgraph: nx.DiGraph,
        question: str = "",
    ) -> str:
        """Serialise the subgraph as a structured text block for the LLM."""
        if subgraph.number_of_nodes() == 0:
            return "[Knowledge Graph] No relevant entities found for this question."

        lines: list[str] = [
            "=== Knowledge Graph Context ===",
            f"Nodes: {subgraph.number_of_nodes()} | Edges: {subgraph.number_of_edges()}",
            "",
        ]

        # ── Node summaries ────────────────────────────────────────────
        lines.append("## Entities (Nodes)")
        for node_id, attrs in subgraph.nodes(data=True):
            label     = attrs.get("label", node_id)
            node_type = attrs.get("type", "UNKNOWN")
            language  = attrs.get("language", "")
            lang_tag  = f" [{language.upper()}]" if language else ""
            lines.append(f"  [{node_type}]{lang_tag} {label}")

        lines.append("")

        # ── Edge / triple summaries ───────────────────────────────────
        lines.append("## Relationships (Triples)")
        for src, dst, attrs in subgraph.edges(data=True):
            src_label  = subgraph.nodes[src].get("label", src)
            dst_label  = subgraph.nodes[dst].get("label", dst)
            relation   = attrs.get("relation", "RELATES_TO")
            year       = attrs.get("year", "")
            time_flag  = " [TIME_MATCH]" if attrs.get("time_match") else ""
            year_str   = f" ({year})" if year else ""
            lines.append(
                f"  ({src_label})  --[{relation}{year_str}]{time_flag}-->  ({dst_label})"
            )

        lines.append("")
        lines.append("=== End of Knowledge Graph Context ===")
        return "\n".join(lines)
