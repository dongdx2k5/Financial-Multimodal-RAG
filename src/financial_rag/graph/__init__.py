"""Knowledge Graph module for GraphRAG-based financial document processing.

This module implements the Knowledge Graph layer that sits alongside the
existing ChromaDB vector store. Together they form the GraphRAG hybrid
retrieval pipeline.

Components
----------
knowledge_graph  : KnowledgeGraphBuilder — builds and manages the nx.DiGraph
entity_extractor : LLMEntityExtractor   — LLM-based (Subject, Relation, Object) extraction
bilingual_linker : BilingualEntityLinker — maps Vietnamese ↔ English financial terms
graph_store      : GraphStore           — persistent JSON/GEXF serialisation
graph_retriever  : GraphRetriever       — subgraph-based context assembly for LLMs
"""

from financial_rag.graph.knowledge_graph import KnowledgeGraphBuilder
from financial_rag.graph.graph_store import GraphStore
from financial_rag.graph.graph_retriever import GraphRetriever

__all__ = [
    "KnowledgeGraphBuilder",
    "GraphStore",
    "GraphRetriever",
]
