"""Quick verification tests for GraphRAG migration."""
import sys
sys.path.insert(0, "src")

errors = []

# Test 1: Settings
try:
    from financial_rag.config.settings import Settings
    s = Settings()
    assert hasattr(s, "graph_dir"), "graph_dir missing"
    assert hasattr(s, "use_graph_rag"), "use_graph_rag missing"
    assert hasattr(s, "graph_max_hops"), "graph_max_hops missing"
    assert hasattr(s, "hf_cache_dir"), "hf_cache_dir missing"
    assert s.use_graph_rag is True
    print("OK  settings.py - GraphRAG fields present, use_graph_rag=True")
except Exception as e:
    errors.append(f"FAIL settings.py: {e}")

# Test 2: KnowledgeGraphBuilder
try:
    from financial_rag.graph.knowledge_graph import KnowledgeGraphBuilder, NodeType, RelationType
    builder = KnowledgeGraphBuilder()
    builder.add_triple("Apple", RelationType.HAS_NET_INCOME, "57.4B", year="2020")
    builder.add_triple("Vinamilk", RelationType.HAS_REVENUE, "60000 ty VND", year="2023")
    builder.add_finqa_sample({
        "id": "test_001",
        "question": "What is net income?",
        "answer": "57.4B",
        "pre_text": ["Apple reported strong results."],
        "post_text": [],
        "table": [
            ["Metric", "2020", "2019"],
            ["Net Income", "57.4B", "55.3B"],
            ["Revenue", "274.5B", "260.2B"],
        ],
    })
    builder.add_vifintab_sample({
        "image_path": "report_q1_2023.png",
        "objects": {
            "class_id": [0, 1],
            "coordinates": [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
        },
    })
    stats = builder.stats()
    assert stats["total_nodes"] > 0, "No nodes created"
    assert stats["total_edges"] > 0, "No edges created"
    print(f"OK  knowledge_graph.py - nodes={stats['total_nodes']}, edges={stats['total_edges']}")
except Exception as e:
    errors.append(f"FAIL knowledge_graph.py: {e}")

# Test 3: BilingualLinker
try:
    from financial_rag.graph.bilingual_linker import BilingualEntityLinker, get_canonical_key
    assert get_canonical_key("loi nhuan rong") is None   # exact match required
    assert get_canonical_key("net income") == "net_income"
    assert get_canonical_key("revenue") == "revenue"
    assert get_canonical_key("doanh thu") == "revenue"
    # Test linking
    from financial_rag.graph.knowledge_graph import KnowledgeGraphBuilder, NodeType
    b = KnowledgeGraphBuilder()
    b.get_or_create_node("net income", NodeType.FINANCIAL_METRIC, label="net income")
    b.get_or_create_node("loi nhuan rong", NodeType.FINANCIAL_METRIC, label="loi nhuan rong")
    linker = BilingualEntityLinker()
    added = linker.link(b)
    print(f"OK  bilingual_linker.py - canonical key mapping works, same_as_edges={added}")
except Exception as e:
    errors.append(f"FAIL bilingual_linker.py: {e}")

# Test 4: GraphStore round-trip
try:
    import tempfile
    from pathlib import Path
    from financial_rag.graph.graph_store import GraphStore
    from financial_rag.graph.knowledge_graph import KnowledgeGraphBuilder, RelationType

    with tempfile.TemporaryDirectory() as tmp:
        store = GraphStore(Path(tmp))
        b = KnowledgeGraphBuilder()
        b.add_triple("TestCo", RelationType.HAS_REVENUE, "100M", year="2022")
        store.save(b)
        assert store.exists()
        b2 = KnowledgeGraphBuilder()
        ok = store.load(b2)
        assert ok, "Load returned False"
        assert b2.graph.number_of_nodes() > 0, "Loaded graph is empty"
    print("OK  graph_store.py - save/load round-trip works")
except Exception as e:
    errors.append(f"FAIL graph_store.py: {e}")

# Test 5: GraphRetriever
try:
    from financial_rag.graph.graph_retriever import GraphRetriever
    from financial_rag.graph.knowledge_graph import KnowledgeGraphBuilder, RelationType

    b = KnowledgeGraphBuilder()
    b.add_triple("Apple", RelationType.HAS_NET_INCOME, "57.4B", year="2020")
    b.add_triple("Apple", RelationType.HAS_REVENUE, "265B", year="2021")
    b.add_triple("Microsoft", RelationType.HAS_NET_INCOME, "44.3B", year="2021")

    retriever = GraphRetriever(b, max_hops=2)
    result = retriever.retrieve("What is Apple net income in 2020?")
    assert result["node_count"] > 0, "No nodes retrieved"
    ctx = result["context_text"]
    assert "Apple" in ctx or "net" in ctx.lower(), "Context missing expected entities"
    print(f"OK  graph_retriever.py - nodes={result['node_count']}, seeds={len(result['seed_nodes'])}")
except Exception as e:
    errors.append(f"FAIL graph_retriever.py: {e}")

# Test 6: HF Downloader
try:
    from financial_rag.ingestion.hf_downloader import HuggingFaceDatasetDownloader
    print("OK  hf_downloader.py - import successful")
except Exception as e:
    errors.append(f"FAIL hf_downloader.py: {e}")

# Test 7: Dataset adapters
try:
    from financial_rag.ingestion.dataset_adapters import FinQADatasetLoader, ViFinTabDatasetLoader
    print("OK  dataset_adapters.py - import successful")
except Exception as e:
    errors.append(f"FAIL dataset_adapters.py: {e}")

# Test 8: Pipeline functions
try:
    from financial_rag.pipeline import build_graph, ingest_dataset, query, graph_stats
    print("OK  pipeline.py - all public functions importable")
except Exception as e:
    errors.append(f"FAIL pipeline.py: {e}")

# Test 9: QA Engines
try:
    from financial_rag.reasoning.financial_qa import GraphRAGQAEngine, FinancialQAEngine
    print("OK  financial_qa.py - GraphRAGQAEngine + FinancialQAEngine importable")
except Exception as e:
    errors.append(f"FAIL financial_qa.py: {e}")

# Test 10: Entity extractor
try:
    from financial_rag.graph.entity_extractor import LLMEntityExtractor
    print("OK  entity_extractor.py - import successful")
except Exception as e:
    errors.append(f"FAIL entity_extractor.py: {e}")

# Test 11: FinQA sample conversion
try:
    from financial_rag.graph.knowledge_graph import KnowledgeGraphBuilder
    b = KnowledgeGraphBuilder()
    sample = {
        "id": "AAPL/2020",
        "question": "What is the growth rate of revenue?",
        "answer": "5.5%",
        "pre_text": ["Apple Inc reported revenue."],
        "post_text": ["Overall performance was strong."],
        "table": [
            ["", "2020", "2019"],
            ["Revenue", "274.5B", "260.2B"],
            ["Net Income", "57.4B", "55.3B"],
        ],
    }
    b.add_finqa_sample(sample, source="finqa")
    s = b.stats()
    assert s["total_nodes"] >= 3, f"Expected >= 3 nodes, got {s['total_nodes']}"
    print(f"OK  FinQA table ingestion - nodes={s['total_nodes']}, edges={s['total_edges']}")
except Exception as e:
    errors.append(f"FAIL FinQA ingestion: {e}")

# Summary
print()
if errors:
    print("=== FAILURES ===")
    for err in errors:
        print("  ", err)
    sys.exit(1)
else:
    print("All", 11, "tests passed! GraphRAG migration successful.")
