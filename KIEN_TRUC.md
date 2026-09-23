# 🏛️ TÀI LIỆU KIẾN TRÚC TỔNG THỂ HỆ THỐNG FINANCIAL GRAPHRAG

Tài liệu này mô tả chi tiết toàn bộ kiến trúc, luồng dữ liệu và các cơ chế kỹ thuật của hệ thống **Financial GraphRAG** — hệ thống Hybrid Knowledge Graph + Vector RAG chuyên sâu cho phân tích báo cáo tài chính đa ngôn ngữ (Tiếng Việt & Tiếng Anh).

---

## 🔀 1. Kiến Trúc Tổng Thể (System Architecture)

```
 ┌─────────────────────────────────────────────────────────────────────┐
 │              NGUỒN DỮ LIỆU (HuggingFace Hub)                       │
 │  ViFinTab (ngnquan/ViFinTab)    FinQA (dreamerdeo/finqa)            │
 │  – Ảnh tài liệu PDF tiếng Việt  – Q&A tài chính tiếng Anh          │
 │  – YOLO OBB bounding boxes       – Table 2D + pre/post text         │
 └──────────────┬──────────────────────────┬──────────────────────────┘
                │                          │
                ▼                          ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  MODULE 1: DATA INGESTION  (ingestion/)                             │
 │  hf_downloader.py     → Tải & cache Parquet từ HuggingFace         │
 │  dataset_adapters.py  → Chuyển Parquet → DocumentElement           │
 │  financial_normalizer.py → Chuẩn hoá giá trị kế toán               │
 └──────────┬──────────────────────────────┬──────────────────────────┘
            │                              │
            ▼                              ▼
 ┌──────────────────────┐   ┌─────────────────────────────────────────┐
 │ MODULE 2: VECTOR      │   │  MODULE 3: GRAPH CONSTRUCTION (graph/)  │
 │ STORAGE (retrieval/)  │   │  knowledge_graph.py → KnowledgeGraph    │
 │ ChromaDB + DocStore   │   │  entity_extractor.py → LLM triples      │
 │ (embedding & search)  │   │  bilingual_linker.py → VI↔EN links      │
 └──────────┬───────────┘   │  graph_store.py → JSON/GEXF persist     │
            │               └─────────────────┬───────────────────────┘
            │                                 │
            ▼                                 ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  MODULE 4: HYBRID RETRIEVAL                                          │
 │  Vector Search (ChromaDB)  +  Subgraph Traversal (GraphRetriever)   │
 │  Reranker (time anchoring + entity matching)                         │
 └──────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  MODULE 5: LLM GENERATION & REASONING (reasoning/)                  │
 │  GraphRAGQAEngine → Hybrid Fusion Prompt → LLM Answer               │
 │  ProgramOfThoughtEngine → Python sandbox → Exact arithmetic          │
 │  UnifiedLLMClient → Gemini / LLaMA / OpenAI-compatible API          │
 └─────────────────────────────────────────────────────────────────────┘
```

---

## 📦 2. Chi Tiết Từng Module

### Module 1 — Data Ingestion

| File | Chức năng |
|------|-----------|
| [`hf_downloader.py`](src/financial_rag/ingestion/hf_downloader.py) | Tải ViFinTab + FinQA từ HuggingFace Hub, cache local Parquet |
| [`dataset_adapters.py`](src/financial_rag/ingestion/dataset_adapters.py) | `FinQADatasetLoader`, `ViFinTabDatasetLoader` — chuyển đổi sang `DocumentElement` |
| [`financial_normalizer.py`](src/financial_rag/ingestion/financial_normalizer.py) | Chuẩn hoá số kế toán: `(125.4)→-125.4`, phát hiện đơn vị tỷ/triệu |

**Cấu trúc Dataset thực tế:**

- **FinQA**: `id`, `pre_text[]`, `post_text[]`, `table[][]`, `question`, `answer`
- **ViFinTab**: `image_path`, `image_base64` (optional), `objects` = `{class_id:[0,1,2], coordinates:[[...]]}`
  - class 0 = Báo cáo KQKD, class 1 = Bảng CĐKT, class 2 = Báo cáo LCTT

---

### Module 2 — Vector Storage (giữ nguyên)

| File | Chức năng |
|------|-----------|
| [`vector_store.py`](src/financial_rag/retrieval/vector_store.py) | ChromaDB persistent index + Gemini/LLaMA embeddings |
| [`doc_store.py`](src/financial_rag/retrieval/doc_store.py) | JSON key-value store lưu raw content |
| [`reranker.py`](src/financial_rag/retrieval/reranker.py) | Time anchoring + financial entity matching re-rank |
| [`chunking.py`](src/financial_rag/retrieval/chunking.py) | Sliding-window chunking cho text |

---

### Module 3 — Knowledge Graph (MỚI)

| File | Chức năng |
|------|-----------|
| [`graph/knowledge_graph.py`](src/financial_rag/graph/knowledge_graph.py) | `KnowledgeGraphBuilder` — NetworkX DiGraph, add_finqa_sample, add_vifintab_sample |
| [`graph/entity_extractor.py`](src/financial_rag/graph/entity_extractor.py) | `LLMEntityExtractor` — trích xuất (S,R,O) triple từ văn bản bằng LLM |
| [`graph/bilingual_linker.py`](src/financial_rag/graph/bilingual_linker.py) | `BilingualEntityLinker` — từ điển ~40 thuật ngữ VI↔EN, thêm cạnh SAME_AS |
| [`graph/graph_store.py`](src/financial_rag/graph/graph_store.py) | `GraphStore` — lưu/tải JSON + GEXF |
| [`graph/graph_retriever.py`](src/financial_rag/graph/graph_retriever.py) | `GraphRetriever` — BFS subgraph expansion, time filtering, text serialisation |

**Kiểu Node (NodeType):**
- `COMPANY`, `FINANCIAL_METRIC`, `YEAR`, `VALUE`
- `REPORT_IMAGE` (ViFinTab), `TABLE_REGION` (ViFinTab)
- `QUESTION`, `ANSWER` (FinQA)

**Kiểu Quan hệ (RelationType):**
- `HAS_REVENUE`, `HAS_PROFIT`, `HAS_NET_INCOME`, `HAS_VALUE`
- `INCREASED_BY`, `DECREASED_BY`, `REPORT_PERIOD`
- `HAS_TABLE_REGION`, `SAME_AS`, `ANSWERS`, `RELATES_TO`

---

### Module 4 — Hybrid Retrieval

```
Câu hỏi → [Vector Search (ChromaDB)]   → Top 2K results
         → [Graph Retrieval (BFS)]      → Subgraph nodes & triples
         → [Reranker]                   → Time anchor + entity boost
         → Fused Context (Vector + Graph)
```

---

### Module 5 — LLM Generation & Reasoning (giữ nguyên core, cập nhật prompt)

| File | Chức năng |
|------|-----------|
| [`reasoning/financial_qa.py`](src/financial_rag/reasoning/financial_qa.py) | `GraphRAGQAEngine` (mới) + `FinancialQAEngine` (legacy) |
| [`reasoning/pot_reasoning.py`](src/financial_rag/reasoning/pot_reasoning.py) | Program-of-Thought: LLM sinh Python code → sandbox execute |
| [`reasoning/sandbox.py`](src/financial_rag/reasoning/sandbox.py) | AST-validated Python sandbox, timeout 10s |
| [`reasoning/llm_client.py`](src/financial_rag/reasoning/llm_client.py) | `UnifiedLLMClient` — Gemini + LLaMA + OpenAI-compatible |

---

## 🚀 3. Hướng Dẫn Sử Dụng

### Bước 1: Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### Bước 2: Tải dữ liệu từ HuggingFace
```bash
# Tải cả hai dataset (mặc định không có ảnh để tiết kiệm dung lượng)
python -m financial_rag.ingestion.hf_downloader

# Chỉ tải FinQA, giới hạn 500 mẫu
python -m financial_rag.ingestion.hf_downloader --dataset finqa --sample 500

# Chỉ tải ViFinTab với ảnh
python -m financial_rag.ingestion.hf_downloader --dataset vifintab --include-images
```

### Bước 3: Xây dựng Knowledge Graph + nạp vào Vector Store
```python
from financial_rag.pipeline import ingest_dataset

# FinQA: 200 mẫu, xây graph + vector store
result = ingest_dataset("finqa", split="train", sample_size=200, build_kg=True)
print(result)
# {'vector_chunks': 200, 'graph_nodes': 1500, 'graph_edges': 800, ...}

# ViFinTab
result = ingest_dataset("vifintab", split="train", sample_size=100, build_kg=True)
```

### Bước 4: Chỉ xây dựng Knowledge Graph (không cần vector store)
```python
from financial_rag.pipeline import build_graph

stats = build_graph(dataset="both", split="train", sample_size=300)
print(stats)
# {'nodes': 2000, 'edges': 1200, 'same_as_edges_added': 45, ...}
```

### Bước 5: Truy vấn GraphRAG
```python
from financial_rag.pipeline import query

# Câu hỏi tiếng Anh (FinQA-style, có suy luận số học)
result = query("What was the percentage change in net income from 2018 to 2019?")
print(result["answer"])

# Câu hỏi tiếng Việt (ViFinTab-style)
result = query("Lợi nhuận ròng của công ty trong quý 1 là bao nhiêu?")
print(result["answer"])
print("Retrieval mode:", result["retrieval_mode"])  # "graphrag" or "vector_only"
```

### Bước 6: Xem thống kê Knowledge Graph
```python
from financial_rag.pipeline import graph_stats
print(graph_stats())
```

---

## 🛠️ 4. Cấu Hình `.env` cho GraphRAG

```env
# LLM Provider
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_key_here
CHAT_MODEL=gemini-3.6-flash
EMBEDDING_MODEL=gemini-embedding-001

# GraphRAG settings (tuỳ chọn, có giá trị mặc định)
# GRAPH_DIR=data/graph           # Thư mục lưu đồ thị
# GRAPH_MAX_HOPS=2               # Số hop mở rộng subgraph
# USE_GRAPH_RAG=true             # Bật GraphRAG mode
# HF_CACHE_DIR=data/raw/datasets/huggingface   # Cache HuggingFace
# GRAPH_ENTITY_EXTRACT_LIMIT=200 # Giới hạn LLM entity extraction calls
```

---

## 📊 5. So Sánh Kiến Trúc Cũ và Mới

| Thành phần | Vector RAG (cũ) | GraphRAG (mới) |
|---|---|---|
| **Dữ liệu** | PDF files, FinQA parquet | **HuggingFace Hub**: ViFinTab + FinQA |
| **Lưu trữ** | ChromaDB vector only | ChromaDB + **NetworkX Knowledge Graph** |
| **Truy xuất** | Vector similarity | Vector + **Subgraph BFS traversal** |
| **Entity linking** | Không | **Bilingual VI↔EN linker** (40+ terms) |
| **Extraction** | Không | **LLM (S,R,O) triple extraction** |
| **Suy luận PoT** | ✅ Giữ nguyên | ✅ Giữ nguyên |
| **Multi-LLM** | ✅ Giữ nguyên | ✅ Giữ nguyên |
| **Graph file** | Không | `data/graph/knowledge_graph.json` + `.gexf` |
