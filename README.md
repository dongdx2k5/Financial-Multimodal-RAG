# Financial Multimodal RAG

A production-grade **Multimodal Retrieval-Augmented Generation (RAG)** pipeline designed for financial document analysis (annual reports, 10-K filings, earnings releases, financial statements).

The system implements the **Multi-Vector Retriever** pattern:
1. **Multimodal Ingestion**: Extracts raw text, financial tables (converted to structured HTML), and high-resolution chart/diagram images from financial PDFs.
2. **LLM/Vision Summarization**: Generates compact, semantically rich textual summaries for each modality (using Google Gemini / Vision).
3. **Multi-Vector Indexing**: Indexes summary embeddings into **ChromaDB** while storing original raw content (full text, raw HTML tables, base64 images) in a persistent **DocStore** linked by unique `element_id`s.
4. **Multimodal Reasoning QA**: Retrieves top relevant contexts and feeds text, structured HTML tables, and high-resolution images (`types.Part.from_bytes`) directly into **Gemini Vision** for comprehensive grounded synthesis with financial caveat analysis.

---

## Architecture Overview

```
                          [Financial PDF Document]
                                     |
              +----------------------+----------------------+
              |                      |                      |
         [Raw Text]            [Tables (HTML)]        [Images / Charts]
              |                      |                      |
       LLM Summarization      LLM Summarization      Vision Summarization
              |                      |                      |
              +----------------------+----------------------+
                                     |
                       +-------------+-------------+
                       |                           |
                  [Summaries]                [Raw Content]
                       |                           |
                 Embeddings                 DocStore (JSON/KV)
                       |                     (doc_id -> raw)
                   ChromaDB                        |
                       |                           |
                       +-------------+-------------+
                                     |
                               [User Query]
                                     |
                         1. Retrieve Top Summaries
                         2. Fetch Raw Content from DocStore
                         3. Multimodal Prompt (Text + HTML + Images)
                                     |
                          [Gemini Vision Model]
                                     |
                       [Structured Financial Answer]
```

---

## Features

- **Multi-Modal Document Extraction**:
  - **Text**: Extracted per page via `pypdf` with sliding-window chunking.
  - **Tables**: Extracted via `pdfplumber`, parsed into structured HTML tables retaining rows, columns, headers, and financial figures.
  - **Images**: Extracted charts and figures, saved as local media files and encoded in base64.
- **Multi-Vector Retrieval**: Solves the semantic mismatch problem by embedding high-level summaries into ChromaDB while passing full raw data to the LLM during answer generation.
- **Multimodal Synthesis**: Sends text, HTML tables, and actual image bytes directly to Google Gemini Vision for accurate, grounded financial QA.
- **FastAPI REST API**: Endpoints for document upload, dataset ingestion, and interactive QA.
- **Rich CLI**: Beautiful terminal user interface with progress bars and syntax formatting for easy local use.

---

## Quick Start

### 1. Prerequisites & Virtual Environment

```powershell
# Clone or navigate to the repository
cd C:\Dxdong\financial-multimodal-rag

# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### 2. Configuration

Copy the example environment file and set your Gemini API key:

```powershell
copy .env.example .env
```

Edit `.env`:

```env
GEMINI_API_KEY=your_google_gemini_api_key_here
APP_ENV=development
DATA_DIR=data
CHROMA_DIR=data/vectorstore
COLLECTION_NAME=financial_reports
EMBEDDING_MODEL=gemini-embedding-001
CHAT_MODEL=gemini-3.6-flash
```

---

## Usage

### CLI Commands

**1. Ingest PDF Reports:**
```powershell
python -m financial_rag.cli ingest data/raw/sample_report.pdf
```

**2. Ingest Financial Benchmark Datasets (ViFinTab, etc.):**
```powershell
python -m financial_rag.cli ingest-dataset data/raw/datasets/huggingface/ViFinTab_git
```

**3. Ask a Financial Question:**
```powershell
python -m financial_rag.cli ask "Doanh thu và biên lợi nhuận của công ty thay đổi như thế nào trong năm qua?"
```

---

### REST API

Start the FastAPI server:

```powershell
uvicorn financial_rag.api.main:app --reload --port 8000
```

Access Swagger UI interactive docs at:
```
http://127.0.0.1:8000/docs
```

Available endpoints:
- `POST /upload` - Upload PDF files, auto-extract elements, summarize, and index.
- `POST /ask` - Query the multimodal RAG pipeline with financial questions.
- `POST /ingest` - Trigger background ingestion on a server directory or dataset path.
- `GET /health` - Service health and status.

---

## Running Tests

Execute test suite with pytest:

```powershell
pytest
```

---

## Project Structure

```text
src/financial_rag/
├── api/
│   └── main.py                  # FastAPI REST API endpoints
├── config/
│   └── settings.py              # Environment configuration & settings
├── ingestion/
│   ├── document_loader.py       # PDF loader (Text, HTML Tables, Images)
│   ├── dataset_adapters.py      # Adapters for ViFinTab, SEC-EDGAR, etc.
│   ├── models.py                # Data models (DocumentElement, Chunk, Modality)
│   └── ocr.py                   # OCR fallback hook
├── reasoning/
│   ├── multimodal_summarizer.py # Summarization for text, tables & images
│   └── financial_qa.py          # Multimodal QA synthesis with Gemini Vision
├── retrieval/
│   ├── chunking.py              # Modality-aware chunking strategies
│   ├── doc_store.py             # Persistent JSON KV-store for raw multimodal assets
│   └── vector_store.py          # ChromaDB integration with Multi-Vector retriever
├── pipeline.py                  # Core pipeline orchestrator (ingest, query)
└── cli.py                       # Rich CLI interface
data/
├── raw/                         # Raw PDF documents
├── processed/                   # Extracted images and docstore.json
└── vectorstore/                 # ChromaDB persistent directory
tests/                           # Test suite (pytest)
```

---

## License

MIT License.

