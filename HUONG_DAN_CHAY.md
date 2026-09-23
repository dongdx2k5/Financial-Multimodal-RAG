# 🚀 HƯỚNG DẪN CÀI ĐẶT, CHẠY & KIỂM THỬ DỰ ÁN

> **Financial Multimodal RAG** — Hệ thống Retrieval-Augmented Generation Đa phương thức chuyên sâu cho Phân tích Báo cáo Tài chính.  
> Kết hợp **Knowledge Graph + Document Tree + ChromaDB + Program-of-Thought (PoT) + Multi-Provider LLM (Gemini / LLaMA)**.

---

## 📑 MỤC LỤC

1. [Yêu Cầu Tiên Quyết & Cài Đặt](#1-yêu-cầu-tiên-quyết--cài-đặt)
2. [Cấu Hình Biến Môi Trường (.env)](#2-cấu-hình-biến-môi-trường-env)
3. [Những Lưu Ý Quan Trọng Hàng Đầu](#3-những-lưu-ý-quan-trọng-hàng-đầu)
4. [Chạy Kiểm Thử Tự Động (Tests)](#4-chạy-kiểm-thử-tự-động-tests)
5. [Chạy Giao Diện Dòng Lệnh (CLI)](#5-chạy-giao-diện-dòng-lệnh-cli)
6. [Chạy Dịch Vụ Web API (FastAPI)](#6-chạy-dịch-vụ-web-api-fastapi)
7. [Mô Phỏng Đồ Thị Tri Thức](#7-mô-phỏng-đồ-thị-tri-thức)
8. [Chạy Thử Nghiệm Với Jupyter / Colab](#8-chạy-thử-nghiệm-với-jupyter--colab)
9. [Khắc Phục Sự Cố Thường Gặp (Troubleshooting)](#9-khắc-phục-sự-cố-thường-gặp-troubleshooting)

---

## 1. Yêu Cầu Tiên Quyết & Cài Đặt

- **Hệ điều hành**: Windows 10/11, macOS, hoặc Linux
- **Python**: Phiên bản **3.10 – 3.12** (khuyên dùng)
- **Git**: Đã cài đặt và cấu hình tài khoản GitHub

### Các Bước Cài Đặt:

```powershell
# 1. Di chuyển vào thư mục dự án
cd d:\HMNC_2\Financial-Multimodal-RAG-main

# 2. (Tuỳ chọn) Tạo và kích hoạt môi trường ảo
python -m venv .venv
.venv\Scripts\Activate.ps1       # Windows PowerShell
# source .venv/bin/activate      # Linux / macOS

# 3. Cài đặt toàn bộ thư viện cần thiết
pip install -r requirements.txt

# 4. (Tuỳ chọn) Cài thêm PyVis để xuất đồ thị tri thức dạng HTML
pip install pyvis
```

---

## 2. Cấu Hình Biến Môi Trường (.env)

Mở file `.env` tại thư mục gốc dự án và điền API key của bạn:

### 🦙 Lựa Chọn 1 — Dùng Groq Cloud (Khuyên Dùng: Miễn phí, Siêu nhanh ~300 tok/s)

Lấy API key miễn phí tại: **[https://console.groq.com](https://console.groq.com)**

```env
LLM_PROVIDER=llama

LLAMA_BASE_URL=https://api.groq.com/openai/v1
LLAMA_API_KEY=gsk_your_groq_api_key_here
LLAMA_CHAT_MODEL=llama-3.3-70b-versatile

# Dùng Gemini làm engine embedding vector (bắt buộc nếu dùng LLaMA)
GEMINI_API_KEY=your_gemini_api_key_here
EMBEDDING_MODEL=gemini-embedding-001
```

> **Mẹo Vision**: Để LLaMA phân tích biểu đồ tài chính (hình ảnh), đổi:
> `LLAMA_CHAT_MODEL=llama-3.2-11b-vision-preview`

### 💻 Lựa Chọn 2 — Chạy LLaMA Offline qua Ollama (Không cần Internet)

Cài Ollama từ [ollama.com](https://ollama.com), sau đó kéo model:

```powershell
ollama run llama3.3:70b
# hoặc model đọc ảnh:
ollama run llama3.2-vision:11b
```

Cấu hình `.env`:

```env
LLM_PROVIDER=llama
LLAMA_BASE_URL=http://localhost:11434/v1
LLAMA_API_KEY=ollama
LLAMA_CHAT_MODEL=llama3.3:70b
```

### 🌐 Lựa Chọn 3 — Dùng Google Gemini

Lấy API key miễn phí tại: **[https://aistudio.google.com/apikey](https://aistudio.google.com/apikey)**

```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key_here
CHAT_MODEL=gemini-3.6-flash
EMBEDDING_MODEL=gemini-embedding-001
```

---

## 3. Những Lưu Ý Quan Trọng Hàng Đầu

> [!IMPORTANT]
> **Bắt buộc set biến `PYTHONPATH` trên Windows mỗi khi mở PowerShell mới:**
> ```powershell
> $env:PYTHONPATH="src"
> ```
> Nếu không, Python sẽ báo lỗi `ModuleNotFoundError: No module named 'financial_rag'`.

> [!TIP]
> **Giới hạn số trang khi nạp PDF lần đầu** để test nhanh và tiết kiệm quota API:
> ```powershell
> python -m financial_rag.cli ingest data/raw/bctc.pdf --max-pages 5
> ```

> [!NOTE]
> **Về Vector Embeddings**: LLaMA chỉ là model chat, không tạo được embeddings. Hệ thống dùng **Gemini Embedding API** (miễn phí) để tạo vector. Bắt buộc cấu hình `GEMINI_API_KEY` dù bạn đang dùng LLaMA làm LLM chính.

---

## 4. Chạy Kiểm Thử Tự Động (Tests)

Hệ thống tích hợp bộ kiểm thử bao phủ toàn pipeline từ bóc tách bảng, chuẩn hóa FinQA, sandbox an toàn đến LLM inference.

### Chạy Toàn Bộ Test Suite:

```powershell
$env:PYTHONPATH="src"
python -m pytest
```

### Chạy Riêng Từng Module:

```powershell
$env:PYTHONPATH="src"

# Kiểm thử LLM Client (LLaMA & Gemini, Text + Vision + PoT):
python -m pytest tests/test_llm_client.py -v

# Kiểm thử bộ chuẩn hóa số liệu kế toán FinQA (số âm, tiền tệ, header):
python -m pytest tests/test_financial_normalizer.py -v

# Kiểm thử Safe Python Sandbox (AST validation & timeout):
python -m pytest tests/test_pot_sandbox.py -v

# Kiểm thử bộ nạp tài liệu đa phương thức (Table HTML, Image Base64):
python -m pytest tests/test_multimodal_loader.py -v

# Kiểm thử bộ tái xếp hạng ngữ cảnh (Context Re-ranker):
python -m pytest tests/test_reranker.py -v

# Kiểm thử Dataset Adapters (FinQA, ViFinTab, SEC EDGAR):
python -m pytest tests/test_dataset_adapters.py -v
```

---

## 5. Chạy Giao Diện Dòng Lệnh (CLI)

### Bước 1 — Set PYTHONPATH (Bắt Buộc):

```powershell
$env:PYTHONPATH="src"
```

### Bước 2 — Nạp Dữ Liệu Vào Cơ Sở Tri Thức (Ingest):

**Cách A: Nạp tập dữ liệu FinQA có sẵn (Không cần PDF, chạy ngay được):**

```powershell
# Nạp 50 mẫu từ tập train FinQA
python -m financial_rag.cli ingest-dataset finqa --split train --sample-size 50

# Nạp tập validation
python -m financial_rag.cli ingest-dataset finqa --split validation --sample-size 20

# Nạp tập test
python -m financial_rag.cli ingest-dataset finqa --split test --sample-size 20
```

**Cách B: Nạp tập dữ liệu ViFinTab (Tiếng Việt):**

```powershell
python -m financial_rag.cli ingest-dataset vifintab --split train --sample-size 30
```

**Cách C: Nạp tập SEC EDGAR:**

```powershell
python -m financial_rag.cli ingest-dataset sec_edgar --sample-size 20
```

**Cách D: Nạp file PDF Báo cáo Tài chính thực tế:**

```powershell
# Nạp có tóm tắt ngữ nghĩa LLM (chất lượng cao - mất thời gian hơn)
python -m financial_rag.cli ingest data/raw/bctc_2023.pdf --max-pages 10

# Nạp siêu tốc không gọi LLM (tiết kiệm API quota)
python -m financial_rag.cli ingest data/raw/bctc_2023.pdf --no-summarize

# Nạp toàn bộ file trong thư mục data/raw/
python -m financial_rag.cli ingest data/raw/
```

### Bước 3 — Đặt Câu Hỏi Phân Tích Tài Chính (Ask):

**Câu hỏi định tính (tự động nhận diện):**

```powershell
python -m financial_rag.cli ask "What is the net change in net revenue during 2015?"

python -m financial_rag.cli ask "Doanh thu thuần và lợi nhuận sau thuế năm 2023 là bao nhiêu?"
```

**Câu hỏi tính toán — Bật Program-of-Thought (`--pot`) để đạt độ chính xác 100%:**

```powershell
python -m financial_rag.cli ask "What is the percentage change in net revenue from 2014 to 2015?" --pot

python -m financial_rag.cli ask "Tính biên lợi nhuận ròng năm 2023 và so sánh với năm 2022?" --pot
```

**Tùy chỉnh số lượng ngữ cảnh trích xuất (`--limit`):**

```powershell
python -m financial_rag.cli ask "Chi tiết các khoản nợ vay ngắn hạn?" --limit 8
```

---

## 6. Chạy Dịch Vụ Web API (FastAPI)

### Khởi Động API Server:

```powershell
$env:PYTHONPATH="src"
python -m uvicorn financial_rag.api.main:app --reload --port 8000
```

### Mở Giao Diện Swagger UI Tương Tác:

Truy cập trình duyệt tại: **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**

### Các Endpoint Chính:

| Endpoint | Method | Chức Năng |
| :--- | :---: | :--- |
| `/health` | GET | Kiểm tra trạng thái hoạt động của service |
| `/upload` | POST | Tải file PDF BCTC lên, hệ thống tự động bóc tách và nạp vào vector store |
| `/ask` | POST | Truy vấn câu hỏi tài chính bằng JSON body |
| `/ingest` | POST | Nạp toàn bộ file trong thư mục `data/raw` |

### Ví Dụ Gọi API `/ask` bằng JSON:

```json
{
  "question": "What was the percentage change in net revenue from 2014 to 2015?",
  "limit": 6,
  "use_pot": true
}
```

---

## 7. Mô Phỏng Đồ Thị Tri Thức

Dự án tích hợp sẵn công cụ mô phỏng Đồ thị Tri thức (Knowledge Graph & Document Tree) trực tiếp từ dữ liệu FinQA thực tế, xuất ra file HTML tương tác.

### Chạy Script Mô Phỏng:

```powershell
python simulate_graph.py
```

Script sẽ:
1. Đọc mẫu dữ liệu từ `data/raw/datasets/huggingface/finqa_parquet/test.parquet`
2. Xây dựng các Node: `Company`, `Document`, `Table`, `Metric`, `TimePeriod`, `Value`, `Footnote`
3. Xây dựng các Edge quan hệ kế toán: `CONTAINS`, `HAS_VALUE`, `REPORTED_IN`, `DERIVED_FROM`, `CROSS_REF`
4. Xuất ra file **`knowledge_graph_simulation.html`**

### Xem Đồ Thị Tương Tác:

Mở file `knowledge_graph_simulation.html` bằng **Google Chrome** hoặc **Microsoft Edge**:

- **Kéo thả node**: Khám phá mạng lưới quan hệ tài chính
- **Phóng to / thu nhỏ**: Cuộn chuột để điều chỉnh tầm nhìn
- **Nhấp vào Node**: Xem chi tiết thuộc tính thực thể trên bảng bên trái

### Đổi Sang Mẫu FinQA Khác:

Mở file `simulate_graph.py`, tìm hàm `main()` và đổi tham số `sample_idx`:

```python
# Đổi thành 1, 2, 3... để xem đồ thị của các câu hỏi khác
sample = load_finqa_sample(sample_idx=0)
```

---

## 8. Chạy Thử Nghiệm Với Jupyter / Colab

Nếu muốn thực thi trực quan từng bước và xem kết quả trực tiếp trong notebook:

- **File notebook**: [`notebooks/financial_multimodal_rag_colab.ipynb`](notebooks/financial_multimodal_rag_colab.ipynb)
- Hỗ trợ chạy cục bộ qua **Jupyter Lab** hoặc upload lên **Google Colab**

```powershell
# Cài Jupyter nếu chưa có
pip install jupyterlab

# Khởi động Jupyter Lab
jupyter lab
```

---

## 9. Khắc Phục Sự Cố Thường Gặp (Troubleshooting)

| Lỗi Gặp Phải | Nguyên Nhân | Cách Khắc Phục |
| :--- | :--- | :--- |
| `ModuleNotFoundError: No module named 'financial_rag'` | Chưa set `PYTHONPATH` | Chạy `$env:PYTHONPATH="src"` trước khi dùng bất kỳ lệnh nào |
| `GEMINI_API_KEY chưa được cấu hình` | Thiếu Gemini key trong `.env` | Mở `.env`, thay `your_gemini_api_key_here` bằng key thật từ [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |
| `LLAMA_API_KEY chưa được cấu hình` | Thiếu Groq key trong `.env` | Mở `.env`, điền key Groq từ [console.groq.com](https://console.groq.com) vào `LLAMA_API_KEY=...` |
| `RateLimitError / 429 Too Many Requests` | Gửi quá nhiều request liên tục lên API | Dùng cờ `--no-summarize` khi ingest hoặc giảm `--max-pages` xuống 3-5 |
| `FileNotFoundError: FinQA dataset file not found` | Thiếu file parquet FinQA | Đảm bảo có file tại `data/raw/datasets/huggingface/finqa_parquet/train.parquet` |
| `Blocked: Mã nguồn vi phạm chính sách AST` | PoT sinh code có lệnh nguy hiểm (`import`, `open`) | Sandbox đã tự chặn và chuyển sang dùng LLM thuần — đây là hành vi bình thường, không phải lỗi |
| `pytest: The term 'pytest' is not recognized` | PowerShell chưa nhận binary của pytest | Dùng `python -m pytest` thay vì `pytest` trực tiếp |
| `UnicodeEncodeError: charmap codec` | Terminal Windows không hỗ trợ UTF-8 emoji | Chạy `[Console]::OutputEncoding = [System.Text.Encoding]::UTF8` hoặc dùng Windows Terminal |
| Đồ thị tri thức không hiển thị trong trình duyệt | Trình duyệt không load được Vis.js CDN | Kiểm tra kết nối Internet. File HTML cần CDN: `https://unpkg.com/vis-network/...` |

---

## 📦 Cấu Trúc Thư Mục Dự Án

```
Financial-Multimodal-RAG-main/
├── src/financial_rag/           # Mã nguồn chính
│   ├── ingestion/               # Bóc tách & chuẩn hóa tài liệu
│   │   ├── document_loader.py   # Load PDF: Text + Table HTML + Image Base64
│   │   ├── financial_normalizer.py  # Chuẩn hóa FinQA: Số âm, quy mô, header
│   │   ├── dataset_adapters.py  # FinQA / ViFinTab / SEC EDGAR adapters
│   │   └── models.py            # Schema: EntityType, RelationType, KnowledgeEntity
│   ├── retrieval/               # Lưu trữ & Truy xuất
│   │   ├── vector_store.py      # ChromaDB Vector Store
│   │   ├── doc_store.py         # DocStore KV-Store (HTML, Text, Base64)
│   │   ├── reranker.py          # Context Re-Ranker đa chiều
│   │   └── chunking.py          # Sliding-window chunking
│   ├── reasoning/               # Suy luận & LLM
│   │   ├── llm_client.py        # UnifiedLLMClient (Gemini & LLaMA)
│   │   ├── financial_qa.py      # FinancialQAEngine (RAG pipeline)
│   │   ├── pot_reasoning.py     # Program-of-Thought Engine
│   │   ├── sandbox.py           # Safe Python Execution Sandbox
│   │   └── multimodal_summarizer.py  # LLM Summarizer bảng & ảnh
│   ├── api/main.py              # FastAPI REST API
│   ├── cli.py                   # CLI interface (typer)
│   └── pipeline.py              # Hàm ingest() và query() dùng chung
├── data/
│   ├── raw/                     # PDF, dataset gốc (không đưa lên git)
│   ├── processed/               # Ảnh bóc tách, knowledge_graph.json
│   └── vectorstore/             # ChromaDB persistent store
├── tests/                       # 9 module kiểm thử tự động
├── docs/images/                 # Sơ đồ kiến trúc (SVG, PNG)
├── notebooks/                   # Jupyter / Google Colab notebook
├── simulate_graph.py            # Script mô phỏng Đồ thị Tri thức (FinQA)
├── knowledge_graph_simulation.html  # HTML tương tác đồ thị (Vis.js)
├── KIEN_TRUC.md                 # Tài liệu kiến trúc hệ thống chi tiết
├── LUU_TRU_VA_TRUY_XUAT.md     # Tài liệu cơ chế lưu trữ & truy xuất
├── requirements.txt             # Danh sách thư viện Python
├── .env                         # API keys & cấu hình (không đưa lên git)
└── .env.example                 # Mẫu cấu hình .env
```

---

## 🔗 Liên Kết Quan Trọng

| Tài Nguyên | Đường Dẫn |
| :--- | :--- |
| Tài liệu Kiến trúc hệ thống | [KIEN_TRUC.md](KIEN_TRUC.md) |
| Tài liệu Lưu trữ & Truy xuất | [LUU_TRU_VA_TRUY_XUAT.md](LUU_TRU_VA_TRUY_XUAT.md) |
| Sơ đồ kiến trúc SVG (Tiếng Việt) | [docs/images/architecture_diagram_vn.svg](docs/images/architecture_diagram_vn.svg) |
| Repository GitHub (Nhánh Thien) | [github.com/dongdx2k5/Financial-Multimodal-RAG/tree/Thien](https://github.com/dongdx2k5/Financial-Multimodal-RAG/tree/Thien) |
| Swagger API UI (khi server đang chạy) | [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) |
| Lấy Gemini API Key (miễn phí) | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |
| Lấy Groq API Key (miễn phí) | [console.groq.com](https://console.groq.com) |
