# 🚀 HƯỚNG DẪN CÀI ĐẶT, CHẠY & KIỂM THỬ DỰ ÁN (RUN & TEST GUIDE)

Tài liệu này hướng dẫn chi tiết từ A-Z cách thiết lập môi trường, cấu hình mô hình (LLaMA / Gemini), thực thi kiểm thử và vận hành hệ thống **Financial Multimodal RAG**.

---

## 📑 MỤC LỤC
1. [Yêu Cầu Tiên Quyết & Cài Đặt](#1-yêu-cầu-tiên-quyết--cài-đặt)
2. [Cấu Hình Biến Môi Trường (.env)](#2-cấu-hình-biến-môi-trường-env)
3. [Những Lưu Ý Quan Trọng Hàng Đầu](#3-những-lưu-ý-quan-trọng-hàng-đầu)
4. [Chạy Kiểm Thử Tự Động (Tests)](#4-chạy-kiểm-thử-tự-động-tests)
5. [Chạy Giao Diện Dòng Lệnh (CLI)](#5-chạy-giao-diện-dòng-lệnh-cli)
6. [Chạy Dịch Vụ Web API (FastAPI)](#6-chạy-dịch-vụ-web-api-fastapi)
7. [Chạy Thử Nghiệm Với Jupyter / Colab](#7-chạy-thử-nghiệm-với-jupyter--colab)
8. [Khắc Phục Sự Cố Thường Gặp (Troubleshooting)](#8-khắc-phục-sự-cố-thường-gặp-troubleshooting)

---

## 1. Yêu Cầu Tiên Quyết & Cài Đặt

* **Hệ điều hành**: Windows 10/11, macOS, hoặc Linux.
* **Python**: Khuyến nghị **Python 3.10 - 3.12**.

### Các bước cài đặt:

```powershell
# 1. Clone hoặc mở thư mục dự án
cd d:\Financial-Multimodal-RAG-main\Financial-Multimodal-RAG-main

# 2. Tạo và kích hoạt môi trường ảo (Khuyên dùng)
python -m venv .venv
.venv\Scripts\Activate.ps1   # Trên Windows PowerShell
# source .venv/bin/activate  # Trên Linux/macOS

# 3. Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

---

## 2. Cấu Hình Biến Môi Trường (.env)

Hệ thống đã được thiết lập mặc định sử dụng mô hình **LLaMA** (`LLM_PROVIDER=llama`).

Mở file `.env` tại thư mục gốc và cấu hình API key:

### 🌟 Tùy chọn 1: Dùng Groq Cloud (Siêu tốc & Miễn phí hạn mức cao - Khuyên dùng)
Lấy API key miễn phí tại: [https://console.groq.com](https://console.groq.com)

```env
LLM_PROVIDER=llama

# Cấu hình Groq
LLAMA_BASE_URL=https://api.groq.com/openai/v1
LLAMA_API_KEY=gsk_your_groq_api_key_here
LLAMA_CHAT_MODEL=llama-3.3-70b-versatile

# Giữ lại GEMINI_API_KEY để hệ thống làm fallback tạo vector embeddings
GEMINI_API_KEY=your_gemini_api_key_here
EMBEDDING_MODEL=gemini-embedding-001
```

> *Mẹo Vision*: Nếu bạn muốn dùng LLaMA phân tích trực tiếp hình ảnh biểu đồ tài chính, đổi thành:  
> `LLAMA_CHAT_MODEL=llama-3.2-11b-vision-preview`

### 💻 Tùy chọn 2: Chạy LLaMA Offline hoàn toàn qua Ollama Local
Cài đặt Ollama từ [ollama.com](https://ollama.com), sau đó kéo mô hình:
```bash
ollama run llama3.3:70b
# hoặc mô hình đọc ảnh:
ollama run llama3.2-vision:11b
```
Cấu hình `.env`:
```env
LLM_PROVIDER=llama
LLAMA_BASE_URL=http://localhost:11434/v1
LLAMA_API_KEY=ollama
LLAMA_CHAT_MODEL=llama3.3:70b
```

### 🌐 Tùy chọn 3: Chuyển lại Google Gemini gốc
```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key_here
CHAT_MODEL=gemini-3.6-flash
EMBEDDING_MODEL=gemini-embedding-001
```

---

## 3. Những Lưu Ý Quan Trọng Hàng Đầu

> [!IMPORTANT]
> **1. Bắt buộc set biến `PYTHONPATH` trên Windows**  
> Vì mã nguồn nằm trong thư mục `src/`, mỗi khi mở cửa sổ PowerShell mới để chạy lệnh, bạn cần chạy:
> ```powershell
> $env:PYTHONPATH="src"
> ```
> *(Nếu không, Python sẽ báo lỗi `ModuleNotFoundError: No module named 'financial_rag'`)*.

> [!TIP]
> **2. Giới hạn số trang khi nạp PDF (`--max-pages`)**  
> Báo cáo tài chính thường rất dài (50 - 100+ trang). Lần đầu thử nghiệm, hãy dùng tham số `--max-pages 5` để nạp nhanh một vài trang quan trọng chứa bảng BCTC:
> ```powershell
> python -m financial_rag.cli ingest data/raw/sample.pdf --max-pages 5
> ```

> [!NOTE]
> **3. Bộ Vector Embeddings**  
> LLaMA là mô hình sinh văn bản (Chat completion). Để chuyển văn bản/bảng biểu thành vector tìm kiếm trong ChromaDB, hệ thống cần một model embedding. Bạn có thể sử dụng `GEMINI_API_KEY` (miễn phí) làm embedding engine hoặc cấu hình `LLAMA_EMBEDDING_MODEL`.

---

## 4. Chạy Kiểm Thử Tự Động (Tests)

Hệ thống tích hợp bộ 98 bài kiểm thử bao phủ toàn bộ pipeline từ bóc tách bảng, chuẩn hóa FinQA, sandbox an toàn đến LLaMA inference.

### Chạy toàn bộ test suite:
```powershell
python -m pytest
```

### Chạy riêng từng module cụ thể:
```powershell
# Kiểm thử client LLaMA & Gemini (Text, Multimodal Vision, PoT):
python -m pytest tests/test_llm_client.py -v

# Kiểm thử bộ chuẩn hóa số liệu kế toán FinQA (số âm, tiền tệ, header):
python -m pytest tests/test_financial_normalizer.py -v

# Kiểm thử Hộp cát an toàn Program-of-Thought (AST validation & sandbox):
python -m pytest tests/test_pot_sandbox.py -v

# Kiểm thử bộ nạp tài liệu đa phương thức (Table HTML, Image Base64):
python -m pytest tests/test_multimodal_loader.py -v

# Kiểm thử bộ tái xếp hạng ngữ cảnh (Context Re-ranker):
python -m pytest tests/test_reranker.py -v
```

---

## 5. Chạy Giao Diện Dòng Lệnh (CLI)

Giao diện CLI cung cấp các lệnh tương tác nhanh với tài liệu tài chính.

### Bước 1: Nạp tài liệu PDF vào cơ sở tri thức (`ingest`)
Đặt file PDF báo cáo tài chính vào `data/raw/` (ví dụ `bctc_2023.pdf`):

```powershell
$env:PYTHONPATH="src"

# Cách 1: Nạp có tóm tắt ngữ nghĩa (Khuyên dùng cho chất lượng cao)
python -m financial_rag.cli ingest data/raw/bctc_2023.pdf --max-pages 10

# Cách 2: Nạp siêu tốc không gọi LLM tóm tắt (Tiết kiệm hạn ngạch API)
python -m financial_rag.cli ingest data/raw/bctc_2023.pdf --no-summarize
```

### Bước 2: Nạp bộ dữ liệu chuẩn (Tùy chọn)
Hệ thống hỗ trợ nạp trực tiếp tập dữ liệu FinQA hoặc ViFinTab:
```powershell
$env:PYTHONPATH="src"
python -m financial_rag.cli ingest-dataset finqa --split train --sample-size 50
```

### Bước 3: Đặt câu hỏi phân tích tài chính (`ask`)

* **Truy vấn thông thường (Tự động nhận diện câu hỏi định tính/định lượng)**:
  ```powershell
  $env:PYTHONPATH="src"
  python -m financial_rag.cli ask "Doanh thu thuần và lợi nhuận sau thuế năm 2023 là bao nhiêu?"
  ```

* **Ép buộc chạy Program-of-Thought (`--pot`) để tính toán chính xác tuyệt đối**:
  ```powershell
  $env:PYTHONPATH="src"
  python -m financial_rag.cli ask "Tính biên lợi nhuận ròng năm 2023 và so sánh với năm 2022?" --pot
  ```
  *(Hệ thống sẽ hiển thị: Đoạn mã Python do LLaMA viết, kết quả tính từ Sandbox, câu trả lời phân tích và bảng các nguồn trích dẫn kèm số trang).*

* **Tùy chỉnh số lượng ngữ cảnh trích xuất (`--limit`)**:
  ```powershell
  $env:PYTHONPATH="src"
  python -m financial_rag.cli ask "Chi tiết các khoản nợ vay ngắn hạn?" --limit 8
  ```

---

## 6. Chạy Dịch Vụ Web API (FastAPI)

Hệ thống có sẵn một backend REST API hoàn chỉnh để kết nối với Web UI hoặc App bên ngoài.

### 1. Khởi động API Server:
```powershell
$env:PYTHONPATH="src"
python -m uvicorn financial_rag.api.main:app --reload --port 8000
```

### 2. Mở giao diện Swagger UI:
Truy cập trình duyệt tại: **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**

### 3. Các API chính:
* `POST /upload`: Tải file PDF BCTC lên, hệ thống tự động bóc tách và nạp vào vector store.
* `POST /ask`: Truy vấn câu hỏi tài chính bằng JSON body:
  ```json
  {
    "question": "Biên lợi nhuận gộp quý 3 thay đổi thế nào?",
    "limit": 6,
    "use_pot": true
  }
  ```
* `POST /ingest`: Nạp toàn bộ file trong thư mục `data/raw`.
* `GET /health`: Kiểm tra tình trạng hoạt động của service.

---

## 7. Chạy Thử Nghiệm Với Jupyter / Colab

Nếu muốn thực thi trực quan theo từng bước, hãy mở sổ tay có sẵn:
* Tệp: [`notebooks/financial_multimodal_rag_colab.ipynb`](notebooks/financial_multimodal_rag_colab.ipynb)
* Hỗ trợ chạy trên máy nội bộ qua Jupyter Lab hoặc import lên Google Colab.

---

## 8. Khắc Phục Sự Cố Thường Gặp (Troubleshooting)

| Lỗi gặp phải | Nguyên nhân | Cách khắc phục |
| :--- | :--- | :--- |
| `No module named 'financial_rag'` | Chưa đặt `PYTHONPATH` trỏ tới `src` | Chạy `$env:PYTHONPATH="src"` trước khi gõ lệnh. |
| `LLAMA_API_KEY chưa được cấu hình` | Chưa có key trong file `.env` | Mở file `.env`, điền API key Groq hoặc OpenAI-compatible vào dòng `LLAMA_API_KEY=...`. |
| `RateLimitError / 429 Too Many Requests` | Gửi quá nhiều request liên tục lên API | Dùng cờ `--no-summarize` khi `ingest` hoặc giảm `--max-pages`. |
| `Blocked: Mã nguồn vi phạm chính sách AST` | PoT sinh code có lệnh nguy hiểm (`import`, `open`) | Hộp cát Sandbox đã tự động chặn lệnh không an toàn và chuyển sang trả lời bằng LLM thuần. |
| `pytest: The term 'pytest' is not recognized` | Môi trường PowerShell chưa nhận binary của pytest | Chạy lệnh qua module Python: `python -m pytest`. |
