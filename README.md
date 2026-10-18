# 📊 Financial Multimodal RAG — Hệ Thống Hỏi Đáp Báo Cáo Tài Chính

Hệ thống RAG đọc và tra cứu Báo cáo tài chính (BCTC) song ngữ (tiếng Việt & tiếng Anh), kết hợp bảng số liệu có cấu trúc **SQLite**, tìm kiếm lai **Hybrid Search (BGE-M3 + FTS5 BM25)**, tính toán tài chính tiền tệ chính xác bằng **Python** và diễn đạt câu trả lời kèm trích dẫn nguồn xác thực với mô hình **Qwen2.5-3B-Instruct**.

Xây dựng tuân thủ 100% theo tài liệu thiết kế [phan_tich_thiet_ke_rag_tai_chinh.md](../phan_tich_thiet_ke_rag_tai_chinh.md).

---

## 🌟 Tính Năng Nổi Bật

1. **Số liệu chuẩn xác (Zero Math Hallucination):** 
   - LLM không tự thực hiện phép tính số học.
   - Các phép tính tài chính (Biên lợi nhuận gộp/ròng, tỷ lệ Nợ/Vốn chủ sở hữu D/E, tăng trưởng cùng kỳ YoY, đối sánh liên công ty) được thực thi bằng Python Calculator.
2. **Cơ sở dữ liệu số liệu có cấu trúc (`fact` table):**
   - Lưu trữ chi tiết từng dòng báo cáo (`line_item`), vai trò cột (`current`, `prior_year_same_period`, `year_start`), giá trị đã quy đổi scale và chuỗi gốc (`value_raw`).
   - Kiểm tra tính cân đối kế toán: $Tổng\ tài\ sản = Nợ\ phải\ trả + Vốn\ chủ\ sở\ hữu$.
3. **Tìm kiếm lai Hybrid Retrieval + Metadata Filtering:**
   - Kết hợp Dense Vector Search (BGE-M3 / FAISS) và Lexical Search (SQLite FTS5 BM25).
   - Lọc theo mã cổ phiếu (`ticker`), năm tài chính (`fiscal_year`) và kỳ báo cáo (`period_type`).
4. **Trích dẫn nguồn minh bạch (Provenance):**
   - Mọi câu trả lời đều ghi rõ: `[MÃ_CK | KỲ/NĂM | Trang X | Bảng/Mục]`.
5. **Đối soát số liệu tự động (Numerical Verification):**
   - Quét kiểm tra mọi con số xuất hiện trong câu trả lời với số liệu gốc từ báo cáo, gắn cờ cảnh báo nếu xuất hiện số lạ.
6. **Xử lý câu hỏi bẫy & từ chối trả lời lịch sự:**
   - Tự động nhận diện và từ chối các câu hỏi dự báo giá cổ phiếu tương lai, khuyến nghị mua/bán hoặc thông tin không có trong tài liệu.
7. **Tương thích Google Colab GPU T4:**
   - Chạy mượt mà trên môi trường Colab T4 miễn phí với mô hình `Qwen2.5-3B-Instruct` (fp16) chỉ chiếm ~8-9 GB VRAM.

---

## 📁 Cấu Trúc Thư Mục

```
Financial-Multimodal-RAG-master/
├── data/
│   ├── manifest.csv               # Danh mục báo cáo và metadata
│   ├── raw/                       # Báo cáo gốc dạng PDF / TXT
│   │   ├── AAA/                   # Nhựa An Phát Xanh (BCTC kiểm toán 2025)
│   │   ├── NVDA/                  # NVIDIA Corporation (10-K Report FY2025)
│   │   ├── HPG/                   # Tập đoàn Hòa Phát (BCTC H1/2026)
│   │   └── VNM/                   # Vinamilk (BCTC H1/2026)
│   ├── parsed/                    # Kết quả parse: document.md, tables.json
│   ├── structured/
│   │   ├── financial.db           # SQLite database
│   │   └── metrics_seed.csv       # Từ điển 16 chỉ tiêu chung & alias Việt - Anh
│   ├── index/
│   │   ├── faiss.index            # Dense vector index
│   │   └── chunk_ids.json         # Mapping vector ID -> chunk_id
│   └── eval/
│       └── questions.csv          # 36 câu hỏi đánh giá benchmark
├── docs/
│   └── ARCHITECTURE.md            # Thiết kế kiến trúc chi tiết
├── notebooks/
│   └── colab_financial_rag.ipynb  # Notebook demo chạy trên Google Colab T4
├── src/
│   └── financial_rag/
│       ├── config.py              # Cấu hình Pydantic Settings
│       ├── cli.py                 # Rich CLI (seed, ingest, ask, eval, serve)
│       ├── db/                    # SQLite schema, repository, seed
│       ├── parser/                # Number normalizer, PDF/Table parser, Chunker
│       ├── extractor/             # Slot extractor, Fact extractor, Validator
│       ├── retrieval/             # BGE-M3 Dense, FTS5 Lexical, Hybrid fusion
│       ├── reasoning/             # Calculator, Router, Numerical Verifier, Engine
│       ├── llm/                   # Prompt templates, Qwen2.5-3B loader
│       ├── ui/                    # Gradio Web UI
│       └── evaluation/            # Automated benchmark evaluator
├── tests/                         # Unit tests toàn diện (100% pass)
├── requirements.txt
└── pyproject.toml
```

---

## 🚀 Hướng Dẫn Cài Đặt & Sử Dụng

### 1. Cài đặt môi trường
```bash
# Clone hoặc mở thư mục dự án
cd "Financial-Multimodal-RAG-master"

# Cài đặt dependencies cơ bản
pip install -r requirements.txt
```

### 2. Khởi tạo Database & Nạp Dữ Liệu
```bash
# Bước 1: Khởi tạo database và từ điển chỉ tiêu
python -m financial_rag.cli seed

# Bước 2: Parse báo cáo, trích xuất facts và build vector index
python -m financial_rag.cli ingest
```

### 3. Đặt câu hỏi trực tiếp trên Terminal
```bash
# Tra cứu số liệu
python -m financial_rag.cli ask "Doanh thu thuần năm 2025 của Công ty Nhựa An Phát Xanh (AAA) là bao nhiêu?"

# Tính toán tài chính
python -m financial_rag.cli ask "Biên lợi nhuận gộp của AAA năm 2025 là bao nhiêu phần trăm?"

# Tra cứu báo cáo tiếng Anh
python -m financial_rag.cli ask "What was NVIDIA's total revenue for fiscal year 2025?"

# Câu hỏi bẫy (Hệ thống từ chối lịch sự, không bịa đặt)
python -m financial_rag.cli ask "Dự báo giá cổ phiếu AAA trong 3 tháng tới sẽ tăng hay giảm?"
```

### 4. Chạy bộ đánh giá Benchmark (36 câu hỏi)
```bash
python -m financial_rag.cli evaluate
```

### 5. Khởi chạy Giao diện Web (Gradio)
```bash
# Chạy local tại http://localhost:7860
python -m financial_rag.cli serve --port 7860

# Tạo link public truy cập từ xa (Google Colab)
python -m financial_rag.cli serve --port 7860 --share
```

### 6. Chạy Kiểm Thử Tự Động (Unit Tests)
```bash
python -m pytest tests
```
Tất cả 13 test cases đều vượt qua thành công (100% pass).

---

## 🖥️ Chạy trên Google Colab với GPU T4

1. Mở file notebook [`notebooks/colab_financial_rag.ipynb`](notebooks/colab_financial_rag.ipynb) trên Google Colab.
2. Chọn Runtime: **Python 3 with T4 GPU**.
3. Chạy lần lượt các cells để nạp mô hình `Qwen/Qwen2.5-3B-Instruct` (fp16) và nhận link Gradio công khai (`share=True`).
