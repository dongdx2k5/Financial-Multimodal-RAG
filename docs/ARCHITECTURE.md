# Kiến Trúc Kỹ Thuật (Architecture & Schema)

Hệ thống `Multimodal RAG Financial` vận hành theo mô hình **Hybrid Local + Google Colab GPU**:
- **Máy Local**: Giữ toàn bộ Web UI, upload PDF gốc, kho lưu trữ `SQLite + FTS5`, Knowledge Graph (`NetworkX`), kiểm tra kế toán và bộ tính toán `Decimal` tất định.
- **Google Colab T4+ (tùy chọn)**: Chỉ nhận các tác vụ GPU nặng qua HTTPS API (`scripts/colab_gpu_server.py`) gồm VLM đọc biểu đồ ảnh (`Qwen2.5-VL`), phân tích ý định câu hỏi, tổng hợp giải thích có trích dẫn và Dense Embeddings (`BAAI/bge-m3`).

---

## 1. Triết Lý Thiết Kế: Deterministic Over Generative

1. **Dữ kiện số học phải có bằng chứng tọa độ nguồn**: Mỗi số trích xuất (`Fact`) gắn liền với trang PDF, tọa độ `bbox`, mã dòng kế toán và giá trị gốc (`raw_value`).
2. **Không dùng LLM để tự tính toán hoặc bịa số**: Mọi phép tăng trưởng, chênh lệch và tỷ số được tính bằng `calculator.py` dùng chuẩn `Decimal`. Khi Colab GPU phân tích ý định câu hỏi tự nhiên (`/v1/qa/parse_intent`), kết quả vẫn phải đi qua luồng truy xuất và tính toán tất định cục bộ.
3. **Hậu kiểm chống ảo giác & Từ chối khi thiếu bằng chứng**: Phản hồi tổng hợp từ Colab (`/v1/qa/synthesize`) bắt buộc qua `colab_client.validate_output` kiểm tra số liệu khớp nguyên văn và trích dẫn `[1]..[n]`.

---

## 2. Các Tầng Kiến Trúc

```text
[PDF / TXT / HTML / MD]
       │
       ▼
[Tầng Ingestion & Layout (Local hoặc Colab GPU)]
       ├── PyMuPDF Text / Tesseract OCR / VietOCR / Colab GPU OCR
       ├── Layout Detector (Heuristic hoặc YOLO ONNX qua LAYOUT_MODEL_PATH)
       ├── Spatial Table Extractor (Nối bảng đa trang, đọc kỳ quý/6T/9T/năm)
       └── ChartReader (Vector heuristic hoặc Colab VLM cho biểu đồ ảnh)
       │
       ▼
[Tầng Kiểm Tra Chất Lượng (Quality Assurance)]
       ├── Công thức nội bộ Thông tư 200 (KQKD, CĐKT, LCTT)
       └── Đối chiếu chéo tiền đầu/cuối kỳ giữa LCTT và BĐKT
       │
       ▼
[Tầng Lưu Trữ & Tri Thức (Local SQLite + Graph)]
       ├── FinancialSQLiteStore (documents, chunks, chunks_fts BM25, facts, jobs)
       └── NetworkX MultiDiGraph (Company, Report, Period, Metric, Fact, Chunk)
       │
       ▼
[Tầng Phân Tích, Hỏi Đáp & Split-Pane UI]
       ├── Deterministic QA (answers.py + Decimal calculator.py)
       ├── Hybrid Retrieval (TF-IDF + SQLite FTS5 BM25 + Graph Boost)
       └── Split-Pane Web Workspace (Chat, KPI & SVG Chart, Xác minh Fact, Graph Canvas, PDF Viewer)
```
