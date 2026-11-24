# Kiến Trúc Kỹ Thuật (Architecture & Schema)

Tài liệu mô tả chi tiết kiến trúc của hệ thống `Multimodal RAG Financial`.

---

## 1. Triết Lý Thiết Kế: Deterministic Over Generative

Hầu hết các hệ thống RAG hiện tại sử dụng Large Language Model (LLM) để đọc văn bản và sinh câu trả lời tự nhiên. Tuy nhiên, trong lĩnh vực báo cáo tài chính, **ảo giác số liệu (hallucination)** và **lỗi suy luận nhân quả** gây ra rủi ro nghiêm trọng.

Hệ thống này được thiết kế theo nguyên tắc:
1. **Dữ kiện số học phải có bằng chứng tọa độ nguồn**: Mỗi số trích xuất (`Fact`) phải gắn liền với trang PDF, tọa độ bounding box (`bbox`), mã dòng kế toán và giá trị văn bản gốc (`raw_value`).
2. **Không dùng LLM để sinh số hoặc tính toán**: Số học được tính toán bằng module `calculator.py` sử dụng chuẩn `Decimal` với các ràng buộc khắt khe.
3. **Từ chối trả lời (Abstention) khi thiếu bằng chứng**: Khi câu hỏi vượt quá bộ luật đã định nghĩa hoặc thiếu sự đồng nhất kỳ/phạm vi, hệ thống chỉ hiển thị đoạn trích tra cứu và nêu rõ chưa đủ bằng chứng.

---

## 2. Các Tầng Kiến Trúc

```text
[PDF / TXT / Scan]
       │
       ▼
[Tầng Ingestion & Layout]
       ├── PyMuPDF Text / Tesseract OCR / VietOCR
       └── Spatial Table Extractor (Tách dòng, nhận diện mốc kỳ DD/MM/YYYY)
       │
       ▼
[Tầng Kiểm Tra Chất Lượng (Quality Assurance)]
       ├── Kiểm tra công thức nội bộ (VD: Lợi nhuận gộp = Doanh thu thuần - Giá vốn)
       └── Đối chiếu chéo (Tiền cuối kỳ trên LCTT = Tiền trên BĐKT tại đúng ngày chốt)
       │
       ▼
[Tầng Lưu Trữ & Tri Thức (Knowledge Graph & Index)]
       ├── NetworkX MultiDiGraph (Nodes: Company, Report, Period, Metric, Fact, Chunk)
       └── TF-IDF / BM25 / SQLite Full-Text Index
       │
       ▼
[Tầng Phân Tích & Hỏi Đáp (Service & Answers)]
       ├── Cổng từ vựng (Vocabulary Gating)
       ├── Calculator (Tăng trưởng, chênh lệch, tỷ số)
       └── API FastAPI & Web Dashboard
```

---

## 3. Lược Đồ Đồ Thị Tri Thức (Knowledge Graph Schema)

Đồ thị được mô hình hóa bằng `networkx.MultiDiGraph`:

### Các loại Node:
- `Company`: Doanh nghiệp, thuộc tính: `ticker`.
- `Report`: Báo cáo tài chính, thuộc tính: `source`, `ticker`, `year`, `scope`, `name`.
- `Period`: Kỳ kế toán, thuộc tính: `year` hoặc `period_label` (VD: `2024-Q1`).
- `Metric`: Chỉ tiêu tài chính theo Thông tư 200 (VD: `Doanh thu thuần`, `Lợi nhuận sau thuế`).
- `Fact`: Số liệu cụ thể trích xuất được từ báo cáo.
- `Chunk`: Đoạn văn bản hoặc dòng bảng gốc, lưu tọa độ `bbox` và số trang.

### Các Quan Hệ (Edges):
- `(Company) -[HAS_REPORT]-> (Report)`
- `(Report) -[FOR_PERIOD]-> (Period)`
- `(Report) -[MENTIONS]-> (Chunk)`
- `(Chunk) -[NEXT]-> (Chunk)` (Quan hệ thứ tự đọc văn bản)
- `(Report) -[HAS_FACT]-> (Fact)`
- `(Fact) -[MEASURES]-> (Metric)`
- `(Fact) -[SUPPORTED_BY]-> (Chunk)`
