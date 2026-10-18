# Kiến Trúc Hệ Thống: RAG Hỏi Đáp Báo Cáo Tài Chính

Tài liệu đặc tả kiến trúc kỹ thuật hệ thống **Financial Multimodal RAG**, xây dựng theo chuẩn tài liệu phân tích thiết kế ngày 18/10/2026.

---

## 1. Bài Toán và Ràng Buộc Kỹ Thuật

- **Mục tiêu:** Hệ thống đọc báo cáo tài chính song ngữ (Tiếng Việt & Tiếng Anh), trả lời câu hỏi bằng ngôn ngữ của người dùng kèm trích dẫn nguồn xác thực (Mã công ty, kỳ báo cáo, số trang, bảng biểu).
- **Ràng buộc phần cứng & mô hình:**
  - LLM phục vụ: `Qwen/Qwen2.5-3B-Instruct` (fp16, chạy trên Google Colab GPU T4 16GB VRAM).
  - VRAM ước tính: Qwen2.5-3B (fp16) ~ 6-7 GB + BGE-M3 ~ 1-1.5 GB $\Rightarrow$ Tổng ~ 8-9 GB VRAM, dư dả cho context window 2-3K tokens.
  - LLM không tự tính toán số học; Python Code đảm nhiệm tính toán chính xác để triệt tiêu lỗi số học và ảo giác (hallucination).

---

## 2. Luồng Xử Lý Dữ Liệu (Pipeline Overview)

```
[Báo cáo PDF / TXT / HTML]
        │
        ▼
[Parser & Number Normalizer]
        │
   ┌────┴──────────────────────────┐
   ▼                               ▼
[Structural Chunker]        [Fact Extractor & Validator]
(300-600 tokens + Prefix)   (Line items, Roles, Unit Scales)
   │                               │
   ├───────────────┐               │
   ▼               ▼               ▼
[FAISS Dense] [SQLite FTS5] [SQLite `fact` Table]
(BGE-M3)      (BM25 Search) (Deterministic Database)
```

### Luồng Trả Lời Trực Tuyến (Online Answering Flow):
```
                       [Câu hỏi người dùng]
                                │
                                ▼
                   [Slot Extractor & Glossary]
                                │
          ┌─────────────────────┼─────────────────────┐
          ▼                     ▼                     ▼
   [Tra cứu số liệu]     [Tính toán / So sánh]   [Tra cứu văn bản]
          │                     │                     │
          ▼                     ▼                     ▼
     SQLite `fact`        Python Calculator     Hybrid Retriever
          │                     │                (Dense + FTS5)
          ▼                     ▼                     │
   Verbalize (3B)        Verbalize (3B)               ▼
          │                     │                Top 3-5 Chunks
          └──────────────┬──────┴─────────────────────┤
                         ▼                            ▼
                 Numerical Verifier              Answer (3B)
                         │                            │
                         └──────────────┬─────────────┘
                                        ▼
                            [Câu trả lời + Trích dẫn]
```

---

## 3. Thiết Kế Cơ Sở Dữ Liệu (SQLite Schema)

Hệ thống lưu trữ toàn bộ dữ liệu trong file SQLite `data/structured/financial.db`:

1. **`company`**: Danh mục doanh nghiệp (`ticker`, `name_vi`, `name_en`, `industry`, `country`).
2. **`report`**: Metadata báo cáo (`company_id`, `period_type`, `fiscal_year`, `currency`, `unit_scale`, `audit_status`, `file_path`, `is_scanned`).
3. **`metric`**: Từ điển chỉ tiêu chuẩn hóa và glossary Việt - Anh (`code`, `name_vi`, `name_en`, `statement`, `aliases`).
4. **`fact`**: Con số tài chính có cấu trúc (`report_id`, `metric_id`, `line_item`, `column_role`, `value`, `value_raw`, `page`, `table_ref`).
5. **`chunk`**: Phân đoạn ngữ cảnh (`report_id`, `section`, `chunk_type`, `page_start`, `page_end`, `context_prefix`, `content`).
6. **`chunk_fts`**: Bảng ảo SQLite FTS5 phục vụ tìm kiếm từ khóa với xếp hạng BM25.
7. **`eval_question`**: Bộ 36 câu hỏi chuẩn đánh giá benchmark đa dạng loại hình và độ khó.

---

## 4. Chuẩn Hóa Số Liệu & Tính Toán Tài Chính

- **Định dạng số:** Hỗ trợ chuẩn xác số Việt (`1.234.567,89`), số Anh (`1,234,567.89`), số âm trong ngoặc đơn `(1.234.567)` và có dấu trừ `-1.234`.
- **Hệ số tỷ lệ (`unit_scale`):** Tự động nhân theo đơn vị khai báo (Đồng, nghìn đồng, triệu đồng, tỷ đồng, USD, triệu USD).
- **Công thức tính toán (Python Financial Calculator):**
  - Biên lợi nhuận gộp: `(Lợi nhuận gộp / Doanh thu thuần) * 100%`
  - Biên lợi nhuận ròng: `(Lợi nhuận sau thuế / Doanh thu thuần) * 100%`
  - Tỷ số Nợ trên Vốn chủ sở hữu (D/E): `Tổng nợ phải trả / Vốn chủ sở hữu`
  - Tốc độ tăng trưởng cùng kỳ (YoY): `((Kỳ này - Kỳ trước) / |Kỳ trước|) * 100%`
  - So sánh liên công ty: Tự động kiểm tra tương thích đơn vị tiền tệ, cảnh báo nếu so sánh số tuyệt đối khác loại tiền.

---

## 5. Đối Soát Số Liệu (Fact Verification)

- Module `NumericalVerifier` trích xuất mọi con số trong câu trả lời sinh ra từ LLM và đối chiếu với tập số liệu gốc trong `fact` hoặc kết quả tính toán của Python.
- Nếu phát hiện con số không có căn cứ từ tài liệu gốc, hệ thống lập tức gắn cờ cảnh báo `[Cảnh báo đối soát]`.

---

## 6. Hướng Dẫn Sử Dụng CLI

```bash
# 1. Khởi tạo database và nạp từ điển chỉ tiêu
python -m financial_rag.cli seed

# 2. Thu nạp, chunking, trích xuất facts và build vector index
python -m financial_rag.cli ingest

# 3. Đặt câu hỏi trực tiếp trên terminal
python -m financial_rag.cli ask "Doanh thu thuần năm 2025 của AAA là bao nhiêu?"

# 4. Chạy bộ đánh giá 36 câu hỏi benchmark
python -m financial_rag.cli evaluate

# 5. Khởi chạy giao diện Gradio
python -m financial_rag.cli serve --port 7860
```
