# Kiến trúc local — cập nhật 29-09-2026

Nâng cấp đã triển khai: lưu PDF theo SHA-256 trong uploads; UI `/`; table chunks riêng;
OCR tùy chọn cho trang ít text; retrieval TF-IDF + traversal Metric/Fact/Chunk;
lọc document_id; Responses API tùy chọn; calculator Decimal qua `/calculate`.
Kho graph/chunks là JSON cho một tiến trình, khóa trong service khi ingest/retrieve.
Tài liệu và đoạn nguồn là dữ liệu không đáng tin: UI dùng textContent, LLM được hướng dẫn
không làm theo chỉ dẫn trong nguồn. Kiểm tra citation ID không xác nhận citation đúng nghĩa.

## Luồng dữ liệu

1. `DocumentLoader` đọc PDF, TXT hoặc HTML; giữ số trang và nguồn.
2. `Chunker` chia theo trang/đoạn với overlap.
3. `FinancialFactExtractor` nhận diện chỉ tiêu, giá trị, đơn vị, năm và ticker.
4. `KnowledgeGraph` ghi nodes/edges cùng provenance.
5. `HybridRetriever` kết hợp TF-IDF cosine và graph expansion.
6. `AnswerService` trả evidence-first hoặc gọi LLM nếu được cấu hình.

## Ranh giới MVP

- Chưa huấn luyện GNN: graph hiện dùng cho entity linking, traversal và provenance. GNN chỉ nên thêm khi đã có bài toán dự đoán node/edge và tập nhãn rõ ràng.
- OCR dùng PyMuPDF/Tesseract, cần tessdata cài riêng. Bảng PDF text dùng find_tables;
  bảng scan có spatial extraction, mã chỉ tiêu và kỳ báo cáo hai cột VND.
  Có OCR phục hồi mã dòng, kiểm tra công thức và đối chiếu số dư tiền giữa báo cáo.
  Chart/VLM và bộ đọc bảng tổng quát chưa triển khai.
- Chưa phụ thuộc Neo4j: graph JSON/NetworkX giúp chạy local ngay. Có thể thay adapter bằng Neo4j khi corpus tăng.
- Calculator cho hai fact rõ ràng đã có; tự lập chương trình từ câu hỏi và benchmark FinQA chưa có.

## Phase tiếp theo

1. Table detector + table structure model từ VietFinTab.
2. LLM/VLM structured extraction theo JSON Schema và entity resolution.
3. Dense multilingual embeddings + reranker.
4. Mở rộng calculator đã có sang các phép phân tích nhiều kỳ.
5. Evaluation: Recall@k cho retrieval, entity/relation F1, execution accuracy, answer EM/F1 và citation accuracy.

