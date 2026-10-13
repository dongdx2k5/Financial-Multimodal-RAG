# Kiến trúc MVP

## Luồng dữ liệu

1. `DocumentLoader` đọc PDF, TXT hoặc HTML; giữ số trang và nguồn.
2. `Chunker` chia theo trang/đoạn với overlap.
3. `FinancialFactExtractor` nhận diện chỉ tiêu, giá trị, đơn vị, năm và ticker.
4. `KnowledgeGraph` ghi nodes/edges cùng provenance.
5. `HybridRetriever` kết hợp TF-IDF cosine và graph expansion.
6. `AnswerService` trả evidence-first hoặc gọi LLM nếu được cấu hình.

## Ranh giới MVP

- Chưa huấn luyện GNN: graph hiện dùng cho entity linking, traversal và provenance. GNN chỉ nên thêm khi đã có bài toán dự đoán node/edge và tập nhãn rõ ràng.
- Chưa chạy OCR ảnh: PDF text được đọc bằng PyMuPDF; dataset VietFinTab được dùng để phát triển table detector/parser ở phase 2.
- Chưa phụ thuộc Neo4j: graph JSON/NetworkX giúp chạy local ngay. Có thể thay adapter bằng Neo4j khi corpus tăng.
- Numerical reasoning hiện ưu tiên truy xuất đúng evidence. Calculator/program induction là phase kế tiếp, được đánh giá bằng FinQA.

## Phase tiếp theo

1. Table detector + table structure model từ VietFinTab.
2. LLM/VLM structured extraction theo JSON Schema và entity resolution.
3. Dense multilingual embeddings + reranker.
4. Calculator tool có kiểm tra đơn vị/kỳ kế toán.
5. Evaluation: Recall@k cho retrieval, entity/relation F1, execution accuracy, answer EM/F1 và citation accuracy.

