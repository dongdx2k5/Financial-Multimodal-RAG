# 💾 LƯU TRỮ ĐA TẦNG & TRUY XUẤT KẾT HỢP (TRIPLE STORE & HYBRID RETRIEVAL)

> *Trích xuất từ tài liệu kiến trúc [KIEN_TRUC.md](KIEN_TRUC.md) (Tầng 2 & Tầng 3)*

---

### 💾 TẦNG 2: Lưu Trữ Đa Tầng Triple Store (Knowledge Graph + Vector + DocStore)

* **Module chịu trách nhiệm**: [`knowledge_graph.py`](src/financial_rag/retrieval/knowledge_graph.py), [`vector_store.py`](src/financial_rag/retrieval/vector_store.py), [`doc_store.py`](src/financial_rag/retrieval/doc_store.py), [`multimodal_summarizer.py`](src/financial_rag/reasoning/multimodal_summarizer.py).
* **Vấn đề giải quyết**: 
  - Vector Database chỉ có khả năng so khớp ngữ nghĩa gần đúng (semantic similarity), hoàn toàn không hiểu được quan hệ kế toán đa chiều (ví dụ: *Lợi nhuận gộp = Doanh thu thuần - Giá vốn*).
  - Ngược lại, nếu chỉ dùng đồ thị thuần túy thì sẽ bỏ lỡ các thông tin định tính dạng văn bản tự do trong phần giải trình của ban điều hành (MD&A).
* **Cơ chế hoạt động**:
  1. **Đồ thị tri thức & Cây tài liệu (NetworkX DiGraph)**:
     - Lưu trữ toàn bộ các Node thực thể và Edges quan hệ dưới dạng đồ thị có hướng (Directed Graph).
     - Embed cây tài liệu ngay trong cùng một không gian đồ thị, cho phép vừa duyệt quan hệ thực thể tài chính vừa duyệt cấu trúc chương mục tài liệu.
     - Cơ chế bền vững: Tự động tuần tự hóa và lưu ra tệp `data/processed/knowledge_graph.json` theo định dạng chuẩn Node-Link JSON.
  2. **Vector Store (ChromaDB)**:
     - Dùng LLM (**LLaMA 3.3 / LLaMA 3.2 Vision**) sinh bản tóm tắt súc tích, giàu từ khóa ngữ nghĩa cho từng bảng biểu và biểu đồ.
     - Chỉ lưu Vector Embeddings của các bản **tóm tắt ngữ nghĩa** kèm định danh `doc_id` để tìm kiếm siêu tốc mà không bị loãng ngữ nghĩa.
  3. **Document Store (DocStore JSON KV-Store)**:
     - Lưu trữ toàn văn bản gốc, mã bảng HTML đầy đủ, và chuỗi ảnh Base64 chất lượng cao.
     - Phục vụ truy xuất nội dung chi tiết đưa vào ngữ cảnh LLM khi có kết quả từ bước tìm kiếm.

---

### 🎯 TẦNG 3: Truy Xuất Kết Hợp Hybrid & Xếp Hạng Đa Chiều (Hybrid Retrieval & Re-ranking)

* **Module chịu trách nhiệm**: [`graph_retriever.py`](src/financial_rag/retrieval/graph_retriever.py), [`vector_store.py`](src/financial_rag/retrieval/vector_store.py), [`reranker.py`](src/financial_rag/retrieval/reranker.py).
* **Vấn đề giải quyết**: Một truy vấn tài chính như *"Biên lợi nhuận hoạt động quý 3/2021 thay đổi thế nào so với cùng kỳ?"* đòi hỏi kết hợp dữ liệu từ nhiều trang khác nhau (bảng P&L quý 3/2021, bảng P&L quý 3/2020, và phần phân tích biên lợi nhuận trong thuyết minh).
* **Cơ chế hoạt động**:
  1. **Truy xuất Đồ thị (Graph k-hop Traversal)**:
     - Trích xuất thực thể từ câu hỏi (ví dụ: `Operating Margin`, `Q3/2021`).
     - Định vị các node tương ứng trên đồ thị và thực hiện thuật toán mở rộng $k$-hop ($k=2$) để thu thập toàn bộ các node lân cận liên quan: giá trị số, kỳ so sánh, và thuyết minh kèm theo.
  2. **Duyệt Cây Phân Cấp (Tree Breadcrumbs Context)**:
     - Từ các chunk được tìm thấy, đi ngược cây phả hệ để lấy tiêu đề chương lớn (Parent Section) và đi xuôi cây để lấy các bảng biểu/chỉ tiêu phụ trợ liên đới.
  3. **Truy xuất Ngữ nghĩa Vector (Over-Retrieval)**:
     - Lấy $2 \times K$ tài liệu ứng viên từ ChromaDB dựa trên độ tương đồng cosine của embedding câu hỏi.
  4. **Hợp nhất Hybrid (Hybrid Fusion Score)**:
     $$\text{Score}_{\text{final}} = \alpha \cdot \text{Score}_{\text{vector}} + \beta \cdot \text{Score}_{\text{graph}}$$
     (Mặc định: $\alpha = 0.6, \beta = 0.4$, có thể tùy chỉnh linh hoạt qua `.env`).
  5. **Tái Xếp Hạng Đa Chiều (`ContextReRanker`)**:
     - **Time Anchoring**: Khớp chính xác năm tài chính (`2021`, `2022`), quý (`Q1`, `Q3`), hoặc niên độ (`FY2023`).
     - **Entity Matching**: Khớp tên chỉ số trọng yếu (*Revenue, Gross Profit, Operating Expense, EBITDA*).
     - **Graph Proximity Score**: Ưu tiên các đoạn có liên kết trực tiếp trên đồ thị tri thức với thực thể câu hỏi.
     - **Tree Depth Score**: Ưu tiên các dữ liệu nằm trong cùng một Section chuyên mục của báo cáo.
