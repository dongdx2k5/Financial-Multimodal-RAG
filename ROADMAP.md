# Lộ Trình Nâng Cấp Hệ Thống Multimodal Financial Graph RAG

Tài liệu này vạch ra lộ trình chi tiết từng bước nâng cấp dự án từ **MVP cơ bản** lên một hệ thống **Multimodal Financial Graph RAG hoàn chỉnh cấp độ Production**.

---

## 1. Tổng quan lộ trình (8 tuần)

```
                              LỘ TRÌNH TỔNG QUAN (8 TUẦN)
┌─────────────────────────┐   ┌──────────────────────────┐   ┌──────────────────────────┐   ┌─────────────────────────┐
│  GIAI ĐOẠN 1 (Tuần 1-2) │   │  GIAI ĐOẠN 2 (Tuần 3-4)  │   │  GIAI ĐOẠN 3 (Tuần 5-6)  │   │  GIAI ĐOẠN 4 (Tuần 7-8) │
│  Graph Traversal &      │──>│  True Multimodal Vision  │──>│  Reranking, Scalable DB  │──>│  Evaluation Framework & │
│  LLM Fact Extraction    │   │  (Scanned PDF & Charts)  │   │  & Time-Series Engine    │   │  Interactive Web UI     │
└─────────────────────────┘   └──────────────────────────┘   └──────────────────────────┘   └─────────────────────────┘
```

---

## 2. Bảng phân công nhiệm vụ theo thứ tự ưu tiên

| Thứ tự | Nhiệm vụ | Module tác động | Độ ưu tiên | Độ phức tạp | Thời gian |
| :---: | :--- | :--- | :---: | :---: | :---: |
| **1** | Bổ sung **Graph Traversal** (truy vấn chuỗi thời gian đa năm) | `graph.py`, `retrieval.py` | 🔴 Rất cao | Vừa | 3 - 4 ngày |
| **2** | Chuẩn hóa **Financial Taxonomy** & LLM Structured Extraction | `extraction.py`, `models.py` | 🔴 Rất cao | Vừa | 3 - 4 ngày |
| **3** | Tích hợp **Gemini Vision OCR** cho PDF Scan & ảnh biểu đồ | `ingestion.py`, `llm.py` | 🟠 Cao | Vừa | 4 - 5 ngày |
| **4** | Tích hợp **Cross-Encoder Reranker** vào luồng tìm kiếm | `retrieval.py` | 🟡 Trung bình | Dễ | 2 - 3 ngày |
| **5** | Chuyển đổi Vector Storage sang **SQLite-vec** | `database.py`, `retrieval.py` | 🟡 Trung bình | Vừa | 3 - 4 ngày |
| **6** | Xây dựng script **Benchmark Evaluation** (`evaluate_rag.py`) | `scripts/` | 🟠 Cao | Vừa | 3 - 4 ngày |
| **7** | Xây dựng **Giao diện Web Dashboard trực quan** (Graph + Chat) | `api.py`, `web/` | 🟡 Trung bình | Vừa | 5 - 7 ngày |

---

## 3. Chi tiết kỹ thuật từng bước nâng cấp

### Giai đoạn 1: Hoàn thiện lõi Graph-RAG & Chuẩn hóa Trích xuất (Tuần 1 – 2)

#### Bước 1.1: Triển khai Graph Traversal (Mở rộng ngữ cảnh chuỗi thời gian đa chặng)
* **Vấn đề giải quyết**: Hiện tại `retrieval.py` chỉ lấy facts cục bộ gắn liền với chunk tìm được. Các câu hỏi so sánh hoặc tính tăng trưởng qua các năm (ví dụ: *"So sánh doanh thu thuần của HPG năm 2023 và 2024"*) bị thiếu dữ liệu lịch sử nếu số liệu nằm ở các báo cáo/chunk khác nhau.
* **File tác động**:
  * `src/financial_rag/graph.py`
  * `src/financial_rag/retrieval.py`
  * `src/financial_rag/service.py`
* **Công việc cụ thể**:
  1. Bổ sung hàm duyệt đồ thị đa bước trong `graph.py`:
     ```python
     def find_historical_facts(self, ticker: str, metric_id: str, limit_years: int = 3) -> list[FinancialFact]:
         """Duyệt đồ thị: Fact(t) -> MEASURES -> Metric <- MEASURES <- Fact(t-1) -> FOR_PERIOD -> Period(t-1)"""
         ...
     ```
  2. Khi `retriever.search()` xác định được ticker và metric chính của câu hỏi, tự động gom chuỗi facts lịch sử từ Knowledge Graph.
  3. Ghép chuỗi facts lịch sử vào context gửi cho Gemini LLM để kích hoạt `SafeCalculator` tính toán tăng trưởng YoY/CAGR.
* **Tiêu chuẩn nghiệm thu**: Hệ thống trả lời chính xác câu hỏi so sánh số liệu tài chính qua 2 - 3 năm ngay cả khi các con số nằm ở 2 file tài liệu riêng biệt.

---

#### Bước 1.2: Chuẩn hóa thực thể tài chính (Financial Taxonomy & LLM Extraction)
* **Vấn đề giải quyết**: Regex match bị lỗi khi thuật ngữ biến thiên (*"Doanh thu thuần về bán hàng"*, *"Doanh thu thuần"*, *"Net Revenue"*) hoặc khi bảng biểu có ô gộp (merged cells), phân cấp nhiều tầng.
* **File tác động**:
  * `src/financial_rag/models.py`
  * `src/financial_rag/extraction.py`
* **Công việc cụ thể**:
  1. Thêm trường `standardized_metric: str` vào `FinancialFact` và tạo bảng từ điển chuẩn hóa (**Chart of Accounts**) theo Thông tư 200/VAS & IFRS (`REVENUE_NET`, `NET_PROFIT`, `TOTAL_ASSETS`, `OPERATING_CASH_FLOW`,...).
  2. Tích hợp chế độ **LLM Structured Extraction fallback**: Khi một chunk chứa bảng phức tạp không thể parse bằng regex, gọi Gemini với `response_schema` để trích xuất JSON có cấu trúc chuẩn xác 100%.
* **Tiêu chuẩn nghiệm thu**: Các chỉ tiêu đồng nghĩa tự động được ánh xạ về cùng một Node `Metric` trong Knowledge Graph.

---

### Giai đoạn 2: Hiện thực hóa tính năng "Multimodal" (Tuần 3 – 4)

#### Bước 2.1: Tích hợp Gemini Vision OCR cho PDF Scan
* **Vấn đề giải quyết**: Báo cáo tài chính kiểm toán ở Việt Nam phần lớn là tài liệu scan con dấu đỏ, PyMuPDF đọc ra văn bản rỗng.
* **File tác động**:
  * `src/financial_rag/ingestion.py`
  * `src/financial_rag/llm.py`
* **Công việc cụ thể**:
  1. Trong `DocumentLoader._load_pdf()`, bổ sung kiểm tra: Nếu số ký tự của trang $< 50$, gắn cờ `is_scanned = True`.
  2. Dùng PyMuPDF render trang scan thành ảnh PNG (`page.get_pixmap(dpi=150)`).
  3. Gửi ảnh sang Gemini Vision API (`gemini-3.8-flash`) với prompt trích xuất cấu trúc văn bản và bảng số liệu dạng Markdown table.
  4. Trả kết quả về pipeline chunking thông thường.
* **Tiêu chuẩn nghiệm thu**: Ingest thành công file PDF scan 100% hình ảnh và trích xuất đầy đủ các bảng biểu tài chính.

---

#### Bước 2.2: Phân tích biểu đồ (Chart Understanding)
* **Vấn đề giải quyết**: Các biểu đồ cơ cấu tài sản, tăng trưởng lợi nhuận trong Báo cáo thường niên (Annual Report) không có dữ liệu bảng trực tiếp.
* **File tác động**:
  * `src/financial_rag/ingestion.py`
  * `src/financial_rag/models.py`
* **Công việc cụ thể**:
  1. Bóc tách hình ảnh trong tài liệu bằng PyMuPDF (`page.get_images()`).
  2. Gửi ảnh sang Gemini Vision nhận diện xem hình ảnh có phải biểu đồ tài chính (bar chart, line chart, pie chart) hay không.
  3. Chuyển đổi dữ liệu số từ đồ thị thành facts dạng JSON và nạp vào Knowledge Graph.
* **Tiêu chuẩn nghiệm thu**: Hệ thống trả lời được câu hỏi dựa trên số liệu chỉ xuất hiện trong biểu đồ minh họa.

---

### Giai đoạn 3: Tối ưu Retrieval, Reranker & Cơ sở dữ liệu Vector (Tuần 5 – 6)

#### Bước 3.1: Tích hợp Cross-Encoder Reranker
* **Vấn đề giải quyết**: Hybrid Retrieval đôi khi đưa các chunk chứa từ khóa nhưng sai ngữ cảnh vào top kết quả.
* **File tác động**:
  * `src/financial_rag/retrieval.py`
  * `pyproject.toml`
* **Công việc cụ thể**:
  1. Bổ sung thư viện reranker gọn nhẹ (`flashrank` hoặc `sentence-transformers` với `bge-reranker-base`).
  2. Quy trình Two-Stage Retrieval:
     * **Giai đoạn 1**: Hybrid Retrieval (TF-IDF + Gemini Dense) lọc ra top 20 chunks ứng viên.
     * **Giai đoạn 2**: Reranker chấm điểm tương quan giữa câu hỏi và từng ứng viên, chọn ra top 5 chunks tinh nhuệ nhất.
* **Tiêu chuẩn nghiệm thu**: Nâng cao chỉ số Hit Rate@3, loại bỏ hoàn toàn các đoạn văn rác gây nhiễu cho LLM.

---

#### Bước 3.2: Chuyển đổi sang SQLite-vec (Lưu trữ Vector mở rộng)
* **Vấn đề giải quyết**: File numpy phẳng `chunks.dense.npy` không hỗ trợ cập nhật mở rộng khi quy mô tài liệu lên hàng ngàn báo cáo.
* **File tác động**:
  * `src/financial_rag/database.py`
  * `src/financial_rag/retrieval.py`
* **Công việc cụ thể**:
  1. Tích hợp extension `sqlite-vec` vào file cơ sở dữ liệu `financial_rag.db`.
  2. Lưu vector embedding trực tiếp vào cùng SQLite bên cạnh bảng `chunks`.
  3. Cho phép truy vấn kết hợp SQL FTS5 (BM25) và Vector KNN chỉ bằng một câu query.
* **Tiêu chuẩn nghiệm thu**: Xóa bỏ hoàn toàn file `.dense.npy`, quản lý thống nhất toàn bộ dữ liệu trong một file database SQLite duy nhất.

---

### Giai đoạn 4: Đánh giá định lượng & Giao diện Web Trực quan (Tuần 7 – 8)

#### Bước 4.1: Xây dựng Pipeline Benchmark & Đánh giá định lượng
* **Vấn đề giải quyết**: Cần có số liệu đo lường định lượng khoa học để chứng minh hiệu quả hệ thống.
* **File tác động**:
  * Tạo mới file `scripts/evaluate_rag.py`
* **Công việc cụ thể**:
  1. Tận dụng tập test có sẵn `data/raw/finqa` và dataset `VNFinsQA`.
  2. Thiết lập bộ metrics:
     * **Retrieval Recall@k & MRR**: Đo lường độ chính xác khi tìm bằng chứng.
     * **Calculation Accuracy**: Tỷ lệ tính toán số học đúng từ `SafeCalculator`.
     * **Faithfulness & Context Relevance**: Đảm bảo LLM không bịa đặt số liệu (hallucination).
     * **Latency**: Đo thời gian phản hồi trung bình (ms/query).
* **Tiêu chuẩn nghiệm thu**: Chạy script xuất ra báo cáo `benchmark_results.json` với Recall@5 $\ge 85\%$ và Calculation Accuracy $\ge 95\%$.

---

#### Bước 4.2: Xây dựng Giao diện Web Dashboard Tương tác (Interactive UI)
* **Vấn đề giải quyết**: Giúp người dùng cuối và nhà phân tích tài chính tương tác trực quan với hệ thống.
* **File tác động**:
  * `src/financial_rag/api.py`
  * Tạo thư mục `src/financial_rag/web/`
* **Công việc cụ thể**:
  1. Xây dựng giao diện Web Dashboard 3 cột (FastAPI Static Files + HTML5/CSS3 + Vanilla JS hoặc Streamlit):
     * **Cột trái**: Upload PDF kéo thả, danh sách báo cáo, bộ lọc mã cổ phiếu/năm.
     * **Cột giữa**: Khung Chatbot hỏi đáp, hiển thị nguồn trích dẫn, các bước tính toán của `SafeCalculator`, hỗ trợ Streaming SSE.
     * **Cột phải**: **Interactive Knowledge Graph Viewer** sử dụng thư viện `vis-network` (kết nối trực tiếp `/graph` để xoay, zoom, lọc node Công ty $\leftrightarrow$ Kỳ báo cáo $\leftrightarrow$ Chỉ tiêu).
* **Tiêu chuẩn nghiệm thu**: Người dùng có thể thao tác hoàn toàn từ giao diện web: nạp file, hỏi đáp, xem đồ thị tri thức và đối chiếu tài liệu gốc.
