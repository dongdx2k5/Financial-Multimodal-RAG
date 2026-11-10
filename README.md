# Kiến Trúc Hợp Nhất & Hướng Dẫn Triển Khai Financial Multimodal RAG Trên Google Colab T4

Tài liệu này mô tả chi tiết kiến trúc hợp nhất giữa hai nhánh của dự án (`src/multimodal_rag` bản chạy Local và `Financial-Multimodal-RAG-review-20261020` bản chạy Colab T4), cách phân bổ bộ nhớ GPU trên Google Colab T4 (16GB VRAM), cùng hướng dẫn chạy thực tế với mã nguồn hợp nhất `unified_financial_rag_colab_t4.py`.

## 1. Tổng Quan Kiến Trúc "Shared VLM Backbone + Deterministic Core"

Thay vì sử dụng `pytesseract` dễ làm lệch dòng/cột trên Báo cáo tài chính (BCTC) Việt Nam hoặc nạp rời rạc quá nhiều mô hình gây tràn bộ nhớ GPU T4, hệ thống hợp nhất sử dụng một mô hình ngôn ngữ thị giác lượng tử hóa 4-bit duy nhất làm xương sống cho cả hai pha (Đọc bảng Scan và Tổng hợp câu trả lời), kết hợp chặt chẽ với bộ kiểm toán đẳng thức kế toán VAS và máy tính số học `Decimal`.

| Tầng Kiến Trúc | Mô Hình / Công Nghệ Sử Dụng | Mức Chiếm Dụng VRAM (T4 16GB) | Chức Năng Trong Pipeline Hợp Nhất | 
| ----- | ----- | ----- | ----- | 
| **1. Core VLM & LLM (2-trong-1)** | [Qwen/Qwen2.5-VL-7B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct-AWQ) (hoặc bản nhẹ [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)) | \~5.8 GB (AWQ 4-bit) | Đọc trực tiếp ảnh chụp bảng BCTC scan ra JSON chuẩn cột VAS; tự động đọc lại (Self-Correction) khi tổng kiểm toán bị lệch; đóng vai trò LLM tổng hợp câu trả lời thuyết minh. | 
| **2. Digital PDF & Layout Parser** | `PyMuPDF` / `pdfplumber` + [IBM Docling](https://github.com/docling-project/docling) ([docling-ibm-models](https://github.com/docling-project/docling-ibm-models)) | \~0.5 GB (hoặc CPU) | Bóc tách nhanh các trang PDF điện tử có sẵn text layer; khoanh vùng tọa độ bảng (`bbox`) để chỉ gửi các trang chứa bảng tài chính scan vào VLM. | 
| **3. Multilingual Embedding** | [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) | \~1.2 GB (FP16) | Biểu diễn ngữ nghĩa đa ngôn ngữ Việt – Anh với độ dài ngữ cảnh tới 8192 tokens, kết hợp cùng tìm kiếm từ khóa `SQLite FTS5` theo thuật toán Reciprocal Rank Fusion (RRF). | 
| **4. Cross-Encoder Reranker** | [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) | \~1.1 GB (FP16) | Chấm điểm xếp hạng lại các đoạn văn/bảng truy xuất được, ngăn chặn việc lấy nhầm số liệu giữa báo cáo "Công ty mẹ" và "Hợp nhất". | 
| **5. Kiểm Toán & Tính Toán Tất Định** | `VASAccountingQualityGate` + `DeterministicFinancialCalculator` (`Decimal`) | 0 GB (CPU thuần) | Kiểm tra chéo 7 đẳng thức kế toán Thông tư 200 ngay khi đọc bảng và tính toán YoY, ROE, ROA, Biên lợi nhuận với độ chính xác tuyệt đối. | 

## 2. Danh Sách 8 Lỗi Tồn Đọng Đã Được Khắc Phục Triệt Để

Mã nguồn hợp nhất `unified_financial_rag_colab_t4.py` đã giải quyết toàn bộ 8 vấn đề được ghi nhận trong tài liệu rà soát kỹ thuật của nhánh `Thien`, đồng thời tích hợp trọn vẹn các module mạnh nhất từ nhánh chính:

| Mã Lỗi | Mức Độ | Vấn Đề Ở Phiên Bản Cũ | Giải Pháp Đã Thực Thi Trong Bản Hợp Nhất | 
| ----- | ----- | ----- | ----- | 
| **#1** | High | File mẫu `NVDA_10K_2024_sample.pdf` thực chất chứa báo cáo năm 2009 của Analog Devices nhưng vẫn bị gắn nhãn `NVDA` (`FY2024`). | Bổ sung hàm `SmartFinancialPDFParser.verify_document_identity()` đối chiếu tên doanh nghiệp (`COMPANY_ALIASES`) và năm tài chính ngay trên trang đầu tiên; chặn và báo lỗi `ValueError` tức thì nếu tài liệu bị tráo ruột. | 
| **#2** | High | PDF scan (`FPT_BCTC_2024_scanned.pdf`) không có text layer trả về 0 chunks / 0 facts nhưng `validate_extraction()` vẫn báo `is_valid=True`. | Tự động phát hiện trang scan (`len(raw_text) < 40`) để kích hoạt [Qwen/Qwen2.5-VL-7B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct-AWQ) đọc bảng; sửa `validate_extraction()` bắt buộc trả về `is_valid=False` khi không có dữ liệu. | 
| **#3** | High | `AnswerGenerator` và giao diện `Gradio` chạy trên dữ liệu demo giả lập, không đi qua `HybridRetriever`, `Router` và `NumericalVerifier`. | Xây dựng class `UnifiedFinancialRAGPipeline` và hàm `create_colab_gradio_interface()` nối khép kín từ bước phân loại ý định (`classify_intent`) đến tính toán `Decimal`, truy xuất lai và kiểm chứng số học. | 
| **#4** | Medium | `SQLiteStore.query_facts()` fallback khi thiếu kỳ báo cáo nhưng không có `ORDER BY period DESC`, trả về kỳ ngẫu nhiên và giấu kỳ thực tế. | Sửa câu truy vấn trong `UnifiedSQLiteStore.query_fact()` luôn dùng `ORDER BY period DESC, id DESC LIMIT 1`, gắn cờ `is_fallback_period=True` và thông báo rõ kỳ thực tế (`actual_period`) trong câu trả lời. | 
| **#5** | Medium | `FactExtractor._clean_number()` lọc bỏ mọi giá trị `<= 0`, làm mất hoàn toàn các khoản lỗ và dòng tiền âm `(1.250.500)`. | Viết lại `FactExtractor.clean_number()` hỗ trợ đầy đủ số âm đặt trong ngoặc đơn kế toán `(1.250.500)`, dấu trừ `-450.25`, số `0`, và phân biệt chuẩn dấu chấm/phẩy giữa BCTC Việt Nam và Quốc tế. | 
| **#6** | Medium | `NumericalVerifier._extract_numbers()` bỏ qua mọi con số `<= 10`, không bắt được lỗi ảo giác trên các tỷ số tăng trưởng và biên lợi nhuận nhỏ. | Cập nhật `NumericalVerifier.extract_numbers()` trích xuất và kiểm chứng mọi con số tài chính (kể cả số âm, số thập phân nhỏ như `6.45%`, `1.82 lần`), chỉ lọc bỏ các nhãn năm/trang/mã số VAS. | 
| **#7** | Low | `requirements.txt` chứa gói không hợp lệ (`sqlite3` vốn thuộc thư viện chuẩn Python) và các thư viện dư thừa (`chromadb`, `camelot-py`, `pytesseract`). | Làm sạch danh sách cài đặt trên Colab T4, chỉ giữ lại các gói thực sự sử dụng cho pipeline hợp nhất. | 
| **#8** | Low | Bộ kiểm thử `test_pipeline.py` có độ phủ hẹp, bỏ lọt toàn bộ các lỗi hồi quy trên. | Xây dựng bộ kiểm thử tự động `test_unified_pipeline.py` kiểm chứng toàn bộ 8 kịch bản lỗi và vòng lặp tự sửa lỗi kế toán VAS. | 

## 3. Cơ Chế Vòng Lặp Tự Sửa Lỗi Kế Toán (VAS Self-Correction Loop)

Một cải tiến quan trọng trong `unified_financial_rag_colab_t4.py` là sự kết hợp giữa `VASAccountingQualityGate` và `SharedQwenVLBackbone`:

1. **Bước 1 (Đọc bảng lần đầu):** Khi gặp trang BCTC scan, [Qwen/Qwen2.5-VL-7B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct-AWQ) trích xuất danh sách dòng theo cấu trúc `["Tên chỉ tiêu", "Mã số", "Thuyết minh", "Số cuối năm", "Số đầu năm"]`.

2. **Bước 2 (Kiểm toán 7 đẳng thức VAS):** Hệ thống tự động kiểm tra:

   * Bảng cân đối kế toán: `Mã 270 = Mã 100 + Mã 200`, `Mã 440 = Mã 300 + Mã 400`, `Mã 270 = Mã 440`.

   * Kết quả kinh doanh: `Mã 20 = Mã 10 - Mã 11`, `Mã 60 = Mã 50 - Mã 51 - Mã 52`.

   * Lưu chuyển tiền tệ: `Mã 50 = Mã 20 + Mã 30 + Mã 40`, `Mã 70 = Mã 50 + Mã 60 + Mã 61`.

3. **Bước 3 (Targeted Re-read khi phát hiện lệch số):** Nếu bất kỳ đẳng thức nào có sai số vượt ngưỡng cho phép (`status == "fail"`), `SmartFinancialPDFParser` tự động tạo `correction_hint` ghi rõ đẳng thức đang lệch và mức chênh lệch, sau đó gọi lại VLM trên trang bảng đó để đối chiếu lại chữ số bị nhầm trước khi lưu vào `UnifiedSQLiteStore`.

## 4. Hướng Dẫn Cài Đặt & Khởi Chạy Trên Google Colab T4

### Bước 1: Cài đặt thư viện chuẩn trên cell đầu tiên của Google Colab

```
pip install -q torch torchvision torchaudio
pip install -q "transformers>=4.49.0" accelerate autoawq qwen-vl-utils
pip install -q sentence-transformers faiss-cpu FlagEmbedding
pip install -q docling pdfplumber gradio pytest

```

### Bước 2: Khởi tạo Pipeline với GPU T4 và mở giao diện Gradio

```
from unified_financial_rag_colab_t4 import (
    SharedQwenVLBackbone,
    UnifiedFinancialRAGPipeline,
    create_colab_gradio_interface,
)

# 1. Khởi tạo Shared Backbone dùng Qwen2.5-VL-7B-Instruct-AWQ (chỉ ~5.8GB VRAM trên Colab T4)
backbone = SharedQwenVLBackbone(
    model_id="Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
    lazy_load=False,
)

# 2. Khởi tạo Pipeline hợp nhất lưu SQLite trên Google Drive hoặc ổ đĩa Colab
pipeline = UnifiedFinancialRAGPipeline(
    db_path="/content/financial_rag_unified.sqlite",
    vlm_backbone=backbone,
)

# 3. Khởi chạy giao diện phân tích & kiểm toán BCTC trên Gradio
demo = create_colab_gradio_interface(pipeline)
demo.launch(share=True)

```