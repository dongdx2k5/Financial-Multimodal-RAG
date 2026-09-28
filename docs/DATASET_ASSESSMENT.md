# Đánh giá dataset (28-09-2026)

## Khuyến nghị

Chọn `AIGuruTinix/ViFinQA` + `VietFinTabGroup/VietFinTab` làm hai nguồn chính. Chúng phủ hai phần còn thiếu của nhau: corpus OCR/report-level QA và ảnh bảng có nhãn cấu trúc.

### AIGuruTinix/ViFinQA — ưu tiên 1

- 1.012 câu hỏi, 1.973 báo cáo, 100 mã cổ phiếu, giai đoạn 2015–2025.
- Báo cáo là OCR text, giữ ranh giới trang và bảng dạng HTML inline.
- Rất phù hợp để xây retrieval, chunking, entity/metric extraction và corpus-level QA.
- Hạn chế: bản public không có answer, gold evidence, bảng CSV chuẩn hóa hoặc split train/validation/test.

### VietFinTabGroup/VietFinTab — ưu tiên 1 cho multimodal

- Khoảng 10,4 nghìn ảnh bảng; có XML cấu trúc, một phần có JSON content.
- Phù hợp huấn luyện/đánh giá table detection và table structure recognition.
- Nhãn content không có cho mọi mẫu, cần lọc `has_content_label=true` cho OCR/content task.

### ngnquan/ViFinTab — ưu tiên 2

- Khoảng 1,98 nghìn mẫu train, ảnh base64 và polygon/object labels.
- Phù hợp baseline table detection nhỏ, tải nhanh.
- Schema/giấy phép khác dataset VietFinTabGroup; không nên gộp trực tiếp trước khi ánh xạ class và kiểm tra nguồn trùng.

### duykhangh/VNFinsQA — evaluation phụ

- 790 câu test, có category, difficulty, multi-document flag và ground truth.
- Hữu ích để smoke-test QA tiếng Việt/Graph RAG.
- Ground truth có thể trộn dữ liệu báo cáo, market và vĩ mô; không luôn kèm evidence document-level nên chưa đủ làm ground truth cho citation/retrieval.

### dreamerdeo/finqa — numerical reasoning phụ

- Tiếng Anh, gồm question, answer, pre/post text và table; phù hợp đo khả năng suy luận số học.
- Không đại diện trực tiếp cho ngôn ngữ và chuẩn trình bày báo cáo Việt Nam.

## Chiến lược sử dụng

- Retrieval corpus: ViFinQA.
- Table vision: VietFinTabGroup/VietFinTab.
- Detector sanity check: ngnquan/ViFinTab.
- Vietnamese QA regression: VNFinsQA.
- Arithmetic executor regression: FinQA.

