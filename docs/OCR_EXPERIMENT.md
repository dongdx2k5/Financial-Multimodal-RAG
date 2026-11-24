# Hướng Dẫn Thử Nghiệm OCR & Nhận Dạng Bảng (OCR Experiment)

Tài liệu hướng dẫn thiết lập và đánh giá các mô hình OCR trong hệ thống: Tesseract mặc định, VietOCR và SLANet_plus.

---

## 1. Tesseract OCR (Mặc định)

- PyMuPDF tích hợp sẵn thư viện Tesseract C++ API mà không cần cài đặt binary bên ngoài.
- Dữ liệu ngôn ngữ tải về:
  - `vie.traineddata`: Tiếng Việt
  - `eng.traineddata`: Tiếng Anh
- Cấu hình trong `.env`:
  ```dotenv
  OCR_ENABLED=true
  OCR_LANGUAGE=vie+eng
  TESSDATA_PREFIX=C:/RAG/data/tessdata
  TESSDATA_BEST_PREFIX=C:/RAG/data/tessdata-best
  ```

---

## 2. Thử Nghiệm VietOCR + PP-OCR Detector

- **Detector**: PaddleOCR `ch_PP-OCRv5_det_mobile.onnx` (phát hiện vùng văn bản dạng polygon/box).
- **Recognizer**: VietOCR `vgg_seq2seq.pth` (nhận dạng dòng chữ tiếng Việt có dấu với độ chính xác cao).
- Cấu hình trong `.env`:
  ```dotenv
  OCR_BACKEND=vietocr
  VIETOCR_WEIGHTS=C:/RAG/data/models/vgg_seq2seq.pth
  OCR_DETECTOR_PATH=C:/RAG/data/models/ch_PP-OCRv5_det_mobile.onnx
  ```

---

## 3. Thử Nghiệm Nhận Dạng Cấu Trúc Bảng SLANet_plus

- RapidTable / SLANet_plus hỗ trợ nhận diện các ô gộp (merged cells), số hàng và số cột.
- Cấu hình:
  ```dotenv
  TABLE_BACKEND=slanet_plus
  TABLE_MODEL_PATH=C:/RAG/data/models/slanet-plus.onnx
  ```
- **Lưu ý**: Nhận diện cấu trúc bảng từ mô hình deep learning được xếp vào nhóm bằng chứng thử nghiệm (`experimental evidence`), không tự động coi là fact đã kiểm toán trước khi vượt qua kiểm tra cân đối kế toán.
