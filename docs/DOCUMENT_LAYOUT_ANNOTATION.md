# Hướng Dẫn Gán Nhãn & Huấn Luyện Nhận Diện Bảng Biểu & Biểu Đồ (Document Layout Analysis)

Tài liệu mô tả quy trình bán tự động từ PDF báo cáo tài chính đến gán nhãn (`table`, `chart`) bằng **X-AnyLabeling** (khuyên dùng) hoặc **Label Studio**, sau đó huấn luyện **YOLO / DocLayout-YOLO** xuất ra ONNX.

> **Lưu ý quan trọng về cấu hình**:
> - `LAYOUT_MODEL_PATH` (+ `LAYOUT_BACKEND=yolo`): Dùng cho model **YOLO phát hiện vùng Bảng & Biểu đồ** trong `layout_detector.py`.
> - `OCR_DETECTOR_PATH`: Dùng riêng cho model **RapidOCR PP-OCRv5 phát hiện dòng chữ** trong `ocr_backends.py`. Không dùng lẫn hai biến này.

---

## 1. Bước 1: Sinh ảnh và nhãn gợi ý (Pre-annotation)

```powershell
.\.venv\Scripts\python.exe scripts/pdf_to_annotations.py "duong/dan/den/bao_cao.pdf" --output data/annotation_workspace
```

Kết quả tại `data/annotation_workspace`:
- `images/`: Ảnh PNG từng trang.
- `xanylabeling/`: File `.json` theo chuẩn X-AnyLabeling / LabelMe (khuyên dùng).
- `label_studio_tasks.json`: File import cho Label Studio (tương thích dự phòng).
- `yolo/`: Dataset YOLO ban đầu.

---

## 2. Bước 2: Hiệu chỉnh nhãn bằng X-AnyLabeling (Khuyên dùng)

**X-AnyLabeling** nhẹ hơn Label Studio khi chạy trên Windows, hỗ trợ tự động khoanh vùng bằng SAM2/YOLO chỉ với vài cú click:

1. Tải/cài X-AnyLabeling (bản CPU/GPU hoặc `pip install x-anylabeling`).
2. Mở thư mục ảnh `data/annotation_workspace/images` (hoặc load kèm thư mục `xanylabeling/`).
3. Chỉnh sửa nhanh 2 nhãn: `table` và `chart`, rồi bấm Save (`Ctrl+S`).
4. Chuyển sang định dạng YOLO train/val:

```powershell
.\.venv\Scripts\python.exe scripts/xanylabeling_to_yolo.py data/annotation_workspace/xanylabeling --images-dir data/annotation_workspace/images --output data/yolo_final
```

*(Nếu dùng Label Studio export JSON, chạy `scripts/label_studio_to_yolo.py` tương tự.)*

---

## 3. Bước 3: Huấn luyện & Xuất ONNX

```powershell
.\.venv\Scripts\python.exe -m pip install ultralytics
.\.venv\Scripts\python.exe scripts/train_layout_yolo.py --data data/yolo_final/data.yaml --model yolov8n.pt --epochs 50 --batch 8
```

Khuyến nghị tốt hơn nếu có trọng số **DocLayout-YOLO** (`doclayout_yolo_docstructbench_imgsz1024.pt`): truyền qua `--model` để tận dụng pre-train chuyên biệt cho tài liệu tài chính.

---

## 4. Bước 4: Tích hợp vào hệ thống

Copy model `.onnx` vào `data/models/layout_detector.onnx` và khai báo trong `.env`:

```dotenv
LAYOUT_BACKEND=yolo
LAYOUT_MODEL_PATH=C:/RAG/data/models/layout_detector.onnx
```

Khi không có model YOLO tùy chỉnh, bạn vẫn có thể đặt `LAYOUT_BACKEND=heuristic` để hệ thống dùng PyMuPDF phát hiện vùng bảng/biểu đồ và chạy `ChartReader`.
