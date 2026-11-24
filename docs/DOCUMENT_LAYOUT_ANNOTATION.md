# Hướng Dẫn Gán Nhãn & Huấn Luyện Nhận Diện Bảng Biểu & Biểu Đồ (Document Layout Analysis)

Tài liệu này hướng dẫn quy trình bán tự động hóa (Semi-automated Workflow) từ file PDF báo cáo tài chính đến việc gán nhãn trên **Label Studio / X-AnyLabeling** và huấn luyện mô hình **YOLO** nhận diện:
1. `table`: Các bảng biểu tài chính (Bảng cân đối kế toán, Báo cáo kết quả kinh doanh, Lưu chuyển tiền tệ...).
2. `chart`: Biểu đồ hình cột, hình tròn, đồ thị tăng trưởng, sơ đồ cơ cấu.

---

## 1. Kiến trúc quy trình

```text
               [PDF Báo cáo tài chính]
                         │
                         ▼
           python scripts/pdf_to_annotations.py
                         │
        ┌────────────────┴────────────────┐
        ▼                                 ▼
   [images/]                   [label_studio_tasks.json]
   (Ảnh từng trang)            (Pre-annotations: table + chart)
        │                                 │
        └───────────────┬─────────────────┘
                        ▼
            [Import vào Label Studio / X-AnyLabeling]
                        │
                        ▼ (Người dùng kiểm tra, chỉnh sửa nhanh)
            [Export kết quả JSON từ Label Studio]
                        │
                        ▼
           python scripts/label_studio_to_yolo.py
                        │
                        ▼
               [YOLO Dataset Format]
               (images/, labels/, data.yaml)
                        │
                        ▼
           python scripts/train_layout_yolo.py
                        │
                        ▼
               [best.onnx (CPU-friendly)]
                        │
                        ▼
        Tích hợp vào Financial-Multimodal-RAG
```

---

## 2. Bước 1: Trích xuất ảnh và tạo nhãn gợi ý tự động (Pre-annotation)

Chạy script sau để tự động phân tích PDF, trích xuất từng trang ra ảnh và tạo sẵn tọa độ hộp cho bảng và biểu đồ:

```powershell
.\.venv\Scripts\python.exe scripts/pdf_to_annotations.py "duong/dan/den/bao_cao.pdf" --output data/annotation_workspace
```

Kết quả sinh ra tại `data/annotation_workspace`:
- `images/`: Chứa ảnh PNG chất lượng cao của từng trang.
- `label_studio_tasks.json`: Chứa dữ liệu kèm dự đoán sơ bộ sẵn sàng nạp vào Label Studio.
- `yolo/`: Dataset định dạng YOLO ban đầu (có sẵn `data.yaml`).

---

## 3. Bước 2: Hiệu chỉnh nhãn với Label Studio

### 3.1. Cài đặt và khởi chạy Label Studio
```powershell
.\.venv\Scripts\python.exe -m pip install label-studio
.\.venv\Scripts\python.exe -m label_studio start
```

### 3.2. Cấu hình Project trong Label Studio
1. Đăng nhập và chọn **Create Project**.
2. Đặt tên: `Financial-Layout-Analysis`.
3. Vào tab **Labeling Setup** -> Chọn **Object Detection with Bounding Boxes**.
4. Dán cấu hình XML sau vào trình chỉnh sửa:

```xml
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="table" background="#FFA39E"/>
    <Label value="chart" background="#D4380D"/>
  </RectangleLabels>
</View>
```

5. Cấu hình đường dẫn thư mục ảnh cục bộ (Local Storage):
   - Vào **Settings** -> **Cloud Storage** -> **Add Source Storage**.
   - Storage Type: `Local files`.
   - Absolute local path: `C:\RAG\data\annotation_workspace\images`.
   - Bật `Treat every bucket object as a source file`.
6. Import dữ liệu:
   - Nhấn **Import** -> Tải file `data/annotation_workspace/label_studio_tasks.json` lên.
   - Khi mở bất kỳ trang nào, bạn sẽ thấy các vùng Bảng và Biểu đồ đã được đánh dấu sẵn. Bạn chỉ việc kéo chỉnh lại góc lệch hoặc thêm/bớt nhãn.

---

## 4. Bước 3: Xuất dữ liệu đã gán nhãn và chuyển sang định dạng YOLO

Sau khi hoàn tất gán nhãn trong Label Studio:
1. Nhấn nút **Export** ở góc trên bên phải -> Chọn định dạng **JSON**.
2. Lưu file export (ví dụ `project-1-export.json`).
3. Chạy script chuyển đổi để tự động chia tập `train` (80%) và `val` (20%):

```powershell
.\.venv\Scripts\python.exe scripts/label_studio_to_yolo.py "project-1-export.json" --images-dir data/annotation_workspace/images --output data/yolo_final
```

---

## 5. Bước 4: Huấn luyện mô hình YOLO & Xuất sang ONNX

Cài thư viện huấn luyện Ultralytics:
```powershell
.\.venv\Scripts\python.exe -m pip install ultralytics
```

Bắt đầu huấn luyện mô hình:
```powershell
.\.venv\Scripts\python.exe scripts/train_layout_yolo.py --data data/yolo_final/data.yaml --model yolov8n.pt --epochs 50 --batch 8
```

Script sẽ:
1. Tải trọng số khởi tạo nhẹ (`yolov8n.pt`).
2. Huấn luyện nhận diện 2 lớp: `table` và `chart`.
3. Đánh giá chỉ số mAP50 và mAP50-95 trên tập validation.
4. Tự động xuất ra file mô hình định dạng ONNX (`best.onnx`) để chạy mượt mà trên CPU mà không cần GPU.

---

## 6. Bước 5: Tích hợp vào hệ thống Financial-Multimodal-RAG

Sau khi có file model ONNX (ví dụ `runs/detect/financial_layout_dla/weights/best.onnx`):
1. Copy model vào thư mục dự án:
```powershell
Copy-Item runs/detect/financial_layout_dla/weights/best.onnx data/models/layout_detector.onnx
```
2. Cập nhật file `.env`:
```dotenv
OCR_DETECTOR_PATH=C:/RAG/data/models/layout_detector.onnx
```
3. Khởi động lại server:
```powershell
powershell -ExecutionPolicy Bypass -File .\start-local.ps1
```
Hệ thống sẽ tự động sử dụng mô hình DLA này để nhận diện bảng và sơ đồ trước khi trích xuất số liệu.
