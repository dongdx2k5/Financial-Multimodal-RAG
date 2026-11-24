# Đánh Giá Các Tập Dữ Liệu Tài Chính Tiếng Việt (Dataset Assessment)

Tổng quan và đánh giá các bộ dữ liệu mã nguồn mở phục vụ nghiên cứu và phát triển Document AI tài chính tiếng Việt.

---

## 1. Danh Sách Các Dataset

| Dataset | Định dạng & Quy mô | Ứng dụng trong dự án |
|---|---|---|
| **[ngnquan/ViFinTab](https://huggingface.co/datasets/ngnquan/ViFinTab)** | COCO format, ảnh trang BCTC tiếng Việt | Huấn luyện bài toán Table Detection (Phát hiện vị trí bảng) |
| **[VietFinTab](https://huggingface.co/datasets/VietFinTabGroup/VietFinTab)** | Cấu trúc ô bảng, nhãn HTML | Nghiên cứu nhận diện cấu trúc hàng/cột (Table Structure Recognition) |
| **[AIGuruTinix/ViFinQA](https://huggingface.co/datasets/AIGuruTinix/ViFinQA)** | Cặp câu hỏi - câu trả lời tài chính | Đánh giá khả năng truy xuất (Retrieval Benchmark) |
| **[duykhangh/VNFinsQA](https://huggingface.co/datasets/duykhangh/VNFinsQA)** | QA kèm bằng chứng | Đánh giá độ phủ của cổng từ vựng và câu hỏi tài chính |
| **[DocLayNet](https://huggingface.co/datasets/pierreguillou/DocLayNet-base)** | 80.000+ trang tài liệu quốc tế | Pre-train mô hình YOLO hoặc LayoutLMv3 cho đa lớp layout (Table, Chart, Text) |

---

## 2. Cách Tải Dataset Thử Nghiệm

Dự án cung cấp script `scripts/download_datasets.py` để tải các mẫu phục vụ nghiên cứu:

```powershell
.\.venv\Scripts\python.exe scripts/download_datasets.py --dataset vnfinsqa --limit 20
```

Các file tải về sẽ được lưu tại `data/raw/`.
Lưu ý: Dữ liệu tải về không tự động nạp vào Knowledge Graph mà phục vụ việc đánh giá benchmark độc lập.
