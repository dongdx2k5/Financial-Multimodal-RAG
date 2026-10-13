# Multimodal RAG Financial

Hệ thống đọc hiểu báo cáo tài chính **song ngữ Việt – Anh** theo luồng:

`PDF / OCR text / table -> chunks + facts -> Knowledge Graph G=(V,E) -> hybrid retrieval (TF-IDF + Dense Embedding) -> LLM answer + citations`

## Dataset được chọn

| Vai trò | Dataset | Quyết định |
|---|---|---|
| Corpus báo cáo và câu hỏi tiếng Việt | `AIGuruTinix/ViFinQA` | Nguồn chính cho retrieval/ingestion |
| Ảnh bảng + cấu trúc bảng | `VietFinTabGroup/VietFinTab` | Nguồn chính cho nhánh multimodal/table |
| Table detection nhỏ hơn | `ngnquan/ViFinTab` | Baseline phụ cho detector, không trộn nhãn mặc định |
| QA tiếng Việt có ground truth | `duykhangh/VNFinsQA` | Evaluation set phụ; cần kiểm tra độ tin cậy bằng chứng |
| Numerical reasoning tiếng Anh | `dreamerdeo/finqa` | Benchmark phụ cho calculator/reasoning |

Lưu ý giấy phép: ViFinQA và VietFinTab dùng dữ liệu CC BY-NC 4.0; MVP mặc định phục vụ nghiên cứu, không mặc định cho mục đích thương mại.

## Chạy nhanh

```powershell
cd C:\DXD\Multimodal-RAG-Financial
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev,llm]"
```

### Cấu hình API key (miễn phí)

Tạo file `.env` từ `.env.example` và điền Google API key (lấy miễn phí tại https://aistudio.google.com/apikey):

```powershell
copy .env.example .env
# Sửa GOOGLE_API_KEY= trong file .env
```

### Demo song ngữ

```powershell
# Demo tiếng Việt
finrag demo
finrag ask "Doanh thu thuần năm 2024 là bao nhiêu?"

# Demo tiếng Anh (Apple Inc.)
finrag demo-en
finrag ask "What was Total Revenue in 2024?"

# Demo cả hai ngôn ngữ
finrag demo-all
finrag ask "Total Revenue của DEMO năm 2024?"
```

### API server

```powershell
uvicorn financial_rag.api:app --reload
```

Mở `http://127.0.0.1:8000/docs` để thử API.

## Chế độ hoạt động

| Chế độ | Điều kiện | Mô tả |
|---|---|---|
| **Evidence-first** | Không có `GOOGLE_API_KEY` | Trả đoạn bằng chứng xếp hạng bằng TF-IDF, chạy offline hoàn toàn |
| **LLM + Dense Embedding** | Có `GOOGLE_API_KEY` | Tìm kiếm chéo ngôn ngữ (Việt ↔ Anh) bằng Gemini Embedding + sinh câu trả lời bằng Gemini LLM |

## Nạp dữ liệu

Nạp một file PDF/TXT/HTML:

```powershell
finrag ingest C:\duong-dan\bao-cao.pdf --ticker HPG --year 2024
```

Tải metadata hoặc subset dataset (cần extra `datasets`):

```powershell
pip install -e ".[datasets]"
python scripts/download_datasets.py --dataset vnfinsqa
python scripts/download_datasets.py --dataset vietfintab --limit 100
python scripts/download_datasets.py --dataset finqa --limit 100
```

Không tải toàn bộ dataset ảnh theo mặc định để tránh dùng nhiều dung lượng. Dữ liệu nằm trong `data/raw`, chỉ mục trong `data/index`.

## API

- `GET /health`: trạng thái service/index.
- `GET /documents`: liệt kê các báo cáo đã nạp.
- `GET /documents/{id}`: chi tiết một báo cáo.
- `DELETE /documents/{id}`: xóa báo cáo.
- `GET /facts`: truy vấn dữ kiện tài chính có cấu trúc.
- `POST /ingest`: upload PDF/TXT/HTML, trích xuất nội dung và graph facts.
- `POST /ask`: hybrid retrieval (TF-IDF + Dense Embedding) + LLM, trả câu trả lời trích dẫn nguồn.
- `GET /graph`: xem nodes/edges (giới hạn bằng query `limit`).

## Graph schema MVP

Nodes: `Company`, `Report`, `Period`, `Metric`, `Fact`, `Chunk`, `Table`.

Edges: `HAS_REPORT`, `FOR_PERIOD`, `MENTIONS`, `HAS_FACT`, `MEASURES`, `SUPPORTED_BY`, `NEXT`.

Mỗi `Fact` giữ `value`, `unit`, `period`, `source`, `page` để câu trả lời có provenance. Regex extractor chỉ là baseline; interface được tách sẵn để thay bằng structured-output LLM/VLM ở giai đoạn tiếp theo.

## Kiểm thử

```powershell
pytest -q
ruff check .
```

Xem thêm [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), [docs/DATASET_ASSESSMENT.md](docs/DATASET_ASSESSMENT.md) và [docs/ROADMAP.md](docs/ROADMAP.md).
