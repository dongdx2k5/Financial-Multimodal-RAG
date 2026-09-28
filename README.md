# Multimodal RAG Financial

MVP đọc hiểu báo cáo tài chính tiếng Việt theo luồng:

`PDF / OCR text / table -> chunks + facts -> Knowledge Graph G=(V,E) -> hybrid retrieval -> answer + citations`

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
pip install -e ".[dev]"
finrag demo
finrag ask "Doanh thu thuần năm 2024 là bao nhiêu?"
uvicorn financial_rag.api:app --reload
```

Mở `http://127.0.0.1:8000/docs` để thử API.

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
```

Không tải toàn bộ dataset ảnh theo mặc định để tránh dùng nhiều dung lượng. Dữ liệu nằm trong `data/raw`, chỉ mục trong `data/index`.

## API

- `GET /health`: trạng thái service/index.
- `POST /ingest`: upload PDF/TXT/HTML, trích xuất nội dung và graph facts.
- `POST /ask`: hybrid retrieval từ lexical index và graph, trả câu trả lời trích dẫn nguồn.
- `GET /graph`: xem nodes/edges (giới hạn bằng query `limit`).

MVP hiện chạy ở chế độ evidence-first, trả các đoạn bằng chứng xếp hạng cao. Adapter sinh câu trả lời bằng LLM/VLM là hạng mục phase 2 để tránh khóa kiến trúc vào một nhà cung cấp ngay từ đầu.

## Graph schema MVP

Nodes: `Company`, `Report`, `Period`, `Metric`, `Fact`, `Chunk`, `Table`.

Edges: `HAS_REPORT`, `FOR_PERIOD`, `MENTIONS`, `HAS_FACT`, `MEASURES`, `SUPPORTED_BY`, `NEXT`.

Mỗi `Fact` giữ `value`, `unit`, `period`, `source`, `page` để câu trả lời có provenance. Regex extractor chỉ là baseline; interface được tách sẵn để thay bằng structured-output LLM/VLM ở giai đoạn tiếp theo.

## Kiểm thử

```powershell
pytest -q
ruff check .
```

Xem thêm [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) và [docs/DATASET_ASSESSMENT.md](docs/DATASET_ASSESSMENT.md).

