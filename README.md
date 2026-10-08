# Multimodal RAG Financial

## Ollama local (mặc định)

Trên máy phát triển, Ollama nằm tại `C:\DXD\Ollama` và model tại `C:\DXD\OllamaModels`.
Máy mới cần tự cài Ollama; repository không chứa executable, model hoặc dữ liệu người dùng.
Smoke test trên máy ngày 29-09-2026: câu hỏi doanh thu báo cáo DEMO trả đúng 125,5 tỷ,
có nguồn [1], khoảng 8,01 giây; CPU, model resident khoảng 2,16 GB, context 4096.
Đây là bài thử ngắn, không đại diện tốc độ/độ chính xác cho mọi báo cáo.

Project dùng `LLM_PROVIDER=ollama`, model `qwen2.5:3b`, endpoint localhost:11434.
Không tự chuyển sang OpenAI khi Ollama lỗi. Bật ô LLM trên giao diện để dùng model local.
`start-local.ps1` khởi động Ollama và ứng dụng; mở http://127.0.0.1:8000.
Model cần tải một lần bằng `ollama pull qwen2.5:3b`; tải xong có thể suy luận offline.
RAM máy đã kiểm tra: khoảng 16 GB; CPU i5-1240P, Intel Iris Xe. Chọn CPU và context
4096, output tối đa 512 token, unload sau 2 phút không dùng để hạn chế RAM.
Model nhỏ có thể sai số liệu hoặc tiếng Việt; luôn đối chiếu nguồn. Model này không OCR ảnh.
`OLLAMA_TIMEOUT=180` có thể tăng nếu CPU chậm. Tắt bớt ứng dụng khi thiếu RAM.
Chi tiết API: https://docs.ollama.com/api/chat ; model: https://ollama.com/library/qwen2.5:3b.
Muốn dùng lại OpenAI phải đặt rõ `LLM_PROVIDER=openai` và API key; đây là lựa chọn có phí.

## Nâng cấp PDF (29-09-2026)

Chạy `.\.venv\Scripts\python.exe -m uvicorn financial_rag.api:app --host 127.0.0.1 --port 8000`
rồi mở http://127.0.0.1:8000 để upload PDF, chọn báo cáo, hỏi đáp, mở nguồn theo trang
và tính toán từ các fact. Tài liệu upload được giữ trong `data/uploads` (đã gitignore).
Chạy một server worker cho kho JSON local; chưa hỗ trợ nhiều worker cùng ghi.

- PDF text: đọc thứ tự trang và trích bảng bằng PyMuPDF; cấu trúc bảng được lưu trong chunk.
- Graph: truy xuất qua Metric → Fact → Chunk kết hợp điểm TF-IDF, lọc theo báo cáo.
- Calculator: tăng trưởng, chênh lệch, tỷ số; kiểm tra đơn vị, kỳ, doanh nghiệp, chia cho 0.
- LLM mặc định: Ollama local, không cần extra `llm` hoặc API key. Chỉ cài extra `llm`
  khi chủ động chuyển sang provider OpenAI. Câu trả lời không vượt qua kiểm tra được
  thay bằng bằng chứng gốc; kiểm tra trích dẫn không bảo đảm đúng ngữ nghĩa.
- OCR: đặt `OCR_ENABLED=true`, `OCR_LANGUAGE=vie+eng`, `TESSDATA_PREFIX` trỏ tới thư mục
  có `vie.traineddata` và `eng.traineddata`. Có bộ đọc bảng scan theo tọa độ cho mẫu
  hai cột kỳ, mã chỉ tiêu và đơn vị VND; phục hồi mã dòng bằng OCR vùng ảnh nhỏ.
  Biểu đồ chưa được hiểu bằng VLM; bố cục bảng khác vẫn có thể không đọc được.
- Đã kiểm thử chọn lọc trên VNM Q1/2026, không phải benchmark độ chính xác tổng quát.
  Bảng trộn đơn vị, đầu cột gộp và OCR mơ hồ cần đối chiếu thủ công.

Chi tiết API LLM đối chiếu [OpenAI Responses](https://developers.openai.com/api/docs/quickstart);
PDF/table/OCR đối chiếu [PyMuPDF Page](https://pymupdf.readthedocs.io/en/latest/page.html).

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
Copy-Item .env.example .env
ollama pull qwen2.5:3b
finrag demo
finrag ask "Doanh thu thuần năm 2024 là bao nhiêu?"
uvicorn financial_rag.api:app --reload
```

Mở `http://127.0.0.1:8000/` để dùng giao diện; `/docs` để thử API.
Lệnh tạo `.env` chỉ dùng khi chưa có file, không ghi đè cấu hình đang sử dụng.

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

Mặc định trả bằng chứng; `POST /ask` hỗ trợ `use_llm=true` và `document_id`.

## Graph schema MVP

Nodes: `Company`, `Report`, `Period`, `Metric`, `Fact`, `Chunk`, `Table`.

Edges: `HAS_REPORT`, `FOR_PERIOD`, `MENTIONS`, `HAS_FACT`, `MEASURES`, `SUPPORTED_BY`, `NEXT`.

Mỗi `Fact` giữ `value`, `unit`, `period`, `source`, `page` để câu trả lời có provenance. Regex extractor chỉ là baseline; interface được tách sẵn để thay bằng structured-output LLM/VLM ở giai đoạn tiếp theo.

## Kiểm thử

### Chạy thử Ollama thật

```powershell
python scripts/smoke_ollama.py --document-id de41cabc2d16430f
```

Script gọi model local đang cấu hình với ba câu hỏi diễn giải, lưu câu trả lời,
bằng chứng, thời gian và lý do fallback trong `work/ollama-smoke.json`.
Giao diện hiển thị tên model và số giây khi thực sự gọi LLM. Câu hỏi số liệu trực tiếp
vẫn dùng calculator/structured facts, dù bật LLM, để tránh giao phép tính cho mô hình.
Phản hồi bị cắt, sai chỉ số trích dẫn hoặc chứa số không xuất hiện trong nguồn được dẫn
sẽ bị thay bằng bằng chứng gốc. Đây là bộ kiểm tra cú pháp/số liệu bảo thủ, không phải
xác minh ngữ nghĩa: câu có dẫn nguồn vẫn có thể diễn giải sai. Các phép làm tròn hợp lệ
cũng có thể bị từ chối; dùng calculator cho phép tính.
Với câu hỏi nguyên nhân, phiên bản này chỉ chấp nhận diễn giải từ chối kết luận;
chưa có bộ xác minh quan hệ nhân quả nên các lời giải thích do LLM sinh ra được
thay bằng nguồn gốc để người dùng kiểm tra. Đây là giới hạn có chủ đích, không phải
bảo đảm rằng bộ lọc có thể phát hiện mọi dạng suy diễn sai.

### Nâng cấp PDF scan và báo cáo VNM Q1/2026

Giao diện tại `http://127.0.0.1:8000/` hỗ trợ nạp nền, tiến độ từng trang,
xem số liệu theo kỳ và mở vùng bằng chứng trên ảnh trang PDF. File gốc không bị sửa.
Hỏi số liệu được ưu tiên trả bằng dữ liệu có cấu trúc; phép tính dùng Decimal,
không giao phép tính cho LLM. Ollama dùng để diễn giải khi bật tùy chọn LLM.

Cài ngôn ngữ OCR trên máy mới:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup-ocr.ps1 -IncludeBest
```

Đặt `OCR_ENABLED=true`, `OCR_LANGUAGE=vie+eng`,
`TESSDATA_PREFIX=C:/DXD/Multimodal-RAG-Financial/data/tessdata` và tùy chọn
`TESSDATA_BEST_PREFIX=C:/DXD/Multimodal-RAG-Financial/data/tessdata-best` trong `.env`.
PyMuPDF sử dụng Tesseract nhúng; không cần cài riêng executable Tesseract.
OCR bổ sung chỉ thử phục hồi ô sai định dạng; vẫn giữ cảnh báo và dấu vết nhận dạng.
Các trang OCR được lưu đệm; lần đọc đầu có thể chậm trên CPU.

API bổ sung: `POST /ingest` với `background=true`, `GET /jobs/{id}`,
`GET /documents/{id}/analysis`, `GET /documents/{id}/pages/{page}?chunk_id=...`,
`GET /documents/{id}/file`, `POST /calculate`.
Chỉ chạy một worker trên localhost. Lịch sử tác vụ lưu tại `data/processed/jobs.json`;
tác vụ đang dở khi restart chuyển sang `interrupted` và cần nạp lại tài liệu.
Không tự tiếp tục tác vụ; OCR cache vẫn được sử dụng khi nạp lại. Giới hạn 8 tác vụ
đang chờ/chạy và lưu tối đa 200 kết quả gần nhất.
Chưa có xác thực người dùng; không mở dịch vụ trực tiếp ra Internet.

### Tổng quan và vận hành (08-10-2026)

- Tổng quan theo từng báo cáo: doanh thu, lợi nhuận, tài sản, vốn, dòng tiền, tăng trưởng
  so với kỳ có trong tài liệu và biên lợi nhuận. Mỗi phép tính giữ công thức và fact đầu vào.
- Graph explorer hiển thị quan hệ Metric → Fact → Chunk, lọc theo báo cáo, không trộn công ty.
- `GET /documents/{id}/export`: tải JSON chứa dữ liệu, chất lượng, nguồn và tổng quan;
  bỏ đường dẫn tuyệt đối trên máy. Không tải dữ liệu ra dịch vụ bên ngoài.
- `GET /documents/{id}/graph`: graph đã lọc; `GET /jobs`: lịch sử xử lý.
- `GET /health/ollama`: kiểm tra kết nối và model local thực tế (timeout 3 giây).
  Chỉ cho phép một lần sinh Ollama đồng thời; câu hỏi khác nhận bằng chứng gốc khi bận.
- Graph/chunks/metadata/jobs được ghi qua file tạm rồi thay thế nguyên tử từng file.
  Đây **không phải** transaction nhiều file: nếu mất điện giữa hai lần ghi, nạp lại báo cáo
  để tái tạo chỉ mục. Vẫn chưa hỗ trợ nhiều worker hoặc triển khai public.
- Các file dữ liệu và PDF hiện có được giữ nguyên. Không tự tải model lớn hay dùng cloud.

Đã kiểm tra báo cáo VNM riêng Q1/2026: OCR 55 trang, trích 44 dòng bảng chuẩn.
Đọc lại vùng mã dòng bị lỗi bằng OCR ảnh cắt; giữ dấu vết phục hồi và không suy số từ công thức.
Kiểm tra cả lưu chuyển tiền tệ và đối chiếu số dư tiền đúng ngày giữa các báo cáo.
Phần so sánh Q1/2025 không có bảng cân đối cùng ngày trong tài liệu này nên vẫn cảnh báo thiếu đối chiếu.
Đây không phải chứng nhận chính xác toàn bộ báo cáo. Bộ đọc bảng hiện tập trung
vào mẫu Việt Nam hai cột kỳ và đơn vị VND; bố cục khác có thể chỉ tìm kiếm được văn bản OCR.
Chưa có GNN, dense embeddings hoặc đọc biểu đồ bằng VLM; truy xuất hiện là TF-IDF + graph.
Không ghép báo cáo riêng/hợp nhất hoặc kỳ không tương thích để tính.

Chạy bộ hồi quy 18 tình huống trên báo cáo thật đã nạp (không chứa PDF trong Git):

```powershell
python scripts/evaluate_vnm.py --document-id de41cabc2d16430f
```

```powershell
pytest -q
ruff check .
```

Xem thêm [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) và [docs/DATASET_ASSESSMENT.md](docs/DATASET_ASSESSMENT.md).

