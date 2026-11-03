# Multimodal RAG Financial

Ứng dụng local đọc hiểu báo cáo tài chính tiếng Việt: tải PDF lên, trích xuất văn bản/bảng,
chuyển số liệu thành đồ thị tri thức, tìm bằng chứng và trả lời kèm nguồn theo trang.
Hỏi đáp **không gọi LLM/Ollama/OpenAI**: dùng bộ luật, công thức và mẫu trả lời có nguồn.
Không cần API key hoặc tải model chat. OCR cho PDF scan mặc định dùng Tesseract.
Có [nhánh thử nghiệm VietOCR + nhận cấu trúc bảng SLANet_plus](docs/OCR_EXPERIMENT.md),
chạy trong môi trường và kho dữ liệu riêng, chưa thay mặc định.
Xem [phạm vi hỏi đáp, mẫu câu và trạng thái kết quả](docs/DETERMINISTIC_QA.md).

Đây là MVP phục vụ nghiên cứu và hỗ trợ đọc báo cáo, không phải công cụ kiểm toán
hay tư vấn đầu tư. Luôn đối chiếu số liệu quan trọng với PDF gốc.

## Mục lục

- [Chức năng và kiến trúc](#chức-năng-và-kiến-trúc)
- [Yêu cầu môi trường](#yêu-cầu-môi-trường)
- [Cài đặt trên Windows](#cài-đặt-trên-windows)
- [Chạy trong Antigravity](#chạy-trong-antigravity)
- [Bật OCR cho PDF scan](#bật-ocr-cho-pdf-scan)
- [Sử dụng ứng dụng](#sử-dụng-ứng-dụng)
- [Cấu hình](#cấu-hình)
- [API và dòng lệnh](#api-và-dòng-lệnh)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Dataset](#dataset)
- [Kiểm thử](#kiểm-thử)
- [Xử lý lỗi thường gặp](#xử-lý-lỗi-thường-gặp)
- [Giới hạn và bảo vệ dữ liệu](#giới-hạn-và-bảo-vệ-dữ-liệu)

## Chức năng và kiến trúc

| Nhóm | Đã triển khai |
|---|---|
| Tài liệu | Upload PDF, TXT, HTML/HTM, MD; giới hạn upload 50 MiB; giữ bản sao tài liệu gốc |
| PDF scan | OCR tiếng Việt/Anh, cache theo trang, trích dòng bảng theo tọa độ và phục hồi mã dòng lỗi |
| Số liệu | Chỉ tiêu, giá trị, đơn vị, kỳ, phạm vi riêng/hợp nhất, trang và vùng bằng chứng |
| Kiểm tra | Công thức kế toán, đối chiếu số dư tiền, cảnh báo dữ liệu thiếu/không khớp |
| Tổng quan | Doanh thu, lợi nhuận, tài sản, vốn chủ sở hữu, dòng tiền, tiền cuối kỳ; tăng trưởng và biên lợi nhuận khi đủ dữ liệu |
| Hỏi đáp | Facts + calculator + mẫu trả lời; TF-IDF/graph trả đoạn liên quan khi chưa có luật đủ để trả lời |
| Truy nguồn | Mở PDF/trang/vùng nguồn; xem quan hệ Metric → Fact → Chunk; xuất phân tích JSON |
| Vận hành | Nạp nền, tiến độ từng trang, lịch sử tác vụ và trạng thái ứng dụng |

```text
PDF / TXT / HTML / MD
        │
        ▼
Đọc text, bảng / OCR nếu là ảnh scan
        │
        ▼
Chunks + Facts + trang / tọa độ / kỳ / phạm vi
        │
        ├── Kiểm tra công thức và chất lượng số liệu
        ▼
Knowledge Graph G = (V, E) + chỉ mục TF-IDF
        │
        ▼
Câu hỏi + báo cáo được chọn
        ├── Số liệu / phép tính hỗ trợ → facts + calculator
        └── Câu chưa có luật → retrieval → đoạn trích, ghi rõ chưa kết luận
                              Không gọi mô hình sinh câu trả lời
```

`V` gồm Company, Report, Period, Metric, Fact, Chunk. `E` lưu quan hệ sở hữu báo cáo,
chỉ tiêu và bằng chứng. Graph được tạo bằng bộ trích xuất theo quy tắc/bố cục,
**không phải do LLM tự vẽ hoặc huấn luyện GNN**. Bảng nằm trong chunk, chưa có node Table riêng.

“Multimodal” ở phiên bản này là xử lý văn bản, bảng và ảnh trang qua OCR.
**Chưa có** mô hình hiểu biểu đồ/VLM, GNN, dense embeddings hoặc Neo4j.
Xem [kiến trúc chi tiết và schema](docs/ARCHITECTURE.md).

## Yêu cầu môi trường

- Python **3.11** là phiên bản dùng kiểm thử; package khai báo Python từ 3.11.
- Git để clone/cập nhật mã nguồn.
- Không cần Ollama hoặc OpenAI SDK để chạy hỏi đáp.
- Internet cho lần cài thư viện và bộ ngôn ngữ OCR. Sau khi tải đủ,
  luồng xử lý tài liệu và hỏi đáp chạy local.
- Máy đã chạy thử: i5-1240P, RAM 16 GB, Intel Iris Xe. Đây không phải cấu hình tối thiểu
  được bảo đảm; OCR CPU có thể chậm.

Repository **không chứa** `.venv`, `.env`, Ollama executable, model, PDF người dùng
hoặc kho chỉ mục đã nạp. Máy mới phải cài môi trường và nạp tài liệu lại.

## Cài đặt trên Windows

Các lệnh dưới đây chạy trong **PowerShell**, không cần activate virtual environment.

### 1. Lấy mã nguồn

Nếu chưa có thư mục dự án:

```powershell
New-Item -ItemType Directory -Force -Path C:\DXD | Out-Null
cd C:\DXD
git clone --branch main https://github.com/dongdx2k5/Financial-Multimodal-RAG.git Multimodal-RAG-Financial
cd .\Multimodal-RAG-Financial
```

Nếu đã có dự án ở `C:\DXD\Multimodal-RAG-Financial`, mở đúng thư mục đó;
không clone đè và không tạo lại môi trường đang dùng tốt.

### 2. Cài packages và tạo cấu hình

Nếu máy không có `py`, dùng `python` sau khi kiểm tra phiên bản.

```powershell
cd C:\DXD\Multimodal-RAG-Financial
if (-not (Test-Path -LiteralPath .venv)) {
    py -3.11 -m venv .venv
}
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .
if (-not (Test-Path -LiteralPath .env)) {
    Copy-Item -LiteralPath .env.example -Destination .env
}
```

Không ghi đè `.env` đang dùng; không cần điền API key.

### 3. Khởi động web — không cần Ollama

```powershell
cd C:\DXD\Multimodal-RAG-Financial
.\.venv\Scripts\python.exe -m uvicorn financial_rag.api:app --host 127.0.0.1 --port 8000
```

- Giao diện: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- Thử API: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Kiểm tra ứng dụng: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

Giữ terminal chạy, nhấn **Ctrl+C** để dừng web. Chỉ dùng **một server worker**;
không chạy nhiều tiến trình cùng ghi vào `data`.

### Lệnh tắt trên máy đã cấu hình tại C:\DXD

```powershell
cd C:\DXD\Multimodal-RAG-Financial
powershell -ExecutionPolicy Bypass -File .\start-local.ps1
```

Script chạy Python của `.venv` trong chính thư mục dự án, không khởi động Ollama.
Script không tự cài Python packages hoặc OCR.

## Chạy trong Antigravity

Mở thư mục chứa `pyproject.toml` và `README.md`, rồi chạy các lệnh PowerShell phía trên.
Antigravity là môi trường mở/chỉnh sửa dự án; ứng dụng chạy bằng Python/FastAPI.
Không cần `npm install` hoặc API key chỉ để chạy ứng dụng này.

Xem [hướng dẫn Antigravity từng bước](docs/ANTIGRAVITY.md): mở project,
chọn Python, chạy trên máy đã cài sẵn và kiểm tra kết nối.

## Bật OCR cho PDF scan

PDF có thể bôi chọn chữ thường đọc được trực tiếp. PDF là ảnh scan cần OCR:

```powershell
cd C:\DXD\Multimodal-RAG-Financial
powershell -ExecutionPolicy Bypass -File .\scripts\setup-ocr.ps1 -IncludeBest
```

Script tải `vie.traineddata` và `eng.traineddata` từ kho Tesseract chính thức vào
`data/tessdata` và `data/tessdata-best`. Bỏ `-IncludeBest` nếu chỉ cần bộ fast.
PyMuPDF dùng Tesseract qua thư viện; luồng này không gọi executable Tesseract riêng.

Sửa các dòng tương ứng trong `.env` (đổi đường dẫn nếu đặt dự án ở nơi khác):

```dotenv
OCR_ENABLED=true
OCR_LANGUAGE=vie+eng
TESSDATA_PREFIX=C:/DXD/Multimodal-RAG-Financial/data/tessdata
TESSDATA_BEST_PREFIX=C:/DXD/Multimodal-RAG-Financial/data/tessdata-best
```

Chỉ đặt `TESSDATA_BEST_PREFIX` nếu đã tải bộ best. Restart web sau khi sửa `.env`,
rồi nạp lại PDF. Lần OCR đầu có thể lâu; những lần sau có thể dùng cache.
Bộ đọc bảng hiện tập trung vào báo cáo Việt Nam hai cột kỳ, mã chỉ tiêu, đơn vị VND.

## Sử dụng ứng dụng

1. Upload tài liệu; điền mã cổ phiếu và năm nếu biết. Bản gốc không bị chỉnh sửa.
2. Theo dõi tiến độ nạp nền. Đợi hoàn tất rồi chọn đúng báo cáo.
3. Xem tổng quan, facts và cảnh báo chất lượng. Giá trị thiếu/không khớp cần kiểm tra nguồn.
4. Đặt câu hỏi phù hợp tài liệu, ví dụ “Doanh thu thuần quý 1 năm 2026 là bao nhiêu?”,
   “Biên lợi nhuận gộp quý 1 năm 2026?”, “Báo cáo quy định khi nào ghi nhận doanh thu bán hàng?”.
5. Đọc nhãn kết quả: đã trả lời / chỉ có đoạn liên quan / cần làm rõ / chưa đủ bằng chứng.
   Mở “Cách tạo kết quả” để xem các bước xử lý. Câu hỏi chính sách chỉ trả đoạn liên quan,
   chưa tự xác định điều kiện hay nguyên nhân.
6. Mở nguồn theo trang/vùng PDF, xem graph hoặc tải JSON phân tích để kiểm tra tiếp.

Số trang nguồn là vị trí trang PDF tính từ 1, có thể khác số trang in trên báo cáo.
Chỉ đánh dấu vùng màu cam khi chunk có tọa độ; nếu thiếu tọa độ, xem trang và đoạn trích.
Kết quả tìm kiếm không được coi là câu trả lời đã xác minh. Câu ngoài phạm vi sẽ không được đoán.

## Cấu hình

Đọc từ `.env` ở gốc repository; khởi động lại server sau khi thay đổi.

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `OCR_ENABLED` | `false` | Bật OCR cho trang scan |
| `OCR_LANGUAGE` | `vie+eng` | Ngôn ngữ OCR đã cài |
| `TESSDATA_PREFIX` | Chưa đặt | Thư mục chứa các file ngôn ngữ OCR |
| `TESSDATA_BEST_PREFIX` | Chưa đặt | Bộ best tùy chọn để hỗ trợ phục hồi ô số |
| `DATA_DIR` | `<repo>/data` | Kho dữ liệu local; nên dùng đường dẫn tuyệt đối nếu đổi |

`EMBEDDING_MODEL` có trong cấu hình cũ nhưng chưa dùng để tạo dense embeddings.
Các cấu hình `LLM_PROVIDER`, `OLLAMA_*`, `OPENAI_*` được giữ để tương thích với
client/thử nghiệm cũ, **không bật được LLM trong `/ask`**. Không cần extra `llm`.

## API và dòng lệnh

Schema request/response đầy đủ nằm trong `/docs` khi server đang chạy.

| Endpoint | Chức năng |
|---|---|
| `GET /health` | `answer_policy=deterministic`, `llm_enabled=false`, kích thước chỉ mục |
| `GET /health/ollama` | Chẩn đoán client cũ tùy chọn; giao diện hỏi đáp không gọi endpoint này |
| `POST /ingest` | Multipart `file`, tùy chọn `ticker`, `year`, `background=true` để nạp nền |
| `GET /jobs`, `GET /jobs/{job_id}` | Lịch sử và tiến độ nạp |
| `GET /documents` | Danh sách báo cáo và ID |
| `GET /documents/{id}/analysis` | Metadata, facts, kiểm tra chất lượng, tổng quan |
| `GET /documents/{id}/export` | Tải JSON phân tích, rút gọn trường đường dẫn nguồn |
| `GET /documents/{id}/graph` | Graph theo báo cáo; `limit` tối đa 1.000 node |
| `GET /documents/{id}/file` | Tài liệu gốc đã lưu |
| `GET /documents/{id}/pages/{page}` | Ảnh PNG của trang PDF; `chunk_id` tùy chọn để đánh dấu nguồn |
| `POST /ask` | `question`, `document_id`, `top_k` (1–20); `use_llm` cũ bị bỏ qua kèm cảnh báo |
| `POST /calculate` | `operation`: `growth`, `difference`, `ratio`; hai fact ID |
| `GET /graph` | Graph toàn kho, giới hạn mặc định 500 |

Ví dụ PowerShell khi đã nạp ít nhất một báo cáo (chọn đúng ID trước khi hỏi):

```powershell
$reports = @(Invoke-RestMethod http://127.0.0.1:8000/documents)
$reports | Format-Table id, name, ticker, year
$reportId = Read-Host 'Nhập id báo cáo trong danh sách'
$body = @{
    question = 'Doanh thu thuần trong báo cáo là bao nhiêu?'
    document_id = $reportId
    top_k = 5
    use_llm = $false
} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/ask -ContentType 'application/json; charset=utf-8' -Body ([System.Text.Encoding]::UTF8.GetBytes($body))
```

CLI dành cho demo/nạp dữ liệu ngoài giao diện. Dừng web trước khi nạp bằng CLI để tránh
hai tiến trình ghi cùng kho; khởi động lại web sau đó để đọc chỉ mục mới.

```powershell
.\.venv\Scripts\python.exe -m financial_rag.cli demo
.\.venv\Scripts\python.exe -m financial_rag.cli ask "Doanh thu thuần năm 2024 là bao nhiêu?"
# Thay đường dẫn dưới đây bằng tài liệu thật của bạn:
.\.venv\Scripts\python.exe -m financial_rag.cli ingest "C:\duong-dan\bao-cao.pdf" --ticker VNM --year 2026
```

CLI `ask` nhận `--document-id ID`. Nếu có nhiều báo cáo mà không chọn ID, hệ thống yêu cầu
chọn báo cáo; nếu chỉ có một báo cáo thì tự chọn. Demo là dữ liệu giả lập, không phải báo cáo thật;
dữ kiện regex trong TXT hiện chỉ hỗ trợ truy xuất đoạn, không tự được coi là fact bảng chuẩn.

## Cấu trúc dự án

```text
src/financial_rag/
  api.py, cli.py                 Web API và dòng lệnh
  app.html, dashboard.js         Giao diện, tổng quan và graph explorer
  ingestion.py                  Đọc tài liệu, OCR, chia chunks
  financial_tables.py           Trích dòng bảng theo tọa độ
  ocr_recovery.py                Phục hồi OCR mã dòng
  extraction.py, models.py       Trích facts và schema dữ liệu
  graph.py, retrieval.py         NetworkX graph và TF-IDF + graph retrieval
  answers.py, calculator.py      Hỏi số liệu và phép tính
  quality.py, analytics.py       Kiểm tra kế toán và tổng quan
  ollama_client.py               Client cũ độc lập, không được service.ask gọi
  service.py, config.py          Điều phối và cấu hình
  storage.py, jobs.py            Ghi JSON và tác vụ nền
scripts/                        Cài OCR, tải dataset, kiểm thử báo cáo/Ollama
tests/                          Kiểm thử tự động
samples/                        Báo cáo demo dạng text
docs/                           Hướng dẫn và kiến trúc
data/uploads/                   Bản sao tài liệu người dùng, không đưa lên Git
data/processed/                 OCR cache, metadata báo cáo, jobs.json
data/index/                     graph.json, chunks.json
data/raw/                       Dataset tải tùy chọn
start-local.ps1                 Lệnh chạy tắt cho cấu hình máy C:\DXD
```

## Dataset

Không cần tải dataset để upload và hỏi đáp PDF của bạn. Các nguồn sau được xem xét
cho nghiên cứu/mở rộng, **không có nghĩa model đã được huấn luyện hay benchmark trên chúng**:

| Dataset | Vai trò dự kiến |
|---|---|
| [AIGuruTinix/ViFinQA](https://huggingface.co/datasets/AIGuruTinix/ViFinQA) | Corpus báo cáo/QA tiếng Việt cho retrieval |
| [VietFinTabGroup/VietFinTab](https://huggingface.co/datasets/VietFinTabGroup/VietFinTab) | Nghiên cứu nhận dạng cấu trúc bảng |
| [ngnquan/ViFinTab](https://huggingface.co/datasets/ngnquan/ViFinTab) | Nghiên cứu table detection |
| [duykhangh/VNFinsQA](https://huggingface.co/datasets/duykhangh/VNFinsQA) | Đánh giá QA tiếng Việt |
| [dreamerdeo/finqa](https://huggingface.co/datasets/dreamerdeo/finqa) | Nghiên cứu suy luận số học tiếng Anh |

Xem [đánh giá dataset ban đầu](docs/DATASET_ASSESSMENT.md). Kiểm tra dataset card,
giấy phép và quyền sử dụng tài liệu gốc trước khi tải, phân phối hoặc dùng thương mại.

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[datasets]"
.\.venv\Scripts\python.exe scripts/download_datasets.py --dataset vnfinsqa --limit 20
```

Script xuất JSONL vào `data/raw`; **chưa tự nạp JSONL vào graph hoặc huấn luyện model**.
`--limit` chỉ giới hạn số dòng xuất mỗi split, không bảo đảm giới hạn dung lượng tải:
`load_dataset` chạy trước bước chọn dòng. Riêng lựa chọn `finqa` hiện bật
`trust_remote_code`; cần xem mã dataset trước khi chạy.

## Kiểm thử

Cài extra cho bộ test; không yêu cầu API key, Ollama hoặc gọi dịch vụ trả phí:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check src tests scripts
```

CI GitHub Actions dùng Python 3.11 với cùng lệnh kiểm thử/lint.
Bộ test không thay cho việc đánh giá độ chính xác trên nhiều loại báo cáo.

Kiểm thử tùy chọn trên dữ liệu thật đã nạp; thay `$reportId` bằng ID từ `GET /documents`:

```powershell
# Chỉ dành cho đúng báo cáo VNM RIÊNG Q1/2026 dùng trong bộ hồi quy:
.\.venv\Scripts\python.exe scripts/evaluate_vnm.py --document-id $reportId
```

- `smoke_ollama.py` cũ đã ngừng dùng; hỏi đáp không còn gọi mô hình sinh.
- Bộ VNM có 18 tình huống cố định. Báo cáo từng kiểm tra có 55 trang, trích 44 dòng bảng chuẩn;
  PDF không đi kèm Git và không được script tự tải. Không dùng bộ này cho báo cáo khác.
- Đây là kiểm thử chọn lọc, không phải cam kết chính xác cho mọi số liệu hoặc mọi PDF.

## Xử lý lỗi thường gặp

| Hiện tượng | Cách kiểm tra/xử lý |
|---|---|
| Không có `py` / sai Python | Chạy `python --version`; cài Python 3.11 và mở lại terminal |
| Thiếu `financial_rag` / `uvicorn` | Đứng ở gốc repo, cài `-m pip install -e .` bằng `.venv\Scripts\python.exe` |
| Chặn `Activate.ps1` | Không cần activate; dùng trực tiếp `.venv\Scripts\python.exe` |
| Không có lệnh `ollama` | Không cần cài: hỏi đáp hiện không dùng Ollama |
| Cổng 8000 đang được dùng | Dùng server đang chạy hoặc Ctrl+C đúng terminal cũ; nếu đổi `--port 8001`, mở URL tương ứng |
| PDF scan không có nội dung | Bật OCR, kiểm tra hai file ngôn ngữ/đường dẫn trong `.env`, restart rồi nạp lại |
| Hỏi đáp chậm lúc OCR | Ingest giữ khóa kho dữ liệu; đợi hoàn tất rồi hỏi. Xem tiến độ qua `/jobs` |
| Chỉ có đoạn liên quan | Câu hỏi chưa thuộc bộ luật hoặc chưa có fact bảng chuẩn; không phải đã trả lời |
| Job `interrupted` sau restart | Nạp lại cùng tài liệu và metadata; OCR cache có thể được dùng lại, không tự resume |
| Thiếu số liệu cho kỳ hỏi | Chọn đúng báo cáo/kỳ/phạm vi; hệ thống không tự suy dữ liệu không có trong nguồn |

## Giới hạn và bảo vệ dữ liệu

- Chỉ triển khai local trên `127.0.0.1`, **chưa có xác thực người dùng**. Không mở trực tiếp
  ra Internet hoặc coi `--host 0.0.0.0` là cấu hình production.
- Dùng một tiến trình ghi dữ liệu; không CLI ingest song song với web hoặc chạy nhiều worker.
  JSON được thay thế nguyên tử từng file, nhưng không có transaction nhiều file.
  Nếu gián đoạn giữa các lần ghi, nạp lại báo cáo để tái tạo dữ liệu liên quan.
- Tối đa 8 tác vụ chờ/chạy; lịch sử giữ tối đa 200 bản ghi, `/jobs` trả 30 gần nhất.
  Job đang dở khi restart chuyển `interrupted`, không tự tiếp tục.
- Bảng khác bố cục, đơn vị trộn, ô gộp hoặc ảnh mờ có thể đọc sai/thiếu.
  Công thức khớp không có nghĩa OCR đúng hoặc báo cáo đã được kiểm toán.
- Calculator từ chối kỳ/phạm vi/đơn vị không tương thích, chia cho 0 và tăng trưởng với nền âm.
  Chưa hỗ trợ so sánh tự do nhiều báo cáo hoặc lập chương trình tài chính tổng quát.
- Không có LLM không đồng nghĩa luôn đúng: OCR, ánh xạ mã dòng và luật vẫn có thể sai.
  Chưa trả lời tự động về danh sách HĐQT, nguyên nhân hay suy luận mở; chỉ truy xuất đoạn liên quan.
- Dữ liệu và `.env` được gitignore nhưng vẫn tồn tại trên máy. Không commit PDF, OCR,
  export JSON hoặc khóa bí mật. Khi sao lưu, dừng tiến trình ghi và giữ cả cấu hình lẫn dữ liệu.
- Hỏi đáp không gửi dữ liệu đến LLM. Đưa tài liệu vào trợ lý AI của IDE là luồng khác;
  cần xem nơi dữ liệu được gửi trước khi sử dụng.

