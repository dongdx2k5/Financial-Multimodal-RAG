# Chạy dự án trong Antigravity trên Windows

Antigravity dùng để mở/chỉnh sửa mã nguồn và chạy terminal. Backend vẫn là Python/FastAPI,
model chạy trong Ollama riêng. Không cần đổi dự án sang framework khác.

## 1. Mở đúng thư mục

Trên máy đã có dự án, chọn `C:\DXD\Multimodal-RAG-Financial` — thư mục chứa
`pyproject.toml`, `README.md`, `src` và `start-local.ps1`.

Trong Antigravity 2.0, chọn **New Project → Add Folder**, chọn thư mục trên, rồi bấm
**Create**, theo [hướng dẫn chính thức](https://antigravity.google/docs/getting-started).
Nếu dùng giao diện IDE khác phiên bản, mở cùng thư mục bằng chức năng mở folder của IDE.

Muốn dùng môi trường và dữ liệu đang có, làm việc với folder local hiện tại.
Clone/worktree mới không mang theo `.env`, `.venv`, PDF và index bị gitignore.

## 2. Máy đã cài sẵn: chạy nhanh

Mở terminal **PowerShell** trong IDE (hoặc PowerShell bên ngoài nếu không thấy terminal):

```powershell
cd C:\DXD\Multimodal-RAG-Financial
Test-Path .\.venv\Scripts\python.exe
Test-Path .\.env
powershell -ExecutionPolicy Bypass -File .\start-local.ps1
```

Hai lệnh kiểm tra phải trả `True` nếu đang dùng môi trường đã thiết lập.
Nếu chưa có `.venv`/`.env`, thực hiện phần máy mới trước khi chạy script.

Khi Uvicorn chạy, mở [http://127.0.0.1:8000/](http://127.0.0.1:8000/).
Giữ terminal mở; **Ctrl+C** dừng web, không nhất thiết dừng Ollama nền.
Nếu cổng 8000 đã có server của dự án đang chạy, dùng server đó, không mở thêm bản thứ hai.

`start-local.ps1` dùng Ollama tại `C:\DXD\Ollama` hoặc PATH và đặt kho
`C:\DXD\OllamaModels` khi phải khởi động server. Nếu model ở kho khác, dùng cách chạy
Ollama/Uvicorn riêng trong [README](../README.md#cài-đặt-trên-windows).

## 3. Máy mới hoặc vừa clone GitHub

Làm đủ các bước [cài đặt trong README](../README.md#cài-đặt-trên-windows):

1. Clone nhánh `main` vào thư mục riêng, mở folder đó trong Antigravity.
2. Cài Python 3.11, tạo `.venv`, chạy `pip install -e .` bằng Python trong `.venv`.
3. Chép `.env.example` sang `.env` **chỉ nếu chưa có `.env`**.
4. Cài/mở Ollama, tải `qwen2.5:3b` và kiểm tra `ollama list`.
5. Chạy Uvicorn bằng Python của `.venv`, không cần activate.
6. Nếu PDF scan, cài tessdata, bật OCR và restart theo [phần OCR](../README.md#bật-ocr-cho-pdf-scan).

Không cần `npm install`, Docker, Neo4j hoặc OpenAI key cho luồng này.
Ollama không tự có model chỉ vì đã clone repository.

## 4. Python interpreter và lệnh chạy

Nếu IDE có chức năng chọn Python interpreter, trỏ đến:

```text
C:\DXD\Multimodal-RAG-Financial\.venv\Scripts\python.exe
```

Kiểm tra môi trường bằng lệnh rõ đường dẫn, chạy từ gốc dự án:

```powershell
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip show multimodal-rag-financial
```

Chọn interpreter không thay thế việc cài dependencies. Nút Run file `api.py` thông thường
không phải lệnh khởi động web; dùng:

```powershell
.\.venv\Scripts\python.exe -m uvicorn financial_rag.api:app --host 127.0.0.1 --port 8000
```

## 5. Kiểm tra trước khi upload

Trong terminal khác khi web đã chạy:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/health/ollama
```

`/health` kiểm tra ứng dụng/chỉ mục; `/health/ollama` mới kiểm tra model local.
Upload PDF, chọn đúng báo cáo, đợi xử lý xong rồi hỏi. Bật LLM khi muốn diễn giải;
các phép tính hỗ trợ vẫn dùng calculator.

Không đọc được PDF scan thì kiểm tra OCR trước — đổi model chat không thay thế OCR.
Thiếu citation hoặc model bận có thể dẫn đến trả bằng chứng gốc kèm cảnh báo.

## 6. Sau khi sửa code hoặc cấu hình

- Ctrl+C dừng web, chạy lại Uvicorn, tải lại trang trình duyệt.
- Đổi dependencies thì cài lại bằng `.\.venv\Scripts\python.exe -m pip install -e .`.
- Chỉ chạy một server ghi cùng kho `data`; không CLI ingest song song với web.
- Không chép `.env.example` đè cấu hình OCR/model đang dùng.
- Không upload `.env`, PDF riêng tư hoặc OCR/index lên GitHub.
- Đưa tài liệu vào chat/agent IDE là luồng riêng với backend Ollama local;
  xem quyền/provider của IDE trước khi gửi dữ liệu nhạy cảm.

Xem thêm [bảng xử lý lỗi](../README.md#xử-lý-lỗi-thường-gặp).
