# Hướng Dẫn Sử Dụng Trong Google Antigravity (Antigravity IDE Guide)

Tài liệu hướng dẫn mở, cấu hình và chạy dự án trong môi trường Google Antigravity.

---

## 1. Mở Dự Án
1. Trong giao diện Antigravity, chọn **File** -> **Open Folder...** -> Chọn thư mục `C:\RAG`.
2. Antigravity sẽ tự động nhận diện cấu trúc file `pyproject.toml` và các thư mục `src/`, `tests/`.

---

## 2. Thiết Lập Môi Trường Ảo
1. Mở Terminal trong Antigravity (PowerShell).
2. Kiểm tra môi trường ảo `.venv`:
   ```powershell
   .\.venv\Scripts\python.exe --version
   ```
3. Nếu cần cài đặt các gói mới:
   ```powershell
   .\.venv\Scripts\python.exe -m pip install -e ".[dev]"
   ```

---

## 3. Khởi Chạy Ứng Dụng
Chạy lệnh khởi động nhanh:
```powershell
powershell -ExecutionPolicy Bypass -File .\start-local.ps1
```
- Web UI: http://127.0.0.1:8000/
- API Docs: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health
