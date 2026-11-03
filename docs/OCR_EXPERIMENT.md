# Thử nghiệm VietOCR và cấu trúc bảng

## Phạm vi

Tesseract vẫn là mặc định. `.env`, PDF nguồn và chỉ mục chính không bị thay đổi bởi
script benchmark hay script khởi động phòng thử nghiệm. Không gọi Ollama/OpenAI.

- PDF có text layer đủ dùng: tiếp tục lấy chữ trực tiếp bằng PyMuPDF.
- PDF scan: tùy chọn PP-OCRv5_mobile_det qua RapidOCR/ONNX Runtime để tìm dòng,
  VietOCR `vgg_seq2seq` để nhận chữ tiếng Việt trên ảnh dòng.
- SLANet_plus qua RapidTable: nhận cấu trúc trên **vùng bảng**, ghép OCR theo tọa độ.
  Ưu tiên vùng bảng PyMuPDF; với một số trang báo cáo tài chính có thể khoanh vùng
  bằng quy tắc và ghi rõ `region_source=statement_heuristic`. Chưa có model layout
  tổng quát cho mọi bảng thuyết minh.
- Dữ liệu trả về gồm ô, hàng, cột, rowspan/colspan, bbox và nội dung OCR nguyên gốc.
  Vùng chữ giao nhiều ô bị bỏ khỏi bảng, không tự chia chuỗi để đoán cột.
- Chunk cấu trúc thử nghiệm chỉ dùng làm bằng chứng truy xuất. Không tự sinh
  FinancialFact từ chunk này vì việc hiểu tiêu đề kỳ/đơn vị/ô gộp chưa được xác minh.
  Trang VietOCR cũng chưa sinh FinancialFact: parser cột cũ đòi bbox từng từ,
  trong khi bbox dòng có thể chứa nhiều cột số. Chế độ thử nghiệm hiện phục vụ
  đọc và truy xuất bằng chứng. Tính toán vẫn dùng nhánh Tesseract/text đã có.

Đây không phải tuyên bố VietOCR chính xác hoặc nhanh hơn Tesseract. Cần chấm trên
cùng báo cáo, đặc biệt là số âm, dấu phân cách và việc gán đúng cột/kỳ.

## Cài môi trường riêng trên Windows

Yêu cầu Python 3.11, mạng để tải thư viện/model, vài GB đĩa trống. Không cần GPU NVIDIA.
Trong terminal tại thư mục dự án:

```powershell
cd C:\DXD\Multimodal-RAG-Financial
powershell -File .\scripts\setup-ocr-lab.ps1
```

Script tạo `work/ocr-lab/.venv`, cài CPU PyTorch, RapidOCR 3.10.0,
RapidTable 3.0.2, ONNX Runtime 1.31.1 và VietOCR từ commit upstream cố định
`fe8c3a7fc714aec57ab81cec844eb3adf0c1636c`.
Không dùng PyPI VietOCR cũ có ràng buộc thư viện dễ xung đột NumPy 2.
Model nhận chữ tải từ máy chủ tác giả; detector và SLANet_plus tải từ nguồn RapidAI,
có kiểm tra SHA256 theo manifest chính thức. Trọng số lưu tại `work/ocr-lab/models`.
Không tự tải model khi gọi `/ingest`.

Nếu Windows báo **Application Control policy has blocked this file**, dừng kiểm tra
và nhờ quản trị viên xem xét DLL/gói chính thức. Không tắt hay sửa chính sách bảo mật.

## Chạy giao diện thử nghiệm, không ảnh hưởng dữ liệu chính

```powershell
powershell -File .\scripts\start-ocr-lab.ps1 -Tables
```

Mở http://127.0.0.1:8001 và nạp PDF. Kho dữ liệu nằm tại
`work/ocr-lab/app-data`, không phải `data/index` của ứng dụng chính.
Chế độ này đặt fallback=false để việc thiếu model không bị che bằng Tesseract.
Nếu VietOCR lỗi ở bất kỳ trang cần OCR nào, việc nạp tài liệu sẽ dừng, tránh thay
chỉ mục bằng một bản tài liệu chỉ đọc được một phần.

Đối chiếu bằng cùng môi trường nhưng Tesseract:

```powershell
powershell -File .\scripts\start-ocr-lab.ps1 -OcrBackend tesseract -Port 8002
```

Không chạy hai server cùng ghi kho dữ liệu lab một lúc. Dừng server thứ nhất bằng
Ctrl+C rồi chạy server còn lại; nạp lại PDF để thay kết quả trong kho lab.

## Benchmark PDF

```powershell
& .\work\ocr-lab\.venv\Scripts\python.exe scripts/benchmark_ocr.py 'C:\path\report.pdf' `
  --pages 7,8,10 --engines tesseract,vietocr `
  --weights work/ocr-lab/models/vgg_seq2seq.pth `
  --detector work/ocr-lab/models/ch_PP-OCRv5_det_mobile.onnx `
  --tessdata data/tessdata --output work/ocr-benchmark --render
```

Xuất `report.json`, `report.md` và PNG vùng OCR. Số trang là thứ tự trang PDF, từ 1,
không nhất thiết trùng số in ở chân trang. Mặc định không ghi đè report có sẵn.
Thời gian lần đầu bao gồm khởi tạo model; không phải benchmark tốc độ thuần model.

Để kiểm tra thêm cấu trúc, thêm `--tables --table-model work/ocr-lab/models/slanet-plus.onnx`.
Có thể khoanh đúng một bảng với `--pages 8 --table-region 60,145,535,665` (tọa độ
chỉ là ví dụ, cần chọn lại cho tài liệu thực tế). JSON lưu từng ô và PNG vẽ bbox;
chưa có nhãn cấu trúc thì không tính độ chính xác hàng/cột.

Nếu **không có nhãn kiểm chứng**, chỉ báo bất đồng giữa hai engine, không tính độ chính xác.
Có thể thêm `--reference path/to/reference.json`, định dạng:

```json
{"samples":[{"id":"p8-cell-1","page":8,"bbox":[100,200,180,215],
"text":"(1.250.000)","kind":"number"}]}
```

Mỗi bbox phải là **một dòng**, theo tọa độ PDF không xoay (đơn vị point, 72 point/inch).
Nhãn phải đọc và kiểm tra thủ công từ ảnh gốc, không lấy OCR của một engine làm đáp án.
Hai engine đọc cùng crop có viền trắng. CER giữ dấu tiếng Việt, dấu âm và dấu phân cách.
`exact_text` yêu cầu đúng toàn chuỗi sau chuẩn hóa Unicode và khoảng trắng.
Điểm crop không đo độ đúng phát hiện dòng, cấu trúc bảng hay trả lời tài chính.

## Cấu hình ứng dụng (chỉ bật sau khi đã đánh giá)

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `OCR_BACKEND` | `tesseract` | `tesseract` hoặc `vietocr` |
| `OCR_FALLBACK_TO_TESSERACT` | `true` | Cho phép dự phòng khi VietOCR lỗi, luôn ghi cảnh báo |
| `VIETOCR_WEIGHTS` | trống | File `vgg_seq2seq.pth` cục bộ |
| `OCR_DETECTOR_PATH` | trống | File PP-OCRv5 mobile detector ONNX cục bộ |
| `TABLE_BACKEND` | `none` | `none` hoặc `slanet_plus` |
| `TABLE_MODEL_PATH` | trống | Đường dẫn bắt buộc đến SLANet_plus ONNX khi bật bảng |

`OCR_ENABLED=true` vẫn cần để OCR trang scan. Đường Tesseract cần bộ `vie+eng` như trước.
Cache phân biệt backend, đường dẫn/model thay đổi và tùy chọn bảng. Kết quả fallback
do lỗi không được cache để lần sau có thể thử lại model đã sửa.

## Tọa độ, an toàn số liệu và giới hạn

- VietOCR trả bbox **dòng**, không được gán giả bbox cho từng từ.
- Mọi bbox lưu theo PDF không xoay; kiểm thử có các góc 0/90/180/270 và cropbox.
- `confidence` là điểm model, không phải xác suất số liệu tài chính chính xác.
- Không đưa bảng SLANet hay trang VietOCR thử nghiệm vào phép tính; không dùng LLM sửa số hoặc dấu.
- Hai OCR giống nhau cũng chưa chứng minh đúng. Phải kiểm tra PDF và đơn vị/kỳ.
- Bộ model là học máy, không phải một giải pháp hoàn toàn không có "hộp đen".
- Bảng thuyết minh nhiều cấp, nhiều trang, bảng nghiêng và vùng không có layout
  vẫn cần cải tiến. Nhánh này chưa thay hoàn toàn parser mẫu tài chính hiện tại.

## Kiểm thử

```powershell
& .\work\ocr-lab\.venv\Scripts\python.exe -m pytest -q
```

Unit tests không tải trọng số và không cần GPU. Benchmark model thật là kiểm tra riêng.

## Kết quả chạy thử ngày 02/11/2026

Đã chạy CPU trên máy i5-1240P, RAM 16 GB, Windows; môi trường lab dùng Python 3.11.
Báo cáo VNM thực tế có 55 trang, thử nghiệm này chỉ chấm **trang PDF thứ 8**.
Chọn thủ công 12 crop một dòng: 10 số (có số âm trong ngoặc) và 2 nhãn tiếng Việt.
Đã kiểm tra ảnh crop, mở rộng một bbox ban đầu cắt sát chữ số đầu, rồi chạy lại cả hai
engine trên cùng bộ crop hoàn chỉnh. Đây là mẫu nhỏ, không đại diện cho toàn báo cáo.

| Chỉ số trên bộ crop này | Tesseract | VietOCR |
|---|---:|---:|
| Chuỗi đúng hoàn toàn | 10/12 | 12/12 |
| Chuỗi số đúng hoàn toàn | 10/10 | 10/10 |
| CER (lỗi ký tự / số ký tự chuẩn) | 2,59% | 0% |
| OCR toàn trang, gồm nạp model | 4,77 giây | 60,71 giây |

Thời gian OCR toàn trang là một lượt chạy CPU, khác phép chấm crop; không phải
đo trung bình nhiều lần, cũng không tách thời gian khởi tạo. Hai nhãn tiếng Việt là
phần VietOCR tốt hơn trên mẫu này. OCR toàn trang vẫn có bất đồng số và mã dòng;
không được suy từ 10/10 crop số sang kết luận mọi số toàn trang đều đúng.

SLANet_plus dự đoán 140 ô trên vùng bảng khoanh thủ công. Ghép được chữ vào 40 ô
với Tesseract và 26 ô với VietOCR; nhiều bbox giao nhau hoặc cắt qua dòng chữ.
Các vùng mơ hồ bị giữ ngoài bảng. Những con số này là số ô có chữ, **không phải
độ chính xác cấu trúc**. Chưa có nhãn hàng/cột để chấm; chưa đủ tốt để thay parser
tài chính hoặc tạo dữ kiện cho phép tính. Cần đánh giá thêm trên nhiều bảng và
kiểm tra tiêu đề kỳ, đơn vị, dòng tổng trước khi bật cho tính toán.

Kết quả kỹ thuật: toàn bộ 162 tests đạt; 30 tests benchmark/hồi quy chạy lại sau khi
thêm manifest cũng đạt. `pip check` không phát hiện xung đột. Có một cảnh báo
deprecation của FastAPI/Starlette test client, không làm test thất bại.

Tệp kết quả local (không commit PDF, trọng số hoặc kho lab):

- `work/ocr-lab/benchmark-page8-verified/report.md` và `report.json`.
- `report.json` chứa hash PDF/model, phiên bản thư viện, kết quả từng crop và từng ô.
- `work/ocr-lab/reference-page8.json`: nhãn nhỏ đã kiểm tra thủ công.
- `work/ocr-lab/runtime-freeze.txt`: phiên bản đầy đủ của môi trường vừa chạy.
- `work/ocr-lab/api-smoke.json`: kiểm tra API thực tế trên bản PDF con chứa trang 8.

Đã chạy server lab tại cổng 8001 và nạp bản PDF con một trang qua `/ingest`: nhận
được 4 chunks, 1 bảng cấu trúc và 0 facts (đúng giới hạn thử nghiệm). Hai câu tìm
đoạn liên quan đều trả `evidence_only`, có nguồn PDF; tải nguồn qua API được đúng
các byte PDF đã nạp. Trong bản PDF con này, trang nguồn mang số 1; đó là trang 8
của báo cáo gốc, không phải đã kiểm tra OCR toàn bộ 55 trang.

Quyết định hiện tại: giữ Tesseract mặc định; dùng VietOCR để thử đọc/truy xuất tiếng
Việt, SLANet để kiểm tra cấu trúc. Chưa bật hai model mới cho tính toán tự động.

Nguồn triển khai: [VietOCR](https://github.com/pbcquoc/vietocr),
[RapidOCR](https://github.com/RapidAI/RapidOCR),
[RapidTable](https://github.com/RapidAI/RapidTable),
[SLANet_plus](https://www.paddleocr.ai/main/en/version3.x/module_usage/table_structure_recognition.html).
