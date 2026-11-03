# Hỏi đáp theo luật, có nguồn — không dùng LLM

Từ phiên bản cập nhật ngày 20-10-2026, `FinancialRAGService.ask` không gọi Ollama/OpenAI.
Tham số `use_llm` cũ được tiếp nhận để tương thích, nhưng bị bỏ qua kèm cảnh báo.
Không có biến môi trường nào tự bật lại mô hình sinh trong luồng này.
Tên repository được giữ nguyên; về kỹ thuật đây là truy xuất + hỏi đáp theo luật,
không còn bước RAG sinh câu trả lời bằng LLM.

## Phạm vi trả lời

Chỉ tiêu bảng chuẩn hiện hỗ trợ: doanh thu thuần/bán hàng/hoạt động tài chính,
lợi nhuận gộp/trước thuế/sau thuế, giá vốn, tổng tài sản, vốn chủ sở hữu,
nợ phải trả, hàng tồn kho, dòng tiền kinh doanh/đầu tư/tài chính, tiền đầu/cuối kỳ.
Các chỉ tiêu phải thực sự được trích ra từ báo cáo, không phải luôn có trong mọi PDF.

Mẫu câu (thay kỳ theo tài liệu của bạn):

- `Doanh thu thuần quý 1 năm 2026 là bao nhiêu?`
- `LNST quý 1/2026 là bao nhiêu?`
- `Tổng tài sản ngày 31/3/2026 là bao nhiêu?`
- `Doanh thu thuần quý 1/2026 tăng bao nhiêu phần trăm so với cùng kỳ 2025?`
- `Chênh lệch doanh thu thuần quý 1/2026 so với 2025?`
- `Biên lợi nhuận gộp quý 1/2026 là bao nhiêu?`

Bộ luật bảo thủ: hỏi một chỉ tiêu hoặc một phép tính được hỗ trợ. Từ ngữ/điều kiện
ngoài bộ luật không bị bỏ qua để trả lời một phần câu hỏi. Biên lợi nhuận hiện chỉ gồm
gộp và sau thuế; chưa tính thay đổi biên giữa hai kỳ. So sánh lấy kỳ mới trừ/chia kỳ cũ.

Chưa tự trả lời danh sách HĐQT, nguyên nhân, kế hoạch, dự báo, nhiều chỉ tiêu cùng lúc
hoặc câu hỏi tổng hợp tự do. Những câu này có thể nhận **đoạn liên quan**, không phải
câu trả lời đã được xác minh. Không kết luận “tài liệu không có” chỉ vì tìm kiếm không thấy.

## Các trạng thái API

| `answer_status` | Ý nghĩa |
|---|---|
| `answered` | Đã điền mẫu từ fact hoặc tính theo công thức; vẫn phải kiểm chứng OCR |
| `evidence_only` | Chỉ truy xuất các chunk liên quan, chưa trả lời câu hỏi |
| `needs_clarification` | Cần chọn báo cáo, làm rõ kỳ hoặc đơn giản hóa yêu cầu |
| `insufficient_evidence` | Không có nguồn phù hợp hoặc dữ kiện không vượt kiểm tra |

`reason_code` chỉ ra nhánh xử lý; `steps` liệt kê các thao tác kiểm chứng, không phải
chuỗi suy nghĩ của mô hình. `mode` giữ các giá trị structured/calculation/evidence.
Giá trị llm và các trường llm_model/llm_elapsed_seconds vẫn có trong schema cũ để
tương thích nhưng không được luồng hỏi đáp này phát sinh.

## Truy vết nguồn

Câu trả lời số liệu có `[1]`; phép tính có `[1] [2]`, công thức và cả hai đầu vào.
Mỗi nguồn giữ `document_id`, `chunk_id`, `page`, `source_url`, `graph_facts`, phương thức
trích xuất và `bbox` nếu có. Mở tài liệu gốc hoặc bấm “Xem nguồn” để đối chiếu.
Trang tính từ 1 của PDF, không nhất thiết giống số trang in.

Vùng màu cam chỉ hiện khi có tọa độ thật. Chunk văn bản có thể chưa có tọa độ:
giao diện ghi rõ cần so đoạn trích với toàn trang. Nội dung đoạn trích có thể đã được
chuẩn hóa khoảng trắng hoặc tái cấu trúc từ dòng bảng; PDF gốc là nơi đối chiếu.

## Các chốt kiểm tra

- Nhiều báo cáo: phải chọn một; một báo cáo: có thể tự chọn.
- Không lấy số quý trả cho câu hỏi cả năm; yêu cầu ghi rõ quý.
- Không bỏ qua kỳ/ngày/phạm vi không khớp hoặc chọn tùy ý fact trùng.
- Không tính khi thiếu đơn vị/kỳ/phạm vi, số không hữu hạn hoặc cần kiểm tra kế toán.
- Nguồn chunk phải khớp báo cáo, trang và đường dẫn của fact.
- Giá trị fact phải khớp chuỗi số `raw_value` và chuỗi đó phải hiện diện trong chunk nguồn.
- Không chia cho 0 hoặc diễn giải tăng trưởng trên nền âm.
- Không thực thi chỉ dẫn nằm trong tài liệu; không đưa nội dung tài liệu cho LLM.

Không thể cam kết chính xác mọi câu hỏi. Bộ luật, OCR và bố cục bảng vẫn có giới hạn.
Kiểm tra kế toán khớp không thay thế xác minh thủ công. TF-IDF/điểm graph không phải
xác suất đúng. Dữ liệu regex từ TXT và bảng không chuẩn hiện chỉ làm nguồn truy xuất.

## Chạy và kiểm thử

Khởi động lại web sau khi cập nhật code, rồi tải lại trang:

```powershell
cd C:\DXD\Multimodal-RAG-Financial
powershell -ExecutionPolicy Bypass -File .\start-local.ps1
```

Không cần OCR lại chỉ để dùng chế độ trả lời mới; giữ nguyên PDF/index hiện tại.
Nếu thay đổi cách trích xuất hoặc nguồn lỗi thì mới cần nạp lại.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m financial_rag.cli ask "LNST quý 1/2026" --document-id ID_BAO_CAO
```

`tests/test_deterministic_answers.py` kiểm tra kỳ, điều kiện câu hỏi, dữ kiện thiếu,
nguồn và trạng thái API. Các test provider xác nhận cấu hình/`use_llm=true` không gọi model.
`scripts/evaluate_vnm.py` là hồi quy riêng cho đúng PDF VNM Q1/2026, không phải benchmark
độ chính xác tổng quát. OCR vẫn là thành phần học máy; “không LLM” không có nghĩa
loại bỏ toàn bộ học máy khỏi dự án.
