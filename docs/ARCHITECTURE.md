# Kiến trúc Multimodal RAG Financial

Mô tả triển khai hiện tại, đối chiếu mã nguồn ngày 11-10-2026.
Hướng dẫn cài/chạy nằm trong [README](../README.md).

## 1. Các thành phần

| Thành phần | Triển khai | Trách nhiệm |
|---|---|---|
| Web | FastAPI, HTML/JavaScript thuần | Upload, hỏi đáp, dashboard, nguồn PDF, graph explorer |
| Đọc tài liệu | PyMuPDF, BeautifulSoup | Text/trang, bảng PDF text, HTML/TXT/MD |
| OCR | PyMuPDF/Tesseract, tessdata vie+eng | Đọc trang scan, tọa độ, cache |
| Bảng tài chính | Bộ phân tích bố cục/quy tắc | Mã chỉ tiêu, hai cột kỳ, số VND, phạm vi |
| Graph | NetworkX MultiDiGraph, JSON | Các thực thể và quan hệ nguồn |
| Retrieval | scikit-learn TF-IDF + graph | Xếp hạng từ khóa và liên kết chỉ tiêu, lọc báo cáo |
| Số học | Calculator dùng Decimal | Tăng trưởng, chênh lệch, tỷ số có kiểm tra đầu vào |
| Diễn giải | Ollama local mặc định | Sinh câu trả lời từ bằng chứng đã truy xuất |
| Vận hành | ThreadPoolExecutor, JobStore, RLock | Một tác vụ ingest chạy mỗi lần, trạng thái bền vững, khóa trong tiến trình |

Không có React/npm, Neo4j, vector database, GNN hoặc mô hình hiểu biểu đồ.
“Hybrid” ở đây là TF-IDF kết hợp quan hệ graph, không phải sparse/dense embeddings.

## 2. Luồng nạp tài liệu

1. API nhận file tối đa 50 MiB, ticker/năm tùy chọn; `background=true` bật nạp nền.
2. Service tính SHA-256, lưu bản sao vào `data/uploads/<hash>.<đuôi>`, không sửa file gốc.
3. `DocumentLoader` đọc nội dung theo trang. Trang scan được OCR nếu bật cấu hình và có tessdata.
   Kết quả giữ text, từ/tọa độ, kích thước trang, cảnh báo và phương thức trích xuất.
4. Bộ bảng chuẩn tìm mã dòng, hai cột kỳ và số VND. OCR bổ sung có thể phục hồi mã dòng
   hoặc ô sai định dạng; giữ dấu vết, không lấy công thức để tự điền số thiếu.
5. `Chunker` tạo đoạn theo trang/overlap và đoạn bảng. `FinancialFactExtractor` tạo facts
   từ bảng có cấu trúc hoặc baseline nhận diện văn bản.
6. `check_facts` kiểm tra công thức theo báo cáo/phạm vi/kỳ/loại bảng; khi đủ dữ liệu,
   đối chiếu số dư tiền giữa bảng cân đối và lưu chuyển tiền tệ.
7. Ghi metadata, graph và chunks. Nạp lại cùng document ID thay Chunk/Fact của báo cáo đó,
   không xóa báo cáo khác. ID phụ thuộc nội dung, ticker/năm và phiên bản đọc;
   cùng PDF với metadata khác có thể tạo báo cáo riêng.

Chất lượng có thể thiếu dữ liệu hoặc cần xem lại. Qua được công thức chỉ có nghĩa
nhất quán với các phép kiểm tra đã triển khai, không chứng minh mọi ô OCR chính xác.
Bảng HTML inline khi đọc văn bản chưa được bảo đảm giữ cấu trúc như bảng PDF.

## 3. Graph G = (V, E)

Đồ thị có hướng, cho phép nhiều cạnh giữa hai node. Schema do `graph.py` tạo:

```text
Company ──HAS_REPORT──▶ Report ──FOR_PERIOD──▶ Period
                         │
                         ├──MENTIONS──▶ Chunk ──NEXT──▶ Chunk
                         │
                         └──HAS_FACT──▶ Fact ──SUPPORTED_BY──▶ Chunk
                                        │
                                        └──MEASURES──▶ Metric
```

| Node | Nội dung chính |
|---|---|
| Company | Ticker; chỉ tạo khi có ticker |
| Report | Nguồn, tên, ticker, năm, phạm vi |
| Period | Năm report được cung cấp; kỳ chi tiết nằm trên Fact |
| Chunk | Text/bảng, trang, document ID, tọa độ, phương thức trích xuất |
| Metric | Tên chỉ tiêu đã liên kết |
| Fact | Giá trị, đơn vị, kỳ/loại kỳ, phạm vi, mã chỉ tiêu, chất lượng và nguồn |

Không có node `Table` riêng: bảng nằm trong chunk. Graph chưa được suy luận/huấn luyện
bởi LLM/GNN; LLM hiện phục vụ bước diễn giải.
Fact giữ `raw_value`, `value`, `period`, `period_label`, `period_kind`, `scope`,
`metric_code`, `statement`, `document_id`, `chunk_id`, `page`, `bbox`, `quality`,
`recognition_note`. Không trích được có thể thiếu; không coi thiếu là 0.

Fact hiện dùng kiểu số trong schema; calculator chuyển sang Decimal để tính.
Điều này không cam kết giữ độ chính xác tùy ý cho mọi cỡ số/đơn vị đầu vào.

## 4. Luồng hỏi đáp

`FinancialRAGService.ask` điều phối:

1. API nhận câu hỏi 3–2.000 ký tự, `top_k` 1–20, `document_id` tùy chọn.
2. `factual_answer` thử câu hỏi số liệu/phép tính hỗ trợ; đủ facts thì trả số và nguồn
   trước khi gọi LLM. Kỳ/phạm vi được hỏi không có thì có thể trả thông báo thiếu.
3. `HybridRetriever` kết hợp TF-IDF và liên kết Metric/Fact/Chunk, bổ sung alias tiếng Việt,
   lọc báo cáo, giảm ưu tiên mục lục và các đoạn không phù hợp.
4. Không bật LLM: trả bằng chứng. Bật LLM: gửi câu hỏi và các đoạn truy xuất cho model,
   không gửi cả PDF như một mô hình thị giác.
5. Với Ollama, kiểm tra phản hồi đầy đủ, trích dẫn hợp lệ và số so với nguồn được dẫn.
   Không đạt hoặc lỗi/bận/timeout: trả bằng chứng gốc, lý do và thời gian gọi nếu có.

Mode trả về: `structured`, `calculation`, `evidence`, `llm`.
`use_llm=true` không có nghĩa mọi câu hỏi đều gọi model.
Nguồn trang là vị trí trang PDF tính từ 1, không nhất thiết là số in trên trang.

### Calculator và tổng quan

- `difference`: mới − cũ, cùng chỉ tiêu/kỳ tương thích.
- `growth`: (mới − cũ) / cũ × 100; từ chối nền âm hoặc mẫu số 0.
- `ratio`: trái / phải × 100; yêu cầu cùng kỳ.
- Kiểm tra báo cáo, doanh nghiệp, đơn vị, phạm vi, loại kỳ và chất lượng số liệu.
- `analytics.py` xây sáu thẻ tổng quan và biên lợi nhuận, giữ công thức/fact đầu vào;
  không nhờ LLM tính hoặc chọn âm thầm khi facts trùng/mâu thuẫn.

## 5. Ollama và độ tin cậy

Mặc định `qwen2.5:3b`, CPU (`num_gpu=0`), context 4.096, output tối đa 512 token,
temperature 0,1, keep-alive 2 phút. Các tham số nằm trong client, không phải tất cả
đều có biến `.env`. Chỉ nhận endpoint localhost, không chấp nhận model cloud.

Một lần sinh Ollama đồng thời trong tiến trình. `/health/ollama` kiểm tra model qua
`/api/tags`, không inference; “ready” không bảo đảm câu trả lời vượt kiểm tra.

Bộ kiểm tra trích dẫn/số **không xác minh ngữ nghĩa**. Câu có dẫn nguồn vẫn có thể hiểu sai;
làm tròn hợp lệ có thể bị từ chối. Với câu hỏi nguyên nhân, hiện xử lý bảo thủ: chỉ chấp nhận
dạng diễn giải từ chối kết luận, còn lại trả nguồn. Không phát hiện được mọi dạng bịa đặt.
Provider OpenAI tùy chọn có nhánh riêng, kiểm tra citation đơn giản hơn;
không có cùng bộ kiểm tra số liệu với Ollama và không tự được bật khi Ollama lỗi.

## 6. Kho dữ liệu và đồng thời

| Đường dẫn mặc định | Nội dung |
|---|---|
| `data/uploads/` | Bản sao tài liệu |
| `data/processed/ocr/` | OCR cache theo nội dung/cấu hình |
| `data/processed/reports/<id>.json` | Metadata, thống kê và kiểm tra chất lượng |
| `data/processed/jobs.json` | Lịch sử tác vụ nạp |
| `data/index/graph.json` | Graph NetworkX dạng JSON |
| `data/index/chunks.json` | Chunks phục vụ dựng lại chỉ mục TF-IDF |

Một worker ingest, tối đa 8 tác vụ chờ/chạy, giữ tối đa 200 bản ghi; `/jobs` trả 30 gần nhất.
Sau restart, jobs `queued`/`running` chuyển `interrupted`; người dùng nạp lại tài liệu,
không có cơ chế tự resume, OCR cache có thể được dùng lại.

`RLock` chỉ bảo vệ trong một tiến trình. Trong lúc ingest/OCR, API cần đọc kho có thể phải chờ;
API tiến độ jobs không cần khóa service. Không chạy nhiều worker hoặc CLI ingest cùng web
trên cùng `DATA_DIR`.

`atomic_json` ghi file tạm cùng thư mục, flush/fsync rồi replace cho graph, chunks, metadata,
jobs. Nguyên tử **từng file**, không có transaction toàn kho hoặc database production.
Nếu gián đoạn giữa các bước ghi, nạp lại báo cáo; sao lưu khi không có tiến trình đang ghi.

## 7. Ranh giới bảo mật

- Bind localhost, chưa có login, phân quyền hoặc triển khai Internet an toàn.
- Tài liệu là dữ liệu không đáng tin. Prompt yêu cầu không làm theo chỉ dẫn trong nguồn;
  đây không phải bảo đảm miễn nhiễm prompt injection.
- Giao diện dùng textContent ở các phần hiển thị dữ liệu; cần kiểm tra lại khi thêm UI mới.
- `/export` rút gọn `source` thành tên file, **không ẩn danh nội dung báo cáo**.
  API nội bộ khác có thể chứa đường dẫn/metadata local.
- Gitignore loại cấu hình bí mật, tài liệu, tessdata, index và kết quả thử nghiệm;
  không mã hóa dữ liệu hoặc bảo vệ endpoint đã mở ra mạng.

## 8. Kiểm thử và hướng phát triển

Test tự động bao phủ trích facts, kỳ/phạm vi, phép tính, truy xuất, fallback Ollama,
nguồn, dashboard, jobs, ghi nguyên tử và API. Hồi quy VNM có 18 tình huống trên một báo cáo
riêng Q1/2026 đã nạp; smoke Ollama là bài thử local tùy chọn, không phải benchmark tổng quát.

Các hướng **chưa triển khai**, cần đánh giá riêng:

1. Bảng nhiều bố cục/đơn vị và hiểu biểu đồ bằng VLM.
2. Structured extraction bằng LLM/VLM, đánh giá entity/relation độc lập.
3. Dense multilingual embeddings, reranker và đo Recall@k.
4. Phân tích nhiều báo cáo/nhiều kỳ với kiểm soát metadata.
5. Database/transaction, authentication, phân quyền và hàng đợi production.
6. Benchmark OCR, execution accuracy, answer/citation accuracy trên corpus đa dạng.

