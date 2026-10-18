# Phân tích thiết kế: hệ thống RAG hỏi đáp báo cáo tài chính

Tài liệu tổng hợp bước 1 của dự án: xác định bài toán, thiết kế kiến trúc, chuẩn bị bộ tài liệu context. Ngày soạn: 18/10/2026.

---

## 1. Xác định bài toán

### 1.1 Mục tiêu
Xây dựng hệ thống RAG đọc báo cáo tài chính (tiếng Việt và tiếng Anh) và trả lời câu hỏi bằng ngôn ngữ của người hỏi, kèm trích dẫn nguồn (báo cáo, trang, bảng).

### 1.2 Người dùng và bối cảnh
- Nhóm người dùng: sinh viên, dùng để demo cho nhau xem.
- Không cần đăng nhập, không cần chịu tải lớn.

### 1.3 Ràng buộc đã chốt
| Ràng buộc | Nội dung |
|---|---|
| LLM | Qwen2.5-3B-Instruct |
| Phần cứng | Google Colab, GPU T4 (16 GB VRAM) |
| Ngôn ngữ tài liệu | Tiếng Việt và tiếng Anh |
| Dữ liệu hiện có | 8 file báo cáo của các công ty khác nhau, thuộc các ngành khác nhau, kỳ báo cáo là 6 tháng đầu năm hoặc cả năm |

### 1.4 Phạm vi
- 2-3 công ty cho bản demo (8 file hiện có là mức tối thiểu để bắt đầu), gồm cả báo cáo tiếng Việt và tiếng Anh.
- Ngoài phạm vi: khuyến nghị mua/bán, dự báo giá, đăng nhập, chịu tải lớn.

### 1.5 Các loại câu hỏi hệ thống phải trả lời
1. Tra cứu số liệu: "Doanh thu thuần 6 tháng đầu năm của công ty X là bao nhiêu?"
2. Tra cứu văn bản: "Công ty giải thích thế nào về việc lợi nhuận giảm?"
3. Tính toán trong một báo cáo: biên lợi nhuận, nợ/vốn chủ sở hữu, tăng trưởng so với cùng kỳ (dựa vào cột số liệu có sẵn).
4. So sánh giữa các công ty: chỉ với chỉ tiêu chung và chỉ khi cùng ngành hoặc cùng kỳ.
5. Câu hỏi bẫy: thông tin không có trong tài liệu, hệ thống phải từ chối.

Lưu ý: nếu mỗi công ty chỉ có một báo cáo thì không hỏi được xu hướng qua nhiều kỳ. Muốn demo loại này, tải thêm 2-3 kỳ cho một công ty.

### 1.6 Tiêu chí thành công
- Số liệu đúng (sai một con số là lỗi nghiêm trọng).
- Mọi câu trả lời có trích dẫn nguồn.
- Biết từ chối khi không có thông tin, không bịa.

---

## 2. Hệ quả của việc dùng Qwen2.5-3B

Model 3B yếu về tính toán, ngữ cảnh dài và tiếng Việt chuyên ngành, nên để code và retrieval làm phần nặng; LLM chỉ đọc vài đoạn đã chọn sẵn rồi diễn đạt lại.

| Điểm yếu của 3B | Cách thiết kế để bù |
|---|---|
| Tính toán sai | LLM không tự tính: trích số liệu, code Python tính, LLM diễn đạt kết quả |
| Dễ lạc khi ngữ cảnh dài | Chỉ đưa top 3-5 chunk, khoảng 2-3K token vào prompt |
| Dễ bịa khi thiếu thông tin | Bắt buộc trích dẫn, ngưỡng điểm retrieval để từ chối, `do_sample=False` (tương đương temperature 0) |
| Text-to-SQL và suy luận nhiều bước kém | Chỉ trích slot dạng JSON (công ty, kỳ, chỉ tiêu); code tra cứu |
| Tiếng Việt yếu hơn tiếng Anh | System prompt bằng tiếng Anh, yêu cầu trả lời theo ngôn ngữ của câu hỏi |

Gợi ý thêm: dùng một model lớn hơn (7B hoặc API) làm mốc tham chiếu khi đánh giá, để biết lỗi nằm ở retrieval hay ở generation.

---

## 3. Kiến trúc

### 3.1 Luồng nạp dữ liệu (offline, chạy một lần)

```
PDF (Việt/Anh) -> Parse + OCR (kiểm tra lỗi font)
-> Chunk theo mục/bảng + metadata, chuẩn hóa số và đơn vị
-> (a) Index hybrid: BGE-M3 (dense) + FTS5/BM25
-> (b) Bảng chỉ tiêu có cấu trúc (SQLite)
```

### 3.2 Luồng trả lời (online)

```
Câu hỏi
-> [3B] trích slot JSON (công ty, kỳ, chỉ tiêu)
-> mở rộng truy vấn bằng glossary Việt-Anh
-> phân loại câu hỏi:
     (a) hỏi số liệu    -> tra bảng chỉ tiêu -> code tính toán -> [3B] diễn đạt
     (b) hỏi giải thích -> tìm hybrid có lọc metadata -> (rerank, tùy chọn) -> top 3-5 chunk
                           -> [3B] trả lời kèm trích dẫn
-> kiểm tra: số trong câu trả lời có khớp nguồn không? Không khớp -> từ chối hoặc cảnh báo
```

### 3.3 Quyết định thiết kế chính
- **Parse PDF là phần khó nhất:** dùng parser giữ cấu trúc bảng (Docling, Unstructured hoặc tương đương); so sánh hai parser trên 2-3 file mẫu trước khi chọn. Bản scan cần OCR có hỗ trợ tiếng Việt.
- **Chunk theo cấu trúc**, không cắt theo số ký tự: mỗi chunk khoảng 300-600 token; bảng lớn chia theo nhóm dòng và lặp lại header. Mỗi chunk có dòng tiền tố ngữ cảnh, ví dụ `[CTY_A | H1/2026 | Hợp nhất | Bảng KQKD | Đơn vị: triệu đồng]`.
- **Bảng chỉ tiêu có cấu trúc ngay từ bản đầu:** tra bằng code đáng tin hơn nhiều so với nhờ 3B đọc số trong bảng markdown.
- **Hybrid search + lọc metadata** (mã công ty, kỳ, phạm vi báo cáo) để không lẫn số liệu giữa các công ty và các kỳ.
- **Reranker** (bge-reranker-v2-m3): làm sau, thêm khi đo thấy retrieval chưa tốt.

---

## 4. Xử lý song ngữ Việt-Anh và chuẩn hóa số liệu

- **Embedding đa ngôn ngữ:** BGE-M3 (hỗ trợ Việt, Anh, tìm chéo ngôn ngữ). Có thể so sánh thêm với multilingual-e5.
- **Glossary Việt-Anh:** ví dụ Doanh thu thuần ↔ Net revenue; lưu trong cột `aliases` của bảng `metric`; dùng mở rộng truy vấn và chuẩn hóa tên chỉ tiêu.
- **Ngôn ngữ trả lời:** theo câu hỏi; phần trích dẫn giữ ngôn ngữ gốc của tài liệu.

| Vấn đề | Cách xử lý |
|---|---|
| Định dạng số | Số Việt `1.234.567,89`, số Anh `1,234,567.89`; chuẩn hóa về một dạng khi nạp |
| Số âm | Thường nằm trong ngoặc `(1.234)`; đổi thành số âm |
| Đơn vị | Đồng, nghìn đồng, triệu đồng, tỷ đồng, nghìn USD; lưu `unit_scale` ở mức từng báo cáo |
| Kỳ báo cáo | Năm tài chính có thể không trùng năm dương lịch; báo cáo 6 tháng là số luỹ kế; ghi rõ `period_type` |
| Lỗi font PDF tiếng Việt | Text copy ra vỡ dấu dù nhìn bình thường; kiểm tra từng file, OCR nếu cần |
| Tiền tệ | Lưu `currency`; so sánh tỷ lệ giữa VND và USD thì ổn, so số tuyệt đối thì nói rõ khác tiền tệ, không tự quy đổi |

---

## 5. Thiết kế dữ liệu

Dùng một file SQLite (kèm FAISS cho vector) lưu trên Google Drive. Không cần database lớn.

### 5.1 Ba nhóm dữ liệu
| Nhóm | Lưu gì | Phục vụ |
|---|---|---|
| Danh mục | công ty, báo cáo, từ điển chỉ tiêu | lọc trước khi tìm |
| Số liệu có cấu trúc (`fact`) | từng con số đã chuẩn hóa, kèm trang nguồn | câu hỏi số liệu và tính toán |
| Văn bản (`chunk`) | chunk text/bảng, FTS5, vector FAISS | câu hỏi giải thích |

### 5.2 Schema (SQLite)

```sql
CREATE TABLE company (
  company_id INTEGER PRIMARY KEY, ticker TEXT, name_vi TEXT, name_en TEXT,
  industry TEXT, country TEXT
);

CREATE TABLE report (
  report_id INTEGER PRIMARY KEY,
  company_id INTEGER REFERENCES company(company_id),
  doc_type TEXT,          -- BCTC | annual_report
  period_type TEXT,       -- H1 | 9M | FY | Q1...
  fiscal_year INTEGER, period_end DATE,
  scope TEXT,             -- consolidated | separate
  language TEXT,          -- vi | en
  currency TEXT, unit_scale INTEGER,
  audit_status TEXT,      -- reviewed | audited | unaudited
  file_path TEXT, source_url TEXT, is_scanned BOOLEAN
);

CREATE TABLE metric (     -- từ điển chỉ tiêu + glossary Việt-Anh
  metric_id INTEGER PRIMARY KEY, code TEXT UNIQUE,
  name_vi TEXT, name_en TEXT,
  statement TEXT,         -- IS | BS | CF
  is_common BOOLEAN,      -- chỉ tiêu chung mọi ngành (10-15 cái)
  aliases TEXT
);

CREATE TABLE fact (
  fact_id INTEGER PRIMARY KEY,
  report_id INTEGER REFERENCES report(report_id),
  metric_id INTEGER REFERENCES metric(metric_id),  -- NULL nếu chỉ tiêu đặc thù ngành
  line_item TEXT,         -- tên dòng gốc trong báo cáo
  column_role TEXT,       -- current | prior_year_same_period | year_start
  period_start DATE, period_end DATE,
  value REAL,             -- đã nhân unit_scale
  value_raw TEXT,         -- chuỗi gốc, ví dụ "(1.234.567)"
  page INTEGER, table_ref TEXT,
  UNIQUE (report_id, line_item, column_role)
);

CREATE TABLE chunk (
  chunk_id INTEGER PRIMARY KEY,
  report_id INTEGER REFERENCES report(report_id),
  section TEXT, chunk_type TEXT,   -- text | table
  page_start INTEGER, page_end INTEGER,
  context_prefix TEXT, content TEXT
);
CREATE VIRTUAL TABLE chunk_fts USING fts5(content, content='chunk', content_rowid='chunk_id');
-- Vector: FAISS index lưu riêng, khóa là chunk_id

CREATE TABLE eval_question (
  q_id INTEGER PRIMARY KEY, question TEXT, language TEXT,
  difficulty TEXT,        -- lookup | text | calc | compare | trap
  gold_answer TEXT, gold_report_id INTEGER, gold_page INTEGER
);
```

### 5.3 Quyết định thiết kế dữ liệu
- `fact` dạng dài (mỗi con số một dòng), vì mỗi ngành có dòng báo cáo khác nhau; dòng chưa map vào `metric` vẫn lưu `line_item`.
- Chỉ chuẩn hóa 10-15 chỉ tiêu chung vào `metric` (doanh thu thuần, lợi nhuận sau thuế, tổng tài sản, nợ phải trả, vốn chủ sở hữu, tiền...); chỉ tiêu đặc thù ngành tìm qua `chunk`.
- Lưu cả `value` và `value_raw` để đối chiếu khi nghi sai.
- `column_role` lưu đủ cột kỳ này, cùng kỳ năm trước, đầu năm, để trả lời câu so sánh mà không cần báo cáo kỳ trước.
- `page` và `table_ref` ở mọi `fact` và `chunk` làm nguồn trích dẫn.
- Không lưu tỷ số (ROE, biên lợi nhuận...); code tính từ `fact` khi hỏi.
- Dùng FTS5 của SQLite thay thư viện BM25 riêng.

### 5.4 Kiểm tra chất lượng khi nạp
- Tổng tài sản có bằng nợ phải trả cộng vốn chủ sở hữu không.
- Các dòng con có cộng ra dòng tổng không.
- Mỗi báo cáo có đủ các chỉ tiêu chung chưa.
- Báo cáo không qua kiểm tra: đánh dấu `needs_manual_check`, xem tay, không cho vào demo.

### 5.5 Ví dụ truy vấn
Câu "Lợi nhuận sau thuế 6 tháng đầu năm của công ty X so với cùng kỳ?": 3B trích slot, code chạy truy vấn, tính phần trăm thay đổi, 3B diễn đạt kèm trang nguồn.

```sql
SELECT column_role, value, page FROM fact f
JOIN metric m USING (metric_id) JOIN report r USING (report_id)
JOIN company c USING (company_id)
WHERE c.ticker = 'X' AND r.period_type = 'H1' AND r.fiscal_year = 2026
  AND m.code = 'net_profit'
  AND f.column_role IN ('current', 'prior_year_same_period');
```

### 5.6 Đồ thị tri thức: không dùng ở bản đầu
- Dữ liệu là các công ty khác nhau, ít quan hệ chéo; các quan hệ cơ bản đã nằm trong database dưới dạng khóa ngoại.
- 3B không đủ tin cậy để trích thực thể và quan hệ; GraphRAG đầy đủ quá nặng cho T4.
- Phần hữu ích đã được bao phủ: `aliases` trong `metric` thay cạnh ALIAS_OF; công thức tỷ số viết trong code.
- Chỉ xét lại nếu đo trên bộ eval thấy có nhóm câu hỏi nhiều bước (công ty con, bên liên quan) mà baseline làm sai, hoặc nếu giảng viên yêu cầu. Khi đó làm bản nhẹ bằng hai bảng `node` và `edge` trong SQLite.
- Có thể ghi vào mục hướng phát triển của báo cáo đồ án.

---

## 6. Bộ tài liệu context

### 6.1 Đánh giá 8 file đã tải
- Đủ để bắt đầu xây và thử pipeline.
- Làm được tốt: tra số liệu, tra văn bản, tính tỷ số trong một kỳ, so sánh giữa các công ty (chỉ tiêu chung), so sánh với cùng kỳ năm trước nhờ cột số liệu có sẵn.
- Hạn chế: không hỏi được xu hướng nhiều kỳ; không so sánh được tỷ số đặc thù ngành giữa các ngành (ngân hàng khác hẳn công ty sản xuất); báo cáo 6 tháng là số luỹ kế, không so trực tiếp tỷ số tính trên lợi nhuận với báo cáo cả năm.
- Ngành khác nhau có lợi cho việc kiểm tra độ phủ của parser và đa dạng văn bản giải thích; nên chọn 2-3 công ty cùng ngành làm cụm so sánh chính.
- Cần kiểm tra từng file: dạng text hay scan, có lỗi font tiếng Việt không, loại báo cáo (hợp nhất hay riêng lẻ), có phần văn bản giải thích hay chỉ có bảng.

### 6.2 Manifest và cấu trúc thư mục
Chi tiết ở hai file đã soạn: `manifest.csv` (mẫu) và `data_structure.md` (cấu trúc thư mục, quy ước đặt tên, từ điển cột). Tóm tắt:

```
rag-financial/
├── data/
│   ├── manifest.csv
│   ├── raw/{TICKER}/{TICKER}_{NĂM}_{KỲ}_{PHẠM-VI}_{NGÔN-NGỮ}.pdf
│   ├── parsed/        # document.md, tables.json, parse_log.txt cho mỗi PDF
│   ├── structured/    # financial.db, metrics_seed.csv
│   ├── index/         # faiss.index, chunk_ids.json
│   └── eval/          # questions.csv
├── notebooks/
└── src/
```

---

## 7. Bộ đánh giá

- 30-40 câu hỏi soạn từ chính các file đã tải, kèm đáp án chuẩn, trang nguồn, đơn vị.
- Chia theo hai trục: **ngôn ngữ** (Việt-Việt, Anh-Anh, Việt-Anh, Anh-Việt) và **độ khó** (tra số liệu, tra văn bản, tính toán, so sánh giữa các công ty, so sánh với kỳ trước dựa vào cột sẵn có).
- Thêm khoảng 5-10 câu bẫy (đáp án đúng là không có thông tin), gồm cả vài câu "không so sánh trực tiếp được vì khác ngành".
- Chỉ số theo dõi: recall của retrieval (có lấy đúng đoạn/bảng không), độ chính xác số liệu, tỷ lệ có trích dẫn đúng, tỷ lệ từ chối đúng với câu bẫy.
- Mọi thay đổi (parser, chunk size, embedding, thêm reranker) đều chấm lại trên bộ này.

---

## 8. Môi trường chạy: Google Colab T4

| Thành phần | VRAM (ước tính) | Ghi chú |
|---|---|---|
| Qwen2.5-3B-Instruct (fp16) | khoảng 6-7 GB | T4 không hỗ trợ bf16 |
| BGE-M3 | khoảng 1-1,5 GB | dense và sparse |
| bge-reranker-v2-m3 | khoảng 1-1,5 GB | tùy chọn, làm chậm mỗi câu hỏi |
| Tổng | khoảng 9-10 GB | còn dư cho ngữ cảnh 2-3K token |

- **Chia offline và online:** parse, OCR, chunk, embedding chạy một lần; lưu kết quả vào Google Drive; buổi demo chỉ nạp lại.
- **Session có thể bị ngắt** khi idle và đĩa tạm bị xóa: cell khởi động tự nạp lại từ Drive. GPU T4 bản miễn phí không phải lúc nào cũng cấp được, nên thử trước giờ demo.
- **Chạy model:** `transformers` với fp16 là đủ; vLLM có thể nhanh hơn nhưng cần thử trước trên T4.
- **Giao diện:** Gradio với `share=True` cho link công khai tạm thời (hết hạn, chỉ chạy khi session còn sống).
- **Cấu hình bản đầu:** PDF đã parse sẵn -> BGE-M3 + FTS5 (FAISS) -> bảng chỉ tiêu SQLite -> Qwen2.5-3B-Instruct fp16 -> Gradio; chưa thêm reranker.
- **Chuẩn bị demo:** soạn sẵn 8-10 câu hỏi mẫu đã chạy ổn, đủ các loại câu hỏi và 1-2 câu bẫy; thử trước các câu hỏi lạ (mơ hồ, công ty không có trong dữ liệu, yêu cầu dự báo).

---

## 9. Mức độ sẵn sàng và việc cần làm tiếp

### 9.1 Hiện trạng
| Hạng mục | Trạng thái |
|---|---|
| Xác định bài toán | Đã chốt phần lớn; các ý nằm rải rác, nên gom thành problem statement 1 trang (mục 1 của tài liệu này đã làm việc đó) |
| Thiết kế kiến trúc | Xong mức tổng quan |
| Thiết kế dữ liệu | Xong schema mức khái quát |
| Bộ tài liệu | Mới có nguyên liệu: đã tải 8 file, có manifest mẫu, chưa kiểm tra từng file |
| Bộ đánh giá | Chưa soạn |

### 9.2 Việc cần làm theo thứ tự ưu tiên
1. Kiểm tra 8 file (text hay scan, đơn vị, kỳ, hợp nhất hay riêng lẻ) và điền `manifest.csv`.
2. Thử parse 2-3 file bằng hai parser, so sánh chất lượng bảng và tiếng Việt. Đây là rủi ro lớn nhất, kiểm chứng sớm.
3. Soạn 30-40 câu hỏi đánh giá từ 8 file; loại câu nào không viết nổi cho biết cần tải thêm dữ liệu.
4. Chốt danh sách 10-15 chỉ tiêu chung cho bảng `metric` (kèm alias Việt-Anh).
5. Viết thiết kế chi tiết ngắn: parser chọn, prompt mẫu cho bước trích slot và bước trả lời, ngưỡng nghiệm thu (ví dụ tỷ lệ số liệu đúng tối thiểu).
6. Xây baseline và đo trên bộ eval, sau đó mới thêm reranker hoặc phần mở rộng.
