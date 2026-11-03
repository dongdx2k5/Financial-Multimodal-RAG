# TF-IDF từ số 0: hiểu nguyên lý, tự tính được và đọc được code dự án

**Dự án:** Multimodal RAG Financial

**Đối chiếu mã nguồn:** 20–21/10/2026

**Đối tượng:** người chưa biết học máy, đại số tuyến tính hoặc lập trình Python

**Mục tiêu:** không chỉ nhớ “TF nhân IDF”, mà hiểu dữ liệu đi đâu, từng dòng code làm gì, kết quả có ý nghĩa gì và không có ý nghĩa gì.

## Đọc trước: tài liệu này giải thích bản nào?

Phần chính giải thích mã nguồn local đang dùng tại:

`C:\DXD\Multimodal-RAG-Financial`

File trung tâm là [retrieval.py](C:/DXD/Multimodal-RAG-Financial/src/financial_rag/retrieval.py).
Bản local có các thay đổi chưa commit, nên không được coi nội dung ở đây là mô tả chính xác mọi nhánh trên GitHub.

Nhánh `Thien` của bạn cùng nhóm, được kiểm tra riêng tại commit `08305cb`, có kiến trúc khác: BGE-M3 + FTS5, và TF-IDF là phương án dự phòng. Phần 28 phân biệt hai bản. Không trộn hai cách triển khai khi thuyết trình.

Tài liệu và chương trình minh họa đi kèm **không thay đổi thuật toán truy xuất, dữ liệu báo cáo hoặc index đang sử dụng**.

### Ba điều phải nhớ ngay từ đầu

1. **TF-IDF tìm đoạn văn có từ/cụm từ liên quan; nó không tự hiểu và tính toán tài chính.**
2. **Trong code hiện tại, `N` là số chunk trong toàn bộ tập được đưa vào `fit_transform`, không phải số PDF, số trang, số từ, hay `top_k`.**
3. **Điểm truy xuất cao không phải xác suất câu trả lời đúng.**

## Lộ trình học

| Lượt đọc | Nội dung | Sau lượt này bạn nên làm được |
| --- | --- | --- |
| Lượt 1 | Phần 1–5 | Biết TF-IDF nằm ở đâu, chunk và từ vựng là gì |
| Lượt 2 | Phần 6–12 | Tính TF, DF, IDF, chuẩn hóa và điểm tương đồng bằng tay |
| Lượt 3 | Phần 13–21 | Theo dõi từng hàm và hiểu điểm cuối cùng trong dự án |
| Lượt 4 | Phần 22–28 | Chạy demo, đọc kết quả, hiểu giới hạn và phân biệt các phiên bản |
| Ôn tập | Phần 29–33 | Làm bài tập, trả lời vấn đáp và giải thích lại bằng lời của mình |

Không cần học thuộc toàn bộ tài liệu một lần. Nếu thấy công thức khó, hãy đọc ví dụ số ngay sau công thức rồi quay lại.

### Mục lục nhanh

- [Bắt đầu từ bài toán thật](#bat-dau) — phần 1–5.
- [TF, DF, N, IDF và chuẩn hóa](#cong-thuc) — phần 6–11.
- [Ví dụ tính tay và cấu hình thật](#tinh-tay) — phần 12–13.
- [Giải thích Python và các hàm khởi tạo](#doc-code) — phần 14–17.
- [Theo dõi toàn bộ search: query, graph, lọc, trích dẫn](#luong-search) — phần 18–21.
- [Chạy chương trình thực hành](#thuc-hanh) — phần 22–23.
- [Debug, tốc độ, giới hạn và đánh giá](#kiem-tra) — phần 24–27.
- [Phân biệt với nhánh Thien](#nhanh-thien) — phần 28.
- [25 câu hỏi thường gặp](#hoi-dap) — phần 29.
- [9 bài tập có đáp án](#bai-tap) — phần 30.
- [Tờ ghi nhớ và phần trình bày vấn đáp](#van-dap) — phần 32.
- [Bản đồ file nguồn và kiểm chứng](#nguon) — phần 33.

---

<a id="bat-dau"></a>

## 1. Bài toán thật mà TF-IDF đang giải quyết

Giả sử bạn tải một báo cáo tài chính 55 trang và hỏi:

> Điều kiện ghi nhận doanh thu là gì?

Máy cần tìm các đoạn như:

> Doanh thu bán hàng được ghi nhận khi chuyển giao rủi ro và lợi ích…

Trong cùng báo cáo còn có bảng tài sản, khoản vay, danh sách công ty con, lưu chuyển tiền tệ, thuyết minh hàng tồn kho…

Nếu đưa tất cả nội dung ra trước mặt người dùng thì chưa phải một hệ thống tìm kiếm có ích. Ta cần xếp các đoạn liên quan lên trước.

TF-IDF giúp biến câu hỏi và các đoạn tài liệu thành những dãy số để máy so sánh.

Cách nghĩ gần gũi:

- Mỗi chunk giống một tấm thẻ ghi nội dung.
- Mỗi từ/cụm từ giống một nhãn trên thẻ.
- Nhãn xuất hiện nhiều trên một thẻ có thể mô tả thẻ đó tốt.
- Nhãn xuất hiện trên gần như mọi thẻ ít giúp phân biệt thẻ nào phù hợp hơn.
- Câu hỏi cũng có các nhãn. Máy tìm các thẻ có những nhãn khớp và trọng số phù hợp.

Từ “có thể” ở đây rất quan trọng. Một từ hiếm có thể là thuật ngữ hữu ích, nhưng cũng có thể là lỗi OCR.

**TF-IDF đánh giá mức độ liên quan về từ ngữ, không tự xác minh sự thật.**

---

## 2. TF-IDF nằm ở đâu trong dự án?

### 2.1. Khi nạp tài liệu

```text
PDF / TXT
    ↓
Đọc văn bản; OCR nếu cần và được cấu hình
    ↓
Chia thành chunk: đoạn văn, bảng hoặc dòng chỉ tiêu
    ↓
Trích xuất facts và xây graph
    ↓
Lưu chunks.json
    ↓
TF-IDF fit_transform trên text của toàn bộ chunk đang có
    ↓
Ma trận TF-IDF trong bộ nhớ, sẵn sàng tìm kiếm
```

Phần điều phối nằm trong [service.py](C:/DXD/Multimodal-RAG-Financial/src/financial_rag/service.py:37).

### 2.2. Khi người dùng hỏi — đúng theo bản local hiện tại

```text
Câu hỏi + báo cáo được chọn
    ↓
Kiểm tra phạm vi báo cáo
    ↓
Thử factual_answer: luật, facts và phép tính xác định
    ├─ Có kết quả trả về → trả kết quả / yêu cầu làm rõ / thông báo thiếu bằng chứng
    │                    → không chạy TF-IDF cho nhánh này
    │
    └─ Không có kết quả phù hợp
           ↓
       Mở rộng câu hỏi bằng alias
           ↓
       TF-IDF + điểm cộng từ graph + các quy tắc lọc
           ↓
       Chọn tối đa top_k đoạn có điểm dương
           ↓
       Trả đoạn trích và nguồn, không coi đó là kết luận đã xác minh
```

Xem [service.py, hàm ask](C:/DXD/Multimodal-RAG-Financial/src/financial_rag/service.py:82).

Vì vậy, phát biểu “mọi câu hỏi đều đi qua TF-IDF rồi Ollama trả lời” **không đúng với bản local hiện tại**.

Ở bản này, `use_llm=True` cũng bị bỏ qua trong `ask()`, kèm cảnh báo. TF-IDF không cần gọi Ollama hoặc API để tính vector.

### 2.3. Ai làm việc gì?

| Thành phần | Nhiệm vụ | Không nên gán nhầm nhiệm vụ |
| --- | --- | --- |
| Đọc PDF / OCR | Chuyển nội dung trang thành chữ, vị trí, bảng | Không xếp hạng câu hỏi |
| Chunker | Chia nội dung thành các đơn vị tìm kiếm | Không tự tính TF-IDF |
| TF-IDF | Biểu diễn từ/cụm từ bằng trọng số | Không hiểu công thức kế toán |
| Graph | Lưu quan hệ fact, chỉ tiêu, chunk; hỗ trợ cộng điểm | Không phải mạng neural được huấn luyện |
| Bộ luật trả lời | Nhận diện yêu cầu được hỗ trợ, kiểm tra dữ kiện | Không bảo đảm xử lý mọi cách hỏi |
| Calculator | Thực hiện phép tính đã lập trình | Không suy ra số đầu vào bằng TF-IDF |
| Evidence | Mang đoạn trích, trang, nguồn, bbox nếu có | Không phải bản thân câu trả lời được chứng minh chỉ nhờ điểm cao |

---

## 3. Từ vựng tối thiểu để không bị “ngợp”

| Từ | Hiểu đơn giản | Ví dụ trong dự án |
| --- | --- | --- |
| Document / báo cáo | Một tài liệu nguồn | Một PDF báo cáo tài chính |
| Chunk | Một phần nội dung được đưa vào tìm kiếm | Đoạn thuyết minh khoảng vài trăm ký tự |
| Corpus | Toàn bộ tập văn bản dùng để lập chỉ mục | Danh sách `self.chunks` |
| Token | Một đơn vị chữ/số sau khi tách | `doanh`, `thu`, `2026` |
| Term / feature | Một đặc trưng được cấp một cột | `doanh` hoặc bigram `doanh thu` |
| Vocabulary | Danh sách term và số thứ tự cột | `{"doanh": 2, "doanh thu": 3, ...}` |
| Vector | Một dãy số theo thứ tự cột cố định | `[0, 0.7, 0.2, 0]` |
| Matrix / ma trận | Nhiều vector xếp thành các hàng | Mỗi hàng là một chunk |
| Sparse / thưa | Phần lớn ô là 0, nên chỉ lưu phần khác 0 | Một chunk chỉ chứa ít term trong kho từ vựng |
| Query | Câu hỏi dùng để tìm kiếm | “Chính sách ghi nhận doanh thu?” |
| Fit | Học thống kê của corpus và xây bộ biến đổi | Từ vựng + IDF |
| Transform | Dùng bộ biến đổi đã có để biến chữ thành số | Biến câu hỏi thành vector |
| Score | Điểm dùng để xếp hạng | `Evidence.score` |
| Top-k | Lấy tối đa k kết quả đứng đầu | `top_k=5` |

Trong sách về TF-IDF, “document” thường là đơn vị được đếm khi tính DF. Trong dự án này, đơn vị đó là **chunk**, không phải nguyên file PDF.

Đây là lý do cùng một từ xuất hiện trong 20 chunk của một PDF vẫn có thể đóng góp 20 vào DF.

---

## 4. Chunk được tạo thế nào và vì sao nó ảnh hưởng TF-IDF?

### 4.1. Không phải cứ một trang là một chunk

[Chunker.split](C:/DXD/Multimodal-RAG-Financial/src/financial_rag/ingestion.py:197) có thể tạo:

1. Chunk dòng chỉ tiêu tài chính từ `extract_statement`.
2. Chunk bảng từ `page.tables`.
3. Chunk đoạn văn từ `page.text`.

Vì vậy:

```text
55 trang ≠ 55 chunk
1 PDF ≠ 1 hàng trong ma trận TF-IDF
```

Một phần nội dung cũng có thể được biểu diễn ở nhiều dạng, chẳng hạn vừa trong dòng chỉ tiêu, vừa trong đoạn văn của trang.

### 4.2. `chunk_size=900` nghĩa là gì?

Trong cách chia đoạn hiện tại:

- `900` là giới hạn kích thước theo **ký tự của chuỗi Python**, không phải 900 token của LLM.
- `overlap=120` là vùng chồng lấn theo ký tự.
- Code cố lùi điểm cắt về một dấu cách trong nửa sau của đoạn để hạn chế cắt cuối một từ.
- Đoạn tiếp theo bắt đầu gần `end - overlap`.
- Việc chia diễn ra riêng từng trang; chunk đoạn văn không nối tùy ý qua nhiều trang.

Ví dụ hình dung:

```text
Chunk A: [--------------- khoảng 900 ký tự ---------------]
Chunk B:                                  [---120---|-------------------]
                                         phần chồng lấn
```

Đây là sơ đồ minh họa, không phải tọa độ cắt chính xác cho mọi trang.

### 4.3. Overlap ảnh hưởng DF

Nếu cụm “doanh thu thuần” nằm trong vùng chồng lấn của hai chunk, nó xuất hiện trong cả hai đơn vị TF-IDF.

Do đó:

- N tăng theo số chunk thực tế.
- DF có thể tăng do chồng lấn.
- IDF phụ thuộc cách chunk, không chỉ phụ thuộc nội dung báo cáo gốc.

Chunk nhỏ giúp tìm đúng đoạn chi tiết hơn nhưng dễ thiếu ngữ cảnh. Chunk lớn giữ nhiều ngữ cảnh nhưng có thêm nhiều từ không liên quan, làm hướng vector khác đi. Không có một kích thước luôn tối ưu cho mọi báo cáo.

### 4.4. N trong lần kiểm tra dự án này

Tại lần kiểm tra ngày 20/10/2026, chỉ mục local có **360 chunk thuộc 3 document ID**: hai nhóm có 179 chunk, một nhóm có 2 chunk.

Con số này chỉ là ảnh chụp tại thời điểm kiểm tra, **không phải hằng số trong code**.

Ở trạng thái đó:

```text
N = 360
```

Dù bạn chỉ tìm trong một document ID có 179 chunk, IDF hiện tại vẫn được fit từ tập 360 chunk. Lý do: code lọc `document_id` sau bước tính điểm, không fit một vectorizer riêng cho báo cáo được chọn.

---

## 5. Trước khi tính trọng số: chữ được biến thành term như thế nào?

### 5.1. Cấu hình thực tế

```python
TfidfVectorizer(
    lowercase=True,
    strip_accents="unicode",
    ngram_range=(1, 2),
    max_features=50_000,
    token_pattern=r"(?u)\b\w+\b",
)
```

Đây là cấu hình trong [retrieval.py](C:/DXD/Multimodal-RAG-Financial/src/financial_rag/retrieval.py:32).

### 5.2. `lowercase=True`

Chuyển chữ hoa thành chữ thường.

```text
DOANH THU → doanh thu
Doanh Thu → doanh thu
```

Máy không cần tạo các cột khác nhau chỉ vì khác chữ hoa/thường.

### 5.3. `strip_accents="unicode"`

Loại các dấu kết hợp bằng chuẩn hóa Unicode:

```text
lợi nhuận → loi nhuan
tăng      → tang
```

Nhưng cần nhớ một chi tiết rất thực tế của tiếng Việt:

```text
đầu tư → đau tu
```

Chữ `đ` **không tự biến thành `d`** trong cơ chế này.

Trong khi đó, hàm `fold()` của dự án có xử lý `đ → d`.
Câu hỏi đi qua `expand_query()` rồi `fold()`, nhưng corpus đưa trực tiếp `chunk.text` vào vectorizer.

Kết quả đã kiểm tra:

```text
Analyzer trên văn bản "Đầu tư":
["đau", "tu", "đau tu"]

Analyzer trên fold("Đầu tư"):
["dau", "tu", "dau tu"]
```

Đây là một điểm chưa đồng nhất của code hiện tại, không phải một tính năng của TF-IDF. Không nên thuyết trình rằng hai phía đã bỏ dấu hoàn toàn giống nhau.

### 5.4. `token_pattern=r"(?u)\b\w+\b"`

Đọc từng phần:

| Ký hiệu | Ý nghĩa |
| --- | --- |
| `r"..."` | Chuỗi raw của Python; giữ dấu `\` để biểu thức chính quy xử lý |
| `(?u)` | Dùng cách hiểu Unicode cho ký tự |
| `\b` | Ranh giới từ |
| `\w` | Ký tự thuộc nhóm chữ/số/gạch dưới |
| `+` | Một hoặc nhiều ký tự |

Điểm quan trọng: regex này cho phép token chỉ có **một ký tự**.

Ví dụ câu:

```text
Doanh thu năm 2026: 1.234,56 tỷ đồng.
```

Các unigram thực tế:

```text
doanh, thu, nam, 2026, 1, 234, 56, ty, đong
```

Do đó, số `1.234,56` bị tách thành các mảnh `1`, `234`, `56`. TF-IDF **không tự hiểu đây là số tiền 1.234,56**.

Dấu âm, dấu phần trăm, ngoặc kế toán cũng không được lưu thành một cấu trúc số tài chính hoàn chỉnh bởi bước tokenizer này.

### 5.5. `ngram_range=(1, 2)`

Code dùng cả:

- **Unigram:** một token.
- **Bigram:** hai token liền nhau.

Ví dụ:

```text
doanh thu thuan
```

Sinh ra:

```text
Unigram: doanh | thu | thuan
Bigram:  doanh thu | thu thuan
```

Không sinh trigram `doanh thu thuan`, vì giới hạn trên là 2.

Một cụm tài chính gồm nhiều âm tiết chưa chắc là một “từ” trong cách tách này. Đây không phải bộ tách từ tiếng Việt chuyên biệt biết tự gộp `lợi_nhuận` hay `hội_đồng_quản_trị`.

Bigram giúp giữ một phần thứ tự cục bộ, nhưng không phải hiểu cú pháp hoặc ý nghĩa cả câu.

### 5.6. `max_features=50_000`

Giới hạn số cột của từ vựng ở tối đa 50.000 feature.

Nó không có nghĩa:

- Chỉ đọc 50.000 chunk.
- Chỉ xét 50.000 từ đầu tiên.
- Chỉ xử lý 50.000 ký tự.
- Chọn 50.000 từ có IDF cao nhất.

Khi cần giới hạn, vectorizer chọn đặc trưng theo tần suất tổng trong corpus sau các bước lọc cấu hình. Unigram và bigram cùng nằm trong giới hạn này.

### 5.7. Các mặc định không viết ra nhưng vẫn có hiệu lực

Đã kiểm tra trong môi trường local dùng scikit-learn `1.9.1`:

| Tham số | Giá trị đang có hiệu lực | Hệ quả |
| --- | --- | --- |
| `analyzer` | `"word"` | Đặc trưng dựa trên token từ, không phải character n-gram |
| `use_idf` | `True` | Có nhân IDF |
| `smooth_idf` | `True` | Công thức IDF có cộng 1 để làm trơn |
| `sublinear_tf` | `False` | TF dùng số lần đếm, không dùng `1 + ln(count)` |
| `binary` | `False` | Đếm lặp, không chỉ ghi có/không |
| `norm` | `"l2"` | Chuẩn hóa độ dài vector về 1 nếu vector khác 0 |
| `stop_words` | `None` | Không có danh sách từ dừng được khai báo |
| `min_df` | `1` | Không loại term chỉ vì nó xuất hiện trong đúng một chunk |
| `max_df` | `1.0` | Không loại term chỉ vì xuất hiện ở mọi chunk |
| `dtype` | `float64` | Trọng số dùng số thực 64-bit |

Có thể đối chiếu tên tham số tại [API chính thức TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html). Phần giải thích trong tài liệu này bám cấu hình và hành vi đã kiểm tra trong dự án, không giả định mọi ví dụ TF-IDF trên mạng dùng cùng mặc định.

---

<a id="cong-thuc"></a>

## 6. TF: một term xuất hiện bao nhiêu lần trong một chunk?

TF là viết tắt của **Term Frequency**.

Trong code hiện tại:

```text
TF(t, d) = số lần term t xuất hiện trong chunk d
```

Viết bằng toán:

\[
\operatorname{TF}(t,d)=c(t,d)
\]

Trong đó:

- `t`: term đang xét.
- `d`: chunk đang xét.
- `c(t,d)`: số lần đếm được.

Ví dụ:

```text
Chunk: doanh thu doanh thu tang
```

| Term | TF |
| --- | ---: |
| `doanh` | 2 |
| `thu` | 2 |
| `tang` | 1 |
| `loi` | 0 |

Nếu bật bigram như code thật:

| Bigram | TF |
| --- | ---: |
| `doanh thu` | 2 |
| `thu doanh` | 1 |
| `thu tang` | 1 |

### Có tài liệu viết TF = số lần xuất hiện / tổng số từ, vậy ai đúng?

Có nhiều biến thể TF.

Một biến thể phổ biến:

\[
\operatorname{TF}_{relative}(t,d)=\frac{c(t,d)}{\text{tổng số token trong }d}
\]

Một biến thể khác:

```text
TF_log = 1 + ln(count), nếu count > 0
```

Nhưng dự án hiện tại dùng **raw count**, rồi nhân IDF, rồi chuẩn hóa L2.

Nếu chỉ chia toàn bộ trọng số một vector cho cùng một số dương trước L2, hệ số đó sẽ bị triệt tiêu khi chuẩn hóa. Điều này giải thích vì sao một số cách tính TF tương đối vẫn cho vector cuối giống raw TF + L2. Tuy nhiên, không nên vì thế mô tả sai các bước mà code thật đang thực hiện.

---

## 7. DF: term có mặt trong bao nhiêu chunk?

DF là **Document Frequency**; “document” ở đây là đơn vị chunk.

\[
\operatorname{DF}(t)=\#\{d:\ t\text{ xuất hiện trong }d\}
\]

Đọc công thức bằng tiếng Việt:

> Đếm số chunk có chứa term t.

Giả sử:

```text
C1: doanh thu doanh thu tang
C2: loi nhuan tang
C3: chi phi tang
```

| Term | Trong C1? | Trong C2? | Trong C3? | DF |
| --- | --- | --- | --- | ---: |
| `doanh` | Có | Không | Không | 1 |
| `thu` | Có | Không | Không | 1 |
| `tang` | Có | Có | Có | 3 |
| `loi` | Không | Có | Không | 1 |

`doanh` xuất hiện hai lần trong C1 nhưng DF vẫn là 1.

### Phân biệt TF, DF và tổng tần suất corpus

Nếu một term xuất hiện 100 lần trong một chunk, không xuất hiện ở chunk nào khác:

```text
TF(term, chunk đó) = 100
DF(term)          = 1
Tổng số lần trong corpus = 100
```

Nếu một term xuất hiện một lần trong mỗi chunk của 100 chunk:

```text
TF ở mỗi chunk = 1
DF(term)       = 100
Tổng số lần trong corpus = 100
```

Hai tình huống có cùng tổng số lần xuất hiện nhưng DF rất khác nhau.

**DF trả lời “phổ biến rộng đến đâu”, không phải “lặp tổng cộng bao nhiêu lần”.**

---

## 8. N là gì? Trả lời dứt điểm chuyện “100 chunk”

Trong code:

```python
self.matrix = self.vectorizer.fit_transform(
    [chunk.text for chunk in self.chunks]
)
```

Vì vậy:

```text
N = len(self.chunks)
```

Chính xác hơn: đó là số mẫu văn bản đưa vào lần fit đang tạo vectorizer.

Nếu danh sách có 100 chunk thì N=100. Nếu có 360 chunk thì N=360. Không có quy tắc mặc định “cứ 100 chunk tính một lần”.

| Đại lượng | Đếm cái gì? | Ví dụ |
| --- | --- | ---: |
| N | Số chunk dùng để fit | 360 |
| M | Số feature thực tế trong vocabulary | Có thể vài nghìn, tối đa 50.000 |
| `top_k` | Số kết quả muốn lấy sau khi xếp hạng | 5 |
| `chunk_size` | Giới hạn ký tự của đoạn văn | 900 |
| `chunk_overlap` | Ký tự chồng lấn | 120 |

### Câu hỏi có làm N tăng thêm 1 không?

**Không.**

Câu hỏi dùng:

```python
self.vectorizer.transform([query])
```

Nó không được thêm vào corpus và không làm IDF thay đổi.

### Chọn một báo cáo có làm N thu nhỏ không?

**Không trong triển khai hiện tại.**

Tất cả chunk đã được fit trước. `document_id` chỉ loại kết quả của báo cáo khác bằng cách đặt điểm về 0 ở bước sau.

### Có chunk trùng nội dung thì sao?

Nếu chúng tồn tại dưới các ID khác nhau trong `self.chunks`, mỗi hàng vẫn là một mẫu.
TF-IDF không tự biết hai chunk là nội dung trùng để bỏ một hàng khi đếm N hoặc DF.

---

## 9. IDF: giảm vai trò của term quá phổ biến

IDF là **Inverse Document Frequency**.

Công thức cơ bản trong nhiều giáo trình:

\[
\operatorname{IDF}_{basic}(t)=\ln\left(\frac{N}{\operatorname{DF}(t)}\right)
\]

Nhưng với mặc định `smooth_idf=True`, code dự án thực tế dùng:

\[
\boxed{\operatorname{IDF}(t)=\ln\left(\frac{N+1}{\operatorname{DF}(t)+1}\right)+1}
\]

`ln` là log tự nhiên. Bạn có thể hiểu nó là một phép biến đổi làm mức tăng “dịu lại”: tỷ số tăng 10 lần không nhất thiết làm trọng số tăng 10 lần.

### Hai dấu cộng 1 làm gì?

- `N+1` và `DF+1`: làm trơn thống kê.
- `+1` bên ngoài log: term xuất hiện ở mọi chunk vẫn có IDF=1, không bị biến thành trọng số 0.

Đối với vocabulary bình thường được học từ corpus, term đã được giữ lại có DF ít nhất 1.
Làm trơn không có nghĩa code tự thêm mọi từ ngoài từ vựng vào để tìm kiếm.

### Ví dụ N=100

| DF của term | IDF thực tế, làm tròn 6 chữ số |
| ---: | ---: |
| 1 | 4.921973 |
| 10 | 3.217225 |
| 100 | 1.000000 |

Cùng một TF, term chỉ xuất hiện ở một chunk có trọng số thô cao hơn term xuất hiện ở tất cả chunk.

### IDF có phải độ quan trọng về tài chính không?

Không.

IDF của một từ là thống kê về độ hiếm trong corpus đang lập chỉ mục.

Một số tài khoản, mã biểu mẫu, từ lỗi OCR có thể hiếm nên IDF cao. Điều đó không chứng minh nó quan trọng đối với câu hỏi hoặc đúng về mặt nghiệp vụ.

---

## 10. TF × IDF mới chỉ là trọng số trước chuẩn hóa

Trọng số thô:

\[
w(t,d)=\operatorname{TF}(t,d)\times\operatorname{IDF}(t)
\]

Ví dụ:

```text
TF("doanh", C1) = 2
IDF("doanh")    = 1.693147
```

Thì:

```text
w("doanh", C1) = 2 × 1.693147 = 3.386294
```

Một chunk sẽ có một trọng số như vậy cho mỗi cột trong vocabulary.

Nếu vocabulary là:

```text
[chi, doanh, loi, nhuan, phi, tang, thu]
```

Thì vector thô của C1 là:

```text
[0, 3.386294, 0, 0, 0, 1, 3.386294]
```

Không có `loi` trong C1 nên cột `loi` là 0. Không phải là máy “quên” term đó; cột vẫn tồn tại và có giá trị 0.

### Tại sao phải cùng thứ tự cột?

Hai dãy số chỉ so sánh được nếu cùng một vị trí chỉ cùng một term.

```text
Vector A: cột 2 = doanh
Vector B: cột 2 = doanh
```

Nếu vector B tự tạo từ vựng riêng khiến cột 2 là `nhuan`, phép nhân hai vector sẽ so nhầm.

Đó là lý do query phải dùng **chính vectorizer đã fit corpus**.

---

## 11. Chuẩn hóa L2: đưa các vector về cùng độ dài

### 11.1. Tại sao cần?

Một chunk dài thường chứa nhiều từ và nhiều lần lặp. Nếu chỉ nhân cộng trọng số thô, độ dài có thể lấn át mức độ liên quan.

Chuẩn hóa L2 giúp so sánh hướng phân bố trọng số thay vì chỉ nhìn độ lớn tuyệt đối.

Nó giảm một loại ảnh hưởng của độ dài; không làm mọi chunk dài/ngắn trở nên tương đương về ngữ nghĩa.

### 11.2. Độ dài vector

Với vector `[a, b, c]`:

\[
\|v\|_2=\sqrt{a^2+b^2+c^2}
\]

Ví dụ dễ:

```text
v = [3, 4]
độ dài = sqrt(3² + 4²) = sqrt(25) = 5
```

Chia từng phần tử cho 5:

```text
v_chuẩn_hóa = [0.6, 0.8]
```

Kiểm tra:

```text
sqrt(0.6² + 0.8²) = 1
```

### 11.3. Công thức tổng quát

\[
\widehat{w}_{t,d}=\frac{w(t,d)}{\sqrt{\sum_{j=1}^{M}w(j,d)^2}}
\]

Giải nghĩa:

1. Bình phương từng trọng số.
2. Cộng tất cả.
3. Lấy căn bậc hai.
4. Chia từng trọng số ban đầu cho kết quả đó.

Nếu cả vector là 0, không chia cho 0; đầu ra được giữ là vector 0.

### 11.4. Phần code nào chuẩn hóa?

Trong `retrieval.py` không có một vòng lặp tự viết để làm điều này.

`TfidfVectorizer` gọi phần biến đổi TF-IDF của scikit-learn, và mặc định `norm="l2"` thực hiện chuẩn hóa.

Vì vậy:

> Giá trị trong `self.matrix` là TF-IDF đã chuẩn hóa L2, không chỉ là TF nhân IDF thô.

---

<a id="tinh-tay"></a>

## 12. Ví dụ tính tay hoàn chỉnh: từ văn bản đến thứ hạng

### 12.1. Quy ước để học

Ví dụ này là **dữ liệu tự tạo**, không phải số liệu báo cáo thật.

Để bảng nhỏ và dễ tính, trước tiên chỉ dùng unigram `(1, 1)`.
Code thật của dự án dùng `(1, 2)`; phần 13 sẽ tính lại bằng cấu hình thật.

```text
C1 = "doanh thu doanh thu tang"
C2 = "loi nhuan tang"
C3 = "chi phi tang"

Q  = "doanh thu"
```

### 12.2. Bước 1: từ vựng

```text
[chi, doanh, loi, nhuan, phi, tang, thu]
```

Có 7 cột nên M=7. Có 3 chunk nên N=3.

### 12.3. Bước 2: bảng TF

| Chunk | chi | doanh | loi | nhuan | phi | tang | thu |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C1 | 0 | 2 | 0 | 0 | 0 | 1 | 2 |
| C2 | 0 | 0 | 1 | 1 | 0 | 1 | 0 |
| C3 | 1 | 0 | 0 | 0 | 1 | 1 | 0 |

### 12.4. Bước 3: DF và IDF

| Term | DF | Phép tính | IDF |
| --- | ---: | --- | ---: |
| chi | 1 | ln(4/2)+1 | 1.693147 |
| doanh | 1 | ln(4/2)+1 | 1.693147 |
| loi | 1 | ln(4/2)+1 | 1.693147 |
| nhuan | 1 | ln(4/2)+1 | 1.693147 |
| phi | 1 | ln(4/2)+1 | 1.693147 |
| tang | 3 | ln(4/4)+1 | 1.000000 |
| thu | 1 | ln(4/2)+1 | 1.693147 |

### 12.5. Bước 4: trọng số thô của C1

```text
[0, 3.386294, 0, 0, 0, 1, 3.386294]
```

Độ dài:

```text
sqrt(3.386294² + 1² + 3.386294²) ≈ 4.892237
```

Chia mỗi phần tử cho độ dài đó, ta được xấp xỉ:

```text
C1_normalized ≈ [0, 0.692, 0, 0, 0, 0.204, 0.692]
```

Đây là số làm tròn để dễ đọc; demo đi kèm in độ chính xác cao hơn và kiểm tra phép tính.

### 12.6. Bước 5: vector của câu hỏi

Q có `doanh` một lần, `thu` một lần. Nó dùng IDF đã học từ C1, C2, C3:

```text
Q_raw = [0, 1.693147, 0, 0, 0, 0, 1.693147]
```

Sau L2:

```text
Q_normalized = [0, 0.707107, 0, 0, 0, 0, 0.707107]
```

### 12.7. Bước 6: nhân các cột tương ứng rồi cộng

```text
score(C1, Q)
  = cột doanh của C1 × cột doanh của Q
  + cột thu   của C1 × cột thu   của Q
  + các cột còn lại
```

Các cột còn lại đóng góp 0 vì ít nhất một phía là 0.

Kết quả:

```text
score(C1, Q) ≈ 0.979
score(C2, Q) = 0
score(C3, Q) = 0
```

C1 đứng đầu.

### 12.8. Tại sao không phải 1?

C1 còn có `tang`, nhưng Q không có. Hai vector chưa cùng hướng hoàn toàn.

Cosine bằng 1 khi hai vector khác 0 có cùng hướng, không nhất thiết phải là hai câu nguyên văn giống hệt nhau.
Ví dụ lặp lại toàn bộ các term theo cùng một tỷ lệ có thể giữ nguyên hướng sau chuẩn hóa.

### 12.9. Cosine similarity là gì?

Công thức chung:

\[
\cos(d,q)=\frac{d\cdot q}{\|d\|_2\|q\|_2}
\]

Trong đó:

```text
d · q = d1*q1 + d2*q2 + ... + dM*qM
```

Vì các vector khác 0 đã chuẩn hóa L2 nên hai độ dài đều bằng 1:

\[
\cos(d,q)=\widehat d\cdot\widehat q
\]

Với TF-IDF không âm, điểm cosine nằm trong khoảng 0 đến 1 về mặt toán học, bỏ qua sai số làm tròn số thực.

- 0: không có đóng góp từ feature chung trong không gian từ vựng.
- Gần 1: phân bố trọng số giống nhau nhiều.
- Không có nghĩa: “đúng 97,9%”.

Quan hệ giữa L2 và phép nhân vô hướng có thể đối chiếu ở [cosine_similarity của scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html).

---

## 13. Tính lại ví dụ bằng cấu hình thật của dự án

Giữ nguyên C1, C2, C3 và Q, nhưng bật unigram + bigram.

Từ vựng có 14 feature:

```text
chi
chi phi
doanh
doanh thu
loi
loi nhuan
nhuan
nhuan tang
phi
phi tang
tang
thu
thu doanh
thu tang
```

Các feature khác 0 của C1 sau L2:

| Feature | Trọng số C1 | Trọng số Q | Có đóng góp vào score? |
| --- | ---: | ---: | --- |
| doanh | 0.527985 | 0.577350 | Có |
| doanh thu | 0.527985 | 0.577350 | Có |
| tang | 0.155918 | 0 | Không |
| thu | 0.527985 | 0.577350 | Có |
| thu doanh | 0.263993 | 0 | Không |
| thu tang | 0.263993 | 0 | Không |

Điểm:

```text
score(C1, Q) ≈ 0.914497
score(C2, Q) = 0
score(C3, Q) = 0
```

Tại sao bật bigram mà điểm C1 lại thấp hơn ví dụ unigram?

Vì không gian vector đã đổi. Ngoài bigram khớp `doanh thu`, C1 cũng có bigram không xuất hiện trong Q: `thu doanh`, `thu tang`. Chúng góp vào chuẩn hóa độ dài.

**Thêm bigram không bảo đảm mọi điểm đều tăng.** Mục đích là bổ sung đặc trưng có thể giúp xếp hạng tốt hơn trên tập câu hỏi, không phải làm số score lớn hơn.

Điểm ở phần này chỉ là TF-IDF thuần. Hàm `search()` của dự án còn có mở rộng truy vấn, graph và quy tắc lọc.

---

<a id="doc-code"></a>

## 14. Đọc Python từ mức cơ bản trước khi vào hàm

Một vài cú pháp xuất hiện liên tục:

| Cú pháp | Hiểu đơn giản |
| --- | --- |
| `class HybridRetriever:` | Định nghĩa một kiểu đối tượng phụ trách truy xuất |
| `def search(...):` | Định nghĩa một hành động/hàm |
| `self` | Chính đối tượng retriever đang làm việc |
| `self.chunks` | Dữ liệu chunk được giữ trong đối tượng |
| `None` | Chưa có giá trị hoặc chưa sẵn sàng |
| `list[Chunk]` | Gợi ý đây là danh sách các đối tượng Chunk |
| `Path` | Đối tượng biểu diễn đường dẫn |
| `[chunk.text for chunk in chunks]` | Lấy `.text` của từng chunk rồi tạo thành danh sách |
| `{chunk.id: i for i, chunk in enumerate(...)}` | Tạo bảng tra chunk ID → số thứ tự hàng |
| `if ...:` | Chỉ thực hiện khối bên dưới khi điều kiện đúng |
| `return []` | Trả danh sách rỗng và kết thúc hàm |
| `.T` | Đổi hàng thành cột, cột thành hàng của ma trận |
| `@` | Phép nhân ma trận |
| `[::-1]` | Đảo thứ tự một dãy |
| `[:top_k]` | Lấy phần đầu, tối đa top_k phần tử |

Type hint như `TfidfVectorizer | None` giúp người đọc và công cụ biết kiểu dữ liệu dự kiến. Bản thân nó không phải một thuật toán học máy.

---

## 15. File retrieval.py: import và khởi tạo

### 15.1. Các import

```python
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .financial_tables import fold
from .graph import KnowledgeGraph
from .models import Chunk, Evidence
from .storage import atomic_json
```

| Thành phần | Dùng để làm gì? |
| --- | --- |
| `json` | Đọc dữ liệu chunk đã lưu |
| `Path` | Làm việc với đường dẫn file |
| `numpy` dưới tên `np` | Xếp hạng các điểm số |
| `TfidfVectorizer` | Xây vocabulary, tính IDF, biến text thành ma trận |
| `fold` | Chuẩn hóa chữ cho các quy tắc so khớp |
| `KnowledgeGraph` | Truy cập graph và facts liên quan chunk |
| `Chunk` | Cấu trúc một đơn vị nội dung |
| `Evidence` | Cấu trúc kết quả trả về |
| `atomic_json` | Ghi JSON qua cơ chế của dự án |

### 15.2. Hàm `__init__`

```python
def __init__(self, chunks_path: Path, graph: KnowledgeGraph):
    self.chunks_path = chunks_path
    self.graph = graph
    self.chunks: list[Chunk] = []
    self.vectorizer: TfidfVectorizer | None = None
    self.matrix = None
    self.load()
```

Diễn giải theo thứ tự:

1. Nhớ vị trí file chứa các chunk.
2. Nhớ đối tượng graph sẽ dùng để hỗ trợ tìm kiếm.
3. Ban đầu chưa có chunk nào trong bộ nhớ.
4. Ban đầu chưa có vectorizer đã fit.
5. Ban đầu chưa có ma trận TF-IDF.
6. Gọi `load()` để đọc dữ liệu cũ và tạo các cấu trúc cần thiết.

Bạn có thể hình dung:

```text
retriever
 ├─ chunks_path : file dữ liệu
 ├─ chunks      : nội dung + metadata
 ├─ vectorizer  : vocabulary + IDF + cấu hình biến đổi
 ├─ matrix      : vector của tất cả chunk
 └─ graph       : quan hệ giữa chỉ tiêu, facts, chunks
```

`vectorizer` và `matrix` không phải cùng một thứ:

- Vectorizer là bộ quy tắc/thống kê để biến chữ thành số.
- Matrix là kết quả đã biến đổi của corpus.

---

## 16. `load()` và `_fit()`: đọc kho và tạo chỉ mục

Nguồn: [retrieval.py, dòng 22](C:/DXD/Multimodal-RAG-Financial/src/financial_rag/retrieval.py:22).

### 16.1. Đọc chunks.json

```python
def load(self) -> None:
    if self.chunks_path.exists():
        raw = json.loads(self.chunks_path.read_text(encoding="utf-8"))
        self.chunks = [Chunk.model_validate(item) for item in raw]
    self._fit()
```

Đọc từng dòng:

- `exists()` kiểm tra file có tồn tại không.
- `read_text(encoding="utf-8")` đọc nội dung, có hỗ trợ chữ tiếng Việt.
- `json.loads(...)` chuyển chuỗi JSON thành cấu trúc Python.
- `Chunk.model_validate(item)` chuyển từng bản ghi thành đối tượng Chunk và kiểm tra theo schema.
- `self._fit()` xây lại chỉ mục TF-IDF trong bộ nhớ.

Không phải cứ khởi động là đọc lại PDF và chạy OCR. Ở đường này, dữ liệu được đọc từ `chunks.json` đã lưu.

### 16.2. Trường hợp chưa có chunk

```python
if not self.chunks:
    self.vectorizer, self.matrix = None, None
    return
```

Không có tài liệu thì không có gì để học vocabulary.
Code đánh dấu chưa có chỉ mục và dừng `_fit()`.

### 16.3. Khởi tạo bộ biến đổi và fit corpus

```python
self.vectorizer = TfidfVectorizer(
    lowercase=True,
    strip_accents="unicode",
    ngram_range=(1, 2),
    max_features=50_000,
    token_pattern=r"(?u)\b\w+\b",
)
self.matrix = self.vectorizer.fit_transform(
    [chunk.text for chunk in self.chunks]
)
```

`fit_transform` thực hiện hai nhóm công việc:

```text
fit:
  học vocabulary
  đếm N và DF
  tính IDF

transform:
  đếm TF của từng chunk
  nhân với IDF
  chuẩn hóa L2
  trả ma trận thưa
```

`[chunk.text ...]` chỉ lấy trường văn bản. Các trường `page`, `bbox`, `document_id` không tự thành feature, trừ khi nội dung tương ứng đã được ghi vào `text`.

### 16.4. Ma trận có hình dạng gì?

Nếu N=360 và M=8.000:

```text
self.matrix.shape == (360, 8000)
```

Ví dụ M=8.000 ở đây là số minh họa, không phải số vocabulary đã đo của chỉ mục hiện tại.

- Hàng 0 tương ứng `self.chunks[0]`.
- Hàng 1 tương ứng `self.chunks[1]`.
- Cột j tương ứng `self.vectorizer.get_feature_names_out()[j]`.

Giữ đúng sự tương ứng hàng-chunk là điều bắt buộc để không trích dẫn nhầm tài liệu.

### 16.5. “Fit” có phải huấn luyện một mô hình AI lớn?

Không.

Ở đây fit là tính thống kê của corpus. Không cần:

- Nhãn câu hỏi–đáp án.
- GPU.
- Backpropagation.
- Fine-tune Ollama.
- Khóa API.

Nói “không học gì cả” cũng chưa chính xác: vocabulary và IDF vẫn được ước lượng từ dữ liệu đầu vào. Nhưng đây không phải huấn luyện một mạng neural sinh ngôn ngữ.

---

## 17. `upsert()`: thêm hoặc thay nội dung rồi fit lại

Nguồn: [retrieval.py, dòng 41](C:/DXD/Multimodal-RAG-Financial/src/financial_rag/retrieval.py:41).

```python
def upsert(self, chunks: list[Chunk]) -> None:
    reports = {chunk.document_id for chunk in chunks}
    current = {
        chunk.id: chunk
        for chunk in self.chunks
        if chunk.document_id not in reports
    }
    current.update({chunk.id: chunk for chunk in chunks})
    self.chunks = list(current.values())
    self.chunks_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(self.chunks_path, [chunk.model_dump() for chunk in self.chunks])
    self._fit()
```

### 17.1. Ý nghĩa

`upsert` là cách gọi thường dùng cho “thêm mới hoặc cập nhật”.

Trong hàm này:

1. Lấy danh sách document ID có trong đợt chunk mới.
2. Giữ chunk cũ của các báo cáo khác.
3. Loại chunk cũ của những báo cáo đang được thay.
4. Thêm chunk mới, dùng chunk ID làm khóa.
5. Lưu lại JSON.
6. Fit lại **toàn bộ tập chunk sau cập nhật**.

### 17.2. Ví dụ

Trước:

```text
A: 100 chunk
B: 200 chunk
N = 300
```

Nạp lại A sau cải tiến parser, được 120 chunk:

```text
A cũ 100 → được thay bởi A mới 120
B giữ 200
N mới = 320
```

Không phải N=420 vì cộng cả A cũ lẫn A mới, nếu document ID đang được thay là cùng một ID.

### 17.3. Vì sao phải fit lại?

Thêm/thay chunk có thể làm:

- Vocabulary đổi.
- N đổi.
- DF của nhiều term đổi.
- IDF đổi.
- Cột được giữ khi có giới hạn `max_features` đổi.

Do đó không thể chỉ giữ nguyên vector của mọi chunk cũ và tùy tiện ghép một vector được fit riêng.

### 17.4. Thêm chunk có luôn làm IDF tăng?

Không.

Với term t:

- Thêm nhiều chunk không chứa t: N tăng, DF(t) giữ nguyên → IDF(t) tăng.
- Thêm chunk có chứa t: cả N và DF(t) tăng; IDF có thể giảm hoặc giữ nguyên tùy trường hợp.

Ví dụ từ hiếm có N=3, DF=1:

```text
IDF cũ = ln(4/2)+1 ≈ 1.693147
```

Thêm một chunk chỉ có `tang`, không có từ đang xét:

```text
N mới = 4, DF vẫn = 1
IDF mới = ln(5/2)+1 ≈ 1.916291
```

Điểm một câu hỏi cũ có thể thay đổi sau khi nạp thêm báo cáo khác, dù nội dung báo cáo đang hỏi không đổi.

---

<a id="luong-search"></a>

## 18. `search()`, phần 1: xử lý câu hỏi và tính cosine

Nguồn: [retrieval.py, dòng 50](C:/DXD/Multimodal-RAG-Financial/src/financial_rag/retrieval.py:50).

### 18.1. Chưa có chỉ mục thì trả rỗng

```python
if not self.chunks or self.vectorizer is None or self.matrix is None:
    return []
```

Ba điều kiện được nối bằng `or`: chỉ cần một điều đúng thì chưa đủ dữ liệu để tìm kiếm.

### 18.2. Mở rộng câu hỏi bằng luật

```python
from .answers import expand_query
query = expand_query(query)
```

Trong [answers.py](C:/DXD/Multimodal-RAG-Financial/src/financial_rag/answers.py:48):

```python
def expand_query(question):
    result = fold(question)
    for alias, full in ALIASES.items():
        result = re.sub(r"\b" + re.escape(alias) + r"\b", full, result)
    return result
```

Ví dụ:

```text
"LNST"              → "loi nhuan sau thue"
"lãi ròng"          → "loi nhuan sau thue"
"vốn CSH"           → "von chu so huu"
"tổng cộng tài sản" → "tong tai san"
```

Ý nghĩa:

- `fold` xử lý chữ hoa/thường và dấu.
- `ALIASES.items()` duyệt bảng viết tắt/cách gọi khác.
- `re.escape(alias)` tránh coi các ký tự đặc biệt trong alias là cú pháp regex.
- `\b` hai đầu giúp thay cụm ở ranh giới từ.
- `re.sub` thay cụm khớp bằng cụm chuẩn.

Đây là từ điển và luật do người viết code khai báo. **TF-IDF không tự học rằng LNST là lợi nhuận sau thuế.**

### 18.3. Alias riêng cho dòng tiền

```python
for alias, full in {
    "dong tien kinh doanh": "luu chuyen tien thuan tu hoat dong kinh doanh",
    "dong tien dau tu": "luu chuyen tien thuan tu hoat dong dau tu",
    "dong tien tai chinh": "luu chuyen tien thuan tu hoat dong tai chinh",
}.items():
    query = query.replace(alias, full)
```

Ví dụ:

```text
Tại sao dòng tiền kinh doanh âm?
    ↓ fold + alias
tai sao dong tien kinh doanh am?
    ↓ thay bằng thuật ngữ dài
tai sao luu chuyen tien thuan tu hoat dong kinh doanh am?
```

Cụm dài thường gần tên chỉ tiêu trong báo cáo hơn, làm tăng cơ hội khớp feature.

Lưu ý: thêm nhiều từ vào query cũng làm vector thay đổi sau L2. Không được suy rằng mọi lần mở rộng đều chắc chắn tốt hơn; hiệu quả cần kiểm tra bằng tập câu hỏi.

### 18.4. Biến câu hỏi thành vector

```python
query_vector = self.vectorizer.transform([query])
```

Tại sao `[query]`, không chỉ `query`?

API nhận một tập văn bản. Dù có một câu hỏi, ta vẫn đưa nó trong một danh sách một phần tử.

Nếu vocabulary có M feature:

```text
query_vector.shape = (1, M)
```

Nó dùng:

- Cùng bộ tách token.
- Cùng vocabulary và thứ tự cột.
- Cùng IDF học từ corpus.
- Cùng phép chuẩn hóa L2.

Nó **không fit lại**, không tính DF từ riêng câu hỏi.

### 18.5. Từ ngoài vocabulary — OOV

Giả sử từ vựng không có `zxqv`.

```text
query = "zxqv"
```

Vector query có tất cả phần tử bằng 0. TF-IDF thuần sẽ cho mọi điểm bằng 0.

Nếu chỉ một phần query ngoài từ vựng, phần đó bị bỏ qua; các feature đã biết vẫn được tính.

Nếu query có một bigram mới nhưng từng unigram đã có, bigram mới bị bỏ qua, các unigram vẫn có thể đóng góp.

Trong `search()` thật, graph còn có thể cộng điểm sau đó. Vì vậy “query vector bằng 0” chắc chắn làm **điểm TF-IDF thuần** bằng 0, nhưng chưa đủ để kết luận điểm hybrid cuối cùng luôn bằng 0 trong mọi trạng thái graph.

### 18.6. Dòng tính điểm quan trọng nhất

```python
scores = (self.matrix @ query_vector.T).toarray().ravel()
```

Phân tích từng bước:

| Biểu thức | Kích thước | Ý nghĩa |
| --- | --- | --- |
| `self.matrix` | N × M | Tất cả vector chunk |
| `query_vector` | 1 × M | Một vector câu hỏi |
| `query_vector.T` | M × 1 | Đổi hướng để nhân ma trận |
| `self.matrix @ query_vector.T` | N × 1 | Một điểm cho mỗi chunk |
| `.toarray()` | N × 1 | Chuyển kết quả điểm sang mảng thường |
| `.ravel()` | N phần tử | Làm phẳng thành dãy điểm |

Ví dụ:

```text
Trước ravel: [[0.8], [0.1], [0.4]]
Sau ravel:  [0.8, 0.1, 0.4]
```

`@` không phải “ký hiệu AI bí mật”. Nó thực hiện các phép nhân và cộng theo từng cột như ví dụ tính tay.

Code chỉ `.toarray()` trên **N × 1 điểm số**, không chuyển toàn bộ N × M ma trận tài liệu sang dense.

---

## 19. `search()`, phần 2: graph cộng điểm như thế nào?

Nếu chỉ đọc đến dòng nhân ma trận, bạn mới hiểu phần TF-IDF, chưa hiểu toàn bộ `HybridRetriever`.

### 19.1. Xác định câu hỏi chính sách

```python
policy_query = any(
    term in query
    for term in ["ghi nhan", "chinh sach", "dieu kien"]
)
```

`any(...)` trả True khi có ít nhất một điều kiện đúng.

Nếu hỏi “điều kiện ghi nhận doanh thu”, code muốn ưu tiên đoạn giải thích, không ưu tiên một dòng số doanh thu chỉ vì nó gắn với Metric doanh thu.

### 19.2. Tạo bảng tra vị trí chunk

```python
graph = self.graph.graph
positions = {chunk.id: i for i, chunk in enumerate(self.chunks)}
query_words = set(fold(query).split())
```

Ví dụ `positions`:

```python
{
    "chunk_A": 0,
    "chunk_B": 1,
    "chunk_C": 2,
}
```

Graph nói “chunk_B liên quan”, retriever tra ra hàng 1 rồi cộng vào `scores[1]`.

`set` loại từ lặp:

```text
["doanh", "thu", "doanh"] → {"doanh", "thu"}
```

Phần graph dùng `.split()` theo khoảng trắng, không phải chính analyzer unigram/bigram của vectorizer. Hai phép so khớp này là hai cơ chế khác nhau.

### 19.3. Chỉ xét node Metric

```python
for node, attrs in graph.nodes(data=True):
    if policy_query:
        break
    if attrs.get("kind") != "Metric":
        continue
```

- `attrs` chứa thuộc tính của node.
- Node không phải chỉ tiêu tài chính thì bỏ qua.
- Câu hỏi chính sách thì dừng phần cộng điểm này.

### 19.4. Tính tỷ lệ khớp tên chỉ tiêu

```python
words = set(fold(attrs["name"]).split())
match = len(words & query_words) / max(len(words), 1)
if match < 0.6:
    continue
```

`&` là giao hai tập: các từ có ở cả hai phía.

Ví dụ:

```text
Tên Metric: "Doanh thu thuần"
words = {"doanh", "thu", "thuan"}

Câu hỏi chứa: "doanh", "thu"
Số từ chung = 2
match = 2/3 ≈ 0.667
```

Vì 0.667 ≥ 0.6, Metric này đủ điều kiện.

Đây không phải cosine similarity và cũng không phải Jaccard. Mẫu số là số từ của tên Metric, không phải số từ trong hợp hai tập.

`max(len(words), 1)` tránh chia cho 0 khi tên chỉ tiêu không tạo được từ.

### 19.5. Đi theo liên kết đến chunk

Các quan hệ được tạo trong [graph.py](C:/DXD/Multimodal-RAG-Financial/src/financial_rag/graph.py:62):

```text
Metric ← MEASURES ← Fact → SUPPORTED_BY → Chunk
```

Code truy xuất:

```python
for fact in graph.predecessors(node):
    for target in graph.successors(fact):
        if target in positions:
            scores[positions[target]] += 0.35 * match
```

Diễn giải:

1. Từ Metric, tìm các node có cạnh đi vào nó; với graph hiện tại, đó là các Fact đo chỉ tiêu.
2. Từ Fact, tìm các node có cạnh đi ra; nếu một node là chunk có trong bảng `positions`, cộng điểm cho chunk đó.
3. Số cộng thêm là `0.35 × match`.

Vòng lặp này dựa vào cấu trúc graph được xây; nó không kiểm tra tường minh nhãn relation ở từng cạnh trong đoạn code trên.

### 19.6. Ví dụ cộng điểm

```text
TF-IDF ban đầu = 0.40
match         = 2/3
Điểm cộng     = 0.35 × 2/3 ≈ 0.233333
Điểm mới      ≈ 0.633333
```

Một chunk có thể được cộng nhiều lần nếu có nhiều fact/chỉ tiêu phù hợp.

Vì vậy:

```text
score cuối có thể > 1
```

Không được viết:

> “Điểm kết quả của dự án luôn là cosine trong [0,1].”

Đúng hơn:

> “Điểm ban đầu là cosine TF-IDF; điểm cuối đã cộng graph và áp dụng luật.”

### 19.7. Hệ số 0.35 từ đâu ra?

Là hệ số được lập trình cố định trong code.

Nó không phải:

- Kết quả tự học từ dữ liệu.
- Xác suất fact đúng.
- Điểm của một Graph Neural Network.
- Một định lý bảo đảm 0.35 luôn tối ưu.

Muốn thay hệ số hợp lý cần bộ đánh giá, không chỉ chọn vì số trông đẹp.

---

## 20. `search()`, phần 3: lọc và chọn kết quả

### 20.1. Giới hạn vào báo cáo đã chọn

```python
if document_id:
    for i, chunk in enumerate(self.chunks):
        if chunk.document_id != document_id:
            scores[i] = 0
```

Lưu ý thứ tự: điểm đã được tính trên chỉ mục toàn cục, rồi mới đặt về 0 cho báo cáo ngoài phạm vi.

Hệ quả:

- Kết quả có điểm dương trả về nằm trong báo cáo được chọn.
- Nhưng IDF vẫn là IDF từ toàn bộ corpus.
- Thời gian tính điểm ban đầu không chỉ dành cho chunk của báo cáo được chọn.

### 20.2. Hạ điểm dòng số khi hỏi về chính sách

```python
if policy_query and chunk.financial_rows:
    scores[i] *= 0.1
```

`financial_rows` chứa dữ liệu chỉ tiêu dạng cấu trúc. Danh sách không rỗng được hiểu là True.

Ví dụ minh họa:

```text
Chunk dòng doanh thu:       điểm 0.80 → 0.08
Chunk thuyết minh chính sách: điểm 0.30 → 0.30
```

Đoạn giải thích vượt lên trên.

Đây là quy tắc dựa trên loại câu hỏi và loại chunk, không phải phép tính TF-IDF tiêu chuẩn.

### 20.3. Loại mục lục bằng dấu hiệu văn bản

```python
compact = fold(chunk.text).replace(" ", "")
if "noidungtrang" in compact and "thuyetminh" in compact:
    scores[i] = 0
```

Mục lục có thể chứa nhiều thuật ngữ tài chính nhưng không chứa lời giải thích hay số liệu cần hỏi.

Quy tắc này chỉ nhận diện mẫu đang viết trong code. Nó không bảo đảm mọi mục lục của mọi mẫu PDF đều bị loại.

### 20.4. Xếp hạng

```python
indices = np.argsort(scores)[::-1][:top_k]
```

Ví dụ:

```text
scores = [0.2, 0.8, 0.4]

np.argsort(scores) → [0, 2, 1]
                    thứ tự từ điểm thấp đến cao

[::-1]            → [1, 2, 0]
                    đảo thành cao đến thấp

[:2]              → [1, 2]
                    lấy 2 kết quả đầu
```

Các chỉ số này dùng để lấy lại chunk tương ứng trong `self.chunks`.

Không nên coi thứ tự giữa các điểm bằng hệt nhau là một quyết định ngữ nghĩa đáng tin cậy.

### 20.5. Chỉ giữ điểm dương

```python
for index in indices
if scores[index] > 0
```

Nếu `top_k=5` nhưng chỉ 2 chunk có điểm dương, trả 2 chunk.

Nếu mọi điểm là 0, trả danh sách rỗng.

Code hiện tại chưa có một ngưỡng “đủ liên quan” lớn hơn 0 được hiệu chỉnh bằng tập đánh giá. Điểm dương rất nhỏ vẫn có thể được trả.

Vì vậy “nằm trong top 5” không tự chứng minh đây là bằng chứng đủ để trả lời.

---

## 21. Kết quả Evidence: điểm số không tự sinh ra trích dẫn

Nguồn: [retrieval.py, dòng 94](C:/DXD/Multimodal-RAG-Financial/src/financial_rag/retrieval.py:94).

```python
Evidence(
    chunk_id=self.chunks[index].id,
    source=self.chunks[index].source,
    page=self.chunks[index].page,
    text=self.chunks[index].text,
    score=float(scores[index]),
    graph_facts=self.graph.facts_for_chunk(self.chunks[index].id),
    document_id=self.chunks[index].document_id,
    source_url=(
        f"/documents/{self.chunks[index].document_id}/file"
        f"#page={self.chunks[index].page}"
    ),
    bbox=self.chunks[index].bbox,
    extraction_method=self.chunks[index].extraction_method,
)
```

Đoạn trên chỉ xuống dòng lại phần `source_url` cho dễ đọc, không đổi ý nghĩa.

| Trường | Nghĩa |
| --- | --- |
| `chunk_id` | Định danh đoạn đã chọn |
| `source` | Nguồn gắn với chunk |
| `page` | Trang đã lưu khi đọc/chia tài liệu |
| `text` | Nguyên văn nội dung chunk |
| `score` | Điểm hybrid cuối cùng |
| `graph_facts` | Facts đã gắn với chunk |
| `document_id` | Báo cáo sở hữu chunk |
| `source_url` | Đường dẫn phục vụ mở PDF tại trang tương ứng |
| `bbox` | Vùng tọa độ nếu quá trình trích xuất có lưu |
| `extraction_method` | Chẳng hạn text hay OCR |

TF-IDF chỉ giúp chọn chunk. Nó không tự đọc lại PDF để chứng minh `page` hoặc `bbox` đúng.

Trích dẫn tốt phụ thuộc cả pipeline:

```text
Đọc đúng trang
    → lưu đúng metadata
    → giữ đúng hàng ma trận ↔ chunk
    → chọn đúng đoạn
    → hiển thị đúng nguồn
```

Một đoạn có score cao nhưng metadata sai vẫn có thể dẫn người dùng đến sai nguồn.

### Câu trả lời cuối hiện tại được viết thế nào?

Nếu đi qua nhánh fallback, service ghi rõ:

> Các đoạn trích dưới đây chỉ là kết quả tìm kiếm liên quan, không phải kết luận đã được xác minh.

Sau đó liệt kê văn bản và trang.

Đây là sự phân biệt giữa:

- **Retrieval:** đã tìm được nội dung có liên quan.
- **Answering:** đã có bằng chứng đủ và phương pháp hợp lệ để đưa ra câu trả lời.

Tìm đúng đoạn là điều cần, nhưng chưa luôn là điều đủ.

---

<a id="thuc-hanh"></a>

## 22. Chương trình thực hành: chạy và nhìn thấy từng con số

File: [scripts/explain_tfidf.py](C:/DXD/Multimodal-RAG-Financial/scripts/explain_tfidf.py).

Chương trình chỉ dùng 3 câu giả lập, không đọc hay sửa index thật, không gọi Ollama/API.

### 22.1. Mở đúng thư mục

Trong PowerShell:

```powershell
cd C:\DXD\Multimodal-RAG-Financial
```

Nếu bạn đang ở `C:\Windows\system32`, phải chuyển thư mục trước. Đây là lỗi thao tác đường dẫn, không phải lỗi TF-IDF.

Không cần activate môi trường ảo nếu gọi thẳng Python của `.venv` như dưới đây.

### 22.2. Chạy bản dễ tính tay

```powershell
.\.venv\Scripts\python.exe scripts\explain_tfidf.py --mode simplified --verify
```

Kết quả quan trọng đã chạy kiểm chứng:

```text
N = 3
M = 7
IDF(tang) = 1.000000
IDF(doanh) = 1.693147

L2 của C1 trước chuẩn hóa = 4.892237
Trọng số chuẩn hóa doanh = 0.692177
Trọng số chuẩn hóa thu   = 0.692177

Điểm C1 = 0.978886
Điểm C2 = 0
Điểm C3 = 0
```

### 22.3. Chạy cấu hình đúng như retriever chính

```powershell
.\.venv\Scripts\python.exe scripts\explain_tfidf.py --mode project --verify
```

Kết quả:

```text
N = 3
M = 14
Điểm C1 = 0.914497
Điểm C2 = 0
Điểm C3 = 0
```

Chữ `project` nghĩa là **đúng cấu hình vectorizer**. Nó không có nghĩa chương trình đang chạy toàn bộ `service.ask`, alias, graph và bộ lọc.

### 22.4. Đổi câu hỏi

```powershell
.\.venv\Scripts\python.exe scripts\explain_tfidf.py --mode project --query "lợi nhuận" --verify
```

Hãy quan sát C2 được khớp nhờ các feature `loi`, `nhuan`, `loi nhuan`.

Thử:

```powershell
.\.venv\Scripts\python.exe scripts\explain_tfidf.py --mode project --query "zxqv" --verify
```

Query không có feature trong vocabulary → vector 0 → điểm TF-IDF đều 0.

### 22.5. Đọc output theo thứ tự

1. Corpus và các feature sau analyzer.
2. TF, DF, IDF của từng cột.
3. Vector thô và sau L2 của mỗi chunk.
4. Vector query dùng cùng vocabulary/IDF.
5. Tổng điểm và đóng góp từng feature.
6. Ví dụ query ngoài vocabulary.
7. Kiểm chứng tự động nếu có `--verify`.

`--verify` không hỏi một LLM khác xem phép tính có đúng không.
Nó dùng assertion số học để so cách tính tường minh với scikit-learn.

### 22.6. Nếu lệnh không chạy

| Hiện tượng | Kiểm tra |
| --- | --- |
| Không tìm thấy script | Đã `cd` vào thư mục dự án chưa? |
| Không tìm thấy `.venv\Scripts\python.exe` | Môi trường ảo của dự án có tồn tại không? |
| `No module named sklearn` | Có đang dùng đúng Python của dự án không? |
| Terminal hiển thị chữ lỗi | Demo đã đặt stdout UTF-8; có thể cần kiểm tra font/encoding terminal |
| Không có số liệu báo cáo thật trong output | Đúng thiết kế: đây là bài thực hành độc lập, không đọc dữ liệu người dùng |

Không cần chạy server ở cổng 8000 để dùng demo.

---

## 23. Giải thích code demo: tự viết lại toán, không coi thư viện là “hộp kín”

Phần này không thay thế code ứng dụng. Mục đích là bóc tách những gì `TfidfVectorizer` làm thành các phép toán bạn có thể nhìn thấy.

### 23.1. Tạo vectorizer

[build_vectorizer](C:/DXD/Multimodal-RAG-Financial/scripts/explain_tfidf.py:32):

```python
ngram_range=(1, 1) if mode == "simplified" else (1, 2)
```

Đây là biểu thức điều kiện:

- `simplified`: chỉ unigram để tính tay.
- `project`: cả unigram và bigram, giống cấu hình chính.

Các tham số còn lại giữ như dự án.

### 23.2. Dùng analyzer thật để tránh tự tách sai

```python
features = vectorizer.get_feature_names_out()
analyzer = vectorizer.build_analyzer()
token_lists = [analyzer(chunk) for chunk in chunks]
counters = [Counter(tokens) for tokens in token_lists]
```

- `get_feature_names_out()` trả tên feature theo đúng thứ tự cột.
- `build_analyzer()` trả một hàm dùng cùng cách tiền xử lý và n-gram với vectorizer.
- `Counter` là bộ đếm của Python.

Ví dụ:

```python
Counter(["doanh", "thu", "doanh"])
```

tương đương bảng:

```text
doanh: 2
thu:   1
```

### 23.3. Xây ma trận TF

```python
counts = np.array([
    [counter[word] for word in features]
    for counter in counters
])
```

Vòng ngoài: đi từng chunk.

Vòng trong: đi từng feature theo đúng thứ tự chung.

Term không có trong Counter trả 0. Kết quả là bảng TF đã tính ở phần 12.

### 23.4. Tính DF

```python
document_frequency = (counts > 0).sum(axis=0)
```

Tách ba ý:

1. `counts > 0` biến số lần đếm thành có/không.
2. True được tính như 1, False như 0 khi cộng.
3. `sum(axis=0)` cộng theo chiều các hàng, để mỗi cột thu được số chunk có term đó.

Ví dụ một cột:

```text
TF       = [2, 0, 0]
TF > 0   = [True, False, False]
Tổng     = 1
DF       = 1
```

### 23.5. Tính IDF và trọng số thô

```python
idf = np.log((1 + len(chunks)) / (1 + document_frequency)) + 1
raw_weights = counts * idf
```

`np.log` là log tự nhiên.

`counts * idf` dùng broadcasting: mỗi cột TF được nhân với IDF tương ứng của cột đó, trên tất cả hàng.

Khác với `@`:

- `*` ở đây là nhân từng phần tử theo cột.
- `@` là nhân ma trận để cộng các tích và tạo score.

### 23.6. Tự chuẩn hóa L2

```python
lengths = np.linalg.norm(rows, axis=1, keepdims=True)
return np.divide(
    rows,
    lengths,
    out=np.zeros_like(rows),
    where=lengths != 0,
)
```

- `axis=1`: tính độ dài riêng cho từng hàng.
- `keepdims=True`: giữ dạng cột độ dài để chia từng hàng đúng cách.
- `where=lengths != 0`: chỉ chia hàng có độ dài khác 0.
- `out=np.zeros_like(rows)`: hàng toàn 0 giữ nguyên 0.

### 23.7. Tách đóng góp của từng feature

```python
contributions = normalized * normalized_query
```

Với một chunk:

\[
\operatorname{contribution}(t)=\widehat w_{t,d}\widehat w_{t,q}
\]

Tổng theo tất cả feature là điểm cosine:

\[
\operatorname{score}(d,q)=\sum_t\operatorname{contribution}(t)
\]

Ở chế độ project, ví dụ C1:

```text
'doanh'     đóng góp ≈ 0.304832
'doanh thu' đóng góp ≈ 0.304832
'thu'       đóng góp ≈ 0.304832
Tổng chưa làm tròn   ≈ 0.914497
```

Tổng của các số **đã làm tròn** có thể lệch 0.000001 so với kết quả làm tròn của tổng chính xác. Đó là làm tròn, không phải thuật toán sai.

Đây là cách trả lời rất cụ thể câu hỏi:

> Vì sao hệ thống xếp đoạn này lên đầu?

Đối với điểm TF-IDF, bạn có thể chỉ ra feature nào góp bao nhiêu.
Đối với điểm hybrid thật, phải giải thích thêm các khoản cộng graph và hệ số lọc.

### 23.8. Kiểm tra cách tính tay có khớp không

```python
np.testing.assert_allclose(...)
```

Demo kiểm tra:

- IDF tự tính bằng IDF thư viện.
- Vector tự chuẩn hóa bằng vector thư viện.
- Query tự tính bằng query thư viện.
- Tích vô hướng bằng `cosine_similarity`.
- Tổng đóng góp từng feature bằng tổng score.

`allclose` so với sai số số thực cho phép, không đòi mọi phép toán floating-point phải có bit giống hệt nhau.

Demo đổi các ma trận nhỏ sang dense để in cho dễ học. Không nên sao chép cách in toàn ma trận dense này cho hàng triệu chunk.

---

<a id="kiem-tra"></a>

## 24. Cách tự kiểm tra và “debug” TF-IDF khi câu hỏi tìm sai đoạn

Đừng bắt đầu bằng việc thay model hoặc tăng top_k một cách ngẫu nhiên. Theo dõi từng lớp.

### Bước A: văn bản gốc có đọc đúng không?

Kiểm tra chunk thật chứa gì:

- Chữ có bị lỗi OCR không?
- Có giữ được tên chỉ tiêu không?
- Văn bản có đang là mục lục không?
- Dữ liệu bảng có bị dồn cột không?

Nếu OCR đọc sai `doanh thu` thành một chuỗi khác, TF-IDF không tự sửa lỗi đó chỉ nhờ công thức IDF.

### Bước B: query sau chuẩn hóa là gì?

```python
from financial_rag.answers import expand_query

print(expand_query("LNST là bao nhiêu?"))
```

Nhớ rằng `search()` còn thay alias dòng tiền sau `expand_query()`.

### Bước C: analyzer tạo những feature nào?

```python
analyzer = retriever.vectorizer.build_analyzer()
print(analyzer("Đầu tư"))
print(analyzer("dau tu"))
```

Đoạn này giả định bạn đang có đối tượng `retriever` đã fit. Không phải lệnh có thể chạy riêng mà không khởi tạo gì.

### Bước D: query có feature trong vocabulary không?

```python
q = retriever.vectorizer.transform(["câu hỏi đã xử lý"])
print(q.shape)
print(q.nnz)
```

- `shape`: kích thước.
- `nnz`: số phần tử khác 0.

Nếu `nnz=0`, nhánh từ vựng không có feature dùng được để so khớp.

### Bước E: điểm thô khác điểm cuối ở đâu?

```python
raw_scores = (retriever.matrix @ q.T).toarray().ravel()
```

Đối chiếu với `Evidence.score`.

Chênh lệch có thể đến từ:

- Query thực tế khác query bạn thử.
- Graph boost.
- Lọc document.
- Hạ điểm financial_rows khi hỏi chính sách.
- Loại mục lục.

Không nên kết luận scikit-learn tính sai chỉ vì score tự thử khác điểm hybrid trả về.

### Bước F: điểm cao nhưng câu trả lời vẫn sai?

Đặt tiếp các câu hỏi:

- Có chọn đúng kỳ không?
- Có đúng báo cáo riêng/hợp nhất không?
- Đơn vị có đúng không?
- Đoạn chỉ nhắc đến chỉ tiêu hay thực sự trả lời yêu cầu?
- Nguồn có chứng minh câu kết luận không?

Các kiểm tra này vượt ra ngoài bản thân TF-IDF.

---

## 25. Bộ nhớ, tốc độ và việc lưu index

### 25.1. Vì sao phải dùng ma trận thưa?

Giả sử:

```text
N = 10.000 chunk
M = 50.000 feature
```

Ma trận đặc có:

```text
10.000 × 50.000 = 500.000.000 ô
```

Nếu mỗi ô là float64 8 byte:

```text
500.000.000 × 8 = 4.000.000.000 byte
```

Khoảng 4 GB theo đơn vị thập phân, chưa tính overhead.

Trong thực tế mỗi chunk chỉ có một phần nhỏ feature, nên đa số ô là 0.
Sparse lưu phần khác 0 cùng cấu trúc chỉ số, thường tiết kiệm đáng kể.

### 25.2. `.toarray()` có nguy hiểm không?

Phải nhìn nó đang áp dụng cho ma trận nào.

Trong code chính:

```python
(self.matrix @ query_vector.T).toarray()
```

Là N×1, không phải N×M.

Với N=10.000, đó là 10.000 điểm số, khác hoàn toàn 500 triệu trọng số của toàn ma trận dense.

### 25.3. Điều gì tốn thời gian?

- Nạp PDF scan: OCR có thể rất tốn thời gian, nhưng không phải TF-IDF.
- `_fit`: phân tích tất cả text, xây vocabulary, tính IDF và ma trận.
- `search`: biến query, nhân sparse matrix, duyệt graph, lọc, sắp xếp.
- `facts_for_chunk`: code hiện tại duyệt các node graph để lấy facts liên quan.

Việc xếp hạng dùng `np.argsort` trên toàn dãy N điểm, có chi phí điển hình cỡ N log N. Không có cơ chế chỉ xét 100 chunk mỗi truy vấn.

Đừng kết luận tốc độ chỉ từ tên TF-IDF: corpus, OCR, graph và cách lưu/fit đều có ảnh hưởng.

### 25.4. Dự án có lưu ma trận TF-IDF xuống ổ đĩa không?

Ở retriever chính hiện tại:

- Lưu chunk text và metadata qua `chunks.json`.
- Graph lưu riêng qua `graph.json`.
- Không có đoạn lưu `vectorizer`/matrix bằng pickle hoặc joblib trong `retrieval.py`.
- Khởi tạo/load sẽ fit lại từ dữ liệu chunk.

Do đó không nên đi tìm một file “model TF-IDF đã huấn luyện” rồi nghĩ thiếu file đó là lỗi.

### 25.5. Tăng max_features có luôn tốt?

Không.

Thêm feature có thể giữ được thuật ngữ hữu ích, nhưng cũng có thể:

- Giữ thêm lỗi OCR.
- Tăng bộ nhớ.
- Tăng thời gian fit.
- Thay đổi chuẩn hóa và xếp hạng.

Nên đo chất lượng retrieval trên câu hỏi đại diện cùng chi phí thực tế.

---

## 26. Các giới hạn cần nói thật khi bảo vệ dự án

### 26.1. Không tự hiểu đồng nghĩa

`lãi ròng` và `lợi nhuận sau thuế` có thể gần nghĩa về nghiệp vụ, nhưng TF-IDF không tự biết điều đó.

Dự án giải quyết một phần bằng alias. Ngoài alias và feature chung, không có bảo đảm khớp.

### 26.2. Không hiểu phủ định hoặc quan hệ nguyên nhân đầy đủ

```text
Doanh thu tăng do mở rộng thị trường.
Doanh thu không tăng dù mở rộng thị trường.
```

Hai câu có nhiều feature chung nhưng ý nghĩa không giống nhau.
Unigram/bigram có thể phân biệt một số cụm, nhưng không phải bộ suy luận ngữ nghĩa hoàn chỉnh.

### 26.3. Không tự thực hiện số học

```text
Doanh thu năm nay 120, năm trước 100.
```

TF-IDF có thể tìm được đoạn đó. Nó không tự kết luận tăng 20%.
Phép tính thuộc calculator hoặc quy tắc riêng.

### 26.4. Không tự ràng buộc kỳ và đơn vị

Các token `2025`, `2026`, `triệu`, `tỷ` có thể ảnh hưởng score, nhưng không tạo một cam kết kiểu:

> Chỉ được dùng giá trị thuộc quý 1/2026, báo cáo riêng, đơn vị VND.

Ràng buộc nghiệp vụ phải được biểu diễn và kiểm tra riêng.

### 26.5. Không đọc trực tiếp ảnh, biểu đồ

TF-IDF nhận text. Nếu hình chưa được OCR hoặc chưa chuyển thành mô tả/số liệu, nó không nhìn thấy hình đó.

Có thể dùng TF-IDF tìm đoạn chú thích biểu đồ, nhưng đó không đồng nghĩa đọc được đường biểu diễn hay cột trong ảnh.

### 26.6. Từ hiếm có thể là nhiễu

Một chữ lỗi chỉ xuất hiện một lần có thể có IDF cao. Số trang, mã biểu mẫu, OCR lặp cũng có thể chiếm feature.

IDF không biết đó là lỗi.

### 26.7. Đoạn được tìm thấy chưa chắc đủ để kết luận

Một câu hỏi “Tại sao lợi nhuận giảm?” đòi hỏi bằng chứng nguyên nhân.
Tìm đúng dòng số “lợi nhuận giảm” chưa chứng minh nguyên nhân.

### 26.8. Các trường hợp kỹ thuật

| Trường hợp | Hành vi / lưu ý |
| --- | --- |
| Danh sách chunk rỗng | Retriever trả rỗng, chưa có vectorizer |
| Có chunk nhưng toàn dấu câu, không tạo được từ | Có thể gặp `empty vocabulary`; `_fit` hiện chỉ kiểm tra danh sách rỗng |
| Query toàn feature ngoài từ vựng | Vector lexical bằng 0 |
| Điểm dương rất nhỏ | Vẫn có thể được trả nếu nằm top-k và không bị lọc |
| Từ chứa `đ` | Có bất đối xứng giữa fold query và strip_accents corpus |
| Corpus có nội dung trùng/chồng lấn | N và DF vẫn đếm theo hàng chunk |
| Nạp thêm tài liệu khác | IDF/ranking tài liệu cũ có thể thay đổi |

### 26.9. Vì sao vẫn dùng TF-IDF?

Đối với dự án học tập và các câu hỏi có thuật ngữ khớp tài liệu:

- Có công thức rõ ràng.
- Chạy trên CPU, không cần dịch vụ embedding.
- Có thể xem feature và đóng góp điểm.
- Không cần tập nhãn để fit vocabulary/IDF.
- Là baseline hữu ích để so sánh cải tiến.

Đây là điểm mạnh về tính kiểm tra được, không phải lời bảo đảm mọi câu hỏi đều được hiểu đúng.

---

## 27. Đánh giá và cải tiến thế nào cho đúng phương pháp?

Các mục dưới đây là hướng kiểm tra/cải tiến, **không phải tuyên bố chúng đã được triển khai**.

### 27.1. Tạo một tập câu hỏi có nguồn đúng

Mỗi câu cần ghi:

```text
Câu hỏi
document_id đúng
trang hoặc chunk chứa bằng chứng đúng
chỉ tiêu/kỳ/phạm vi cần xét nếu có
đáp án và cách tính nếu là câu số
```

Nên có câu:

- Dùng đúng thuật ngữ.
- Dùng viết tắt.
- Không dấu.
- Hỏi chính sách thay vì số liệu.
- Chứa số âm, 0, đơn vị tiền.
- Hỏi kỳ không có.
- Không có câu trả lời trong tài liệu.
- Đi qua OCR.

### 27.2. Đo retrieval riêng với answering

Ví dụ mỗi câu có một chunk bằng chứng đúng:

- **Hit@5:** bằng chứng đúng có nằm trong 5 kết quả đầu không?
- **MRR:** nghịch đảo vị trí của kết quả đúng đầu tiên, rồi lấy trung bình.

Nếu vị trí đúng của ba câu lần lượt là 1, 2 và không tìm thấy:

```text
Hit@5 = 2/3 ≈ 66.7%
MRR   = (1 + 1/2 + 0)/3 = 0.5
```

Nếu một câu có nhiều chunk liên quan:

```text
Recall@k = số chunk liên quan được tìm thấy trong top-k
           / tổng số chunk liên quan đã gán nhãn
```

Với một nhãn đúng duy nhất, hit và recall ở mức câu có thể trùng về giá trị. Khi có nhiều nhãn thì không nên dùng hai tên thay nhau một cách tùy tiện.

Các chỉ số này không thay thế:

- Độ đúng số tiền.
- Đúng đơn vị/kỳ.
- Đúng công thức.
- Trích dẫn đúng nguồn và thực sự hỗ trợ câu trả lời.
- Từ chối đúng khi thiếu bằng chứng.

### 27.3. Thay một yếu tố mỗi lần

So sánh:

```text
A: TF-IDF unigram
B: TF-IDF unigram + bigram
C: B + alias
D: C + graph boost
E: D + policy/mục lục filter
```

Đây là cách tách đóng góp của từng thành phần. Nếu đổi tất cả cùng lúc, rất khó biết yếu tố nào giúp hoặc làm giảm chất lượng.

### 27.4. Các hướng cải tiến gần với code hiện tại

1. Đồng nhất chuẩn hóa corpus và query, đặc biệt `đ/d`, rồi reindex và kiểm thử.
2. Bổ sung alias có kiểm soát từ câu hỏi thực tế.
3. So sánh word n-gram với character n-gram cho lỗi OCR/không dấu.
4. Kiểm thử cách chunk và overlap.
5. Hiệu chỉnh hệ số graph/điểm ngưỡng bằng tập phát triển.
6. Cân nhắc tách index theo phạm vi nếu muốn IDF riêng từng báo cáo.
7. Đánh giá BM25 hoặc dense retrieval như các baseline khác.

Không nên chọn cấu hình chỉ vì cho điểm similarity cao hơn trên một câu duy nhất.

---

<a id="nhanh-thien"></a>

## 28. Phân biệt TF-IDF bản chính với nhánh Thien

Đây là phụ lục về phiên bản đã review tại commit `08305cb`, không phải mô tả bản chính đang dùng.

Nguồn:
[dense.py của bản review](C:/DXD/Financial-Multimodal-RAG-review-20261020/src/financial_rag/retrieval/dense.py).

| Nội dung | Bản chính local | Nhánh Thien đã review |
| --- | --- | --- |
| File | `src/financial_rag/retrieval.py` | `src/financial_rag/retrieval/dense.py` |
| Vai trò TF-IDF | Nhánh truy xuất từ vựng chính của HybridRetriever | Fallback khi SentenceTransformer/model không sẵn sàng |
| Nội dung vector hóa | `chunk.text` | `context_prefix + content` |
| N-gram TF-IDF | `(1, 2)` | Mặc định `(1, 1)` |
| Giới hạn feature | 50.000 | 512 khi build fallback |
| Bỏ dấu | `strip_accents="unicode"` | Không khai báo bỏ dấu trong fallback |
| Token một ký tự | Có | Mặc định tokenizer yêu cầu ít nhất hai ký tự |
| Lưu ma trận | Sparse trong bộ nhớ; fit từ JSON khi load | Fallback đổi thành dense array, lưu `.npy`, có thể dùng FAISS |
| Xử lý query | Tái dùng vectorizer đã fit | Fallback tạo/fit lại vectorizer từ các chunk hiện có |
| Hybrid kết hợp | TF-IDF + graph/rule boosts | Dense/BGE-M3 hoặc fallback + FTS5 |

### Một nhầm lẫn rất dễ mắc

Tên class `DenseRetriever` không có nghĩa mọi đường trong class đều dùng neural embedding.

```python
TfidfVectorizer(...).fit_transform(texts).toarray()
```

vẫn là TF-IDF từ vựng, chỉ đổi cách lưu từ sparse sang mảng dense. Nó không tự trở thành embedding ngữ nghĩa được học bởi BGE-M3.

### Vì sao không nên lấy query fallback nhánh mới làm mẫu chuẩn?

Vectorizer của lần build không được giữ/lưu để dùng lại; query fallback fit một vectorizer mới từ corpus hiện tại.

Nếu corpus và cấu hình hoàn toàn không đổi, vocabulary có thể được tái tạo khớp. Nhưng nếu dữ liệu/index không đồng bộ, tọa độ và IDF có thể khác.

Nguyên tắc đáng học vẫn là:

> Ma trận tài liệu và vector query phải dùng cùng không gian đặc trưng và thống kê đã fit.

Không cần học hết nhánh mới để hiểu TF-IDF của bản chính. Khi thuyết trình, hãy nói rõ đang giải thích phiên bản nào.

---

<a id="hoi-dap"></a>

## 29. Câu hỏi thường gặp và câu trả lời dễ nhớ

### 1. TF-IDF viết tắt của gì?

Term Frequency – Inverse Document Frequency: kết hợp số lần xuất hiện trong một đoạn với độ hiếm của term trong toàn corpus.

### 2. Tên đúng là TF-IDF hay DF-IDF?

Thuật toán đang dùng là **TF-IDF**. DF là một thành phần để tính IDF, không thay thế chữ TF trong tên.

### 3. N là tổng số chunk hay cứ xét 100 chunk một lần?

N là tổng số chunk đưa vào lần fit. Code không chia mặc định thành lô 100 để tính IDF riêng.

### 4. Một PDF có 100 trang thì N=100?

Không. N bằng số chunk tạo ra và đưa vào index; một trang có thể có nhiều chunk.

### 5. Một từ lặp 100 lần trong một chunk có DF=100?

Không. DF vẫn là 1 nếu chỉ chunk đó chứa từ.

### 6. Từ có trong mọi chunk có bị xóa không?

Không bị loại chỉ vì xuất hiện ở mọi chunk. Nếu feature được giữ trong vocabulary, IDF của nó bằng 1; không có stop-word list được khai báo. Việc giới hạn `max_features` vẫn có thể loại feature khi từ vựng quá lớn.

### 7. Câu hỏi có được đưa vào corpus khi hỏi không?

Không. Query chỉ `transform`; không tăng N hay cập nhật IDF.

### 8. Tại sao không `fit_transform` riêng câu hỏi?

Vì sẽ học từ vựng/IDF riêng, không còn bảo đảm cùng tọa độ với ma trận chunk.

### 9. 50.000 là số chunk?

Không. Là giới hạn số feature/cột, tính cả unigram và bigram.

### 10. Top-k=5 có nghĩa chỉ so với 5 chunk?

Không. Code tính score trên index rồi lấy tối đa 5 kết quả đứng đầu.

### 11. Score=0.8 có nghĩa câu trả lời đúng 80%?

Không. Score là độ tương đồng hoặc điểm hybrid phục vụ xếp hạng.

### 12. Vì sao score có thể lớn hơn 1?

Vì điểm trả về đã cộng graph; một chunk có thể nhận nhiều khoản cộng.

### 13. TF-IDF có tự hiểu viết tắt LNST?

Không. Dự án có bảng alias chuyển LNST thành cụm từ chuẩn trước khi tính vector.

### 14. TF-IDF có cần Ollama?

Không. Nó dùng thư viện chạy local. Bản local hiện tại còn bỏ qua yêu cầu gọi LLM trong đường `ask`.

### 15. TF-IDF có phải embedding không?

Theo nghĩa rộng, nó là cách biểu diễn text thành vector. Nhưng trong ngữ cảnh dự án, nên gọi rõ “vector TF-IDF từ vựng”, để phân biệt với neural/dense semantic embeddings.

### 16. Có cần huấn luyện không?

Cần fit vocabulary/IDF từ corpus. Không cần huấn luyện mạng neural hoặc fine-tune trên tập câu hỏi–đáp án.

### 17. Khi thêm tài liệu thì chuyện gì xảy ra?

`upsert` lưu danh sách chunk cập nhật rồi `_fit` lại toàn corpus. N, DF, IDF và ranking có thể đổi.

### 18. Tại sao từ hiếm không luôn thắng?

Điểm còn phụ thuộc TF, chuẩn hóa L2, feature chung với query, bigram, graph và bộ lọc.

### 19. Từ không có trong vocabulary thì sao?

Không được cấp cột mới tại thời điểm query; nó bị bỏ qua. Nếu không còn feature nào, query vector bằng 0.

### 20. TF-IDF có hiểu số âm và đơn vị tiền?

Không theo nghĩa nghiệp vụ. Tokenizer nhìn chuỗi chữ/số; parser và calculator phải xử lý số, dấu âm, đơn vị, kỳ.

### 21. Tại sao hỏi chính sách không nên ưu tiên dòng số?

Một dòng số có thể chứa đúng tên chỉ tiêu nhưng không chứa điều kiện ghi nhận. Vì vậy code bỏ graph boost và hạ điểm financial_rows cho các câu hỏi chính sách được nhận diện.

### 22. Nếu tìm sai thì chỉ cần tăng top-k?

Không. Tăng top-k có thể giúp lấy thêm bằng chứng nhưng cũng thêm nhiễu. Phải kiểm tra OCR, token, alias, metadata và score breakdown.

### 23. Lọc một báo cáo có làm IDF chỉ tính trong báo cáo đó?

Không với code hiện tại. Bộ lọc nằm sau bước tính điểm trên chỉ mục toàn cục.

### 24. TF-IDF tự tạo trích dẫn PDF?

Không. Nó chọn chunk; trang, file, bbox được lấy từ metadata đã lưu.

### 25. Có thể nói hệ thống không dùng “hộp đen” không?

Có thể nói riêng nhánh truy xuất TF-IDF + luật có công thức và cách cộng điểm kiểm tra được. Nhưng vẫn phải kiểm chứng dữ liệu OCR/trích xuất và toàn pipeline. Tính giải thích được không đồng nghĩa tự động đúng.

---

<a id="bai-tap"></a>

## 30. Bài tập tự làm — có đáp án

### Bài 1: phân biệt TF và DF

```text
C1: no no tai san
C2: tai san
C3: doanh thu
```

Hỏi TF(`no`,C1), DF(`no`), DF(`tai`) và N.

**Đáp án:** lần lượt 2, 1, 2 và 3. Dùng chế độ unigram.

### Bài 2: tính IDF

Với corpus bài 1, tính IDF(`no`) và IDF(`tai`) theo công thức đang dùng.

**Đáp án:**

```text
IDF(no)  = ln(4/2)+1 ≈ 1.693147
IDF(tai) = ln(4/3)+1 ≈ 1.287682
```

`no` hiếm hơn về số chunk nên IDF lớn hơn.

### Bài 3: chuẩn hóa

Vector thô `[3,4,0]` có vector L2-normalized là gì?

**Đáp án:** `[0.6,0.8,0]`.

### Bài 4: cosine

Hai vector đã chuẩn hóa:

```text
d = [0.6,0.8,0]
q = [1,0,0]
```

Tính score.

**Đáp án:** `0.6*1 + 0.8*0 + 0*0 = 0.6`.

### Bài 5: query ngoài từ vựng

Vocabulary chỉ có `doanh`, `thu`, `tang`. Query là `xyz`.

**Đáp án:** vector query toàn 0, điểm TF-IDF thuần của mọi chunk là 0. Query không tự làm từ vựng có thêm `xyz`.

### Bài 6: điểm hybrid

Chunk có TF-IDF=0.42, nhận hai khoản boost:

```text
Fact A: match=1
Fact B: match=0.8
```

Chưa có bộ lọc khác, điểm mới là bao nhiêu?

**Đáp án:**

```text
0.42 + 0.35*1 + 0.35*0.8 = 1.05
```

Đây là lý do không coi điểm hybrid là xác suất.

### Bài 7: N và lọc báo cáo

Index có báo cáo A 80 chunk, B 120 chunk. Query chỉ tìm A, `top_k=5`.

**Đáp án:**

```text
N dùng fit = 200
Số kết quả trả tối đa = 5
B bị đặt điểm về 0 trước khi trả kết quả
IDF không tự fit lại trên 80 chunk của A
```

### Bài 8: tìm lỗi trong đoạn code

```python
documents = vectorizer.fit_transform(chunks)
query_vector = TfidfVectorizer().fit_transform([question])
scores = documents @ query_vector.T
```

**Đáp án:** query dùng một vectorizer fit riêng. Hai phía có thể khác số cột; kể cả tình cờ cùng số cột, ý nghĩa cột và IDF có thể khác nhau.

Cách đúng về nguyên tắc:

```python
documents = vectorizer.fit_transform(chunks)
query_vector = vectorizer.transform([question])
scores = documents @ query_vector.T
```

### Bài 9: đọc code thật

Tìm trong `retrieval.py`:

1. Dòng nào học corpus?
2. Dòng nào biến query mà không fit lại?
3. Dòng nào cộng graph?
4. Dòng nào loại mục lục?
5. Dòng nào chỉ giữ score > 0?

**Đáp án ở snapshot này:** lần lượt dòng 39, 62, 81, 89–91, 107.
Số dòng có thể đổi sau các lần chỉnh code; tên biểu thức/hàm mới là dấu mốc bền vững.

---

## 31. Từ điển ngắn: khi gặp thuật ngữ thì tra ở đây

| Thuật ngữ | Nghĩa cần nhớ |
| --- | --- |
| Lexical retrieval | Tìm kiếm dựa trên từ/cụm từ xuất hiện |
| Semantic embedding | Vector do mô hình biểu diễn để hỗ trợ quan hệ ngữ nghĩa; không đồng nghĩa TF-IDF |
| Unigram | Feature gồm một token |
| Bigram | Feature gồm hai token liền nhau |
| TF | Số lần term xuất hiện trong một chunk, với cấu hình hiện tại |
| DF | Số chunk có term |
| IDF | Trọng số giảm khi term phổ biến trong nhiều chunk |
| L2 norm | Căn tổng bình phương các phần tử |
| Dot product | Nhân phần tử tương ứng rồi cộng |
| Cosine similarity | Tích vô hướng chia tích độ dài hai vector |
| OOV | Feature ngoài vocabulary đã fit |
| Sparse matrix | Ma trận lưu hiệu quả các phần tử khác 0 |
| Dense array | Mảng lưu các ô, kể cả 0; không tự có nghĩa là neural embedding |
| Broadcasting | NumPy tự áp dụng một dãy số theo chiều tương thích, như nhân IDF cho mọi hàng |
| Hyperparameter | Tham số cấu hình do người xây hệ thống chọn, như n-gram hoặc hệ số boost |
| Metadata | Dữ liệu mô tả nguồn: trang, document ID, bbox, phương pháp trích xuất |
| Provenance | Khả năng truy ngược kết quả về nguồn |
| Baseline | Cách làm cơ sở dùng để so sánh cải tiến |
| Regression test | Test giúp phát hiện thay đổi làm hỏng hành vi đã biết |
| Ablation | Thử bỏ/thêm từng thành phần để đo đóng góp |

---

<a id="van-dap"></a>

## 32. Tờ ghi nhớ và cách trình bày khi vấn đáp

### 32.1. Tám dòng cốt lõi

```text
1. Mỗi chunk là một document của TF-IDF.
2. N = số chunk được fit; M = số feature; top_k = số kết quả lấy ra.
3. TF = số lần term xuất hiện trong chunk.
4. DF = số chunk có term, không phải tổng số lần lặp.
5. IDF = ln((N+1)/(DF+1))+1.
6. Trọng số = TF*IDF, sau đó chuẩn hóa L2.
7. Query dùng transform của cùng vectorizer; nhân ma trận để lấy cosine.
8. Dự án còn cộng graph và lọc; điểm cuối không phải xác suất đúng.
```

### 32.2. Câu trả lời khoảng một phút

> Trong dự án của em, báo cáo được đọc và chia thành các chunk. Mỗi chunk là một đơn vị lập chỉ mục TF-IDF. Bộ vectorizer xây từ vựng gồm từ đơn và cụm hai token, đếm số chunk chứa từng feature để tính IDF theo công thức có làm trơn. Với mỗi chunk, số lần xuất hiện được nhân IDF rồi chuẩn hóa L2. Khi có câu hỏi cần truy xuất, hệ thống chuẩn hóa và thay alias, dùng cùng vectorizer để transform câu hỏi, sau đó nhân với ma trận chunk để lấy điểm cosine. Điểm còn được cộng từ các quan hệ graph và áp dụng bộ lọc trước khi lấy top-k. Nguồn trích dẫn được lấy từ metadata của chunk. TF-IDF chỉ xếp hạng bằng chứng, không tự tính số liệu hay bảo đảm câu trả lời đúng.

### 32.3. Nếu thầy hỏi “học máy nằm ở đâu?”

> TF-IDF là bước biểu diễn văn bản bằng thống kê học từ corpus: vocabulary và IDF. Nó không phải mạng neural được fine-tune. Phần đang giải thích chạy CPU và có thể kiểm tra từng trọng số. Em phân biệt bước truy xuất này với OCR đọc ảnh thành chữ khi được bật trong bản chính; còn đường hỏi đáp hiện tại không gọi LLM.

### 32.4. Nếu thầy hỏi “vì sao không chỉ tìm từ khóa?”

> TF-IDF không chỉ kiểm tra có/không. Nó kết hợp số lần xuất hiện với độ phổ biến trong toàn corpus, biểu diễn nhiều feature cùng lúc và chuẩn hóa trước khi so độ tương đồng. Nhưng nó vẫn là truy xuất từ vựng, không phải hiểu nghĩa đầy đủ.

### 32.5. Nếu thầy hỏi “có chứng minh được vì sao chọn đoạn này không?”

> Có thể tách tổng điểm thành đóng góp của từng feature: trọng số feature trong chunk nhân trọng số feature trong query. Sau đó giải thích riêng graph boost và các quy tắc lọc. Demo của em kiểm tra tổng các đóng góp bằng điểm cosine thực tế.

### 32.6. Ba câu không nên nói

| Không nên nói | Nên nói |
| --- | --- |
| “Score 0.9 nghĩa là chính xác 90%.” | “Đây là điểm dùng xếp hạng, không phải xác suất đúng.” |
| “TF-IDF hiểu mọi từ đồng nghĩa.” | “Chúng em bổ sung một số alias bằng luật; TF-IDF không tự học mọi đồng nghĩa.” |
| “Tất cả câu hỏi đều qua TF-IDF rồi Ollama.” | “Bản local hiện tại thử trả lời xác định trước, sau đó mới fallback tìm đoạn trích.” |

---

<a id="nguon"></a>

## 33. Bản đồ code, kiểm chứng và nguồn tham khảo

### 33.1. Code dự án nên mở theo thứ tự

| Thứ tự | File / điểm vào | Mục đích |
| --- | --- | --- |
| 1 | [models.py](C:/DXD/Multimodal-RAG-Financial/src/financial_rag/models.py:31) | Hiểu Chunk và Evidence |
| 2 | [config.py](C:/DXD/Multimodal-RAG-Financial/src/financial_rag/config.py:20) | Size, overlap, top-k |
| 3 | [ingestion.py](C:/DXD/Multimodal-RAG-Financial/src/financial_rag/ingestion.py:190) | Tạo chunk |
| 4 | [retrieval.py](C:/DXD/Multimodal-RAG-Financial/src/financial_rag/retrieval.py:13) | Toàn bộ TF-IDF + graph ranking |
| 5 | [answers.py](C:/DXD/Multimodal-RAG-Financial/src/financial_rag/answers.py:48) | Chuẩn hóa/alias và nhánh trả lời xác định |
| 6 | [financial_tables.py](C:/DXD/Multimodal-RAG-Financial/src/financial_rag/financial_tables.py:14) | Hàm fold |
| 7 | [graph.py](C:/DXD/Multimodal-RAG-Financial/src/financial_rag/graph.py:62) | Quan hệ Fact–Metric–Chunk |
| 8 | [service.py](C:/DXD/Multimodal-RAG-Financial/src/financial_rag/service.py:82) | Khi nào retriever được gọi |
| 9 | [explain_tfidf.py](C:/DXD/Multimodal-RAG-Financial/scripts/explain_tfidf.py:51) | Thực hành từng phép tính |
| 10 | [test_tfidf_lesson.py](C:/DXD/Multimodal-RAG-Financial/tests/test_tfidf_lesson.py) | Kiểm chứng công thức và cấu hình demo |

### 33.2. Các kiểm tra đã thực hiện cho tài liệu

- Đọc mã nguồn local, bao gồm các thay đổi deterministic QA chưa commit.
- Kiểm tra cấu hình và mặc định scikit-learn trong `.venv` của dự án.
- Chạy analyzer để xác minh cách xử lý tiếng Việt, `đ`, số và n-gram.
- Đối chiếu ví dụ tính tay với vectorizer thật.
- Chạy demo ở cả hai chế độ với `--verify`.
- Bộ test demo có 11 trường hợp, gồm DF không đếm lặp, query OOV, IDF không đổi theo query, CLI tiếng Việt và cấu hình khớp retriever.
- Chạy chung test demo, test alias retrieval và test deterministic answering: **42 passed**. Có một cảnh báo deprecation từ thư viện test HTTP, không có test lỗi.
- Không thay đổi code nghiệp vụ, không refit/ghi lại index sản xuất để làm tài liệu, không cần bật Ollama.

### 33.3. Tài liệu thư viện để đọc thêm

- [TfidfVectorizer: tham số và phương thức](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html).
- [Hướng dẫn trích xuất đặc trưng và TF-IDF](https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting).
- [Cosine similarity](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html).

Phần hướng dẫn và ví dụ trong tài liệu này được diễn giải cho mã nguồn của dự án. Các con số ví dụ được tính từ corpus giả lập đã nêu, không lấy làm kết luận về báo cáo tài chính thật.

**Nếu chỉ nhớ một câu:** TF-IDF biến chunk và câu hỏi thành vector từ vựng có trọng số để tìm nội dung liên quan; việc trả lời đúng còn cần dữ liệu nguồn đúng, ràng buộc nghiệp vụ, phép tính và trích dẫn được kiểm chứng.
