# Phạm Vi Hỏi Đáp Tất Định (Deterministic QA Guide)

Tài liệu quy định chi tiết phạm vi câu hỏi được hỗ trợ, cơ chế cổng từ vựng (vocabulary gating) và các trạng thái trả lời của hệ thống.

---

## 1. Các Trạng Thái Kết Quả (Answer Status)

Mỗi phản hồi từ `/ask` trả về một trong các trạng thái:

| Trạng thái (`answer_status`) | Ý nghĩa | Hành động của hệ thống |
|---|---|---|
| `answered` | Câu hỏi đã được trả lời chính xác bằng dữ kiện bảng hoặc phép tính toán học có nguồn. | Trả lời trực tiếp kèm trích dẫn số trang, mã chỉ tiêu và tọa độ box. |
| `evidence_only` | Câu hỏi ngoài phạm vi số liệu bảng (VD: câu hỏi về chính sách, ban điều hành). | Chỉ trả về các đoạn trích liên quan tìm được; ghi rõ chưa xác minh kết luận. |
| `needs_clarification` | Câu hỏi bị mơ hồ (thiếu báo cáo, hỏi nhiều kỳ xung đột, hỏi phạm vi riêng lẫn hợp nhất). | Yêu cầu người dùng chọn lại báo cáo hoặc làm rõ kỳ kế toán. |
| `insufficient_evidence` | Không tìm thấy bằng chứng phù hợp trong tài liệu đã nạp. | Thông báo thiếu bằng chứng, từ chối đoán mò. |

---

## 2. Các Chỉ Tiêu Được Hỗ Trợ (Supported Intents)

Hệ thống hỗ trợ các chỉ tiêu chuẩn hóa theo Thông tư 200:

1. **Báo cáo Kết quả Kinh doanh (`income`)**:
   - `01`: Doanh thu bán hàng và cung cấp dịch vụ
   - `10`: Doanh thu thuần
   - `11`: Giá vốn hàng bán
   - `20`: Lợi nhuận gộp
   - `22`: Doanh thu hoạt động tài chính
   - `23`: Chi phí tài chính
   - `24`: Chi phí lãi vay
   - `25`: Chi phí bán hàng
   - `26`: Chi phí quản lý doanh nghiệp
   - `30`: Lợi nhuận thuần từ hoạt động kinh doanh
   - `50`: Lợi nhuận trước thuế
   - `60`: Lợi nhuận sau thuế

2. **Bảng Cân đối Kế toán (`balance`)**:
   - `100`: Tài sản ngắn hạn
   - `110`: Tiền và các khoản tương đương tiền
   - `140`: Hàng tồn kho
   - `200`: Tài sản dài hạn
   - `280`: Tổng tài sản
   - `300`: Nợ phải trả
   - `400`: Vốn chủ sở hữu
   - `440`: Tổng nguồn vốn

3. **Báo cáo Lưu chuyển Tiền tệ (`cashflow`)**:
   - `20`: Lưu chuyển tiền thuần từ hoạt động kinh doanh
   - `30`: Lưu chuyển tiền thuần từ hoạt động đầu tư
   - `40`: Lưu chuyển tiền thuần từ hoạt động tài chính
   - `50`: Lưu chuyển tiền thuần trong kỳ
   - `60`: Tiền đầu kỳ
   - `70`: Tiền cuối kỳ

---

## 3. Các Phép Tính Số Học Được Phép

- **Tăng trưởng cùng kỳ (Growth)**: $(\text{Giá trị kỳ sau} - \text{Giá trị kỳ trước}) / \text{Giá trị kỳ trước} \times 100\%$ (Chặn tính khi nền âm).
- **Chênh lệch tuyệt đối (Difference)**: $\text{Giá trị kỳ sau} - \text{Giá trị kỳ trước}$.
- **Tỷ số tài chính (Ratio)**: Biên lợi nhuận gộp, Biên lợi nhuận ròng.
- **Ràng buộc an toàn**: Chỉ cho phép tính hai fact thuộc cùng doanh nghiệp, cùng báo cáo, cùng đơn vị tiền tệ và cùng phạm vi (riêng hoặc hợp nhất).
