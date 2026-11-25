# Phạm Vi Hỏi Đáp Tất Định (Deterministic QA Guide)

Tài liệu mô tả phạm vi câu hỏi được xử lý trực tiếp bằng bộ luật tất định tại `src/financial_rag/answers.py` và các trạng thái kết quả của hệ thống.

---

## 1. Các Trạng Thái Kết Quả (`answer_status`)

| Trạng thái | Ý nghĩa | Hành động của hệ thống |
|---|---|---|
| `answered` | Trả lời từ dữ kiện bảng chuẩn, phép tính `Decimal`, hoặc tổng hợp có trích dẫn đã qua hậu kiểm từ Colab GPU. | Hiển thị kết quả kèm trích dẫn `[1]`, `[2]` bấm để nhảy tới trang PDF và vùng `bbox`. |
| `evidence_only` | Câu hỏi chưa có quy tắc khớp hoàn toàn hoặc thiếu kiểm chứng (và chưa bật/không dùng Colab GPU). | Chỉ hiển thị đoạn trích liên quan; ghi rõ chưa kết luận. |
| `needs_clarification` | Câu hỏi mơ hồ (nhiều báo cáo, xung đột kỳ, trộn riêng/hợp nhất, so sánh nền âm/khác quý). | Yêu cầu người dùng chọn báo cáo hoặc làm rõ kỳ. |
| `insufficient_evidence` | Không tìm thấy bằng chứng hoặc dữ kiện không vượt qua kiểm tra provenance/chất lượng. | Dừng trả số, báo rõ lý do thiếu hoặc sai lệch. |

---

## 2. Các Chỉ Tiêu & Tỷ Số Hỗ Trợ Trong `answers.py`

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
   - `60`: Lợi nhuận sau thuế (`LNST`, `lãi ròng`)

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

4. **Tỷ số tài chính hỗ trợ trực tiếp**:
   - Biên lợi nhuận gộp (`20 / 10`), Biên lợi nhuận ròng (`60 / 10`)
   - Hệ số nợ trên vốn chủ sở hữu (`300 / 400`)
   - Khả năng thanh toán hiện hành (`100 / 310`)

---

## 3. Phép Tính & Ràng Buộc An Toàn (`calculator.py`)

- **Tăng trưởng cùng kỳ (`growth`)**: $(\text{Kỳ sau} - \text{Kỳ trước}) / \text{Kỳ trước} \times 100\%$ (từ chối khi nền âm hoặc chia cho `0`).
- **Chênh lệch (`difference`)** & **Tỷ số (`ratio`)**: Hỗ trợ tự động quy đổi cùng hệ tiền tệ (`VND`, `nghìn`, `triệu`, `tỷ`, `nghìn tỷ`) bằng `Decimal`.
- **Ràng buộc**: Không trộn phạm vi (`separate` vs `consolidated`), không trộn loại kỳ (`instant` vs `quarter`), không tính trên fact `needs_review`.
