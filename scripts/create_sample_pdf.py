"""Creates a sample PDF containing a financial table and a chart for testing annotation workflows."""

from pathlib import Path

import pymupdf


def generate_sample_pdf(output_path: Path):
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)  # A4 size in points

    # Page Title
    page.insert_text(pymupdf.Point(50, 60), "BÁO CÁO KẾT QUẢ KINH DOANH VÀ CƠ CẤU DOANH THU", fontsize=16, fontname="helv", color=(0.1, 0.1, 0.3))
    page.insert_text(pymupdf.Point(50, 85), "Công ty Cổ phần Mẫu Demo (Năm 2024)", fontsize=11, fontname="helv", color=(0.4, 0.4, 0.4))

    # 1. DRAW A FINANCIAL TABLE (Table Box)
    table_rect = pymupdf.Rect(50, 110, 545, 290)
    page.draw_rect(table_rect, color=(0.2, 0.2, 0.2), width=1)
    # Header fill
    header_rect = pymupdf.Rect(50, 110, 545, 140)
    page.draw_rect(header_rect, color=(0.9, 0.9, 0.95), fill=(0.9, 0.9, 0.95))

    # Table Header text
    page.insert_text(pymupdf.Point(60, 130), "Chỉ tiêu tài chính (VND)", fontsize=10, fontname="helv")
    page.insert_text(pymupdf.Point(340, 130), "Mã số", fontsize=10, fontname="helv")
    page.insert_text(pymupdf.Point(420, 130), "Năm 2024", fontsize=10, fontname="helv")

    # Table Rows
    rows = [
        ("1. Doanh thu bán hàng và CCDV", "01", "150.000.000.000"),
        ("2. Các khoản giảm trừ doanh thu", "02", "5.000.000.000"),
        ("3. Doanh thu thuần", "10", "145.000.000.000"),
        ("4. Giá vốn hàng bán", "11", "90.000.000.000"),
        ("5. Lợi nhuận gộp", "20", "55.000.000.000"),
    ]

    y_pos = 140
    for title, code, val in rows:
        y_pos += 30
        page.draw_line(pymupdf.Point(50, y_pos - 10), pymupdf.Point(545, y_pos - 10), color=(0.85, 0.85, 0.85), width=0.5)
        page.insert_text(pymupdf.Point(60, y_pos + 8), title, fontsize=9, fontname="helv")
        page.insert_text(pymupdf.Point(350, y_pos + 8), code, fontsize=9, fontname="helv")
        page.insert_text(pymupdf.Point(420, y_pos + 8), val, fontsize=9, fontname="helv")

    # 2. DRAW A BAR CHART (Vector Graphic Cluster)
    page.insert_text(pymupdf.Point(50, 340), "Biểu đồ 1: Tăng trưởng doanh thu và lợi nhuận qua các năm (Tỷ VND)", fontsize=12, fontname="helv", color=(0.1, 0.1, 0.3))

    chart_bg = pymupdf.Rect(50, 360, 545, 580)
    page.draw_rect(chart_bg, color=(0.8, 0.8, 0.8), fill=(0.98, 0.98, 0.98), width=0.8)

    # Grid lines
    for gy in [400, 450, 500, 550]:
        page.draw_line(pymupdf.Point(60, gy), pymupdf.Point(535, gy), color=(0.9, 0.9, 0.9), width=0.5)

    # Bars (vector rects)
    bars = [
        (80, 550, 110, 480, (0.2, 0.4, 0.8)),   # 2022 Rev
        (115, 550, 140, 510, (0.2, 0.7, 0.4)),  # 2022 Profit
        (180, 550, 210, 440, (0.2, 0.4, 0.8)),  # 2023 Rev
        (215, 550, 240, 490, (0.2, 0.7, 0.4)),  # 2023 Profit
        (280, 550, 310, 390, (0.2, 0.4, 0.8)),  # 2024 Rev
        (315, 550, 340, 460, (0.2, 0.7, 0.4)),  # 2024 Profit
    ]
    for x0, y0, x1, y1, color in bars:
        page.draw_rect(pymupdf.Rect(x0, y1, x1, y0), fill=color, color=color)

    page.insert_text(pymupdf.Point(90, 565), "2022", fontsize=9, fontname="helv")
    page.insert_text(pymupdf.Point(190, 565), "2023", fontsize=9, fontname="helv")
    page.insert_text(pymupdf.Point(290, 565), "2024", fontsize=9, fontname="helv")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_path))
    doc.close()
    print(f"Sample PDF created at '{output_path}'")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parents[1]
    generate_sample_pdf(base_dir / "samples" / "demo_report.pdf")
