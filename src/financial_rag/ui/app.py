"""Interactive Gradio Web UI for demoing the Financial RAG system."""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import pandas as pd

from financial_rag.config import settings
from financial_rag.db.repository import FinancialRepository
from financial_rag.reasoning.engine import FinancialRAGEngine


EXAMPLE_QUESTIONS = [
    "Doanh thu thuần năm 2025 của Công ty Nhựa An Phát Xanh (AAA) là bao nhiêu?",
    "Lợi nhuận sau thuế năm 2025 của AAA đạt bao nhiêu và tăng hay giảm so với năm 2024?",
    "Biên lợi nhuận gộp của AAA năm 2025 là bao nhiêu phần trăm?",
    "Tỷ lệ nợ trên vốn chủ sở hữu (D/E) của AAA tại ngày 31/12/2025 là bao nhiêu?",
    "What was NVIDIA's total revenue and net income for fiscal year 2025?",
    "What was NVIDIA's gross profit margin for fiscal year 2025?",
    "So sánh biên lợi nhuận ròng 6 tháng đầu năm 2026 giữa Hòa Phát (HPG) và Vinamilk (VNM)?",
    "Ý kiến của kiểm toán viên độc lập về báo cáo tài chính của AAA là gì?",
    "Dự báo giá cổ phiếu AAA trong 3 tháng tới sẽ tăng hay giảm?",
    "Doanh thu của Công ty Bất động sản Novaland năm 2025 là bao nhiêu?",
]


def create_gradio_app(engine: FinancialRAGEngine, repo: FinancialRepository):
    """Construct Gradio UI components and event handlers."""
    try:
        import gradio as gr
    except ImportError:
        raise ImportError("Gradio is not installed. Install with `pip install gradio`.")

    def handle_ask(query: str, history: Optional[List[Any]] = None):
        if not query or not query.strip():
            return history or [], "", "Vui lòng nhập câu hỏi.", ""

        response = engine.answer_question(query)

        ans_md = response.answer
        citations_md = "\n".join([f"- **Nguồn**: `{c}`" for c in response.citations]) or "Không có trích dẫn cụ thể."
        
        # Meta info
        meta_info = f"**Loại câu hỏi**: `{response.question_type}`\n"
        if response.slots.get("ticker"):
            meta_info += f"- **Công ty**: `{response.slots.get('ticker')}`\n"
        if response.calculation:
            meta_info += f"- **Tính toán Python**: `{response.calculation.get('result', '')}`\n"
        if response.verification:
            is_v = response.verification.get("is_verified", True)
            v_badge = "✅ Số liệu khớp 100% với báo cáo" if is_v else "⚠️ Có cảnh báo đối soát"
            meta_info += f"- **Kiểm tra đối soát**: {v_badge}\n"
            if response.verification.get("warning"):
                meta_info += f"  > {response.verification.get('warning')}\n"

        history = list(history or [])
        # Support both Gradio 5+ (messages format: list of dicts) and legacy Gradio (tuples)
        if getattr(chatbot, "type", None) == "tuples" or (history and isinstance(history[0], (list, tuple))):
            history.append((query, ans_md))
        else:
            history.append({"role": "user", "content": query})
            history.append({"role": "assistant", "content": ans_md})

        return history, "", citations_md, meta_info

    def load_facts(ticker_filter: str):
        ticker = None if ticker_filter == "TẤT CẢ" else ticker_filter
        facts = repo.query_facts(ticker=ticker)
        data = []
        for f in facts:
            data.append({
                "Mã CK": f["ticker"],
                "Năm": f["fiscal_year"],
                "Kỳ": f["period_type"],
                "Chỉ tiêu gốc": f["line_item"],
                "Chỉ tiêu chuẩn hóa": f.get("metric_code") or "N/A",
                "Giá trị": f"{f['value']:,.0f}" if f['value'] is not None else "0",
                "Chuỗi gốc": f.get("value_raw", ""),
                "Đơn vị": f.get("currency", "VND"),
                "Trang": f.get("page", 1),
            })
        return pd.DataFrame(data)

    with gr.Blocks(title="Financial RAG - Hỏi Đáp BCTC", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 📊 Hệ thống RAG Hỏi Đáp Báo Cáo Tài Chính
            *Hệ thống RAG song ngữ (Việt - Anh) kết hợp bảng số liệu SQLite có cấu trúc, tìm kiếm Hybrid (BGE-M3 + FTS5), tính toán tài chính Python và Qwen2.5-3B-Instruct.*
            """
        )

        with gr.Tabs():
            # Tab 1: Chat QA
            with gr.Tab("💬 Hỏi Đáp Báo Cáo (QA)"):
                with gr.Row():
                    with gr.Column(scale=7):
                        try:
                            chatbot = gr.Chatbot(height=520, label="Hội thoại BCTC", type="messages")
                        except TypeError:
                            chatbot = gr.Chatbot(height=520, label="Hội thoại BCTC")
                        with gr.Row():
                            query_input = gr.Textbox(
                                placeholder="Đặt câu hỏi về BCTC (ví dụ: Doanh thu thuần của AAA năm 2025 là bao nhiêu?)...",
                                label="Câu hỏi của bạn",
                                scale=8,
                            )
                            send_btn = gr.Button("Gửi câu hỏi", variant="primary", scale=2)

                        gr.Examples(
                            examples=EXAMPLE_QUESTIONS,
                            inputs=query_input,
                            label="💡 Câu hỏi mẫu (đủ các loại: Tra cứu, Tính toán, So sánh, Giải thích, Bẫy)",
                        )

                    with gr.Column(scale=4):
                        gr.Markdown("### 🔍 Nguồn trích dẫn (Provenance)")
                        citations_box = gr.Markdown("Chưa có trích dẫn.")
                        gr.Markdown("### ⚙️ Đối soát & Thông số (Audit & Verification)")
                        meta_box = gr.Markdown("Chưa có thông tin.")

                send_btn.click(
                    handle_ask,
                    inputs=[query_input, chatbot],
                    outputs=[chatbot, query_input, citations_box, meta_box],
                )
                query_input.submit(
                    handle_ask,
                    inputs=[query_input, chatbot],
                    outputs=[chatbot, query_input, citations_box, meta_box],
                )

            # Tab 2: Fact Explorer
            with gr.Tab("📋 Kho Số Liệu Có Cấu Trúc (Facts Explorer)"):
                gr.Markdown("Dữ liệu chỉ tiêu tài chính đã chuẩn hóa trong bảng `fact` của SQLite.")
                with gr.Row():
                    companies = repo.get_all_companies()
                    ticker_choices = ["TẤT CẢ"] + [c["ticker"] for c in companies]
                    ticker_dropdown = gr.Dropdown(choices=ticker_choices, value="TẤT CẢ", label="Chọn công ty:")
                    refresh_btn = gr.Button("Làm mới dữ liệu", variant="secondary")

                facts_table = gr.DataFrame(value=load_facts("TẤT CẢ"), label="Danh sách số liệu đã nạp")
                refresh_btn.click(load_facts, inputs=[ticker_dropdown], outputs=[facts_table])
                ticker_dropdown.change(load_facts, inputs=[ticker_dropdown], outputs=[facts_table])

            # Tab 3: Upload Report PDF
            with gr.Tab("📤 Nạp BCTC Mới (Upload PDF)"):
                gr.Markdown(
                    """
                    ### 📤 Nạp Báo cáo tài chính trực tiếp (PDF / TXT)
                    *Tải file trực tiếp vào đây mà không cần tạo thư mục mã chứng khoán hay chỉnh sửa file cấu hình thủ công.*
                    """
                )
                with gr.Row():
                    with gr.Column(scale=6):
                        pdf_file = gr.File(label="Chọn hoặc kéo thả file BCTC (.pdf / .txt)", file_types=[".pdf", ".txt", ".md"])
                        with gr.Row():
                            up_ticker = gr.Textbox(label="Mã cổ phiếu (Ticker)", placeholder="VD: FPT, VCB, MWG...")
                            up_company = gr.Textbox(label="Tên công ty (Tùy chọn)", placeholder="VD: Công ty Cổ phần FPT")
                        with gr.Row():
                            up_year = gr.Number(label="Năm tài chính", value=2025, precision=0)
                            up_period = gr.Dropdown(choices=["FY", "H1", "Q1", "Q2", "Q3", "Q4"], value="FY", label="Kỳ báo cáo")
                        with gr.Row():
                            up_scale = gr.Dropdown(
                                choices=[
                                    ("Đồng (1)", 1),
                                    ("Nghìn đồng (1,000)", 1000),
                                    ("Triệu đồng (1,000,000)", 1000000),
                                    ("Tỷ đồng (1,000,000,000)", 1000000000),
                                ],
                                value=1000000,
                                label="Đơn vị tính số tiền trong BCTC",
                            )
                            up_currency = gr.Dropdown(choices=["VND", "USD"], value="VND", label="Tiền tệ")
                            up_scope = gr.Dropdown(choices=["consolidated", "separate"], value="consolidated", label="Phạm vi")
                        upload_btn = gr.Button("🚀 Bắt đầu Nạp & Xử Lý Báo Cáo", variant="primary")

                    with gr.Column(scale=4):
                        gr.Markdown("### 📊 Kết quả xử lý (Status)")
                        upload_status = gr.Markdown("Chưa nạp file nào.")

                def handle_upload(file_obj, ticker, company, year, period, scale, curr, sc):
                    if not file_obj:
                        return "❌ Vui lòng chọn hoặc kéo thả file PDF cần nạp."
                    if not ticker or not str(ticker).strip():
                        return "❌ Vui lòng nhập mã cổ phiếu (Ticker)."

                    from financial_rag.parser.ingest_pipeline import ingest_single_file

                    # file_obj is either a filepath string or a tempfile object depending on Gradio version
                    file_path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
                    try:
                        res = ingest_single_file(
                            file_path=file_path,
                            ticker=ticker,
                            fiscal_year=int(year),
                            period_type=period,
                            unit_scale=int(scale),
                            currency=curr,
                            scope=sc,
                            company_name_vi=company,
                            repo=repo,
                        )
                        validation_badge = "✅ Cân đối hoàn hảo ($Tổng\\ Tài\\ Sản = Nợ + VCSH$)" if res['is_valid'] else f"⚠️ Cảnh báo: {res['validation_errors']}"
                        return f"""
### ✅ Nạp thành công báo cáo **{res['ticker']}** ({res['period_type']}/{res['fiscal_year']})!
- **Số trang đã xử lý**: `{res['pages_parsed']}` trang
- **Đoạn văn bản đánh chỉ mục (Chunks)**: `{res['chunks_indexed']}`
- **Số liệu có cấu trúc (`fact`)**: `{res['facts_extracted']}` chỉ tiêu
- **FAISS Vector Index**: `{res['total_vector_indexed']}` vectors
- **Kiểm tra cân đối kế toán**: {validation_badge}

*Bây giờ bạn có thể sang tab **💬 Hỏi Đáp Báo Cáo (QA)** để đặt câu hỏi về doanh nghiệp này ngay lập tức!*
"""
                    except Exception as e:
                        return f"❌ Lỗi khi nạp báo cáo: `{str(e)}`"

                upload_btn.click(
                    handle_upload,
                    inputs=[pdf_file, up_ticker, up_company, up_year, up_period, up_scale, up_currency, up_scope],
                    outputs=[upload_status],
                )

            # Tab 4: System Design
            with gr.Tab("🏛️ Thiết Kế Kiến Trúc"):
                gr.Markdown(
                    """
                    ### Kiến Trúc Hệ Thống (Theo tài liệu phân tích thiết kế)
                    
                    1. **Luồng Offline (Ingestion)**:
                       - PDF BCTC (Việt/Anh) -> Parse bảng & văn bản kèm số trang.
                       - Chuẩn hóa số kiểu Việt (`1.234.567,89`) và Anh (`1,234,567.89`), số âm `(1.234)`.
                       - Chunk theo ngữ cảnh: `[CTY | KỲ | PHẠM VI | BẢNG | ĐƠN VỊ]`.
                       - Lưu trữ 2 nhánh:
                         - **(a) Bảng SQLite**: `company`, `report`, `metric` (glossary Vi-En), `fact` (số liệu có cấu trúc), `chunk`, `chunk_fts`.
                         - **(b) FAISS Dense Index**: Embedding đa ngôn ngữ BGE-M3.
                    
                    2. **Luồng Online (Answering)**:
                       - Câu hỏi -> Slot Extraction JSON -> Mở rộng truy vấn từ điển Việt - Anh.
                       - **Hỏi số liệu & tính toán**: Tra trực tiếp SQLite `fact` -> Code Python tính toán chuẩn xác (Biên LN, D/E, YoY) -> Qwen2.5-3B diễn đạt.
                       - **Hỏi giải thích**: Hybrid Search (BGE-M3 + FTS5) có lọc metadata -> Top 3-5 chunk -> Qwen2.5-3B trả lời kèm trích dẫn trang.
                       - **Câu hỏi bẫy / ngoài phạm vi**: Từ chối lịch sự, triệt tiêu ảo giác (hallucination).
                       - **Fact Verification**: Đối soát mọi con số trong câu trả lời với số liệu gốc.
                    """
                )

    return demo
