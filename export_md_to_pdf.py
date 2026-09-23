"""
Script chuyển đổi Markdown (.md) sang PDF chất lượng cao.
Sử dụng Mistune + Pygments + Mermaid.js + KaTeX + Microsoft Edge Headless.

Sử dụng:
    python export_md_to_pdf.py                  # Xuất cả KIEN_TRUC.md và README.md
    python export_md_to_pdf.py KIEN_TRUC.md     # Xuất riêng KIEN_TRUC.pdf
    python export_md_to_pdf.py README.md        # Xuất riêng README.pdf
"""

import sys
import os

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import re
import html
import base64
import shutil
import pathlib
import subprocess
import pypdf
import mistune
from pygments import highlight
from pygments.lexers import get_lexer_by_name, TextLexer
from pygments.formatters import HtmlFormatter

def find_browser_executable() -> str:
    """Tìm đường dẫn Edge hoặc Chrome trên hệ thống Windows."""
    candidates = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        shutil.which("msedge"),
        shutil.which("chrome"),
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    raise FileNotFoundError("Không tìm thấy Microsoft Edge hoặc Google Chrome để xuất PDF.")

def image_to_base64(url: str, base_dir: pathlib.Path) -> str:
    """Chuyển đổi đường dẫn ảnh cục bộ sang Base64 Data URI để nhúng trực tiếp vào PDF."""
    if url.startswith("data:") or url.startswith("http://") or url.startswith("https://"):
        return url
    
    img_path = (base_dir / url).resolve()
    if img_path.exists() and img_path.is_file():
        suffix = img_path.suffix.lower()
        mime = "image/png"
        if suffix == ".svg":
            mime = "image/svg+xml"
        elif suffix in [".jpg", ".jpeg"]:
            mime = "image/jpeg"
        elif suffix == ".webp":
            mime = "image/webp"
        elif suffix == ".gif":
            mime = "image/gif"
        
        encoded = base64.b64encode(img_path.read_bytes()).decode("utf-8")
        return f"data:{mime};base64,{encoded}"
    return url

class ElegantPDFRenderer(mistune.HTMLRenderer):
    def __init__(self, base_dir: pathlib.Path, **kwargs):
        super().__init__(**kwargs)
        self.base_dir = base_dir

    def block_code(self, code: str, info: str = None) -> str:
        info = info.strip() if info else ""
        lang = info.split()[0] if info else ""
        
        # Nhánh Mermaid Diagram
        if lang.lower() == "mermaid":
            # Unescape nếu cần
            clean_code = code.strip()
            return f'<div class="mermaid-wrapper"><pre class="mermaid">\n{clean_code}\n</pre></div>\n'
        
        # Nhánh Syntax Highlighting
        try:
            lexer = get_lexer_by_name(lang)
        except Exception:
            lexer = TextLexer()
        
        formatter = HtmlFormatter(style="friendly", noclasses=True)
        highlighted = highlight(code, lexer, formatter)
        tag_html = f'<span class="lang-tag">{lang}</span>' if lang else ''
        return f'<div class="code-wrapper">{tag_html}{highlighted}</div>\n'

    def image(self, text: str, url: str, title: str = None) -> str:
        b64_url = image_to_base64(url, self.base_dir)
        caption_html = f'<div class="img-caption">{text}</div>' if text else ""
        return f'<div class="img-container"><img src="{b64_url}" alt="{text}" />{caption_html}</div>'

def build_html(title: str, body_html: str) -> str:
    """Tạo tài liệu HTML hoàn chỉnh với typography cao cấp và hỗ trợ in A4 chuẩn."""
    return f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{html.escape(title)}</title>

<!-- KaTeX: Toán học & Ký hiệu tài chính -->
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"></script>

<!-- Mermaid.js: Sơ đồ luồng đa phương thức -->
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>

<script>
document.addEventListener("DOMContentLoaded", function() {{
    renderMathInElement(document.body, {{
        delimiters: [
            {{left: '$$', right: '$$', display: true}},
            {{left: '$', right: '$', display: false}}
        ],
        throwOnError: false
    }});
    mermaid.initialize({{
        startOnLoad: true,
        theme: 'neutral',
        securityLevel: 'loose',
        fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
    }});
}});
</script>

<style>
@page {{
    size: A4;
    margin: 18mm 16mm;
}}

* {{
    box-sizing: border-box;
}}

body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    color: #1e293b;
    line-height: 1.6;
    font-size: 13px;
    background-color: #ffffff;
    margin: 0;
    padding: 0;
}}

h1 {{
    font-size: 23px;
    font-weight: 700;
    color: #0f172a;
    border-bottom: 2.5px solid #2563eb;
    padding-bottom: 8px;
    margin-top: 0;
    margin-bottom: 16px;
    letter-spacing: -0.3px;
}}

h2 {{
    font-size: 16.5px;
    font-weight: 600;
    color: #1e3a8a;
    border-left: 4px solid #2563eb;
    padding-left: 10px;
    margin-top: 26px;
    margin-bottom: 12px;
}}

h3 {{
    font-size: 14.5px;
    font-weight: 600;
    color: #1e40af;
    margin-top: 20px;
    margin-bottom: 8px;
}}

h4 {{
    font-size: 13.5px;
    font-weight: 600;
    color: #334155;
    margin-top: 14px;
    margin-bottom: 6px;
}}

p {{
    margin-top: 0;
    margin-bottom: 10px;
}}

ul, ol {{
    margin-top: 4px;
    margin-bottom: 12px;
    padding-left: 22px;
}}

li {{
    margin-bottom: 4px;
}}

table {{
    width: 100%;
    border-collapse: collapse;
    margin: 14px 0;
    font-size: 12px;
    page-break-inside: avoid;
}}

th, td {{
    border: 1px solid #cbd5e1;
    padding: 8px 11px;
    text-align: left;
    vertical-align: top;
}}

th {{
    background-color: #f1f5f9;
    font-weight: 600;
    color: #0f172a;
}}

tr:nth-child(even) {{
    background-color: #f8fafc;
}}

blockquote {{
    margin: 12px 0;
    padding: 8px 14px;
    background-color: #f0f9ff;
    border-left: 4px solid #0284c7;
    color: #0369a1;
    border-radius: 0 6px 6px 0;
    page-break-inside: avoid;
    font-size: 12.5px;
}}

blockquote p {{
    margin: 0;
}}

.code-wrapper {{
    position: relative;
    margin: 12px 0;
    page-break-inside: avoid;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    background: #f8fafc;
    overflow: hidden;
}}

.lang-tag {{
    display: inline-block;
    padding: 2px 8px;
    font-size: 9.5px;
    font-weight: bold;
    color: #64748b;
    background: #e2e8f0;
    border-bottom: 1px solid #cbd5e1;
    border-right: 1px solid #cbd5e1;
    border-radius: 0 0 4px 0;
    text-transform: uppercase;
}}

pre {{
    margin: 0;
    padding: 10px 12px;
    font-family: Consolas, "Cascadia Code", "Fira Code", monospace;
    font-size: 11.5px;
    line-height: 1.45;
    background: #f8fafc;
    overflow-x: auto;
}}

code {{
    font-family: Consolas, "Cascadia Code", "Fira Code", monospace;
    font-size: 11.5px;
    background-color: #f1f5f9;
    padding: 2px 5px;
    border-radius: 4px;
    color: #b91c1c;
}}

.img-container {{
    text-align: center;
    margin: 16px 0;
    page-break-inside: avoid;
}}

.img-container img {{
    max-width: 100%;
    height: auto;
    border-radius: 6px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    border: 1px solid #e2e8f0;
    display: block;
    margin: 0 auto;
}}

.img-caption {{
    font-size: 11px;
    color: #64748b;
    margin-top: 6px;
    font-style: italic;
}}

.mermaid-wrapper {{
    text-align: center;
    margin: 16px auto;
    page-break-inside: avoid;
}}

.mermaid svg {{
    max-width: 100% !important;
    height: auto !important;
    display: block;
    margin: 0 auto;
}}

a {{
    color: #2563eb;
    text-decoration: none;
    font-weight: 500;
}}

a code {{
    color: #2563eb;
}}

hr {{
    border: none;
    border-top: 1px solid #e2e8f0;
    margin: 20px 0;
}}

@media print {{
    body {{
        -webkit-print-color-adjust: exact !important;
        print-color-adjust: exact !important;
    }}
    h1, h2, h3 {{
        page-break-after: avoid;
    }}
    pre, table, blockquote, .img-container, .mermaid-wrapper {{
        page-break-inside: avoid;
    }}
}}
</style>
</head>
<body>
{body_html}
</body>
</html>
"""

def convert_md_to_pdf(md_file_path: pathlib.Path, output_pdf_path: pathlib.Path = None) -> pathlib.Path:
    """Chuyển đổi một file Markdown sang PDF."""
    if not md_file_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {md_file_path}")
    
    if output_pdf_path is None:
        output_pdf_path = md_file_path.with_suffix(".pdf")
    
    base_dir = md_file_path.parent
    raw_md = md_file_path.read_text(encoding="utf-8")
    
    # Renderer tùy biến
    renderer = ElegantPDFRenderer(base_dir=base_dir)
    md_parser = mistune.create_markdown(renderer=renderer, plugins=['table', 'url'])
    
    # Render Markdown sang HTML
    rendered_body = md_parser(raw_md)
    
    # Khôi phục các thẻ <br/> trong bảng bị escape thành &lt;br/&gt;
    rendered_body = re.sub(r'&lt;br\s*/?&gt;', '<br/>', rendered_body, flags=re.IGNORECASE)
    
    # Tạo HTML hoàn chỉnh
    doc_title = md_file_path.stem
    full_html_content = build_html(title=doc_title, body_html=rendered_body)
    
    tmp_html = base_dir / f"~tmp_{md_file_path.stem}.html"
    tmp_html.write_text(full_html_content, encoding="utf-8")
    
    browser_exe = find_browser_executable()
    
    cmd = [
        browser_exe,
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        "--virtual-time-budget=6000",
        f"--print-to-pdf={output_pdf_path.resolve()}",
        f"file:///{str(tmp_html.resolve()).replace('\\', '/')}"
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    finally:
        tmp_html.unlink(missing_ok=True)
        
    if not output_pdf_path.exists():
        raise RuntimeError(f"Không thể tạo PDF tại {output_pdf_path}")
    
    reader = pypdf.PdfReader(str(output_pdf_path))
    size_mb = output_pdf_path.stat().st_size / (1024 * 1024)
    print(f"✅ Đã tạo thành công: {output_pdf_path.name} ({len(reader.pages)} trang, {size_mb:.2f} MB)")
    return output_pdf_path

def main():
    root_dir = pathlib.Path(__file__).parent.resolve()
    
    if len(sys.argv) > 1:
        target = pathlib.Path(sys.argv[1])
        if not target.is_absolute():
            target = root_dir / target
        out_target = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else None
        convert_md_to_pdf(target, out_target)
    else:
        # Xuất cả 2 file tài liệu chính của dự án
        print("🚀 Bắt đầu chuyển đổi tài liệu Markdown sang PDF...")
        docs = ["KIEN_TRUC.md", "README.md"]
        for d in docs:
            doc_path = root_dir / d
            if doc_path.exists():
                convert_md_to_pdf(doc_path)
            else:
                print(f"⚠️ Không tìm thấy {d}")
        print("\n🎉 Hoàn tất toàn bộ chuyển đổi!")

if __name__ == "__main__":
    main()
