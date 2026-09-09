import base64
import html
import uuid
from pathlib import Path

# pyrefly: ignore [missing-import]
import pdfplumber
from pypdf import PdfReader

from financial_rag.ingestion.models import DocumentElement, Modality

SUPPORTED_DOCUMENT_EXTENSIONS = {".pdf", ".txt", ".md"}
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
ALL_SUPPORTED_EXTENSIONS = SUPPORTED_DOCUMENT_EXTENSIONS | SUPPORTED_IMAGE_EXTENSIONS


def _table_to_html(table_data: list[list[str | None]]) -> str:
    """Converts a 2D table grid into clean HTML table markup."""
    if not table_data or not any(table_data):
        return ""
    
    html_parts = ["<table>"]
    header_processed = False
    
    for row_idx, row in enumerate(table_data):
        if not row or not any(cell for cell in row):
            continue
        cleaned_cells = [html.escape(str(cell or "").strip()) for cell in row]
        if not header_processed and row_idx == 0:
            header_cells = "".join(f"<th>{cell}</th>" for cell in cleaned_cells)
            html_parts.append(f"<thead><tr>{header_cells}</tr></thead><tbody>")
            header_processed = True
        else:
            data_cells = "".join(f"<td>{cell}</td>" for cell in cleaned_cells)
            html_parts.append(f"<tr>{data_cells}</tr>")
            
    if header_processed:
        html_parts.append("</tbody>")
    html_parts.append("</table>")
    return "".join(html_parts)


class DocumentLoader:
    def __init__(self, image_output_dir: Path | None = None, max_pages: int | None = None):
        self.image_output_dir = image_output_dir or Path("data/processed/images")
        self.max_pages = max_pages

    def load_path(self, path: Path) -> list[DocumentElement]:
        if path.is_dir():
            elements: list[DocumentElement] = []
            for child in sorted(path.rglob("*")):
                if child.suffix.lower() in ALL_SUPPORTED_EXTENSIONS:
                    elements.extend(self.load_file(child))
            return elements
        return self.load_file(path)

    def load_file(self, path: Path) -> list[DocumentElement]:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return self._load_pdf(path)
        if suffix in {".txt", ".md"}:
            content = path.read_text(encoding="utf-8")
            return [
                DocumentElement(
                    content=content,
                    raw_content=content,
                    modality=Modality.TEXT,
                    source_path=path,
                    element_id=f"{path.name}:text:0",
                )
            ]
        if suffix in SUPPORTED_IMAGE_EXTENSIONS:
            return self._load_standalone_image(path)
        raise ValueError(f"Unsupported file type: {path}")

    def _load_standalone_image(self, path: Path) -> list[DocumentElement]:
        with open(path, "rb") as f:
            raw_bytes = f.read()
        b64_str = base64.b64encode(raw_bytes).decode("utf-8")
        element_id = f"{path.name}:img:{uuid.uuid4().hex[:8]}"
        return [
            DocumentElement(
                content=f"[Image: {path.name}]",
                raw_content=b64_str,
                modality=Modality.IMAGE,
                source_path=path,
                element_id=element_id,
                metadata={
                    "filename": path.name,
                    "mime_type": f"image/{path.suffix.lstrip('.').lower()}",
                    "image_path": str(path),
                },
            )
        ]

    def _load_pdf(self, path: Path) -> list[DocumentElement]:
        elements: list[DocumentElement] = []
        self.image_output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Extract Text
        reader = PdfReader(str(path))
        pdf_pages = reader.pages[: self.max_pages] if self.max_pages else reader.pages
        for page_idx, page in enumerate(pdf_pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                elements.append(
                    DocumentElement(
                        content=text,
                        raw_content=text,
                        modality=Modality.TEXT,
                        source_path=path,
                        page=page_idx,
                        element_id=f"{path.name}:p{page_idx}:text",
                    )
                )

        # 2. Extract Tables (HTML structured) and Images using pdfplumber / pypdf
        try:
            with pdfplumber.open(path) as pdf:
                plumber_pages = pdf.pages[: self.max_pages] if self.max_pages else pdf.pages
                for page_idx, page in enumerate(plumber_pages, start=1):
                    # Extract Tables
                    tables = page.extract_tables() or []
                    for table_idx, table in enumerate(tables, start=1):
                        table_html = _table_to_html(table)
                        if table_html:
                            elements.append(
                                DocumentElement(
                                    content=table_html,
                                    raw_content=table_html,
                                    modality=Modality.TABLE,
                                    source_path=path,
                                    page=page_idx,
                                    element_id=f"{path.name}:p{page_idx}:tbl{table_idx}",
                                    metadata={"table_index": table_idx},
                                )
                            )
        except Exception:
            # Fallback if pdfplumber fails on specific pdf
            pass

        # 3. Extract Images from PDF pages
        for page_idx, page in enumerate(pdf_pages, start=1):
            try:
                for img_idx, img_obj in enumerate(page.images, start=1):
                    try:
                        img_bytes = img_obj.data
                        # Use original name from pypdf to preserve the correct format
                        orig_name = getattr(img_obj, "name", f"img{img_idx}.png")
                        ext = Path(orig_name).suffix.lower() or ".png"
                        mime_type = {
                            ".png": "image/png",
                            ".jpg": "image/jpeg",
                            ".jpeg": "image/jpeg",
                            ".jp2": "image/jp2",
                            ".tiff": "image/tiff",
                            ".tif": "image/tiff",
                            ".bmp": "image/bmp",
                        }.get(ext, "image/png")

                        img_name = f"{path.stem}_p{page_idx}_img{img_idx}{ext}"
                        img_file_path = self.image_output_dir / img_name
                        img_file_path.write_bytes(img_bytes)

                        b64_str = base64.b64encode(img_bytes).decode("utf-8")
                        elements.append(
                            DocumentElement(
                                content=f"[Chart/Figure extracted from {path.name} page {page_idx}]",
                                raw_content=b64_str,
                                modality=Modality.IMAGE,
                                source_path=path,
                                page=page_idx,
                                element_id=f"{path.name}:p{page_idx}:img{img_idx}",
                                metadata={
                                    "image_name": img_name,
                                    "image_path": str(img_file_path),
                                    "mime_type": mime_type,
                                    "page": page_idx,
                                },
                            )
                        )
                    except Exception:
                        continue
            except Exception:
                continue


        return elements
