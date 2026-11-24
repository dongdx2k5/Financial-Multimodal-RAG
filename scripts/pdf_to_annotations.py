"""Extracts PDF pages as images and produces pre-annotations for Label Studio and YOLO.

Detects candidate:
  - Tables (using PyMuPDF find_tables and spatial heuristics)
  - Charts / Diagrams (vector drawing clusters and graphical figures)

Outputs:
  - images/: High-resolution PNG images of each page
  - label_studio_tasks.json: Pre-annotated tasks ready to import into Label Studio
  - yolo_dataset/: Standard YOLO format (images/, labels/, data.yaml)
"""

import argparse
import json
from pathlib import Path

import pymupdf


def normalize_box(bbox, width, height):
    """Ensure coordinates are within image bounds."""
    x0, y0, x1, y1 = bbox
    x0 = max(0.0, min(float(x0), float(width)))
    y0 = max(0.0, min(float(y0), float(height)))
    x1 = max(0.0, min(float(x1), float(width)))
    y1 = max(0.0, min(float(y1), float(height)))
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def detect_tables(page, page_rect):
    """Detect candidate table bounding boxes using PyMuPDF table finder."""
    boxes = []
    try:
        tabs = page.find_tables()
        for t in tabs.tables:
            b = normalize_box(t.bbox, page_rect.width, page_rect.height)
            if b:
                w, h = b[2] - b[0], b[3] - b[1]
                # Filter out accidental tiny single-line detections
                if w > page_rect.width * 0.2 and h > 30:
                    boxes.append(b)
    except (RuntimeError, ValueError, TypeError, OSError):
        return []
    return boxes


def detect_charts(page, page_rect, table_boxes):
    """Detect candidate charts and diagram figures."""
    boxes = []
    # 1. Check embedded bitmap images
    try:
        images = page.get_image_info()
        for img in images:
            b = normalize_box(img["bbox"], page_rect.width, page_rect.height)
            if not b:
                continue
            w, h = b[2] - b[0], b[3] - b[1]
            # Ignore tiny icons / logos and full-page background scans
            if w < 80 or h < 60:
                continue
            if w > page_rect.width * 0.95 and h > page_rect.height * 0.95:
                continue
            # Avoid overlaps with table boxes
            img_rect = pymupdf.Rect(b)
            if any((img_rect & pymupdf.Rect(tb)).get_area() > 0.5 * img_rect.get_area() for tb in table_boxes):
                continue
            boxes.append(b)
    except (RuntimeError, ValueError, TypeError, OSError):
        boxes = []

    # 2. Check vector drawing clusters (charts drawn with vector lines/rectangles)
    try:
        drawings = page.get_drawings()
        if drawings and len(drawings) >= 15:
            # Cluster drawings by bounding box
            cluster_rect = None
            for d in drawings:
                r = d["rect"]
                if r.width < 5 and r.height < 5:
                    continue
                # Ignore full-page borders or thin separator lines
                if (r.width > page_rect.width * 0.9 and r.height < 3) or (r.height > page_rect.height * 0.9 and r.width < 3):
                    continue
                if cluster_rect is None:
                    cluster_rect = pymupdf.Rect(r)
                else:
                    cluster_rect |= r

            if cluster_rect:
                b = normalize_box(list(cluster_rect), page_rect.width, page_rect.height)
                if b:
                    w, h = b[2] - b[0], b[3] - b[1]
                    # Check reasonable chart size
                    if w > page_rect.width * 0.25 and h > 80:
                        crect = pymupdf.Rect(b)
                        if not any((crect & pymupdf.Rect(tb)).get_area() > 0.3 * crect.get_area() for tb in table_boxes):
                            boxes.append(b)
    except (RuntimeError, ValueError, TypeError, OSError):
        return boxes

    return boxes


def process_pdf(pdf_path: Path, output_dir: Path, dpi: int = 150):
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    yolo_labels_dir = output_dir / "yolo" / "labels"
    yolo_images_dir = output_dir / "yolo" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    yolo_labels_dir.mkdir(parents=True, exist_ok=True)
    yolo_images_dir.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open(pdf_path)
    label_studio_tasks = []

    print(f"Processing '{pdf_path.name}' ({len(doc)} pages)...")

    for page_idx, page in enumerate(doc):
        page_num = page_idx + 1
        page_rect = page.rect
        scale = dpi / 72.0
        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale))

        img_filename = f"{pdf_path.stem}_p{page_num:03d}.png"
        img_path = images_dir / img_filename
        yolo_img_path = yolo_images_dir / img_filename

        pixmap.save(img_path)
        pixmap.save(yolo_img_path)

        img_w, img_h = pixmap.width, pixmap.height
        pdf_w, pdf_h = page_rect.width, page_rect.height

        table_boxes = detect_tables(page, page_rect)
        chart_boxes = detect_charts(page, page_rect, table_boxes)

        annotations = []
        yolo_lines = []

        # Process Tables (Class 0)
        for idx, (x0, y0, x1, y1) in enumerate(table_boxes):
            # Label Studio uses 0..100 percentage values
            ls_x = (x0 / pdf_w) * 100.0
            ls_y = (y0 / pdf_h) * 100.0
            ls_w = ((x1 - x0) / pdf_w) * 100.0
            ls_h = ((y1 - y0) / pdf_h) * 100.0

            annotations.append({
                "id": f"table_{page_num}_{idx}",
                "type": "rectanglelabels",
                "from_name": "label",
                "to_name": "image",
                "original_width": img_w,
                "original_height": img_h,
                "value": {
                    "x": ls_x,
                    "y": ls_y,
                    "width": ls_w,
                    "height": ls_h,
                    "rotation": 0,
                    "rectanglelabels": ["table"]
                }
            })

            # YOLO uses normalized center_x, center_y, width, height (0..1)
            cx = (x0 + x1) / (2.0 * pdf_w)
            cy = (y0 + y1) / (2.0 * pdf_h)
            bw = (x1 - x0) / pdf_w
            bh = (y1 - y0) / pdf_h
            yolo_lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        # Process Charts (Class 1)
        for idx, (x0, y0, x1, y1) in enumerate(chart_boxes):
            ls_x = (x0 / pdf_w) * 100.0
            ls_y = (y0 / pdf_h) * 100.0
            ls_w = ((x1 - x0) / pdf_w) * 100.0
            ls_h = ((y1 - y0) / pdf_h) * 100.0

            annotations.append({
                "id": f"chart_{page_num}_{idx}",
                "type": "rectanglelabels",
                "from_name": "label",
                "to_name": "image",
                "original_width": img_w,
                "original_height": img_h,
                "value": {
                    "x": ls_x,
                    "y": ls_y,
                    "width": ls_w,
                    "height": ls_h,
                    "rotation": 0,
                    "rectanglelabels": ["chart"]
                }
            })

            cx = (x0 + x1) / (2.0 * pdf_w)
            cy = (y0 + y1) / (2.0 * pdf_h)
            bw = (x1 - x0) / pdf_w
            bh = (y1 - y0) / pdf_h
            yolo_lines.append(f"1 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        # Write YOLO label file
        label_file = yolo_labels_dir / f"{pdf_path.stem}_p{page_num:03d}.txt"
        label_file.write_text("\n".join(yolo_lines), encoding="utf-8")

        # Add Label Studio task with pre-annotations
        label_studio_tasks.append({
            "data": {
                "image": f"/data/local-files/?d=images/{img_filename}",
                "page": page_num,
                "filename": img_filename,
            },
            "predictions": [
                {
                    "model_version": "heuristic_v1",
                    "result": annotations
                }
            ]
        })

    # Save Label Studio JSON
    tasks_file = output_dir / "label_studio_tasks.json"
    tasks_file.write_text(json.dumps(label_studio_tasks, ensure_ascii=False, indent=2), encoding="utf-8")

    # Save YOLO data.yaml
    data_yaml = output_dir / "yolo" / "data.yaml"
    data_yaml.write_text(
        f"path: {output_dir.resolve() / 'yolo'}\n"
        "train: images\n"
        "val: images\n"
        "names:\n"
        "  0: table\n"
        "  1: chart\n",
        encoding="utf-8"
    )

    print("Generated:")
    print(f"  - Images: {len(doc)} files in '{images_dir}'")
    print(f"  - Label Studio Import file: '{tasks_file}'")
    print(f"  - YOLO Dataset: '{output_dir / 'yolo'}'")


def main():
    parser = argparse.ArgumentParser(description="Convert PDF to images and candidate annotations.")
    parser.add_argument("pdf", type=str, help="Path to PDF file or directory containing PDFs")
    parser.add_argument("--output", "-o", type=str, default="data/annotation_workspace", help="Output directory")
    parser.add_argument("--dpi", type=int, default=150, help="Render DPI (default: 150)")
    args = parser.parse_args()

    target = Path(args.pdf)
    output = Path(args.output)

    if target.is_file() and target.suffix.lower() == ".pdf":
        process_pdf(target, output, dpi=args.dpi)
    elif target.is_dir():
        pdfs = list(target.glob("*.pdf")) + list(target.glob("*.PDF"))
        if not pdfs:
            print(f"No PDF files found in '{target}'")
            return
        for pdf in pdfs:
            process_pdf(pdf, output, dpi=args.dpi)
    else:
        print(f"Invalid target: '{target}'")


if __name__ == "__main__":
    main()
