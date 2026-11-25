"""Extracts PDF pages as images and produces pre-annotations for X-AnyLabeling, Label Studio, and YOLO.

Reuses ``financial_rag.layout_detector.detect_heuristic_layout`` for candidate boxes:
  - Tables (PyMuPDF find_tables)
  - Charts / Diagrams (image regions & vector drawing clusters)

Outputs:
  - images/: High-resolution PNG images of each page
  - xanylabeling/: Per-image JSON files ready for X-AnyLabeling (recommended)
  - label_studio_tasks.json: Pre-annotated tasks for Label Studio
  - yolo/: Standard YOLO format (images/, labels/, data.yaml)
"""

import argparse
import json
import sys
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from financial_rag.layout_detector import detect_heuristic_layout, normalize_box


def detect_tables(page, page_rect):
    boxes = []
    try:
        for t in page.find_tables().tables:
            b = normalize_box(t.bbox, page_rect.width, page_rect.height)
            if b and (b[2] - b[0]) > page_rect.width * 0.2 and (b[3] - b[1]) > 30:
                boxes.append(b)
    except (RuntimeError, ValueError, TypeError, OSError):
        return []
    return boxes


def detect_charts(page, page_rect, table_boxes):
    del table_boxes
    return detect_heuristic_layout(page)["charts"]


def process_pdf(pdf_path: Path, output_dir: Path, dpi: int = 150):
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    xany_dir = output_dir / "xanylabeling"
    yolo_labels_dir = output_dir / "yolo" / "labels"
    yolo_images_dir = output_dir / "yolo" / "images"
    for folder in (images_dir, xany_dir, yolo_labels_dir, yolo_images_dir):
        folder.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open(pdf_path)
    label_studio_tasks = []

    print(f"Processing '{pdf_path.name}' ({len(doc)} pages)...")

    for page_idx, page in enumerate(doc):
        page_num = page_idx + 1
        page.set_rotation(0)
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

        layout = detect_heuristic_layout(page)
        table_boxes = layout["tables"]
        chart_boxes = layout["charts"]

        annotations = []
        yolo_lines = []
        xany_shapes = []

        for class_id, label_name, boxes in [
            (0, "table", table_boxes),
            (1, "chart", chart_boxes),
        ]:
            for idx, (x0, y0, x1, y1) in enumerate(boxes):
                ls_x = (x0 / pdf_w) * 100.0
                ls_y = (y0 / pdf_h) * 100.0
                ls_w = ((x1 - x0) / pdf_w) * 100.0
                ls_h = ((y1 - y0) / pdf_h) * 100.0

                annotations.append(
                    {
                        "id": f"{label_name}_{page_num}_{idx}",
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
                            "rectanglelabels": [label_name],
                        },
                    }
                )

                cx = (x0 + x1) / (2.0 * pdf_w)
                cy = (y0 + y1) / (2.0 * pdf_h)
                bw = (x1 - x0) / pdf_w
                bh = (y1 - y0) / pdf_h
                yolo_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

                xany_shapes.append(
                    {
                        "label": label_name,
                        "points": [
                            [round(x0 * scale, 2), round(y0 * scale, 2)],
                            [round(x1 * scale, 2), round(y1 * scale, 2)],
                        ],
                        "group_id": None,
                        "description": "",
                        "difficult": False,
                        "shape_type": "rectangle",
                        "flags": {},
                    }
                )

        (yolo_labels_dir / f"{pdf_path.stem}_p{page_num:03d}.txt").write_text(
            "\n".join(yolo_lines), encoding="utf-8"
        )
        (xany_dir / f"{pdf_path.stem}_p{page_num:03d}.json").write_text(
            json.dumps(
                {
                    "version": "2.4.0",
                    "flags": {},
                    "shapes": xany_shapes,
                    "imagePath": f"../images/{img_filename}",
                    "imageData": None,
                    "imageHeight": img_h,
                    "imageWidth": img_w,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        label_studio_tasks.append(
            {
                "data": {
                    "image": f"/data/local-files/?d=images/{img_filename}",
                    "page": page_num,
                    "filename": img_filename,
                },
                "predictions": [{"model_version": "heuristic_v2", "result": annotations}],
            }
        )

    tasks_file = output_dir / "label_studio_tasks.json"
    tasks_file.write_text(
        json.dumps(label_studio_tasks, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    data_yaml = output_dir / "yolo" / "data.yaml"
    data_yaml.write_text(
        f"path: {output_dir.resolve() / 'yolo'}\n"
        "train: images\n"
        "val: images\n"
        "names:\n"
        "  0: table\n"
        "  1: chart\n",
        encoding="utf-8",
    )

    print("Generated:")
    print(f"  - Images: {len(doc)} files in '{images_dir}'")
    print(f"  - X-AnyLabeling JSONs: '{xany_dir}'")
    print(f"  - Label Studio Import file: '{tasks_file}'")
    print(f"  - YOLO Dataset: '{output_dir / 'yolo'}'")


def main():
    parser = argparse.ArgumentParser(description="Convert PDF to images and candidate annotations.")
    parser.add_argument("pdf", type=str, help="Path to PDF file or directory containing PDFs")
    parser.add_argument(
        "--output", "-o", type=str, default="data/annotation_workspace", help="Output directory"
    )
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
