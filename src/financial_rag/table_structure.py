"""Optional, CPU-only SLANet_plus table structure experiments.

RapidTable supplies *structure only*. Text is assigned from existing OCR boxes,
never generated from HTML or guessed from a nearby cell. Coordinates are always
unrotated PDF points, as in PyMuPDF's ``get_text('words')`` output.
"""

import math
import re
import unicodedata
from pathlib import Path

import pymupdf


class TableStructureError(RuntimeError):
    """Optional table engine is unavailable or returned invalid structure."""


def _box(value):
    """Accept a rectangle or a flat/nested quadrilateral; reject bad geometry."""
    if hasattr(value, "tolist"):
        value = value.tolist()
    if len(value) == 4 and isinstance(value[0], (list, tuple)):
        value = [coordinate for point in value for coordinate in point]
    values = [float(item) for item in value]
    if len(values) == 8:
        values = [min(values[::2]), min(values[1::2]), max(values[::2]), max(values[1::2])]
    if len(values) != 4 or not all(math.isfinite(item) for item in values):
        raise TableStructureError("Invalid table/OCR bounding box.")
    if values[2] <= values[0] or values[3] <= values[1]:
        raise TableStructureError("Empty table/OCR bounding box.")
    return values


def _lines(lines):
    result = []
    for line in lines:
        if isinstance(line, dict):
            text, bbox = line["text"], line["bbox"]
            confidence = line.get("confidence")
        elif hasattr(line, "text"):
            text, bbox = line.text, line.bbox
            confidence = getattr(line, "confidence", None)
        else:
            bbox, text, confidence = line[:4], line[4], None
        if not str(text).strip():
            continue
        if confidence is not None:
            confidence = float(confidence)
            if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                confidence = None
        result.append({"text": str(text), "bbox": _box(bbox), "confidence": confidence})
    return result


def _overlap_fraction(text_box, cell_box):
    intersection = pymupdf.Rect(text_box) & pymupdf.Rect(cell_box)
    return intersection.get_area() / pymupdf.Rect(text_box).get_area()


def assign_text_to_cells(cells, lines):
    """Assign each OCR region at most once; reject regions crossing cell borders.

    A whole-line detector may return several columns as one box. Splitting its
    string by spaces would fabricate coordinates, so it is deliberately omitted.
    All omitted text remains available in the parent page's ordinary OCR output.
    """
    assigned = [[] for _ in cells]
    unmatched = 0
    ambiguous = 0
    for line in _lines(lines):
        matches = sorted(
            ((_overlap_fraction(line["bbox"], cell["bbox"]), i) for i, cell in enumerate(cells)),
            reverse=True,
        )
        if not matches or matches[0][0] < 0.60:
            unmatched += 1
        elif len(matches) > 1 and matches[1][0] >= 0.20:
            ambiguous += 1
        else:
            assigned[matches[0][1]].append(line)
    result = []
    for cell, parts in zip(cells, assigned):
        parts.sort(key=lambda item: (round(item["bbox"][1] / 3), item["bbox"][0]))
        scores = [part["confidence"] for part in parts]
        result.append(
            {
                **cell,
                "text": " ".join(part["text"] for part in parts),
                # This is OCR confidence, NOT a calibrated structure probability.
                "confidence": min(scores)
                if scores and all(s is not None for s in scores)
                else None,
            }
        )
    warnings = []
    if unmatched:
        warnings.append(f"{unmatched} vùng chữ chưa ghép được vào ô; giữ trong OCR trang.")
    if ambiguous:
        warnings.append(f"{ambiguous} vùng chữ giao nhiều ô; không tự chia hoặc đoán cột.")
    return result, warnings


def cells_from_prediction(cell_boxes, logic_points, *, scale, pixel_origin, region):
    """Validate inclusive RapidTable [row_start,row_end,col_start,col_end] spans."""
    if len(cell_boxes) != len(logic_points) or not len(cell_boxes):
        raise TableStructureError("SLANet_plus returned inconsistent or empty cells.")
    if len(cell_boxes) > 4000 or scale <= 0:
        raise TableStructureError("Table prediction exceeds safe limits.")
    cells, occupied = [], set()
    clip = pymupdf.Rect(_box(region))
    for bounds, logic in zip(cell_boxes, logic_points):
        if len(logic) != 4 or any(not math.isfinite(float(v)) or int(v) != v for v in logic):
            raise TableStructureError("Invalid table row/column indexes.")
        row, end_row, col, end_col = map(int, logic)
        if not (0 <= row <= end_row < 500 and 0 <= col <= end_col < 100):
            raise TableStructureError("Invalid table row/column spans.")
        positions = {(r, c) for r in range(row, end_row + 1) for c in range(col, end_col + 1)}
        if positions & occupied:
            raise TableStructureError("Overlapping logical cells; table rejected.")
        occupied.update(positions)
        x0, y0, x1, y1 = _box(bounds)
        mapped = (
            pymupdf.Rect(
                (x0 + pixel_origin[0]) / scale,
                (y0 + pixel_origin[1]) / scale,
                (x1 + pixel_origin[0]) / scale,
                (y1 + pixel_origin[1]) / scale,
            )
            & clip
        )
        if mapped.is_empty:
            raise TableStructureError("Predicted cell is outside the table crop.")
        cells.append(
            {
                "row": row,
                "col": col,
                "row_span": end_row - row + 1,
                "col_span": end_col - col + 1,
                "bbox": list(mapped),
                "text": "",
                "confidence": None,
            }
        )
    return cells


def _statement_region(lines, page_rect):
    """Conservative fallback candidate, explicitly NOT a trained layout detector."""
    ordered = sorted(lines, key=lambda line: (line["bbox"][1], line["bbox"][0]))
    heading = " ".join(line["text"] for line in ordered if line["bbox"][1] < page_rect.height * 0.3)
    heading = unicodedata.normalize("NFD", heading.lower().replace("đ", "d"))
    heading = "".join(c for c in heading if unicodedata.category(c) != "Mn")
    heading = re.sub(r"\s+", "", heading)
    if not any(
        title in heading
        for title in ("bangcandoiketoan", "baocaotinhhinhtaichinh", "baocaoketqua", "baocaoluu")
    ):
        return None
    number = re.compile(r"^\(?-?(?:\d{1,3}(?:[.,]\d{3})+|\d{4,})\)?$")
    numeric_rows = []
    for line in ordered:
        box = line["bbox"]
        if box[0] < page_rect.width * 0.4 or not number.fullmatch(line["text"].strip()):
            continue
        center = (box[1] + box[3]) / 2
        if numeric_rows and abs(center - numeric_rows[-1][0]) <= 6:
            numeric_rows[-1][1].append(box)
        else:
            numeric_rows.append((center, [box]))
    aligned = [
        boxes
        for _, boxes in numeric_rows
        if len(boxes) >= 2
        and max(b[0] for b in boxes) - min(b[0] for b in boxes) > page_rect.width * 0.08
    ]
    if len(aligned) < 4:
        return None
    y0 = max(0, min(b[1] for row in aligned for b in row) - 45)
    y1 = min(page_rect.height, max(b[3] for row in aligned for b in row) + 8)
    body = [line["bbox"] for line in ordered if y0 <= line["bbox"][1] <= y1]
    region = pymupdf.Rect(
        max(0, min(b[0] for b in body) - 5),
        y0,
        min(page_rect.width, max(b[2] for b in body) + 5),
        y1,
    )
    return list(region) if region.height < page_rect.height * 0.85 else None


class SLANetTableBackend:
    """Lazy local-only RapidTable 3.x adapter. No automatic model downloads."""

    def __init__(self, model_path=None, dpi=200, allow_statement_region=True):
        if not 72 <= dpi <= 400:
            raise ValueError("Table DPI must be between 72 and 400.")
        self.model_path = Path(model_path) if model_path else None
        self.dpi = dpi
        self.allow_statement_region = allow_statement_region
        self._engine = None

    def _load_engine(self):
        if self._engine is not None:
            return self._engine
        try:
            from rapid_table import ModelType, RapidTable, RapidTableInput
        except ImportError as exc:
            raise TableStructureError(
                "Cần cài rapid-table>=3,<4 và onnxruntime cho SLANet_plus."
            ) from exc
        path = self.model_path
        if path is None or not path.is_file() or path.stat().st_size == 0:
            raise TableStructureError(
                "Thiếu TABLE_MODEL_PATH trỏ đến slanet-plus.onnx cục bộ; hãy chạy bước cài model."
            )
        try:
            self._engine = RapidTable(
                RapidTableInput(
                    model_type=ModelType.SLANETPLUS,
                    model_dir_or_path=str(path.resolve()),
                    use_ocr=False,
                    engine_cfg={"use_cuda": False, "use_dml": False, "intra_op_num_threads": 4},
                )
            )
        except Exception as exc:
            raise TableStructureError(f"Không khởi tạo được SLANet_plus: {exc}") from exc
        return self._engine

    def extract_tables(self, page, lines, regions=None):
        """Return structured dictionaries; optional regions are unrotated PDF boxes.

        Explicit regions are ideal for the benchmark. Otherwise PyMuPDF detections
        are used first, then a labeled financial-statement heuristic. Arbitrary
        whole pages are never silently treated as tables.
        """
        lines = _lines(lines)
        rotation = page.rotation
        # Only in-memory page rendering is changed; always restore it, never save.
        try:
            page.set_rotation(0)
            bounds = pymupdf.Rect(page.rect)
            region_source = "explicit"
            if regions is None:
                region_source = "pymupdf"
                try:
                    regions = [list(table.bbox) for table in page.find_tables().tables]
                except (RuntimeError, ValueError, TypeError):
                    regions = []
                if not regions and self.allow_statement_region:
                    region = _statement_region(lines, bounds)
                    regions = [region] if region else []
                    region_source = "statement_heuristic"
            if not regions:
                return []
            engine = self._load_engine()
            import numpy as np

            tables = []
            for raw_region in regions[:20]:
                region = pymupdf.Rect(_box(raw_region)) & bounds
                if region.is_empty:
                    raise TableStructureError("Table region does not intersect the PDF page.")
                scale = self.dpi / 72
                if region.get_area() * scale**2 > 24_000_000:
                    raise TableStructureError("Vùng bảng quá lớn; hãy giảm DPI hoặc chia vùng.")
                pix = page.get_pixmap(
                    matrix=pymupdf.Matrix(scale, scale),
                    clip=region,
                    colorspace=pymupdf.csRGB,
                    alpha=False,
                )
                # RapidTable numpy input convention is OpenCV BGR.
                image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
                try:
                    prediction = engine(image[:, :, ::-1].copy())
                except Exception as exc:
                    # ONNX Runtime failures do not consistently inherit RuntimeError.
                    raise TableStructureError(f"SLANet_plus suy luận thất bại: {exc}") from exc
                try:
                    cells = cells_from_prediction(
                        prediction.cell_bboxes[0],
                        prediction.logic_points[0],
                        scale=scale,
                        pixel_origin=(pix.x, pix.y),
                        region=list(region),
                    )
                except (AttributeError, IndexError) as exc:
                    raise TableStructureError(
                        "Unexpected RapidTable output; expected version 3.x."
                    ) from exc
                inside = [line for line in lines if _overlap_fraction(line["bbox"], region) >= 0.6]
                cells, warnings = assign_text_to_cells(cells, inside)
                warnings.insert(0, "Cấu trúc bảng thử nghiệm; chưa xác minh ô, kỳ và đơn vị.")
                if region_source == "statement_heuristic":
                    warnings.append(
                        "Vùng bảng được khoanh bằng quy tắc báo cáo tài chính, chưa có model layout."
                    )
                tables.append(
                    {
                        "bbox": list(region),
                        "cells": cells,
                        "engine": "slanet_plus",
                        "warnings": warnings,
                        "region_source": region_source,
                    }
                )
            return tables
        finally:
            page.set_rotation(rotation)
