"""Document layout analysis (tables and charts) separated from OCR text detection.

Supports:
- ``heuristic``: PyMuPDF table detection + image/vector-drawing clustering.
- ``yolo``: Local ONNX YOLOv8 / DocLayout-YOLO model (``LAYOUT_MODEL_PATH``).

All returned coordinates are clamped unrotated PDF points matching PyMuPDF text space.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf


class LayoutDetectionError(RuntimeError):
    """Raised when the optional layout detector is missing or returns invalid output."""


def normalize_box(bbox: list[float] | tuple[float, ...], width: float, height: float) -> list[float] | None:
    x0, y0, x1, y1 = bbox
    x0 = max(0.0, min(float(x0), float(width)))
    y0 = max(0.0, min(float(y0), float(height)))
    x1 = max(0.0, min(float(x1), float(width)))
    y1 = max(0.0, min(float(y1), float(height)))
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def detect_heuristic_layout(page: pymupdf.Page) -> dict[str, list[list[float]]]:
    """Extract candidate table and chart bounding boxes using PyMuPDF primitives."""
    rotation = page.rotation
    try:
        page.set_rotation(0)
        page_rect = page.rect
        tables: list[list[float]] = []
        try:
            for t in page.find_tables().tables:
                box = normalize_box(t.bbox, page_rect.width, page_rect.height)
                if box and (box[2] - box[0]) > page_rect.width * 0.2 and (box[3] - box[1]) > 30:
                    tables.append(box)
        except (RuntimeError, ValueError, TypeError, OSError):
            tables = []

        charts: list[list[float]] = []
        try:
            for img in page.get_image_info():
                box = normalize_box(img["bbox"], page_rect.width, page_rect.height)
                if not box:
                    continue
                w, h = box[2] - box[0], box[3] - box[1]
                if w < 80 or h < 60:
                    continue
                if w > page_rect.width * 0.95 and h > page_rect.height * 0.95:
                    continue
                img_rect = pymupdf.Rect(box)
                if any(
                    (img_rect & pymupdf.Rect(tb)).get_area() > 0.5 * img_rect.get_area()
                    for tb in tables
                ):
                    continue
                charts.append(box)
        except (RuntimeError, ValueError, TypeError, OSError):
            charts = []

        try:
            drawings = page.get_drawings()
            if drawings and len(drawings) >= 15:
                cluster_rect = None
                for d in drawings:
                    r = d["rect"]
                    if r.width < 5 and r.height < 5:
                        continue
                    if (r.width > page_rect.width * 0.9 and r.height < 3) or (
                        r.height > page_rect.height * 0.9 and r.width < 3
                    ):
                        continue
                    # Ignore items inside known table regions.
                    if any((r & pymupdf.Rect(tb)).get_area() > 0.5 * max(r.get_area(), 1e-6) for tb in tables):
                        continue
                    cluster_rect = pymupdf.Rect(r) if cluster_rect is None else (cluster_rect | r)
                if cluster_rect:
                    box = normalize_box(list(cluster_rect), page_rect.width, page_rect.height)
                    if box and (box[2] - box[0]) > page_rect.width * 0.25 and (box[3] - box[1]) > 80:
                        crect = pymupdf.Rect(box)
                        if not any(
                            (crect & pymupdf.Rect(tb)).get_area() > 0.3 * crect.get_area()
                            for tb in tables
                        ):
                            charts.append(box)
        except (RuntimeError, ValueError, TypeError, OSError):
            pass

        return {"tables": tables, "charts": charts}
    finally:
        page.set_rotation(rotation)


class YOLOLayoutDetector:
    """Lazy ONNX Runtime detector for 2-class layout (0: table, 1: chart)."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        imgsz: int = 1024,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
    ):
        self.model_path = Path(model_path) if model_path else None
        self.imgsz = imgsz
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self._session = None

    def _load_session(self):
        if self._session is not None:
            return self._session
        if self.model_path is None or not self.model_path.is_file() or self.model_path.stat().st_size == 0:
            raise LayoutDetectionError(
                "Thiếu LAYOUT_MODEL_PATH trỏ đến file ONNX mô hình YOLO (table/chart)."
            )
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise LayoutDetectionError("Cần cài onnxruntime để chạy YOLO layout detector.") from exc
        try:
            self._session = ort.InferenceSession(
                str(self.model_path.resolve()),
                providers=["CPUExecutionProvider"],
            )
        except Exception as exc:
            raise LayoutDetectionError(f"Không nạp được YOLO layout ONNX: {exc}") from exc
        return self._session

    @staticmethod
    def _nms(boxes, scores, iou_threshold: float) -> list[int]:
        import numpy as np

        if len(boxes) == 0:
            return []
        x0, y0, x1, y1 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x1 - x0) * (y1 - y0)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = int(order[0])
            keep.append(i)
            xx0 = np.maximum(x0[i], x0[order[1:]])
            yy0 = np.maximum(y0[i], y0[order[1:]])
            xx1 = np.minimum(x1[i], x1[order[1:]])
            yy1 = np.minimum(y1[i], y1[order[1:]])
            inter = np.maximum(0.0, xx1 - xx0) * np.maximum(0.0, yy1 - yy0)
            union = areas[i] + areas[order[1:]] - inter
            iou = np.where(union > 0, inter / union, 0.0)
            order = order[1:][iou <= iou_threshold]
        return keep

    def detect(self, page: pymupdf.Page) -> dict[str, list[list[float]]]:
        import numpy as np
        from PIL import Image

        session = self._load_session()
        rotation = page.rotation
        try:
            page.set_rotation(0)
            page_rect = page.rect
            scale = min(self.imgsz / max(page_rect.width, 1.0), self.imgsz / max(page_rect.height, 1.0))
            pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), colorspace=pymupdf.csRGB, alpha=False)
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            canvas = Image.new("RGB", (self.imgsz, self.imgsz), (114, 114, 114))
            canvas.paste(image, (0, 0))
            arr = np.asarray(canvas, dtype=np.float32) / 255.0
            tensor = np.transpose(arr, (2, 0, 1))[None, ...]
            input_name = session.get_inputs()[0].name
            outputs = session.run(None, {input_name: tensor})[0]
            preds = np.squeeze(outputs, axis=0)
            if preds.shape[0] < preds.shape[1]:
                preds = preds.T  # Shape (N, 4 + num_classes)
            if preds.ndim != 2 or preds.shape[1] < 6:
                raise LayoutDetectionError("Output của model YOLO ONNX không đúng định dạng.")
            xywh = preds[:, :4]
            class_scores = preds[:, 4:6]
            class_ids = np.argmax(class_scores, axis=1)
            confidences = class_scores[np.arange(len(class_scores)), class_ids]
            mask = confidences >= self.conf_threshold
            if not np.any(mask):
                return {"tables": [], "charts": []}
            xywh = xywh[mask]
            confidences = confidences[mask]
            class_ids = class_ids[mask]
            boxes_xyxy = np.empty_like(xywh)
            boxes_xyxy[:, 0] = (xywh[:, 0] - xywh[:, 2] / 2) / scale
            boxes_xyxy[:, 1] = (xywh[:, 1] - xywh[:, 3] / 2) / scale
            boxes_xyxy[:, 2] = (xywh[:, 0] + xywh[:, 2] / 2) / scale
            boxes_xyxy[:, 3] = (xywh[:, 1] + xywh[:, 3] / 2) / scale

            result: dict[str, list[list[float]]] = {"tables": [], "charts": []}
            for cls_id, label in [(0, "tables"), (1, "charts")]:
                cls_mask = class_ids == cls_id
                if not np.any(cls_mask):
                    continue
                cls_boxes = boxes_xyxy[cls_mask]
                cls_scores = confidences[cls_mask]
                for idx in self._nms(cls_boxes, cls_scores, self.iou_threshold):
                    norm = normalize_box(cls_boxes[idx].tolist(), page_rect.width, page_rect.height)
                    if norm:
                        result[label].append(norm)
            return result
        finally:
            page.set_rotation(rotation)
