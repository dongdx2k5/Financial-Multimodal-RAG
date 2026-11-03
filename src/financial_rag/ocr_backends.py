"""Opt-in Vietnamese OCR: a PP-OCR detector and a VietOCR line recognizer.

Neither model is imported or downloaded by the normal Tesseract path. Model files
must be provisioned explicitly. Coordinates always use unrotated PDF points,
matching PyMuPDF text extraction; line boxes are never presented as word boxes.

Inference configuration follows pbcquoc/vietocr's Apache-2.0 base.yml and
vgg-seq2seq.yml. Detector API follows RapidAI/RapidOCR 3.x (detector-only mode).
"""

from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from typing import Any

import pymupdf


class OCREngineUnavailable(RuntimeError):
    """The optional OCR runtime or explicitly provisioned model is unavailable."""


@dataclass
class OCRLine:
    text: str
    bbox: list[float]
    confidence: float | None = None


@dataclass
class OCRResult:
    lines: list[OCRLine]
    engine: str = "vietocr_vgg_seq2seq+ppocrv5_mobile_det"
    warnings: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)

    @property
    def words(self) -> list[list[Any]]:
        """Compatibility records; each record is a *line*, not a detected word.

        Consumers must inspect the backend/coordinate granularity. Splitting a
        line's text into tokens and inventing their x coordinates is unsafe for
        financial columns, so this method deliberately does not do that.
        """
        return [[*line.bbox, line.text, 0, i, 0] for i, line in enumerate(self.lines)]


def pixel_box_to_pdf(
    box: Any,
    page: pymupdf.Page,
    scale: float,
    origin: tuple[float, float] = (0, 0),
) -> list[float]:
    """Map an image rectangle/quad to a clamped unrotated PDF bounding box.

    ``origin`` is the rendered pixmap's (x, y), in scaled device pixels. It is
    important for cropped renders. ``scale`` is the actual render matrix scale,
    not rounded pixmap width divided by page width.
    """
    import numpy as np

    if not isfinite(scale) or scale <= 0:
        raise ValueError("OCR render scale must be positive and finite")
    points = np.asarray(box, dtype=float)
    if points.shape == (4,):
        x0, y0, x1, y1 = points
        points = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError("OCR box must be a finite rectangle or quadrilateral")
    transformed = [
        pymupdf.Point((x + origin[0]) / scale, (y + origin[1]) / scale) * page.derotation_matrix
        for x, y in points
    ]
    rect = pymupdf.Rect(
        min(p.x for p in transformed),
        min(p.y for p in transformed),
        max(p.x for p in transformed),
        max(p.y for p in transformed),
    )
    rect &= page.rect * page.derotation_matrix
    if rect.is_empty or rect.is_infinite:
        raise ValueError("OCR box is outside the PDF page or has zero area")
    return [float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)]


def _local_model(path: str | Path | None, label: str) -> Path:
    if not path:
        raise OCREngineUnavailable(f"Thiếu {label}; hãy chạy bước chuẩn bị model OCR trước.")
    candidate = Path(path).expanduser().resolve()
    if not candidate.is_file() or candidate.stat().st_size == 0:
        raise OCREngineUnavailable(f"Không tìm thấy model {label}: {candidate}")
    return candidate


def vietocr_inference_config(weights_path: Path, device: str = "cpu") -> dict:
    """Build the pretrained vgg_seq2seq inference config without HTTP requests.

    Vocabulary order is part of the weight contract: do not sort or normalize it.
    ``pretrained=False`` prevents torchvision from downloading ImageNet weights.
    """
    return {
        "vocab": (
            "aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐ"
            "eEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnN"
            "oOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStT"
            "uUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ"
            "0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ "
        ),
        "device": device,
        "weights": str(weights_path),
        "backbone": "vgg19_bn",
        "cnn": {
            "pretrained": False,
            "ss": [[2, 2], [2, 2], [2, 1], [2, 1], [1, 1]],
            "ks": [[2, 2], [2, 2], [2, 1], [2, 1], [1, 1]],
            "hidden": 256,
        },
        "seq_modeling": "seq2seq",
        "transformer": {
            "encoder_hidden": 256,
            "decoder_hidden": 256,
            "img_channel": 256,
            "decoder_embedded": 256,
            "dropout": 0.1,
        },
        "dataset": {"image_height": 32, "image_min_width": 32, "image_max_width": 512},
        "predictor": {"beamsearch": False},
        "quiet": True,
    }


class VietOCRBackend:
    """Experimental CPU-friendly OCR; explicit local models, lazy initialization.

    ``recognize_images`` can compare recognizers on identical PIL line crops.
    ``extract`` additionally requires a local PP-OCRv5_mobile_det ONNX model.
    Confidence is the recognizer's score, not a calibrated correctness estimate.
    """

    def __init__(
        self,
        weights_path: str | Path | None = None,
        device: str = "cpu",
        min_confidence: float = 0.0,
        detector_path: str | Path | None = None,
        cpu_threads: int = 4,
        batch_size: int = 8,
    ):
        if not 0 <= min_confidence <= 1:
            raise ValueError("min_confidence must be between 0 and 1")
        if cpu_threads < 1 or batch_size < 1:
            raise ValueError("cpu_threads and batch_size must be positive")
        self.weights_path = weights_path
        self.detector_path = detector_path
        self.device = device
        self.min_confidence = min_confidence
        self.cpu_threads = cpu_threads
        self.batch_size = batch_size
        self._predictor = None
        self._detector = None

    def _load_predictor(self):
        if self._predictor is not None:
            return self._predictor
        weights = _local_model(self.weights_path, "VIETOCR_WEIGHTS")
        try:
            import torch
            from vietocr.tool.predictor import Predictor

            if self.device == "cpu":
                torch.set_num_threads(self.cpu_threads)
            self._predictor = Predictor(vietocr_inference_config(weights, self.device))
        except Exception as exc:
            raise OCREngineUnavailable(
                "Không khởi tạo được VietOCR. Kiểm tra môi trường OCR riêng và model "
                f"vgg_seq2seq tương thích. Chi tiết: {type(exc).__name__}: {exc}"
            ) from exc
        return self._predictor

    def _load_detector(self):
        if self._detector is not None:
            return self._detector
        model_path = _local_model(self.detector_path, "OCR_DETECTOR_PATH")
        try:
            from rapidocr import EngineType, ModelType, OCRVersion, RapidOCR

            self._detector = RapidOCR(
                params={
                    "Global.use_det": True,
                    "Global.use_cls": False,
                    "Global.use_rec": False,
                    "Global.max_side_len": 3000,
                    "Det.model_path": str(model_path),
                    "Det.engine_type": EngineType.ONNXRUNTIME,
                    "Det.ocr_version": OCRVersion.PPOCRV5,
                    "Det.model_type": ModelType.MOBILE,
                    "Det.limit_side_len": 1600,
                    "Det.limit_type": "max",
                    "Det.mean": [0.485, 0.456, 0.406],
                    "Det.std": [0.229, 0.224, 0.225],
                    "EngineConfig.onnxruntime.intra_op_num_threads": self.cpu_threads,
                    "EngineConfig.onnxruntime.inter_op_num_threads": 1,
                    "EngineConfig.onnxruntime.use_cuda": False,
                    "EngineConfig.onnxruntime.use_dml": False,
                }
            )
        except Exception as exc:
            raise OCREngineUnavailable(
                "Không khởi tạo được bộ phát hiện chữ RapidOCR/ONNX Runtime. "
                f"Chi tiết: {type(exc).__name__}: {exc}"
            ) from exc
        return self._detector

    def recognize_images(self, images: list) -> list[tuple[str, float | None]]:
        """Recognize PIL RGB line crops, preserving ordering and numeric glyphs."""
        if not images:
            return []
        predictor = self._load_predictor()
        results = []
        try:
            for start in range(0, len(images), self.batch_size):
                batch = images[start : start + self.batch_size]
                texts, scores = predictor.predict_batch(batch, return_prob=True)
                if len(texts) != len(batch) or len(scores) != len(batch):
                    raise RuntimeError("VietOCR trả số dòng không khớp số ảnh đầu vào")
                for text, score in zip(texts, scores, strict=True):
                    # The recognizer returns NaN for an empty sequence in some versions.
                    confidence = float(score) if score is not None else None
                    if confidence is not None and not isfinite(confidence):
                        confidence = None
                    if confidence is not None:
                        confidence = max(0.0, min(1.0, confidence))
                    results.append((str(text).strip(), confidence))
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            raise RuntimeError(f"VietOCR nhận dạng thất bại: {exc}") from exc
        return results

    def extract(self, page: pymupdf.Page, dpi: int = 300) -> OCRResult:
        # PDF /Rotate is display metadata; a line recognizer needs upright crops.
        # The temporary change is in memory only, and all stored boxes remain
        # unrotated PDF coordinates. Restore it even if a model raises an error.
        rotation = page.rotation
        try:
            page.set_rotation(0)
            return self._extract_upright(page, dpi)
        finally:
            page.set_rotation(rotation)

    def _extract_upright(self, page: pymupdf.Page, dpi: int) -> OCRResult:
        import numpy as np

        if not 72 <= dpi <= 600:
            raise ValueError("OCR DPI must be between 72 and 600")
        scale = dpi / 72
        if page.rect.width * page.rect.height * scale**2 > 24_000_000:
            raise RuntimeError("Trang quá lớn cho OCR thử nghiệm; giảm DPI hoặc cắt vùng trước.")
        detector = self._load_detector()
        self._load_predictor()
        try:
            import cv2
            from PIL import Image
        except ImportError as exc:
            raise OCREngineUnavailable("OCR thử nghiệm cần Pillow và OpenCV.") from exc
        pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), colorspace=pymupdf.csRGB)
        rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        # RapidOCR's ndarray input contract is OpenCV BGR, whereas VietOCR uses PIL RGB.
        try:
            detected = detector(rgb[:, :, :3][:, :, ::-1].copy(), use_cls=False, use_rec=False)
        except Exception as exc:
            raise OCREngineUnavailable(f"Bộ phát hiện chữ suy luận thất bại: {exc}") from exc
        boxes = getattr(detected, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return OCRResult([], warnings=["Bộ phát hiện chữ không tìm thấy dòng trên trang."])
        crops, pdf_boxes = [], []
        warnings = []
        for box in boxes:
            try:
                quad = np.asarray(box, dtype=np.float32)
                if quad.shape != (4, 2) or not np.isfinite(quad).all():
                    raise ValueError("invalid detector quadrilateral")
                quad[:, 0] = np.clip(quad[:, 0], 0, pix.width - 1)
                quad[:, 1] = np.clip(quad[:, 1], 0, pix.height - 1)
                pdf_box = pixel_box_to_pdf(quad, page, scale, (pix.x, pix.y))
                width = round(
                    max(np.linalg.norm(quad[0] - quad[1]), np.linalg.norm(quad[2] - quad[3]))
                )
                height = round(
                    max(np.linalg.norm(quad[0] - quad[3]), np.linalg.norm(quad[1] - quad[2]))
                )
                if width < 2 or height < 2:
                    raise ValueError("empty detector crop")
                target = np.array(
                    [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
                    dtype=np.float32,
                )
                matrix = cv2.getPerspectiveTransform(quad, target)
                crop = cv2.warpPerspective(
                    rgb[:, :, :3], matrix, (width, height), borderMode=cv2.BORDER_REPLICATE
                )
                crops.append(Image.fromarray(crop))
                pdf_boxes.append(pdf_box)
            except (ValueError, cv2.error):
                warnings.append("Bỏ qua vùng chữ không có tọa độ hợp lệ.")
        predictions = self.recognize_images(crops)
        lines = []
        for (text, confidence), bbox in zip(predictions, pdf_boxes, strict=True):
            if not text:
                continue
            if self.min_confidence > 0 and (confidence is None or confidence < self.min_confidence):
                warnings.append("Một dòng OCR dưới ngưỡng tin cậy đã bị loại; cần kiểm tra PDF.")
                continue
            lines.append(OCRLine(text=text, bbox=bbox, confidence=confidence))
        # Keep detector reading order. A global sort on unrotated y would scramble
        # rotated pages and a naive sort by exact y can scramble aligned columns.
        return OCRResult(lines, warnings=warnings)
