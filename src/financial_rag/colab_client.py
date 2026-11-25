"""Remote Google Colab GPU inference client.

Local host keeps Web UI, uploads, SQLite/FTS5 store, Knowledge Graph, and Decimal
arithmetic. Only heavy GPU tasks (VLM chart parsing, dense embeddings, remote OCR,
intent parsing, and grounded qualitative synthesis) are dispatched to Colab T4+.
"""

from __future__ import annotations

import base64
import json
import re
from threading import Lock
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


class ColabGPUError(RuntimeError):
    """Raised when the remote Colab GPU worker is unconfigured, unreachable, or unsafe."""


_generation_lock = Lock()


def remote_base(settings) -> str:
    raw = (getattr(settings, "colab_gpu_url", None) or "").strip().rstrip("/")
    if not raw:
        raise ColabGPUError("Chưa cấu hình COLAB_GPU_URL để gọi GPU API trên Google Colab.")
    parsed = urlparse(raw)
    is_local = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
    allowed_schemes = {"https"} | (
        {"http"} if (is_local and getattr(settings, "colab_allow_http_local", True)) else set()
    )
    if (
        parsed.scheme not in allowed_schemes
        or not parsed.netloc
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ColabGPUError(
            "COLAB_GPU_URL phải dùng HTTPS hợp lệ (hoặc http://127.0.0.1 khi test cục bộ)."
        )
    return raw


def _headers(settings) -> dict[str, str]:
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    token = (getattr(settings, "colab_api_token", None) or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _post_json(settings, path: str, payload: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
    base = remote_base(settings)
    req = Request(
        base + path,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=_headers(settings),
        method="POST",
    )
    try:
        with urlopen(req, timeout=timeout or getattr(settings, "colab_timeout", 120.0)) as resp:
            data = json.load(resp)
        if not isinstance(data, dict):
            raise ColabGPUError("Phản hồi Colab GPU API không phải JSON object hợp lệ.")
        return data
    except HTTPError as exc:
        if exc.code in {401, 403}:
            raise ColabGPUError("Colab GPU API từ chối xác thực; kiểm tra COLAB_API_TOKEN.") from exc
        if exc.code == 404:
            raise ColabGPUError(f"Endpoint Colab không tồn tại ({path}).") from exc
        raise ColabGPUError(f"Colab GPU API trả lỗi HTTP {exc.code}.") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise ColabGPUError(
            "Không kết nối được Google Colab GPU API hoặc đã hết thời gian chờ."
        ) from exc
    except (ValueError, TypeError) as exc:
        raise ColabGPUError("Phản hồi JSON từ Colab GPU API không hợp lệ.") from exc


def probe(settings) -> dict[str, Any]:
    raw_url = (getattr(settings, "colab_gpu_url", None) or "").strip()
    result: dict[str, Any] = {
        "provider": "colab_gpu",
        "configured": bool(raw_url),
        "url": raw_url or None,
        "ready": False,
        "busy": _generation_lock.locked(),
        "gpu": None,
        "models": {},
    }
    if not raw_url:
        return {
            **result,
            "message": "Chưa cấu hình COLAB_GPU_URL; hệ thống đang chạy chế độ tất định cục bộ.",
        }
    try:
        base = remote_base(settings)
        req = Request(base + "/health", headers=_headers(settings), method="GET")
        with urlopen(req, timeout=5) as response:
            payload = json.load(response)
        if not isinstance(payload, dict):
            raise TypeError("Invalid health payload")
        return {
            **result,
            "ready": payload.get("status") == "ok",
            "gpu": payload.get("gpu"),
            "models": payload.get("models", {}),
            "message": payload.get("message") or "Colab GPU API sẵn sàng.",
        }
    except (ColabGPUError, URLError, OSError, ValueError, TypeError) as exc:
        return {**result, "message": f"Không kết nối được Colab GPU API: {exc}"}


def validate_output(output: str, evidence: list, question: str = "") -> str | None:
    """Conservative lexical guard ensuring all numbers and citations match evidence."""
    citations = [int(n) for n in re.findall(r"\[(\d+)\]", output)]
    if not output.strip() or not citations or any(n < 1 or n > len(evidence) for n in citations):
        return "Phản hồi Colab thiếu trích dẫn hợp lệ; đang trả bằng chứng gốc."
    from .financial_tables import fold

    if any(term in fold(question) for term in ["tai sao", "nguyen nhan", "vi sao"]):
        abstentions = [
            "chua du bang chung",
            "khong du bang chung",
            "khong giai thich",
            "khong cung cap",
            "khong neu",
            "khong the xac dinh",
        ]
        if any(
            not any(term in fold(line) for term in abstentions)
            for line in output.splitlines()
            if line.strip()
        ):
            return (
                "Chưa xác minh được lập luận nguyên nhân; chỉ hiển thị bằng chứng gốc, "
                "không suy biến động số liệu thành nguyên nhân."
            )
    for paragraph in output.splitlines():
        if not paragraph.strip():
            continue
        refs = [int(n) for n in re.findall(r"\[(\d+)\]", paragraph)]
        prose = re.sub(r"\[\d+\]", "", paragraph)
        prose = re.sub(r"^\s*\d+[.)]\s+", "", prose)
        values = re.findall(r"(?<!\w)\d+(?:[.,]\d+)*(?!\w)", prose)
        if values:
            source = " ".join(evidence[n - 1].text for n in refs)
            allowed = set(re.findall(r"(?<!\w)\d+(?:[.,]\d+)*(?!\w)", source))
            if any(value not in allowed for value in values):
                return "Phản hồi Colab nêu số không đối chiếu được với nguồn được dẫn; đang trả bằng chứng gốc."
    return None


def synthesize(settings, question: str, evidence_text: str) -> tuple[str, str | None, float | None]:
    remote_base(settings)
    if len(evidence_text) > 12000:
        raise ColabGPUError("Bằng chứng quá dài; giảm top_k hoặc hỏi cụ thể hơn.")
    if not _generation_lock.acquire(blocking=False):
        raise ColabGPUError("Colab GPU đang xử lý yêu cầu khác; đang trả bằng chứng gốc.")
    try:
        payload = _post_json(
            settings,
            "/v1/qa/synthesize",
            {"question": question, "evidence": evidence_text},
        )
        answer = payload.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            raise ColabGPUError("Colab GPU trả câu trả lời rỗng hoặc sai định dạng.")
        return (
            answer.strip(),
            payload.get("model"),
            float(payload["elapsed_seconds"]) if payload.get("elapsed_seconds") is not None else None,
        )
    finally:
        _generation_lock.release()


def parse_intent(settings, question: str, metrics: list[dict[str, str]]) -> dict[str, Any]:
    remote_base(settings)
    if not _generation_lock.acquire(blocking=False):
        raise ColabGPUError("Colab GPU đang bận.")
    try:
        return _post_json(
            settings,
            "/v1/qa/parse_intent",
            {"question": question, "metrics": metrics},
        )
    finally:
        _generation_lock.release()


def ocr_page_remote(settings, png_bytes: bytes, *, dpi: int = 200) -> dict[str, Any]:
    return _post_json(
        settings,
        "/v1/ocr",
        {"image_base64": base64.b64encode(png_bytes).decode("ascii"), "dpi": dpi},
    )


def parse_chart_remote(settings, png_bytes: bytes, *, title: str = "") -> dict[str, Any]:
    return _post_json(
        settings,
        "/v1/chart/parse",
        {"image_base64": base64.b64encode(png_bytes).decode("ascii"), "title": title},
    )


def embed_texts(settings, texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    payload = _post_json(settings, "/v1/embed", {"texts": texts})
    vectors = payload.get("embeddings")
    if not isinstance(vectors, list) or len(vectors) != len(texts):
        raise ColabGPUError("Colab GPU trả số lượng vector embedding không khớp.")
    return vectors
