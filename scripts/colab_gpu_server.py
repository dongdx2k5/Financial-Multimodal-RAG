"""Google Colab T4/L4/A100 GPU Inference Worker for Multimodal RAG Financial.

Run this server inside Google Colab and expose it via HTTPS tunnel (e.g. cloudflared)
to handle heavy GPU tasks while keeping Web UI, SQLite/FTS5 DB, and Decimal calculator
safely on the user's local machine.
"""

from __future__ import annotations

import base64
import hmac
import io
import json
import os
import re
import time
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Colab GPU Worker — Multimodal RAG Financial", version="0.2.0")

API_TOKEN = os.environ.get("COLAB_API_TOKEN", "").strip()
VLM_MODEL_ID = os.environ.get("COLAB_VLM_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
EMBED_MODEL_ID = os.environ.get("COLAB_EMBED_MODEL", "BAAI/bge-m3")
USE_4BIT = os.environ.get("COLAB_4BIT", "true").lower() in {"1", "true", "yes"}

_STATE: dict[str, Any] = {"vlm": None, "processor": None, "embedder": None, "ocr": None}


def verify_auth(authorization: str | None = Header(default=None)):
    if not API_TOKEN:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    token = authorization.removeprefix("Bearer ").strip()
    if not hmac.compare_digest(token, API_TOKEN):
        raise HTTPException(status_code=403, detail="Invalid COLAB_API_TOKEN")


class SynthesizeRequest(BaseModel):
    question: str
    evidence: str


class IntentRequest(BaseModel):
    question: str
    metrics: list[dict[str, str]]


class EmbedRequest(BaseModel):
    texts: list[str]


class ImageRequest(BaseModel):
    image_base64: str
    dpi: int = 200
    title: str = ""


def _gpu_info() -> dict[str, Any]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"available": False, "name": "CPU"}
        free, total = torch.cuda.mem_get_info(0)
        return {
            "available": True,
            "name": torch.cuda.get_device_name(0),
            "vram_free_gb": round(free / 1024**3, 2),
            "vram_total_gb": round(total / 1024**3, 2),
        }
    except ImportError:
        return {"available": False, "name": "unknown"}


def _get_vlm():
    if _STATE["vlm"] is not None:
        return _STATE["vlm"], _STATE["processor"]
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    kwargs: dict[str, Any] = {"device_map": "auto", "torch_dtype": torch.float16}
    if USE_4BIT:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
    processor = AutoProcessor.from_pretrained(VLM_MODEL_ID)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(VLM_MODEL_ID, **kwargs)
    model.eval()
    _STATE["vlm"] = model
    _STATE["processor"] = processor
    return model, processor


def _get_embedder():
    if _STATE["embedder"] is not None:
        return _STATE["embedder"]
    from sentence_transformers import SentenceTransformer

    device = "cuda" if _gpu_info().get("available") else "cpu"
    model = SentenceTransformer(EMBED_MODEL_ID, device=device)
    _STATE["embedder"] = model
    return model


def _run_vlm(messages: list[dict[str, Any]], *, images=None, max_new_tokens: int = 512) -> str:
    import torch

    model, processor = _get_vlm()
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=images, padding=True, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = generated[0][inputs["input_ids"].shape[1] :]
    return processor.decode(trimmed, skip_special_tokens=True).strip()


@app.get("/health", dependencies=[Depends(verify_auth)])
def health() -> dict[str, Any]:
    gpu = _gpu_info()
    return {
        "status": "ok",
        "gpu": gpu,
        "models": {
            "vlm": VLM_MODEL_ID,
            "embed": EMBED_MODEL_ID,
            "vlm_loaded": _STATE["vlm"] is not None,
            "embedder_loaded": _STATE["embedder"] is not None,
        },
        "message": f"Colab GPU Worker sẵn sàng ({gpu.get('name', 'CPU')}).",
    }


@app.post("/v1/qa/synthesize", dependencies=[Depends(verify_auth)])
def synthesize(req: SynthesizeRequest) -> dict[str, Any]:
    started = time.perf_counter()
    messages = [
        {
            "role": "system",
            "content": (
                "Trả lời ngắn gọn bằng tiếng Việt chỉ dựa trên bằng chứng. "
                "Mỗi ý một dòng và bắt buộc gắn trích dẫn [1], [2] đúng với bằng chứng. "
                "Không tự tính toán hoặc làm tròn số; giữ nguyên số và đơn vị gốc. "
                "Nếu nguồn không giải thích nguyên nhân thì phải nói rõ chưa đủ bằng chứng."
            ),
        },
        {
            "role": "user",
            "content": f"Câu hỏi: {req.question}\nBằng chứng:\n{req.evidence}",
        },
    ]
    answer = _run_vlm(messages, max_new_tokens=450)
    return {
        "answer": answer,
        "model": VLM_MODEL_ID,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


@app.post("/v1/qa/parse_intent", dependencies=[Depends(verify_auth)])
def parse_intent(req: IntentRequest) -> dict[str, Any]:
    catalog_text = "\n".join(
        f"- {m['statement']}.{m['metric_code']}: {m['phrase']}" for m in req.metrics
    )
    messages = [
        {
            "role": "system",
            "content": (
                "Trích xuất ý định câu hỏi tài chính thành JSON duy nhất với các khóa: "
                '{"statement": str|null, "metric_code": str|null, "periods": [str], '
                '"scope": "separate"|"consolidated"|null, "operation": "lookup"|"growth"|"difference"}. '
                "Chỉ chọn statement và metric_code từ danh sách bên dưới; nếu không khớp trả null."
            ),
        },
        {
            "role": "user",
            "content": f"Danh sách chỉ tiêu:\n{catalog_text}\n\nCâu hỏi: {req.question}",
        },
    ]
    raw = _run_vlm(messages, max_new_tokens=200)
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {"statement": None, "metric_code": None, "periods": [], "operation": "lookup"}
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {"statement": None, "metric_code": None, "periods": [], "operation": "lookup"}


@app.post("/v1/chart/parse", dependencies=[Depends(verify_auth)])
def parse_chart(req: ImageRequest) -> dict[str, Any]:
    from PIL import Image

    image = Image.open(io.BytesIO(base64.b64decode(req.image_base64))).convert("RGB")
    prompt = (
        f"Đọc biểu đồ tài chính (tiêu đề tham khảo: {req.title or 'không rõ'}) và trả về JSON duy nhất: "
        '{"title": str, "chart_type": str, "unit": str|null, '
        '"series": [{"name": str, "points": [{"label": str, "value": float, "raw_value": str}]}]}. '
        "Không đoán số không nhìn thấy trên biểu đồ."
    )
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": prompt}],
        }
    ]
    raw = _run_vlm(messages, images=[image], max_new_tokens=600)
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {"title": req.title, "series": []}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"title": req.title, "series": []}


@app.post("/v1/embed", dependencies=[Depends(verify_auth)])
def embed(req: EmbedRequest) -> dict[str, Any]:
    model = _get_embedder()
    vectors = model.encode(req.texts, normalize_embeddings=True).tolist()
    return {"embeddings": vectors, "model": EMBED_MODEL_ID}


@app.post("/v1/ocr", dependencies=[Depends(verify_auth)])
def remote_ocr(req: ImageRequest) -> dict[str, Any]:
    import numpy as np
    from PIL import Image

    image = Image.open(io.BytesIO(base64.b64decode(req.image_base64))).convert("RGB")
    if _STATE["ocr"] is None:
        from rapidocr import RapidOCR

        _STATE["ocr"] = RapidOCR()
    result = _STATE["ocr"](np.asarray(image)[:, :, ::-1])
    lines = []
    boxes = getattr(result, "boxes", None) or []
    txts = getattr(result, "txts", None) or []
    scores = getattr(result, "scores", None) or []
    for box, txt, score in zip(boxes, txts, scores, strict=False):
        lines.append(
            {
                "text": str(txt),
                "bbox": np.asarray(box, dtype=float).tolist(),
                "confidence": float(score) if score is not None else None,
            }
        )
    return {"lines": lines, "warnings": []}
