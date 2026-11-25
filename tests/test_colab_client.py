import io
import json
from urllib.error import HTTPError, URLError

import pytest

from financial_rag.colab_client import (
    ColabGPUError,
    probe,
    remote_base,
    synthesize,
    validate_output,
)
from financial_rag.config import Settings
from financial_rag.models import Evidence
from financial_rag.service import FinancialRAGService


def test_service_uses_local_evidence_when_colab_unconfigured(tmp_path):
    service = FinancialRAGService(Settings(data_dir=tmp_path / "data", colab_gpu_url=None))
    path = tmp_path / "report.txt"
    path.write_text("Doanh thu năm 2024 đạt 100 tỷ đồng.", encoding="utf-8")
    service.ingest(path)
    result = service.ask("Doanh thu", use_llm=True)
    assert result.mode == "evidence" and "[1]" in result.answer
    assert result.answer_status == "evidence_only"
    assert any("COLAB_GPU_URL" in w for w in result.warnings)
    assert result.llm_model is None


def test_colab_grounded_synthesis_and_intent_fallback(tmp_path, monkeypatch):
    def fake_urlopen(req, timeout):
        url = req.full_url
        assert req.get_header("Authorization") == "Bearer secret-tok"
        if url.endswith("/v1/qa/parse_intent"):
            return io.BytesIO(json.dumps({"statement": None, "metric_code": None}).encode())
        if url.endswith("/v1/qa/synthesize"):
            return io.BytesIO(
                json.dumps(
                    {
                        "answer": "Doanh thu năm 2024 đạt 100 tỷ đồng [1].",
                        "model": "Qwen2.5-VL-3B-Instruct",
                        "elapsed_seconds": 0.42,
                    }
                ).encode()
            )
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr("financial_rag.colab_client.urlopen", fake_urlopen)
    service = FinancialRAGService(
        Settings(
            data_dir=tmp_path / "data",
            colab_gpu_url="https://demo-worker.trycloudflare.com",
            colab_api_token="secret-tok",
        )
    )
    path = tmp_path / "report.txt"
    path.write_text("Doanh thu năm 2024 đạt 100 tỷ đồng.", encoding="utf-8")
    service.ingest(path)
    result = service.ask("Tình hình doanh thu ra sao?", use_llm=True)
    assert result.mode == "llm"
    assert result.answer_status == "answered"
    assert result.llm_model == "Qwen2.5-VL-3B-Instruct"
    assert "[1]" in result.answer


@pytest.mark.parametrize(
    "error,expected",
    [
        (URLError("offline"), "Không kết nối"),
        (HTTPError("https://x.trycloudflare.com", 401, "unauth", {}, None), "COLAB_API_TOKEN"),
    ],
)
def test_unavailable_colab_gpu(monkeypatch, error, expected):
    monkeypatch.setattr("financial_rag.colab_client.urlopen", lambda *a, **k: (_ for _ in ()).throw(error))
    with pytest.raises(ColabGPUError, match=expected):
        synthesize(Settings(colab_gpu_url="https://x.trycloudflare.com"), "Question", "[1] Evidence")


def test_reject_insecure_remote_http_endpoint():
    with pytest.raises(ColabGPUError, match="HTTPS"):
        remote_base(Settings(colab_gpu_url="http://example.com"))
    assert remote_base(Settings(colab_gpu_url="http://127.0.0.1:8008")) == "http://127.0.0.1:8008"


@pytest.mark.parametrize(
    "output,valid",
    [
        ("Doanh thu 100 tỷ [1].", True),
        ("Doanh thu 999 tỷ [1].", False),
        ("Doanh thu 100 tỷ [2].", False),
        ("Doanh thu 100 tỷ.", False),
        ("Có doanh thu [1].\nDoanh thu 999 tỷ.", False),
    ],
)
def test_numeric_citation_guard(output, valid):
    evidence = [Evidence(chunk_id="a", source="a.txt", page=1, text="Doanh thu 100 tỷ", score=1)]
    assert (validate_output(output, evidence) is None) == valid


def test_causal_claim_not_proven_by_table():
    evidence = [
        Evidence(
            chunk_id="a", source="a.pdf", page=11, text="Dòng tiền kinh doanh (100) VND", score=1
        )
    ]
    assert validate_output(
        "Dòng tiền âm có thể do doanh thu không đủ bù chi phí [1].",
        evidence,
        "Tại sao dòng tiền âm?",
    )
    assert (
        validate_output(
            "Nguồn không giải thích nguyên nhân [1].", evidence, "Tại sao dòng tiền âm?"
        )
        is None
    )


def test_probe_unconfigured():
    status = probe(Settings(colab_gpu_url=None))
    assert not status["configured"]
    assert not status["ready"]
