import io
import json
from urllib.error import HTTPError, URLError

import pytest

from financial_rag.config import Settings
from financial_rag.models import Evidence
from financial_rag.ollama_client import OllamaError, generate, validate_output
from financial_rag.service import FinancialRAGService


def test_local_request_and_service(tmp_path, monkeypatch):
    requests = []

    def respond(req, timeout):
        requests.append(json.loads(req.data))
        return io.BytesIO(
            json.dumps({"done": True, "message": {"content": "100 tỷ [1]."}}).encode()
        )

    monkeypatch.setattr("financial_rag.ollama_client.urlopen", respond)
    service = FinancialRAGService(Settings(data_dir=tmp_path / "data", llm_provider="ollama"))
    path = tmp_path / "report.txt"
    path.write_text("Doanh thu năm 2024 đạt 100 tỷ đồng.", encoding="utf-8")
    service.ingest(path)
    result = service.ask("Doanh thu", use_llm=True)
    assert result.mode == "llm" and "[1]" in result.answer
    assert requests[0]["stream"] is False
    assert requests[0]["options"]["num_ctx"] == 4096


@pytest.mark.parametrize(
    "error,expected",
    [
        (URLError("offline"), "kết nối"),
        (HTTPError("http://localhost", 404, "missing", {}, None), "ollama pull"),
    ],
)
def test_unavailable_ollama(monkeypatch, error, expected):
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr("financial_rag.ollama_client.urlopen", fail)
    with pytest.raises(OllamaError, match=expected):
        generate(Settings(), "Question", "[1] Evidence")


def test_reject_external_endpoint():
    with pytest.raises(OllamaError, match="localhost"):
        generate(Settings(ollama_base_url="https://example.com"), "Question", "[1] Evidence")


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


def test_truncated_response_rejected(monkeypatch):
    monkeypatch.setattr(
        "financial_rag.ollama_client.urlopen",
        lambda *a, **k: io.BytesIO(
            json.dumps(
                {"done": True, "done_reason": "length", "message": {"content": "Bị cắt [1]"}}
            ).encode()
        ),
    )
    with pytest.raises(OllamaError, match="giới hạn"):
        generate(Settings(), "Question", "[1] Evidence")


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
