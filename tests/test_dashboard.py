import io
import json

import pytest
from fastapi.testclient import TestClient

from financial_rag import api
from financial_rag.analytics import summarize
from financial_rag.config import Settings
from financial_rag.jobs import JobStore
from financial_rag.models import FinancialFact
from financial_rag.ollama_client import OllamaError, generate, probe
from financial_rag.service import FinancialRAGService
from financial_rag.storage import atomic_json


def fact(year, value, code="10", quality="consistent"):
    return FinancialFact(
        id=f"{year}-{code}",
        metric="Revenue" if code == "10" else "Profit",
        value=value,
        raw_value=str(value),
        unit="VND",
        period=year,
        chunk_id="row",
        source="report.pdf",
        page=10,
        confidence=0.7,
        document_id="a",
        period_label=f"{year}-Q1",
        period_kind="quarter",
        scope="separate",
        metric_code=code,
        statement="income",
        quality=quality,
    ).model_dump()


def test_summary_growth_ratios_and_sources():
    result = summarize(
        [fact(2026, 120), fact(2025, 100), fact(2026, 24, "60")], [{"status": "pass"}]
    )
    assert result["cards"][0]["comparison"]["value"] == "20.0"
    assert result["ratios"][0]["value"] == "20.0"
    assert result["ratios"][0]["fact_ids"] == ["2026-60", "2026-10"]
    assert result["checks"] == {"pass": 1}


@pytest.mark.parametrize(
    "values",
    [
        [fact(2026, 120, quality="needs_review")],
        [fact(2026, 120), fact(2026, 121)],
    ],
)
def test_summary_does_not_select_unreliable_value(values):
    result = summarize(values, [])
    assert result["cards"][0]["value"] is None
    assert result["cards"][0]["warning"]


def test_negative_baseline_not_growth():
    result = summarize([fact(2026, 120), fact(2025, -100)], [])
    assert result["cards"][0]["comparison"] is None


def test_atomic_failed_serialization_preserves_file(tmp_path):
    target = tmp_path / "state.json"
    atomic_json(target, {"valid": 1})
    with pytest.raises(ValueError):
        atomic_json(target, {"value": float("nan")})
    assert json.loads(target.read_text()) == {"valid": 1}
    assert not list(tmp_path.glob("*.tmp"))


def test_jobs_survive_restart_and_interruption(tmp_path):
    path = tmp_path / "jobs.json"
    store = JobStore(path)
    complete = store.create("a.pdf")
    store.update(complete, status="complete", result={"pages": 1})
    interrupted = store.create("b.pdf")
    store.update(interrupted, status="running", completed_pages=5)
    restored = JobStore(path)
    assert restored.get(complete)["status"] == "complete"
    assert restored.get(interrupted)["status"] == "interrupted"
    assert restored.get(interrupted)["completed_pages"] == 5
    assert restored.get(interrupted)["error"]


def test_job_queue_is_bounded(tmp_path):
    store = JobStore(tmp_path / "jobs.json")
    for _ in range(8):
        store.create("a.pdf")
    with pytest.raises(ValueError, match="8"):
        store.create("b.pdf")


def test_ollama_probe_and_busy_guard(monkeypatch):
    from financial_rag import ollama_client

    monkeypatch.setattr(
        ollama_client,
        "urlopen",
        lambda *a, **k: io.BytesIO(json.dumps({"models": [{"name": "qwen2.5:3b"}]}).encode()),
    )
    assert probe(Settings(llm_provider="ollama"))["ready"]
    assert not probe(Settings(llm_provider="ollama", ollama_model="missing"))["ready"]
    ollama_client._generation_lock.acquire()
    try:
        with pytest.raises(OllamaError, match="đang xử lý"):
            generate(Settings(), "Question", "[1] Evidence")
    finally:
        ollama_client._generation_lock.release()


@pytest.fixture
def client(tmp_path, monkeypatch):
    service = FinancialRAGService(Settings(data_dir=tmp_path / "data", ocr_enabled=False))
    monkeypatch.setattr(api, "service", service)
    monkeypatch.setattr(api, "jobs", JobStore(tmp_path / "jobs.json"))
    with TestClient(api.app) as client:
        yield client


def test_api_dashboard_export_graph_and_limits(client):
    ids = []
    for ticker in ["ABC", "XYZ"]:
        response = client.post(
            "/ingest",
            data={"ticker": ticker},
            files={
                "file": ("report.txt", f"Doanh thu năm 2024 đạt 100 tỷ đồng. {ticker}".encode())
            },
        )
        assert response.status_code == 200
        ids.append(response.json()["document_id"])
    analysis = client.get(f"/documents/{ids[0]}/analysis")
    assert analysis.status_code == 200 and "summary" in analysis.json()
    exported = client.get(f"/documents/{ids[0]}/export")
    assert exported.status_code == 200 and "attachment" in exported.headers["content-disposition"]
    assert str(api.service.settings.data_dir) not in exported.text
    graph = client.get(f"/documents/{ids[0]}/graph").json()
    assert all(n.get("document_id") in {None, ids[0]} for n in graph["nodes"])
    assert ids[1] not in {n["id"] for n in graph["nodes"]}
    allowed = {n["id"] for n in graph["nodes"]}
    assert all(e["source"] in allowed and e["target"] in allowed for e in graph["edges"])
    assert client.get(f"/documents/{ids[0]}/graph?limit=1").json()["truncated"]
    assert (
        client.post("/ask", json={"question": "abc", "document_id": "missing"}).status_code == 404
    )
    assert client.post("/ask", json={"question": "x" * 2001}).status_code == 422
    assert client.post("/ingest", files={"file": ("a.txt", b"")}).status_code == 400
    assert client.get("/assets/dashboard.js").status_code == 200


def test_background_job_completed_and_persisted(client, monkeypatch):
    class ImmediateExecutor:
        def submit(self, function, *args):
            function(*args)

    monkeypatch.setattr(api, "executor", ImmediateExecutor())
    response = client.post(
        "/ingest", data={"background": "true"}, files={"file": ("a.txt", b"Revenue 2026: 100 VND")}
    )
    job_id = response.json()["job_id"]
    assert client.get("/jobs/" + job_id).json()["status"] == "complete"
    assert JobStore(api.jobs.path).get(job_id)["result"]["pages"] == 1
    assert client.get("/jobs").json()[0]["id"] == job_id
