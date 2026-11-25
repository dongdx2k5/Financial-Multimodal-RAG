import io
import json

import pytest
from fastapi.testclient import TestClient

from financial_rag import api
from financial_rag.analytics import summarize
from financial_rag.colab_client import ColabGPUError, probe, synthesize
from financial_rag.config import Settings
from financial_rag.jobs import JobStore
from financial_rag.models import FinancialFact
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


def test_colab_probe_and_busy_guard(monkeypatch):
    from financial_rag import colab_client

    monkeypatch.setattr(
        colab_client,
        "urlopen",
        lambda *a, **k: io.BytesIO(
            json.dumps({"status": "ok", "gpu": {"name": "Tesla T4"}}).encode()
        ),
    )
    res = probe(Settings(colab_gpu_url="https://demo.trycloudflare.com"))
    assert res["ready"]
    assert res["gpu"]["name"] == "Tesla T4"
    colab_client._generation_lock.acquire()
    try:
        with pytest.raises(ColabGPUError, match="đang xử lý"):
            synthesize(Settings(colab_gpu_url="https://demo.trycloudflare.com"), "Question", "[1] Evidence")
    finally:
        colab_client._generation_lock.release()


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
    assert client.get("/assets/styles.css").status_code == 200
    assert client.get("/health/colab").status_code == 200

    # Verify human-in-the-loop fact verification endpoint + cross-document compare endpoint
    analysis_data = client.get(f"/documents/{ids[0]}/analysis").json()
    first_fact_id = analysis_data["facts"][0]["id"]
    patched = client.patch(
        f"/facts/{first_fact_id}",
        json={"value": 105.0, "raw_value": "105", "quality": "verified"},
    )
    assert patched.status_code == 200
    assert patched.json()["fact"]["quality"] == "verified"

    cmp_resp = client.post(
        "/compare",
        json={
            "left_document_id": ids[0],
            "right_document_id": ids[1],
            "statement": "income",
            "metric_code": "10",
            "operation": "difference",
        },
    )
    assert cmp_resp.status_code == 422  # TXT regex facts have no circular-200 metric_code

    assert client.delete(f"/documents/{ids[1]}").status_code == 200
    assert len(client.get("/documents").json()) == 1


def test_human_verified_fact_updates_analysis_checks_and_allows_qa(client):
    from financial_rag.models import Chunk, Document, FinancialFact

    service = api.service
    chunks = [
        Chunk(
            id="c10",
            document_id="doc-v",
            source="v.pdf",
            page=10,
            text="Doanh thu thuần: 2026-Q1: 100 VND",
            financial_rows=[{"value": 100}],
            scope="separate",
            bbox=[10, 10, 200, 30],
        ),
        Chunk(
            id="c11",
            document_id="doc-v",
            source="v.pdf",
            page=10,
            text="Giá vốn hàng bán: 2026-Q1: 40 VND",
            financial_rows=[{"value": 40}],
            scope="separate",
            bbox=[10, 40, 200, 60],
        ),
        Chunk(
            id="c20",
            document_id="doc-v",
            source="v.pdf",
            page=10,
            text="Lợi nhuận gộp: 2026-Q1: 50 VND",  # OCR misread 60 as 50
            financial_rows=[{"value": 50}],
            scope="separate",
            bbox=[10, 70, 200, 90],
        ),
    ]
    facts = [
        FinancialFact(
            id="f10",
            document_id="doc-v",
            chunk_id="c10",
            source="v.pdf",
            page=10,
            metric="Doanh thu thuần",
            metric_code="10",
            value=100,
            raw_value="100",
            unit="VND",
            period=2026,
            period_label="2026-Q1",
            period_kind="quarter",
            scope="separate",
            statement="income",
            confidence=0.7,
        ),
        FinancialFact(
            id="f11",
            document_id="doc-v",
            chunk_id="c11",
            source="v.pdf",
            page=10,
            metric="Giá vốn hàng bán",
            metric_code="11",
            value=40,
            raw_value="40",
            unit="VND",
            period=2026,
            period_label="2026-Q1",
            period_kind="quarter",
            scope="separate",
            statement="income",
            confidence=0.7,
        ),
        FinancialFact(
            id="f20",
            document_id="doc-v",
            chunk_id="c20",
            source="v.pdf",
            page=10,
            metric="Lợi nhuận gộp",
            metric_code="20",
            value=50,
            raw_value="50",
            unit="VND",
            period=2026,
            period_label="2026-Q1",
            period_kind="quarter",
            scope="separate",
            statement="income",
            confidence=0.7,
            quality="needs_review",
        ),
    ]
    doc = Document(id="doc-v", source="v.pdf", name="v.pdf", pages=[], scope="separate")
    service.store.save_document(doc, metadata={"checks": [{"statement": "income", "status": "fail"}]})
    service.graph.add_document(doc, chunks, facts)
    service.retriever.upsert(chunks)
    service.store.save_facts(facts)

    res = client.patch(
        "/facts/f20",
        json={"value": 60.0, "raw_value": "60", "quality": "verified"},
    )
    assert res.status_code == 200
    assert res.json()["fact"]["quality"] == "verified"
    analysis = client.get("/documents/doc-v/analysis").json()
    income_20_checks = [c for c in analysis["metadata"]["checks"] if c.get("target_code") == "20"]
    assert income_20_checks and income_20_checks[0]["status"] == "pass"

    qa = client.post(
        "/ask", json={"question": "Lợi nhuận gộp quý 1/2026 là bao nhiêu?", "document_id": "doc-v"}
    ).json()
    assert qa["answer_status"] == "answered"
    assert "60 VND" in qa["answer"]


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
