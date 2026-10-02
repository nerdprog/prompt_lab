from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.schemas.task import OptimizationStartRequest
from app.services.optimizer.optimization_service import BudgetExhausted, OptimizationService
from app.state.session_state import create_session, get_session


def test_mock_session_completes_and_writes_report(monkeypatch, tmp_path):
    monkeypatch.setenv("ALLOW_MOCK_LLMS", "true")
    monkeypatch.chdir(tmp_path)
    service = OptimizationService()
    session = service.start(OptimizationStartRequest(
        prompt="Explain photosynthesis I am five.",
        active_candidates=3,
        edited_candidates=1,
        max_iterations=1,
        final_top_n=2,
        max_llm_calls=20,
    ))

    service.confirm_and_prepare(session["session_id"], session["task_spec"])
    service.run_optimization(session["session_id"])

    completed = get_session(session["session_id"])
    assert completed is not None
    assert completed["status"] == "completed"
    assert completed["final_evaluation"]["top3"]
    assert completed["final_evaluation"]["baseline"]["score"] is not None
    assert completed["iterations"][0]["edited_candidates"]
    assert completed["insights"]
    assert all(candidate["source"] == "mock" for candidate in completed["candidates"])
    assert completed.get("report_error") is None, completed.get("report_error")
    assert Path(completed["report_path"]).read_bytes().startswith(b"%PDF")


def test_insights_keep_evaluation_provenance_and_merge(monkeypatch):
    monkeypatch.setenv("ALLOW_MOCK_LLMS", "true")
    session = create_session("An original prompt", config={})
    service = OptimizationService()
    from app.services.insights.insight_store import generate_insights

    first = generate_insights(
        {"task_category": "explain_concept"},
        [{
            "candidate_id": "A1",
            "evaluation_id": "eval-a1",
            "weaknesses": ["Missing the core mechanism."],
            "improvement_suggestion": ["Explain sunlight and water."],
            "root_cause": ["The prompt omitted key details."],
            "evidence": ["No mechanism was present."],
            "confidence": 0.85,
        }],
        1,
    )
    service._merge_insights(session, first, session["session_id"])
    repeated = generate_insights(
        {"task_category": "explain_concept"},
        [{
            "candidate_id": "A2",
            "evaluation_id": "eval-a2",
            "weaknesses": ["Missing the core mechanism."],
            "improvement_suggestion": [],
            "root_cause": ["The response skipped details."],
            "evidence": [],
            "confidence": 0.9,
        }],
        2,
    )
    service._merge_insights(session, repeated, session["session_id"])

    matching = [item for item in session["insights"] if item["text"] == "Missing the core mechanism."]
    assert len(matching) == 1
    assert matching[0]["confidence"] == 0.9
    assert {item["candidate_id"] for item in matching[0]["provenance"]} == {"A1", "A2"}


def test_call_budget_rejects_calls_after_limit(monkeypatch):
    monkeypatch.setenv("ALLOW_MOCK_LLMS", "false")
    service = OptimizationService()
    session = create_session("Prompt with sufficient length", config={"max_llm_calls": 1})
    service._charge(session)
    try:
        service._charge(session)
    except BudgetExhausted:
        pass
    else:
        raise AssertionError("Expected the call budget to reject the second provider call.")
    assert session["llm_call_count"] == 1


def test_cancelled_session_is_not_restarted_and_gets_partial_report(monkeypatch, tmp_path):
    monkeypatch.setenv("ALLOW_MOCK_LLMS", "true")
    monkeypatch.chdir(tmp_path)
    service = OptimizationService()
    session = create_session("Prompt with sufficient length", config={"max_iterations": 1})
    session["status"] = "cancelled"
    service.run_optimization(session["session_id"])
    assert session["status"] == "cancelled"
    assert session["current_iteration"] == 0
    assert Path(session["report_path"]).read_bytes().startswith(b"%PDF")


def test_api_confirmation_runs_mock_optimization_and_downloads_pdf(monkeypatch, tmp_path):
    monkeypatch.setenv("ALLOW_MOCK_LLMS", "true")
    monkeypatch.chdir(tmp_path)
    from app.api.routes_optimization import service as api_service
    from app.main import app

    settings = get_settings()
    monkeypatch.setattr(api_service, "settings", settings)
    monkeypatch.setattr(api_service.performer, "settings", settings)
    monkeypatch.setattr(api_service.judge, "settings", settings)
    monkeypatch.setattr(api_service.generator.optimizer, "settings", settings)

    client = TestClient(app)
    rejected = client.post("/api/optimization/start", json={"prompt": "x" * 12001})
    assert rejected.status_code == 422
    home = client.get("/")
    assert home.headers["x-content-type-options"] == "nosniff"
    assert home.headers["x-frame-options"] == "DENY"
    assert "default-src 'self'" in home.headers["content-security-policy"]
    html = home.text
    assert 'data-view="session"' in html
    assert 'data-view="about"' in html
    assert 'id="session-view"' in html
    assert 'id="about-view"' in html
    app_js = client.get("/static/js/app.js")
    assert app_js.status_code == 200
    assert "function refreshCurrentSession()" in app_js.text
    assert "function renderCurrentSession(session)" in app_js.text
    started = client.post("/api/optimization/start", json={
        "prompt": "Explain photosynthesis I am five.",
        "active_candidates": 3,
        "edited_candidates": 1,
        "max_iterations": 1,
        "final_top_n": 2,
    })
    assert started.status_code == 200
    session_id = started.json()["data"]["sessionId"]
    task_spec = started.json()["data"]["taskSpec"]

    confirmed = client.post(
        f"/api/optimization/{session_id}/confirm-understanding",
        json={"task_spec": task_spec},
    )
    assert confirmed.status_code == 200
    result = client.get(f"/api/optimization/{session_id}/results")
    assert result.json()["data"]["status"] == "completed"
    current_session = client.get(f"/api/optimization/{session_id}")
    assert current_session.status_code == 200
    assert current_session.json()["data"]["final_evaluation"]["top3"]
    report = client.get(f"/api/optimization/{session_id}/report")
    assert report.status_code == 200
    assert report.headers["content-type"] == "application/pdf"
    assert report.content.startswith(b"%PDF")
