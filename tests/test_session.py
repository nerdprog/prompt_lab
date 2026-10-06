import os

os.environ.setdefault("ALLOW_MOCK_LLMS", "true")

from fastapi.testclient import TestClient

from app.main import app
from app.state.session_state import create_session, get_session


client = TestClient(app)


def test_create_session_in_memory() -> None:
    session = create_session("Create a study plan for me.")
    assert session["session_id"].startswith("sess-")
    assert get_session(session["session_id"]) is not None


def test_start_endpoint_returns_session() -> None:
    response = client.post("/api/optimization/start", json={"prompt": "Create a study plan for me.", "active_candidates": 3})
    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert "sessionId" in payload["data"]


def test_backend_exposes_optimization_default_config(monkeypatch) -> None:
    monkeypatch.setenv("DEFAULT_ACTIVE_CANDIDATES", "6")
    monkeypatch.setenv("DEFAULT_EDITED_CANDIDATES", "4")
    monkeypatch.setenv("DEFAULT_MAX_ITERATIONS", "7")
    monkeypatch.setenv("DEFAULT_FINAL_TOP_N", "2")
    monkeypatch.setenv("DEFAULT_UCB_C", "1.5")
    monkeypatch.setenv("DEFAULT_STAGNATION_LIMIT", "3")
    monkeypatch.setenv("DEFAULT_MIN_IMPROVEMENT", "0.02")
    monkeypatch.setenv("DEFAULT_MAX_LLM_CALLS", "150")

    response = client.get("/api/optimization/defaults")
    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["data"]["active_candidates"] == 6
    assert payload["data"]["edited_candidates"] == 4
    assert payload["data"]["max_iterations"] == 7
    assert payload["data"]["final_top_n"] == 2
    assert payload["data"]["ucb_c"] == 1.5
    assert payload["data"]["stagnation_limit"] == 3
    assert payload["data"]["min_improvement"] == 0.02
    assert payload["data"]["max_llm_calls"] == 150


def test_final_eval_groq_key_can_be_configured_separately(monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "primary-groq-key")
    monkeypatch.setenv("FINAL_EVAL_GROQ_API_KEY", "final-eval-groq-key")

    from app.core.config import get_settings

    settings = get_settings()
    assert settings.groq_api_key == "primary-groq-key"
    assert settings.final_evaluation_groq_api_key == "final-eval-groq-key"
