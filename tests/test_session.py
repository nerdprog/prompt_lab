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
