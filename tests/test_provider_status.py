from __future__ import annotations

from datetime import datetime as RealDateTime, timedelta, timezone
from threading import Event, Thread

from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import app
from app.services.optimizer.optimization_service import OptimizationService
from app.services.providers.health import (
    classify_provider_exception,
    classify_provider_error,
    provider_status,
    safe_provider_error_message,
)
from app.state.session_state import create_session


def test_gemini_roles_do_not_fall_back_to_legacy_shared_key(monkeypatch):
    monkeypatch.setenv("GEMINI_OPTIMIZER_API_KEY", "")
    monkeypatch.setenv("GEMINI_PERFORMER_API_KEY", "")
    monkeypatch.setenv("GEMINI_API_KEY", "legacy-only-test-value")
    monkeypatch.setenv("GEMINI_OPTIMIZER_MODEL", "optimizer-test-model")
    monkeypatch.setenv("GEMINI_PERFORMER_MODEL", "performer-test-model")
    settings = get_settings()

    assert settings.optimizer_key is None
    assert settings.performer_key is None
    assert not settings.has_optimizer_config
    assert not settings.has_performer_config
    assert settings.optimizer_model == "optimizer-test-model"
    assert settings.performer_model == "performer-test-model"

    monkeypatch.setenv("GEMINI_OPTIMIZER_API_KEY", "optimizer-test-value")
    monkeypatch.setenv("GEMINI_PERFORMER_API_KEY", "performer-test-value")
    settings = get_settings()
    assert settings.optimizer_key == "optimizer-test-value"
    assert settings.performer_key == "performer-test-value"


def test_provider_status_is_role_specific_and_never_claims_mock_working(monkeypatch):
    monkeypatch.setenv("ALLOW_MOCK_LLMS", "true")
    monkeypatch.setenv("GEMINI_OPTIMIZER_API_KEY", "optimizer-test-value")
    monkeypatch.setenv("GEMINI_PERFORMER_API_KEY", "")
    monkeypatch.setenv("GROQ_API_KEY", "")
    statuses = provider_status(force_refresh=True)

    assert set(statuses) == {"optimizer", "performer", "judge"}
    assert statuses["optimizer"]["configured"] is True
    assert statuses["optimizer"]["working"] is False
    assert statuses["optimizer"]["status"] == "mock"
    assert statuses["performer"]["configured"] is False
    assert statuses["judge"]["configured"] is False
    assert "test-value" not in str(statuses)


def test_provider_status_api_has_exactly_three_safe_provider_records(monkeypatch):
    monkeypatch.setenv("ALLOW_MOCK_LLMS", "true")
    monkeypatch.setenv("GEMINI_OPTIMIZER_API_KEY", "")
    monkeypatch.setenv("GEMINI_PERFORMER_API_KEY", "")
    monkeypatch.setenv("GROQ_API_KEY", "")
    response = TestClient(app).get("/api/providers/status")

    assert response.status_code == 200
    assert set(response.json()["data"]) == {"optimizer", "performer", "judge"}
    assert all(record["working"] is False for record in response.json()["data"].values())


def test_rate_limit_classifier_recognizes_quota_and_authentication():
    assert classify_provider_error("HTTP 429 too many requests") == "RATE_LIMITED"
    assert classify_provider_error("RESOURCE_EXHAUSTED quota") == "QUOTA_EXCEEDED"
    assert classify_provider_error("401 UNAUTHENTICATED") == "AUTHENTICATION_FAILED"


def test_temporary_provider_outage_is_classified_and_safely_described():
    message = "503 UNAVAILABLE: This model is currently experiencing high demand."
    assert classify_provider_error(message) == "TEMPORARILY_UNAVAILABLE"
    assert "temporarily unavailable" in safe_provider_error_message(
        "TEMPORARILY_UNAVAILABLE", "Gemini Optimizer"
    )


def test_provider_exception_classification_uses_nested_structured_error():
    try:
        try:
            raise ValueError("response schema validation failed")
        except ValueError as cause:
            raise RuntimeError("provider request failed after repair") from cause
    except RuntimeError as error:
        assert classify_provider_exception(error) == "MALFORMED_RESPONSE"


def test_missing_schema_fields_are_not_classified_as_missing_credentials():
    assert classify_provider_error(
        "2 validation errors for CandidateGenerationOutput: "
        "prompt_text Field required [type=missing]"
    ) == "MALFORMED_RESPONSE"


def test_start_endpoint_surfaces_safe_temporary_provider_error(monkeypatch):
    from app.api.routes_optimization import service as api_service

    def raise_unavailable(_request):
        raise RuntimeError("503 UNAVAILABLE high demand; secret=not-for-response")

    monkeypatch.setattr(api_service, "start", raise_unavailable)
    response = TestClient(app).post(
        "/api/optimization/start",
        json={"prompt": "Create a useful study guide with examples."},
    )

    assert response.status_code == 503
    body = response.json()
    assert body["error"]["code"] == "temporarily_unavailable"
    assert "high demand" in body["error"]["message"]
    assert "not-for-response" not in response.text


def test_rate_limited_provider_call_exposes_pause_then_retries(monkeypatch):
    monkeypatch.setenv("ALLOW_MOCK_LLMS", "false")
    session = create_session("A sufficiently long prompt", config={"max_llm_calls": 5})
    session["status"] = "running"
    service = OptimizationService()
    monkeypatch.setattr(service, "_rate_limit_delay", lambda _: 20)
    entered_wait = Event()
    resume = Event()
    clock = [0]

    class FakeDateTime:
        @classmethod
        def now(cls, tz=None):
            return RealDateTime(2030, 1, 1, tzinfo=tz or timezone.utc) + timedelta(seconds=clock[0])

    def fake_sleep(seconds):
        entered_wait.set()
        assert resume.wait(2)
        clock[0] += seconds

    monkeypatch.setattr("app.services.optimizer.optimization_service.datetime", FakeDateTime)
    monkeypatch.setattr("app.services.optimizer.optimization_service.time.sleep", fake_sleep)
    results = iter([
        {"status": "failed", "error_code": "RATE_LIMITED"},
        {"status": "success"},
    ])
    returned = []
    worker = Thread(target=lambda: returned.append(
        service._retry_provider_call(session, "Gemini Performer", lambda: next(results))
    ))
    worker.start()
    assert entered_wait.wait(2)
    assert session["status"] == "paused_rate_limit"
    assert service.get_status(session["session_id"])["rate_limit_status"] == "waiting"
    resume.set()
    worker.join(2)

    assert not worker.is_alive()
    assert returned[0]["status"] == "success"
    assert session["retry_count"] == 1
    assert session["rate_limit_status"] == "cleared"
    assert session["status"] == "running"


def test_rate_limited_provider_call_uses_two_bounded_retries(monkeypatch):
    monkeypatch.setenv("ALLOW_MOCK_LLMS", "false")
    session = create_session("A sufficiently long prompt", config={"max_llm_calls": 5})
    session["status"] = "running"
    session["current_stage"] = "judge_evaluation"
    session["progress_stages"]["judge_evaluation"] = "running"
    service = OptimizationService()
    monkeypatch.setattr(service, "_rate_limit_delay", lambda _: 0)
    calls = []

    def rate_limited_operation():
        calls.append(1)
        return {"error_code": "RATE_LIMITED"}

    try:
        service._retry_provider_call(session, "Groq Judge", rate_limited_operation)
    except RuntimeError as exc:
        assert "after 2 retries" in str(exc)
    else:
        raise AssertionError("Expected the bounded provider retries to exhaust.")

    assert len(calls) == 3
    assert session["retry_count"] == 2
    assert session["status"] == "failed"
    assert session["stop_reason"] == "rate_limit_exhausted"
    assert session["rate_limit_status"] == "exhausted"


def test_rate_limit_delay_uses_provider_retry_hint():
    assert OptimizationService._rate_limit_delay(
        RuntimeError("Please try again in 3.0225s.")
    ) == 4


def test_cancel_endpoint_marks_the_session_cancelled_and_requested():
    session = create_session("A sufficiently long prompt", config={})
    session["status"] = "running"

    response = TestClient(app).post(f"/api/optimization/{session['session_id']}/cancel")

    assert response.status_code == 200
    assert session["status"] == "cancelled"
    assert session["cancellation_requested"] is True
