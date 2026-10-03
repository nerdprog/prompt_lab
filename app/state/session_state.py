from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

SESSIONS: dict[str, dict[str, Any]] = {}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_session(original_prompt: str, optional_context: str | None = None, preferences: str | None = None, config: dict[str, Any] | None = None) -> dict[str, Any]:
    session_id = f"sess-{uuid.uuid4().hex[:12]}"
    session = {
        "session_id": session_id,
        "original_prompt": original_prompt,
        "optional_context": optional_context,
        "preferences": preferences,
        "configuration": config or {},
        "status": "initialized",
        "stage": "pending",
        "stage_status": "pending",
        "progress_stages": {
            "understanding_task": "pending",
            "building_rubric": "pending",
            "generating_candidates": "pending",
            "ucb_selection": "pending",
            "performer_execution": "pending",
            "judge_evaluation": "pending",
            "stage_a_optimization": "pending",
            "insight_update": "pending",
            "next_iteration": "pending",
            "final_evaluation": "pending",
            "report_generation": "pending",
        },
        "stop_reason": None,
        "task_spec": None,
        "rubric": None,
        "samples": [],
        "current_iteration": 0,
        "max_iterations": config.get("max_iterations", 5) if config else 5,
        "current_stage": "understanding_task",
        "active_candidate_count": 0,
        "total_candidate_count": 0,
        "completed_performer_calls": 0,
        "completed_judge_calls": 0,
        "model_calls_used": 0,
        "model_call_budget": config.get("max_llm_calls", 30) if config else 30,
        "current_best_quality": None,
        "current_best_candidate_id": None,
        "current_best_prompt_token_count": None,
        "token_count_error": None,
        "retry_count": 0,
        "rate_limit_status": None,
        "rate_limit_provider": None,
        "rate_limit_retry_after": None,
        "rate_limit_retry_at": None,
        "rate_limit_message": None,
        "cancellation_requested": False,
        "error_code": None,
        "user_message": None,
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
        "completed_at": None,
        "candidates": [],
        "evaluations": [],
        "iterations": [],
        "active_candidate_ids": [],
        "ucb_history": [],
        "insights": [],
        "llm_call_count": 0,
        "started_at": None,
        "final_evaluation": None,
        "report_path": None,
        "provider_status": None,
    }
    SESSIONS[session_id] = session
    return session


def get_session(session_id: str) -> dict[str, Any] | None:
    return SESSIONS.get(session_id)


def update_session(session_id: str, **kwargs: Any) -> dict[str, Any]:
    session = SESSIONS.get(session_id)
    if session is None:
        raise KeyError(f"Session not found: {session_id}")
    session.update(kwargs)
    session["updated_at"] = utc_now_iso()
    return session


def append_candidate(session_id: str, candidate: dict[str, Any]) -> dict[str, Any]:
    session = get_session(session_id)
    if session is None:
        raise KeyError(f"Session not found: {session_id}")
    session.setdefault("candidates", []).append(candidate)
    session["updated_at"] = utc_now_iso()
    return candidate


def append_evaluation(session_id: str, evaluation: dict[str, Any]) -> dict[str, Any]:
    session = get_session(session_id)
    if session is None:
        raise KeyError(f"Session not found: {session_id}")
    session.setdefault("evaluations", []).append(evaluation)
    session["updated_at"] = utc_now_iso()
    return evaluation


def append_iteration(session_id: str, iteration: dict[str, Any]) -> dict[str, Any]:
    session = get_session(session_id)
    if session is None:
        raise KeyError(f"Session not found: {session_id}")
    session.setdefault("iterations", []).append(iteration)
    session["updated_at"] = utc_now_iso()
    return iteration


def append_insight(session_id: str, insight: dict[str, Any]) -> dict[str, Any]:
    session = get_session(session_id)
    if session is None:
        raise KeyError(f"Session not found: {session_id}")
    session.setdefault("insights", []).append(insight)
    session["updated_at"] = utc_now_iso()
    return insight


def list_candidates(session_id: str) -> list[dict[str, Any]]:
    return (get_session(session_id) or {}).get("candidates", [])


def clear_session(session_id: str) -> None:
    SESSIONS.pop(session_id, None)
