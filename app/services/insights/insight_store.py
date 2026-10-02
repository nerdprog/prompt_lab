from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _category(text: str) -> str:
    lowered = text.lower()
    if any(word in lowered for word in ("audience", "vocabulary", "jargon", "technical")):
        return "audience"
    if any(word in lowered for word in ("missing", "omitted", "does not", "doesn't", "unclear")):
        return "coverage"
    if any(word in lowered for word in ("assumption", "unsupported")):
        return "assumptions"
    return "task_quality"


def generate_insights(
    task_intent: dict[str, Any],
    feedback_entries: list[dict[str, Any]],
    iteration: int = 0,
) -> list[dict[str, Any]]:
    insights: list[dict[str, Any]] = []
    for evaluation in feedback_entries:
        weaknesses = evaluation.get("weaknesses", [])
        suggestions = evaluation.get("improvement_suggestion", [])
        root_causes = evaluation.get("root_cause", [])
        evidence = evaluation.get("evidence", [])
        observations = list(dict.fromkeys([*weaknesses, *suggestions]))
        for observation in observations:
            if not isinstance(observation, str) or not observation.strip():
                continue
            insight_evidence = list(dict.fromkeys([
                *[str(item) for item in evidence if item],
                *[str(item) for item in root_causes if item],
            ]))
            insights.append({
                "insight_id": f"ins-{uuid.uuid4().hex[:10]}",
                "category": _category(observation),
                "text": observation.strip(),
                "evidence": insight_evidence,
                "source_feedback": observation.strip(),
                "source_candidate_id": evaluation.get("candidate_id"),
                "source_iteration": iteration,
                "source_evaluation_id": evaluation.get("evaluation_id"),
                "task_category": task_intent.get("task_category", "general"),
                "scope": "session-level",
                "confidence": max(0.0, min(1.0, float(evaluation.get("confidence", 0.5)))),
                "provenance": [{
                    "candidate_id": evaluation.get("candidate_id"),
                    "iteration": iteration,
                    "evaluation_id": evaluation.get("evaluation_id"),
                    "evidence": insight_evidence,
                }],
                "created_at": _now(),
                "updated_at": _now(),
            })
    return insights
