from __future__ import annotations

from typing import Any

from app.services.insights.insight_store import generate_insights


class FeedbackOptimizer:
    def __init__(self) -> None:
        pass

    def prepare(self, task_intent: dict[str, Any], feedback_entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return generate_insights(task_intent, feedback_entries)
