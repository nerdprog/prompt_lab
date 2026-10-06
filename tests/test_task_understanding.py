from __future__ import annotations

import json
from types import SimpleNamespace

from google import genai

from app.services.intent.task_understanding import understand_task, understand_task_from_llm


def test_semantic_roles_are_preserved_for_background_and_complexity_mismatch():
    task = understand_task("I'm an experienced DevOps engineer. Explain Docker like I'm completely new to it.")

    assert task.user_background == "Experienced DevOps engineer"
    assert task.user_knowledge_level is None
    assert task.audience is None
    assert task.desired_complexity is not None
    assert "new" in task.desired_complexity.lower() or "beginner" in task.desired_complexity.lower()
    assert task.output_format is None
    assert any("advanced" in item.lower() for item in task.forbidden_assumptions) or task.forbidden_assumptions


def test_task_understanding_uses_schema_and_repairs_invalid_field_names(monkeypatch):
    monkeypatch.setenv("ALLOW_MOCK_LLMS", "false")
    monkeypatch.setenv("GEMINI_OPTIMIZER_API_KEY", "optimizer-test-key")
    monkeypatch.setenv("GEMINI_PERFORMER_API_KEY", "performer-test-key")
    monkeypatch.setenv("GEMINI_OPTIMIZER_MODEL", "optimizer-test-model")
    responses = iter([
        '{"primary_goal":"Create a study plan"}',
        json.dumps({
            "primary_intent": "Create a 7-day Python study plan",
            "task_category": "planning",
        }),
    ])
    calls = []

    class FakeModels:
        def generate_content(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(text=next(responses))

    class FakeClient:
        def __init__(self, *, api_key):
            assert api_key == "optimizer-test-key"
            self.models = FakeModels()

    monkeypatch.setattr(genai, "Client", FakeClient)
    charged_calls = []
    result = understand_task_from_llm(
        "Create a 7-day Python study plan",
        on_llm_call=lambda: charged_calls.append(True),
    )

    assert result.primary_intent == "Create a 7-day Python study plan"
    assert result.source == "gemini"
    assert len(calls) == len(charged_calls) == 2
    assert all(
        call["config"].response_schema["properties"]["primary_intent"]["type"] == "STRING"
        and "primary_intent" in call["config"].response_schema["required"]
        for call in calls
    )
    assert "primary_intent" in calls[1]["contents"]
