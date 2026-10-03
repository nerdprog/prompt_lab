import json
import os
from types import SimpleNamespace

from google import genai

os.environ["ALLOW_MOCK_LLMS"] = "true"

from app.services.intent.task_understanding import understand_task
from app.services.rubric.rubric_generator import build_rubric


def test_rubric_has_valid_weights() -> None:
    task = understand_task("Create a study plan for me.")
    rubric = build_rubric(task)
    assert rubric.task_type in {"planning", "general", "explain_concept"}
    assert abs(rubric.total_weight - 1.0) < 1e-9


def test_photosynthesis_intent_is_preserved() -> None:
    task = understand_task("Explain photosynthesis I am five.")
    assert "5-year-old" in task.audience
    assert any("Avoid scientific jargon" in constraint for constraint in task.constraints)


def test_live_rubric_uses_string_scoring_scale_schema(monkeypatch) -> None:
    monkeypatch.setenv("ALLOW_MOCK_LLMS", "false")
    monkeypatch.setenv("GEMINI_OPTIMIZER_API_KEY", "optimizer-test-key")
    monkeypatch.setenv("GEMINI_OPTIMIZER_MODEL", "optimizer-test-model")
    captured: dict = {}

    class FakeModels:
        def generate_content(self, **kwargs):
            captured["request"] = kwargs
            return SimpleNamespace(text=json.dumps({
                "task_type": "planning",
                "criteria": [{
                    "criterion": "Intent Fidelity",
                    "description": "Follows the request.",
                    "weight": 1.0,
                    "scoring_scale": "0-10: fully satisfies the request.",
                }],
            }))

    class FakeClient:
        def __init__(self, *, api_key):
            assert api_key == "optimizer-test-key"
            self.models = FakeModels()

    monkeypatch.setattr(genai, "Client", FakeClient)
    task = understand_task("Create a study plan for me.")
    rubric = build_rubric(task, "Create a study plan for me.")

    assert rubric.criteria[0].scoring_scale == "0-10: fully satisfies the request."
    response_schema = captured["request"]["config"].response_schema
    scoring_scale_schema = response_schema["properties"]["criteria"]["items"]["properties"]["scoring_scale"]
    assert scoring_scale_schema["type"] == "STRING"
