from __future__ import annotations

import json
from types import SimpleNamespace

import openai

from app.services.llm.judge import JudgeClient


def test_judge_uses_groq_openai_compatible_client(monkeypatch):
    rubric = [
        {"criterion": "Intent Fidelity", "weight": 0.4},
        {"criterion": "Clarity", "weight": 0.6},
    ]
    response_data = {
        "overall_score": 8.0,
        "criterion_scores": [
            {"criterion": "Intent Fidelity", "score": 9.0, "weight": 0.4, "reason": "Faithful."},
            {"criterion": "Clarity", "score": 7.0, "weight": 0.6, "reason": "Clear."},
        ],
        "strengths": ["Addresses the task."],
        "weaknesses": ["Could be clearer."],
        "evidence": ["The answer covers the requested points."],
        "root_cause": ["Some detail is missing."],
        "improvement_suggestion": ["Add one clarifying detail."],
        "confidence": 0.9,
        "intent_fidelity": 9.0,
        "unsupported_assumptions": [],
    }
    captured: dict = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured["request"] = kwargs
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(response_data)))]
            )

    class FakeClient:
        def __init__(self, **kwargs):
            captured["client"] = kwargs
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(openai, "OpenAI", FakeClient)
    judge = JudgeClient()
    judge.settings.allow_mock_llms = False
    judge.settings.groq_api_key = "offline-test-key"
    judge.settings.judge_model = "llama-3.3-70b-versatile"

    result = judge.evaluate(
        task_spec={"primary_intent": "Explain the topic"},
        rubric=rubric,
        candidate_prompt="Explain this clearly.",
        sample="Explain this",
        response_text="A response from the performer.",
    )

    assert captured["client"] == {
        "api_key": "offline-test-key",
        "base_url": "https://api.groq.com/openai/v1",
        "timeout": 45.0,
        "max_retries": 0,
    }
    request = captured["request"]
    assert request["model"] == "llama-3.3-70b-versatile"
    assert request["response_format"] == {"type": "json_object"}
    assert result["source"] == "groq"
    assert result["model"] == "llama-3.3-70b-versatile"
    assert result["authoritative_score"] == 0.78
