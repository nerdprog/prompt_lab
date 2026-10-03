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


def test_judge_batch_sends_shared_sample_once_and_keeps_candidate_results(monkeypatch):
    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured["request"] = kwargs
            items = []
            for candidate_id in ("cand-a", "cand-b"):
                items.append({
                    "candidate_id": candidate_id,
                    "overall_score": 8.0,
                    "criterion_scores": {
                        "Intent Fidelity": {
                            "score": 8.0,
                            "reason": "Meets the task.",
                        },
                    },
                    "strengths": ["Clear."],
                    "weaknesses": [],
                    "evidence": ["The response includes the requested plan."],
                    "root_cause": [],
                    "improvement_suggestion": [],
                    "confidence": 0.9,
                    "intent_fidelity": 9.0,
                    "unsupported_assumptions": [],
                })
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({"evaluations": items})))]
            )

    class FakeClient:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(openai, "OpenAI", FakeClient)
    judge = JudgeClient()
    judge.settings.allow_mock_llms = False
    judge.settings.groq_api_key = "offline-test-key"
    judge.settings.judge_model = "openai/gpt-oss-20b"
    shared_sample = "Create a 7-day Python study plan."

    results = judge.evaluate_batch(
        {"primary_intent": "Create a Python study plan."},
        [{"criterion": "Intent Fidelity", "weight": 1.0}],
        [
            {
                "candidate_id": candidate_id,
                "candidate_prompt": f"Candidate prompt for {candidate_id}",
                "sample": shared_sample,
                "response_text": f"Response for {candidate_id}",
            }
            for candidate_id in ("cand-a", "cand-b")
        ],
    )

    request = captured["request"]
    payload = json.loads(request["messages"][1]["content"])
    assert request["max_tokens"] == 4096
    assert payload["required_criterion_names"] == ["Intent Fidelity"]
    assert payload["shared_sample"] == shared_sample
    assert [item["candidate_id"] for item in payload["evaluations"]] == ["cand-a", "cand-b"]
    assert all("candidate_prompt" not in item and "sample" not in item for item in payload["evaluations"])
    item_schema = request["response_format"]["json_schema"]["schema"]["properties"]["evaluations"]["items"]
    assert item_schema["properties"]["candidate_id"]["enum"] == ["cand-a", "cand-b"]
    assert list(item_schema["properties"]["criterion_scores"]["properties"]) == ["Intent Fidelity"]
    assert item_schema["required"] == [
        "candidate_id",
        "overall_score",
        "criterion_scores",
        "strengths",
        "weaknesses",
        "evidence",
        "root_cause",
        "improvement_suggestion",
        "confidence",
        "intent_fidelity",
        "unsupported_assumptions",
    ]
    assert set(results) == {"cand-a", "cand-b"}
    assert all(item["source"] == "groq" for item in results.values())
