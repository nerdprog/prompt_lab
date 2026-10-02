import os

os.environ.setdefault("ALLOW_MOCK_LLMS", "true")

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
