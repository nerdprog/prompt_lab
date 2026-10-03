from __future__ import annotations

import json
from typing import Callable

from pydantic import ValidationError

from app.core.config import get_settings
from app.schemas.task import RubricCriterion, TaskIntent, TaskRubric
from app.services.providers.health import classify_provider_error

_RUBRIC_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "task_type": {"type": "STRING"},
        "criteria": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "criterion": {"type": "STRING"},
                    "description": {"type": "STRING"},
                    "weight": {"type": "NUMBER"},
                    "scoring_scale": {"type": "STRING"},
                },
                "required": ["criterion", "description", "weight", "scoring_scale"],
            },
        },
    },
    "required": ["task_type", "criteria"],
}


def _criterion(name: str, description: str, weight: float) -> RubricCriterion:
    return RubricCriterion(
        criterion=name,
        description=description,
        weight=weight,
        scoring_scale="0-10: 0 = unmet or incorrect; 10 = fully satisfied with clear evidence.",
    )


def _compose_rubric(intent: TaskIntent, original_task: str) -> TaskRubric:
    category = intent.task_category.lower()
    text = (original_task + " " + intent.primary_intent).lower()
    audience = (intent.audience or "").lower()
    dimensions: list[tuple[str, str, float]] = [
        ("Intent Fidelity", "How well the response satisfies the user's actual objective without distortion or unsupported assumptions.", 0.24)
    ]

    if category == "coding" or any(word in text for word in ("code", "function", "algorithm", "sort")):
        dimensions.extend([
            ("Functional Correctness", "Whether the code implements the requested behavior and handles relevant inputs correctly.", 0.31),
            ("Algorithmic Requirements", "Whether the requested algorithm or implementation constraint is followed.", 0.22),
            ("Code Quality", "Whether the implementation is readable, appropriately structured, and robust.", 0.13),
            ("Explanation Clarity", "Whether any explanation needed to use or understand the code is clear.", 0.10),
        ])
    elif category == "email" or "email" in text:
        dimensions.extend([
            ("Purpose and Completeness", "Whether the email clearly makes the requested request and includes the specified details.", 0.25),
            ("Professional Tone", "Whether the tone is appropriate for the recipient and workplace context.", 0.22),
            ("Email Structure", "Whether the response has useful email conventions such as a subject, greeting, body, and closing.", 0.16),
            ("Clarity", "Whether the request is concise, understandable, and unambiguous.", 0.13),
        ])
    elif category == "explain_concept" or any(word in text for word in ("explain", "describe", "understand")):
        dimensions.extend([
            ("Correctness", "Whether the explanation is accurate and avoids misleading claims.", 0.28),
            ("Audience Appropriateness", "Whether vocabulary and complexity suit the intended audience.", 0.24),
            ("Clarity", "Whether the explanation is easy to follow and well organized.", 0.15),
            ("Requirement Coverage", "Whether important requested aspects of the concept are addressed.", 0.09),
        ])
    elif category == "planning":
        dimensions.extend([
            ("Feasibility", "Whether the plan is realistic given stated constraints and resources.", 0.25),
            ("Coverage and Sequencing", "Whether relevant steps or topics are covered in a useful order.", 0.25),
            ("Actionability", "Whether the user can readily act on the plan.", 0.15),
            ("Clarity", "Whether the plan is organized and easy to understand.", 0.11),
        ])
    else:
        dimensions.extend([
            ("Instruction Coverage", "Whether explicit requirements and constraints are followed.", 0.27),
            ("Task Quality", "Whether the response is useful, correct for the task, and sufficiently complete.", 0.27),
            ("Clarity", "Whether the response is understandable and appropriately structured.", 0.22),
        ])

    if any(token in audience for token in ("child", "year-old", "beginner")) and not any(
        name == "Audience Appropriateness" for name, _, _ in dimensions
    ):
        dimensions.append(
            ("Audience Appropriateness", "Whether the response is adapted to the stated audience and level.", 0.10)
        )

    weight_total = sum(weight for _, _, weight in dimensions)
    criteria = [
        _criterion(name, description, weight / weight_total)
        for name, description, weight in dimensions
    ]
    return TaskRubric(task_type=category, criteria=criteria)


def _generate_with_gemini(
    intent: TaskIntent,
    original_task: str,
    on_llm_call: Callable[[], None] | None = None,
) -> TaskRubric:
    settings = get_settings()
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=settings.optimizer_key)
    request_data = {
        "original_task": original_task,
        "task_intent": intent.model_dump(),
        "requirements": [
            "Create criteria relevant to this specific task, not a generic prompt-writing rubric.",
            "Include Intent Fidelity as a first-class criterion.",
            "Weights must be positive and sum to 1.0.",
            "Use a 0-10 scoring scale and provide a clear scoring guide for each criterion as one string in scoring_scale, not an object.",
            "Return JSON only with task_type and criteria containing criterion, description, weight, scoring_scale.",
        ],
    }
    last_error: Exception | None = None
    for attempt in range(2):
        instruction = "Generate a task-specific rubric from the input JSON. " + json.dumps(request_data)
        if attempt:
            instruction += (
                " Repair the prior output to match the schema exactly. Each scoring_scale must be a string, "
                "not an object, and all weights must sum to 1.0."
            )
        if on_llm_call:
            on_llm_call()
        try:
            response = client.models.generate_content(
                model=settings.optimizer_model,
                contents=instruction,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=_RUBRIC_RESPONSE_SCHEMA,
                ),
            )
            text = getattr(response, "text", None) or ""
            if not text:
                raise ValueError("Rubric model returned empty output.")
            return TaskRubric.model_validate_json(text)
        except Exception as exc:
            if classify_provider_error(str(exc)) == "RATE_LIMITED":
                raise
            last_error = exc
    if isinstance(last_error, (json.JSONDecodeError, ValidationError, ValueError)):
        raise RuntimeError(
            "Task-specific rubric generation returned invalid structured output after one repair attempt."
        ) from last_error
    error_code = classify_provider_error(str(last_error))
    raise RuntimeError(
        f"Task-specific rubric generation failed ({error_code})."
    ) from last_error


def build_rubric(
    task_intent: TaskIntent,
    original_task: str = "",
    on_llm_call: Callable[[], None] | None = None,
) -> TaskRubric:
    settings = get_settings()
    if settings.has_optimizer_config and not settings.allow_mock_llms:
        return _generate_with_gemini(task_intent, original_task, on_llm_call)
    if not settings.allow_mock_llms:
        raise RuntimeError("GEMINI_OPTIMIZER_API_KEY is required to generate a task-specific rubric.")
    return _compose_rubric(task_intent, original_task)
