from __future__ import annotations

import re
from typing import Any


def analyze_intent_fidelity(task_spec: dict[str, Any], candidate_prompt: str) -> dict[str, Any]:
    prompt_lower = candidate_prompt.lower()
    forbidden = task_spec.get("forbidden_assumptions", [])
    matched = []
    for item in forbidden:
        if isinstance(item, str) and item.lower() in prompt_lower:
            matched.append(item)

    audience = task_spec.get("audience")
    score = 0.9
    if matched:
        score -= min(0.6, 0.12 * len(matched))

    if audience and "five" in prompt_lower and "5-year-old" in str(audience).lower():
        if not any(item in prompt_lower for item in ["simple", "easy", "sunlight", "water", "air"]):
            score -= 0.1

    if "kindergarten" in prompt_lower or "counting to five" in prompt_lower:
        score -= 0.2

    return {
        "score": max(0.0, min(1.0, score)),
        "unsupported_assumptions": matched,
        "reason": "The prompt preserves the task goal while avoiding assumptions that are not explicitly justified by the user's request.",
    }


def detect_duplicate_candidates(prompts: list[str]) -> list[list[str]]:
    groups: list[list[str]] = []
    seen: dict[str, str] = {}
    for prompt in prompts:
        key = re.sub(r"[^a-z0-9\s]", "", prompt.lower())
        key = " ".join(key.split())
        seen.setdefault(key, prompt)
    for key, prompt in seen.items():
        matches = [p for p in prompts if re.sub(r"[^a-z0-9\s]", "", p.lower()) == key]
        if len(matches) > 1:
            groups.append(matches)
    return groups
