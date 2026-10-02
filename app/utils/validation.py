from __future__ import annotations

from app.core.config import get_settings


def validate_prompt(prompt: str) -> tuple[bool, str | None]:
    settings = get_settings()
    cleaned = (prompt or "").strip()
    if not cleaned:
        return False, "Please enter a prompt to optimize."
    if len(cleaned) < settings.min_prompt_chars:
        return False, f"Please enter at least {settings.min_prompt_chars} characters of meaningful text."
    if len(cleaned) > settings.max_prompt_chars:
        return False, "Your prompt is too long for reliable optimization. Please shorten it or provide the essential requirements."
    return True, None


def validate_config(payload: dict) -> list[str]:
    errors: list[str] = []
    if payload.get("active_candidates") is not None:
        try:
            active = int(payload["active_candidates"])
            if not 3 <= active <= 10:
                errors.append("Active candidates must be between 3 and 10.")
        except (TypeError, ValueError):
            errors.append("Active candidates must be an integer.")
    if payload.get("edited_candidates") is not None:
        try:
            edited = int(payload["edited_candidates"])
            if not 1 <= edited <= 6:
                errors.append("Edited candidates must be between 1 and 6.")
        except (TypeError, ValueError):
            errors.append("Edited candidates must be an integer.")
    if payload.get("max_iterations") is not None:
        try:
            iterations = int(payload["max_iterations"])
            if not 1 <= iterations <= 10:
                errors.append("Maximum iterations must be between 1 and 10.")
        except (TypeError, ValueError):
            errors.append("Maximum iterations must be an integer.")
    if payload.get("final_top_n") is not None:
        try:
            final_top_n = int(payload["final_top_n"])
            if not 1 <= final_top_n <= 10:
                errors.append("Final top N must be between 1 and 10.")
        except (TypeError, ValueError):
            errors.append("Final top N must be an integer.")
    if payload.get("ucb_c") is not None:
        try:
            ucb_c = float(payload["ucb_c"])
            if ucb_c <= 0:
                errors.append("UCB exploration constant must be greater than zero.")
        except (TypeError, ValueError):
            errors.append("UCB exploration constant must be numeric.")
    if payload.get("stagnation_limit") is not None:
        try:
            stagnation = int(payload["stagnation_limit"])
            if stagnation < 0:
                errors.append("Stagnation limit must be zero or greater.")
        except (TypeError, ValueError):
            errors.append("Stagnation limit must be an integer.")
    if payload.get("min_improvement") is not None:
        try:
            min_improvement = float(payload["min_improvement"])
            if min_improvement < 0:
                errors.append("Minimum improvement threshold cannot be negative.")
        except (TypeError, ValueError):
            errors.append("Minimum improvement threshold must be numeric.")
    return errors
