from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    gemini_optimizer_api_key: str | None = None
    gemini_performer_api_key: str | None = None
    gemini_api_key: str | None = None
    groq_api_key: str | None = None
    port: int = 8000
    optimizer_model: str = "gemini-2.0-flash"
    performer_model: str = "gemini-2.0-flash"
    judge_model: str = "llama-3.3-70b-versatile"
    default_active_candidates: int = 6
    default_edited_candidates: int = 3
    default_max_iterations: int = 5
    default_final_top_n: int = 3
    default_ucb_c: float = 1.414
    default_stagnation_limit: int = 2
    default_min_improvement: float = 0.01
    default_max_llm_calls: int = 200
    min_prompt_chars: int = 5
    max_prompt_chars: int = 12000
    allow_mock_llms: bool = False

    @property
    def optimizer_key(self) -> str | None:
        return self.gemini_optimizer_api_key

    @property
    def performer_key(self) -> str | None:
        return self.gemini_performer_api_key

    @property
    def has_optimizer_config(self) -> bool:
        return bool(self.optimizer_key)

    @property
    def has_performer_config(self) -> bool:
        return bool(self.performer_key)

    @property
    def has_gemini_config(self) -> bool:
        return self.has_optimizer_config or self.has_performer_config

    @property
    def has_groq_config(self) -> bool:
        return bool(self.groq_api_key)

    @property
    def has_llm_config(self) -> bool:
        return self.has_optimizer_config and self.has_performer_config and self.has_groq_config


def _coerce_int(value: Any, default: int) -> int:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any, default: float) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def get_settings() -> Settings:
    legacy_gemini = os.getenv("GEMINI_API_KEY") or None
    return Settings(
        gemini_optimizer_api_key=os.getenv("GEMINI_OPTIMIZER_API_KEY") or None,
        gemini_performer_api_key=os.getenv("GEMINI_PERFORMER_API_KEY") or None,
        gemini_api_key=legacy_gemini,
        groq_api_key=os.getenv("GROQ_API_KEY") or None,
        port=_coerce_int(os.getenv("PORT"), 8000),
        optimizer_model=os.getenv("GEMINI_OPTIMIZER_MODEL") or os.getenv("OPTIMIZER_MODEL") or "gemini-2.0-flash",
        performer_model=os.getenv("GEMINI_PERFORMER_MODEL") or os.getenv("PERFORMER_MODEL") or "gemini-2.0-flash",
        judge_model=os.getenv("JUDGE_MODEL") or "llama-3.3-70b-versatile",
        default_active_candidates=_coerce_int(os.getenv("DEFAULT_ACTIVE_CANDIDATES"), 6),
        default_edited_candidates=_coerce_int(os.getenv("DEFAULT_EDITED_CANDIDATES"), 3),
        default_max_iterations=_coerce_int(os.getenv("DEFAULT_MAX_ITERATIONS"), 5),
        default_final_top_n=_coerce_int(os.getenv("DEFAULT_FINAL_TOP_N"), 3),
        default_ucb_c=_coerce_float(os.getenv("DEFAULT_UCB_C"), 1.414),
        default_stagnation_limit=_coerce_int(os.getenv("DEFAULT_STAGNATION_LIMIT"), 2),
        default_min_improvement=_coerce_float(os.getenv("DEFAULT_MIN_IMPROVEMENT"), 0.01),
        default_max_llm_calls=_coerce_int(os.getenv("DEFAULT_MAX_LLM_CALLS"), 30),
        min_prompt_chars=_coerce_int(os.getenv("MIN_PROMPT_CHARS"), 5),
        max_prompt_chars=_coerce_int(os.getenv("MAX_PROMPT_CHARS"), 12000),
        allow_mock_llms=str(os.getenv("ALLOW_MOCK_LLMS", "false")).lower() in {"1", "true", "yes"},
    )


def require_runtime_keys() -> None:
    """Validate configuration for production path when actual external LLM calls are required."""
    settings = get_settings()
    if not settings.has_optimizer_config or not settings.has_performer_config or not settings.has_groq_config:
        raise RuntimeError(
            "Missing required LLM API keys. Configure GEMINI_OPTIMIZER_API_KEY, GEMINI_PERFORMER_API_KEY, and GROQ_API_KEY in the environment."
        )
