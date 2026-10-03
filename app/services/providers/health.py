from __future__ import annotations

import time
from typing import Any

from app.core.config import get_settings

_CACHE_TTL_SECONDS = 60.0
_PROVIDER_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}


def classify_provider_error(message: str) -> str:
    lowered = (message or "").lower()
    if any(token in lowered for token in ("quota", "resource_exhausted")):
        return "QUOTA_EXCEEDED"
    if any(token in lowered for token in ("rate limit", "429", "too many requests")):
        return "RATE_LIMITED"
    if any(token in lowered for token in ("503", "service unavailable", "currently experiencing high demand")):
        return "TEMPORARILY_UNAVAILABLE"
    if any(token in lowered for token in ("json", "malformed", "parse", "schema", "validation")):
        return "MALFORMED_RESPONSE"
    if any(token in lowered for token in ("permission", "permission_denied", "forbidden", "403")):
        return "PERMISSION_DENIED"
    if any(token in lowered for token in ("unauthenticated", "authentication", "401", "invalid api key")):
        return "AUTHENTICATION_FAILED"
    if any(token in lowered for token in ("missing", "not configured", "no api key")):
        return "MISSING_CREDENTIAL"
    if "model" in lowered and any(token in lowered for token in ("not found", "unsupported", "invalid")):
        return "INVALID_MODEL"
    if any(token in lowered for token in ("timeout", "timed out")):
        return "TIMEOUT"
    if any(token in lowered for token in ("network", "connection", "dns", "econn")):
        return "NETWORK_ERROR"
    if "import" in lowered or "module" in lowered:
        return "SDK_ERROR"
    return "PROVIDER_ERROR"


def classify_provider_exception(error: BaseException) -> str:
    chain: list[BaseException] = []
    current: BaseException | None = error
    while current is not None:
        chain.append(current)
        current = current.__cause__ or current.__context__
    for item in reversed(chain):
        error_code = classify_provider_error(str(item))
        if error_code != "PROVIDER_ERROR":
            return error_code
    return "PROVIDER_ERROR"


def safe_provider_error_message(error_code: str, provider: str) -> str:
    messages = {
        "MISSING_CREDENTIAL": f"{provider} API key is not configured.",
        "AUTHENTICATION_FAILED": "Authentication failed. Check the configured API key.",
        "INVALID_MODEL": "The configured model is invalid or unavailable.",
        "PERMISSION_DENIED": "The configured key does not have permission to use this model.",
        "QUOTA_EXCEEDED": "The provider quota has been exceeded.",
        "RATE_LIMITED": "The provider is rate limited. Try again later.",
        "TEMPORARILY_UNAVAILABLE": f"{provider} is temporarily unavailable due to high demand. Please retry shortly.",
        "TIMEOUT": "The provider did not respond before the request timed out.",
        "NETWORK_ERROR": "The provider could not be reached because of a network error.",
        "SDK_ERROR": "The provider client SDK could not be loaded.",
        "MALFORMED_RESPONSE": "The provider returned an invalid response.",
    }
    return messages.get(error_code, "Provider check failed. Check the provider configuration and try again.")


def _configured_status(configured: bool, model_name: str, provider: str) -> dict[str, Any]:
    if not configured:
        return {
            "configured": False,
            "working": False,
            "model": model_name,
            "status": "missing_credential",
            "error_code": "MISSING_CREDENTIAL",
            "message": safe_provider_error_message("MISSING_CREDENTIAL", provider),
        }
    return {
        "configured": True,
        "working": False,
        "model": model_name,
        "status": "not_verified",
        "error_code": None,
        "message": "Configured but not verified.",
    }


def _gemini_probe(api_key: str | None, model_name: str) -> dict[str, Any]:
    if not api_key:
        return _configured_status(False, model_name, "Gemini")
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model_name,
            contents="Reply with OK.",
            config=types.GenerateContentConfig(response_mime_type="text/plain"),
        )
        text = (getattr(response, "text", None) or "").strip()
        if not text:
            raise ValueError("Gemini returned an empty test response.")
        return {
            "configured": True,
            "working": True,
            "model": model_name,
            "status": "healthy",
            "error_code": None,
            "message": "Working",
        }
    except Exception as exc:  # pragma: no cover - defensive provider-specific classification
        error_code = classify_provider_error(str(exc))
        return {
            "configured": True,
            "working": False,
            "model": model_name,
            "status": error_code.lower(),
            "error_code": error_code,
            "message": safe_provider_error_message(error_code, "Gemini"),
        }


def _groq_probe(api_key: str | None, model_name: str) -> dict[str, Any]:
    if not api_key:
        return _configured_status(False, model_name, "Groq")
    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            timeout=45.0,
            max_retries=0,
        )
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Reply with OK."}],
            temperature=0.0,
            max_tokens=5,
        )
        _ = response.choices[0].message.content
        return {
            "configured": True,
            "working": True,
            "model": model_name,
            "status": "healthy",
            "error_code": None,
            "message": "Working",
        }
    except Exception as exc:  # pragma: no cover - defensive provider-specific classification
        error_code = classify_provider_error(str(exc))
        return {
            "configured": True,
            "working": False,
            "model": model_name,
            "status": error_code.lower(),
            "error_code": error_code,
            "message": safe_provider_error_message(error_code, "Groq"),
        }


def provider_check(role: str, force_refresh: bool = False) -> dict[str, Any]:
    settings = get_settings()
    if settings.allow_mock_llms:
        if role == "optimizer":
            result = _configured_status(bool(settings.gemini_optimizer_api_key), settings.optimizer_model, "Gemini")
        elif role == "performer":
            result = _configured_status(bool(settings.gemini_performer_api_key), settings.performer_model, "Gemini")
        elif role == "judge":
            result = _configured_status(bool(settings.groq_api_key), settings.judge_model, "Groq")
        else:
            raise ValueError(f"Unsupported provider role: {role}")
        result.update(status="mock", error_code=None, message="Mock mode is enabled; provider connectivity has not been tested.")
        return result

    cache_key = f"{role}:{settings.optimizer_model}:{settings.performer_model}:{settings.judge_model}:{settings.allow_mock_llms}"
    now = time.monotonic()
    if not force_refresh and cache_key in _PROVIDER_CACHE:
        cached_at, cached = _PROVIDER_CACHE[cache_key]
        if now - cached_at < _CACHE_TTL_SECONDS:
            return cached

    if role == "optimizer":
        payload = _gemini_probe(settings.gemini_optimizer_api_key, settings.optimizer_model)
    elif role == "performer":
        payload = _gemini_probe(settings.gemini_performer_api_key, settings.performer_model)
    elif role == "judge":
        payload = _groq_probe(settings.groq_api_key, settings.judge_model)
    else:
        raise ValueError(f"Unsupported provider role: {role}")

    _PROVIDER_CACHE[cache_key] = (now, payload)
    return payload


def provider_status(force_refresh: bool = False) -> dict[str, Any]:
    return {
        "optimizer": provider_check("optimizer", force_refresh=force_refresh),
        "performer": provider_check("performer", force_refresh=force_refresh),
        "judge": provider_check("judge", force_refresh=force_refresh),
    }
