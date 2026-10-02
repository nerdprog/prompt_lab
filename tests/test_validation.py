import os

os.environ.setdefault("ALLOW_MOCK_LLMS", "true")

from app.utils.validation import validate_config, validate_prompt


def test_empty_prompt_is_invalid() -> None:
    valid, error = validate_prompt("")
    assert valid is False
    assert error == "Please enter a prompt to optimize."


def test_short_prompt_is_invalid() -> None:
    valid, error = validate_prompt("abc")
    assert valid is False
    assert "at least" in error


def test_valid_prompt_is_allowed() -> None:
    valid, error = validate_prompt("Create a study plan for me.")
    assert valid is True
    assert error is None


def test_config_validation_flags_bad_ranges() -> None:
    errors = validate_config({"active_candidates": 2, "max_iterations": 11})
    assert any("between 3 and 10" in e for e in errors)
    assert any("between 1 and 10" in e for e in errors)
