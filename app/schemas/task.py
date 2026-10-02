from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class TaskIntent(BaseModel):
    task_category: str = "general"
    primary_intent: str = Field(..., min_length=1, max_length=4000)
    audience: str | None = Field(default=None, max_length=1000)
    desired_complexity: str | None = Field(default=None, max_length=1000)
    language: str | None = Field(default=None, max_length=1000)
    output_format: str | None = Field(default=None, max_length=1000)
    style: str | None = Field(default=None, max_length=1000)
    explicit_requirements: list[str] = Field(default_factory=list, max_length=100)
    inferred_requirements: list[str] = Field(default_factory=list, max_length=100)
    constraints: list[str] = Field(default_factory=list, max_length=100)
    forbidden_assumptions: list[str] = Field(default_factory=list, max_length=100)
    ambiguities: list[str] = Field(default_factory=list, max_length=100)
    confidence: float = Field(ge=0.0, le=1.0, default=0.8)
    source: Literal["gemini", "mock"] = "mock"


class RubricCriterion(BaseModel):
    criterion: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    weight: float = Field(..., gt=0.0, le=1.0)
    scoring_scale: str = Field(..., min_length=1)

    @property
    def name(self) -> str:
        return self.criterion

    @property
    def scoring_guidance(self) -> str:
        return self.scoring_scale

    @field_validator("weight")
    @classmethod
    def validate_weight(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("Weight must be positive.")
        return value


class TaskRubric(BaseModel):
    task_type: str
    criteria: list[RubricCriterion]

    @property
    def total_weight(self) -> float:
        return sum(item.weight for item in self.criteria)

    @field_validator("criteria")
    @classmethod
    def validate_weights(cls, value: list[RubricCriterion]) -> list[RubricCriterion]:
        if not value:
            raise ValueError("Rubric must have at least one criterion.")
        if abs(sum(item.weight for item in value) - 1.0) > 0.01:
            raise ValueError("Rubric weights must sum to 1.0.")
        return value


class CandidatePrompt(BaseModel):
    candidate_id: str
    parent_id: str | None = None
    generation: int = 0
    prompt_text: str = Field(..., min_length=1)
    generation_reason: str | None = None
    preserved_requirements: list[str] = Field(default_factory=list)
    new_assumptions: list[str] = Field(default_factory=list)
    removed_assumptions: list[str] = Field(default_factory=list)
    created_at: str
    status: Literal["new", "active", "retained", "dropped", "invalid", "failed"] = "new"
    pull_count: int = Field(default=0, ge=0)
    total_reward: float = Field(default=0.0, ge=0.0)
    mean_reward: float = Field(default=0.0, ge=0.0, le=1.0)
    source: Literal["gemini", "mock"] = "gemini"
    provenance: dict[str, Any] = Field(default_factory=dict)


class CandidateGenerationItem(BaseModel):
    prompt_text: str = Field(..., min_length=1)
    generation_reason: str = Field(..., min_length=1)
    preserved_requirements: list[str] = Field(default_factory=list)
    new_assumptions: list[str] = Field(default_factory=list)
    removed_assumptions: list[str] = Field(default_factory=list)


class CandidateGenerationOutput(BaseModel):
    candidates: list[CandidateGenerationItem] = Field(min_length=1)


class JudgeCriterionScore(BaseModel):
    criterion: str = Field(..., min_length=1)
    score: float = Field(..., ge=0.0, le=10.0)
    weight: float = Field(..., gt=0.0, le=1.0)
    reason: str = Field(..., min_length=1)


class JudgeOutput(BaseModel):
    overall_score: float = Field(..., ge=0.0, le=10.0)
    criterion_scores: list[JudgeCriterionScore] = Field(min_length=1)
    strengths: list[str]
    weaknesses: list[str]
    evidence: list[str]
    root_cause: list[str]
    improvement_suggestion: list[str]
    confidence: float = Field(..., ge=0.0, le=1.0)
    intent_fidelity: float = Field(..., ge=0.0, le=10.0)
    unsupported_assumptions: list[str]

    @model_validator(mode="after")
    def validate_feedback_fields(self) -> "JudgeOutput":
        for field_name in ("strengths", "weaknesses", "evidence", "root_cause", "improvement_suggestion", "unsupported_assumptions"):
            if any(not item.strip() for item in getattr(self, field_name)):
                raise ValueError(f"{field_name} entries cannot be blank")
        return self


class PerformerResult(BaseModel):
    response_text: str = ""
    model: str
    status: Literal["success", "failed", "mock"] = "success"
    source: Literal["gemini", "mock"] = "gemini"
    error: str | None = None
    latency_ms: int = Field(..., ge=0)
    token_usage: dict[str, Any] = Field(default_factory=dict)
    created_at: str


class CandidateEvaluation(BaseModel):
    candidate_id: str
    sample_id: str
    response_text: str
    model: str
    status: Literal["success", "failed", "timeout", "rate_limited"] = "success"
    error: str | None = None
    latency_ms: int | None = None
    token_usage: dict[str, Any] | None = None
    criterion_scores: list[dict[str, Any]] = Field(default_factory=list)
    overall_score: float | None = None
    normalized_reward: float | None = None
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    root_cause: list[str] = Field(default_factory=list)
    improvement_suggestion: list[str] = Field(default_factory=list)
    confidence: float | None = None
    intent_fidelity: float | None = None
    unsupported_assumptions: list[str] = Field(default_factory=list)
    created_at: str


class OptimizationStartRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=12000)
    optional_context: str | None = Field(default=None, max_length=12000)
    preferences: str | None = Field(default=None, max_length=2000)
    active_candidates: int | None = None
    edited_candidates: int | None = None
    max_iterations: int | None = None
    final_top_n: int | None = None
    ucb_c: float | None = None
    stagnation_limit: int | None = None
    min_improvement: float | None = None
    max_llm_calls: int | None = Field(default=None, ge=1, le=500)


class TaskUnderstandingRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=12000)
    optional_context: str | None = Field(default=None, max_length=12000)
    preferences: str | None = Field(default=None, max_length=2000)
    active_candidates: int | None = None
    edited_candidates: int | None = None
    max_iterations: int | None = None
    final_top_n: int | None = None
    ucb_c: float | None = None
    stagnation_limit: int | None = None
    min_improvement: float | None = None
    max_llm_calls: int | None = Field(default=None, ge=1, le=500)


class IntentEditRequest(BaseModel):
    task_spec: TaskIntent
    desired_complexity: str | None = None
    language: str | None = None
    output_format: str | None = None
    style: str | None = None
    explicit_requirements: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    ambiguities: list[str] = Field(default_factory=list)
    confidence: float | None = None


class ApiError(BaseModel):
    code: str
    message: str
    details: dict[str, Any] | None = None


class ApiResponse(BaseModel):
    success: bool
    data: dict[str, Any] | None = None
    error: ApiError | None = None
