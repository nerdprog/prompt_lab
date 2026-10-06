from __future__ import annotations

import math
import math
import re
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from app.core.config import get_settings
from app.schemas.task import OptimizationStartRequest, TaskIntent
from app.services.insights.insight_store import generate_insights
from app.services.intent.task_understanding import understand_task, understand_task_from_llm
from app.services.llm.judge import JudgeClient
from app.services.llm.performer import PerformerClient
from app.services.optimizer.candidate_generator import CandidateGenerator
from app.services.rubric.rubric_generator import build_rubric
from app.services.selector.ucb import compute_ucb, select_active_candidates, update_candidate_stats
from app.services.pdf.report_generator import ReportGenerator
from app.services.providers.health import (
    classify_provider_exception,
    classify_provider_error,
    safe_provider_error_message,
)
from app.state.session_state import (
    append_candidate,
    append_evaluation,
    append_insight,
    append_iteration,
    create_session,
    get_session,
    update_session,
)
from app.utils.validation import validate_config, validate_prompt


class BudgetExhausted(RuntimeError):
    pass


STAGE_A_MAX_CHILD_ATTEMPTS = 5
FINAL_EVAL_MAX_ATTEMPTS = 5


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class OptimizationService:
    STAGE_A_MAX_CHILD_ATTEMPTS = STAGE_A_MAX_CHILD_ATTEMPTS
    FINAL_EVAL_MAX_ATTEMPTS = FINAL_EVAL_MAX_ATTEMPTS

    def __init__(self) -> None:
        self.settings = get_settings()
        self.generator = CandidateGenerator()
        self.performer = PerformerClient()
        self.judge = JudgeClient()

    def validate_start_request(self, request: OptimizationStartRequest) -> tuple[bool, str | None]:
        ok, error = validate_prompt(request.prompt)
        if not ok:
            return False, error
        if not self.settings.allow_mock_llms and (
            not self.settings.has_optimizer_config or not self.settings.has_performer_config or not self.settings.has_groq_config
        ):
            return False, "Missing required LLM API keys. Configure GEMINI_OPTIMIZER_API_KEY, GEMINI_PERFORMER_API_KEY, and GROQ_API_KEY in the environment before starting optimization."
        errors = validate_config(request.model_dump())
        return (False, errors[0]) if errors else (True, None)

    def start(self, request: OptimizationStartRequest) -> dict[str, Any]:
        valid, error = self.validate_start_request(request)
        if not valid:
            raise ValueError(error or "Invalid optimization request.")
        config = {
            "active_candidates": request.active_candidates or self.settings.default_active_candidates,
            "edited_candidates": request.edited_candidates or self.settings.default_edited_candidates,
            "max_iterations": request.max_iterations or self.settings.default_max_iterations,
            "final_top_n": request.final_top_n or self.settings.default_final_top_n,
            "ucb_c": request.ucb_c or self.settings.default_ucb_c,
            "stagnation_limit": request.stagnation_limit or self.settings.default_stagnation_limit,
            "min_improvement": request.min_improvement if request.min_improvement is not None else self.settings.default_min_improvement,
            "max_llm_calls": request.max_llm_calls or self.settings.default_max_llm_calls,
        }
        session = create_session(request.prompt, request.optional_context, request.preferences, config)
        session["model_configuration"] = {
            "optimizer": self.settings.optimizer_model,
            "performer": self.settings.performer_model,
            "judge": self.settings.judge_model,
            "provider_mode": "mock" if self.settings.allow_mock_llms else "live",
        }
        try:
            self._set_stage(session["session_id"], "understanding_task", "running")
            if not self.settings.allow_mock_llms and self.settings.has_optimizer_config:
                task_intent = self._retry_provider_call(
                    session,
                    "Gemini Optimizer",
                    lambda: understand_task_from_llm(
                        request.prompt, request.optional_context, on_llm_call=lambda: self._charge(session)
                    ),
                )
            else:
                task_intent = understand_task(request.prompt, request.optional_context)
                task_intent.source = "mock"
            session["task_spec"] = task_intent.model_dump()
            session["status"] = "awaiting_confirmation"
            self._set_stage(session["session_id"], "understanding_task", "completed")
            update_session(
                session["session_id"],
                current_stage="awaiting_confirmation",
                stage="awaiting_confirmation",
                stage_status="pending",
            )
            session["samples"] = [
                {"sample_id": "sample-1", "origin": "original_prompt", "content": request.prompt},
                {"sample_id": "sample-2", "origin": "generated_variant", "content": request.prompt},
                {"sample_id": "sample-3", "origin": "generated_variant", "content": request.prompt},
            ]
            session["started_at"] = _now()
            return session
        except Exception:
            current_stage = session.get("current_stage")
            if current_stage in session.get("progress_stages", {}):
                self._set_stage(session["session_id"], current_stage, "failed")
            update_session(session["session_id"], status="failed", stop_reason="critical_failure")
            raise

    def confirm_and_prepare(self, session_id: str, task_spec_data: dict[str, Any]) -> dict[str, Any]:
        session = get_session(session_id)
        if session is None:
            raise KeyError(f"Session {session_id} was not found.")
        if session.get("status") != "awaiting_confirmation":
            raise ValueError("Task understanding can only be confirmed once before optimization starts.")
        source = (session.get("task_spec") or {}).get("source", "mock")
        task_intent = TaskIntent.model_validate({**task_spec_data, "source": source})
        session["task_spec"] = task_intent.model_dump()
        self._set_stage(session_id, "building_rubric", "running")
        try:
            rubric = self._retry_provider_call(
                session,
                "Gemini Optimizer",
                lambda: build_rubric(
                    task_intent, session["original_prompt"], on_llm_call=lambda: self._charge(session)
                ),
            )
        except Exception:
            self._set_stage(session_id, "building_rubric", "failed")
            raise
        session["rubric"] = rubric.model_dump()
        self._set_stage(session_id, "building_rubric", "completed")
        samples = session["samples"]
        samples[1]["content"] = (
            f"Task: {session['original_prompt']}\nContext variation: preserve the same objective and audience."
        )
        samples[2]["content"] = (
            f"Task: {session['original_prompt']}\nResponse variation: test the same task with a concise-answer expectation."
        )
        self._set_stage(session_id, "generating_candidates", "running")
        try:
            initial_candidates = self._retry_provider_call(
                session,
                "Gemini Optimizer",
                lambda: self.generator.generate_initial(
                    task_intent.model_dump(),
                    session["original_prompt"],
                    count=session["configuration"]["active_candidates"],
                    rubric=rubric.model_dump(),
                    on_llm_call=lambda: self._charge(session),
                ),
            )
        except Exception:
            self._set_stage(session_id, "generating_candidates", "failed")
            raise
        for candidate in initial_candidates:
            append_candidate(session_id, candidate)
        session["active_candidate_ids"] = [candidate["candidate_id"] for candidate in initial_candidates]
        self._set_stage(session_id, "generating_candidates", "completed")
        update_session(
            session_id,
            status="ready",
            task_spec=session["task_spec"],
            rubric=session["rubric"],
            active_candidate_count=len(initial_candidates),
            total_candidate_count=len(session["candidates"]),
            current_stage="ucb_selection",
            stage="ucb_selection",
            stage_status="pending",
        )
        return session

    def _charge(
        self, session: dict[str, Any], count: int = 1, reserve_calls: int = 0
    ) -> None:
        if self.settings.allow_mock_llms:
            return
        used = int(session.get("llm_call_count", 0))
        budget = int(session["configuration"]["max_llm_calls"])
        if used + count + reserve_calls > budget:
            raise BudgetExhausted("Optimization call budget exhausted.")
        session["llm_call_count"] = used + count
        session["model_calls_used"] = session["llm_call_count"]

    def _set_stage(self, session_id: str, stage: str, status: str) -> None:
        session = get_session(session_id)
        if session is None:
            raise KeyError(f"Session {session_id} was not found.")
        stages = dict(session.get("progress_stages", {}))
        stages[stage] = status
        update_session(
            session_id,
            current_stage=stage,
            stage=stage,
            stage_status=status,
            progress_stages=stages,
        )

    @staticmethod
    def _rate_limit_delay(error: Exception | None) -> int:
        if error is not None:
            response = getattr(error, "response", None)
            headers = getattr(response, "headers", None)
            if headers:
                try:
                    return max(1, min(120, int(float(headers.get("retry-after", 20)))))
                except (TypeError, ValueError):
                    pass
            match = re.search(r"(?:retry|try again) in\s+([0-9]+(?:\.[0-9]+)?)\s*s", str(error), re.IGNORECASE)
            if match:
                return max(1, min(120, math.ceil(float(match.group(1)))))
        return 20

    @staticmethod
    def _result_is_rate_limited(result: Any) -> bool:
        return isinstance(result, dict) and result.get("error_code") == "RATE_LIMITED"

    @staticmethod
    def _exception_is_rate_limited(error: Exception) -> bool:
        current: BaseException | None = error
        while current is not None:
            if classify_provider_error(str(current)) == "RATE_LIMITED":
                return True
            current = current.__cause__ or current.__context__
        return False

    def _retry_provider_call(
        self,
        session: dict[str, Any],
        provider: str,
        operation: Callable[[], Any],
        charge_retry: bool = False,
    ) -> Any:
        max_attempts = 3
        last_error: Exception | None = None
        result: Any = None
        for attempt in range(max_attempts):
            try:
                result = operation()
                last_error = None
            except Exception as exc:
                if not self._exception_is_rate_limited(exc):
                    raise
                last_error = exc
                result = None

            if last_error is None and not self._result_is_rate_limited(result):
                if attempt:
                    update_session(
                        session["session_id"],
                        rate_limit_status="cleared",
                        rate_limit_message="Rate limit cleared. Continuing optimization.",
                        rate_limit_cleared_at=_now(),
                    )
                return result

            if attempt == max_attempts - 1:
                break

            delay = self._rate_limit_delay(last_error)
            retry_at = datetime.now(timezone.utc) + timedelta(seconds=delay)
            update_session(
                session["session_id"],
                status="paused_rate_limit",
                rate_limit_status="waiting",
                rate_limit_provider=provider,
                rate_limit_retry_after=delay,
                rate_limit_retry_at=retry_at.isoformat(),
                rate_limit_message=f"{provider} is temporarily rate limited.",
                retry_count=int(session.get("retry_count", 0)) + 1,
            )
            while datetime.now(timezone.utc) < retry_at:
                if session.get("status") == "cancelled":
                    return None
                time.sleep(min(1.0, (retry_at - datetime.now(timezone.utc)).total_seconds()))

            if session.get("status") == "cancelled":
                return None
            update_session(
                session["session_id"],
                status="running",
                rate_limit_status="retrying",
                rate_limit_retry_after=None,
                rate_limit_retry_at=None,
                rate_limit_message="Retrying the provider request.",
            )
            if charge_retry:
                self._charge(session)

        if last_error is not None or self._result_is_rate_limited(result):
            current_stage = session.get("current_stage")
            if current_stage in session.get("progress_stages", {}):
                self._set_stage(session["session_id"], current_stage, "failed")
            update_session(
                session["session_id"],
                status="failed",
                stop_reason="rate_limit_exhausted",
                rate_limit_status="exhausted",
                rate_limit_message=f"{provider} is still rate limited after {max_attempts - 1} retries.",
                error_code="RATE_LIMITED",
                user_message=f"{provider} remains rate limited. Optimization could not continue.",
            )
            if last_error is not None:
                raise last_error
            raise RuntimeError(f"{provider} remains rate limited after {max_attempts - 1} retries.")
        return result

    def _record_stage_a_child_failure(
        self,
        session: dict[str, Any],
        parent: dict[str, Any],
        attempt_count: int,
        reason: str,
    ) -> None:
        failure = {
            "candidate_id": parent.get("candidate_id"),
            "parent_id": parent.get("parent_id"),
            "attempts": attempt_count,
            "reason": reason,
            "created_at": _now(),
        }
        session.setdefault("errors", []).append(failure)
        session.setdefault("stage_a_failures", []).append(failure)

    def _generate_stage_a_child(
        self,
        session: dict[str, Any],
        parent: dict[str, Any],
        feedback: dict[str, Any],
        task_spec: dict[str, Any],
        *,
        rubric: dict[str, Any] | None = None,
        insights: list[dict[str, Any]] | None = None,
        original_prompt: str = "",
        iteration: int = 0,
        existing_prompts: list[str] | None = None,
        reserve_calls: int = 0,
    ) -> list[dict[str, Any]]:
        for attempt in range(1, STAGE_A_MAX_CHILD_ATTEMPTS + 1):
            try:
                return self._retry_provider_call(
                    session,
                    "Gemini Optimizer",
                    lambda: self.generator.generate_edited(
                        parent,
                        feedback,
                        task_spec,
                        count=1,
                        rubric=rubric,
                        insights=insights,
                        original_prompt=original_prompt,
                        iteration=iteration,
                        existing_prompts=existing_prompts,
                        on_llm_call=lambda: self._charge(
                            session,
                            reserve_calls=reserve_calls,
                        ),
                    ),
                )
            except BudgetExhausted:
                session["stop_reason"] = "budget_exhausted"
                return []
            except Exception as exc:
                if attempt >= STAGE_A_MAX_CHILD_ATTEMPTS:
                    reason = "Stage A child-generation failed after 5 attempts."
                    self._record_stage_a_child_failure(session, parent, attempt, reason)
                    return []
                if isinstance(exc, RuntimeError) and "budget" in str(exc).lower():
                    session["stop_reason"] = "budget_exhausted"
                    return []
        return []

    def _update_best_metrics(self, session: dict[str, Any]) -> bool:
        scored = [
            candidate for candidate in session.get("candidates", [])
            if int(candidate.get("pull_count", 0)) > 0
        ]
        if not scored:
            return False
        best = max(scored, key=lambda candidate: float(candidate.get("mean_reward", 0.0)))
        score = float(best.get("mean_reward", 0.0))
        if score < float(session.get("current_best_quality") or 0.0):
            return False
        changed = best["candidate_id"] != session.get("current_best_candidate_id")
        session["current_best_quality"] = score
        session["current_best_candidate_id"] = best["candidate_id"]
        return changed

    def _update_best_prompt_token_count(
        self, session: dict[str, Any], reserve_calls: int
    ) -> None:
        if self.settings.allow_mock_llms:
            session["current_best_prompt_token_count"] = None
            session["token_count_error"] = "Exact token counting is unavailable in mock mode."
            return
        candidate_id = session.get("current_best_candidate_id")
        candidate = next(
            (
                item for item in session.get("candidates", [])
                if item.get("candidate_id") == candidate_id
            ),
            None,
        )
        if candidate is None:
            return
        try:
            self._charge(session, reserve_calls=reserve_calls)
            session["current_best_prompt_token_count"] = self.performer.count_tokens(
                candidate["prompt_text"], self.settings.performer_model
            )
            session["token_count_error"] = None
        except BudgetExhausted:
            session["current_best_prompt_token_count"] = None
            session["token_count_error"] = "Token count unavailable to preserve calls for remaining optimization stages."
        except Exception as exc:
            session["current_best_prompt_token_count"] = None
            session["token_count_error"] = f"Token count unavailable ({type(exc).__name__})."

    @staticmethod
    def _eligible_candidates(session: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            candidate for candidate in session.get("candidates", [])
            if candidate.get("status") not in {"invalid", "failed"}
        ]

    def run_round(self, session_id: str, round_number: int) -> dict[str, Any]:
        session = get_session(session_id)
        if not session:
            raise KeyError(f"Session {session_id} was not found.")
        if session.get("status") == "cancelled":
            return {"status": "cancelled"}
        update_session(session_id, current_iteration=round_number)
        self._set_stage(session_id, "ucb_selection", "running")
        candidates = self._eligible_candidates(session)
        if not candidates:
            self._set_stage(session_id, "ucb_selection", "completed")
            return {"status": "insufficient_valid_candidates", "evaluated": [], "new_candidates": []}

        sample_index = (round_number - 1) % len(session["samples"])
        sample = session["samples"][sample_index]
        k = min(len(candidates), int(session["configuration"]["active_candidates"]))
        active_ids = select_active_candidates(candidates, k=k, c=float(session["configuration"]["ucb_c"]))
        session["active_candidate_ids"] = active_ids
        update_session(
            session_id,
            active_candidate_count=len(active_ids),
            total_candidate_count=len(session.get("candidates", [])),
        )
        total_pulls = max(1, sum(int(item.get("pull_count", 0)) for item in candidates))
        decisions = []
        for candidate in candidates:
            pulls = int(candidate.get("pull_count", 0))
            mean = float(candidate.get("mean_reward", 0.0))
            value = compute_ucb(mean, pulls, total_pulls, float(session["configuration"]["ucb_c"]))
            decisions.append({
                "candidate_id": candidate["candidate_id"],
                "pull_count": pulls,
                "mean_reward": mean,
                "exploration_term": value - mean if math.isfinite(value) else None,
                "ucb": value if math.isfinite(value) else None,
                "selected": candidate["candidate_id"] in active_ids,
            })
            candidate["status"] = "active" if candidate["candidate_id"] in active_ids else "dropped"
        session.setdefault("ucb_history", []).append({
            "iteration": round_number,
            "total_pulls_before_round": total_pulls,
            "selected_candidate_ids": list(active_ids),
            "decisions": decisions,
            "created_at": _now(),
        })
        self._set_stage(session_id, "ucb_selection", "completed")

        evaluated_batch: list[dict[str, Any]] = []
        successful_count = 0
        edited_count = min(
            len(active_ids), int(session["configuration"]["edited_candidates"])
        )
        remaining_rounds = max(
            0, int(session["configuration"]["max_iterations"]) - round_number
        )
        future_round_reserve = remaining_rounds * (
            2 + int(session["configuration"]["edited_candidates"]) + 1
        )
        final_evaluation_reserve = len(session["samples"]) * 2
        token_count_reserve = 1
        self._set_stage(session_id, "performer_execution", "running")
        performer_results = self._retry_provider_call(
            session,
            "Gemini Performer",
            lambda: self.performer.execute_batch(
                [
                    {
                        "candidate_id": candidate_id,
                        "prompt_text": next(
                            item["prompt_text"] for item in candidates
                            if item["candidate_id"] == candidate_id
                        ),
                        "sample": sample["content"],
                    }
                    for candidate_id in active_ids
                ],
                self.settings.performer_model,
                on_llm_call=lambda: self._charge(
                    session,
                    reserve_calls=(
                        1
                        + edited_count
                        + future_round_reserve
                        + token_count_reserve
                        + final_evaluation_reserve
                    ),
                ),
            ),
        )
        if session.get("status") == "cancelled" or performer_results is None:
            return {
                "iteration": round_number,
                "evaluated": [],
                "new_candidates": [],
                "stop_reason": "cancelled",
            }
        update_session(
            session_id,
            completed_performer_calls=int(session.get("completed_performer_calls", 0))
            + len(performer_results),
        )
        self._set_stage(session_id, "performer_execution", "completed")
        self._set_stage(session_id, "judge_evaluation", "running")
        judge_results = self._retry_provider_call(
            session,
            "Groq Judge",
            lambda: self.judge.evaluate_batch(
                session["task_spec"],
                session["rubric"]["criteria"],
                [
                    {
                        "candidate_id": candidate_id,
                        "candidate_prompt": next(
                            item["prompt_text"] for item in candidates
                            if item["candidate_id"] == candidate_id
                        ),
                        "sample": sample["content"],
                        "response_text": performer_results[candidate_id]["response_text"],
                    }
                    for candidate_id in active_ids
                ],
                on_llm_call=lambda: self._charge(
                    session,
                    reserve_calls=(
                        edited_count
                        + future_round_reserve
                        + token_count_reserve
                        + final_evaluation_reserve
                    ),
                ),
            ),
        )
        if session.get("status") == "cancelled" or judge_results is None:
            return {
                "iteration": round_number,
                "evaluated": [],
                "new_candidates": [],
                "stop_reason": "cancelled",
            }
        update_session(
            session_id,
            completed_judge_calls=int(session.get("completed_judge_calls", 0))
            + len(judge_results),
        )
        self._set_stage(session_id, "judge_evaluation", "completed")
        for candidate_id in active_ids:
            candidate = next(
                item for item in candidates if item["candidate_id"] == candidate_id
            )
            performer_result = performer_results[candidate_id]
            judge_result = judge_results[candidate_id]
            reward = float(judge_result["authoritative_score"])
            if not math.isfinite(reward) or not 0.0 <= reward <= 1.0:
                raise ValueError("Judge returned an invalid authoritative reward.")
            update_candidate_stats(candidate, reward)
            successful_count += 1
            evaluation = {
                "evaluation_id": f"eval-{uuid.uuid4().hex[:10]}",
                "candidate_id": candidate_id,
                "session_id": session_id,
                "iteration": round_number,
                "sample_id": sample["sample_id"],
                "response": performer_result["response_text"],
                "criterion_scores": judge_result["criterion_scores"],
                "judge_overall_score": judge_result["overall_score"],
                "authoritative_score": reward,
                "normalized_reward": reward,
                "strengths": judge_result["strengths"],
                "weaknesses": judge_result["weaknesses"],
                "evidence": judge_result["evidence"],
                "root_cause": judge_result["root_cause"],
                "improvement_suggestion": judge_result["improvement_suggestion"],
                "confidence": judge_result["confidence"],
                "intent_fidelity": judge_result["intent_fidelity"],
                "unsupported_assumptions": judge_result["unsupported_assumptions"],
                "performer_model": performer_result["model"],
                "judge_model": judge_result["model"],
                "performer_source": performer_result["source"],
                "judge_source": judge_result["source"],
                "evaluation_source": "mock"
                if "mock" in {performer_result["source"], judge_result["source"]}
                else "live",
                "latency": performer_result["latency_ms"],
                "status": "success"
                if "mock" not in {performer_result["source"], judge_result["source"]}
                else "mock",
                "created_at": _now(),
            }
            append_evaluation(session_id, evaluation)
            evaluated_batch.append({
                "candidate_id": candidate_id,
                "score": reward,
                "feedback": evaluation,
            })
        best_candidate_changed = self._update_best_metrics(session)

        if session.get("status") == "failed":
            session["stop_reason"] = session.get("stop_reason") or "provider_failure"
            update_session(session_id, current_iteration=round_number)
            return {
                "iteration": round_number,
                "evaluated": evaluated_batch,
                "new_candidates": [],
                "best_score": max((item["score"] for item in evaluated_batch), default=None),
                "stop_reason": session["stop_reason"],
            }
        self._set_stage(session_id, "performer_execution", "completed")
        if session.get("progress_stages", {}).get("judge_evaluation") == "running":
            self._set_stage(session_id, "judge_evaluation", "completed")
        if not successful_count and session.get("stop_reason") != "budget_exhausted":
            session["stop_reason"] = "critical_failure"
        top_candidates = sorted(evaluated_batch, key=lambda item: item["score"], reverse=True)[
            : int(session["configuration"]["edited_candidates"])
        ]
        new_candidates: list[dict[str, Any]] = []
        previous_insights = list(session.get("insights", []))
        self._set_stage(session_id, "stage_a_optimization", "running")
        for edit_index, parent_item in enumerate(top_candidates):
            if session.get("status") in {"cancelled", "failed"}:
                break
            parent = next(item for item in candidates if item["candidate_id"] == parent_item["candidate_id"])
            evaluation = parent_item["feedback"]
            remaining_edits = len(top_candidates) - edit_index - 1
            feedback = {
                "evaluation_id": evaluation["evaluation_id"],
                "candidate_id": parent["candidate_id"],
                "score": parent_item["score"],
                "weaknesses": evaluation["weaknesses"],
                "root_cause": evaluation["root_cause"],
                "improvement_suggestion": evaluation["improvement_suggestion"],
                "criterion_scores": evaluation["criterion_scores"],
                "evidence": evaluation["evidence"],
            }
            children = self._generate_stage_a_child(
                session,
                parent,
                feedback,
                session["task_spec"],
                rubric=session["rubric"],
                insights=previous_insights,
                original_prompt=session["original_prompt"],
                iteration=round_number,
                existing_prompts=[item["prompt_text"] for item in session["candidates"]],
                reserve_calls=(
                    remaining_edits
                    + future_round_reserve
                    + token_count_reserve
                    + final_evaluation_reserve
                ),
            )
            if not children:
                if session.get("stop_reason") == "budget_exhausted":
                    break
                continue
            for child in children:
                append_candidate(session_id, child)
                new_candidates.append(child)
        if session.get("status") == "failed":
            update_session(session_id, current_iteration=round_number)
            return {
                "iteration": round_number,
                "evaluated": evaluated_batch,
                "new_candidates": new_candidates,
                "best_score": max((item["score"] for item in evaluated_batch), default=None),
                "stop_reason": session.get("stop_reason"),
            }
        self._set_stage(session_id, "stage_a_optimization", "completed")
        if best_candidate_changed:
            self._update_best_prompt_token_count(
                session, future_round_reserve + final_evaluation_reserve
            )

        self._set_stage(session_id, "insight_update", "running")
        insight_inputs = [
            {**item["feedback"], "candidate_id": item["candidate_id"]}
            for item in evaluated_batch
        ]
        created_insights = generate_insights(session["task_spec"], insight_inputs, round_number)
        self._merge_insights(session, created_insights, session_id)
        self._set_stage(session_id, "insight_update", "completed")
        self._set_stage(session_id, "next_iteration", "completed")
        iteration_record = {
            "iteration": round_number,
            "sample_id": sample["sample_id"],
            "active_candidates": active_ids,
            "evaluated_candidates": [item["candidate_id"] for item in evaluated_batch],
            "parent_candidates": [item["candidate_id"] for item in top_candidates],
            "edited_candidates": [item["candidate_id"] for item in new_candidates],
            "ucb_decisions": decisions,
            "best_score": max((item["score"] for item in evaluated_batch), default=None),
            "created_at": _now(),
        }
        append_iteration(session_id, iteration_record)
        session["current_iteration"] = round_number
        session.setdefault("best_score_by_iteration", []).append(iteration_record["best_score"])
        if session.get("status") == "cancelled":
            session["stop_reason"] = "cancelled"
            update_session(session_id, current_iteration=round_number, stop_reason="cancelled")
        else:
            update_session(
                session_id,
                current_iteration=round_number,
                status="running",
                total_candidate_count=len(session.get("candidates", [])),
            )
        return {
            "iteration": round_number,
            "evaluated": evaluated_batch,
            "new_candidates": new_candidates,
            "best_score": iteration_record["best_score"],
            "stop_reason": session.get("stop_reason"),
        }

    def _record_failed_evaluation(
        self, session_id: str, candidate: dict[str, Any], sample: dict[str, Any],
        performer_result: dict[str, Any], evaluation_id: str, reason: str, error: str | None = None,
    ) -> None:
        append_evaluation(session_id, {
            "evaluation_id": evaluation_id,
            "candidate_id": candidate["candidate_id"],
            "sample_id": sample["sample_id"],
            "response": performer_result.get("response_text", ""),
            "status": "failed",
            "failure_reason": reason,
            "error": error or performer_result.get("error"),
            "error_code": performer_result.get("error_code"),
            "performer_source": performer_result.get("source"),
            "judge_source": None,
            "created_at": _now(),
        })

    def _merge_insights(self, session: dict[str, Any], new_insights: list[dict[str, Any]], session_id: str) -> None:
        existing_by_text = {item["text"].casefold(): item for item in session.get("insights", [])}
        for insight in new_insights:
            key = insight["text"].casefold()
            existing = existing_by_text.get(key)
            if existing:
                existing["evidence"] = list(dict.fromkeys(existing.get("evidence", []) + insight.get("evidence", [])))
                existing["provenance"].extend(insight.get("provenance", []))
                existing["confidence"] = min(1.0, max(existing["confidence"], insight["confidence"]))
                existing["updated_at"] = _now()
            else:
                append_insight(session_id, insight)
                existing_by_text[key] = insight

    def stopping_condition(self, session: dict[str, Any], round_result: dict[str, Any]) -> str | None:
        if session.get("status") == "cancelled":
            return "cancelled"
        if session.get("stop_reason") == "budget_exhausted":
            return "budget_exhausted"
        if session.get("stop_reason") == "critical_failure":
            return "critical_failure"
        if not self._eligible_candidates(session):
            return "insufficient_valid_candidates"
        if int(session["current_iteration"]) >= int(session["configuration"]["max_iterations"]):
            return "max_iterations"
        scores = session.get("best_score_by_iteration", [])
        if len(scores) >= int(session["configuration"]["stagnation_limit"]) + 1:
            recent = scores[-(int(session["configuration"]["stagnation_limit"]) + 1):]
            best = max((score for score in recent if score is not None), default=None)
            previous_best = max((score for score in scores[:-int(session["configuration"]["stagnation_limit"])] if score is not None), default=None)
            min_improvement = float(session["configuration"]["min_improvement"])
            if best is not None and previous_best is not None:
                relative_improvement = (best - previous_best) / max(abs(previous_best), 1e-9)
                if relative_improvement < min_improvement:
                    return "stagnation"
        return None

    def _write_report(self, session_id: str, session: dict[str, Any]) -> None:
        session["completed_at"] = _now()
        self._set_stage(session_id, "report_generation", "running")
        try:
            report_path = ReportGenerator(output_dir="reports").create_report(session)
            self._set_stage(session_id, "report_generation", "completed")
            updates: dict[str, Any] = {"report_path": report_path, "report_error": None}
            terminal_status = session.get("terminal_status_after_report")
            if terminal_status and session.get("status") != "cancelled":
                updates["status"] = terminal_status
            update_session(session_id, **updates)
        except Exception as exc:
            self._set_stage(session_id, "report_generation", "failed")
            updates = {"report_error": "PDF report generation failed."}
            terminal_status = session.get("terminal_status_after_report")
            if terminal_status and session.get("status") != "cancelled":
                updates["status"] = terminal_status
            update_session(session_id, **updates)
            session.setdefault("errors", []).append({
                "stage": "report_generation",
                "error": type(exc).__name__,
            })

    def run_optimization(self, session_id: str) -> None:
        session = get_session(session_id)
        if not session:
            return
        if session.get("status") == "cancelled":
            session["stop_reason"] = "cancelled"
            self._write_report(session_id, session)
            return
        update_session(
            session_id,
            status="running",
            current_stage="ucb_selection",
            stage="ucb_selection",
            stage_status="pending",
        )
        try:
            for iteration in range(1, int(session["configuration"]["max_iterations"]) + 1):
                if session.get("status") == "cancelled":
                    session["stop_reason"] = "cancelled"
                    break
                if int(session.get("current_iteration", 0)) >= int(session["configuration"]["max_iterations"]):
                    session["stop_reason"] = "max_iterations"
                    break
                result = self.run_round(session_id, iteration)
                if session.get("status") == "cancelled":
                    session["stop_reason"] = "cancelled"
                    break
                if session.get("status") == "failed":
                    break
                stop_reason = self.stopping_condition(session, result)
                if stop_reason:
                    session["stop_reason"] = stop_reason
                    break
            if not session.get("stop_reason"):
                session["stop_reason"] = "max_iterations"
            if session.get("status") not in {"cancelled", "failed"}:
                self._set_stage(session_id, "final_evaluation", "running")
                session["status"] = "final_evaluation"
                self.finalize(session_id)
            elif session.get("status") == "cancelled":
                update_session(session_id, status="cancelled", stop_reason="cancelled")
            else:
                update_session(session_id, status="failed")
        except BudgetExhausted:
            session["stop_reason"] = "budget_exhausted"
            session["status"] = "final_evaluation"
            self.finalize(session_id)
        except Exception as exc:
            current_stage = session.get("current_stage")
            if current_stage in session.get("progress_stages", {}):
                self._set_stage(session_id, current_stage, "failed")
            error_code = classify_provider_exception(exc)
            provider = {
                "performer_execution": "Gemini Performer",
                "judge_evaluation": "Groq Judge",
            }.get(current_stage, "Gemini Optimizer")
            if session.get("stop_reason") != "rate_limit_exhausted":
                session["stop_reason"] = "critical_failure"
                session["status"] = "failed"
                session["error_code"] = error_code
                session["user_message"] = safe_provider_error_message(error_code, provider)
            session.setdefault("errors", []).append({
                "stage": current_stage or "optimization",
                "error": type(exc).__name__,
                "error_code": error_code,
            })
            update_session(session_id, stop_reason="critical_failure", status="failed")
        finally:
            self._write_report(session_id, session)

    def finalize(self, session_id: str) -> dict[str, Any]:
        session = get_session(session_id)
        if not session:
            raise KeyError(f"Session {session_id} was not found.")
        candidates = [
            candidate for candidate in session.get("candidates", [])
            if candidate.get("status") not in {"invalid", "failed"}
            and any(
                evaluation.get("candidate_id") == candidate["candidate_id"]
                and evaluation.get("status") in {"success", "mock"}
                for evaluation in session.get("evaluations", [])
            )
        ]
        candidates.sort(key=lambda item: (-float(item.get("mean_reward", 0.0)), item["candidate_id"]))
        self._set_stage(session_id, "final_evaluation", "running")
        finalists = candidates[: int(session["configuration"]["final_top_n"])]
        sample_evaluations: list[dict[str, Any]] = []
        baseline_evaluations: list[dict[str, Any]] = []
        finalist_scores: dict[str, list[float]] = {
            candidate["candidate_id"]: [] for candidate in finalists
        }
        baseline_id = "original-baseline"
        final_targets = [
            {
                "candidate_id": candidate["candidate_id"],
                "prompt_text": candidate["prompt_text"],
            }
            for candidate in finalists
        ]
        final_targets.append({
            "candidate_id": baseline_id,
            "prompt_text": session["original_prompt"],
        })
        for sample_index, sample in enumerate(session["samples"]):
            if session.get("status") == "cancelled":
                break
            remaining_samples = len(session["samples"]) - sample_index - 1
            remaining_final_calls = remaining_samples * 2
            for attempt in range(1, FINAL_EVAL_MAX_ATTEMPTS + 1):
                try:
                    performer_results = self._retry_provider_call(
                        session,
                        "Gemini Performer",
                        lambda: self.performer.execute_batch(
                            [
                                {
                                    "candidate_id": item["candidate_id"],
                                    "prompt_text": item["prompt_text"],
                                    "sample": sample["content"],
                                }
                                for item in final_targets
                            ],
                            self.settings.performer_model,
                            on_llm_call=lambda: self._charge(
                                session, reserve_calls=1 + remaining_final_calls
                            ),
                        ),
                    )
                except BudgetExhausted:
                    session["stop_reason"] = "budget_exhausted"
                    break
                except Exception as exc:
                    if attempt >= FINAL_EVAL_MAX_ATTEMPTS:
                        session.setdefault("errors", []).append({
                            "stage": "final_evaluation",
                            "attempts": attempt,
                            "error": type(exc).__name__,
                            "error_code": classify_provider_exception(exc),
                        })
                        session["stop_reason"] = "critical_failure"
                        break
                    continue

                if session.get("status") == "cancelled" or performer_results is None:
                    break
                update_session(
                    session_id,
                    completed_performer_calls=int(session.get("completed_performer_calls", 0))
                    + len(performer_results),
                )
                try:
                    judge_results = self._retry_provider_call(
                        session,
                        "Groq Judge",
                        lambda: self.judge.evaluate_batch(
                            session["task_spec"],
                            session["rubric"]["criteria"],
                            [
                                {
                                    "candidate_id": item["candidate_id"],
                                    "candidate_prompt": item["prompt_text"],
                                    "sample": sample["content"],
                                    "response_text": performer_results[item["candidate_id"]]["response_text"],
                                }
                                for item in final_targets
                            ],
                            on_llm_call=lambda: self._charge(
                                session, reserve_calls=remaining_final_calls
                            ),
                            final_evaluation=True,
                        ),
                    )
                except BudgetExhausted:
                    session["stop_reason"] = "budget_exhausted"
                    break
                except Exception as exc:
                    if attempt >= FINAL_EVAL_MAX_ATTEMPTS:
                        session.setdefault("errors", []).append({
                            "stage": "final_evaluation",
                            "attempts": attempt,
                            "error": type(exc).__name__,
                            "error_code": classify_provider_exception(exc),
                        })
                        session["stop_reason"] = "critical_failure"
                        break
                    continue

                if session.get("status") == "cancelled" or judge_results is None:
                    break
                update_session(
                    session_id,
                    completed_judge_calls=int(session.get("completed_judge_calls", 0))
                    + len(judge_results),
                )
                for item in final_targets:
                    candidate_id = item["candidate_id"]
                    performer_result = performer_results[candidate_id]
                    judge_result = judge_results[candidate_id]
                    result = {
                        "sample_id": sample["sample_id"],
                        "response": performer_result["response_text"],
                        "authoritative_score": judge_result["authoritative_score"],
                        "criterion_scores": judge_result["criterion_scores"],
                        "feedback": {
                            "strengths": judge_result["strengths"],
                            "weaknesses": judge_result["weaknesses"],
                            "evidence": judge_result["evidence"],
                            "root_cause": judge_result["root_cause"],
                            "improvement_suggestion": judge_result["improvement_suggestion"],
                        },
                        "source": "mock"
                        if "mock" in {performer_result["source"], judge_result["source"]}
                        else "live",
                        "status": "mock"
                        if "mock" in {performer_result["source"], judge_result["source"]}
                        else "success",
                        "performer_model": performer_result["model"],
                        "judge_model": judge_result["model"],
                    }
                    if candidate_id == baseline_id:
                        baseline_evaluations.append(result)
                    else:
                        finalist_scores[candidate_id].append(
                            float(result["authoritative_score"])
                        )
                        sample_evaluations.append({"candidate_id": candidate_id, **result})
                break
        for candidate in finalists:
            scores = finalist_scores[candidate["candidate_id"]]
            candidate["final_score"] = (
                sum(scores) / len(scores)
                if len(scores) == len(session["samples"]) and scores
                and session.get("status") != "cancelled" else None
            )
        baseline_score = (
            sum(item["authoritative_score"] for item in baseline_evaluations) / len(baseline_evaluations)
            if len(baseline_evaluations) == len(session["samples"]) and baseline_evaluations
            and session.get("status") != "cancelled" else None
        )
        ranked = [
            {
                "candidate_id": item["candidate_id"],
                "prompt_text": item["prompt_text"],
                "score": item.get("final_score"),
                "parent_id": item.get("parent_id"),
                "generation": item.get("generation", 0),
                "generation_reason": item.get("generation_reason"),
                "preserved_requirements": item.get("preserved_requirements", []),
                "final_evaluations": [
                    evaluation for evaluation in sample_evaluations
                    if evaluation["candidate_id"] == item["candidate_id"]
                ],
            }
            for item in finalists if item.get("final_score") is not None
        ]
        ranked.sort(key=lambda item: (-item["score"], item["candidate_id"]))
        top_n = int(session["configuration"]["final_top_n"])
        termination_reason = session.get("stop_reason")
        final_data = {
            "finalists": ranked,
            "baseline": {
                "prompt_text": session["original_prompt"],
                "score": baseline_score,
                "evaluations": baseline_evaluations,
            },
            "final_ranking": ranked,
            "top3": ranked[:min(3, top_n)],
            "termination_reason": termination_reason,
            "stop_reason": None if termination_reason in {None, "max_iterations", "stagnation"} and ranked else termination_reason,
            "created_at": _now(),
        }
        terminal_status = (
            "cancelled" if session.get("status") == "cancelled"
            else "completed" if ranked else "failed"
        )
        self._set_stage(session_id, "final_evaluation", "completed")
        update_session(
            session_id,
            final_evaluation=final_data,
            status="cancelled" if terminal_status == "cancelled" else "report_generation",
            terminal_status_after_report=terminal_status,
            completed_at=_now(),
        )
        return final_data

    def _final_score(self, session: dict[str, Any], prompt: str, sample: dict[str, Any]) -> dict[str, Any] | None:
        try:
            self._charge(session)
            performer_result = self._retry_provider_call(
                session,
                "Gemini Performer",
                lambda: self.performer.execute(prompt, sample["content"], self.settings.performer_model),
                charge_retry=True,
            )
            if session.get("status") == "cancelled" or performer_result is None:
                return None
            update_session(
                session["session_id"],
                completed_performer_calls=int(session.get("completed_performer_calls", 0)) + 1,
            )
            if performer_result["status"] not in {"success", "mock"}:
                return None
            judge_result = self._retry_provider_call(
                session,
                "Groq Judge",
                lambda: self.judge.evaluate(
                    session["task_spec"], session["rubric"]["criteria"], prompt, sample["content"],
                    performer_result["response_text"],
                    on_llm_call=lambda: self._charge(session),
                ),
            )
            if session.get("status") == "cancelled" or judge_result is None:
                return None
            update_session(
                session["session_id"],
                completed_judge_calls=int(session.get("completed_judge_calls", 0)) + 1,
            )
            score = float(judge_result["authoritative_score"])
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                return None
            return {
                "sample_id": sample["sample_id"],
                "response": performer_result["response_text"],
                "authoritative_score": score,
                "criterion_scores": judge_result["criterion_scores"],
                "feedback": {
                    "strengths": judge_result["strengths"],
                    "weaknesses": judge_result["weaknesses"],
                    "evidence": judge_result["evidence"],
                    "root_cause": judge_result["root_cause"],
                    "improvement_suggestion": judge_result["improvement_suggestion"],
                },
                "source": "mock" if "mock" in {performer_result["source"], judge_result["source"]} else "live",
                "status": "success",
            }
        except BudgetExhausted:
            session["stop_reason"] = "budget_exhausted"
            session.setdefault("errors", []).append({
                "stage": "final_evaluation",
                "error": "BudgetExhausted",
            })
            return None
        except Exception as exc:
            session.setdefault("errors", []).append({
                "stage": "final_evaluation",
                "error": type(exc).__name__,
            })
            return None

    def get_status(self, session_id: str) -> dict[str, Any]:
        session = get_session(session_id)
        if not session:
            raise KeyError(f"Session {session_id} was not found.")
        return {
            "session_id": session_id,
            "status": session.get("status"),
            "current_iteration": session.get("current_iteration", 0),
            "max_iterations": session.get("max_iterations", self.settings.default_max_iterations),
            "stop_reason": session.get("stop_reason"),
            "current_stage": session.get("current_stage"),
            "stage_status": session.get("stage_status"),
            "progress_stages": session.get("progress_stages", {}),
            "active_candidate_count": session.get("active_candidate_count", 0),
            "best_score": max(
                (candidate.get("mean_reward", 0.0) for candidate in session.get("candidates", [])),
                default=0.0,
            ),
            "current_best_quality": session.get("current_best_quality"),
            "current_best_prompt_token_count": session.get("current_best_prompt_token_count"),
            "token_count_error": session.get("token_count_error"),
            "candidate_count": len(session.get("candidates", [])),
            "evaluations": len(session.get("evaluations", [])),
            "llm_call_count": session.get("llm_call_count", 0),
            "max_llm_calls": session.get("configuration", {}).get("max_llm_calls"),
            "completed_performer_calls": session.get("completed_performer_calls", 0),
            "completed_judge_calls": session.get("completed_judge_calls", 0),
            "rate_limit_status": session.get("rate_limit_status"),
            "rate_limit_provider": session.get("rate_limit_provider"),
            "rate_limit_retry_after": session.get("rate_limit_retry_after"),
            "rate_limit_retry_at": session.get("rate_limit_retry_at"),
            "rate_limit_message": session.get("rate_limit_message"),
            "error_code": session.get("error_code"),
            "user_message": session.get("user_message"),
            "report_ready": bool(session.get("report_path")),
            "report_error": session.get("report_error"),
        }
