from __future__ import annotations

import math
import uuid
from datetime import datetime, timezone
from typing import Any

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


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class OptimizationService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.generator = CandidateGenerator()
        self.performer = PerformerClient()
        self.judge = JudgeClient()

    def validate_start_request(self, request: OptimizationStartRequest) -> tuple[bool, str | None]:
        ok, error = validate_prompt(request.prompt)
        if not ok:
            return False, error
        if not self.settings.allow_mock_llms and (not self.settings.has_gemini_config or not self.settings.has_groq_config):
            return False, "Missing required LLM API keys. Configure GEMINI_API_KEY and GROQ_API_KEY in the environment before starting optimization."
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
            if not self.settings.allow_mock_llms and self.settings.has_gemini_config:
                task_intent = understand_task_from_llm(
                    request.prompt, request.optional_context, on_llm_call=lambda: self._charge(session)
                )
            else:
                task_intent = understand_task(request.prompt, request.optional_context)
                task_intent.source = "mock"
            session["task_spec"] = task_intent.model_dump()
            session["status"] = "awaiting_confirmation"
            session["samples"] = [
                {"sample_id": "sample-1", "origin": "original_prompt", "content": request.prompt},
                {"sample_id": "sample-2", "origin": "generated_variant", "content": request.prompt},
                {"sample_id": "sample-3", "origin": "generated_variant", "content": request.prompt},
            ]
            session["started_at"] = _now()
            return session
        except Exception:
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
        rubric = build_rubric(
            task_intent, session["original_prompt"], on_llm_call=lambda: self._charge(session)
        )
        session["rubric"] = rubric.model_dump()
        samples = session["samples"]
        samples[1]["content"] = (
            f"Task: {session['original_prompt']}\nContext variation: preserve the same objective and audience."
        )
        samples[2]["content"] = (
            f"Task: {session['original_prompt']}\nResponse variation: test the same task with a concise-answer expectation."
        )
        initial_candidates = self.generator.generate_initial(
            task_intent.model_dump(),
            session["original_prompt"],
            count=session["configuration"]["active_candidates"],
            rubric=rubric.model_dump(),
            on_llm_call=lambda: self._charge(session),
        )
        for candidate in initial_candidates:
            append_candidate(session_id, candidate)
        session["active_candidate_ids"] = [candidate["candidate_id"] for candidate in initial_candidates]
        update_session(session_id, status="ready", task_spec=session["task_spec"], rubric=session["rubric"])
        return session

    def _charge(self, session: dict[str, Any], count: int = 1) -> None:
        if self.settings.allow_mock_llms:
            return
        used = int(session.get("llm_call_count", 0))
        budget = int(session["configuration"]["max_llm_calls"])
        if used + count > budget:
            raise BudgetExhausted("Optimization call budget exhausted.")
        session["llm_call_count"] = used + count

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
        candidates = self._eligible_candidates(session)
        if not candidates:
            return {"status": "insufficient_valid_candidates", "evaluated": [], "new_candidates": []}

        sample_index = (round_number - 1) % len(session["samples"])
        sample = session["samples"][sample_index]
        k = min(len(candidates), int(session["configuration"]["active_candidates"]))
        active_ids = select_active_candidates(candidates, k=k, c=float(session["configuration"]["ucb_c"]))
        session["active_candidate_ids"] = active_ids
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

        evaluated_batch: list[dict[str, Any]] = []
        successful_count = 0
        for candidate_id in active_ids:
            if session.get("status") == "cancelled":
                break
            candidate = next(item for item in candidates if item["candidate_id"] == candidate_id)
            try:
                self._charge(session)
            except BudgetExhausted:
                session["stop_reason"] = "budget_exhausted"
                break
            performer_result = self.performer.execute(
                candidate["prompt_text"], sample["content"], self.settings.performer_model
            )
            evaluation_id = f"eval-{uuid.uuid4().hex[:10]}"
            if performer_result["status"] not in {"success", "mock"}:
                self._record_failed_evaluation(session_id, candidate, sample, performer_result, evaluation_id, "performer_failed")
                continue
            try:
                judge_result = self.judge.evaluate(
                    session["task_spec"],
                    session["rubric"]["criteria"],
                    candidate["prompt_text"],
                    sample["content"],
                    performer_result["response_text"],
                    on_llm_call=lambda: self._charge(session),
                )
            except BudgetExhausted:
                session["stop_reason"] = "budget_exhausted"
                break
            except Exception as exc:
                self._record_failed_evaluation(
                    session_id, candidate, sample, performer_result, evaluation_id,
                    "judge_failed", f"Judge evaluation failed: {type(exc).__name__}",
                )
                continue
            reward = float(judge_result["authoritative_score"])
            if not math.isfinite(reward) or not 0.0 <= reward <= 1.0:
                self._record_failed_evaluation(
                    session_id, candidate, sample, performer_result, evaluation_id, "invalid_reward",
                    "Judge returned an invalid authoritative reward.",
                )
                continue
            update_candidate_stats(candidate, reward)
            successful_count += 1
            evaluation = {
                "evaluation_id": evaluation_id,
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
                "evaluation_source": "mock" if "mock" in {performer_result["source"], judge_result["source"]} else "live",
                "latency": performer_result["latency_ms"],
                "status": "success" if "mock" not in {performer_result["source"], judge_result["source"]} else "mock",
                "created_at": _now(),
            }
            append_evaluation(session_id, evaluation)
            evaluated_batch.append({"candidate_id": candidate_id, "score": reward, "feedback": evaluation})

        if not successful_count and session.get("stop_reason") != "budget_exhausted":
            session["stop_reason"] = "critical_failure"
        top_candidates = sorted(evaluated_batch, key=lambda item: item["score"], reverse=True)[
            : int(session["configuration"]["edited_candidates"])
        ]
        new_candidates: list[dict[str, Any]] = []
        previous_insights = list(session.get("insights", []))
        for parent_item in top_candidates:
            if session.get("status") == "cancelled":
                break
            parent = next(item for item in candidates if item["candidate_id"] == parent_item["candidate_id"])
            evaluation = parent_item["feedback"]
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
            try:
                self._charge(session)
                children = self.generator.generate_edited(
                    parent,
                    feedback,
                    session["task_spec"],
                    count=1,
                    rubric=session["rubric"],
                    insights=previous_insights,
                    original_prompt=session["original_prompt"],
                    iteration=round_number,
                    existing_prompts=[item["prompt_text"] for item in session["candidates"]],
                    on_llm_call=lambda: self._charge(session),
                )
            except BudgetExhausted:
                session["stop_reason"] = "budget_exhausted"
                break
            except Exception as exc:
                session.setdefault("errors", []).append({
                    "iteration": round_number, "candidate_id": parent["candidate_id"],
                    "stage": "stage_a", "error": type(exc).__name__,
                })
                continue
            for child in children:
                append_candidate(session_id, child)
                new_candidates.append(child)

        insight_inputs = [
            {**item["feedback"], "candidate_id": item["candidate_id"]}
            for item in evaluated_batch
        ]
        created_insights = generate_insights(session["task_spec"], insight_inputs, round_number)
        self._merge_insights(session, created_insights, session_id)
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
            update_session(session_id, current_iteration=round_number, status="running")
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
        try:
            report_path = ReportGenerator(output_dir="reports").create_report(session)
            update_session(session_id, report_path=report_path, report_error=None)
        except Exception as exc:
            session["report_error"] = f"Report generation failed: {type(exc).__name__}"
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
        session["status"] = "running"
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
                stop_reason = self.stopping_condition(session, result)
                if stop_reason:
                    session["stop_reason"] = stop_reason
                    break
            if not session.get("stop_reason"):
                session["stop_reason"] = "max_iterations"
            if session.get("status") != "cancelled":
                session["status"] = "final_evaluation"
                self.finalize(session_id)
            else:
                update_session(session_id, status="cancelled", stop_reason="cancelled")
        except BudgetExhausted:
            session["stop_reason"] = "budget_exhausted"
            session["status"] = "final_evaluation"
            self.finalize(session_id)
        except Exception as exc:
            session["stop_reason"] = "critical_failure"
            session["status"] = "failed"
            session.setdefault("errors", []).append({"stage": "optimization", "error": type(exc).__name__})
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
        finalists = candidates[: int(session["configuration"]["final_top_n"])]
        sample_evaluations: list[dict[str, Any]] = []
        baseline_evaluations: list[dict[str, Any]] = []
        for candidate in finalists:
            scores = []
            for sample in session["samples"]:
                if session.get("status") == "cancelled":
                    break
                if session.get("stop_reason") == "budget_exhausted":
                    break
                result = self._final_score(session, candidate["prompt_text"], sample)
                if result:
                    scores.append(result)
                    sample_evaluations.append({"candidate_id": candidate["candidate_id"], **result})
            candidate["final_score"] = (
                sum(item["authoritative_score"] for item in scores) / len(scores)
                if len(scores) == len(session["samples"]) and scores
                and session.get("status") != "cancelled" else None
            )
            if session.get("stop_reason") == "budget_exhausted" or session.get("status") == "cancelled":
                break
        for sample in session["samples"]:
            if session.get("status") == "cancelled":
                break
            if session.get("stop_reason") == "budget_exhausted":
                break
            result = self._final_score(session, session["original_prompt"], sample)
            if result:
                baseline_evaluations.append(result)
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
        final_data = {
            "finalists": ranked,
            "baseline": {
                "prompt_text": session["original_prompt"],
                "score": baseline_score,
                "evaluations": baseline_evaluations,
            },
            "final_ranking": ranked,
            "top3": ranked[:min(3, top_n)],
            "stop_reason": session.get("stop_reason"),
            "created_at": _now(),
        }
        update_session(
            session_id,
            final_evaluation=final_data,
            status=(
                "cancelled" if session.get("status") == "cancelled"
                else "completed" if ranked else "failed"
            ),
            completed_at=_now(),
        )
        return final_data

    def _final_score(self, session: dict[str, Any], prompt: str, sample: dict[str, Any]) -> dict[str, Any] | None:
        try:
            self._charge(session)
            performer_result = self.performer.execute(prompt, sample["content"], self.settings.performer_model)
            if performer_result["status"] not in {"success", "mock"}:
                return None
            judge_result = self.judge.evaluate(
                session["task_spec"], session["rubric"]["criteria"], prompt, sample["content"],
                performer_result["response_text"],
                on_llm_call=lambda: self._charge(session),
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
            "best_score": max(
                (candidate.get("mean_reward", 0.0) for candidate in session.get("candidates", [])),
                default=0.0,
            ),
            "candidate_count": len(session.get("candidates", [])),
            "evaluations": len(session.get("evaluations", [])),
            "llm_call_count": session.get("llm_call_count", 0),
            "max_llm_calls": session.get("configuration", {}).get("max_llm_calls"),
        }
