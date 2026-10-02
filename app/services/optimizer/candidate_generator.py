from __future__ import annotations

import uuid
from difflib import SequenceMatcher
from datetime import datetime, timezone
from typing import Callable

from app.services.intent.intent_fidelity import analyze_intent_fidelity
from app.services.llm.optimizer import OptimizerClient


class CandidateGenerator:
    def __init__(self) -> None:
        self.optimizer = OptimizerClient()

    def _utc_now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _validate_diversity(candidates: list[dict], existing_prompts: list[str] | None = None) -> None:
        seen = list(existing_prompts or [])
        for candidate in candidates:
            text = " ".join(candidate["prompt_text"].lower().split())
            for previous in seen:
                normalized = " ".join(previous.lower().split())
                if SequenceMatcher(None, text, normalized).ratio() >= 0.92:
                    raise ValueError("Candidate generation returned an exact or near-duplicate prompt.")
            seen.append(candidate["prompt_text"])

    @staticmethod
    def _validate_intent(candidate: dict, task_spec: dict) -> None:
        fidelity = analyze_intent_fidelity(task_spec, candidate["prompt_text"])
        if fidelity["unsupported_assumptions"] or fidelity["score"] < 0.7:
            raise ValueError("Generated candidate introduced a forbidden assumption or failed intent-fidelity validation.")
        if candidate.get("new_assumptions"):
            raise ValueError("Generated candidate reports unsupported new assumptions.")

    def generate_initial(
        self,
        task_spec: dict,
        prompt: str,
        count: int = 6,
        rubric: dict | None = None,
        on_llm_call: Callable[[], None] | None = None,
    ) -> list[dict]:
        candidates = self.optimizer.generate_initial_candidates(
            task_spec, prompt, count, rubric, on_llm_call
        )
        if len(candidates) != count:
            raise ValueError(f"Candidate generator returned {len(candidates)} candidates; requested {count}.")
        for candidate in candidates:
            candidate.setdefault("candidate_id", f"cand-{uuid.uuid4().hex[:8]}")
            candidate.setdefault("created_at", self._utc_now())
            candidate.setdefault("status", "new")
            candidate.setdefault("pull_count", 0)
            candidate.setdefault("mean_reward", 0.0)
            candidate.setdefault("total_reward", 0.0)
            self._validate_intent(candidate, task_spec)
        self._validate_diversity(candidates, [prompt])
        return candidates

    def generate_edited(
        self,
        parent: dict,
        feedback: dict,
        task_spec: dict,
        count: int = 1,
        rubric: dict | None = None,
        insights: list[dict] | None = None,
        original_prompt: str = "",
        iteration: int = 0,
        existing_prompts: list[str] | None = None,
        on_llm_call: Callable[[], None] | None = None,
    ) -> list[dict]:
        edited = self.optimizer.generate_edited_candidate(
            parent, feedback, task_spec, count, rubric, insights, original_prompt, iteration,
            on_llm_call,
        )
        if len(edited) != count:
            raise ValueError(f"Candidate editor returned {len(edited)} children; requested {count}.")
        for candidate in edited:
            candidate.setdefault("candidate_id", f"cand-{uuid.uuid4().hex[:8]}")
            candidate.setdefault("generated_at", self._utc_now())
            candidate.setdefault("parent_id", parent.get("candidate_id"))
            candidate.setdefault("generation", int(parent.get("generation", 0)) + 1)
            candidate.setdefault("created_at", self._utc_now())
            candidate.setdefault("status", "new")
            candidate.setdefault("pull_count", 0)
            candidate.setdefault("mean_reward", 0.0)
            candidate.setdefault("total_reward", 0.0)
            if not candidate.get("prompt_text", "").strip():
                raise ValueError("Generated child prompt is empty.")
            if candidate["prompt_text"].strip() == parent["prompt_text"].strip():
                raise ValueError("Generated child prompt duplicates its parent.")
            self._validate_intent(candidate, task_spec)
        self._validate_diversity(edited, existing_prompts)
        return edited
