from __future__ import annotations

import math
from typing import Any


def compute_ucb(mean_reward: float, pull_count: int, total_pulls: int, c: float = 1.414) -> float:
    if not math.isfinite(mean_reward) or not 0.0 <= mean_reward <= 1.0:
        raise ValueError("Mean reward must be a finite value between 0 and 1.")
    if pull_count < 0 or total_pulls < 0 or not math.isfinite(c) or c <= 0:
        raise ValueError("UCB counts must be nonnegative and exploration constant must be positive.")
    if pull_count == 0:
        return float("inf")
    if total_pulls <= 0:
        return mean_reward
    exploration_term = c * math.sqrt(math.log(total_pulls) / pull_count)
    return mean_reward + exploration_term


def select_active_candidates(candidates: list[dict[str, Any]], k: int = 6, c: float = 1.414) -> list[str]:
    if not candidates:
        return []
    total_pulls = max(1, sum(int(item.get("pull_count", 0)) for item in candidates))
    ranked = []
    for candidate in candidates:
        pull_count = int(candidate.get("pull_count", 0))
        mean_reward = float(candidate.get("mean_reward", 0.0))
        ucb_value = compute_ucb(mean_reward, pull_count, total_pulls, c)
        exploration_term = ucb_value - mean_reward if math.isfinite(ucb_value) else float("inf")
        ranked.append({
            "candidate_id": candidate["candidate_id"],
            "pull_count": pull_count,
            "mean_reward": mean_reward,
            "exploration_term": exploration_term,
            "ucb": ucb_value,
        })
    ranked.sort(key=lambda x: x["ucb"], reverse=True)
    selected = [item["candidate_id"] for item in ranked[:k]]
    return selected


def update_candidate_stats(candidate: dict[str, Any], reward: float) -> dict[str, Any]:
    reward = float(reward)
    if not math.isfinite(reward) or not 0.0 <= reward <= 1.0:
        raise ValueError("Reward must be finite and between 0 and 1.")
    candidate.setdefault("pull_count", 0)
    candidate.setdefault("total_reward", 0.0)
    candidate.setdefault("mean_reward", 0.0)
    candidate["pull_count"] = int(candidate["pull_count"]) + 1
    candidate["total_reward"] = float(candidate.get("total_reward", 0.0)) + float(reward)
    candidate["mean_reward"] = candidate["total_reward"] / candidate["pull_count"]
    candidate["last_score"] = reward
    return candidate
