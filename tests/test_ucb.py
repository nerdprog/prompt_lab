import math

from app.services.selector.ucb import compute_ucb, select_active_candidates, update_candidate_stats


def test_ucb_handles_untested_candidates() -> None:
    assert math.isinf(compute_ucb(0.0, 0, 1, 1.414))


def test_ucb_selects_best_candidates() -> None:
    candidates = [
        {"candidate_id": "a", "pull_count": 3, "mean_reward": 0.8},
        {"candidate_id": "b", "pull_count": 1, "mean_reward": 0.7},
        {"candidate_id": "c", "pull_count": 0, "mean_reward": 0.0},
    ]
    chosen = select_active_candidates(candidates, k=2, c=1.414)
    assert isinstance(chosen, list)
    assert len(chosen) == 2


def test_update_candidate_stats_works() -> None:
    candidate = {"candidate_id": "x", "pull_count": 0, "total_reward": 0.0, "mean_reward": 0.0}
    updated = update_candidate_stats(candidate, 0.5)
    assert updated["pull_count"] == 1
    assert updated["mean_reward"] == 0.5
