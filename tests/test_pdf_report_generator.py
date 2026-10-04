from __future__ import annotations

from app.services.pdf.report_generator import ReportGenerator


def test_candidates_evaluated_counts_unique_successful_candidate_ids():
    session = {
        "evaluations": [
            {"candidate_id": "candidate-a", "status": "success"},
            {"candidate_id": "candidate-a", "status": "mock"},
            {"candidate_id": "candidate-b", "status": "success"},
            {"candidate_id": "candidate-c", "status": "failed"},
            {"status": "success"},
        ]
    }

    assert ReportGenerator._count_evaluated_candidates(session) == 2


def test_summary_uses_recorded_feedback_not_task_intent():
    candidate = {
        "candidate_id": "candidate-a",
        "final_evaluations": [
            {"feedback": {"strengths": ["The response follows the requested structure."]}}
        ],
    }

    assert ReportGenerator._build_summary_sentence(candidate) == (
        "Final evaluation feedback for the top-ranked prompt highlighted: "
        "The response follows the requested structure."
    )


def test_summary_and_candidate_explanations_are_neutral_without_evidence():
    candidate = {"candidate_id": "candidate-a", "final_evaluations": []}

    assert ReportGenerator._build_summary_sentence(candidate) == (
        "No specific improvement evidence was recorded for the best candidate."
    )
    assert ReportGenerator._collect_feedback_texts(candidate, "strengths") == []
