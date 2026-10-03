from app.services.optimizer.candidate_generator import CandidateGenerator


def test_initial_candidates_get_one_semantic_repair(monkeypatch):
    monkeypatch.setenv("ALLOW_MOCK_LLMS", "false")
    generator = CandidateGenerator()
    requests = []
    responses = [
        [
            {
                "prompt_text": "Create a structured seven-day Python plan for beginners.",
                "generation_reason": "Organizes the required progression.",
                "new_assumptions": ["Study two hours every day."],
            },
            {
                "prompt_text": "Design a progressive Python learning schedule with daily coding practice.",
                "generation_reason": "Emphasizes daily practice.",
                "new_assumptions": [],
            },
        ],
        [
            {
                "prompt_text": "Create a seven-day Python curriculum progressing from beginner to intermediate.",
                "generation_reason": "Organizes concepts by increasing difficulty.",
                "new_assumptions": [],
            },
            {
                "prompt_text": "Build a weeklong Python study sequence with varied exercises and a final project.",
                "generation_reason": "Emphasizes hands-on practice and a culminating project.",
                "new_assumptions": [],
            },
        ],
    ]

    def generate_initial_candidates(
        task_spec, prompt, count, rubric, on_llm_call, validation_feedback=None
    ):
        requests.append(validation_feedback)
        if on_llm_call:
            on_llm_call()
        return [dict(item) for item in responses[len(requests) - 1]]

    generator.optimizer.generate_initial_candidates = generate_initial_candidates
    calls = []
    candidates = generator.generate_initial(
        {"audience": "beginner programmers", "forbidden_assumptions": []},
        "Create a seven-day beginner Python study plan.",
        count=2,
        on_llm_call=lambda: calls.append(1),
    )

    assert len(candidates) == 2
    assert requests[0] is None
    assert "intent-fidelity or diversity validation" in requests[1]
    assert len(calls) == 2
    assert all(not candidate["new_assumptions"] for candidate in candidates)
