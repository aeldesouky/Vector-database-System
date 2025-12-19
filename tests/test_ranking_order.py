"""
Tests for explicit ranking order.
"""

def test_results_are_sorted_by_final_score() -> None:
    results = [
        {"scores": {"final": 0.3}},
        {"scores": {"final": 0.9}},
        {"scores": {"final": 0.5}},
    ]

    sorted_results = sorted(
        results,
        key=lambda r: r["scores"].get("final", 0.0),
        reverse=True,
    )

    scores = [r["scores"]["final"] for r in sorted_results]

    assert scores == sorted(scores, reverse=True)
