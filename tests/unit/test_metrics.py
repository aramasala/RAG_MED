"""Unit tests for evaluation metrics (RAGAS-based)."""

from rag_med.evaluation.metrics import compare_two_answers, evaluate_answer_pair


def test_compare_two_answers_cosine_similarity() -> None:
    """compare_two_answers includes cosine_similarity in [0, 1]."""
    m = compare_two_answers("one two three", "one two four")
    assert "cosine_similarity" in m
    assert 0.0 <= m["cosine_similarity"] <= 1.0
    m2 = compare_two_answers("same", "same")
    assert m2["cosine_similarity"] == 1.0
    m3 = compare_two_answers("a b c", "x y z")
    assert m3["cosine_similarity"] == 0.0


def test_evaluate_answer_pair_empty_context() -> None:
    """With empty retrieved_contexts, returns zero faithfulness."""
    metrics = evaluate_answer_pair(
        question="Какой препарат назначают?",
        response="Назначают препарат X.",
        retrieved_contexts=[],
    )
    assert metrics["faithfulness"] == 0.0


def test_evaluate_answer_pair_return_keys() -> None:
    """Return dict has faithfulness (with empty context)."""
    metrics = evaluate_answer_pair(
        question="Test question",
        response="Test answer",
        retrieved_contexts=[],
    )
    assert "faithfulness" in metrics
    assert 0.0 <= metrics["faithfulness"] <= 1.0
