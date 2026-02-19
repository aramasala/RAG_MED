"""Unit tests for evaluation metrics."""

from rag_med.evaluation.metrics import evaluate_answer_pair


def test_evaluate_answer_pair_basic() -> None:
    ref = "Повышение артериального давления"
    cand = "Повышение артериального давления."

    metrics = evaluate_answer_pair(ref, cand)

    assert metrics["exact_match"] is True
    assert 0.0 <= metrics["primary_score"] <= 1.0
    assert metrics["token_f1"]["f1"] == 1.0
    assert metrics["rouge_l"]["f1"] == 1.0
