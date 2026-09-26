from src.evaluation import entity_fbeta


def test_empty_singleton_scores_one():
    assert entity_fbeta([], []) == 1.0


def test_false_positive_on_singleton_scores_zero():
    assert entity_fbeta(["S2-1"], []) == 0.0
