from collections.abc import Iterable


def fbeta(precision: float, recall: float, beta: float = 0.5) -> float:
    if precision == 0.0 and recall == 0.0:
        return 0.0
    beta2 = beta * beta
    denominator = beta2 * precision + recall
    return (1 + beta2) * precision * recall / denominator if denominator else 0.0


def entity_fbeta(predicted: Iterable[str], actual: Iterable[str], beta: float = 0.5) -> float:
    predicted_set = set(predicted)
    actual_set = set(actual)
    if not predicted_set and not actual_set:
        return 1.0
    if not predicted_set:
        return 0.0
    tp = len(predicted_set & actual_set)
    precision = tp / len(predicted_set)
    recall = tp / len(actual_set) if actual_set else 0.0
    return fbeta(precision, recall, beta)


def macro_f05(predictions: dict[str, set[str]], ground_truth: dict[str, set[str]]) -> float:
    scores = [entity_fbeta(predictions.get(s1_id, set()), truth, beta=0.5) for s1_id, truth in ground_truth.items()]
    return sum(scores) / len(scores) if scores else 0.0
