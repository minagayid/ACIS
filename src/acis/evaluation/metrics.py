"""Small, explicit metrics for a leakage-aware baseline evaluation."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable


def majority_class_accuracy(labels: Iterable[str], truth: list[str] | None = None) -> float:
    labels = list(labels)
    if truth is None:
        truth = labels
    if not truth:
        return 0.0
    majority = Counter(labels).most_common(1)[0][0] if labels else "unknown"
    return sum(label == majority for label in truth) / len(truth)


def macro_f1(truth: list[str], predicted: list[str], labels: list[str]) -> float:
    if not truth:
        return 0.0
    scores: list[float] = []
    for label in labels:
        true_positive = sum(actual == label and guess == label for actual, guess in zip(truth, predicted))
        false_positive = sum(actual != label and guess == label for actual, guess in zip(truth, predicted))
        false_negative = sum(actual == label and guess != label for actual, guess in zip(truth, predicted))
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def balanced_accuracy(truth: list[str], predicted: list[str], labels: list[str]) -> float:
    recalls = []
    for label in labels:
        total = sum(actual == label for actual in truth)
        if total:
            recalls.append(sum(actual == label and guess == label for actual, guess in zip(truth, predicted)) / total)
    return sum(recalls) / len(recalls) if recalls else 0.0


def brier_score(truth: list[str], probabilities: list[dict[str, float]]) -> float:
    if not truth:
        return 0.0
    labels = sorted(set(truth) | {label for row in probabilities for label in row})
    total = 0.0
    for actual, row in zip(truth, probabilities):
        total += sum((row.get(label, 0.0) - float(label == actual)) ** 2 for label in labels)
    return total / len(truth)


def evaluate_predictions(
    truth: list[str],
    predicted: list[str],
    probabilities: list[dict[str, float]],
    *,
    abstained: list[bool] | None = None,
    animal_ids: list[str] | None = None,
    baseline_labels: list[str] | None = None,
) -> dict[str, object]:
    labels = sorted(set(truth) | set(predicted))
    accuracy = sum(actual == guess for actual, guess in zip(truth, predicted)) / len(truth) if truth else 0.0
    abstained = abstained or [False] * len(truth)
    non_abstained = [index for index, value in enumerate(abstained) if not value]
    selective_accuracy = (
        sum(truth[index] == predicted[index] for index in non_abstained) / len(non_abstained)
        if non_abstained else 0.0
    )
    per_animal: dict[str, dict[str, float]] = {}
    if animal_ids:
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, animal_id in enumerate(animal_ids):
            grouped[animal_id].append(index)
        for animal_id, indices in grouped.items():
            per_animal[animal_id] = {
                "accuracy": sum(truth[i] == predicted[i] for i in indices) / len(indices),
                "count": float(len(indices)),
            }
    return {
        "count": len(truth),
        "labels": labels,
        "accuracy": round(accuracy, 6),
        "macro_f1": round(macro_f1(truth, predicted, labels), 6),
        "balanced_accuracy": round(balanced_accuracy(truth, predicted, labels), 6),
        "brier_score": round(brier_score(truth, probabilities), 6),
        "abstention_rate": round(sum(abstained) / len(abstained), 6) if abstained else 0.0,
        "selective_accuracy": round(selective_accuracy, 6),
        "majority_accuracy": round(majority_class_accuracy(baseline_labels or truth, truth), 6),
        "per_animal": per_animal,
    }
