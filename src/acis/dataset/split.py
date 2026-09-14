"""Leakage-resistant grouped splits.

The default grouping unit is the animal. Session and site IDs are returned so
callers can audit whether later pipelines need stricter holdouts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .loaders import TrainingExample


@dataclass(frozen=True)
class GroupSplit:
    train: list[TrainingExample]
    validation: list[TrainingExample]
    test: list[TrainingExample]
    train_animals: tuple[str, ...]
    validation_animals: tuple[str, ...]
    test_animals: tuple[str, ...]

    def assert_no_animal_overlap(self) -> None:
        groups = [set(self.train_animals), set(self.validation_animals), set(self.test_animals)]
        if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
            raise AssertionError("animal IDs overlap across split partitions")


def grouped_split(
    examples: Iterable[TrainingExample],
    *,
    train_ratio: float = 0.6,
    validation_ratio: float = 0.2,
) -> GroupSplit:
    if train_ratio <= 0 or validation_ratio < 0 or train_ratio + validation_ratio >= 1:
        raise ValueError("train_ratio must be > 0 and train_ratio + validation_ratio must be < 1")
    examples = list(examples)
    animals = sorted({example.animal_id for example in examples})
    if len(animals) < 3:
        raise ValueError("at least three animals are required for train/validation/test splitting")
    train_count = max(1, round(len(animals) * train_ratio))
    validation_count = max(1, round(len(animals) * validation_ratio)) if validation_ratio else 0
    if train_count + validation_count >= len(animals):
        train_count = max(1, len(animals) - 2)
        validation_count = 1
    train_animals = tuple(animals[:train_count])
    validation_animals = tuple(animals[train_count:train_count + validation_count])
    test_animals = tuple(animals[train_count + validation_count:])
    train_set, validation_set, test_set = set(train_animals), set(validation_animals), set(test_animals)
    split = GroupSplit(
        [item for item in examples if item.animal_id in train_set],
        [item for item in examples if item.animal_id in validation_set],
        [item for item in examples if item.animal_id in test_set],
        train_animals,
        validation_animals,
        test_animals,
    )
    split.assert_no_animal_overlap()
    if not split.validation or not split.test:
        raise ValueError("grouped split must contain non-empty validation and test partitions")
    return split
