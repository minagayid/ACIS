"""Deterministic acoustic feature utilities.

The fixture corpus supplies measured summaries. A future audio backend can replace
the measurement adapter without changing the model or prediction contracts.
"""

from __future__ import annotations

import math

from acis.schemas.annotation import AnnotationSegment


FEATURE_NAMES = (
    "duration_ms",
    "peak_frequency_hz",
    "spectral_centroid_hz",
    "energy_db",
    "repetition_count",
    "voiced_fraction",
)


def acoustic_vector(segment: AnnotationSegment) -> tuple[float, ...]:
    supplied = segment.acoustic_features
    # Duration is derived from the validated annotation interval. All other
    # values must come from an explicit feature extractor measurement.
    missing = [name for name in FEATURE_NAMES if name != "duration_ms" and name not in supplied]
    if missing:
        raise ValueError(f"missing acoustic features: {', '.join(missing)}")
    vector = tuple(
        float(segment.end_ms - segment.start_ms) if name == "duration_ms" else float(supplied[name])
        for name in FEATURE_NAMES
    )
    if any(not math.isfinite(value) for value in vector):
        raise ValueError("acoustic features must be finite")
    return vector


def standardize(vectors: list[tuple[float, ...]]) -> tuple[list[tuple[float, ...]], tuple[float, ...], tuple[float, ...]]:
    if not vectors:
        return [], (), ()
    width = len(vectors[0])
    means = tuple(sum(row[index] for row in vectors) / len(vectors) for index in range(width))
    stds = tuple(
        math.sqrt(sum((row[index] - means[index]) ** 2 for row in vectors) / len(vectors)) or 1.0
        for index in range(width)
    )
    normalized = [tuple((row[index] - means[index]) / stds[index] for index in range(width)) for row in vectors]
    return normalized, means, stds


def apply_standardization(vector: tuple[float, ...], means: tuple[float, ...], stds: tuple[float, ...]) -> tuple[float, ...]:
    if not means:
        return vector
    return tuple((value - means[index]) / (stds[index] or 1.0) for index, value in enumerate(vector))


def cosine_similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)
