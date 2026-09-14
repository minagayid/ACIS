"""Interval utilities for reproducible segmentation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SegmentWindow:
    start_ms: int
    end_ms: int

    def __post_init__(self) -> None:
        if self.start_ms < 0 or self.end_ms <= self.start_ms:
            raise ValueError("segment window must have 0 <= start_ms < end_ms")

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


def intersection_over_union(left: SegmentWindow, right: SegmentWindow) -> float:
    intersection = max(0, min(left.end_ms, right.end_ms) - max(left.start_ms, right.start_ms))
    union = max(left.end_ms, right.end_ms) - min(left.start_ms, right.start_ms)
    return intersection / union if union else 0.0
