"""WAV validation without silently modifying source recordings."""

from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WavSpec:
    sample_rate_hz: int
    channels: int
    sample_width_bytes: int
    frame_count: int
    duration_ms: int


def validate_wav(path: str | Path, *, max_duration_ms: int | None = None) -> WavSpec:
    path = Path(path)
    try:
        with wave.open(str(path), "rb") as handle:
            sample_rate = handle.getframerate()
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            frame_count = handle.getnframes()
    except (FileNotFoundError, wave.Error) as exc:
        raise ValueError(f"invalid WAV file {path}: {exc}") from exc
    if sample_rate <= 0 or channels <= 0 or sample_width <= 0:
        raise ValueError(f"invalid WAV metadata in {path}")
    duration_ms = round(frame_count / sample_rate * 1000)
    if duration_ms <= 0:
        raise ValueError(f"WAV file {path} has no audio frames")
    if max_duration_ms is not None and duration_ms > max_duration_ms:
        raise ValueError(f"WAV file {path} is {duration_ms}ms, over limit {max_duration_ms}ms")
    return WavSpec(sample_rate, channels, sample_width, frame_count, duration_ms)
