"""Runtime configuration loaded from a small JSON file."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from acis.audio.features import FEATURE_NAMES
from acis.schemas.annotation import ALLOWED_CONTEXTS


@dataclass(frozen=True)
class ProjectConfig:
    schema_version: str
    species: str
    model: str
    features: tuple[str, ...]
    labels: tuple[str, ...]
    abstention_threshold: float
    claim_scope: str


def load_config(path: str | Path) -> ProjectConfig:
    path = Path(path)
    try:
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid config {path}: {exc}") from exc
    required = {"schema_version", "species", "model", "features", "labels", "abstention_threshold", "claim_scope"}
    missing = sorted(required - set(data))
    if missing:
        raise ValueError(f"config missing fields: {', '.join(missing)}")
    features = tuple(data["features"])
    labels = tuple(data["labels"])
    if features != FEATURE_NAMES:
        raise ValueError(f"config features must exactly match {list(FEATURE_NAMES)}")
    if not labels or any(label not in ALLOWED_CONTEXTS for label in labels):
        raise ValueError(f"config labels must use only {sorted(ALLOWED_CONTEXTS)}")
    try:
        threshold = float(data["abstention_threshold"])
    except (TypeError, ValueError) as exc:
        raise ValueError("config abstention_threshold must be numeric") from exc
    if not math.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("config abstention_threshold must be in (0, 1]")
    return ProjectConfig(
        schema_version=str(data["schema_version"]),
        species=str(data["species"]),
        model=str(data["model"]),
        features=features,
        labels=labels,
        abstention_threshold=threshold,
        claim_scope=str(data["claim_scope"]),
    )
