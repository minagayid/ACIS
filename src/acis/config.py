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
    species_scope: tuple[str, ...]
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
    raw_species = data["species"]
    configured_species = str(raw_species) if isinstance(raw_species, str) else "multi_species"
    raw_scope = data.get("species_scope", raw_species if isinstance(raw_species, list) else [raw_species])
    if not isinstance(raw_scope, list) or not raw_scope or any(not isinstance(item, str) or not item.strip() for item in raw_scope):
        raise ValueError("config species_scope must be a non-empty list of species identifiers")
    species_scope = tuple(dict.fromkeys(item.strip() for item in raw_scope))
    if configured_species != "multi_species" and configured_species not in species_scope:
        raise ValueError("config species must be included in species_scope")
    return ProjectConfig(
        schema_version=str(data["schema_version"]),
        species=configured_species,
        species_scope=species_scope,
        model=str(data["model"]),
        features=features,
        labels=labels,
        abstention_threshold=threshold,
        claim_scope=str(data["claim_scope"]),
    )
