"""Recording manifest contract: provenance and raw-media metadata only."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .common import ValidationError, copy_unknown_keys, optional_string, parse_timestamp, require_int, require_string


CONSENT_STATUSES = frozenset({"documented", "pending", "not_applicable", "withdrawn", "restricted"})


@dataclass(frozen=True)
class Recording:
    recording_id: str
    animal_id: str
    species: str
    file_uri: str
    captured_at: str
    session_id: str
    site_id: str
    device_id: str
    location_class: str
    consent_status: str
    collector_id: str | None = None
    # Optional taxonomy and social metadata let the same contract represent
    # domestic, wild, terrestrial, marine, and invertebrate studies without
    # requiring exact locations or species-specific columns.
    subspecies: str | None = None
    population: str | None = None
    region: str | None = None
    habitat: str | None = None
    wild_or_domestic: str | None = None
    social_group_id: str | None = None
    age_class: str | None = None
    sex: str | None = None
    duration_ms: int | None = None
    sample_rate_hz: int | None = None
    channels: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Recording":
        allowed = {
            "recording_id", "animal_id", "species", "file_uri", "captured_at", "session_id",
            "site_id", "device_id", "location_class", "collector_id", "duration_ms",
            "sample_rate_hz", "channels", "consent_status", "subspecies", "population",
            "region", "habitat", "wild_or_domestic", "social_group_id", "age_class", "sex",
        }
        copy_unknown_keys(data, allowed)
        duration = require_int(data, "duration_ms", minimum=1) if data.get("duration_ms") is not None else None
        sample_rate = require_int(data, "sample_rate_hz", minimum=1) if data.get("sample_rate_hz") is not None else None
        channels = require_int(data, "channels", minimum=1) if data.get("channels") is not None else None
        consent_status = require_string(data, "consent_status")
        if consent_status not in CONSENT_STATUSES:
            raise ValidationError(f"consent_status must be one of {sorted(CONSENT_STATUSES)}")
        return cls(
            recording_id=require_string(data, "recording_id"),
            animal_id=require_string(data, "animal_id"),
            species=require_string(data, "species"),
            file_uri=require_string(data, "file_uri"),
            captured_at=parse_timestamp(require_string(data, "captured_at")),
            session_id=require_string(data, "session_id"),
            site_id=require_string(data, "site_id"),
            device_id=require_string(data, "device_id"),
            location_class=require_string(data, "location_class"),
            collector_id=optional_string(data, "collector_id"),
            duration_ms=duration,
            sample_rate_hz=sample_rate,
            channels=channels,
            consent_status=consent_status,
            subspecies=optional_string(data, "subspecies"),
            population=optional_string(data, "population"),
            region=optional_string(data, "region"),
            habitat=optional_string(data, "habitat"),
            wild_or_domestic=optional_string(data, "wild_or_domestic"),
            social_group_id=optional_string(data, "social_group_id"),
            age_class=optional_string(data, "age_class"),
            sex=optional_string(data, "sex"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
