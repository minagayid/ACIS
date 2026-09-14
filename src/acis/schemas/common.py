"""Small dependency-free validation helpers used by the data contracts."""

from __future__ import annotations

from datetime import datetime
import math
from typing import Any, Iterable


class ValidationError(ValueError):
    """Raised when an input does not satisfy an ACIS contract."""


def require_string(data: dict[str, Any], key: str, *, allow_empty: bool = False) -> str:
    value = data.get(key)
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ValidationError(f"{key} must be a non-empty string")
    return value


def optional_string(data: dict[str, Any], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValidationError(f"{key} must be a string or null")
    return value


def require_number(data: dict[str, Any], key: str, *, minimum: float | None = None, maximum: float | None = None) -> float:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{key} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValidationError(f"{key} must be finite")
    if minimum is not None and number < minimum:
        raise ValidationError(f"{key} must be >= {minimum}")
    if maximum is not None and number > maximum:
        raise ValidationError(f"{key} must be <= {maximum}")
    return number


def require_int(data: dict[str, Any], key: str, *, minimum: int | None = None, maximum: int | None = None) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"{key} must be an integer")
    if minimum is not None and value < minimum:
        raise ValidationError(f"{key} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValidationError(f"{key} must be <= {maximum}")
    return value


def require_list(data: dict[str, Any], key: str, *, item_type: type = str) -> list[Any]:
    value = data.get(key)
    if not isinstance(value, list) or not all(isinstance(item, item_type) for item in value):
        raise ValidationError(f"{key} must be a list of {item_type.__name__}")
    return value


def optional_list(data: dict[str, Any], key: str, *, item_type: type = str) -> list[Any]:
    if key not in data or data[key] is None:
        return []
    return require_list(data, key, item_type=item_type)


def parse_timestamp(value: str, key: str = "captured_at") -> str:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValidationError(f"{key} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValidationError(f"{key} must include a timezone")
    return value


def copy_unknown_keys(data: dict[str, Any], allowed: Iterable[str]) -> None:
    unknown = sorted(set(data) - set(allowed))
    if unknown:
        raise ValidationError(f"unknown fields: {', '.join(unknown)}")
