"""Dataset schema objects for ZEBRAID."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, Mapping

Side = Literal["left", "right"]


class DataSchemaError(ValueError):
    """Raised when a manifest record does not satisfy the dataset schema."""


def _normalize_image_id(value: Any) -> str:
    image_id = str(value).strip()
    if not image_id:
        raise DataSchemaError("image_id cannot be empty")
    return image_id


def _normalize_gps(value: Any) -> str:
    gps = str(value).strip()
    if not gps:
        raise DataSchemaError("gps cannot be empty")

    parts = [part.strip() for part in gps.split(",")]
    if len(parts) != 2:
        raise DataSchemaError("gps must be formatted as 'lat,long'")

    try:
        latitude = float(parts[0])
        longitude = float(parts[1])
    except ValueError as exc:
        raise DataSchemaError("gps must contain numeric latitude and longitude") from exc

    if not -90.0 <= latitude <= 90.0:
        raise DataSchemaError("gps latitude must be between -90 and 90")
    if not -180.0 <= longitude <= 180.0:
        raise DataSchemaError("gps longitude must be between -180 and 180")

    return f"{parts[0]},{parts[1]}"


def _normalize_timestamp(value: Any) -> str:
    timestamp = str(value).strip()
    if not timestamp:
        raise DataSchemaError("timestamp cannot be empty")

    candidate = timestamp.replace("Z", "+00:00")
    try:
        datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise DataSchemaError("timestamp must be a valid ISO 8601 value") from exc

    return timestamp


def _normalize_side(value: Any) -> Side:
    side = str(value).strip().lower()
    if side not in {"left", "right"}:
        raise DataSchemaError("side must be either 'left' or 'right'")
    return side  # type: ignore[return-value]


def _normalize_quality_score(value: Any) -> float:
    try:
        quality_score = float(value)
    except (TypeError, ValueError) as exc:
        raise DataSchemaError("quality_score must be numeric") from exc

    if not 0.0 <= quality_score <= 1.0:
        raise DataSchemaError("quality_score must be between 0 and 1")
    return quality_score


@dataclass(frozen=True, slots=True)
class ZebraDataRecord:
    """Canonical record for a zebra image in the dataset manifest."""

    image_id: str
    gps: str
    timestamp: str
    side: Side
    quality_score: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "image_id", _normalize_image_id(self.image_id))
        object.__setattr__(self, "gps", _normalize_gps(self.gps))
        object.__setattr__(self, "timestamp", _normalize_timestamp(self.timestamp))
        object.__setattr__(self, "side", _normalize_side(self.side))
        object.__setattr__(self, "quality_score", _normalize_quality_score(self.quality_score))

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ZebraDataRecord":
        """Create a record from a mapping with the required schema fields."""

        missing_fields = [field for field in cls.required_fields() if field not in data]
        if missing_fields:
            missing_text = ", ".join(missing_fields)
            raise DataSchemaError(f"missing required fields: {missing_text}")

        return cls(
            image_id=data["image_id"],
            gps=data["gps"],
            timestamp=data["timestamp"],
            side=data["side"],
            quality_score=data["quality_score"],
        )

    @staticmethod
    def required_fields() -> tuple[str, ...]:
        return ("image_id", "gps", "timestamp", "side", "quality_score")

    def to_mapping(self) -> dict[str, Any]:
        """Serialize the record to the canonical manifest schema."""

        return {
            "image_id": self.image_id,
            "gps": self.gps,
            "timestamp": self.timestamp,
            "side": self.side,
            "quality_score": self.quality_score,
        }