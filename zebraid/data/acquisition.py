"""Helpers for collecting and serializing zebra dataset manifests."""

from __future__ import annotations

import csv
import json
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

from .schema import ZebraDataRecord

PathResolver = Callable[[ZebraDataRecord], Path]

IMAGE_SUFFIXES: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def discover_image_files(
    source_root: str | Path,
    *,
    suffixes: Sequence[str] = IMAGE_SUFFIXES,
) -> list[Path]:
    """Find image files under a camera-trap or public dataset directory."""

    root = Path(source_root)
    suffix_set = {suffix.lower() for suffix in suffixes}
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in suffix_set
    )


def load_records_from_csv(
    csv_path: str | Path,
    *,
    column_map: dict[str, str] | None = None,
) -> list[ZebraDataRecord]:
    """Read a CSV export into canonical dataset records.

    ``column_map`` can be used when the source dataset uses different column
    names, e.g. ``{"image_id": "filename", "timestamp": "captured_at"}``.
    """

    csv_path = Path(csv_path)
    column_map = column_map or {}

    records: list[ZebraDataRecord] = []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {csv_path}")

        for row in reader:
            normalized: dict[str, Any] = {}
            for field in ZebraDataRecord.required_fields():
                source_key = column_map.get(field, field)
                if source_key not in row:
                    raise ValueError(
                        f"CSV file {csv_path} is missing required column '{source_key}'"
                    )
                normalized[field] = row[source_key]
            records.append(ZebraDataRecord.from_mapping(normalized))

    return records


def load_records_from_jsonl(jsonl_path: str | Path) -> list[ZebraDataRecord]:
    """Read a newline-delimited JSON manifest."""

    jsonl_path = Path(jsonl_path)
    records: list[ZebraDataRecord] = []

    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON on line {line_number} of {jsonl_path}"
                ) from exc
            records.append(ZebraDataRecord.from_mapping(payload))

    return records


def load_records_from_json(json_path: str | Path) -> list[ZebraDataRecord]:
    """Read a JSON manifest containing a list of records or a ``records`` key."""

    json_path = Path(json_path)
    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict):
        payload = payload.get("records", payload)

    if not isinstance(payload, list):
        raise ValueError("JSON manifest must contain a list of records")

    return [ZebraDataRecord.from_mapping(item) for item in payload]


def load_manifest(
    manifest_path: str | Path,
    *,
    column_map: dict[str, str] | None = None,
) -> list[ZebraDataRecord]:
    """Load a manifest from CSV, JSON, or JSONL."""

    manifest_path = Path(manifest_path)
    suffix = manifest_path.suffix.lower()
    if suffix == ".csv":
        return load_records_from_csv(manifest_path, column_map=column_map)
    if suffix in {".jsonl", ".ndjson"}:
        return load_records_from_jsonl(manifest_path)
    if suffix == ".json":
        return load_records_from_json(manifest_path)

    raise ValueError(f"unsupported manifest format: {manifest_path.suffix}")


def save_manifest(
    records: Iterable[ZebraDataRecord],
    manifest_path: str | Path,
) -> Path:
    """Write records to JSON or JSONL for reuse in later stages."""

    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    records = list(records)

    if manifest_path.suffix.lower() == ".json":
        payload = [record.to_mapping() for record in records]
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        return manifest_path

    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_mapping(), ensure_ascii=False))
            handle.write("\n")

    return manifest_path


def build_path_resolver(
    image_root: str | Path,
    *,
    suffixes: Sequence[str] = IMAGE_SUFFIXES,
) -> PathResolver:
    """Return a helper that resolves image paths from a record's image_id."""

    root = Path(image_root)
    suffix_set = tuple(dict.fromkeys(suffixes))

    def resolve(record: ZebraDataRecord) -> Path:
        candidates: list[Path] = []
        raw_id = Path(record.image_id)

        if raw_id.suffix:
            candidates.append(root / raw_id)
        else:
            candidates.append(root / record.image_id)
            for suffix in suffix_set:
                candidates.append(root / f"{record.image_id}{suffix}")

        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                return candidate

        for candidate in root.rglob(f"{record.image_id}*"):
            if candidate.is_file() and candidate.suffix.lower() in suffix_set:
                return candidate

        raise FileNotFoundError(
            f"unable to resolve image path for {record.image_id!r} in {root}"
        )

    return resolve