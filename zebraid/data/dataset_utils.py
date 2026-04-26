"""Utilities for validating and splitting zebra datasets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def validate_dataset(dataset: Sequence[Mapping[str, Any]]) -> None:
	"""Validate that dataset records contain required keys.

	This intentionally follows an assert-style contract requested for the
	training pipeline bootstrap phase.
	"""

	assert len(dataset) > 0, "dataset must not be empty"
	for index, item in enumerate(dataset):
		assert "image_id" in item, f"dataset[{index}] missing 'image_id'"
		assert "gps" in item, f"dataset[{index}] missing 'gps'"


def split_dataset(
	data: Sequence[Any],
	*,
	train_ratio: float = 0.8,
	val_ratio: float = 0.1,
	test_ratio: float = 0.1,
) -> tuple[list[Any], list[Any], list[Any]]:
	"""Split dataset into train/val/test partitions.

	Default behavior matches:
	- train: data[:80%]
	- val: data[80:90%]
	- test: data[90:]
	"""

	ratio_total = train_ratio + val_ratio + test_ratio
	if abs(ratio_total - 1.0) > 1e-6:
		raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

	if any(r < 0 for r in (train_ratio, val_ratio, test_ratio)):
		raise ValueError("split ratios must be non-negative")

	items = list(data)
	n_total = len(items)
	train_end = int(n_total * train_ratio)
	val_end = train_end + int(n_total * val_ratio)

	train = items[:train_end]
	val = items[train_end:val_end]
	test = items[val_end:]
	return train, val, test