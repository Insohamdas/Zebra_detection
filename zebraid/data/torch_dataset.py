"""Torch Dataset wrapper for zebra training batches."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

try:  # pragma: no cover - exercised based on environment availability
	import torch
	from torch.utils.data import Dataset
except Exception:  # pragma: no cover
	torch = None

	class Dataset:  # type: ignore[override]
		"""Fallback Dataset when torch is unavailable."""

		pass


class ZebraDataset(Dataset):
	"""Training dataset that returns ``(image, label)`` for each sample."""

	def __init__(
		self,
		samples: Sequence[Any],
		*,
		image_key: str = "image",
		label_key: str = "label",
		transform: Callable[[Any], Any] | None = None,
		target_transform: Callable[[Any], Any] | None = None,
	) -> None:
		if len(samples) == 0:
			raise ValueError("samples must not be empty")

		self.samples = list(samples)
		self.image_key = image_key
		self.label_key = label_key
		self.transform = transform
		self.target_transform = target_transform

	def __len__(self) -> int:
		return len(self.samples)

	def _extract(self, sample: Any) -> tuple[Any, Any]:
		if isinstance(sample, Mapping):
			if self.image_key not in sample or self.label_key not in sample:
				raise KeyError(
					f"sample mapping must contain keys '{self.image_key}' and '{self.label_key}'"
				)
			return sample[self.image_key], sample[self.label_key]

		if isinstance(sample, (tuple, list)) and len(sample) >= 2:
			return sample[0], sample[1]

		raise TypeError("sample must be a mapping with image/label keys or a tuple(image, label)")

	def _to_torch(self, image: Any) -> Any:
		if torch is None:
			return image

		if isinstance(image, np.ndarray):
			tensor = torch.from_numpy(image)
			if tensor.ndim == 3 and tensor.shape[-1] in (1, 3, 4):
				tensor = tensor.permute(2, 0, 1)
			if tensor.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
				tensor = tensor.float() / 255.0
			return tensor

		return image

	def __getitem__(self, idx: int) -> tuple[Any, Any]:
		image, label = self._extract(self.samples[idx])

		if self.transform is not None:
			image = self.transform(image)
		else:
			image = self._to_torch(image)

		if self.target_transform is not None:
			label = self.target_transform(label)

		return image, label