"""Dataset loading, resizing, normalization, and quality filtering."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable, Iterator
from pathlib import Path

import cv2
import numpy as np

from .acquisition import PathResolver, build_path_resolver, load_manifest
from .quality import QualityDecision, QualityFilterConfig, evaluate_quality
from .schema import ZebraDataRecord


@dataclass(frozen=True, slots=True)
class LoadedSample:
    """A loaded and preprocessed zebra image plus its manifest record."""

    record: ZebraDataRecord
    image_path: Path
    image: np.ndarray
    quality: QualityDecision


class ZebraDataLoader:
    """Load zebra images, resize them to 512x512, and normalize pixel values."""

    def __init__(
        self,
        records: Iterable[ZebraDataRecord],
        resolve_path: PathResolver,
        *,
        image_size: tuple[int, int] = (512, 512),
        quality_config: QualityFilterConfig | None = None,
        drop_rejected: bool = True,
    ) -> None:
        self._records = list(records)
        self._resolve_path = resolve_path
        self.image_size = image_size
        self.quality_config = quality_config or QualityFilterConfig()
        self.drop_rejected = drop_rejected

    @classmethod
    def from_manifest(
        cls,
        manifest_path: str | Path,
        resolve_path: PathResolver,
        *,
        column_map: dict[str, str] | None = None,
        image_size: tuple[int, int] = (512, 512),
        quality_config: QualityFilterConfig | None = None,
        drop_rejected: bool = True,
    ) -> "ZebraDataLoader":
        records = load_manifest(manifest_path, column_map=column_map)
        return cls(
            records,
            resolve_path,
            image_size=image_size,
            quality_config=quality_config,
            drop_rejected=drop_rejected,
        )

    @classmethod
    def from_image_root(
        cls,
        manifest_path: str | Path,
        image_root: str | Path,
        *,
        column_map: dict[str, str] | None = None,
        image_size: tuple[int, int] = (512, 512),
        quality_config: QualityFilterConfig | None = None,
        drop_rejected: bool = True,
    ) -> "ZebraDataLoader":
        return cls.from_manifest(
            manifest_path,
            build_path_resolver(image_root),
            column_map=column_map,
            image_size=image_size,
            quality_config=quality_config,
            drop_rejected=drop_rejected,
        )

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[LoadedSample]:
        return self.iter_samples()

    def _load_image(self, image_path: Path) -> np.ndarray:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"unable to read image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        width, height = self.image_size
        resized = cv2.resize(
            image,
            (width, height),
            interpolation=cv2.INTER_AREA
            if image.shape[0] > height or image.shape[1] > width
            else cv2.INTER_CUBIC,
        )
        normalized = resized.astype(np.float32) / 255.0
        return normalized

    def load_record(self, record: ZebraDataRecord) -> LoadedSample | None:
        image_path = self._resolve_path(record)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"unable to read image: {image_path}")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        quality = evaluate_quality(
            image_rgb,
            record_quality_score=record.quality_score,
            config=self.quality_config,
        )

        if self.drop_rejected and not quality.passed:
            return None

        normalized = self._load_image(image_path)
        return LoadedSample(
            record=record,
            image_path=image_path,
            image=normalized,
            quality=quality,
        )

    def iter_samples(self) -> Iterator[LoadedSample]:
        for record in self._records:
            sample = self.load_record(record)
            if sample is not None:
                yield sample

    def load_all(self) -> list[LoadedSample]:
        return list(self.iter_samples())