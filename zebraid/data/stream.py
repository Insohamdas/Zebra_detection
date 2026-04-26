"""Live CCTV stream ingestion helpers for ZEBRAID."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

import cv2
import numpy as np

Side = Literal["left", "right"]


class StreamError(RuntimeError):
    """Base error raised for CCTV stream failures."""


class StreamOpenError(StreamError):
    """Raised when a video capture source cannot be opened."""


class StreamReadError(StreamError):
    """Raised when a stream cannot produce frames."""


@dataclass(frozen=True, slots=True)
class CCTVStreamConfig:
    """Configuration for a live CCTV or RTSP stream."""

    source: str | int
    stream_id: str = "cctv"
    side: Side = "left"
    frame_stride: int = 1
    max_frames: int | None = None
    resize_to: tuple[int, int] | None = (512, 512)
    normalize: bool = True
    color_space: Literal["rgb", "bgr"] = "rgb"

    def __post_init__(self) -> None:
        if self.frame_stride <= 0:
            raise ValueError("frame_stride must be a positive integer")
        if self.max_frames is not None and self.max_frames <= 0:
            raise ValueError("max_frames must be positive when provided")
        if self.resize_to is not None and any(d <= 0 for d in self.resize_to):
            raise ValueError("resize_to must contain positive dimensions")


@dataclass(frozen=True, slots=True)
class LiveFrameRecord:
    """A single frame captured from a CCTV stream."""

    frame_id: str
    stream_id: str
    source: str
    side: Side
    timestamp: str
    frame_index: int
    frame: np.ndarray
    quality_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_mapping(self) -> dict[str, Any]:
        """Serialize the record into a JSON-friendly payload."""

        return {
            "frame_id": self.frame_id,
            "stream_id": self.stream_id,
            "source": self.source,
            "side": self.side,
            "timestamp": self.timestamp,
            "frame_index": self.frame_index,
            "quality_score": self.quality_score,
            "metadata": dict(self.metadata),
        }


class VideoCaptureStreamSource:
    """Read frames from a local camera index, RTSP URL, or HTTP video stream."""

    def __init__(
        self,
        config: CCTVStreamConfig,
        *,
        capture_factory: Callable[[str | int], Any] = cv2.VideoCapture,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._config = config
        self._capture_factory = capture_factory
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    @property
    def config(self) -> CCTVStreamConfig:
        return self._config

    def _prepare_frame(self, frame: np.ndarray) -> np.ndarray:
        prepared = frame

        if prepared.ndim == 2:
            prepared = cv2.cvtColor(prepared, cv2.COLOR_GRAY2RGB)
        elif prepared.ndim == 3 and prepared.shape[2] == 4:
            prepared = cv2.cvtColor(prepared, cv2.COLOR_BGRA2RGB)
        elif self._config.color_space == "rgb":
            prepared = cv2.cvtColor(prepared, cv2.COLOR_BGR2RGB)

        if self._config.resize_to is not None:
            target_width, target_height = self._config.resize_to
            interpolation = (
                cv2.INTER_AREA
                if prepared.shape[0] > target_height or prepared.shape[1] > target_width
                else cv2.INTER_CUBIC
            )
            prepared = cv2.resize(prepared, (target_width, target_height), interpolation=interpolation)

        if self._config.normalize:
            prepared = prepared.astype(np.float32, copy=False) / 255.0

        return prepared

    def _build_record(self, frame: np.ndarray, frame_index: int) -> LiveFrameRecord:
        timestamp = self._clock().astimezone(timezone.utc).isoformat()
        frame_id = f"{self._config.stream_id}_{frame_index:06d}"
        metadata = {
            "capture_source": str(self._config.source),
            "color_space": self._config.color_space,
        }
        return LiveFrameRecord(
            frame_id=frame_id,
            stream_id=self._config.stream_id,
            source=str(self._config.source),
            side=self._config.side,
            timestamp=timestamp,
            frame_index=frame_index,
            frame=self._prepare_frame(frame),
            metadata=metadata,
        )

    def iter_frames(self) -> Iterator[LiveFrameRecord]:
        """Yield processed frames from the configured stream source."""

        capture = self._capture_factory(self._config.source)
        try:
            if not hasattr(capture, "isOpened") or not capture.isOpened():
                raise StreamOpenError(f"Unable to open video source: {self._config.source}")

            emitted = 0
            frame_index = 0

            while True:
                if self._config.max_frames is not None and emitted >= self._config.max_frames:
                    break

                success, frame = capture.read()
                if not success:
                    if emitted == 0:
                        raise StreamReadError(f"No frames available from {self._config.source}")
                    break

                if frame is None:
                    frame_index += 1
                    continue

                if frame_index % self._config.frame_stride != 0:
                    frame_index += 1
                    continue

                yield self._build_record(frame, frame_index)
                emitted += 1
                frame_index += 1
        finally:
            if hasattr(capture, "release"):
                capture.release()
