"""Live CCTV identification pipeline with optional DeepSORT tracking."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Literal, Protocol
from uuid import uuid4

import numpy as np

from zebraid.data.quality import QualityDecision, QualityFilterConfig, evaluate_quality
from zebraid.data.stream import LiveFrameRecord
from zebraid.preprocessing import ZebraSegmenter, segment_and_clean


LOGGER = logging.getLogger(__name__)


BBox = tuple[int, int, int, int]


def _bbox_iou(a: BBox, b: BBox) -> float:
    """Compute IoU for two boxes in (x1, y1, x2, y2) format."""

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def default_bbox_provider(frame: LiveFrameRecord) -> BBox:
    """Fallback bbox provider using the full frame extent."""

    height, width = frame.frame.shape[:2]
    return (0, 0, int(width), int(height))


@dataclass
class _TrackState:
    """Minimal track state for live ID stabilization."""

    bbox: BBox
    zebra_id: str | None = None
    missed_frames: int = 0


@dataclass(frozen=True, slots=True)
class Detection:
    """Detector output used by tracker integration."""

    bbox: BBox
    confidence: float = 1.0
    class_name: str = "zebra"


def default_detector(frame: LiveFrameRecord) -> list[Detection]:
    """Fallback detector that treats the full frame as one zebra crop."""

    return [Detection(bbox=default_bbox_provider(frame), confidence=1.0)]


class SortLikeStabilizer:
    """A lightweight SORT-like tracker for suppressing ID flicker.

    This stabilizer keeps one active track per stream and reuses the previous
    zebra ID when bbox overlap stays high across adjacent frames.
    """

    def __init__(self, *, iou_threshold: float = 0.5, max_missed_frames: int = 3) -> None:
        self.iou_threshold = iou_threshold
        self.max_missed_frames = max_missed_frames
        self._tracks: dict[str, _TrackState] = {}

    def stabilize(
        self,
        *,
        stream_id: str,
        bbox: BBox,
        candidate_id: str | None,
    ) -> tuple[str | None, bool]:
        """Return stabilized ID and whether it was reused from track history."""

        track = self._tracks.get(stream_id)
        if track is None or _bbox_iou(track.bbox, bbox) < self.iou_threshold:
            self._tracks[stream_id] = _TrackState(bbox=bbox, zebra_id=candidate_id)
            return candidate_id, False

        track.bbox = bbox
        track.missed_frames = 0

        if track.zebra_id is not None:
            if candidate_id is None or candidate_id != track.zebra_id:
                return track.zebra_id, True
            return candidate_id, False

        if candidate_id is not None:
            track.zebra_id = candidate_id
        return candidate_id, False

    def assign(self, *, stream_id: str, zebra_id: str) -> None:
        """Attach a newly created zebra ID to the active track for this stream."""

        track = self._tracks.get(stream_id)
        if track is not None and track.zebra_id is None:
            track.zebra_id = zebra_id


class DeepSortStabilizer:
    """DeepSORT wrapper that combines motion + appearance association.

    Uses ``deep_sort_realtime`` when available. If the package is missing,
    falls back to ``SortLikeStabilizer`` while preserving the same API.
    """

    def __init__(
        self,
        *,
        max_age: int = 30,
        n_init: int = 2,
        max_cosine_distance: float = 0.2,
        use_fallback: bool = True,
    ) -> None:
        self._fallback: SortLikeStabilizer | None = None
        self.max_age = max_age

        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort

            self._tracker = DeepSort(
                max_age=max_age,
                n_init=n_init,
                max_cosine_distance=max_cosine_distance,
            )
            self._using_deepsort = True
        except Exception as exc:
            if not use_fallback:
                raise
            LOGGER.warning(
                "DeepSORT import failed (%s). Falling back to sort-like stabilizer.",
                exc,
            )
            self._tracker = None
            self._using_deepsort = False
            self._fallback = SortLikeStabilizer()

    @property
    def using_deepsort(self) -> bool:
        return self._using_deepsort

    def update(
        self,
        *,
        frame_uint8_bgr: np.ndarray,
        detections: list[Detection],
        stream_id: str,
    ) -> list[tuple[str, BBox]]:
        """Update tracker and return active tracks as (track_id, bbox)."""

        if not self._using_deepsort:
            if not detections:
                return []
            det = detections[0]
            stabilized_id, _ = self._fallback.stabilize(
                stream_id=stream_id,
                bbox=det.bbox,
                candidate_id="TRACK-0",
            )
            if stabilized_id is None:
                return []
            return [(stabilized_id, det.bbox)]

        tracker_dets: list[tuple[list[float], float, str]] = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            w = max(1, x2 - x1)
            h = max(1, y2 - y1)
            tracker_dets.append(([float(x1), float(y1), float(w), float(h)], float(det.confidence), det.class_name))

        tracks = self._tracker.update_tracks(tracker_dets, frame=frame_uint8_bgr)

        outputs: list[tuple[str, BBox]] = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = str(track.track_id)
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = (int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3]))
            outputs.append((track_id, (x1, y1, x2, y2)))
        return outputs


@dataclass(frozen=True, slots=True)
class IdentificationCandidate:
    """A candidate match produced by the identification model/registry."""

    zebra_id: str | None
    confidence: float


@dataclass(frozen=True, slots=True)
class LiveIdentificationResult:
    """Result emitted after processing one live CCTV frame."""

    frame: LiveFrameRecord
    quality: QualityDecision
    zebra_id: str | None
    confidence: float
    is_new: bool
    accepted: bool
    match: IdentificationCandidate | None
    generated_id: str | None
    reasons: tuple[str, ...]


class FrameSource(Protocol):
    """Protocol for a source that yields live frames."""

    def iter_frames(self) -> Iterator[LiveFrameRecord]:
        """Yield live frames for processing."""


IdentifyFrame = Callable[[np.ndarray], IdentificationCandidate | None]
IdentityFactory = Callable[[LiveFrameRecord], str]
BBoxProvider = Callable[[LiveFrameRecord], BBox]
Detector = Callable[[LiveFrameRecord], list[Detection]]


def default_identity_factory(frame: LiveFrameRecord) -> str:
    """Generate a unique zebra ID for a new live observation."""

    return f"ZEBRA-{frame.stream_id.upper()}-{uuid4().hex[:10].upper()}"


class LiveIdentificationPipeline:
    """Connect a CCTV stream to quality filtering and identity assignment."""

    def __init__(
        self,
        frame_source: FrameSource,
        *,
        identify_frame: IdentifyFrame | None = None,
        identity_factory: IdentityFactory | None = None,
        segmenter: ZebraSegmenter | None = None,
        detector: Detector | None = None,
        bbox_provider: BBoxProvider | None = None,
        stabilizer: SortLikeStabilizer | DeepSortStabilizer | None = None,
        tracking_backend: Literal["deepsort", "sortlike", "none"] = "deepsort",
        max_track_age: int = 30,
        quality_config: QualityFilterConfig | None = None,
        match_threshold: float = 0.85,
        drop_rejected: bool = True,
    ) -> None:
        self.frame_source = frame_source
        self.identify_frame = identify_frame
        self.identity_factory = identity_factory or default_identity_factory
        self.segmenter = segmenter or ZebraSegmenter(backend="sam")
        self.bbox_provider = bbox_provider or default_bbox_provider
        if detector is None:
            self.detector = lambda f: [Detection(bbox=self.bbox_provider(f), confidence=1.0)]
        else:
            self.detector = detector
        self._track_to_zebra: dict[str, str] = {}
        self._track_last_seen: dict[str, int] = {}
        self._track_age_limit = max_track_age
        if stabilizer is not None:
            self.stabilizer = stabilizer
        elif tracking_backend == "deepsort":
            self.stabilizer = DeepSortStabilizer(max_age=max_track_age)
        elif tracking_backend == "sortlike":
            self.stabilizer = SortLikeStabilizer()
        else:
            self.stabilizer = None
        self.quality_config = quality_config or QualityFilterConfig()
        self.match_threshold = match_threshold
        self.drop_rejected = drop_rejected

    @staticmethod
    def _to_uint8_bgr(image: np.ndarray) -> np.ndarray:
        """Normalize image into uint8 BGR for tracker compatibility."""

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("expected 3-channel frame")

        frame = image
        if frame.dtype != np.uint8:
            if np.issubdtype(frame.dtype, np.floating) and float(frame.max(initial=0.0)) <= 1.0:
                frame = frame * 255.0
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        # Stream default is RGB; convert to BGR for OpenCV/DeepSORT expectations.
        return frame[:, :, ::-1].copy()

    @staticmethod
    def _clip_bbox(bbox: BBox, width: int, height: int) -> BBox:
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(x1 + 1, min(x2, width))
        y2 = max(y1 + 1, min(y2, height))
        return (x1, y1, x2, y2)

    def _cleanup_stale_tracks(self, current_frame_index: int) -> None:
        stale = [
            track_id
            for track_id, seen_idx in self._track_last_seen.items()
            if current_frame_index - seen_idx > self._track_age_limit
        ]
        for track_id in stale:
            self._track_last_seen.pop(track_id, None)
            self._track_to_zebra.pop(track_id, None)

    def _process_detection(
        self,
        *,
        frame: LiveFrameRecord,
        quality: QualityDecision,
        frame_uint8_bgr: np.ndarray,
        track_id: str,
        bbox: BBox,
    ) -> LiveIdentificationResult:
        x1, y1, x2, y2 = bbox
        crop = frame_uint8_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            generated_id = self.identity_factory(frame)
            self._track_to_zebra[track_id] = generated_id
            self._track_last_seen[track_id] = frame.frame_index
            return LiveIdentificationResult(
                frame=frame,
                quality=quality,
                zebra_id=generated_id,
                confidence=0.0,
                is_new=True,
                accepted=True,
                match=None,
                generated_id=generated_id,
                reasons=("empty_crop",),
            )

        h, w = crop.shape[:2]
        crop_box = np.array([0, 0, w, h])
        clean_crop = segment_and_clean(crop, segmenter=self.segmenter, box=crop_box)
        candidate = self.identify_frame(clean_crop) if self.identify_frame is not None else None

        tracked_zebra_id = self._track_to_zebra.get(track_id)
        self._track_last_seen[track_id] = frame.frame_index

        if tracked_zebra_id is not None:
            confidence = 0.0 if candidate is None else max(candidate.confidence, self.match_threshold)
            return LiveIdentificationResult(
                frame=frame,
                quality=quality,
                zebra_id=tracked_zebra_id,
                confidence=confidence,
                is_new=False,
                accepted=True,
                match=IdentificationCandidate(zebra_id=tracked_zebra_id, confidence=confidence),
                generated_id=None,
                reasons=("track_reused",),
            )

        if candidate is not None and candidate.zebra_id and candidate.confidence >= self.match_threshold:
            self._track_to_zebra[track_id] = candidate.zebra_id
            return LiveIdentificationResult(
                frame=frame,
                quality=quality,
                zebra_id=candidate.zebra_id,
                confidence=candidate.confidence,
                is_new=False,
                accepted=True,
                match=candidate,
                generated_id=None,
                reasons=(),
            )

        generated_id = self.identity_factory(frame)
        self._track_to_zebra[track_id] = generated_id
        return LiveIdentificationResult(
            frame=frame,
            quality=quality,
            zebra_id=generated_id,
            confidence=0.0 if candidate is None else candidate.confidence,
            is_new=True,
            accepted=True,
            match=candidate,
            generated_id=generated_id,
            reasons=("no_match_found",) if candidate is None else ("match_below_threshold",),
        )

    def process_frame_all(self, frame: LiveFrameRecord) -> list[LiveIdentificationResult]:
        """Process a frame and emit one result per tracked zebra."""

        record_quality = 1.0 if frame.quality_score is None else frame.quality_score
        quality = evaluate_quality(
            frame.frame,
            record_quality_score=record_quality,
            config=self.quality_config,
        )

        if self.drop_rejected and not quality.passed:
            return [
                LiveIdentificationResult(
                    frame=frame,
                    quality=quality,
                    zebra_id=None,
                    confidence=0.0,
                    is_new=False,
                    accepted=False,
                    match=None,
                    generated_id=None,
                    reasons=quality.reasons,
                )
            ]

        # Preserve old behavior when no identifier is provided.
        if self.identify_frame is None:
            generated_id = self.identity_factory(frame)
            return [
                LiveIdentificationResult(
                    frame=frame,
                    quality=quality,
                    zebra_id=generated_id,
                    confidence=0.0,
                    is_new=True,
                    accepted=True,
                    match=None,
                    generated_id=generated_id,
                    reasons=("no_match_found",),
                )
            ]

        frame_uint8_bgr = self._to_uint8_bgr(frame.frame)
        height, width = frame_uint8_bgr.shape[:2]

        detections = self.detector(frame)
        if not detections:
            return [
                LiveIdentificationResult(
                    frame=frame,
                    quality=quality,
                    zebra_id=None,
                    confidence=0.0,
                    is_new=False,
                    accepted=False,
                    match=None,
                    generated_id=None,
                    reasons=("no_detections",),
                )
            ]

        clipped_detections = [
            Detection(
                bbox=self._clip_bbox(det.bbox, width, height),
                confidence=det.confidence,
                class_name=det.class_name,
            )
            for det in detections
        ]

        tracks: list[tuple[str, BBox]]
        if self.stabilizer is None:
            tracks = [
                (f"det-{idx}", det.bbox)
                for idx, det in enumerate(clipped_detections)
            ]
        elif isinstance(self.stabilizer, DeepSortStabilizer) or hasattr(self.stabilizer, "update"):
            tracks = self.stabilizer.update(
                frame_uint8_bgr=frame_uint8_bgr,
                detections=clipped_detections,
                stream_id=frame.stream_id,
            )
        else:
            tracks = []
            for idx, det in enumerate(clipped_detections):
                tid, _ = self.stabilizer.stabilize(
                    stream_id=f"{frame.stream_id}:{idx}",
                    bbox=det.bbox,
                    candidate_id=f"track-{idx}",
                )
                if tid is not None:
                    tracks.append((tid, det.bbox))

        if not tracks:
            # If DeepSORT is still warming up (unconfirmed tracks), bootstrap with raw detections.
            tracks = [(f"bootstrap-{idx}", det.bbox) for idx, det in enumerate(clipped_detections)]

        results = [
            self._process_detection(
                frame=frame,
                quality=quality,
                frame_uint8_bgr=frame_uint8_bgr,
                track_id=track_id,
                bbox=self._clip_bbox(bbox, width, height),
            )
            for track_id, bbox in tracks
        ]

        self._cleanup_stale_tracks(frame.frame_index)
        return results

    def process_frame(self, frame: LiveFrameRecord) -> LiveIdentificationResult:
        """Process a single live frame and return the first track result.

        For backward compatibility this returns only the first emitted result.
        Use :meth:`process_frame_all` or :meth:`run` for full multi-track output.
        """

        return self.process_frame_all(frame)[0]

    def run(self) -> Iterator[LiveIdentificationResult]:
        """Iterate over the frame source and yield one result per tracked zebra."""

        for frame in self.frame_source.iter_frames():
            for result in self.process_frame_all(frame):
                yield result
