from datetime import datetime, timezone

import cv2
import numpy as np
import torch

from zebraid.data.stream import CCTVStreamConfig, LiveFrameRecord, VideoCaptureStreamSource
from zebraid.data.quality import QualityFilterConfig
from zebraid.pipelines.live_identification import (
    Detection,
    IdentificationCandidate,
    LiveIdentificationPipeline,
)
from zebraid.pipelines.real_identify import create_real_identifier


def _make_test_image() -> np.ndarray:
    image = np.zeros((96, 144, 3), dtype=np.uint8)
    cv2.rectangle(image, (18, 16), (126, 80), (255, 255, 255), thickness=-1)
    cv2.line(image, (0, 95), (143, 0), (0, 0, 255), thickness=3)
    return image


class FakeCapture:
    def __init__(self, frames: list[np.ndarray]) -> None:
        self._frames = frames
        self._index = 0
        self.released = False

    def isOpened(self) -> bool:
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._index >= len(self._frames):
            return False, None
        frame = self._frames[self._index]
        self._index += 1
        return True, frame

    def release(self) -> None:
        self.released = True


class StaticFrameSource:
    def __init__(self, frames: list[LiveFrameRecord]) -> None:
        self._frames = frames

    def iter_frames(self):
        yield from self._frames


def test_video_capture_stream_source_iterates_normalized_frames() -> None:
    frames = [_make_test_image(), np.flipud(_make_test_image()), _make_test_image()]
    fake_capture = FakeCapture(frames)

    source = VideoCaptureStreamSource(
        CCTVStreamConfig(
            source=0,
            stream_id="cam-01",
            side="left",
            frame_stride=2,
            max_frames=1,
            resize_to=(32, 32),
            normalize=True,
        ),
        capture_factory=lambda _: fake_capture,
        clock=lambda: datetime(2026, 4, 22, 12, 0, 0, tzinfo=timezone.utc),
    )

    records = list(source.iter_frames())

    assert len(records) == 1
    record = records[0]
    assert record.frame_id == "cam-01_000000"
    assert record.stream_id == "cam-01"
    assert record.source == "0"
    assert record.side == "left"
    assert record.timestamp == "2026-04-22T12:00:00+00:00"
    assert record.frame.shape == (32, 32, 3)
    assert record.frame.dtype == np.float32
    assert 0.0 <= float(record.frame.min()) <= 1.0
    assert 0.0 <= float(record.frame.max()) <= 1.0
    assert fake_capture.released is True


def test_live_pipeline_generates_new_id_when_no_match() -> None:
    frame = LiveFrameRecord(
        frame_id="cam-01_000001",
        stream_id="cam-01",
        source="rtsp://camera.local/stream",
        side="right",
        timestamp="2026-04-22T12:00:01+00:00",
        frame_index=1,
        frame=_make_test_image().astype(np.float32) / 255.0,
    )

    pipeline = LiveIdentificationPipeline(
        StaticFrameSource([frame]),
        identify_frame=lambda _: None,
        identity_factory=lambda current: f"ZEBRA-{current.frame_index:04d}",
        quality_config=QualityFilterConfig(
            min_record_quality_score=0.5,
            min_visual_quality_score=0.2,
            min_blur_variance=10.0,
            min_brightness=5.0,
            min_contrast=5.0,
        ),
    )

    results = list(pipeline.run())

    assert len(results) == 1
    result = results[0]
    assert result.accepted is True
    assert result.is_new is True
    assert result.zebra_id == "ZEBRA-0001"
    assert result.generated_id == "ZEBRA-0001"
    assert result.match is None
    assert result.reasons == ("no_match_found",)


def test_live_pipeline_uses_existing_match_when_confident() -> None:
    frame = LiveFrameRecord(
        frame_id="cam-01_000002",
        stream_id="cam-01",
        source="rtsp://camera.local/stream",
        side="right",
        timestamp="2026-04-22T12:00:02+00:00",
        frame_index=2,
        frame=_make_test_image().astype(np.float32) / 255.0,
    )

    pipeline = LiveIdentificationPipeline(
        StaticFrameSource([frame]),
        identify_frame=lambda _: IdentificationCandidate("ZEBRA-007", 0.97),
        quality_config=QualityFilterConfig(
            min_record_quality_score=0.5,
            min_visual_quality_score=0.2,
            min_blur_variance=10.0,
            min_brightness=5.0,
            min_contrast=5.0,
        ),
    )

    results = list(pipeline.run())

    assert len(results) == 1
    result = results[0]
    assert result.accepted is True
    assert result.is_new is False
    assert result.zebra_id == "ZEBRA-007"
    assert result.generated_id is None
    assert result.match == IdentificationCandidate("ZEBRA-007", 0.97)
    assert result.reasons == ()


def test_live_pipeline_segments_before_identify() -> None:
    frame = LiveFrameRecord(
        frame_id="cam-01_000003",
        stream_id="cam-01",
        source="rtsp://camera.local/stream",
        side="right",
        timestamp="2026-04-22T12:00:03+00:00",
        frame_index=3,
        frame=_make_test_image().astype(np.float32) / 255.0,
    )

    class RecordingSegmenter:
        def __init__(self) -> None:
            self.calls = 0

        def segment(self, image: np.ndarray) -> np.ndarray:
            self.calls += 1
            assert image.shape == (96, 144, 3)
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            mask[:, :72] = 1
            return mask

    seen_shapes: list[tuple[int, ...]] = []

    def identify_frame(clean_frame: np.ndarray) -> IdentificationCandidate:
        seen_shapes.append(clean_frame.shape)
        assert clean_frame.shape == (256, 256, 3)
        assert clean_frame.dtype == np.uint8
        return IdentificationCandidate("ZEBRA-SEG", 0.99)

    pipeline = LiveIdentificationPipeline(
        StaticFrameSource([frame]),
        identify_frame=identify_frame,
        segmenter=RecordingSegmenter(),
        quality_config=QualityFilterConfig(
            min_record_quality_score=0.5,
            min_visual_quality_score=0.2,
            min_blur_variance=10.0,
            min_brightness=5.0,
            min_contrast=5.0,
        ),
    )

    results = list(pipeline.run())

    assert len(results) == 1
    result = results[0]
    assert result.accepted is True
    assert result.is_new is False
    assert result.zebra_id == "ZEBRA-SEG"
    assert seen_shapes == [(256, 256, 3)]


def test_live_pipeline_with_real_identifier_segments_once() -> None:
    frame = LiveFrameRecord(
        frame_id="cam-01_000004",
        stream_id="cam-01",
        source="rtsp://camera.local/stream",
        side="right",
        timestamp="2026-04-22T12:00:04+00:00",
        frame_index=4,
        frame=_make_test_image().astype(np.float32) / 255.0,
    )

    class LiveSegmenter:
        def __init__(self) -> None:
            self.calls = 0

        def segment(self, image: np.ndarray) -> np.ndarray:
            self.calls += 1
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            mask[:, :72] = 1
            return mask

    class IdentifierSegmenter:
        def __init__(self) -> None:
            self.calls = 0

        def segment(self, image: np.ndarray) -> np.ndarray:
            self.calls += 1
            raise AssertionError("real identifier should not segment pre-cleaned live frames")

    class DummyEncoder:
        def encode(self, image_tensor: torch.Tensor) -> torch.Tensor:
            assert image_tensor.shape == (1, 3, 256, 256)
            return torch.ones((1, 2048), dtype=torch.float32)

    identifier_segmenter = IdentifierSegmenter()
    identify_fn = create_real_identifier(
        encoder=DummyEncoder(),
        segmenter=identifier_segmenter,
        segment_input=False,
    )

    live_segmenter = LiveSegmenter()
    pipeline = LiveIdentificationPipeline(
        StaticFrameSource([frame]),
        identify_frame=identify_fn,
        segmenter=live_segmenter,
        quality_config=QualityFilterConfig(
            min_record_quality_score=0.5,
            min_visual_quality_score=0.2,
            min_blur_variance=10.0,
            min_brightness=5.0,
            min_contrast=5.0,
        ),
    )

    results = list(pipeline.run())

    assert len(results) == 1
    result = results[0]
    assert result.accepted is True
    assert result.is_new is False
    assert result.zebra_id is not None
    assert live_segmenter.calls == 1
    assert identifier_segmenter.calls == 0


def test_live_pipeline_reuses_id_when_bbox_stays_same() -> None:
    frame1 = LiveFrameRecord(
        frame_id="cam-01_000010",
        stream_id="cam-01",
        source="rtsp://camera.local/stream",
        side="right",
        timestamp="2026-04-22T12:00:10+00:00",
        frame_index=10,
        frame=_make_test_image().astype(np.float32) / 255.0,
    )
    frame2 = LiveFrameRecord(
        frame_id="cam-01_000011",
        stream_id="cam-01",
        source="rtsp://camera.local/stream",
        side="right",
        timestamp="2026-04-22T12:00:11+00:00",
        frame_index=11,
        frame=_make_test_image().astype(np.float32) / 255.0,
    )

    calls = {"count": 0}

    def flickery_identify(_: np.ndarray) -> IdentificationCandidate:
        calls["count"] += 1
        if calls["count"] == 1:
            return IdentificationCandidate("ZEBRA-A", 0.97)
        return IdentificationCandidate("ZEBRA-B", 0.99)

    pipeline = LiveIdentificationPipeline(
        StaticFrameSource([frame1, frame2]),
        identify_frame=flickery_identify,
        bbox_provider=lambda _: (10, 10, 70, 70),
        quality_config=QualityFilterConfig(
            min_record_quality_score=0.5,
            min_visual_quality_score=0.2,
            min_blur_variance=10.0,
            min_brightness=5.0,
            min_contrast=5.0,
        ),
    )

    results = list(pipeline.run())
    assert len(results) == 2
    assert results[0].zebra_id == "ZEBRA-A"
    assert results[1].zebra_id == "ZEBRA-A"
    assert results[1].reasons == ("track_reused",)


def test_live_pipeline_reuses_previous_id_when_second_match_is_missing() -> None:
    frame1 = LiveFrameRecord(
        frame_id="cam-01_000020",
        stream_id="cam-01",
        source="rtsp://camera.local/stream",
        side="right",
        timestamp="2026-04-22T12:00:20+00:00",
        frame_index=20,
        frame=_make_test_image().astype(np.float32) / 255.0,
    )
    frame2 = LiveFrameRecord(
        frame_id="cam-01_000021",
        stream_id="cam-01",
        source="rtsp://camera.local/stream",
        side="right",
        timestamp="2026-04-22T12:00:21+00:00",
        frame_index=21,
        frame=_make_test_image().astype(np.float32) / 255.0,
    )

    calls = {"count": 0}

    def intermittently_missing_identify(_: np.ndarray) -> IdentificationCandidate | None:
        calls["count"] += 1
        if calls["count"] == 1:
            return IdentificationCandidate("ZEBRA-STABLE", 0.96)
        return None

    pipeline = LiveIdentificationPipeline(
        StaticFrameSource([frame1, frame2]),
        identify_frame=intermittently_missing_identify,
        bbox_provider=lambda _: (12, 12, 72, 72),
        quality_config=QualityFilterConfig(
            min_record_quality_score=0.5,
            min_visual_quality_score=0.2,
            min_blur_variance=10.0,
            min_brightness=5.0,
            min_contrast=5.0,
        ),
    )

    results = list(pipeline.run())
    assert len(results) == 2
    assert results[0].zebra_id == "ZEBRA-STABLE"
    assert results[1].zebra_id == "ZEBRA-STABLE"
    assert results[1].is_new is False
    assert results[1].reasons == ("track_reused",)


def test_live_pipeline_processes_multiple_detections_in_one_frame() -> None:
    frame = LiveFrameRecord(
        frame_id="cam-01_000030",
        stream_id="cam-01",
        source="rtsp://camera.local/stream",
        side="right",
        timestamp="2026-04-22T12:00:30+00:00",
        frame_index=30,
        frame=_make_test_image().astype(np.float32) / 255.0,
    )

    identify_ids = iter(["ZEBRA-M1", "ZEBRA-M2"])

    def identify_frame(_: np.ndarray) -> IdentificationCandidate:
        return IdentificationCandidate(next(identify_ids), 0.97)

    pipeline = LiveIdentificationPipeline(
        StaticFrameSource([frame]),
        identify_frame=identify_frame,
        detector=lambda _: [
            Detection((5, 5, 70, 80), 0.95),
            Detection((72, 8, 140, 90), 0.93),
        ],
        tracking_backend="none",
        quality_config=QualityFilterConfig(
            min_record_quality_score=0.5,
            min_visual_quality_score=0.2,
            min_blur_variance=10.0,
            min_brightness=5.0,
            min_contrast=5.0,
        ),
    )

    results = list(pipeline.run())
    assert len(results) == 2
    assert {r.zebra_id for r in results} == {"ZEBRA-M1", "ZEBRA-M2"}
    assert all(r.accepted for r in results)


def test_live_pipeline_track_lifecycle_reuses_id_across_frames() -> None:
    frame1 = LiveFrameRecord(
        frame_id="cam-01_000040",
        stream_id="cam-01",
        source="rtsp://camera.local/stream",
        side="right",
        timestamp="2026-04-22T12:00:40+00:00",
        frame_index=40,
        frame=_make_test_image().astype(np.float32) / 255.0,
    )
    frame2 = LiveFrameRecord(
        frame_id="cam-01_000041",
        stream_id="cam-01",
        source="rtsp://camera.local/stream",
        side="right",
        timestamp="2026-04-22T12:00:41+00:00",
        frame_index=41,
        frame=_make_test_image().astype(np.float32) / 255.0,
    )

    calls = {"count": 0}

    def identify_frame(_: np.ndarray) -> IdentificationCandidate:
        calls["count"] += 1
        return IdentificationCandidate(f"ZEBRA-LIFE-{calls['count']}", 0.98)

    class FixedTrackDeepSort:
        def update(self, *, frame_uint8_bgr, detections, stream_id):
            return [("track-42", detections[0].bbox)]

    pipeline = LiveIdentificationPipeline(
        StaticFrameSource([frame1, frame2]),
        identify_frame=identify_frame,
        detector=lambda _: [Detection((10, 10, 70, 70), 0.95)],
        stabilizer=FixedTrackDeepSort(),
        quality_config=QualityFilterConfig(
            min_record_quality_score=0.5,
            min_visual_quality_score=0.2,
            min_blur_variance=10.0,
            min_brightness=5.0,
            min_contrast=5.0,
        ),
    )

    results = list(pipeline.run())
    assert len(results) == 2
    assert results[0].zebra_id == "ZEBRA-LIFE-1"
    assert results[1].zebra_id == "ZEBRA-LIFE-1"
    assert results[1].reasons == ("track_reused",)