"""Tests for the asynchronous video identification API endpoints."""

from __future__ import annotations

import os
import time
import importlib
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from zebraid.api.app import create_app

app_module = importlib.import_module("zebraid.api.app")



def _make_test_video(video_path: Path, frame_count: int = 45) -> None:
    """Create a simple AVI video with varying brightness."""

    width, height = 96, 64
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(video_path), fourcc, 15.0, (width, height))
    assert writer.isOpened(), "VideoWriter failed to open"

    for index in range(frame_count):
        value = (index * 5) % 255
        frame = np.full((height, width, 3), value, dtype=np.uint8)
        writer.write(frame)

    writer.release()



def _create_client() -> TestClient:
    os.environ["IDENTIFY_MOCK"] = "1"
    return TestClient(create_app())



def test_process_video_returns_async_job_and_completes(tmp_path: Path) -> None:
    client = _create_client()
    video_path = tmp_path / "sample.avi"
    _make_test_video(video_path)

    with video_path.open("rb") as fh:
        response = client.post(
            "/process-video",
            files={"video": ("sample.avi", fh, "video/x-msvideo")},
        )

    assert response.status_code == 202
    payload = response.json()
    assert payload["status"] == "queued"
    assert isinstance(payload["job_id"], str)
    assert payload["job_id"]

    job_id = payload["job_id"]

    status_payload = None
    for _ in range(50):
        status = client.get(f"/video-status/{job_id}")
        assert status.status_code == 200
        status_payload = status.json()
        if status_payload["status"] == "completed":
            break
        time.sleep(0.05)

    assert status_payload is not None
    assert status_payload["status"] == "completed"
    assert status_payload["sampled_frames"] == 3
    assert status_payload["estimated_total_samples"] == 3
    assert status_payload["progress"] == 1.0
    assert status_payload["total_frames_processed"] == 3
    assert len(status_payload["unique_zebras"]) == 3
    assert {item["zebra_id"] for item in status_payload["unique_zebras"]} == {
        "MOCK_ZEBRA_00",
        "MOCK_ZEBRA_03",
        "MOCK_ZEBRA_06",
    }



def test_process_video_rejects_unsupported_format(tmp_path: Path) -> None:
    client = _create_client()
    fake_video = tmp_path / "sample.txt"
    fake_video.write_bytes(b"not a video")

    with fake_video.open("rb") as fh:
        response = client.post(
            "/process-video",
            files={"video": ("sample.txt", fh, "text/plain")},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "unsupported_format"


def test_process_video_aggregates_best_confidence_thumbnail_and_flags_drift(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = _create_client()
    video_path = tmp_path / "sample.avi"
    _make_test_video(video_path, frame_count=3)

    monkeypatch.setattr(app_module, "_VIDEO_SAMPLE_RATE", 1)
    monkeypatch.setattr(app_module, "_VIDEO_MAX_SAMPLES", 10)

    code_zero = np.zeros(512, dtype=np.uint8)
    code_drift = np.zeros(512, dtype=np.uint8)
    code_drift[:200] = 1
    code_one = np.ones(512, dtype=np.uint8)

    def fake_identify_zebras_in_frame(frame, *, frame_id, ref_image):
        frame_no = int(frame_id.rsplit("_", 1)[-1])
        if frame_no == 0:
            return [
                app_module._FrameIdentification(
                    result=app_module.VideoIdentificationItem(
                        zebra_id="ZEBRA-A",
                        confidence=0.70,
                        is_new=False,
                    ),
                    global_code=code_zero.copy(),
                    quality_score=0.70,
                    thumbnail_bytes=b"A-low",
                ),
                app_module._FrameIdentification(
                    result=app_module.VideoIdentificationItem(
                        zebra_id="ZEBRA-B",
                        confidence=0.80,
                        is_new=False,
                    ),
                    global_code=code_one.copy(),
                    quality_score=0.80,
                    thumbnail_bytes=b"B-best",
                ),
            ]
        if frame_no == 1:
            return [
                app_module._FrameIdentification(
                    result=app_module.VideoIdentificationItem(
                        zebra_id="ZEBRA-A",
                        confidence=0.92,
                        is_new=False,
                    ),
                    global_code=code_zero.copy(),
                    quality_score=0.92,
                    thumbnail_bytes=b"A-best",
                )
            ]
        return [
            app_module._FrameIdentification(
                result=app_module.VideoIdentificationItem(
                    zebra_id="ZEBRA-A",
                    confidence=0.85,
                    is_new=False,
                ),
                global_code=code_drift.copy(),
                quality_score=0.75,
                thumbnail_bytes=b"A-drift",
            )
        ]

    monkeypatch.setattr(app_module, "_identify_zebras_in_frame", fake_identify_zebras_in_frame)

    with video_path.open("rb") as fh:
        response = client.post(
            "/process-video",
            files={"video": ("sample.avi", fh, "video/x-msvideo")},
        )

    assert response.status_code == 202
    job_id = response.json()["job_id"]

    status_payload = None
    for _ in range(50):
        status = client.get(f"/video-status/{job_id}")
        assert status.status_code == 200
        status_payload = status.json()
        if status_payload["status"] == "completed":
            break
        time.sleep(0.05)

    assert status_payload is not None
    assert status_payload["status"] == "completed"
    assert status_payload["total_frames_processed"] == 3
    assert len(status_payload["unique_zebras"]) == 2

    zebras = {item["zebra_id"]: item for item in status_payload["unique_zebras"]}
    assert zebras["ZEBRA-A"]["confidence"] == 0.92
    assert zebras["ZEBRA-A"]["flagged_for_review"] is True
    assert zebras["ZEBRA-A"]["thumbnail_jpeg_base64"] == "QS1iZXN0"
    assert zebras["ZEBRA-B"]["confidence"] == 0.80
    assert zebras["ZEBRA-B"]["flagged_for_review"] is False
    assert zebras["ZEBRA-B"]["thumbnail_jpeg_base64"] == "Qi1iZXN0"
