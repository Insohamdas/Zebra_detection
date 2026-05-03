"""Tests for the asynchronous video identification API endpoints."""

from __future__ import annotations

import os
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from zebraid.api.app import create_app



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
        status = client.get(f"/process-video/{job_id}")
        assert status.status_code == 200
        status_payload = status.json()
        if status_payload["status"] == "completed":
            break
        time.sleep(0.05)

    assert status_payload is not None
    assert status_payload["status"] == "completed"
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
