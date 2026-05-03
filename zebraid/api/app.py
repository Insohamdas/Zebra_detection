"""FastAPI foundation for the ZEBRAID API."""

from __future__ import annotations

import logging
import os
import threading
import tempfile
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from typing import Annotated

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from zebraid.feature_engine import (
    FeatureEncoder, 
    FlankClassifier,
    engineered_stripe_features,
    combine_features
)
from zebraid.matching import MatchingEngine
from zebraid.registry import FaissStore
from zebraid.preprocessing import ZebraSegmenter, prepare_tensor
from zebraid.id_generator import global_itq_code, local_patch_codes

LOGGER = logging.getLogger(__name__)

# Global state for matching pipeline
_registry: FaissStore | None = None
_engine: MatchingEngine | None = None
_encoder: FeatureEncoder | None = None
_segmenter: ZebraSegmenter | None = None
_flank_classifier: FlankClassifier | None = None
_detector = None


class IdentificationResponse(BaseModel):
    """Response from zebra identification endpoint."""

    zebra_id: str
    confidence: float
    is_new: bool


class VideoIdentificationResponse(BaseModel):
    """Response from video identification endpoint."""

    unique_zebras: list[IdentificationResponse]
    total_frames_processed: int


class VideoJobAcceptedResponse(BaseModel):
    """Response returned when a video job has been queued."""

    job_id: str
    status: str


class VideoJobStatusResponse(BaseModel):
    """Current status for an asynchronous video identification job."""

    job_id: str
    status: str
    unique_zebras: list[IdentificationResponse] | None = None
    total_frames_processed: int | None = None
    error: str | None = None


@dataclass
class _VideoJobRecord:
    status: str
    result: VideoIdentificationResponse | None = None
    error: str | None = None


_VIDEO_JOBS: dict[str, _VideoJobRecord] = {}
_VIDEO_JOBS_LOCK = threading.Lock()
_VIDEO_SAMPLE_RATE = 15
_VIDEO_MAX_SAMPLES = 100


def _mock_identification_from_frame(frame: np.ndarray) -> IdentificationResponse:
    """Return a deterministic mock zebra identity for a frame."""

    mean_value = float(np.mean(frame)) if frame.size else 0.0
    bucket = int(mean_value * 10) if mean_value <= 1.0 else int(mean_value // 25)
    bucket = max(0, min(bucket, 99))
    return IdentificationResponse(
        zebra_id=f"MOCK_ZEBRA_{bucket:02d}",
        confidence=0.90,
        is_new=False,
    )


def _get_video_job(job_id: str) -> _VideoJobRecord | None:
    with _VIDEO_JOBS_LOCK:
        return _VIDEO_JOBS.get(job_id)


def _set_video_job(job_id: str, record: _VideoJobRecord) -> None:
    with _VIDEO_JOBS_LOCK:
        _VIDEO_JOBS[job_id] = record


def _update_video_job(job_id: str, **updates: object) -> None:
    with _VIDEO_JOBS_LOCK:
        record = _VIDEO_JOBS[job_id]
        for key, value in updates.items():
            setattr(record, key, value)


def _identify_frame_with_pipeline(
    frame: np.ndarray,
    *,
    frame_id: str,
    ref_image: bytes,
) -> IdentificationResponse | None:
    """Run the full zebra identification pipeline on a single frame."""

    import cv2
    import torch
    from zebraid.pipelines.real_identify import prefilter_decision

    if os.getenv("IDENTIFY_MOCK", "") == "1":
        return _mock_identification_from_frame(frame)

    prefilter = prefilter_decision(frame)
    if not prefilter.passed:
        return None
    qual_val = prefilter.score

    _, engine, encoder, segmenter, flank_classifier, detector = get_pipeline()

    zebra_boxes = detector.detect_boxes(frame)
    if not zebra_boxes:
        return None
    zebra_box = zebra_boxes[0]

    frame_tensor = prepare_tensor(frame, segmenter=segmenter, box=zebra_box)

    with torch.no_grad():
        if hasattr(encoder, "encode_multiscale"):
            resnet_embedding = encoder.encode_multiscale(frame_tensor)
        else:
            resnet_embedding = encoder.encode(frame_tensor)

        engineered_feats = engineered_stripe_features(frame)
        engineered_tensor = torch.from_numpy(engineered_feats).unsqueeze(0).to(resnet_embedding.device)

        embedding = combine_features(resnet_embedding, [engineered_tensor], alpha=0.7)
        global_code = global_itq_code(resnet_embedding.squeeze().detach().cpu().numpy())
        zone_feats = engineered_feats[:96].reshape(3, 32)
        patch_codes = local_patch_codes(
            {
                "shoulder": zone_feats[0],
                "torso": zone_feats[1],
                "neck": zone_feats[2],
            }
        )

    embedding_np = embedding.squeeze().detach().cpu().numpy().astype(np.float32)
    flank = flank_classifier.classify(frame)

    zebra_id, confidence, is_new = engine.match_with_confidence(
        embedding_np,
        flank=flank,
        frame_id=frame_id,
        quality_score=qual_val,
        ref_image=ref_image,
        global_code=global_code,
        local_codes={
            "shoulder": patch_codes.shoulder,
            "torso": patch_codes.torso,
            "neck": patch_codes.neck,
        },
    )

    return IdentificationResponse(zebra_id=zebra_id, confidence=confidence, is_new=is_new)


def _process_video_job(job_id: str, tmp_path: str, filename: str) -> None:
    import cv2

    unique_ids: dict[str, IdentificationResponse] = {}
    frame_index = 0
    sampled_count = 0
    temp_path = Path(tmp_path)

    try:
        _update_video_job(job_id, status="processing")
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise RuntimeError("Could not open video file")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index % _VIDEO_SAMPLE_RATE != 0:
                frame_index += 1
                continue

            if sampled_count >= _VIDEO_MAX_SAMPLES:
                break

            sampled_count += 1

            encoded_ok, buffer = cv2.imencode(".jpg", frame)
            if not encoded_ok:
                frame_index += 1
                continue

            result = _identify_frame_with_pipeline(
                frame,
                frame_id=f"video_{filename}_{frame_index}",
                ref_image=buffer.tobytes(),
            )
            if result is not None:
                existing = unique_ids.get(result.zebra_id)
                if existing is None or result.confidence > existing.confidence:
                    unique_ids[result.zebra_id] = result

            frame_index += 1

        cap.release()
        _update_video_job(
            job_id,
            status="completed",
            result=VideoIdentificationResponse(
                unique_zebras=list(unique_ids.values()),
                total_frames_processed=sampled_count,
            ),
        )
    except Exception as exc:  # pragma: no cover - defensive path
        LOGGER.exception("Video processing failed")
        _update_video_job(job_id, status="failed", error=str(exc))
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except Exception:
            LOGGER.warning("Failed to remove temporary video file: %s", tmp_path)


def get_pipeline():
    """Get or initialize the identification pipeline.
    
    Lazily initializes the pipeline on first request.
    """
    global _registry, _engine, _encoder, _segmenter, _flank_classifier, _detector
    
    if _registry is None:
        registry_path = os.getenv("REGISTRY_PATH", None)
        _registry = FaissStore(embedding_dim=1138, store_path=registry_path)
        _engine = MatchingEngine(registry=_registry, similarity_threshold=0.75)
        _encoder = FeatureEncoder()
        _segmenter = ZebraSegmenter(backend="sam")
        _flank_classifier = FlankClassifier()
        from zebraid.preprocessing.detector import ZebraDetector
        _detector = ZebraDetector()
    
    return _registry, _engine, _encoder, _segmenter, _flank_classifier, _detector


def create_app() -> FastAPI:
    """Build the Phase 0 FastAPI application."""

    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(
        title="ZEBRAID API",
        version="0.1.0",
        description="Phase 0 foundation for zebra detection and re-identification",
    )

    allowed_origins = os.getenv(
        "ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000"
    ).split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health", tags=["health"])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/", include_in_schema=False)
    def root() -> dict[str, str]:
        return {"message": "ZEBRAID API foundation is ready"}

    @app.post("/identify", tags=["identification"], response_model=IdentificationResponse)
    async def identify(
        image: Annotated[UploadFile, File(description="Image file (JPG, PNG)")],
    ) -> IdentificationResponse:
        """Identify zebra from image.
        
        Full pipeline: image → encode → match → return ZebraID
        """
        try:
            import cv2

            # Read image from upload
            contents = await image.read()
            nparr = np.frombuffer(contents, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                raise HTTPException(
                    status_code=400, detail="Could not decode image file"
                )
            # Basic decode & validation
            # Check file extension for accepted image types when available
            _, ext = os.path.splitext(image.filename or "")
            ext = ext.lower().lstrip(".")
            accepted_exts = {
                "jpg",
                "jpeg",
                "png",
                "tif",
                "tiff",
                "raw",
                "nef",
                "cr2",
                "arw",
            }
            if ext and ext not in accepted_exts:
                raise HTTPException(status_code=400, detail="unsupported_format")

            # Resolution check (>= 5MP)
            height, width = frame.shape[:2]
            if (width * height) < 5_000_000:
                raise HTTPException(status_code=422, detail="low_resolution")

            # Aspect-ratio sanity check (avoid extremely tall/flat images)
            aspect = float(width) / float(height) if height > 0 else 0.0
            if aspect <= 0.0 or aspect < 0.5 or aspect > 2.0:
                raise HTTPException(status_code=422, detail="bad_aspect_ratio")

            # Ensure color image (convert grayscale to BGR)
            if frame.ndim == 2 or (frame.ndim == 3 and frame.shape[2] == 1):
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            
            result = _identify_frame_with_pipeline(
                frame,
                frame_id="api_upload",
                ref_image=contents,
            )
            if result is None:
                raise HTTPException(status_code=422, detail="low_quality_or_no_zebra")

            return result

        except HTTPException:
            raise
        except Exception as e:
            LOGGER.exception("Identification failed")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post(
        "/process-video",
        tags=["identification"],
        response_model=VideoJobAcceptedResponse,
        status_code=202,
    )
    async def process_video(
        video: Annotated[UploadFile, File(description="Video file (MP4, AVI, MOV)")],
    ) -> VideoJobAcceptedResponse:
        """Queue a video file for zebra identification and return a job ID."""
        import cv2

        # Save video to temp file
        suffix = os.path.splitext(video.filename or "")[1].lower()
        accepted_exts = {".mp4", ".mov", ".avi"}
        if suffix and suffix not in accepted_exts:
            raise HTTPException(status_code=400, detail="unsupported_format")

        job_id = uuid4().hex

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix or ".mp4") as tmp:
            tmp.write(await video.read())
            tmp_path = tmp.name

        _set_video_job(job_id, _VideoJobRecord(status="queued"))

        worker = threading.Thread(
            target=_process_video_job,
            args=(job_id, tmp_path, video.filename or "upload"),
            daemon=True,
        )
        worker.start()

        return VideoJobAcceptedResponse(job_id=job_id, status="queued")

    @app.get(
        "/process-video/{job_id}",
        tags=["identification"],
        response_model=VideoJobStatusResponse,
    )
    def get_process_video_job(job_id: str) -> VideoJobStatusResponse:
        """Get the current status for a queued video job."""

        record = _get_video_job(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="job_not_found")

        return VideoJobStatusResponse(
            job_id=job_id,
            status=record.status,
            unique_zebras=(record.result.unique_zebras if record.result else None),
            total_frames_processed=(record.result.total_frames_processed if record.result else None),
            error=record.error,
        )

    return app


app = create_app()


def main() -> None:
    """Run the API with uvicorn."""

    import uvicorn

    uvicorn.run("zebraid.api.app:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
