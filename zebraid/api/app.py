"""FastAPI foundation for the ZEBRAID API."""

from __future__ import annotations

import logging
import os
import threading
import tempfile
import base64
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from typing import Annotated

import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from zebraid.feature_engine import (
    FeatureEncoder, 
    FlankClassifier,
    engineered_stripe_features,
    combine_features
)
from zebraid.matching import MatchingEngine
from zebraid.registry import FaissStore
from zebraid.registry.faiss_store import hamming_distance
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


class VideoIdentificationItem(IdentificationResponse):
    """Per-zebra output item for video processing."""

    thumbnail_jpeg_base64: str | None = None
    flagged_for_review: bool = False


class VideoIdentificationResponse(BaseModel):
    """Response from video identification endpoint."""

    unique_zebras: list[VideoIdentificationItem]
    total_frames_processed: int


class VideoJobAcceptedResponse(BaseModel):
    """Response returned when a video job has been queued."""

    job_id: str
    status: str


class VideoJobStatusResponse(BaseModel):
    """Current status for an asynchronous video identification job."""

    job_id: str
    status: str
    sampled_frames: int = 0
    estimated_total_samples: int = 0
    progress: float = 0.0
    unique_zebras: list[VideoIdentificationItem] | None = None
    total_frames_processed: int | None = None
    error: str | None = None


@dataclass
class _VideoJobRecord:
    status: str
    sampled_frames: int = 0
    estimated_total_samples: int = 0
    progress: float = 0.0
    result: VideoIdentificationResponse | None = None
    error: str | None = None


@dataclass
class _FrameIdentification:
    """Internal payload for a single zebra detection in a frame."""

    result: VideoIdentificationItem
    global_code: np.ndarray | None
    quality_score: float
    thumbnail_bytes: bytes | None


_VIDEO_JOBS: dict[str, _VideoJobRecord] = {}
_VIDEO_JOBS_LOCK = threading.Lock()
_VIDEO_SAMPLE_RATE = int(os.getenv("VIDEO_SAMPLE_RATE", "15"))
_VIDEO_MAX_SAMPLES = int(os.getenv("VIDEO_MAX_SAMPLES", "100"))
_REVIEW_CONFIDENCE_MIN = float(os.getenv("REVIEW_CONFIDENCE_MIN", "0.82"))
_AUTO_ENROLL_MIN_QUALITY = float(os.getenv("AUTO_ENROLL_MIN_QUALITY", "0.70"))


def _estimate_total_samples(frame_count: int, sample_rate: int, max_samples: int) -> int:
    """Estimate how many frames will be sampled from a video."""

    if frame_count <= 0:
        return max_samples
    return min(max_samples, (frame_count + sample_rate - 1) // sample_rate)


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


def _identify_zebras_in_frame(
    frame: np.ndarray,
    *,
    frame_id: str,
    ref_image: bytes,
) -> list[_FrameIdentification]:
    """Run the full zebra identification pipeline on all detected zebras in a frame."""

    import cv2
    import torch
    from zebraid.pipelines.real_identify import prefilter_decision

    if os.getenv("IDENTIFY_MOCK", "") == "1":
        mock_result = _mock_identification_from_frame(frame)
        return [
            _FrameIdentification(
                result=VideoIdentificationItem(
                    zebra_id=mock_result.zebra_id,
                    confidence=mock_result.confidence,
                    is_new=mock_result.is_new,
                ),
                global_code=None,
                quality_score=1.0,
                thumbnail_bytes=ref_image,
            )
        ]

    _, engine, encoder, segmenter, flank_classifier, detector = get_pipeline()
    zebra_boxes = detector.detect_boxes(frame, conf_threshold=0.3)
    if not zebra_boxes:
        return []

    frame_results: list[_FrameIdentification] = []
    height, width = frame.shape[:2]
    for det_idx, zebra_box in enumerate(zebra_boxes):
        x1, y1, x2, y2 = [int(v) for v in zebra_box]
        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height))
        if x2 <= x1 or y2 <= y1:
            continue
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        prefilter = prefilter_decision(crop)
        if not prefilter.passed:
            continue
        qual_val = prefilter.score

        crop_ok, crop_buffer = cv2.imencode(".jpg", crop)
        crop_bytes = crop_buffer.tobytes() if crop_ok else ref_image
        frame_tensor = prepare_tensor(frame, segmenter=segmenter, box=zebra_box)

        with torch.no_grad():
            if hasattr(encoder, "encode_multiscale"):
                resnet_embedding = encoder.encode_multiscale(frame_tensor)
            else:
                resnet_embedding = encoder.encode(frame_tensor)

            engineered_feats = engineered_stripe_features(crop)
            engineered_tensor = (
                torch.from_numpy(engineered_feats).unsqueeze(0).to(resnet_embedding.device)
            )

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
        flank = flank_classifier.classify(crop)

        zebra_id, confidence, is_new = engine.match_with_confidence(
            embedding_np,
            flank=flank,
            frame_id=f"{frame_id}_det{det_idx}",
            quality_score=qual_val,
            ref_image=crop_bytes,
            global_code=global_code,
            local_codes={
                "shoulder": patch_codes.shoulder,
                "torso": patch_codes.torso,
                "neck": patch_codes.neck,
            },
        )
        frame_results.append(
            _FrameIdentification(
                result=VideoIdentificationItem(zebra_id=zebra_id, confidence=confidence, is_new=is_new),
                global_code=np.asarray(global_code, dtype=np.uint8).ravel(),
                quality_score=qual_val,
                thumbnail_bytes=crop_bytes,
            )
        )

    return frame_results


def _identify_frame_with_pipeline(
    frame: np.ndarray,
    *,
    frame_id: str,
    ref_image: bytes,
) -> IdentificationResponse | None:
    """Compatibility wrapper for single-image endpoint."""

    frame_results = _identify_zebras_in_frame(frame, frame_id=frame_id, ref_image=ref_image)
    if not frame_results:
        return None
    best = frame_results[0].result
    return IdentificationResponse(
        zebra_id=best.zebra_id,
        confidence=best.confidence,
        is_new=best.is_new,
    )


def _process_video_job(job_id: str, tmp_path: str, filename: str) -> None:
    import cv2

    unique_ids: dict[str, _FrameIdentification] = {}
    frame_index = 0
    sampled_count = 0
    temp_path = Path(tmp_path)
    cap = None
    drift_threshold = 0.35

    try:
        _update_video_job(job_id, status="processing")
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise RuntimeError("Could not open video file")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        estimated_total_samples = _estimate_total_samples(
            frame_count,
            _VIDEO_SAMPLE_RATE,
            _VIDEO_MAX_SAMPLES,
        )
        _update_video_job(
            job_id,
            estimated_total_samples=estimated_total_samples,
            progress=0.0,
        )

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
            progress = (
                min(sampled_count / estimated_total_samples, 1.0)
                if estimated_total_samples
                else 0.0
            )
            _update_video_job(
                job_id,
                sampled_frames=sampled_count,
                progress=progress,
            )

            encoded_ok, buffer = cv2.imencode(".jpg", frame)
            if not encoded_ok:
                frame_index += 1
                continue

            detections = _identify_zebras_in_frame(
                frame,
                frame_id=f"video_{filename}_{frame_index}",
                ref_image=buffer.tobytes(),
            )
            for detection in detections:
                zebra_id = detection.result.zebra_id
                existing = unique_ids.get(zebra_id)
                if (
                    detection.result.confidence < _REVIEW_CONFIDENCE_MIN
                    or detection.quality_score < _AUTO_ENROLL_MIN_QUALITY
                ):
                    detection.result.flagged_for_review = True
                if (
                    existing is not None
                    and existing.global_code is not None
                    and detection.global_code is not None
                    and hamming_distance(existing.global_code, detection.global_code) > drift_threshold
                ):
                    existing.result.flagged_for_review = True
                    detection.result.flagged_for_review = True

                should_replace = (
                    existing is None
                    or detection.result.confidence > existing.result.confidence
                    or (
                        detection.result.confidence == existing.result.confidence
                        and detection.quality_score > existing.quality_score
                    )
                )
                if should_replace:
                    if existing is not None and existing.result.flagged_for_review:
                        detection.result.flagged_for_review = True
                    unique_ids[zebra_id] = detection

            frame_index += 1

        unique_results: list[IdentificationResponse] = []
        for aggregate in unique_ids.values():
            if aggregate.thumbnail_bytes:
                aggregate.result.thumbnail_jpeg_base64 = base64.b64encode(
                    aggregate.thumbnail_bytes
                ).decode("ascii")
            unique_results.append(aggregate.result)
        unique_results.sort(key=lambda item: item.confidence, reverse=True)

        _update_video_job(
            job_id,
            status="completed",
            sampled_frames=sampled_count,
            progress=1.0,
            result=VideoIdentificationResponse(
                unique_zebras=unique_results,
                total_frames_processed=sampled_count,
            ),
        )
    except Exception as exc:  # pragma: no cover - defensive path
        LOGGER.exception("Video processing failed")
        _update_video_job(job_id, status="failed", error=str(exc))
    finally:
        if cap is not None:
            cap.release()
        try:
            temp_path.unlink(missing_ok=True)
        except Exception:
            LOGGER.warning("Failed to remove temporary video file: %s", tmp_path)


def _video_status_response(job_id: str) -> VideoJobStatusResponse:
    """Build an API response for a video job record."""

    record = _get_video_job(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="job_not_found")

    return VideoJobStatusResponse(
        job_id=job_id,
        status=record.status,
        sampled_frames=record.sampled_frames,
        estimated_total_samples=record.estimated_total_samples,
        progress=record.progress,
        unique_zebras=(record.result.unique_zebras if record.result else None),
        total_frames_processed=(record.result.total_frames_processed if record.result else None),
        error=record.error,
    )


def get_pipeline():
    """Get or initialize the identification pipeline.
    
    Lazily initializes the pipeline on first request.
    Ensures all components are properly initialized.
    """
    global _registry, _engine, _encoder, _segmenter, _flank_classifier, _detector
    
    if _registry is None or _engine is None or _encoder is None or _detector is None:
        # Initialize Registry & Engine
        if _registry is None:
            registry_path = os.getenv("REGISTRY_PATH", None)
            _registry = FaissStore(embedding_dim=1138, store_path=registry_path)
            
        if _engine is None:
            _engine = MatchingEngine(
                registry=_registry,
                similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.75")),
                review_similarity_threshold=float(os.getenv("REVIEW_SIMILARITY_THRESHOLD", "0.67")),
                min_enroll_quality=_AUTO_ENROLL_MIN_QUALITY,
            )
            
        # Initialize Encoder & Segmenter
        if _encoder is None:
            _encoder = FeatureEncoder()
        if _segmenter is None:
            _segmenter = ZebraSegmenter(backend="otsu")
        if _flank_classifier is None:
            _flank_classifier = FlankClassifier()
            
        # Initialize Detector with fallback
        if _detector is None:
            from zebraid.preprocessing.detector import ZebraDetector
            detector_model = os.getenv(
                "DETECTOR_MODEL_PATH",
                str(
                    Path(__file__).resolve().parents[2]
                    / "data"
                    / "runs"
                    / "zebra_combined_v1"
                    / "weights"
                    / "best.pt"
                ),
            )
            
            try:
                if not os.path.exists(detector_model):
                    LOGGER.warning(f"Custom detector model not found at {detector_model}. Falling back to yolov8n.pt")
                    detector_model = "yolov8n.pt"
                _detector = ZebraDetector(model_name=detector_model)
            except Exception as e:
                LOGGER.error(f"Failed to load detector {detector_model}: {e}. Trying absolute fallback.")
                _detector = ZebraDetector(model_name="yolov8n.pt")
    
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

            # Aspect-ratio sanity check (avoid extremely tall/flat images)
            # widened to allow more flexibility
            height, width = frame.shape[:2]
            aspect = float(width) / float(height) if height > 0 else 0.0
            if aspect <= 0.0 or aspect < 0.1 or aspect > 10.0:
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
        background_tasks: BackgroundTasks,
        video: Annotated[UploadFile, File(description="Video file (MP4, AVI, MOV)")],
    ) -> VideoJobAcceptedResponse:
        """Queue a video file for zebra identification and return a job ID."""

        # Save video to temp file
        suffix = os.path.splitext(video.filename or "")[1].lower()
        accepted_exts = {".mp4", ".mov", ".avi"}
        if suffix and suffix not in accepted_exts:
            raise HTTPException(status_code=400, detail="unsupported_format")

        job_id = uuid4().hex

        # Enforce maximum video size (e.g. 500 MB)
        video_bytes = await video.read()
        if len(video_bytes) > 500 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Video exceeds maximum allowed size (500MB).")

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix or ".mp4") as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        _set_video_job(job_id, _VideoJobRecord(status="queued"))
        background_tasks.add_task(
            _process_video_job,
            job_id,
            tmp_path,
            video.filename or "upload",
        )

        return VideoJobAcceptedResponse(job_id=job_id, status="queued")

    @app.get(
        "/video-status/{job_id}",
        tags=["identification"],
        response_model=VideoJobStatusResponse,
    )
    def get_video_status(job_id: str) -> VideoJobStatusResponse:
        """Get the current status for a queued video job."""

        return _video_status_response(job_id)

    @app.get(
        "/process-video/{job_id}",
        tags=["identification"],
        response_model=VideoJobStatusResponse,
    )
    def get_process_video_job(job_id: str) -> VideoJobStatusResponse:
        """Get the current status for a queued video job."""

        return _video_status_response(job_id)

    return app


app = create_app()


def main() -> None:
    """Run the API with uvicorn."""

    import uvicorn

    uvicorn.run("zebraid.api.app:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
