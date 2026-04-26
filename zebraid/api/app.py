"""FastAPI foundation for the ZEBRAID API."""

from __future__ import annotations

import io
import os
import logging

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
            import torch
            import cv2

            # Read image from upload
            contents = await image.read()
            nparr = np.frombuffer(contents, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                raise HTTPException(
                    status_code=400, detail="Could not decode image file"
                )
            
            from zebraid.pipelines.real_identify import prefilter_decision
            prefilter = prefilter_decision(frame)
            if not prefilter.passed:
                raise HTTPException(status_code=422, detail="low_quality")
            qual_val = prefilter.score

            # Get pipeline components
            registry, engine, encoder, segmenter, flank_classifier, detector = get_pipeline()

            # Detect zebra and pass the best box as a SAM prompt for body masking.
            zebra_boxes = detector.detect_boxes(frame)
            if not zebra_boxes:
                raise HTTPException(status_code=400, detail="no_zebra")
            zebra_box = zebra_boxes[0]

            # Segment, clean, and convert to tensor before encoding.
            frame_tensor = prepare_tensor(frame, segmenter=segmenter, box=zebra_box)

            # Extract embedding
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
                patch_codes = local_patch_codes({
                    "shoulder": zone_feats[0],
                    "torso": zone_feats[1],
                    "neck": zone_feats[2],
                })

            # Convert to numpy for matching
            embedding_np = embedding.squeeze().detach().cpu().numpy().astype(np.float32)

            # Classify flank
            flank = flank_classifier.classify(frame)

            # Match against registry
            zebra_id, confidence, is_new = engine.match_with_confidence(
                embedding_np,
                flank=flank,
                frame_id="api_upload",
                quality_score=qual_val,
                ref_image=contents,
                global_code=global_code,
                local_codes={
                    "shoulder": patch_codes.shoulder,
                    "torso": patch_codes.torso,
                    "neck": patch_codes.neck,
                },
            )

            return IdentificationResponse(
                zebra_id=zebra_id, confidence=confidence, is_new=is_new
            )

        except HTTPException:
            raise
        except Exception as e:
            LOGGER.exception("Identification failed")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post("/process-video", tags=["identification"], response_model=VideoIdentificationResponse)
    async def process_video(
        video: Annotated[UploadFile, File(description="Video file (MP4, AVI, MOV)")],
    ) -> VideoIdentificationResponse:
        """Identify all unique zebras in a video file."""
        import tempfile
        import cv2
        import torch
        from zebraid.pipelines.real_identify import prefilter_decision

        # Save video to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video.filename)[1]) as tmp:
            tmp.write(await video.read())
            tmp_path = tmp.name

        try:
            cap = cv2.VideoCapture(tmp_path)
            if not cap.isOpened():
                raise HTTPException(status_code=400, detail="Could not open video file")

            registry, engine, encoder, segmenter, flank_classifier, detector = get_pipeline()
            
            unique_ids = {}  # zebra_id -> IdentificationResponse
            frame_count = 0
            processed_count = 0
            
            # Sample every 15 frames (approx 0.5s for 30fps video)
            sample_rate = 15
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % sample_rate == 0:
                    processed_count += 1
                    
                    # 1. Detection
                    zebra_boxes = detector.detect_boxes(frame)
                    if not zebra_boxes:
                        frame_count += 1
                        continue
                    zebra_box = zebra_boxes[0]
                        
                    # 2. Quality
                    prefilter = prefilter_decision(frame)
                    if not prefilter.passed:
                        frame_count += 1
                        continue
                    qual_val = prefilter.score
                        
                    # 3. Processing
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
                        patch_codes = local_patch_codes({
                            "shoulder": zone_feats[0],
                            "torso": zone_feats[1],
                            "neck": zone_feats[2],
                        })
                        
                    embedding_np = embedding.squeeze().detach().cpu().numpy().astype(np.float32)
                    flank = flank_classifier.classify(frame)
                    
                    # Convert frame to bytes for storage if new
                    _, buffer = cv2.imencode('.jpg', frame)
                    frame_bytes = buffer.tobytes()
                    
                    # Match
                    zebra_id, confidence, is_new = engine.match_with_confidence(
                        embedding_np, flank=flank, frame_id=f"video_{video.filename}_{frame_count}", 
                        quality_score=qual_val,
                        ref_image=frame_bytes,
                        global_code=global_code,
                        local_codes={
                            "shoulder": patch_codes.shoulder,
                            "torso": patch_codes.torso,
                            "neck": patch_codes.neck,
                        },
                    )
                    
                    # Store unique ID results (keep the highest confidence one)
                    if zebra_id not in unique_ids or confidence > unique_ids[zebra_id].confidence:
                        unique_ids[zebra_id] = IdentificationResponse(
                            zebra_id=zebra_id, confidence=confidence, is_new=is_new
                        )
                
                frame_count += 1
                
                # Limit processing to prevent timeouts (e.g., max 100 sampled frames)
                if processed_count > 100:
                    break

            cap.release()
            return VideoIdentificationResponse(
                unique_zebras=list(unique_ids.values()),
                total_frames_processed=processed_count
            )

        except Exception as e:
            LOGGER.exception("Video processing failed")
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    return app


app = create_app()


def main() -> None:
    """Run the API with uvicorn."""

    import uvicorn

    uvicorn.run("zebraid.api.app:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
