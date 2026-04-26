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
    gabor_features,
    combine_features
)
from zebraid.matching import MatchingEngine
from zebraid.registry import FaissStore
from zebraid.preprocessing import ZebraSegmenter, prepare_tensor

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


def get_pipeline():
    """Get or initialize the identification pipeline.
    
    Lazily initializes the pipeline on first request.
    """
    global _registry, _engine, _encoder, _segmenter, _flank_classifier, _detector
    
    if _registry is None:
        registry_path = os.getenv("REGISTRY_PATH", None)
        _registry = FaissStore(embedding_dim=160, store_path=registry_path)
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
        
        Args:
            image: Image file (JPG or PNG)
        
        Returns:
            IdentificationResponse with zebra_id, confidence, and is_new flag
        
        Raises:
            HTTPException: If image processing fails
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
            
            from zebraid.pipelines.real_identify import quality_score
            is_good, qual_val = quality_score(frame)
            if not is_good:
                raise HTTPException(status_code=422, detail="low_quality")

            # Get pipeline components
            registry, engine, encoder, segmenter, flank_classifier, detector = get_pipeline()

            # Detect zebras
            if not detector.detect(frame):
                raise HTTPException(status_code=400, detail="no_zebra")

            # Segment, clean, and convert to tensor before encoding.
            frame_tensor = prepare_tensor(frame, segmenter=segmenter)

            # Extract embedding
            with torch.no_grad():
                resnet_embedding = encoder.encode(frame_tensor)
                
                # Extract Gabor features
                g_feats = gabor_features(frame)
                gabor_tensor = torch.from_numpy(g_feats).unsqueeze(0).to(resnet_embedding.device)
                
                # Combine ResNet and Gabor features with alpha weighting
                embedding = combine_features(resnet_embedding, [gabor_tensor], alpha=0.7)

            # Convert to numpy for matching
            embedding_np = embedding.squeeze().detach().cpu().numpy().astype(np.float32)

            # Classify flank
            flank = flank_classifier.classify(frame)

            # Match against registry
            # Returns: (zebra_id, cosine_similarity, is_new)
            zebra_id, confidence, is_new = engine.match_with_confidence(
                embedding_np, flank=flank, frame_id="api_upload", quality_score=qual_val, ref_image=contents
            )

            return IdentificationResponse(
                zebra_id=zebra_id, confidence=confidence, is_new=is_new
            )

        except HTTPException:
            raise
        except Exception as e:
            LOGGER.exception("Identification failed")
            raise HTTPException(status_code=500, detail=str(e)) from e

    return app


app = create_app()


def main() -> None:
    """Run the API with uvicorn."""

    import uvicorn

    uvicorn.run("zebraid.api.app:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
