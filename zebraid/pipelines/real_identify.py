"""Real zebra identification pipeline using encoding and matching."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np
import torch

from zebraid.feature_engine import (
    FeatureEncoder, 
    FlankClassifier,
    engineered_stripe_features,
    combine_features
)
from zebraid.matching import MatchingEngine
from zebraid.pipelines.live_identification import IdentificationCandidate
from zebraid.registry import FaissStore
from zebraid.preprocessing import FramePrefilter, FramePrefilterDecision, ZebraSegmenter, prepare_tensor
from zebraid.id_generator import global_itq_code, local_patch_codes


@dataclass(frozen=True)
class QualityRejected:
    """Explicitly marks a rejection due to image quality issues."""
    reason: str


def quality_score(crop: np.ndarray) -> tuple[bool, float]:
    """Assess whether a crop meets minimal acceptable quality for embeddings.

    The score is now the pre-filter score in the [0, 1] range. It includes
    blur and overexposure checks before the expensive identification stages.
    """
    decision = prefilter_decision(crop)
    return decision.passed, decision.score


def prefilter_decision(crop: np.ndarray) -> FramePrefilterDecision:
    """Return detailed pre-filter decision for blur/overexposure rejection."""

    return FramePrefilter().evaluate(crop)


LOGGER = logging.getLogger(__name__)


def create_real_identifier(
    registry: FaissStore | None = None,
    encoder: FeatureEncoder | None = None,
    segmenter: ZebraSegmenter | None = None,
    flank_classifier: FlankClassifier | None = None,
    detector: object | None = None,
    segment_input: bool = True,
    match_threshold: float = 0.75,
) -> callable:
    """Factory for creating a real zebra identifier using matching engine.
    
    Step 6.2 — Integrate in LIVE PIPELINE
    
    Creates a function that implements the full pipeline:
    frame → detect zebra → process_image → encode → match
    
    Args:
        registry: FAISS registry for storing/searching embeddings
        encoder: FeatureEncoder for extracting embeddings from frames
        flank_classifier: Optional classifier for auto-detecting side
        match_threshold: Cosine similarity threshold for identity matching
    
    Returns:
        Function that takes a frame and returns IdentificationCandidate
    """
    # Initialize defaults if not provided
    if registry is None:
        registry = FaissStore(embedding_dim=1138)
    
    if encoder is None:
        encoder = FeatureEncoder()

    if segmenter is None:
        segmenter = ZebraSegmenter(backend="sam")
    
    if flank_classifier is None:
        flank_classifier = FlankClassifier()
    
    matching_engine = MatchingEngine(registry, similarity_threshold=match_threshold)
    
    def identify_frame(frame: np.ndarray) -> IdentificationCandidate | QualityRejected | None:
        """Process frame through full identification pipeline.
        
        Pipeline flow:
        1. Validate frame
        2. Normalize/process image
        3. Extract embedding using ResNet-50 encoder
        4. Match against registry using FAISS
        5. Return candidate with zebra_id and confidence
        
        Args:
            frame: Input frame (BGR image or array)
        
        Returns:
            IdentificationCandidate with matched/new zebra_id and confidence
        """
        if frame is None or frame.size == 0:
            return None
        
        prefilter = prefilter_decision(frame)
        if not prefilter.passed:
            return QualityRejected(reason=",".join(prefilter.reasons) or "low_quality")
        qual_val = prefilter.score
        
        try:
            # Step 1: Segment and clean the crop before encoding when requested.
            if segment_input:
                zebra_box = None
                if detector is not None and hasattr(detector, "best_box"):
                    zebra_box = detector.best_box(frame)
                    if zebra_box is None:
                        return None

                frame_tensor = prepare_tensor(frame, segmenter=segmenter, box=zebra_box)
            else:
                if frame.ndim != 3 or frame.shape[2] != 3:
                    return None

                if frame.dtype != np.uint8:
                    frame = np.clip(frame, 0, 255).astype(np.uint8)

                frame_tensor = (
                    torch.from_numpy(frame)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .float()
                    / 255.0
                )

            # Step 2: Extract embedding using encoder
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
            
            # Classify flank (left vs right)
            flank = flank_classifier.classify(frame)
            
            # Serialize frame to JPG bytes for reference image storage
            _, buffer = cv2.imencode('.jpg', frame)
            ref_image_bytes = buffer.tobytes()
            
            # Step 4: Match embedding against registry (flank-specific)
            zebra_id, confidence, is_new = matching_engine.match_with_confidence(
                embedding_np,
                flank=flank,
                frame_id="live_crop",
                quality_score=qual_val,
                ref_image=ref_image_bytes,
                global_code=global_code,
                local_codes={
                    "shoulder": patch_codes.shoulder,
                    "torso": patch_codes.torso,
                    "neck": patch_codes.neck,
                },
            )
            
            return IdentificationCandidate(
                zebra_id=zebra_id,
                confidence=confidence,  # Already cosine similarity [0, 1]
            )
        
        except Exception as e:
            LOGGER.warning(f"Identification failed: {e}")
            return None
    
    # Attach matching engine and registry to function for external access
    identify_frame.matching_engine = matching_engine  # type: ignore
    identify_frame.registry = registry  # type: ignore
    identify_frame.encoder = encoder  # type: ignore
    identify_frame.segmenter = segmenter  # type: ignore
    identify_frame.detector = detector  # type: ignore
    identify_frame.segment_input = segment_input  # type: ignore
    
    return identify_frame
