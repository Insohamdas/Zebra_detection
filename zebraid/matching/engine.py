"""Matching engine for zebra identity resolution using FAISS registry."""
from __future__ import annotations

import json
import logging
import time

import numpy as np

from zebraid.registry import FaissStore

log = logging.getLogger("zebraid.match")


class MatchingEngine:
    """Matches embeddings against a FAISS registry to resolve zebra identities.
    
    Uses cosine similarity threshold for deciding between existing matches
    and new IDs. With L2-normalized vectors and IndexFlatIP, the FAISS
    distance IS the cosine similarity (inner product).
    """
    
    def __init__(
        self,
        registry: FaissStore,
        similarity_threshold: float = 0.75,
        drift_hamming_threshold: float = 0.35,
    ):
        """Initialize matching engine with a registry store.
        
        Args:
            registry: FaissStore instance containing known embeddings and IDs
            similarity_threshold: Cosine similarity threshold (0-1, default: 0.75)
        """
        self.registry = registry
        self.similarity_threshold = similarity_threshold
        self.drift_hamming_threshold = drift_hamming_threshold
    
    def match(self, embedding: np.ndarray, flank: str = "left") -> str:
        """Attempt to match an embedding to an existing zebra ID.
        
        Args:
            embedding: 1D numpy array of embedding values
            flank: 'left' or 'right' side of the zebra
        
        Returns:
            Existing zebra_id if similarity > threshold, otherwise newly generated ID
        """
        # If the specific flank index is empty, create new ID
        try:
            if self.registry.indices[flank].ntotal == 0:
                return self._create_new_id(embedding, flank=flank)
        except KeyError:
            raise ValueError(f"Invalid flank '{flank}'")
        
        # Search registry for nearest neighbor within the specified flank
        # With IndexFlatIP + L2-normalized vectors, distance = cosine similarity
        zebra_id, cosine_sim = self.registry.search(embedding, flank=flank)
        
        # If similarity > threshold, return existing ID
        if cosine_sim > self.similarity_threshold:
            self.registry.update_embedding(zebra_id, embedding, flank=flank, alpha=0.1)
            return zebra_id
        else:
            # Create new ID for this flank
            return self._create_new_id(embedding, flank=flank)
    
    def _create_new_id(
        self, 
        embedding: np.ndarray, 
        flank: str = "left",
        ref_image: bytes | None = None,
        global_code: np.ndarray | None = None,
        local_codes: dict[str, np.ndarray] | None = None,
        ssi_profile: np.ndarray | None = None,
    ) -> str:
        """Create a new zebra ID by adding to registry with flank label.
        
        Args:
            embedding: 1D numpy array of embedding values
            flank: 'left' or 'right'
            ref_image: Reference image bytes for future re-encoding
        
        Returns:
            Registry-assigned zebra ID
        """
        # Let registry assign the ID when adding embedding
        new_id = self.registry.add_and_get_id(
            embedding,
            flank=flank,
            ref_image=ref_image,
            global_code=global_code,
            local_codes=local_codes,
            ssi_profile=ssi_profile,
        )
        
        return new_id
    
    def match_with_confidence(
        self, 
        embedding: np.ndarray, 
        flank: str = "left",
        frame_id: str | None = None,
        quality_score: float | None = None,
        ref_image: bytes | None = None,
        global_code: np.ndarray | None = None,
        local_codes: dict[str, np.ndarray] | None = None,
        ssi_profile: np.ndarray | None = None,
    ) -> tuple[str, float, bool]:
        """Match embedding and return ID, confidence, and match status.
        
        With IndexFlatIP on L2-normalized vectors, the FAISS distance IS
        the cosine similarity directly (range -1 to 1, typically 0 to 1
        for normalized non-negative embeddings).
        
        Confidence is mapped from cosine_sim ∈ [-1, 1] to [0, 1].
        
        Args:
            embedding: 1D numpy array of embedding values (should be L2-normalized)
            flank: 'left' or 'right' side
            frame_id: Optional string identifying the origin frame/scene
            quality_score: Optional blur variance or quality metric for logging
            ref_image: Binary content of original image for enrollment
            
        Returns:
            Tuple of (zebra_id, confidence, is_new)
        """
        # If flank index is empty, create new ID
        if self.registry.indices[flank].ntotal == 0:
            zebra_id = self._create_new_id(
                embedding,
                flank=flank,
                ref_image=ref_image,
                global_code=global_code,
                local_codes=local_codes,
                ssi_profile=ssi_profile,
            )
            confidence = 1.0
            is_new = True
            cosine_sim = 1.0
            drift_flag = False
        else:
            # With IndexFlatIP, distance = cosine similarity (higher = more similar)
            zebra_id, cosine_sim = self.registry.search(embedding, flank=flank)
            
            # Map cosine similarity [-1, 1] -> confidence [0, 1]
            confidence = float(np.clip((cosine_sim + 1.0) / 2.0, 0.0, 1.0))
            
            if cosine_sim > self.similarity_threshold:
                drift_flag = self.registry.update_embedding(
                    zebra_id,
                    embedding,
                    flank=flank,
                    alpha=0.1,
                    global_code=global_code,
                    drift_threshold=self.drift_hamming_threshold,
                )
                is_new = False
            else:
                zebra_id = self._create_new_id(
                    embedding,
                    flank=flank,
                    ref_image=ref_image,
                    global_code=global_code,
                    local_codes=local_codes,
                    ssi_profile=ssi_profile,
                )
                is_new = True
                drift_flag = False
                
        decision = {
            "ts": time.time(),
            "frame_id": frame_id,
            "zebra_id": zebra_id,
            "cosine_sim": round(float(cosine_sim), 4),
            "confidence": round(float(confidence), 4),
            "is_new": is_new,
            "quality": round(float(quality_score), 3) if quality_score is not None else None,
            "flank": flank,
            "drift_flag": drift_flag,
        }
        log.info(json.dumps(decision))

        return zebra_id, confidence, is_new

    def match_three_phase(
        self,
        embedding: np.ndarray,
        *,
        global_code: np.ndarray,
        local_codes: dict[str, np.ndarray] | None = None,
        flank: str = "left",
        coarse_k: int = 20,
        hamming_k: int = 5,
        borderline_hamming: float = 0.20,
    ) -> tuple[str | None, float, str]:
        """Run coarse FAISS filter, Hamming search, and local patch refinement."""

        candidates = self.registry.search_candidates(embedding, flank=flank, k=coarse_k)
        if not candidates:
            return None, 1.0, "empty"

        candidate_ids = [zebra_id for zebra_id, _ in candidates]
        hamming_matches = self.registry.hamming_search(
            global_code,
            flank=flank,
            candidate_ids=candidate_ids,
            k=hamming_k,
        )
        if not hamming_matches:
            return candidate_ids[0], 1.0 - candidates[0][1], "coarse"

        best_id, best_hamming = hamming_matches[0]
        if local_codes and best_hamming <= borderline_hamming:
            refined = self.registry.local_refine(
                local_codes,
                [zebra_id for zebra_id, _ in hamming_matches],
                flank=flank,
            )
            if refined:
                refined_id, refined_distance = refined[0]
                return refined_id, 1.0 - refined_distance, "local_refine"

        return best_id, 1.0 - best_hamming, "hamming"
    
    def add_zebra(self, embedding: np.ndarray, zebra_id: str, flank: str = "left") -> None:
        """Add a new zebra embedding to the registry.
        
        Args:
            embedding: 1D numpy array of embedding values
            zebra_id: Unique identifier for this zebra
            flank: 'left' or 'right'
        """
        self.registry.add(embedding, zebra_id, flank=flank)
