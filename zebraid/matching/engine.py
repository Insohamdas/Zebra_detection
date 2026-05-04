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
        similarity_threshold: float | None = None,
        distance_threshold: float | None = None,
        drift_hamming_threshold: float = 0.35,
        review_similarity_threshold: float | None = None,
        min_enroll_quality: float = 0.0,
    ):
        """Initialize matching engine with a registry store.
        
        Args:
            registry: FaissStore instance containing known embeddings and IDs
            similarity_threshold: Cosine similarity threshold (0-1, default: 0.75)
        """
        self.registry = registry
        if similarity_threshold is None and distance_threshold is None:
            similarity_threshold = 0.75
        if similarity_threshold is None:
            similarity_threshold = float(distance_threshold)
        self.similarity_threshold = float(similarity_threshold)
        self.drift_hamming_threshold = drift_hamming_threshold
        if review_similarity_threshold is None:
            review_similarity_threshold = max(0.0, self.similarity_threshold - 0.08)
        self.review_similarity_threshold = float(review_similarity_threshold)
        self.min_enroll_quality = float(min_enroll_quality)

    @property
    def distance_threshold(self) -> float:
        """Backward-compatible alias for similarity_threshold."""

        return self.similarity_threshold

    @distance_threshold.setter
    def distance_threshold(self, value: float) -> None:
        self.similarity_threshold = float(value)
    
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
        stripe_stats: np.ndarray | None = None,
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
            stripe_stats=stripe_stats,
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
        stripe_stats: np.ndarray | None = None,
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
                stripe_stats=stripe_stats,
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
            elif cosine_sim >= self.review_similarity_threshold:
                # Borderline similarity: keep existing ID and require downstream review,
                # avoid creating noisy new identities.
                is_new = False
                drift_flag = False
            elif quality_score is not None and quality_score < self.min_enroll_quality:
                # Unmatched but low-quality crop: avoid auto-enrollment.
                is_new = False
                drift_flag = False
            else:
                zebra_id = self._create_new_id(
                    embedding,
                    flank=flank,
                    ref_image=ref_image,
                    stripe_stats=stripe_stats,
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

    def resolve_three_phase_identity(
        self,
        embedding: np.ndarray,
        *,
        global_code: np.ndarray,
        local_codes: dict[str, np.ndarray] | None = None,
        flank: str = "left",
        stripe_stats: np.ndarray | None = None,
        coarse_k: int = 20,
        hamming_k: int = 5,
        borderline_hamming: float = 0.20,
        frame_id: str | None = None,
        quality_score: float | None = None,
        ref_image: bytes | None = None,
        ssi_profile: np.ndarray | None = None,
    ) -> tuple[str, float, bool, str]:
        """Resolve an identity using coarse FAISS -> Hamming -> optional local refinement.

        Returns ``(zebra_id, confidence, is_new, phase)`` where phase is one of
        ``enroll``, ``hamming``, or ``local_refine``.
        """

        if self.registry.indices[flank].ntotal == 0:
            zebra_id = self._create_new_id(
                embedding,
                flank=flank,
                ref_image=ref_image,
                stripe_stats=stripe_stats,
                global_code=global_code,
                local_codes=local_codes,
                ssi_profile=ssi_profile,
            )
            return zebra_id, 1.0, True, "enroll"

        candidates = self.registry.search_candidates(embedding, flank=flank, k=coarse_k)
        if not candidates:
            zebra_id = self._create_new_id(
                embedding,
                flank=flank,
                ref_image=ref_image,
                stripe_stats=stripe_stats,
                global_code=global_code,
                local_codes=local_codes,
                ssi_profile=ssi_profile,
            )
            return zebra_id, 1.0, True, "enroll"

        candidate_ids = [zebra_id for zebra_id, _ in candidates]
        hamming_matches = self.registry.hamming_search(
            global_code,
            flank=flank,
            candidate_ids=candidate_ids,
            k=hamming_k,
        )

        if not hamming_matches:
            zebra_id = self._create_new_id(
                embedding,
                flank=flank,
                ref_image=ref_image,
                stripe_stats=stripe_stats,
                global_code=global_code,
                local_codes=local_codes,
                ssi_profile=ssi_profile,
            )
            return zebra_id, 1.0, True, "enroll"

        best_id, best_hamming = hamming_matches[0]
        phase = "hamming"

        if local_codes and best_hamming <= borderline_hamming:
            refined = self.registry.local_refine(
                local_codes,
                [zebra_id for zebra_id, _ in hamming_matches],
                flank=flank,
            )
            if refined:
                refined_id, refined_distance = refined[0]
                if refined_distance <= borderline_hamming:
                    best_id = refined_id
                    best_hamming = refined_distance
                    phase = "local_refine"

        if best_hamming <= borderline_hamming:
            self.registry.update_embedding(
                best_id,
                embedding,
                flank=flank,
                alpha=0.1,
                global_code=global_code,
            )
            confidence = float(np.clip(1.0 - best_hamming, 0.0, 1.0))
            return best_id, confidence, False, phase

        zebra_id = self._create_new_id(
            embedding,
            flank=flank,
            ref_image=ref_image,
            stripe_stats=stripe_stats,
            global_code=global_code,
            local_codes=local_codes,
            ssi_profile=ssi_profile,
        )
        return zebra_id, 1.0, True, "enroll"

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
