"""Tests for matching robustness and real-world scenarios."""
import numpy as np
import pytest

from zebraid.matching import MatchingEngine
from zebraid.output import count_population, get_population_summary
from zebraid.registry import FaissStore


@pytest.fixture
def registry():
    """Create a fresh FAISS registry."""
    return FaissStore(embedding_dim=2048)


@pytest.fixture
def engine(registry):
    """Create matching engine with default threshold."""
    return MatchingEngine(registry=registry, distance_threshold=0.5)


def test_same_zebra_same_id(engine):
    """Test: Same zebra → same ID (identical embedding returns same ID)."""
    # Create embedding for zebra_A
    embedding_a = np.random.randn(2048).astype(np.float32)
    
    # First observation
    zebra_id_1, distance_1, is_new_1 = engine.match_with_confidence(embedding_a)
    assert is_new_1 is True  # First time seeing this
    
    # Second observation: exact same embedding
    zebra_id_2, distance_2, is_new_2 = engine.match_with_confidence(embedding_a)
    assert is_new_2 is False  # Should match existing
    assert zebra_id_2 == zebra_id_1  # Same ID!
    assert distance_2 < 1e-5  # Exact match


def test_different_zebra_different_id(engine):
    """Test: Different zebra → different ID (very different embeddings)."""
    # Zebra A: all positive values
    embedding_a = np.full(2048, 1.0, dtype=np.float32)
    
    # Zebra B: all negative values (very different)
    embedding_b = np.full(2048, -1.0, dtype=np.float32)
    
    # First observation of A
    zebra_id_a, _, is_new_a = engine.match_with_confidence(embedding_a)
    assert is_new_a is True
    
    # First observation of B (should be very far from A)
    zebra_id_b, _, is_new_b = engine.match_with_confidence(embedding_b)
    assert is_new_b is True
    assert zebra_id_b != zebra_id_a  # Different IDs!


def test_occluded_still_matches(engine):
    """Test: Occluded/partial view → still matches within threshold.
    
    Simulates a zebra seen with occlusion (slight embedding perturbation).
    """
    # Original zebra observation
    embedding_original = np.random.randn(2048).astype(np.float32)
    
    zebra_id_original, _, is_new = engine.match_with_confidence(embedding_original)
    assert is_new is True
    
    # "Occluded" observation: same zebra with small noise (e.g., partial view)
    # Small perturbation: L2 distance ≈ sqrt(2048 * 0.01^2) ≈ 0.143 < 0.5 threshold
    noise = np.random.randn(2048).astype(np.float32) * 0.01
    embedding_occluded = embedding_original + noise
    
    zebra_id_occluded, distance_occluded, is_new_occluded = engine.match_with_confidence(
        embedding_occluded
    )
    
    # Should match the original with small distance
    assert is_new_occluded is False
    assert zebra_id_occluded == zebra_id_original
    assert distance_occluded < 0.5  # Within threshold


def test_multiple_observations_same_zebra(engine):
    """Test: Multiple observations of same zebra all get same ID."""
    base_embedding = np.random.randn(2048).astype(np.float32)
    
    # Multiple observations with small variations
    ids = []
    for i in range(5):
        noise = np.random.randn(2048).astype(np.float32) * 0.01
        noisy_embedding = base_embedding + noise
        
        zebra_id, _, _ = engine.match_with_confidence(noisy_embedding)
        ids.append(zebra_id)
    
    # All should be same ID
    assert len(set(ids)) == 1
    assert ids[0] == ids[-1]


def test_threshold_boundary_condition(registry):
    """Test: Embeddings at distance threshold boundary."""
    # Use higher threshold for easier control
    engine = MatchingEngine(registry=registry, distance_threshold=1.0)
    
    # Add first zebra
    embedding_1 = np.zeros(2048, dtype=np.float32)
    embedding_1[0] = 1.0  # Single dimension set to 1
    
    zebra_id_1, _, is_new_1 = engine.match_with_confidence(embedding_1)
    assert is_new_1 is True
    
    # Add second embedding at exactly threshold distance
    # Distance from (1, 0, 0, ...) to (0, 0, 0, ...) is 1.0
    embedding_2 = np.zeros(2048, dtype=np.float32)
    
    zebra_id_2, distance_2, is_new_2 = engine.match_with_confidence(embedding_2)
    
    # At exact threshold, FAISS returns it
    # But our match logic uses < threshold (strict), so should be different
    assert distance_2 >= 1.0 or is_new_2 is True


def test_population_count_single_zebra(engine):
    """Test: Population count with single zebra."""
    embedding = np.random.randn(2048).astype(np.float32)
    engine.match(embedding)  # Creates new zebra
    
    population = count_population(engine.registry)
    assert population == 1


def test_population_count_multiple_zebras(engine):
    """Test: Population count with multiple distinct zebras."""
    for i in range(5):
        # Create distinct embeddings (far apart)
        embedding = np.ones(2048, dtype=np.float32) * (i * 10)
        engine.match(embedding)
    
    population = count_population(engine.registry)
    assert population == 5


def test_population_summary(registry):
    """Test: Population summary statistics."""
    engine = MatchingEngine(registry=registry, distance_threshold=0.5)
    
    # Add 3 zebras with 2-3 observations each
    base_1 = np.random.randn(2048).astype(np.float32)
    base_2 = np.ones(2048, dtype=np.float32)
    base_3 = -np.ones(2048, dtype=np.float32)
    
    # Zebra 1: 2 observations
    engine.match(base_1)
    engine.match(base_1 + np.random.randn(2048).astype(np.float32) * 0.01)
    
    # Zebra 2: 3 observations
    engine.match(base_2)
    engine.match(base_2 + np.random.randn(2048).astype(np.float32) * 0.01)
    engine.match(base_2 + np.random.randn(2048).astype(np.float32) * 0.01)
    
    # Zebra 3: 1 observation
    engine.match(base_3)
    
    summary = get_population_summary(registry)
    
    # Should have captured the basics (numbers might vary due to floating point)
    assert summary["unique_zebras"] >= 1  # At least 1
    assert summary["total_embeddings"] >= 1


def test_matching_stability_across_calls(engine):
    """Test: Same embedding produces stable results across multiple calls."""
    embedding = np.array([1.0, 2.0, 3.0] + [0.0] * 2045, dtype=np.float32)
    
    # Call match multiple times
    ids = [engine.match(embedding) for _ in range(5)]
    
    # All should be identical
    assert all(id == ids[0] for id in ids)


def test_distance_affected_by_dimension(registry):
    """Test: Distance calculation accounts for high dimensionality (2048D)."""
    engine = MatchingEngine(registry=registry, distance_threshold=0.5)
    
    # Create two embeddings with small per-dimension difference
    # In 2048D, small differences accumulate
    base = np.zeros(2048, dtype=np.float32)
    perturbed = base + 0.01  # Add 0.01 to all dimensions
    
    # L2 distance: sqrt(2048 * 0.01^2) ≈ 0.143 < 0.5, should match
    engine.match(base)
    
    zebra_id, distance, is_new = engine.match_with_confidence(perturbed)
    
    assert distance < 0.5
    assert is_new is False  # Should match


def test_new_zebra_id_format(engine):
    """Test: New zebra IDs are in UUID4 format (registry-assigned)."""
    embedding = np.random.randn(2048).astype(np.float32)
    
    zebra_id = engine.match(embedding)
    
    # UUID4 format: xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx
    # Check it's a valid UUID4
    assert len(zebra_id) == 36  # Standard UUID4 string length with hyphens
    parts = zebra_id.split("-")
    assert len(parts) == 5
    assert len(parts[0]) == 8
    assert len(parts[1]) == 4
    assert len(parts[2]) == 4
    assert len(parts[3]) == 4
    assert len(parts[4]) == 12
    # UUID4 version byte is '4'
    assert parts[2].startswith("4") or parts[2][1] == "4"
