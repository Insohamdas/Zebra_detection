"""Tests for the matching engine."""
import numpy as np
import pytest

from zebraid.matching import MatchingEngine
from zebraid.registry import FaissStore


@pytest.fixture
def registry():
    """Create a fresh FAISS registry for each test."""
    return FaissStore(embedding_dim=160)


@pytest.fixture
def engine(registry):
    """Create a matching engine with default threshold."""
    return MatchingEngine(registry=registry, similarity_threshold=0.75)


def test_matching_engine_init():
    """Test MatchingEngine initialization."""
    registry = FaissStore(embedding_dim=160)
    engine = MatchingEngine(registry=registry, similarity_threshold=0.75)
    
    assert engine.registry is registry
    assert engine.similarity_threshold == 0.75


def test_matching_engine_exact_match(engine):
    """Test that exact embeddings are matched within threshold."""
    # Create an embedding and add to registry
    embedding = np.array([1.0, 2.0, 3.0] + [0.0] * 157, dtype=np.float32)
    engine.add_zebra(embedding, "zebra_001", flank="left")
    
    # Search for the exact same embedding with correct flank
    matched_id = engine.match(embedding, flank="left")
    
    assert matched_id == "zebra_001"


def test_matching_engine_new_zebra_outside_threshold(engine):
    """Test that embeddings outside threshold create new IDs."""
    # Add one zebra to registry
    embedding1 = np.array([1.0] * 160, dtype=np.float32)
    engine.add_zebra(embedding1, "zebra_001", flank="left")
    
    # Query with very different embedding (far outside threshold)
    # Normed [1,1...] vs Normed [-1,-1...] has cos_sim = -1
    embedding2 = np.array([-1.0] * 160, dtype=np.float32)
    matched_id = engine.match(embedding2, flank="left")
    
    # Should create new ID, not return zebra_001
    assert matched_id != "zebra_001"
    assert matched_id.startswith("ZEB-")
    assert matched_id.count("-") >= 5


def test_matching_engine_match_with_confidence(engine):
    """Test match_with_confidence returns correct tuple."""
    embedding = np.array([1.0, 2.0, 3.0] + [0.0] * 157, dtype=np.float32)
    engine.add_zebra(embedding, "zebra_existing", flank="left")
    
    # Exact match should have is_new=False and high confidence
    matched_id, confidence, is_new = engine.match_with_confidence(embedding, flank="left")
    assert matched_id == "zebra_existing"
    assert confidence > 0.99  # Should be very close to 1.0 for exact match
    assert is_new is False
    
    # New embedding should have is_new=True
    new_embedding = np.array([-1.0] * 160, dtype=np.float32)
    new_id, new_confidence, is_new_flag = engine.match_with_confidence(new_embedding, flank="left")
    assert is_new_flag is True
    assert new_id != "zebra_existing"

def test_matching_engine_confidence_calibration(engine):
    """Test that confidence scoring properly maps cosine domain [-1, 1] to [0, 1]."""
    base = np.array([1.0] + [0.0] * 159, dtype=np.float32)

    exact_engine = MatchingEngine(registry=FaissStore(embedding_dim=160), similarity_threshold=0.75)
    exact_engine.add_zebra(base, "zebra_base", flank="left")
    _, conf_exact, _ = exact_engine.match_with_confidence(base, flank="left")
    assert conf_exact > 0.99
    
    orthogonal = np.array([0.0, 1.0] + [0.0] * 158, dtype=np.float32)
    ortho_engine = MatchingEngine(registry=FaissStore(embedding_dim=160), similarity_threshold=0.75)
    ortho_engine.add_zebra(base, "zebra_base", flank="left")
    _, conf_ortho, _ = ortho_engine.match_with_confidence(orthogonal, flank="left")
    assert 0.49 < conf_ortho < 0.51
    
    opposite = np.array([-1.0] + [0.0] * 159, dtype=np.float32)
    opp_engine = MatchingEngine(registry=FaissStore(embedding_dim=160), similarity_threshold=0.75)
    opp_engine.add_zebra(base, "zebra_base", flank="left")
    _, conf_opp, _ = opp_engine.match_with_confidence(opposite, flank="left")
    assert conf_opp < 0.01


def test_matching_engine_near_match_within_threshold(engine, registry):
    """Test that nearby embeddings (within threshold) are matched."""
    # Create base embedding and add to registry
    base = np.array([1.0] * 160, dtype=np.float32)
    engine.add_zebra(base, "zebra_close", flank="left")
    
    # Create a slight perturbation (within default 0.75 similarity threshold)
    # Norm is handled by registry. Small perturbation keeps similarity high.
    perturbed = base + np.random.randn(160).astype(np.float32) * 0.01
    
    matched_id = engine.match(perturbed, flank="left")
    
    # Should match the existing zebra
    assert matched_id == "zebra_close"


def test_matching_engine_custom_threshold(registry):
    """Test matching engine with custom similarity threshold."""
    engine = MatchingEngine(registry=registry, similarity_threshold=0.9)
    
    # Add embedding to registry with manual ID
    embedding1 = np.array([1.0] * 160, dtype=np.float32)
    engine.add_zebra(embedding1, "zebra_threshold_test", flank="left")
    
    # Query with identical embedding (similarity should be 1.0)
    matched_id = engine.match(embedding1, flank="left")
    
    # Should match the existing zebra
    assert matched_id == "zebra_threshold_test"
    
    # Verify custom threshold is applied
    assert engine.similarity_threshold == 0.9


def test_matching_engine_add_zebra(engine, registry):
    """Test adding zebras to registry through matching engine."""
    embedding1 = np.array([1.0, 2.0, 3.0] + [0.0] * 157, dtype=np.float32)
    embedding2 = np.array([4.0, 5.0, 6.0] + [0.0] * 157, dtype=np.float32)
    
    engine.add_zebra(embedding1, "zebra_A", flank="left")
    engine.add_zebra(embedding2, "zebra_B", flank="right")
    
    # Both should be findable
    assert engine.match(embedding1, flank="left") == "zebra_A"
    assert engine.match(embedding2, flank="right") == "zebra_B"
    assert engine.registry.ntotal == 2


def test_matching_engine_deterministic_new_ids(engine):
    """Test that engine returns the exact same ID for repeated occurrences by querying the registry."""
    embedding = np.array([7.0, 8.0, 9.0] + [0.0] * 157, dtype=np.float32)
    
    # Query multiple times with no registry entries initially
    id1 = engine.match(embedding, flank="left")
    id2 = engine.match(embedding, flank="left")
    
    # Should return identical IDs because the first match registered the embedding
    assert id1 == id2


def test_matching_engine_empty_registry_creates_new(engine):
    """Test that empty registry creates new ID for any embedding."""
    embedding = np.random.randn(160).astype(np.float32)
    
    # Registry is empty, so should create new ID
    matched_id = engine.match(embedding, flank="left")
    
    assert matched_id.startswith("ZEB-")
    assert matched_id.count("-") >= 5


def test_matching_engine_multiple_zebras_returns_closest(engine):
    """Test that search returns closest zebra when multiple exist."""
    # Add multiple zebras at different distances
    close = np.array([1.0] * 160, dtype=np.float32)
    far = np.array([-1.0] * 160, dtype=np.float32)
    
    engine.add_zebra(close, "zebra_close", flank="left")
    engine.add_zebra(far, "zebra_far", flank="left")
    
    # Query near the close one
    query = close + np.random.randn(160).astype(np.float32) * 0.01
    
    matched_id = engine.match(query, flank="left")
    
    # Should match the closer one
    assert matched_id == "zebra_close"


def test_matching_engine_float64_input(engine):
    """Test that float64 embeddings are handled correctly."""
    # NumPy defaults to float64
    embedding = np.random.randn(160)  # float64
    
    # Should work without error
    matched_id = engine.match(embedding, flank="left")
    
    # Should return a valid ID
    assert isinstance(matched_id, str)
    assert len(matched_id) > 0


def test_matching_engine_flank_separation(engine):
    """Test that embeddings on different flanks do not match."""
    embedding = np.array([1.0] * 160, dtype=np.float32)
    
    # Add to left flank
    id_left = engine.match(embedding, flank="left")
    
    # Search same embedding on right flank - should create a NEW ID
    id_right = engine.match(embedding, flank="right")
    
    assert id_left != id_right
    
    # Search again on left - should match first ID
    assert engine.match(embedding, flank="left") == id_left
    
    # Search again on right - should match second ID
    assert engine.match(embedding, flank="right") == id_right


def test_matching_engine_three_phase_with_local_refine(engine):
    embedding_a = np.array([1.0] + [0.0] * 159, dtype=np.float32)
    embedding_b = np.array([0.9, 0.1] + [0.0] * 158, dtype=np.float32)
    code_a = np.zeros(512, dtype=np.uint8)
    code_b = np.zeros(512, dtype=np.uint8)
    code_b[:20] = 1
    local_a = {
        "shoulder": np.zeros(128, dtype=np.uint8),
        "torso": np.zeros(128, dtype=np.uint8),
        "neck": np.zeros(64, dtype=np.uint8),
    }
    local_b = {
        "shoulder": np.ones(128, dtype=np.uint8),
        "torso": np.ones(128, dtype=np.uint8),
        "neck": np.ones(64, dtype=np.uint8),
    }

    engine.registry.add(embedding_a, "zebra_A", flank="left", global_code=code_a, local_codes=local_a)
    engine.registry.add(embedding_b, "zebra_B", flank="left", global_code=code_b, local_codes=local_b)

    matched_id, score, phase = engine.match_three_phase(
        embedding_a,
        global_code=code_b,
        local_codes=local_b,
        flank="left",
    )

    assert matched_id == "zebra_B"
    assert score > 0.9
    assert phase in {"hamming", "local_refine"}


def test_temporal_drift_flag_on_hamming_change(engine):
    embedding = np.array([1.0] + [0.0] * 159, dtype=np.float32)
    enrolled_code = np.zeros(512, dtype=np.uint8)
    drifted_code = np.ones(512, dtype=np.uint8)

    engine.registry.add(embedding, "zebra_drift", flank="left", global_code=enrolled_code)
    _, _, is_new = engine.match_with_confidence(embedding, flank="left", global_code=drifted_code)

    assert is_new is False
    assert engine.registry.drift_flags["left"]["zebra_drift"] is True


def test_resolve_three_phase_identity_enrolls_when_no_code_match(registry):
    engine = MatchingEngine(registry=registry, similarity_threshold=0.75)
    embedding = np.array([1.0, 0.0] + [0.0] * 158, dtype=np.float32)
    global_code = np.ones(512, dtype=np.uint8)

    zebra_id, confidence, is_new, phase = engine.resolve_three_phase_identity(
        embedding,
        global_code=global_code,
        flank="left",
        stripe_stats=np.ones(18, dtype=np.float32),
    )

    assert zebra_id.startswith("ZEB-")
    assert confidence == 1.0
    assert is_new is True
    assert phase == "enroll"


def test_resolve_three_phase_identity_matches_existing_by_hamming(registry):
    engine = MatchingEngine(registry=registry, similarity_threshold=0.75)
    base = np.array([1.0, 0.0] + [0.0] * 158, dtype=np.float32)
    query = np.array([0.9, 0.1] + [0.0] * 158, dtype=np.float32)
    code = np.zeros(512, dtype=np.uint8)
    local_codes = {
        "shoulder": np.zeros(128, dtype=np.uint8),
        "torso": np.zeros(128, dtype=np.uint8),
        "neck": np.zeros(64, dtype=np.uint8),
    }

    engine.registry.add(base, "zebra_code_match", flank="left", global_code=code, local_codes=local_codes)

    zebra_id, confidence, is_new, phase = engine.resolve_three_phase_identity(
        query,
        global_code=code,
        local_codes=local_codes,
        flank="left",
        stripe_stats=np.zeros(18, dtype=np.float32),
    )

    assert zebra_id == "zebra_code_match"
    assert confidence > 0.99
    assert is_new is False
    assert phase in {"hamming", "local_refine"}
