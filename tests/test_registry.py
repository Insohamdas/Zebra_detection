"""Tests for FAISS-based registry store."""
import numpy as np
import pytest
from zebraid.registry import FaissStore


def test_faiss_store_init():
    """Test FaissStore initialization creates correct index."""
    store = FaissStore(embedding_dim=2048)
    assert store.index.d == 2048  # d is the dimension property of FAISS index
    assert store.ids == []
    assert store.index.ntotal == 0  # No vectors added yet


def test_faiss_store_add_single_embedding():
    """Test adding a single embedding to the store."""
    store = FaissStore(embedding_dim=2048)
    embedding = np.random.randn(2048).astype(np.float32)
    
    store.add(embedding, "zebra_001")
    
    assert store.index.ntotal == 1
    assert store.ids == ["zebra_001"]


def test_faiss_store_add_multiple_embeddings():
    """Test adding multiple embeddings preserves order."""
    store = FaissStore(embedding_dim=2048)
    embeddings = [np.random.randn(2048).astype(np.float32) for _ in range(5)]
    zebra_ids = [f"zebra_{i:03d}" for i in range(5)]
    
    for embedding, zebra_id in zip(embeddings, zebra_ids):
        store.add(embedding, zebra_id)
    
    assert store.index.ntotal == 5
    assert store.ids == zebra_ids


def test_faiss_store_search_returns_correct_type():
    """Test search returns tuple of (zebra_id, cosine_similarity)."""
    store = FaissStore(embedding_dim=2048)
    embedding = np.array([1.0] * 2048, dtype=np.float32)
    
    store.add(embedding, "zebra_001")
    
    zebra_id, distance = store.search(embedding)
    
    assert isinstance(zebra_id, str)
    assert zebra_id == "zebra_001"
    assert isinstance(distance, (float, np.floating))
    assert 0.0 <= distance <= 1.0


def test_faiss_store_search_exact_match():
    """Test searching for identical embedding returns cosine similarity near one."""
    store = FaissStore(embedding_dim=2048)
    embedding = np.array([1.0, 2.0, 3.0] + [0.0] * 2045, dtype=np.float32)
    
    store.add(embedding, "zebra_exact")
    
    zebra_id, distance = store.search(embedding)
    
    assert zebra_id == "zebra_exact"
    assert distance > 0.99


def test_faiss_store_search_nearest_neighbor():
    """Test that search returns nearest neighbor correctly."""
    store = FaissStore(embedding_dim=2048)
    
    # Add embeddings: base + small perturbation, and base + large perturbation
    base = np.random.randn(2048).astype(np.float32)
    close_embedding = base + np.random.randn(2048).astype(np.float32) * 0.01
    far_embedding = base + np.random.randn(2048).astype(np.float32) * 1.0
    
    store.add(close_embedding, "zebra_close")
    store.add(far_embedding, "zebra_far")
    
    zebra_id, distance = store.search(base)
    
    # Nearest neighbor should be the close one
    assert zebra_id == "zebra_close"


def test_faiss_store_search_finds_all_entries():
    """Test that each added embedding can be found as nearest neighbor."""
    store = FaissStore(embedding_dim=2048)
    embeddings = []
    for i in range(5):
        embedding = np.zeros(2048, dtype=np.float32)
        embedding[i] = 1.0
        embeddings.append(embedding)
    zebra_ids = [f"zebra_{i:03d}" for i in range(5)]
    
    for embedding, zebra_id in zip(embeddings, zebra_ids):
        store.add(embedding, zebra_id)
    
    # Each embedding should find itself as nearest neighbor
    for embedding, expected_id in zip(embeddings, zebra_ids):
        found_id, distance = store.search(embedding)
        assert found_id == expected_id
        assert distance > 0.99


def test_faiss_store_with_custom_dimension():
    """Test FaissStore with custom embedding dimension."""
    store = FaissStore(embedding_dim=512)
    embedding = np.random.randn(512).astype(np.float32)
    
    store.add(embedding, "zebra_512")
    
    found_id, distance = store.search(embedding)
    assert found_id == "zebra_512"
    assert store.index.d == 512


def test_faiss_store_handles_float64_input():
    """Test that float64 arrays are handled correctly."""
    store = FaissStore(embedding_dim=2048)
    # Create float64 array (numpy default)
    embedding = np.random.randn(2048)  # float64 by default
    
    # Should work with type conversion
    store.add(embedding, "zebra_float64")
    
    found_id, distance = store.search(embedding)
    assert found_id == "zebra_float64"


def test_faiss_store_stores_binary_codes_and_ssi_profile():
    store = FaissStore(embedding_dim=2048)
    embedding = np.random.randn(2048).astype(np.float32)
    global_code = np.zeros(512, dtype=np.uint8)
    local_codes = {
        "shoulder": np.zeros(128, dtype=np.uint8),
        "torso": np.ones(128, dtype=np.uint8),
        "neck": np.zeros(64, dtype=np.uint8),
    }
    ssi_profile = np.ones((3, 38), dtype=np.float32)

    store.add(
        embedding,
        "zebra_codes",
        flank="right",
        global_code=global_code,
        local_codes=local_codes,
        ssi_profile=ssi_profile,
    )

    assert store.global_codes["right"]["zebra_codes"].shape == (512,)
    assert store.local_codes["right"]["zebra_codes"]["shoulder"].shape == (128,)
    assert store.ssi_profiles["right"]["zebra_codes"].shape == (3, 38)


def test_faiss_store_hamming_and_local_refine():
    store = FaissStore(embedding_dim=2048)
    embedding = np.random.randn(2048).astype(np.float32)
    global_code = np.zeros(512, dtype=np.uint8)
    local_codes = {
        "shoulder": np.zeros(128, dtype=np.uint8),
        "torso": np.zeros(128, dtype=np.uint8),
        "neck": np.zeros(64, dtype=np.uint8),
    }
    store.add(embedding, "zebra_binary", global_code=global_code, local_codes=local_codes)

    assert store.hamming_search(global_code)[0][0] == "zebra_binary"
    assert store.local_refine(local_codes, ["zebra_binary"])[0][0] == "zebra_binary"
