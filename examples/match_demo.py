"""
Interactive matching demo - Use this in a Python REPL to test matching.

Usage:
    source .venv/bin/activate
    python -i examples/match_demo.py
    
    # Then in the REPL:
    >>> match_crop_by_index(0)  # Match first crop
    >>> match_crop_by_index(10)  # Match another
    >>> find_similar_crops(50, threshold=0.9)  # Find high-similarity crops
"""

import csv
import sqlite3
from pathlib import Path

import faiss
import numpy as np


# Load resources globally
INDEX = faiss.read_index("zebra.index")
CONN = sqlite3.connect("zebra.db")

print(f"✓ Loaded index with {INDEX.ntotal} vectors")
print(f"✓ Connected to database with {CONN.execute('SELECT COUNT(*) FROM crops').fetchone()[0]} crops\n")


def load_embedding_by_index(idx: int) -> np.ndarray:
    """Load embedding for the i-th crop from CSV."""
    with open("embeddings.csv", "r") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i == idx:
                return np.array([float(row[f"emb_{j}"]) for j in range(512)], dtype="float32")
    raise IndexError(f"Crop index {idx} not found")


def match_crop_by_index(crop_idx: int, top_k: int = 5, threshold: float = 0.85):
    """
    Match a crop against the index using its position in the CSV.
    
    Args:
        crop_idx: Index of crop in embeddings.csv (0-based)
        top_k: Number of nearest neighbors to return
        threshold: Minimum similarity score to accept match
    """
    # Load query embedding
    query_emb = load_embedding_by_index(crop_idx)
    
    # Search
    vec = query_emb.reshape(1, -1)
    scores, ids = INDEX.search(vec, top_k)
    
    # Get query metadata
    query_row = CONN.execute(
        "SELECT crop_path, confidence FROM crops WHERE faiss_id = ?",
        (crop_idx,)
    ).fetchone()
    
    print(f"\nQuery: Crop {crop_idx} → {query_row[0] if query_row else 'Unknown'}")
    print(f"Embedding norm: {np.linalg.norm(query_emb):.4f}")
    print(f"\nTop {top_k} matches:")
    print("-" * 80)
    
    for rank, (score, faiss_id) in enumerate(zip(scores[0], ids[0]), 1):
        row = CONN.execute(
            "SELECT crop_path, confidence FROM crops WHERE faiss_id = ?",
            (int(faiss_id),)
        ).fetchone()
        
        if row:
            status = "✓ MATCH" if score >= threshold else "✗ REJECT"
            self_tag = " [SELF]" if rank == 1 else ""
            print(f"{rank}. {status} | FAISS ID {int(faiss_id)} | score: {score:.4f}{self_tag}")
            print(f"   Path: {row[0]}")
            print(f"   Confidence: {row[1]:.3f}\n")
    
    best_score = scores[0][0]
    print(f"Decision: {'ACCEPT' if best_score >= threshold else 'REJECT'} (threshold={threshold})")
    return scores[0], ids[0]


def find_similar_crops(crop_idx: int, threshold: float = 0.9, max_results: int = 10):
    """
    Find all crops with similarity >= threshold.
    
    Args:
        crop_idx: Query crop index
        threshold: Minimum similarity score
        max_results: Maximum number of results to return
    """
    query_emb = load_embedding_by_index(crop_idx)
    vec = query_emb.reshape(1, -1)
    
    # Search with larger k to find all similar ones
    scores, ids = INDEX.search(vec, min(100, INDEX.ntotal))
    
    # Filter by threshold
    similar = [(s, i) for s, i in zip(scores[0], ids[0]) if s >= threshold][:max_results]
    
    print(f"\nFound {len(similar)} crops with similarity >= {threshold}:")
    print("-" * 80)
    
    for rank, (score, faiss_id) in enumerate(similar, 1):
        row = CONN.execute(
            "SELECT crop_path FROM crops WHERE faiss_id = ?",
            (int(faiss_id),)
        ).fetchone()
        
        if row:
            print(f"{rank}. FAISS ID {int(faiss_id)} | score: {score:.4f} | {row[0]}")
    
    return similar


def get_crop_info(faiss_id: int):
    """Get detailed info for a specific FAISS ID."""
    row = CONN.execute(
        "SELECT * FROM crops WHERE faiss_id = ?",
        (faiss_id,)
    ).fetchone()
    
    if not row:
        print(f"No crop found with FAISS ID {faiss_id}")
        return None
    
    print(f"\nCrop Info for FAISS ID {faiss_id}:")
    print("-" * 80)
    print(f"DB ID: {row[0]}")
    print(f"Path: {row[2]}")
    print(f"Source: {row[3]}")
    print(f"BBox: ({row[4]:.1f}, {row[5]:.1f}, {row[6]:.1f}, {row[7]:.1f})")
    print(f"Confidence: {row[8]:.3f}")
    print(f"Class ID: {row[9]}")
    
    return dict(zip(
        ["id", "faiss_id", "crop_path", "source_image", "x1", "y1", "x2", "y2", "confidence", "class_id"],
        row
    ))


if __name__ == "__main__":
    print("Available functions:")
    print("  - match_crop_by_index(idx, top_k=5, threshold=0.85)")
    print("  - find_similar_crops(idx, threshold=0.9, max_results=10)")
    print("  - get_crop_info(faiss_id)")
    print("\nTry: match_crop_by_index(0)")
