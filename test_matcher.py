#!/usr/bin/env python3
"""Test script demonstrating FAISS matching functionality without PyTorch conflicts."""

import csv
import sqlite3
from pathlib import Path

import faiss
import numpy as np


def test_matching():
    """Demonstrate matching using pre-extracted embeddings."""
    
    # Load FAISS index
    index = faiss.read_index("zebra.index")
    print(f"✓ Loaded FAISS index: {index.ntotal} vectors, {index.d} dimensions\n")
    
    # Load a test embedding from CSV (simulates a query)
    with open("embeddings.csv", "r") as f:
        reader = csv.DictReader(f)
        test_row = next(reader)
        crop_path = test_row["crop_path"]
        test_emb = np.array([float(test_row[f"emb_{i}"]) for i in range(512)], dtype="float32")
    
    print(f"Query: {crop_path[:60]}...")
    print(f"Embedding L2 norm: {np.linalg.norm(test_emb):.4f} (should be ~1.0 for normalized)\n")
    
    # Search for top 5 matches
    vec = test_emb.reshape(1, -1)
    scores, ids = index.search(vec, k=5)
    
    # Query database for metadata
    conn = sqlite3.connect("zebra.db")
    print("Top 5 matches:")
    print("-" * 80)
    
    for rank, (score, faiss_id) in enumerate(zip(scores[0], ids[0]), 1):
        row = conn.execute(
            "SELECT crop_path, confidence, source_image FROM crops WHERE faiss_id = ?",
            (int(faiss_id),)
        ).fetchone()
        
        if row:
            crop_path, conf, src_img = row
            status = "SELF-MATCH" if rank == 1 else "SIMILAR"
            print(f"{rank}. [{status}] FAISS ID {int(faiss_id)} | score: {score:.4f}")
            print(f"   Confidence: {conf:.3f} | Path: {crop_path}")
            print(f"   Source: {src_img}\n")
    
    conn.close()
    
    # Demonstrate threshold-based matching
    threshold = 0.85
    best_score = scores[0][0]
    best_id = int(ids[0][0])
    
    print(f"Match decision (threshold={threshold}):")
    if best_score >= threshold:
        print(f"✓ MATCH ACCEPTED: score {best_score:.4f} >= {threshold} → FAISS ID {best_id}")
    else:
        print(f"✗ MATCH REJECTED: score {best_score:.4f} < {threshold} → Would create new identity")


if __name__ == "__main__":
    test_matching()
