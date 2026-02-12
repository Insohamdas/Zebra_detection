"""Utilities to match query zebras against indexed embeddings."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Optional

# Allow FAISS + PyTorch to coexist when both bring OpenMP runtimes (macOS workaround).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import torch
from torchvision import transforms
import faiss
import numpy as np
from PIL import Image

from models.reid import build_model


def _faiss_search(index_path: Path, vector: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    """Search FAISS index for top-k matches."""
    index = faiss.read_index(str(index_path))
    vec = vector.reshape(1, -1)
    scores, ids = index.search(vec, top_k)
    return scores[0], ids[0]


def _faiss_add(index_path: Path, vector: np.ndarray, faiss_id: int) -> None:
    """Add vector to FAISS index with specific ID."""
    index = faiss.read_index(str(index_path))
    vec = vector.reshape(1, -1)
    ids = np.array([faiss_id], dtype="int64")
    index.add_with_ids(vec, ids)
    faiss.write_index(index, str(index_path))


def _device_choice(force: Optional[str]) -> torch.device:
    if force:
        return torch.device(force)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_embedding_for_image(
    image_path: Path,
    *,
    weights: Optional[Path] = None,
    embedding_dim: int = 512,
    image_size: int = 224,
    device: Optional[str] = None,
) -> np.ndarray:
    """Encode a crop image into a normalized embedding vector."""

    device_obj = _device_choice(device)
    model = build_model(
        embedding_dim=embedding_dim,
        pretrained=weights is None,
        weights_path=str(weights) if weights else None,
        device=device_obj,
    )
    model.eval()

    transform = _transform(image_size)
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device_obj)

    with torch.inference_mode():
        embedding = model(tensor).cpu().numpy()[0].astype("float32")
    return embedding


def match(query_embedding: np.ndarray, index_path: Path, top_k: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Return FAISS scores + ids for the query embedding."""

    scores, ids = _faiss_search(index_path, query_embedding.astype("float32"), top_k)
    return scores, ids


def _next_faiss_id(conn: sqlite3.Connection) -> int:
    result = conn.execute("SELECT MAX(faiss_id) FROM crops").fetchone()[0]
    return (result or 0) + 1


def _insert_metadata(conn: sqlite3.Connection, faiss_id: int, metadata: dict) -> None:
    conn.execute(
        """
        INSERT INTO crops (
            faiss_id, crop_path, source_image, x1, y1, x2, y2, confidence, class_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            faiss_id,
            metadata.get("crop_path"),
            metadata.get("source_image"),
            metadata.get("x1"),
            metadata.get("y1"),
            metadata.get("x2"),
            metadata.get("y2"),
            metadata.get("confidence"),
            metadata.get("class_id"),
        ),
    )


def match_or_create(
    query_embedding: np.ndarray,
    conn: sqlite3.Connection,
    *,
    index_path: Path,
    metadata: Optional[dict] = None,
    threshold: float = 0.8,
    top_k: int = 5,
) -> dict:
    """Return best match; optionally register new identity when similarity < threshold."""

    scores, ids = match(query_embedding, index_path, top_k=top_k)
    best_score = float(scores[0]) if scores.size else 0.0
    best_id = int(ids[0]) if ids.size else -1

    if best_id >= 0 and best_score >= threshold:
        row = conn.execute(
            "SELECT * FROM crops WHERE faiss_id = ?",
            (best_id,),
        ).fetchone()
        return {
            "status": "match",
            "score": best_score,
            "faiss_id": best_id,
            "record": dict(row) if row else None,
        }

    if metadata is None:
        return {"status": "new", "score": best_score, "faiss_id": None}

    new_id = _next_faiss_id(conn)
    vector = query_embedding.astype("float32")
    _faiss_add(index_path, vector, new_id)
    _insert_metadata(conn, new_id, metadata)
    conn.commit()

    return {"status": "created", "faiss_id": new_id, "score": best_score}


def _default_crop(db_path: Path) -> Path:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute("SELECT crop_path FROM crops ORDER BY id LIMIT 1").fetchone()
    finally:
        conn.close()
    if not row:
        raise SystemExit("No crops found in database. Run scripts/index_and_db.py first.")
    return Path(row["crop_path"])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Match a crop against the FAISS index")
    parser.add_argument("--image", type=Path, default=None, help="Path to crop image. Defaults to first DB entry")
    parser.add_argument("--db", type=Path, default=Path("zebra.db"), help="SQLite DB with metadata")
    parser.add_argument("--index", type=Path, default=Path("zebra.index"), help="FAISS index path")
    parser.add_argument("--weights", type=Path, default=None, help="Optional ReID checkpoint")
    parser.add_argument("--device", type=str, default=None, help="Force device (cpu/cuda/mps)")
    parser.add_argument("--threshold", type=float, default=0.8, help="Similarity threshold to accept matches")
    parser.add_argument("--top-k", type=int, default=5, help="Nearest neighbors to fetch")
    parser.add_argument("--image-size", type=int, default=224, help="Resize dimension before encoding")
    parser.add_argument("--embedding-dim", type=int, default=512, help="Embedding dimension of the model")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    image_path = args.image or _default_crop(args.db)

    embedding = get_embedding_for_image(
        image_path,
        weights=args.weights,
        embedding_dim=args.embedding_dim,
        image_size=args.image_size,
        device=args.device,
    )

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row

    metadata = {
        "crop_path": str(image_path),
        "source_image": str(image_path),
        "x1": None,
        "y1": None,
        "x2": None,
        "y2": None,
        "confidence": None,
        "class_id": None,
    }

    result = match_or_create(
        embedding,
        conn,
        index_path=args.index,
        metadata=metadata,
        threshold=args.threshold,
        top_k=args.top_k,
    )

    conn.close()
    print(json.dumps(result, indent=2, default=float))


if __name__ == "__main__":
    main()
