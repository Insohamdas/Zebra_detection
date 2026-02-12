"""Index embeddings and persist metadata into the project database."""

from __future__ import annotations

import argparse
import csv
import sqlite3
from pathlib import Path
from typing import Iterable, List, Tuple

import faiss
import numpy as np


BASE_COLUMNS = [
    "crop_path",
    "source_image",
    "x1",
    "y1",
    "x2",
    "y2",
    "confidence",
    "class_id",
]


def _float_or_none(value: str | None) -> float | None:
    if value in (None, "", "None"):
        return None
    return float(value)


def _int_or_none(value: str | None) -> int | None:
    if value in (None, "", "None"):
        return None
    return int(float(value))


def _read_embeddings(path: Path) -> Tuple[np.ndarray, List[dict]]:
    with path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError("Embeddings CSV must include headers")

        emb_cols = [name for name in reader.fieldnames if name.startswith("emb_")]
        if not emb_cols:
            raise ValueError("No embedding columns (emb_*) found in CSV")

        vectors: List[List[float]] = []
        records: List[dict] = []

        for row in reader:
            metadata = {
                "crop_path": row.get("crop_path"),
                "source_image": row.get("source_image"),
                "x1": _float_or_none(row.get("x1")),
                "y1": _float_or_none(row.get("y1")),
                "x2": _float_or_none(row.get("x2")),
                "y2": _float_or_none(row.get("y2")),
                "confidence": _float_or_none(row.get("confidence")),
                "class_id": _int_or_none(row.get("class_id")),
            }
            vector = [_float_or_none(row.get(col)) for col in emb_cols]
            if None in vector:
                raise ValueError(f"Encountered missing values in embedding columns for {metadata['crop_path']}")

            vectors.append([float(v) for v in vector])
            records.append(metadata)

    embeddings = np.asarray(vectors, dtype="float32")
    return embeddings, records


def _prepare_database(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS crops (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            faiss_id INTEGER UNIQUE,
            crop_path TEXT UNIQUE,
            source_image TEXT,
            x1 REAL,
            y1 REAL,
            x2 REAL,
            y2 REAL,
            confidence REAL,
            class_id INTEGER
        )
        """
    )


def _sync_metadata(conn: sqlite3.Connection, records: List[dict]) -> None:
    conn.execute("DELETE FROM crops")
    for faiss_id, record in enumerate(records):
        conn.execute(
            """
            INSERT INTO crops (
                faiss_id, crop_path, source_image, x1, y1, x2, y2, confidence, class_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                faiss_id,
                record["crop_path"],
                record["source_image"],
                record["x1"],
                record["y1"],
                record["x2"],
                record["y2"],
                record["confidence"],
                record["class_id"],
            ),
        )


def build_index(embeddings_path: Path, db_path: Path, index_path: Path, metric: str = "cosine") -> faiss.Index:
    """Create/update FAISS index and sync information with the SQL database."""

    embeddings, records = _read_embeddings(embeddings_path)
    if embeddings.size == 0:
        raise SystemExit("Embeddings CSV is empty")

    dim = embeddings.shape[1]
    if metric == "cosine":
        base_index = faiss.IndexFlatIP(dim)
    else:
        base_index = faiss.IndexFlatL2(dim)
    index = faiss.IndexIDMap(base_index)

    ids = np.arange(len(records), dtype="int64")
    index.add_with_ids(embeddings, ids)

    conn = sqlite3.connect(db_path)
    try:
        _prepare_database(conn)
        _sync_metadata(conn, records)
        conn.commit()
    finally:
        conn.close()

    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    print(f"Indexed {len(records)} embeddings (dim={dim}) -> {index_path}")
    print(f"SQLite metadata stored in {db_path}")
    return index


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FAISS index + SQLite metadata from embeddings CSV")
    parser.add_argument("--embeddings", default="embeddings.csv", type=Path, help="Path to embeddings CSV")
    parser.add_argument("--db", default="zebra.db", type=Path, help="SQLite DB output")
    parser.add_argument("--index", default="zebra.index", type=Path, help="FAISS index output path")
    parser.add_argument("--metric", choices=["cosine", "l2"], default="cosine", help="Similarity metric")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    build_index(args.embeddings, args.db, args.index, metric=args.metric)
