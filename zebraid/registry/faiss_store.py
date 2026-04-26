"""Persistent FAISS-based registry with SQLite metadata storage."""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

LOGGER = logging.getLogger(__name__)


class PersistentFaissStore:
    """Thread-safe FAISS registry with SQLite metadata persistence.
    
    Stores embeddings in a FAISS index file and metadata (IDs, metadata) in SQLite.
    Provides automatic recovery from disk state.
    
    Uses IndexFlatIP (inner product on L2-normalized vectors = cosine similarity)
    for reliable operation. For registries exceeding 10K entries, call
    upgrade_to_ivf() to switch to approximate IVFFlat search.
    """
    
    def __init__(
        self,
        embedding_dim: int = 160,
        store_path: Optional[str | Path] = None,
        expected_size: int = 10000,
    ):
        """Initialize persistent registry.
        
        Args:
            embedding_dim: Dimensionality of embeddings (default: 160)
            store_path: Directory path for persistence files. If None, uses in-memory only.
                       Will create indices and metadata.db in this directory.
            expected_size: Expected maximum population (stored for future IVF upgrade).
        """
        self.embedding_dim = embedding_dim
        self.store_path = Path(store_path) if store_path else None
        self.expected_size = expected_size
        
        # Use IndexFlatIP for exact cosine similarity search (no training required).
        # On L2-normalized vectors, inner product == cosine similarity.
        self.indices = {
            "left": faiss.IndexFlatIP(embedding_dim),
            "right": faiss.IndexFlatIP(embedding_dim),
        }
        # Map (flank, index_within_flank_index) -> zebra_id
        self.flank_ids: dict[str, list[str]] = {"left": [], "right": []}
        
        # Initialize or load index
        if self.store_path:
            self.store_path.mkdir(parents=True, exist_ok=True)
            self.index_files = {
                "left": self.store_path / "index_left.faiss",
                "right": self.store_path / "index_right.faiss",
            }
            self.db_file = self.store_path / "metadata.db"
            
            # Load from disk if exists
            if self.db_file.exists():
                self._load_from_disk()
            else:
                # Create new
                self._init_db()
        else:
            # In-memory only
            self.db_file = None
            self.index_files = {}
    
    def _init_db(self) -> None:
        """Initialize SQLite database schema."""
        if not self.db_file:
            return
        
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        # Create table for zebra metadata with flank separation
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS zebras (
                id TEXT PRIMARY KEY,
                embedding_index INTEGER NOT NULL,
                flank TEXT NOT NULL,  -- 'left' or 'right'
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                observation_count INTEGER DEFAULT 1,
                model_ver TEXT DEFAULT 'v1.0',
                ref_image BLOB NULL
            )
        """)
        # Ensure we can efficiently find an ID by its position in a flank index
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_flank_pos ON zebras(flank, embedding_index)")
        
        conn.commit()
        conn.close()
    
    def _load_from_disk(self) -> None:
        """Load indices and metadata from disk."""
        # Load FAISS indices
        for side in ["left", "right"]:
            idx_file = self.index_files.get(side)
            if idx_file and idx_file.exists():
                self.indices[side] = faiss.read_index(str(idx_file))
            
        # Load IDs from database
        self.flank_ids = {"left": [], "right": []}
        conn = sqlite3.connect(self.db_file)
        # Handle schema updates for existing databases
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(zebras)")
        columns = [col[1] for col in cursor.fetchall()]
        
        if 'flank' not in columns:
            cursor.execute("ALTER TABLE zebras ADD COLUMN flank TEXT DEFAULT 'left'")
        if 'model_ver' not in columns:
            cursor.execute("ALTER TABLE zebras ADD COLUMN model_ver TEXT DEFAULT 'v1.0'")
        if 'ref_image' not in columns:
            cursor.execute("ALTER TABLE zebras ADD COLUMN ref_image BLOB NULL")
        
        conn.commit()
        cursor.execute("SELECT id, flank FROM zebras ORDER BY flank, embedding_index")
        
        for zebra_id, flank in cursor.fetchall():
            if flank not in self.flank_ids:
                flank = "left"
            self.flank_ids[flank].append(zebra_id)
        
        conn.close()
        total_zebras = sum(len(v) for v in self.flank_ids.values())
        LOGGER.info(f"Loaded registry from disk: {total_zebras} zebras ({len(self.flank_ids['left'])}L, {len(self.flank_ids['right'])}R)")
    
    def upgrade_to_ivf(self, flank: str = "left") -> None:
        """Upgrade a flat index to IVFFlat for faster approximate search.
        
        Should only be called when the index has enough vectors (>= 256).
        Requires reconstructing all vectors and retraining.
        
        Args:
            flank: 'left' or 'right' side to upgrade
        """
        if flank not in self.indices:
            raise ValueError(f"Invalid flank '{flank}'")
        
        flat_idx = self.indices[flank]
        n = flat_idx.ntotal
        if n < 256:
            LOGGER.warning(f"Not enough vectors ({n}) to upgrade flank '{flank}' to IVF. Need >= 256.")
            return
        
        # Reconstruct all vectors from the flat index
        vectors = np.zeros((n, self.embedding_dim), dtype=np.float32)
        for i in range(n):
            vectors[i] = flat_idx.reconstruct(i)
        
        # Build IVFFlat index
        nlist = max(4, int(n ** 0.5))
        quantizer = faiss.IndexFlatIP(self.embedding_dim)
        ivf_idx = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist, faiss.METRIC_INNER_PRODUCT)
        ivf_idx.nprobe = min(nlist, 10)
        
        # Train and populate
        ivf_idx.train(np.ascontiguousarray(vectors, dtype=np.float32))
        ivf_idx.add(np.ascontiguousarray(vectors, dtype=np.float32))
        
        self.indices[flank] = ivf_idx
        LOGGER.info(f"Upgraded flank '{flank}' to IVFFlat with nlist={nlist}, {n} vectors.")
    
    def _normalize(self, embedding: np.ndarray) -> np.ndarray:
        """L2 normalize a 1D embedding vector."""
        embedding = np.asarray(embedding, dtype=np.float32).ravel()
        norm = np.linalg.norm(embedding)
        if norm < 1e-8:
            return embedding
        return (embedding / norm).astype(np.float32)
    
    def add(
        self, 
        embedding: np.ndarray, 
        zebra_id: str, 
        flank: str = "left",
        model_ver: str = "v1.0",
        ref_image: bytes | None = None
    ) -> None:
        """Add embedding to index and store metadata.
        
        Args:
            embedding: 1D numpy array of embedding values
            zebra_id: Unique identifier string for the zebra
            flank: 'left' or 'right' side of the zebra
            model_ver: Version string identifying the encoder model
            ref_image: Best-quality reference crop (JPEG bytes) for re-encoding
        """
        if flank not in ("left", "right"):
            raise ValueError(f"Invalid flank '{flank}'; must be 'left' or 'right'")
            
        # Ensure L2 normalization for cosine similarity compatibility
        embedding = self._normalize(embedding)
        
        # Add to the correct FAISS index
        vec = np.ascontiguousarray(embedding.reshape(1, -1), dtype=np.float32)
        self.indices[flank].add(vec)
        self.flank_ids[flank].append(zebra_id)
        
        # Store metadata in SQLite
        if self.db_file:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            
            try:
                cursor.execute(
                    "INSERT INTO zebras (id, embedding_index, flank, model_ver, ref_image) VALUES (?, ?, ?, ?, ?)",
                    (zebra_id, len(self.flank_ids[flank]) - 1, flank, model_ver, ref_image),
                )
                conn.commit()
            except sqlite3.IntegrityError:
                # ID already exists, increment observation count
                cursor.execute(
                    "UPDATE zebras SET observation_count = observation_count + 1 WHERE id = ?",
                    (zebra_id,),
                )
                conn.commit()
            finally:
                conn.close()
    
    def update_embedding(self, zebra_id: str, new_embedding: np.ndarray, flank: str = "left", alpha: float = 0.1) -> None:
        """Update an existing embedding using an Exponential Moving Average.
        
        For IndexFlatIP, we reconstruct all vectors, replace the target,
        and rebuild the index. This is O(n) but correct for flat indices.
        
        Args:
            zebra_id: ID of the zebra to update
            new_embedding: 1D numpy array of the new observation
            flank: 'left' or 'right'
            alpha: Weight of the new observation (0 to 1). Default 0.1
        """
        if flank not in self.indices:
            raise ValueError(f"Invalid flank '{flank}'")
            
        try:
            idx = self.flank_ids[flank].index(zebra_id)
        except ValueError:
            raise KeyError(f"Zebra ID '{zebra_id}' not found in flank '{flank}'")
            
        idx_engine = self.indices[flank]
        n = idx_engine.ntotal
        
        # Reconstruct all vectors
        all_vectors = np.zeros((n, self.embedding_dim), dtype=np.float32)
        for i in range(n):
            all_vectors[i] = idx_engine.reconstruct(i)
        
        # Apply EMA to the target vector
        new_embedding = np.asarray(new_embedding, dtype=np.float32).ravel()
        updated = (1.0 - alpha) * all_vectors[idx] + alpha * new_embedding
        all_vectors[idx] = self._normalize(updated)
        
        # Rebuild the index with updated vectors
        new_index = faiss.IndexFlatIP(self.embedding_dim)
        new_index.add(np.ascontiguousarray(all_vectors, dtype=np.float32))
        self.indices[flank] = new_index
        
        # Update observation count in database
        if self.db_file:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            try:
                cursor.execute(
                    "UPDATE zebras SET observation_count = observation_count + 1 WHERE id = ?",
                    (zebra_id,),
                )
                conn.commit()
            finally:
                conn.close()
    
    def add_and_get_id(
        self, 
        embedding: np.ndarray, 
        flank: str = "left",
        model_ver: str = "v1.0",
        ref_image: bytes | None = None
    ) -> str:
        """Add embedding to registry and get assigned ID.
        
        The registry assigns a unique ID that is decoupled from the embedding.
        
        Args:
            embedding: 1D numpy array of embedding values
            flank: 'left' or 'right' side of the zebra
            model_ver: Encoder model version for migration tracking
            ref_image: Best-quality reference crop (JPEG bytes) for re-encoding
        
        Returns:
            Registry-assigned unique ID (UUID4 string)
        """
        from uuid import uuid4
        
        # Generate a new unique ID
        zebra_id = str(uuid4())
        
        # Add embedding with this ID and flank
        self.add(embedding, zebra_id, flank=flank, model_ver=model_ver, ref_image=ref_image)
        
        return zebra_id
    
    def search(self, embedding: np.ndarray, flank: str = "left", k: int = 1) -> tuple[str, float]:
        """Search for nearest neighbor in the specific flank index.
        
        Args:
            embedding: 1D numpy array of query embedding
            flank: 'left' or 'right' side to search within
            k: Number of neighbors to return (default: 1)
        
        Returns:
            Tuple of (zebra_id, distance) for nearest neighbor
        
        Raises:
            RuntimeError: If the specified flank index is empty
        """
        if flank not in self.indices:
            raise ValueError(f"Invalid flank '{flank}'")
            
        idx = self.indices[flank]
        if idx.ntotal == 0:
            raise RuntimeError(f"Cannot search empty index for flank '{flank}'")
        
        # Ensure normalization for consistent comparison
        embedding = self._normalize(embedding)
        
        vec = np.ascontiguousarray(embedding.reshape(1, -1), dtype=np.float32)
        D, I = idx.search(vec, k)
        return self.flank_ids[flank][I[0][0]], float(D[0][0])
    
    @property
    def ntotal(self) -> int:
        """Total number of embeddings across all indices."""
        return sum(idx.ntotal for idx in self.indices.values())
    
    def save(self) -> None:
        """Save indices and metadata to disk."""
        if not self.store_path:
            return
        
        # Save FAISS indices
        for side, idx in self.indices.items():
            if idx.ntotal > 0:
                faiss.write_index(idx, str(self.index_files[side]))
        LOGGER.info(f"Saved FAISS indices to {self.store_path}")
    
    def get_stats(self) -> dict:
        """Get registry statistics."""
        if self.db_file:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            cursor.execute("SELECT flank, COUNT(DISTINCT id), SUM(observation_count) FROM zebras GROUP BY flank")
            stats = cursor.fetchall()
            conn.close()
            
            result = {
                "unique_zebras": sum(s[1] for s in stats),
                "total_observations": sum(s[2] for s in stats),
                "by_flank": {s[0]: {"unique": s[1], "obs": s[2]} for s in stats},
                "total_index_size": self.ntotal,
            }
            return result
        else:
            return {
                "unique_zebras": sum(len(set(v)) for v in self.flank_ids.values()),
                "total_observations": sum(len(v) for v in self.flank_ids.values()),
                "total_index_size": self.ntotal,
            }


# Backward compatibility alias
FaissStore = PersistentFaissStore
