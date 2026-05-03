"""Analytics and reporting utilities for zebra population tracking."""
from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zebraid.registry import PersistentFaissStore


def count_population(registry: PersistentFaissStore) -> int:
    """Count unique zebras in the registry.
    
    Uses SQL to get accurate unique count, handles duplicates properly.
    
    Args:
        registry: FAISS registry containing all known zebra embeddings
    
    Returns:
        Number of unique zebra IDs in the registry
    """
    if hasattr(registry, 'db_file') and registry.db_file:
        # Use database for accurate count
        import sqlite3
        conn = sqlite3.connect(registry.db_file)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(DISTINCT id) FROM zebras")
        result = cursor.fetchone()
        conn.close()
        return result[0] if result[0] else 0
    else:
        # In-memory registry
        all_ids = []
        for flank_list in getattr(registry, 'flank_ids', {}).values():
            all_ids.extend(flank_list)
        return len(set(all_ids))


def get_unique_zebras(registry: PersistentFaissStore) -> list[str]:
    """Get list of all unique zebra IDs.
    
    Args:
        registry: FAISS registry containing all known zebra embeddings
    
    Returns:
        List of unique zebra ID strings
    """
    if hasattr(registry, 'db_file') and registry.db_file:
        import sqlite3
        conn = sqlite3.connect(registry.db_file)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT id FROM zebras ORDER BY created_at")
        result = [row[0] for row in cursor.fetchall()]
        conn.close()
        return result
    else:
        all_ids = []
        for flank_list in getattr(registry, 'flank_ids', {}).values():
            all_ids.extend(flank_list)
        return list(dict.fromkeys(all_ids))


def get_population_summary(registry: PersistentFaissStore) -> dict[str, int | float]:
    """Get population summary statistics.
    
    Args:
        registry: FAISS registry containing all known zebra embeddings
    
    Returns:
        Dictionary with:
            - total_embeddings: Total number of embeddings in index
            - unique_zebras: Count of unique zebra IDs
            - avg_observations: Average observations per zebra
    """
    if hasattr(registry, 'get_stats'):
        # Use built-in stats for persistent registry
        stats = registry.get_stats()
        if "total_embeddings" not in stats:
            stats["total_embeddings"] = stats.get("total_index_size", stats.get("index_size", 0))
        return stats
    else:
        total = registry.ntotal if hasattr(registry, 'ntotal') else 0
        all_ids = []
        for flank_list in getattr(registry, 'flank_ids', {}).values():
            all_ids.extend(flank_list)
        unique = len(set(all_ids))
        
        return {
            "total_embeddings": total,
            "unique_zebras": unique,
            "avg_observations": total / unique if unique > 0 else 0,
            "index_size": total,
        }


def get_zebra_observation_counts(registry: PersistentFaissStore) -> dict[str, int]:
    """Get observation count per zebra (how many times each zebra was seen).
    
    Args:
        registry: FAISS registry containing all known zebra embeddings
    
    Returns:
        Dictionary mapping zebra_id → observation_count
    """
    if hasattr(registry, 'db_file') and registry.db_file:
        import sqlite3
        conn = sqlite3.connect(registry.db_file)
        cursor = conn.cursor()
        cursor.execute("SELECT id, observation_count FROM zebras")
        result = {row[0]: row[1] for row in cursor.fetchall()}
        conn.close()
        return result
    else:
        all_ids = []
        for flank_list in getattr(registry, 'flank_ids', {}).values():
            all_ids.extend(flank_list)
        return dict(Counter(all_ids))


def get_top_observed_zebras(
    registry: PersistentFaissStore,
    top_n: int = 10,
) -> list[tuple[str, int]]:
    """Get most frequently observed zebras.
    
    Args:
        registry: FAISS registry containing all known zebra embeddings
        top_n: Number of top zebras to return
    
    Returns:
        List of (zebra_id, observation_count) tuples, sorted by count descending
    """
    if hasattr(registry, 'db_file') and registry.db_file:
        import sqlite3
        conn = sqlite3.connect(registry.db_file)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, observation_count FROM zebras ORDER BY observation_count DESC LIMIT ?",
            (top_n,),
        )
        result = [(row[0], row[1]) for row in cursor.fetchall()]
        conn.close()
        return result
    else:
        all_ids = []
        for flank_list in getattr(registry, 'flank_ids', {}).values():
            all_ids.extend(flank_list)
        counts = Counter(all_ids)
        return counts.most_common(top_n)

