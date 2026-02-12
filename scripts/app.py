"""Lightweight Flask API exposing zebra ReID endpoints."""

import csv
import os
import sqlite3
import sys
import uuid
from io import BytesIO
from pathlib import Path

import faiss
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image

# Allow FAISS + PyTorch to coexist
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
INDEX_PATH = PROJECT_ROOT / "zebra.index"
DB_PATH = PROJECT_ROOT / "zebra.db"
EMBEDDINGS_CSV = PROJECT_ROOT / "embeddings.csv"
MATCH_THRESHOLD = 0.85
TOP_K = 5

# Global cache for FAISS index
_index_cache = None
_embeddings_cache = None


def get_index():
    """Load FAISS index (cached)."""
    global _index_cache
    if _index_cache is None:
        _index_cache = faiss.read_index(str(INDEX_PATH))
        print(f"✓ Loaded FAISS index: {_index_cache.ntotal} vectors")
    return _index_cache


def get_embeddings():
    """Load embeddings from CSV (cached)."""
    global _embeddings_cache
    if _embeddings_cache is None:
        _embeddings_cache = {}
        with open(EMBEDDINGS_CSV, 'r') as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                # Use row index as FAISS ID (matching how index was built)
                emb = np.array([float(row[f'emb_{i}']) for i in range(512)], dtype='float32')
                _embeddings_cache[idx] = emb
        print(f"✓ Loaded {len(_embeddings_cache)} embeddings from CSV")
    return _embeddings_cache


def match_embedding(embedding: np.ndarray):
    """Match embedding against index and return result."""
    index = get_index()
    vec = embedding.reshape(1, -1)
    scores, ids = index.search(vec, TOP_K)

    best_score = float(scores[0][0])
    best_id = int(ids[0][0])

    if best_score < MATCH_THRESHOLD:
        return None, best_score

    # Get metadata from database
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM crops WHERE faiss_id = ?",
        (best_id,)
    ).fetchone()
    conn.close()

    if row:
        return dict(row), best_score
    return None, best_score


def create_identity(embedding: np.ndarray, image_data: bytes = None):
    """Create new zebra identity in index and database."""
    index = get_index()
    conn = sqlite3.connect(DB_PATH)

    # Get next FAISS ID
    max_id_row = conn.execute("SELECT MAX(faiss_id) FROM crops").fetchone()
    next_id = (max_id_row[0] + 1) if max_id_row and max_id_row[0] is not None else 0

    # Add to FAISS
    vec = embedding.reshape(1, -1)
    ids = np.array([next_id], dtype="int64")
    index.add_with_ids(vec, ids)
    faiss.write_index(index, str(INDEX_PATH))

    # Generate unique identifier
    zebra_uuid = str(uuid.uuid4())

    # Add to database
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO crops (faiss_id, crop_path, source_image, x1, y1, x2, y2, confidence, class_id)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (next_id, f"api_upload_{zebra_uuid}.jpg", "api_upload", None, None, None, None, None, 0)
    )
    conn.commit()
    db_id = cur.lastrowid
    conn.close()

    return {
        "id": db_id,
        "faiss_id": next_id,
        "zebra_id": zebra_uuid,
    }


@app.get("/health")
def health():
    """Basic health endpoint for deployment smoke tests."""
    return jsonify(status="ok"), 200


@app.post("/identify")
def identify():
    """
    Identify a zebra from an uploaded image.

    Accepts:
        - multipart/form-data with 'image' file
        - JSON with 'faiss_id' to test with existing embedding

    Returns:
        - If match found: zebra metadata + match score
        - If new: creates new identity and returns zebra_id
    """
    try:
        # Option 1: Upload image file
        if 'image' in request.files:
            file = request.files['image']
            if file.filename == '':
                return jsonify(error="No file selected"), 400

            # For now, return error since PyTorch model loading has conflicts
            # In production, this would extract embedding from the image
            return jsonify(
                error="Image upload not yet supported due to PyTorch conflicts",
                workaround="Use /identify with faiss_id parameter to test with existing embeddings"
            ), 501

        # Option 2: Test with existing FAISS ID
        elif request.is_json:
            data = request.get_json()
            faiss_id = data.get('faiss_id')

            if faiss_id is None:
                return jsonify(error="Missing 'faiss_id' in JSON body"), 400

            # Load embedding from cache
            embeddings = get_embeddings()
            if faiss_id not in embeddings:
                return jsonify(error=f"FAISS ID {faiss_id} not found"), 404

            embedding = embeddings[faiss_id]

            # Match against index
            match_result, score = match_embedding(embedding)

            if match_result:
                return jsonify({
                    "matched": True,
                    "zebra_id": match_result.get('faiss_id'),
                    "score": score,
                    "crop_path": match_result.get('crop_path'),
                    "confidence": match_result.get('confidence'),
                    "threshold": MATCH_THRESHOLD
                }), 200
            else:
                # Create new identity
                new_identity = create_identity(embedding)
                return jsonify({
                    "matched": False,
                    "created": True,
                    "zebra_id": new_identity['zebra_id'],
                    "faiss_id": new_identity['faiss_id'],
                    "score": score,
                    "threshold": MATCH_THRESHOLD
                }), 201

        else:
            return jsonify(error="Request must be multipart/form-data or JSON"), 400

    except Exception as e:
        return jsonify(error=str(e)), 500


@app.get("/stats")
def stats():
    """Get system statistics."""
    try:
        index = get_index()
        conn = sqlite3.connect(DB_PATH)
        crop_count = conn.execute("SELECT COUNT(*) FROM crops").fetchone()[0]
        conn.close()

        return jsonify({
            "index_vectors": index.ntotal,
            "index_dimension": index.d,
            "database_records": crop_count,
            "match_threshold": MATCH_THRESHOLD,
            "top_k": TOP_K
        }), 200
    except Exception as e:
        return jsonify(error=str(e)), 500


@app.get("/zebra/<int:faiss_id>")
def get_zebra(faiss_id):
    """Get metadata for a specific zebra by FAISS ID."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM crops WHERE faiss_id = ?",
            (faiss_id,)
        ).fetchone()
        conn.close()

        if row:
            return jsonify(dict(row)), 200
        else:
            return jsonify(error=f"FAISS ID {faiss_id} not found"), 404
    except Exception as e:
        return jsonify(error=str(e)), 500


@app.get("/export/csv")
def export_csv():
    """Export all zebra data as CSV."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.execute(
            """SELECT
                faiss_id as 'Zebra ID',
                crop_path as 'Image Path',
                source_image as 'Source Image',
                confidence as 'Detection Confidence',
                x1 as 'Bounding Box X1',
                y1 as 'Bounding Box Y1',
                x2 as 'Bounding Box X2',
                y2 as 'Bounding Box Y2',
                class_id as 'Class ID'
            FROM crops
            ORDER BY faiss_id"""
        )

        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return jsonify(error="No data to export"), 404

        # Build CSV
        import io
        output = io.StringIO()
        output.write("Zebra ID,Image Path,Source Image,Detection Confidence,Bounding Box X1,Bounding Box Y1,Bounding Box X2,Bounding Box Y2,Class ID\n")

        for row in rows:
            output.write(','.join(str(val) if val is not None else '' for val in row) + '\n')

        csv_content = output.getvalue()
        output.close()

        from flask import make_response
        response = make_response(csv_content)
        response.headers['Content-Type'] = 'text/csv'
        response.headers['Content-Disposition'] = f'attachment; filename=zebra_database_{__import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        return response

    except Exception as e:
        return jsonify(error=str(e)), 500


if __name__ == "__main__":
    # Preload index and embeddings on startup
    print("Starting Flask API...")
    get_index()
    get_embeddings()
    print(f"✓ Ready to serve on http://0.0.0.0:5001")
    print(f"✓ Match threshold: {MATCH_THRESHOLD}")
    print(f"\nEndpoints:")
    print("  GET  /health           - Health check")
    print("  GET  /stats            - System statistics")
    print("  POST /identify         - Identify zebra (JSON with faiss_id)")
    print("  GET  /zebra/<faiss_id> - Get zebra metadata")
    print("  GET  /export/csv       - Export all zebra data as CSV")
    print("\nPress Ctrl+C to stop")
    app.run(host="0.0.0.0", port=5001, debug=False)
