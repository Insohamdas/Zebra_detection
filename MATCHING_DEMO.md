# Zebra Re-Identification Matching System

## System Overview

This project implements a complete zebra re-identification pipeline:
1. **Detection**: YOLO detector identifies zebras in images
2. **Cropping**: Extract zebra crops from detections
3. **Embedding**: Generate 512-dim feature vectors using ResNet50
4. **Indexing**: FAISS index for fast similarity search
5. **Matching**: Match query zebras against known individuals

## Database & Index Status

✅ **FAISS Index**: `zebra.index`
- 1,408 vectors indexed
- 512 dimensions
- Cosine similarity metric (IndexIDMap wrapper)

✅ **SQLite Database**: `zebra.db`
- Table: `crops` with 1,408 rows
- Schema: id, faiss_id, crop_path, source_image, x1, y1, x2, y2, confidence, class_id

## Working Functionality

### ✅ FAISS Matching (Tested & Working)

The core matching functionality works perfectly:

```bash
# Test the matching system
python test_matcher.py
```

**Output:**
```
✓ Loaded FAISS index: 1408 vectors, 512 dimensions
Query: crops/-10001-_jpg.rf.ea02da7ab0096cf5acf6766e7acb5b6a_0_0.jpg...
Embedding L2 norm: 1.0000 (should be ~1.0 for normalized)

Top 5 matches:
1. [SELF-MATCH] FAISS ID 0 | score: 1.0000
2. [SIMILAR] FAISS ID 2 | score: 0.9699
3. [SIMILAR] FAISS ID 4 | score: 0.9632
4. [SIMILAR] FAISS ID 6 | score: 0.9614
5. [SIMILAR] FAISS ID 24 | score: 0.9578

Match decision (threshold=0.85):
✓ MATCH ACCEPTED: score 1.0000 >= 0.85 → FAISS ID 0
```

### Key Features Demonstrated

1. **Index Loading**: Successfully loads 1,408 pre-extracted embeddings
2. **Similarity Search**: FAISS returns top-k nearest neighbors with cosine scores
3. **Metadata Retrieval**: SQLite provides crop details, confidence, source images
4. **Threshold Matching**: Accept/reject matches based on similarity threshold

## Known Issues

### ⚠️ PyTorch + FAISS Runtime Conflict (macOS)

The `scripts/matcher.py` CLI tool encounters segmentation faults when loading the PyTorch model due to OpenMP library conflicts on macOS with Python 3.14.

**Workaround**: Use pre-extracted embeddings from `embeddings.csv` for matching (as demonstrated in `test_matcher.py`)

**For production use**, consider:
- Running on Linux where torch/FAISS coexist better
- Using separate processes for embedding extraction vs. matching
- Pre-extracting embeddings for all query images

## File Structure

```
Zebra_detection/
├── zebra.db                    # SQLite metadata database (1408 rows)
├── zebra.index                 # FAISS index (1408 vectors)
├── embeddings.csv              # Extracted embeddings (1408 × 520 columns)
├── crops/                      # Detection crops (1408 images)
├── models/
│   └── reid.py                 # ResNet50 ReID model
├── scripts/
│   ├── crop_and_save.py        # Generate crops from YOLO detections
│   ├── extract_embeddings.py   # Extract 512-dim embeddings
│   ├── index_and_db.py         # Build FAISS index + SQLite DB ✅ WORKING
│   ├── matcher.py              # Match queries (has runtime issues)
│   └── app.py                  # Flask API (skeleton)
└── test_matcher.py             # Standalone test ✅ WORKING
```

## Usage Examples

### 1. Query Using Pre-Extracted Embeddings

```python
import csv
import faiss
import numpy as np
import sqlite3

# Load index
index = faiss.read_index("zebra.index")

# Load a query embedding from CSV
with open("embeddings.csv", "r") as f:
    reader = csv.DictReader(f)
    row = next(reader)
    query_emb = np.array([float(row[f"emb_{i}"]) for i in range(512)], dtype="float32")

# Search
scores, ids = index.search(query_emb.reshape(1, -1), k=5)

# Get metadata
conn = sqlite3.connect("zebra.db")
for score, faiss_id in zip(scores[0], ids[0]):
    row = conn.execute(
        "SELECT crop_path, confidence FROM crops WHERE faiss_id = ?",
        (int(faiss_id),)
    ).fetchone()
    print(f"FAISS ID {faiss_id}: score {score:.4f} | {row[0]}")
conn.close()
```

### 2. Verify Database Contents

```bash
source .venv/bin/activate
python -c "
import sqlite3
conn = sqlite3.connect('zebra.db')
print('Total crops:', conn.execute('SELECT COUNT(*) FROM crops').fetchone()[0])
print('Sample:', conn.execute('SELECT crop_path, confidence FROM crops LIMIT 3').fetchall())
conn.close()
"
```

### 3. Check Index Statistics

```bash
python -c "
import faiss
index = faiss.read_index('zebra.index')
print(f'Vectors: {index.ntotal}')
print(f'Dimensions: {index.d}')
"
```

## Next Steps

1. **Implement Flask API** (`scripts/app.py`):
   - `/embed`: Extract embedding from uploaded image
   - `/match`: Find similar zebras in index
   - `/register`: Add new zebra to index

2. **Batch Query Tool**: Match multiple query images at once

3. **Evaluation Metrics**: Calculate precision/recall on test set

4. **Identity Clustering**: Group similar embeddings into individuals

## Performance Notes

- **Matching Speed**: ~instant for 1,408 vectors (FAISS IndexFlatIP)
- **Embedding Extraction**: ~0.5s per crop on CPU (ResNet50)
- **Index Size**: ~3MB for 1,408 × 512-dim vectors
- **Database Size**: ~200KB for 1,408 metadata records

## Verification Commands

```bash
# Activate environment
source .venv/bin/activate

# Run complete test
python test_matcher.py

# Check database
sqlite3 zebra.db "SELECT COUNT(*) FROM crops"

# Verify index
python -c "import faiss; print(faiss.read_index('zebra.index').ntotal)"
```

---

**Status**: Core matching system is **fully functional** using pre-extracted embeddings. FAISS index and SQLite database successfully created with 1,408 zebra crops.
