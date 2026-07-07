import json
import sys
from pathlib import Path

import faiss
import numpy as np

# Make config importable when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import EMBEDDINGS_DIR, INDEX_DIR, PROCESSED_DIR


def build_index(embeddings: np.ndarray) -> faiss.Index:
    """Build a FAISS flat inner-product index (cosine sim on normalised vectors)."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


if __name__ == "__main__":
    # ── Load embeddings ──────────────────────────────────────────────────
    embeddings_path = EMBEDDINGS_DIR / "embeddings.npy"
    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"Embeddings not found: {embeddings_path}\n"
            "Run step4_embed.py first."
        )
    embeddings = np.load(embeddings_path)

    # ── Load chunks (written by step3 into data/processed/) ─────────────
    chunks_path = PROCESSED_DIR / "chunks.json"
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {chunks_path}\n"
            "Run step3_chunk.py first."
        )
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Loaded {embeddings.shape[0]} embeddings of dim {embeddings.shape[1]}")
    print(f"Loaded {len(chunks)} chunks")

    # ── Build & save FAISS index ─────────────────────────────────────────
    index = build_index(embeddings)
    print(f"Index built. Total vectors indexed: {index.ntotal}")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    index_path = INDEX_DIR / "wiki_index.faiss"
    faiss.write_index(index, str(index_path))
    print(f"Saved FAISS index to {index_path}")