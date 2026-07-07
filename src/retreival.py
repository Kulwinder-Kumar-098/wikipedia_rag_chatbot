import json
import sys
from pathlib import Path

import faiss
import numpy as np

# Make config importable when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import EMBEDDINGS_DIR, INDEX_DIR, PROCESSED_DIR

TOP_K = 3


def load_embedder():
    """
    Read the embedding method saved by step4 and return a matching embed function.
    Raises RuntimeError if the method file is missing or unrecognised.
    """
    method_path = EMBEDDINGS_DIR / "embedding_method.txt"
    if not method_path.exists():
        raise FileNotFoundError(
            f"Embedding method file not found: {method_path}\n"
            "Run step4_embed.py first."
        )

    method = method_path.read_text(encoding="utf-8").strip()

    if method.startswith("sentence-transformers"):
        from sentence_transformers import SentenceTransformer

        # Extract model name after the slash, e.g. "sentence-transformers/all-MiniLM-L6-v2"
        model_name = method.split("/", 1)[-1] if "/" in method else "all-MiniLM-L6-v2"
        model = SentenceTransformer(model_name)

        def embed(query: str) -> np.ndarray:
            vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
            return vec.astype(np.float32)

        return embed

    raise RuntimeError(
        f"Unrecognised embedding method '{method}' in {method_path}.\n"
        "Only 'sentence-transformers/*' is currently supported."
    )


def search(query: str, index: faiss.Index, chunks: list, embed_fn, top_k: int = TOP_K):
    query_vec = embed_fn(query)
    scores, indices = index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        results.append({"score": float(score), "text": chunks[idx]["text"], "id": int(idx)})
    return results


if __name__ == "__main__":
    # Load FAISS index
    index_path = INDEX_DIR / "wiki_index.faiss"
    if not index_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {index_path}\n"
            "Run step5_faiss.py first."
        )
    index = faiss.read_index(str(index_path))

    # Load chunks 
    chunks_path = PROCESSED_DIR / "chunks.json"
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {chunks_path}\n"
            "Run step3_chunk.py first."
        )
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    embed_fn = load_embedder()

    print(f"Index loaded: {index.ntotal} vectors | Chunks loaded: {len(chunks)}\n")

    test_queries = [
        "What are the risks of artificial intelligence?",
        "How does deep learning work?",
        "Who founded AI as an academic field?",
    ]

    for q in test_queries:
        print(f"=== Query: {q!r} ===")
        results = search(q, index, chunks, embed_fn)
        for r in results:
            print(f"  [score={r['score']:.1f}, chunk#{r['id']}] {r['text'][:150]}...")
        print()