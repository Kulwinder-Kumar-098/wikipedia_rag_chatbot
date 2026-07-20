import json
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import EMBEDDING_MODEL, PROCESSED_DIR, EMBEDDINGS_DIR

MODEL_NAME = EMBEDDING_MODEL
path_to_chunks = PROCESSED_DIR / "chunks.json"
output_path = EMBEDDINGS_DIR / "embeddings.npy"
method_path = EMBEDDINGS_DIR / "embedding_method.txt"

def load_chunks(path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def embed_with_sentence_transformers(texts: list[str]):
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # normalize so cosine similarity == dot product
    )
    return embeddings, "sentence-transformers/all-MiniLM-L6-v2"


if __name__ == "__main__":
    chunks = load_chunks(path_to_chunks)
    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(chunks)} chunks...")

    try:
        print(f"Attempting to load model: {MODEL_NAME} (downloads ~80MB on first run)")
        embeddings, method = embed_with_sentence_transformers(texts)
    except Exception as e:
        print(f"\nCould not reach HuggingFace Hub ({type(e).__name__}).")
        
        
    print(f"Embedding method used: {method}")
    print(f"Embeddings shape: {embeddings.shape}")

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embeddings)
    method_path.write_text(method, encoding="utf-8")

    print(f"Saved embeddings to {output_path}")
    print(f"Saved embedding method to {method_path}")