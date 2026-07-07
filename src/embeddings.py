import json
import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"
path_to_chunks = "C:\\Users\\hp\\OneDrive\\Desktop\\The_Wikipedia_RAG_ChatBot\\data\\processed\\chunks.json"

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

    np.save("C:\\Users\\hp\\OneDrive\\Desktop\\The_Wikipedia_RAG_ChatBot\\data\\embeddings\\embeddings.npy", embeddings)
    with open("C:\\Users\\hp\\OneDrive\\Desktop\\The_Wikipedia_RAG_ChatBot\\data\\embeddings\\embedding_method.txt", "w") as f:
        f.write(method)

    print("Saved embeddings.npy and embedding_method.txt to disk.")