import json
import sys
from pathlib import Path
import faiss

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from config import INDEX_DIR, PROCESSED_DIR
from retreival import search, load_embedder
from llm_intergration import ask_llm

TOP_K = 3


def load_pipeline():
    index_path = INDEX_DIR / "wiki_index.faiss"
    chunks_path = PROCESSED_DIR / "chunks.json"

    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found at {index_path}. Run build index step first.")
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found at {chunks_path}. Run chunking step first.")

    index = faiss.read_index(str(index_path))
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    embed_fn = load_embedder()
    return index, chunks, embed_fn


def main():
    print("=" * 60)
    print("Wikipedia RAG Chatbot  (article: Artificial Intelligence)")
    print("Type 'exit' or 'quit' to stop.")
    print("=" * 60)

    try:
        index, chunks, embed_fn = load_pipeline()
    except Exception as e:
        print(f"Error loading pipeline: {e}")
        return

    while True:
        try:
            question = input("\nYou: ").strip()
            if not question:
                continue
            if question.lower() in ("exit", "quit"):
                print("Goodbye!")
                break

            # 1. Semantic search: top-3 relevant chunks
            retrieved = search(question, index, chunks, embed_fn, top_k=TOP_K)

            print("\n[Retrieved context]")
            for r in retrieved:
                preview = r["text"][:90].replace("\n", " ")
                print(f"  - (score={r['score']:.3f}) {preview}...")

            # 2. Send question + context to the LLM
            answer = ask_llm(question, retrieved)

            print(f"\nBot: {answer}")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError occurred: {e}")


if __name__ == "__main__":
    main()