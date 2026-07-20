import os
import sys
from pathlib import Path

# Ensure src/ is on the path so sibling modules (config, step6_search) are importable
# regardless of which directory the script is launched from.
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from groq import Groq
from dotenv import load_dotenv
from config import LLM_MODEL, INDEX_DIR, PROCESSED_DIR

load_dotenv(_SRC_DIR.parent / ".env")

GROQ_MODEL = LLM_MODEL  

def build_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    """Format retrieved chunks + question into a grounded RAG prompt."""
    context = "\n\n".join(
        f"[Context {i+1}]\n{c['text']}" for i, c in enumerate(retrieved_chunks)
    )
    prompt = f"""You are a helpful assistant answering questions using ONLY the context below,
which was retrieved from a Wikipedia article. If the answer isn't in the context,
say you don't have enough information.

{context}

Question: {question}

Answer clearly and concisely, grounded only in the context above."""
    return prompt


def ask_llm(question: str, retrieved_chunks: list[dict]) -> str:
    """
    Send the question + retrieved context to Groq and return the answer string.
    Reads GROQ_API_KEY from the environment (loaded via .env).
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return (
            "No GROQ_API_KEY found. Add it to your .env file"
        )

    client = Groq(api_key=api_key)
    prompt = build_prompt(question, retrieved_chunks)

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0.2,   # low temp → factual, grounded answers
    )
    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    # Guarantee src/ is on sys.path however this script is launched
    _src = str(Path(__file__).resolve().parent)
    if _src not in sys.path:
        sys.path.insert(0, _src)

    from retreival import load_embedder, search
    import faiss
    import json

    index_path = INDEX_DIR / "wiki_index.faiss"
    chunks_path = PROCESSED_DIR / "chunks.json"

    index = faiss.read_index(str(index_path))
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    embed_fn = load_embedder()

    question = "What year was AI founded as an academic discipline?"
    results = search(question, index, chunks, embed_fn, top_k=3)

    print(f"Q: {question}\n")
    answer = ask_llm(question, results)
    print(f"A: {answer}")