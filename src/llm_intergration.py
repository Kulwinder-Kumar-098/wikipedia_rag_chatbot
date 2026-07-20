import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from config import LLM_MODEL, INDEX_DIR, PROCESSED_DIR

_tokenizer = None
_model = None


def _load_model():
    global _tokenizer, _model
    if _model is not None:
        return
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    _tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    _model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL,
        quantization_config=bnb_config,
        device_map="auto"
    )


def build_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    context = "\n\n".join(
        f"[Context {i+1}]\n{c['text']}" for i, c in enumerate(retrieved_chunks)
    )
    return (
        f"You are a helpful assistant answering questions using ONLY the context below,\n"
        f"which was retrieved from a Wikipedia article. If the answer isn't in the context,\n"
        f"say you don't have enough information.\n\n"
        f"{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer clearly and concisely, grounded only in the context above."
    )


def ask_llm(question: str, retrieved_chunks: list[dict]) -> str:
    _load_model()
    prompt = build_prompt(question, retrieved_chunks)

    messages = [{"role": "user", "content": prompt}]
    text = _tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = _tokenizer(text, return_tensors="pt").to(_model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output = _model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.2,
            do_sample=True,
            pad_token_id=_tokenizer.eos_token_id
        )

    # Decode only the newly generated tokens (skip the prompt)
    new_tokens = output[0][input_len:]
    return _tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


if __name__ == "__main__":
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
