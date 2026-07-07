import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RAW_DIR, PROCESSED_DIR


def clean_text(raw_text: str) -> str:
    text = raw_text


    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\[[a-z]\]", "", text)
    text = re.sub(r"\[citation needed\]", "", text, flags=re.IGNORECASE)

    text = re.sub(r"^Source:.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^License:.*$", "", text, flags=re.MULTILINE)


    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    return text


if __name__ == "__main__":

    in_path = RAW_DIR / "article_raw.txt"
    if not in_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {in_path}\n"
            "Run step1_fetch.py first to generate article_raw.txt"
        )

    with open(in_path, "r", encoding="utf-8") as f:
        raw = f.read()

    cleaned = clean_text(raw)

    print(f"Raw length:     {len(raw)} chars")
    print(f"Cleaned length: {len(cleaned)} chars")
    print("---- Preview (first 400 chars) ----")
    print(cleaned[:400])


    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "article_clean.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(cleaned)
    print(f"\nSaved cleaned text to {out_path}")