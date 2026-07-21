import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RAW_DIR, PROCESSED_DIR


def clean_text(raw_text: str) -> str:
    text = raw_text

    # Remove Markdown links
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

    # Remove citation markers
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\[[a-z]\]", "", text)
    text = re.sub(r"\[citation needed\]", "", text, flags=re.IGNORECASE)

    # Remove metadata lines
    text = re.sub(r"^Source:.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^License:.*$", "", text, flags=re.MULTILINE)

    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


if __name__ == "__main__":

    # Default file if no argument is given
    input_filename = (
        sys.argv[1] if len(sys.argv) > 1 else "History_of_spaceflight.txt"
    )

    in_path = RAW_DIR / input_filename

    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    with open(in_path, "r", encoding="utf-8") as f:
        raw = f.read()

    cleaned = clean_text(raw)

    print(f"Raw length:     {len(raw)} chars")
    print(f"Cleaned length: {len(cleaned)} chars")
    print("---- Preview (first 400 chars) ----")
    print(cleaned[:400])

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Save using the same filename
    out_path = PROCESSED_DIR / input_filename

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(cleaned)

    print(f"\nSaved cleaned text to {out_path}")