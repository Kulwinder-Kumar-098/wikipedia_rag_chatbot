import sys
from pathlib import Path

import wikipediaapi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RAW_DIR


def fetch_article(title: str, lang: str = "en") -> str:
    wiki = wikipediaapi.Wikipedia(
        language=lang,
        user_agent="RAG-Chatbot-Tutorial/1.0 (contact@example.com)"
    )
    page = wiki.page(title)

    if not page.exists():
        raise ValueError(f"Wikipedia page '{title}' does not exist.")

    return page.text  # full plain-text article, no HTML


if __name__ == "__main__":
    title = sys.argv[1] if len(sys.argv) > 1 else "History of spaceflight"

    text = fetch_article(title)

    print(f"Fetched article: '{title}'")
    print(f"Total characters: {len(text)}")
    print("---- Preview (first 500 chars) ----")
    print(text[:500])

    # Save raw text to disk for the next steps to use
    out_path = RAW_DIR / "article_raw.txt"
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\nSaved full article text to {out_path}")