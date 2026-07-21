import re
import sys
from pathlib import Path

import wikipediaapi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RAW_DIR


def fetch_article(title: str, lang: str = "en"):
    wiki = wikipediaapi.Wikipedia(
        language=lang,
        user_agent="RAG-Chatbot-Tutorial/1.0 (contact@example.com)"
    )

    page = wiki.page(title)

    if not page.exists():
        raise ValueError(f"Wikipedia page '{title}' does not exist.")

    return page.title, page.text


def sanitize_filename(name: str) -> str:
    """Convert article title into a safe filename."""
    name = re.sub(r'[<>:"/\\|?*]', "", name)  # Remove invalid characters
    name = name.replace(" ", "_")            # Replace spaces with underscores
    return name


if __name__ == "__main__":
    title = sys.argv[1] if len(sys.argv) > 1 else "History of spaceflight"

    article_title, text = fetch_article(title)

    print(f"Fetched article: '{article_title}'")
    print(f"Total characters: {len(text)}")
    print("---- Preview (first 500 chars) ----")
    print(text[:500])

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    filename = sanitize_filename(article_title) + ".txt"
    out_path = RAW_DIR / filename

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"\nSaved full article text to {out_path}")